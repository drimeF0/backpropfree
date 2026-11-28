"""Milestone 3 (Iters 9-13): O(1)-Depth & 0-Backprop DPO Alignment (<= 4 Loss Terms, Zero Ref Model in VRAM).

Implements:
  1. Standard_FullBackprop_DPO baseline (logs OOM/VRAM spike on 9B vs 360M).
  2. Cached-RefLogp Single-Layer ResSkip DPO (O(1) depth, x_in.detach(), <= 4 loss terms, cartography-targeted).
  3. 0-Backprop Contrastive Preference Ridge LoRA (no_grad() Delta_h = h(chosen)-h(rejected) + Null-Space Ridge/SVD).
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from eval_harness import disable_adapters_ctx
from guards import BaselineBackpropProfiler, LoRAManager, SynthesisMetrics, cleanup_cuda
from iter_0_universal_baselines import FullBackpropLoRAManager, _cleanup_failed_backprop
from iter_1_transfer_cot_block import UnifiedTransferManager
from probes import gather_positions
from unified_model_utils import (
    ModelSpec,
    apply_chat_safe,
    get_layer_type,
    get_layers,
    get_lm_head,
    get_lm_norm,
    get_rmsnorm_gamma,
    get_text_model,
    prepare_layer_inputs_unified,
    slice_cm_map,
    slice_pe,
    step_layer_unified,
)
from universal_eval import SYS_COT
from universal_sft import _chunked_step_range


def _tokenize_pref_pairs(
    tokenizer,
    pref_pairs: List[Dict[str, str]],
    device: str,
    max_seq_len: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]], torch.Tensor, List[int], torch.Tensor, torch.Tensor, List[List[int]], torch.Tensor, List[int]]:
    """Tokenize (prompt, chosen) and (prompt, rejected) response token slices cleanly."""
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    eos_id = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None

    def _build(resp_key: str):
        seqs, pos_l, tgt_l, counts = [], [], [], []
        for pair in pref_pairs:
            p_str = apply_chat_safe(
                tokenizer,
                [{"role": "system", "content": SYS_COT}, {"role": "user", "content": pair["prompt"]}],
                add_generation_prompt=True,
                enable_thinking=False,
            )
            p_ids = tokenizer.encode(p_str, add_special_tokens=False)
            r_ids = tokenizer.encode(pair[resp_key], add_special_tokens=False)
            if eos_id is not None:
                r_ids = r_ids + [eos_id]
            if len(p_ids) + len(r_ids) > max_seq_len:
                p_ids = p_ids[-max(16, max_seq_len - len(r_ids)) :]
                r_ids = r_ids[: max_seq_len - len(p_ids)]
            seqs.append(p_ids + r_ids)
            pos_l.append(list(range(len(p_ids) - 1, len(p_ids) - 1 + len(r_ids))))
            tgt_l.extend(r_ids)
            counts.append(len(r_ids))
        ml = max(len(s) for s in seqs)
        ids_t = torch.full((len(seqs), ml), pad_id, dtype=torch.long, device=device)
        msk_t = torch.zeros((len(seqs), ml), dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            ids_t[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            msk_t[i, : len(s)] = 1
        return ids_t, msk_t, pos_l, torch.tensor(tgt_l, dtype=torch.long, device=device), counts

    w_ids, w_msk, w_pos, w_tgt, w_cnt = _build("chosen")
    l_ids, l_msk, l_pos, l_tgt, l_cnt = _build("rejected")
    return w_ids, w_msk, w_pos, w_tgt, w_cnt, l_ids, l_msk, l_pos, l_tgt, l_cnt


def _sum_logps_from_logits(logits: torch.Tensor, tgt_ids: torch.Tensor, counts: List[int]) -> List[torch.Tensor]:
    lp = F.log_softmax(logits.float(), dim=-1)
    tok_lp = lp[torch.arange(tgt_ids.shape[0], device=logits.device), tgt_ids]
    out: List[torch.Tensor] = []
    off = 0
    for c in counts:
        out.append(tok_lp[off : off + c].sum())
        off += c
    return out


# ---------------------------------------------------------------------------
# 1. Full-Backprop DPO Baseline (Memory/OOM Benchmark)
# ---------------------------------------------------------------------------

def run_full_backprop_dpo_baseline(
    model: nn.Module,
    tokenizer,
    spec: ModelSpec,
    pref_pairs: List[Dict[str, str]],
    device: str,
    static_vram_mb: float,
    beta: float = 0.1,
    steps: int = 8,
    lr: float = 2e-4,
    use_grad_ckpt: bool = False,
) -> Tuple[Optional[FullBackpropLoRAManager], Dict[str, object]]:
    """Standard full-backprop DPO (all layers, holding ref logps + full activation graphs) for VRAM/OOM comparison."""
    w_ids, w_msk, w_pos, w_tgt, w_cnt, l_ids, l_msk, l_pos, l_tgt, l_cnt = _tokenize_pref_pairs(
        tokenizer, pref_pairs, device
    )
    mgr = FullBackpropLoRAManager(model, r=16, lora_alpha=32.0)

    # Compute reference logps before training (note: holding a 2nd 9B model directly in 15GB T4 is impossible)
    with torch.no_grad():
        for w in mgr.wrapped.values():
            w.active = False
        ref_w_list: List[float] = []
        ref_l_list: List[float] = []
        off_w = off_l = 0
        for i in range(len(pref_pairs)):
            lg_w = model(input_ids=w_ids[i : i + 1], attention_mask=w_msk[i : i + 1], use_cache=False).logits
            pts_w = gather_positions(lg_w, [w_pos[i]])
            cw = w_cnt[i]
            ref_w_list.append(float(_sum_logps_from_logits(pts_w, w_tgt[off_w : off_w + cw], [cw])[0].item()))
            off_w += cw

            lg_l = model(input_ids=l_ids[i : i + 1], attention_mask=l_msk[i : i + 1], use_cache=False).logits
            pts_l = gather_positions(lg_l, [l_pos[i]])
            cl = l_cnt[i]
            ref_l_list.append(float(_sum_logps_from_logits(pts_l, l_tgt[off_l : off_l + cl], [cl])[0].item()))
            off_l += cl
            del lg_w, pts_w, lg_l, pts_l
        for w in mgr.wrapped.values():
            w.active = True

    ref_w_t = torch.tensor(ref_w_list, dtype=torch.float32, device=device)
    ref_l_t = torch.tensor(ref_l_list, dtype=torch.float32, device=device)

    model.train()
    lm = get_text_model(model)
    emb_hook = None
    if use_grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        emb_hook = lm.embed_tokens.register_forward_hook(
            lambda m, inp, out: out.requires_grad_(True) if isinstance(out, torch.Tensor) else out
        )

    params = mgr.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=lr)
    bs = 4 if spec.scale == "large" else 8

    try:
        with BaselineBackpropProfiler(model, static_vram_mb=static_vram_mb) as prof:
            n = len(pref_pairs)
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                off_w = off_l = 0
                for s in range(0, n, bs):
                    e = min(n, s + bs)
                    cw_sum = sum(w_cnt[s:e])
                    cl_sum = sum(l_cnt[s:e])
                    lg_w = model(input_ids=w_ids[s:e], attention_mask=w_msk[s:e], use_cache=False).logits
                    pts_w = gather_positions(lg_w, w_pos[s:e])
                    pol_w = torch.stack(_sum_logps_from_logits(pts_w, w_tgt[off_w : off_w + cw_sum], w_cnt[s:e]))
                    del lg_w, pts_w

                    lg_l = model(input_ids=l_ids[s:e], attention_mask=l_msk[s:e], use_cache=False).logits
                    pts_l = gather_positions(lg_l, l_pos[s:e])
                    pol_l = torch.stack(_sum_logps_from_logits(pts_l, l_tgt[off_l : off_l + cl_sum], l_cnt[s:e]))
                    del lg_l, pts_l

                    logits_dpo = beta * ((pol_w - ref_w_t[s:e]) - (pol_l - ref_l_t[s:e]))
                    loss = -F.logsigmoid(logits_dpo).sum() / float(n)
                    loss.backward()
                    off_w += cw_sum
                    off_l += cl_sum
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
        return mgr, {
            "oom": False,
            "use_grad_ckpt": use_grad_ckpt,
            "batch_size": bs,
            "peak_vram_mb": round(prof.metrics.peak_vram_mb, 2),
            "overhead_mb": round(prof.metrics.activation_vram_overhead_mb, 2),
            "wall_time_sec": round(prof.metrics.wall_time_sec, 2),
            "metrics": prof.metrics,
        }
    except torch.cuda.OutOfMemoryError as err:
        peak_oom = float(torch.cuda.max_memory_allocated() / (1024.0 ** 2))
        _cleanup_failed_backprop(model)
        return None, {
            "oom": True,
            "error": str(err)[:180],
            "use_grad_ckpt": use_grad_ckpt,
            "batch_size": bs,
            "peak_vram_mb": round(peak_oom, 2),
            "overhead_mb": round(max(0.0, peak_oom - static_vram_mb), 2),
        }
    finally:
        if emb_hook is not None:
            emb_hook.remove()
        if use_grad_ckpt and hasattr(model, "gradient_checkpointing_disable"):
            try:
                model.gradient_checkpointing_disable()
            except Exception:
                pass
        model.eval()


# ---------------------------------------------------------------------------
# 2. Cached-RefLogp Single-Layer ResSkip DPO (O(1) Depth, <= 4 Terms)
# ---------------------------------------------------------------------------

def synthesize_o1_resskip_dpo(
    model: nn.Module,
    tokenizer,
    spec: ModelSpec,
    pref_pairs: List[Dict[str, str]],
    cal_texts: List[str],
    target_layers: List[int],
    device: str,
    r: int = 16,
    lora_alpha: float = 24.0,
    alpha_null: float = 8.0,
    beta: float = 0.15,
    lambda_nll: float = 0.08,
    lambda_cal_kl: float = 3.5,
    lambda_reg: float = 0.002,
    steps_per_layer: int = 8,
    existing_mgr: Optional[UnifiedTransferManager] = None,
) -> Tuple[UnifiedTransferManager, Dict[str, object]]:
    """O(1)-depth Detached Single-Layer DPO using cached ref_logps & skip states (<= 4 loss terms)."""
    t0 = time.perf_counter()
    target_layers = sorted(set(int(l) for l in target_layers))
    layers = get_layers(model)
    n_layers = len(layers)
    lm_norm = get_lm_norm(model)
    lm_head = get_lm_head(model)
    is_large = spec.hidden_size >= 2048
    fwd_bs = 8 if is_large else 24

    # Note: if existing_mgr is passed (e.g. stacking DPO onto SFT), preserve its frozen LoRA weights as base!
    # We install a fresh layer-targeted UnifiedTransferManager wrapping current modules cleanly.
    if existing_mgr is not None:
        existing_mgr.freeze_all()
        existing_mgr.set_all_active(True)
    mgr = UnifiedTransferManager(model, target_layers=target_layers, r=r, lora_alpha=lora_alpha)

    with torch.no_grad():
        mgr.freeze_all()
        mgr.set_all_active(False)

        w_ids, w_msk, w_pos, w_tgt, w_cnt, l_ids, l_msk, l_pos, l_tgt, l_cnt = _tokenize_pref_pairs(
            tokenizer, pref_pairs, device
        )
        prev_side = tokenizer.padding_side
        tokenizer.padding_side = "right"
        cal_enc = tokenizer(cal_texts[:16], return_tensors="pt", padding=True, truncation=True, max_length=96).to(device)
        cal_lens = cal_enc.attention_mask.sum(dim=1).tolist()
        cal_pos = [[max(0, min(L - 1, int(k * L / 4))) for k in range(1, 4)] for L in cal_lens]
        tokenizer.padding_side = prev_side
        cal_mask_bool = cal_enc.attention_mask.bool()

        def _entry_and_logps(ids, mask, pos_l, tgt_t, cnts, compute_ref_lp: bool):
            h0, cm0, pe0, tp0 = prepare_layer_inputs_unified(model, ids, mask)
            h_ent = _chunked_step_range(model, 0, target_layers[0], h0, cm0, pe0, tp0, fwd_bs)
            if not compute_ref_lp:
                return h_ent, cm0, pe0, tp0, None
            h_fin = _chunked_step_range(model, target_layers[0], n_layers, h_ent, cm0, pe0, tp0, fwd_bs)
            pts = gather_positions(h_fin, pos_l)
            lp_list = []
            for s in range(0, pts.shape[0], 32):
                e = min(pts.shape[0], s + 32)
                lp_list.append(F.log_softmax(lm_head(lm_norm(pts[s:e])).float(), dim=-1))
            lp_all = torch.cat(lp_list, dim=0)
            tok_lp = lp_all[torch.arange(tgt_t.shape[0], device=device), tgt_t]
            seq_lps = []
            off = 0
            for c in cnts:
                seq_lps.append(float(tok_lp[off : off + c].sum().item()))
                off += c
            del h_fin, pts, lp_all
            return h_ent, cm0, pe0, tp0, torch.tensor(seq_lps, dtype=torch.float32, device=device)

        H_w, cm_w, pe_w, tp_w, ref_w = _entry_and_logps(w_ids, w_msk, w_pos, w_tgt, w_cnt, True)
        H_l, cm_l, pe_l, tp_l, ref_l = _entry_and_logps(l_ids, l_msk, l_pos, l_tgt, l_cnt, True)

        # Calibration base probs
        h0_c, cm_c, pe_c, tp_c = prepare_layer_inputs_unified(model, cal_enc.input_ids, cal_enc.attention_mask)
        H_cal = _chunked_step_range(model, 0, target_layers[0], h0_c, cm_c, pe_c, tp_c, fwd_bs)
        h_fin_c = _chunked_step_range(model, target_layers[0], n_layers, H_cal, cm_c, pe_c, tp_c, fwd_bs)
        pts_c = gather_positions(h_fin_c, cal_pos)
        p_cal_chunks = []
        for s in range(0, pts_c.shape[0], 32):
            e = min(pts_c.shape[0], s + 32)
            p_cal_chunks.append(F.softmax(lm_head(lm_norm(pts_c[s:e])).float(), dim=-1).half().cpu())
        p_cal_store = torch.cat(p_cal_chunks, dim=0).to("cpu" if is_large else device)
        del h_fin_c, pts_c
        if is_large:
            torch.cuda.empty_cache()

    def _compute_skip(l_cur: int, h_in: torch.Tensor, cm_m, pe_t, tp_t, pos_l):
        n = h_in.shape[0]
        skips = []
        for s in range(0, n, fwd_bs):
            e = min(n, s + fwd_bs)
            cm_s, pe_s, tp_s = slice_cm_map(cm_m, s, e, n), slice_pe(pe_t, s, e, n), tp_t[s:e]
            hl = step_layer_unified(model, l_cur, h_in[s:e], cm_s, pe_s, tp_s)
            hc = hl
            for j in range(l_cur + 1, n_layers):
                hc = step_layer_unified(model, j, hc, cm_s, pe_s, tp_s)
            skips.append(gather_positions(hc - hl, pos_l[s:e]).detach())
        return torch.cat(skips, dim=0)

    lr_b = 2.4e-3 * ((4096.0 / float(spec.hidden_size)) ** 0.25)
    m_cal = max(sum(len(p) for p in cal_pos), 1)
    m_w_toks = max(int(w_tgt.shape[0]), 1)
    n_pairs = len(pref_pairs)
    pair_bs = 4 if is_large else 12
    cal_bs = 8 if is_large else 16

    layer_logs = []
    for idx_s, l_idx in enumerate(target_layers):
        Hw_in, Hl_in, Hc_in = H_w.detach(), H_l.detach(), H_cal.detach()

        with torch.no_grad():
            mgr.freeze_all()
            mgr.set_all_active(False)
            spaces = ("attn_in", "mixer_out", "mlp_in", "down_proj")
            hooks = [(sp, mgr.rep_module_for_space(l_idx, sp)) for sp in spaces]
            acts_by_sp: Dict[str, List[torch.Tensor]] = {sp: [] for sp in spaces}

            for s in range(0, Hc_in.shape[0], fwd_bs):
                e = min(Hc_in.shape[0], s + fwd_bs)
                mf = cal_mask_bool[s:e].reshape(-1)
                def _mk(sp_n: str):
                    def hk(m, inp, out):
                        xg = inp[0].detach().reshape(-1, inp[0].shape[-1])[mf].float()
                        if xg.shape[0] > 260:
                            xg = xg[:: max(1, xg.shape[0] // 260)][:260]
                        acts_by_sp[sp_n].append(xg.cpu())
                    return hk
                live = [mod.register_forward_hook(_mk(sp)) for sp, mod in hooks]
                _ = step_layer_unified(model, l_idx, Hc_in[s:e], slice_cm_map(cm_c, s, e, Hc_in.shape[0]), slice_pe(pe_c, s, e, Hc_in.shape[0]), tp_c[s:e])
                for hr in live:
                    hr.remove()

            for idx_m, ((li, m_name), (_, _, sp)) in enumerate(mgr.mod_loc.items()):
                if li != l_idx:
                    continue
                x_all = torch.cat(acts_by_sp[sp], dim=0).to(device=device, dtype=torch.float32)
                x_tilde = x_all / math.sqrt(max(x_all.shape[0], 1))
                mean_var = max(float((x_tilde * x_tilde).sum().item()) / float(max(x_tilde.shape[1], 1)), 1e-8)
                q_k = min(96, x_tilde.shape[0] - 2, x_tilde.shape[1] - 2)
                _, sv, v_mat = torch.svd_lowrank(x_tilde, q=max(8, q_k), niter=2)
                norm_eig = (sv * sv) / mean_var
                c_damp = (1.0 - torch.rsqrt(1.0 + alpha_null * norm_eig)).float().detach()
                mgr.wrapped[(li, m_name)].set_precond_and_init(v_mat.half().detach(), c_damp, seed_val=9100 * l_idx + idx_m)
                del x_all, x_tilde, sv, v_mat, norm_eig, c_damp
            acts_by_sp.clear()

            mgr.set_all_active(True)
            mgr.freeze_all()
            skip_w = _compute_skip(l_idx, Hw_in, cm_w, pe_w, tp_w, w_pos)
            skip_l = _compute_skip(l_idx, Hl_in, cm_l, pe_l, tp_l, l_pos)
            skip_c = _compute_skip(l_idx, Hc_in, cm_c, pe_c, tp_c, cal_pos)

        mgr.activate_for_layer(l_idx)
        opt_groups, params_all = [], []
        for (li, _), w in mgr.wrapped.items():
            if li == l_idx:
                lr_a = lr_b * 0.40 * math.sqrt(960.0 / float(max(w.in_features, 1)))
                opt_groups.append({"params": [w.lora_A], "lr": lr_a, "weight_decay": 1e-3})
                opt_groups.append({"params": [w.lora_B], "lr": lr_b, "weight_decay": 2e-3})
                params_all.extend([w.lora_A, w.lora_B])
        opt = torch.optim.AdamW(opt_groups)

        last_dpo = last_cal = 0.0
        for _ in range(steps_per_layer):
            opt.zero_grad(set_to_none=True)
            dpo_acc = 0.0
            off_w = off_l = 0
            for s in range(0, n_pairs, pair_bs):
                e = min(n_pairs, s + pair_bs)
                cw = sum(w_cnt[s:e])
                cl = sum(l_cnt[s:e])
                with torch.enable_grad():
                    ow = step_layer_unified(model, l_idx, Hw_in[s:e], slice_cm_map(cm_w, s, e, n_pairs), slice_pe(pe_w, s, e, n_pairs), tp_w[s:e])
                    hw_fs = gather_positions(ow, w_pos[s:e]) + skip_w[off_w : off_w + cw]
                    lg_w = lm_head(lm_norm(hw_fs)).float()
                    pol_w = torch.stack(_sum_logps_from_logits(lg_w, w_tgt[off_w : off_w + cw], w_cnt[s:e]))
                    del ow, hw_fs, lg_w

                    ol = step_layer_unified(model, l_idx, Hl_in[s:e], slice_cm_map(cm_l, s, e, n_pairs), slice_pe(pe_l, s, e, n_pairs), tp_l[s:e])
                    hl_fs = gather_positions(ol, l_pos[s:e]) + skip_l[off_l : off_l + cl]
                    lg_l = lm_head(lm_norm(hl_fs)).float()
                    pol_l = torch.stack(_sum_logps_from_logits(lg_l, l_tgt[off_l : off_l + cl], l_cnt[s:e]))
                    del ol, hl_fs, lg_l

                    # Term 1: DPO Loss + Term 2: Gentle Chosen NLL stabilizer
                    margin = (pol_w - ref_w[s:e]) - (pol_l - ref_l[s:e])
                    l_dpo = -F.logsigmoid(beta * margin).sum() / float(n_pairs)
                    l_nll = lambda_nll * (-pol_w.sum() / float(m_w_toks))
                    (l_dpo + l_nll).backward()
                    dpo_acc += float(l_dpo.item())
                off_w += cw
                off_l += cl
            last_dpo = dpo_acc

            # Term 3: Calibration KL + Term 4: Frobenius Reg
            cal_off = 0
            cal_acc = 0.0
            for cs0 in range(0, Hc_in.shape[0], cal_bs):
                cs1 = min(Hc_in.shape[0], cs0 + cal_bs)
                ccnt = sum(len(p) for p in cal_pos[cs0:cs1])
                ct0, ct1 = cal_off, cal_off + ccnt
                cal_off = ct1
                with torch.enable_grad():
                    oc = step_layer_unified(model, l_idx, Hc_in[cs0:cs1], slice_cm_map(cm_c, cs0, cs1, Hc_in.shape[0]), slice_pe(pe_c, cs0, cs1, Hc_in.shape[0]), tp_c[cs0:cs1])
                    ptsc = gather_positions(oc, cal_pos[cs0:cs1]) + skip_c[ct0:ct1]
                    lpc = F.log_softmax(lm_head(lm_norm(ptsc)).float(), dim=-1)
                    pcs = p_cal_store[ct0:ct1].to(device=device, dtype=torch.float32)
                    l_cal = (pcs * (pcs.clamp(min=1e-12).log() - lpc)).sum() / float(m_cal)
                    loss_c = lambda_cal_kl * l_cal
                    if cs0 == 0:
                        loss_c = loss_c + lambda_reg * mgr.get_layer_frob_reg(l_idx)
                    loss_c.backward()
                    cal_acc += float(l_cal.item())
                del oc, ptsc, lpc, pcs, l_cal, loss_c
            last_cal = cal_acc

            torch.nn.utils.clip_grad_norm_(params_all, 1.0)
            opt.step()

        mgr.merge_and_freeze_layer(l_idx)
        del opt, skip_w, skip_l, skip_c
        layer_logs.append({"layer": l_idx, "dpo_loss": round(last_dpo, 4), "cal_kl": round(last_cal, 5)})

        if idx_s + 1 < len(target_layers):
            nxt = target_layers[idx_s + 1]
            with torch.no_grad():
                H_w = _chunked_step_range(model, l_idx, nxt, Hw_in, cm_w, pe_w, tp_w, fwd_bs)
                H_l = _chunked_step_range(model, l_idx, nxt, Hl_in, cm_l, pe_l, tp_l, fwd_bs)
                H_cal = _chunked_step_range(model, l_idx, nxt, Hc_in, cm_c, pe_c, tp_c, fwd_bs)
        del Hw_in, Hl_in, Hc_in
        if is_large:
            torch.cuda.empty_cache()

    mgr.freeze_all()
    mgr.set_all_active(True)
    return mgr, {
        "method": "CachedRef_O1_ResSkip_DPO",
        "target_layers": target_layers,
        "loss_terms": 4,
        "layer_logs": layer_logs,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }


# ---------------------------------------------------------------------------
# 3. 0-Backprop Contrastive Preference Ridge LoRA (no_grad() Delta_h + SVD)
# ---------------------------------------------------------------------------

@torch.no_grad()
def synthesize_zero_backprop_contrastive_dpo(
    model: nn.Module,
    tokenizer,
    spec: ModelSpec,
    pref_pairs: List[Dict[str, str]],
    cal_texts: List[str],
    target_layers: List[int],
    device: str,
    r: int = 16,
    lora_alpha: float = 16.0,
    eta_steer: float = 0.25,
    existing_mgr: Optional[UnifiedTransferManager] = None,
) -> Tuple[UnifiedTransferManager, Dict[str, object]]:
    """Pure torch.no_grad() Contrastive Preference Ridge LoRA using Delta_h = h(chosen) - h(rejected)."""
    t0 = time.perf_counter()
    target_layers = sorted(set(int(l) for l in target_layers))
    if existing_mgr is not None:
        existing_mgr.freeze_all()
        existing_mgr.set_all_active(True)
    mgr = UnifiedTransferManager(model, target_layers=target_layers, r=r, lora_alpha=lora_alpha)
    mgr.freeze_all()
    mgr.set_all_active(False)

    n_layers = len(get_layers(model))
    lm_norm = get_lm_norm(model)
    lm_head = get_lm_head(model)
    eps = float(getattr(lm_norm, "variance_epsilon", getattr(lm_norm, "eps", 1e-6)))
    gamma = get_rmsnorm_gamma(lm_norm).to(device=device, dtype=torch.float32)
    W_lm_raw = lm_head.weight
    is_large = spec.hidden_size >= 2048
    fwd_bs = 8 if is_large else 24

    w_ids, w_msk, w_pos, w_tgt, w_cnt, l_ids, l_msk, l_pos, l_tgt, l_cnt = _tokenize_pref_pairs(
        tokenizer, pref_pairs, device
    )
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    cal_enc = tokenizer(cal_texts[:20], return_tensors="pt", padding=True, truncation=True, max_length=96).to(device)
    tokenizer.padding_side = prev_side
    cal_mask_bool = cal_enc.attention_mask.bool()

    hw, cm_w, pe_w, tp_w = prepare_layer_inputs_unified(model, w_ids, w_msk)
    hw = _chunked_step_range(model, 0, target_layers[0], hw, cm_w, pe_w, tp_w, fwd_bs)
    hl, cm_l, pe_l, tp_l = prepare_layer_inputs_unified(model, l_ids, l_msk)
    hl = _chunked_step_range(model, 0, target_layers[0], hl, cm_l, pe_l, tp_l, fwd_bs)
    hc, cm_c, pe_c, tp_c = prepare_layer_inputs_unified(model, cal_enc.input_ids, cal_enc.attention_mask)
    hc = _chunked_step_range(model, 0, target_layers[0], hc, cm_c, pe_c, tp_c, fwd_bs)

    layer_logs = []
    for idx_s, l_idx in enumerate(target_layers):
        # Compute contrastive preference vector Delta_h = mean(h(chosen) - h(rejected)) at layer l_idx
        # PLUS analytical DPO hidden target at lm_head on chosen vs rejected tokens!
        h_fin_w = _chunked_step_range(model, l_idx, n_layers, hw, cm_w, pe_w, tp_w, fwd_bs)
        h_fin_l = _chunked_step_range(model, l_idx, n_layers, hl, cm_l, pe_l, tp_l, fwd_bs)
        pts_w = gather_positions(h_fin_w, w_pos).float()
        pts_l = gather_positions(h_fin_l, l_pos).float()

        # Contrastive mean difference Delta_h across chosen vs rejected final states
        w_boundary_pos = [[p[0]] for p in w_pos]
        l_boundary_pos = [[p[0]] for p in l_pos]
        hb_w = gather_positions(h_fin_w, w_boundary_pos).float()
        hb_l = gather_positions(h_fin_l, l_boundary_pos).float()
        delta_h_mean = (pts_w.mean(dim=0, keepdim=True) - pts_l.mean(dim=0, keepdim=True))
        delta_h_dir = delta_h_mean / delta_h_mean.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        del h_fin_w, h_fin_l, hb_w, hb_l

        # Analytical DPO hidden shifts: push chosen tokens up (+u_w + steer*Delta_h) and rejected tokens down (-u_l)
        def _ce_dir(pts: torch.Tensor, tids: torch.Tensor, sign: float) -> torch.Tensor:
            out_dh = torch.zeros_like(pts)
            for s in range(0, pts.shape[0], 32):
                e = min(pts.shape[0], s + 32)
                hv = pts[s:e]
                td = tids[s:e]
                inv_rms = torch.rsqrt((hv * hv).mean(dim=-1, keepdim=True) + eps)
                h_hat = hv * inv_rms
                logits = lm_head((h_hat * gamma.unsqueeze(0)).to(W_lm_raw.dtype)).float()
                probs = F.softmax(logits, dim=-1)
                ar = torch.arange(hv.shape[0], device=device)
                gz = probs
                gz[ar, td] -= 1.0
                g_norm = (gz.to(W_lm_raw.dtype) @ W_lm_raw).float() * gamma.unsqueeze(0)
                g_h = inv_rms * (g_norm - (g_norm * h_hat).mean(dim=-1, keepdim=True) * h_hat)
                h_rms = hv.norm(dim=-1, keepdim=True)
                u = (-g_h) / g_h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                if sign > 0:
                    u = (u + 0.35 * delta_h_dir).renorm(p=2, dim=-1, maxnorm=1.0)
                out_dh[s:e] = sign * (eta_steer * h_rms / max(len(target_layers), 1)) * u
                del logits, probs, gz, g_norm, g_h
            return out_dh

        dH_w = _ce_dir(pts_w, w_tgt, +1.0)
        dH_l = _ce_dir(pts_l, l_tgt, -0.45)
        dH_all = torch.cat([dH_w, dH_l], dim=0)
        del pts_w, pts_l, dH_w, dH_l

        lt = get_layer_type(model, l_idx)
        for mod_key in ("o_proj" if lt != "linear_attention" else "out_proj", "down_proj"):
            w = mgr.wrapped[(l_idx, mod_key)]
            xw_list, xl_list, xc_list = [], [], []

            def _cap(hb, cmb, peb, tpb, g_or_m, is_pos: bool, dest: List[torch.Tensor]):
                nb = hb.shape[0]
                for s0 in range(0, nb, fwd_bs):
                    e0 = min(nb, s0 + fwd_bs)
                    bc, sl = e0 - s0, hb.shape[1]
                    def hk(m, inp, out):
                        if is_pos:
                            dest.append(gather_positions(inp[0].detach().reshape(bc, sl, -1), g_or_m[s0:e0]).float())
                        else:
                            mf = g_or_m[s0:e0].reshape(-1)
                            xg = inp[0].detach().reshape(-1, inp[0].shape[-1])[mf].float()
                            if xg.shape[0] > 260:
                                xg = xg[:: max(1, xg.shape[0] // 260)][:260]
                            dest.append(xg.cpu())
                    hr = w.register_forward_hook(hk)
                    _ = step_layer_unified(model, l_idx, hb[s0:e0], slice_cm_map(cmb, s0, e0, nb), slice_pe(peb, s0, e0, nb), tpb[s0:e0])
                    hr.remove()

            _cap(hw, cm_w, pe_w, tp_w, w_pos, True, xw_list)
            _cap(hl, cm_l, pe_l, tp_l, l_pos, True, xl_list)
            _cap(hc, cm_c, pe_c, tp_c, cal_mask_bool, False, xc_list)
            X_pref = torch.cat(xw_list + xl_list, dim=0)
            X_cal = torch.cat(xc_list, dim=0).to(device)
            if X_cal.shape[0] > 768:
                X_cal = X_cal[:: max(1, X_cal.shape[0] // 768)][:768]

            d_in = X_pref.shape[1]
            c_mat = (X_cal.T @ X_cal) / max(X_cal.shape[0], 1)
            c_scale = max(float(torch.trace(c_mat).item() / max(d_in, 1)), 1e-8)
            c_norm = c_mat * (6.0 / c_scale)
            if d_in <= 4096:
                ev, evec = torch.linalg.eigh(0.5 * (c_norm + c_norm.T) + torch.eye(d_in, device=device))
                S_inv = (evec / torch.sqrt(ev.clamp(min=0.5)).unsqueeze(0)) @ evec.T
                Xw = X_pref @ S_inv
                apply_sinv = lambda M: M @ S_inv
                del ev, evec
            else:
                Uk, Sk, _ = torch.svd_lowrank(c_norm, q=min(384, d_in - 2), niter=2)
                shrink = 1.0 - torch.rsqrt(1.0 + Sk.clamp(min=0.0))
                apply_sinv = lambda M: M - ((M @ Uk) * shrink.unsqueeze(0)) @ Uk.T
                Xw = apply_sinv(X_pref)

            dY = 0.5 * dH_all
            mn = Xw.shape[0]
            ridge = 0.1 * max(float((Xw * Xw).sum().item() / max(mn, 1)), 1e-6)
            if mn < d_in:
                Theta_w = Xw.T @ torch.linalg.solve(Xw @ Xw.T + ridge * torch.eye(mn, device=device), dY)
            else:
                Theta_w = torch.linalg.solve(Xw.T @ Xw + ridge * torch.eye(d_in, device=device), Xw.T @ dY)
            U, S, Vh = torch.linalg.svd(Theta_w.T, full_matrices=False)
            r_eff = min(r, S.shape[0])
            B_fit = torch.zeros((w.out_features, r), device=device, dtype=torch.float32)
            A_fit = torch.zeros((r, d_in), device=device, dtype=torch.float32)
            B_fit[:, :r_eff] = U[:, :r_eff] * S[:r_eff].unsqueeze(0)
            A_fit[:r_eff, :] = apply_sinv(Vh[:r_eff, :])
            fro = float((B_fit @ A_fit).norm().item())
            if fro > 3.5:
                B_fit = B_fit * (3.5 / (fro + 1e-8))
            w.lora_A.data.copy_(A_fit / max(w.scaling, 1e-6))
            w.lora_B.data.copy_(B_fit)
            w.active = True
            del X_pref, X_cal, c_mat, c_norm, Xw, Theta_w, U, S, Vh

        layer_logs.append({"layer": l_idx, "delta_h_norm": round(float(delta_h_mean.norm().item()), 4)})
        if idx_s + 1 < len(target_layers):
            nxt = target_layers[idx_s + 1]
            hw = _chunked_step_range(model, l_idx, nxt, hw, cm_w, pe_w, tp_w, fwd_bs)
            hl = _chunked_step_range(model, l_idx, nxt, hl, cm_l, pe_l, tp_l, fwd_bs)
            hc = _chunked_step_range(model, l_idx, nxt, hc, cm_c, pe_c, tp_c, fwd_bs)

    return mgr, {
        "method": "ZeroBackprop_Contrastive_Pref_Ridge",
        "target_layers": target_layers,
        "loss_terms": 1,
        "layer_logs": layer_logs,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }
