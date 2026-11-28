"""Milestone 4 (Iters 14-20): O(1)-Depth & 0-Backprop Verifiable-Reward RL (GRPO / Advantage-Weighted Null Ridge).

Pure deterministic programmatic verifier (R in {0.0, 1.0}):
  - Code tasks: isolated 2s subprocess sandbox unit tests (`assert`).
  - Deduction tasks: exact regex/numeric pattern match on deduced answer.
Strictly O(1)-depth (x_in.detach() + cached ref_logps OR pure torch.no_grad() Ridge/SVD), <= 4 loss terms,
targeted to Reasoning Layers identified in Milestone 1 skill_cartography.json.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from guards import cleanup_cuda
from iter_1_transfer_cot_block import UnifiedTransferManager
from probes import gather_positions
from unified_model_utils import (
    ModelSpec,
    apply_chat_safe,
    get_eos_token_ids,
    get_layer_type,
    get_layers,
    get_lm_head,
    get_lm_norm,
    get_rmsnorm_gamma,
    prepare_layer_inputs_unified,
    slice_cm_map,
    slice_pe,
    step_layer_unified,
)
from universal_eval import SYS_CODE, SYS_COT, check_patterns
from universal_sandbox import run_code_in_sandbox
from universal_sft import _chunked_step_range


@torch.no_grad()
def sample_and_verify_rl_rollouts(
    model: nn.Module,
    tokenizer,
    spec: ModelSpec,
    rl_tasks: List[Dict[str, object]],
    device: str,
    group_size: int = 6,
    temperature: float = 0.85,
    mock_module_name: Optional[str] = None,
    mock_module_source: Optional[str] = None,
    seed: int = 42,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    """Generate G rollouts per task in no_grad(), score deterministically, and compute group advantages."""
    is_code = any("test_code" in t for t in rl_tasks)
    sys_m = SYS_CODE if is_code else SYS_COT
    max_new = 180 if is_code else 120
    eos_ids = get_eos_token_ids(tokenizer)
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    rollout_records: List[Dict[str, object]] = []
    all_rewards: List[float] = []
    non_trivial_groups = 0

    try:
        for t in rl_tasks:
            q_str = str(t.get("prompt") or t.get("q"))
            p_str = apply_chat_safe(
                tokenizer,
                [{"role": "system", "content": sys_m}, {"role": "user", "content": q_str}],
                add_generation_prompt=True,
                enable_thinking=False,
            )
            enc = tokenizer([p_str] * group_size, return_tensors="pt", padding=True).to(device)
            in_len = enc.input_ids.shape[1]
            gen = model.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_ids,
                use_cache=True,
            )
            texts = [tokenizer.decode(row[in_len:], skip_special_tokens=True).strip() for row in gen]
            del enc, gen

            rewards: List[float] = []
            for txt in texts:
                if is_code:
                    res = run_code_in_sandbox(
                        txt,
                        str(t["test_code"]),
                        expected_fn=str(t["fn_name"]),
                        mock_module_name=mock_module_name,
                        mock_module_source=mock_module_source,
                    )
                    rewards.append(1.0 if res.passed else 0.0)
                else:
                    ok = check_patterns(txt, patterns=t.get("patterns"))  # type: ignore[arg-type]
                    rewards.append(1.0 if ok else 0.0)

            all_rewards.extend(rewards)
            mean_r = sum(rewards) / len(rewards)
            var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
            std_r = math.sqrt(var_r)
            if std_r > 1e-6:
                non_trivial_groups += 1
                for txt, r_val in zip(texts, rewards):
                    adv = (r_val - mean_r) / (std_r + 1e-6)
                    rollout_records.append({
                        "prompt": p_str,
                        "completion": txt,
                        "reward": r_val,
                        "advantage": float(adv),
                    })
    finally:
        tokenizer.padding_side = prev_side

    stats = {
        "n_tasks": len(rl_tasks),
        "group_size": group_size,
        "mean_rollout_reward": round(sum(all_rewards) / max(len(all_rewards), 1), 4),
        "non_trivial_groups": non_trivial_groups,
        "kept_trajectories": len(rollout_records),
    }
    return rollout_records, stats


def _tokenize_rollouts(
    tokenizer,
    records: List[Dict[str, object]],
    device: str,
    max_seq_len: int = 320,
) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]], torch.Tensor, List[int], torch.Tensor]:
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    eos_id = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None
    seqs, pos_l, tgt_l, cnts, advs = [], [], [], [], []
    for rec in records:
        p_ids = tokenizer.encode(str(rec["prompt"]), add_special_tokens=False)
        c_ids = tokenizer.encode(str(rec["completion"]), add_special_tokens=False)
        if eos_id is not None:
            c_ids = c_ids + [eos_id]
        if not c_ids:
            continue
        if len(p_ids) + len(c_ids) > max_seq_len:
            p_ids = p_ids[-max(16, max_seq_len - len(c_ids)) :]
            c_ids = c_ids[: max_seq_len - len(p_ids)]
        if not c_ids:
            continue
        seqs.append(p_ids + c_ids)
        pos_l.append(list(range(len(p_ids) - 1, len(p_ids) - 1 + len(c_ids))))
        tgt_l.extend(c_ids)
        cnts.append(len(c_ids))
        advs.append(float(rec["advantage"]))
    ml = max(len(s) for s in seqs)
    ids_t = torch.full((len(seqs), ml), pad_id, dtype=torch.long, device=device)
    msk_t = torch.zeros((len(seqs), ml), dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        ids_t[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        msk_t[i, : len(s)] = 1
    return (
        ids_t,
        msk_t,
        pos_l,
        torch.tensor(tgt_l, dtype=torch.long, device=device),
        cnts,
        torch.tensor(advs, dtype=torch.float32, device=device),
    )


# ---------------------------------------------------------------------------
# Rule (a): Single-Layer Detached GRPO (O(1) Depth, <= 4 Loss Terms)
# ---------------------------------------------------------------------------

def synthesize_o1_detached_grpo(
    model: nn.Module,
    tokenizer,
    spec: ModelSpec,
    rl_tasks: List[Dict[str, object]],
    cal_texts: List[str],
    target_layers: List[int],
    device: str,
    mock_module_name: Optional[str] = None,
    mock_module_source: Optional[str] = None,
    r: int = 16,
    lora_alpha: float = 16.0,
    alpha_null: float = 8.0,
    group_size: int = 6,
    kl_beta: float = 0.04,
    clip_eps: float = 0.2,
    lambda_cal_kl: float = 4.0,
    lambda_reg: float = 0.003,
    steps_per_layer: int = 6,
    existing_mgr: Optional[UnifiedTransferManager] = None,
    seed: int = 42,
) -> Tuple[Optional[UnifiedTransferManager], Dict[str, object]]:
    """O(1)-depth Detached Single-Layer GRPO targeted to Reasoning Layers (<= 4 loss terms)."""
    t0 = time.perf_counter()
    target_layers = sorted(set(int(l) for l in target_layers))
    if existing_mgr is not None:
        existing_mgr.freeze_all()
        existing_mgr.set_all_active(True)

    rollout_records, r_stats = sample_and_verify_rl_rollouts(
        model,
        tokenizer,
        spec,
        rl_tasks,
        device,
        group_size=group_size,
        mock_module_name=mock_module_name,
        mock_module_source=mock_module_source,
        seed=seed,
    )
    if not rollout_records:
        return None, {
            "method": "O1_Detached_GRPO",
            "target_layers": target_layers,
            "skipped_zero_variance": True,
            "rollout_stats": r_stats,
            "elapsed_sec": round(time.perf_counter() - t0, 2),
        }

    mgr = UnifiedTransferManager(model, target_layers=target_layers, r=r, lora_alpha=lora_alpha)
    n_layers = len(get_layers(model))
    lm_norm = get_lm_norm(model)
    lm_head = get_lm_head(model)
    is_large = spec.hidden_size >= 2048
    fwd_bs = 6 if is_large else 20

    with torch.no_grad():
        mgr.freeze_all()
        mgr.set_all_active(False)
        r_ids, r_msk, r_pos, r_tgt, r_cnt, r_adv = _tokenize_rollouts(tokenizer, rollout_records, device)

        prev_side = tokenizer.padding_side
        tokenizer.padding_side = "right"
        cal_enc = tokenizer(cal_texts[:16], return_tensors="pt", padding=True, truncation=True, max_length=96).to(device)
        cal_lens = cal_enc.attention_mask.sum(dim=1).tolist()
        cal_pos = [[max(0, min(L - 1, int(k * L / 4))) for k in range(1, 4)] for L in cal_lens]
        tokenizer.padding_side = prev_side
        cal_mask_bool = cal_enc.attention_mask.bool()

        # Precompute cached ref token log-probs & entry states in no_grad()
        h0_r, cm_r, pe_r, tp_r = prepare_layer_inputs_unified(model, r_ids, r_msk)
        H_r = _chunked_step_range(model, 0, target_layers[0], h0_r, cm_r, pe_r, tp_r, fwd_bs)
        h_fin_r = _chunked_step_range(model, target_layers[0], n_layers, H_r, cm_r, pe_r, tp_r, fwd_bs)
        pts_r = gather_positions(h_fin_r, r_pos)
        ref_tok_lps: List[torch.Tensor] = []
        for s in range(0, pts_r.shape[0], 32):
            e = min(pts_r.shape[0], s + 32)
            lp = F.log_softmax(lm_head(lm_norm(pts_r[s:e])).float(), dim=-1)
            ref_tok_lps.append(lp[torch.arange(e - s, device=device), r_tgt[s:e]])
        ref_tok_lp_all = torch.cat(ref_tok_lps, dim=0).detach()
        del h_fin_r, pts_r

        # Expand sequence advantage to per-token advantage
        tok_adv_list = []
        for adv_v, c in zip(r_adv.tolist(), r_cnt):
            tok_adv_list.extend([adv_v] * c)
        tok_adv = torch.tensor(tok_adv_list, dtype=torch.float32, device=device)

        h0_c, cm_c, pe_c, tp_c = prepare_layer_inputs_unified(model, cal_enc.input_ids, cal_enc.attention_mask)
        H_c = _chunked_step_range(model, 0, target_layers[0], h0_c, cm_c, pe_c, tp_c, fwd_bs)
        h_fin_c = _chunked_step_range(model, target_layers[0], n_layers, H_c, cm_c, pe_c, tp_c, fwd_bs)
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

    lr_b = 1.6e-3 * ((4096.0 / float(spec.hidden_size)) ** 0.25)
    m_r_toks = max(int(r_tgt.shape[0]), 1)
    m_cal = max(sum(len(p) for p in cal_pos), 1)
    opt_bs = 4 if is_large else 12
    cal_bs = 8 if is_large else 16

    layer_logs = []
    for idx_s, l_idx in enumerate(target_layers):
        Hr_in, Hc_in = H_r.detach(), H_c.detach()
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
                mgr.wrapped[(li, m_name)].set_precond_and_init(v_mat.half().detach(), c_damp, seed_val=14000 * l_idx + idx_m)
                del x_all, x_tilde, sv, v_mat, norm_eig, c_damp
            acts_by_sp.clear()

            mgr.set_all_active(True)
            mgr.freeze_all()
            skip_r = _compute_skip(l_idx, Hr_in, cm_r, pe_r, tp_r, r_pos)
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

        last_grpo = last_cal = 0.0
        for _ in range(steps_per_layer):
            opt.zero_grad(set_to_none=True)
            tok_off = 0
            grpo_acc = 0.0
            n_r = Hr_in.shape[0]
            for s0 in range(0, n_r, opt_bs):
                s1 = min(n_r, s0 + opt_bs)
                cnt = sum(len(p) for p in r_pos[s0:s1])
                t0_i, t1_i = tok_off, tok_off + cnt
                tok_off = t1_i
                with torch.enable_grad():
                    or_out = step_layer_unified(model, l_idx, Hr_in[s0:s1], slice_cm_map(cm_r, s0, s1, n_r), slice_pe(pe_r, s0, s1, n_r), tp_r[s0:s1])
                    hr_fs = gather_positions(or_out, r_pos[s0:s1]) + skip_r[t0_i:t1_i]
                    lg = lm_head(lm_norm(hr_fs)).float()
                    lp = F.log_softmax(lg, dim=-1)
                    pol_lp = lp[torch.arange(t1_i - t0_i, device=device), r_tgt[t0_i:t1_i]]
                    ref_sl = ref_tok_lp_all[t0_i:t1_i]
                    adv_sl = tok_adv[t0_i:t1_i]
                    # Term 1: Clipped GRPO policy objective + Term 2: Cached-RefLogp KL penalty
                    ratio = torch.exp((pol_lp - ref_sl).clamp(-4.0, 4.0))
                    surr1 = ratio * adv_sl
                    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_sl
                    l_pg = -torch.min(surr1, surr2).sum() / float(m_r_toks)
                    l_ref_kl = kl_beta * (torch.exp((ref_sl - pol_lp).clamp(-4.0, 4.0)) - (ref_sl - pol_lp) - 1.0).sum() / float(m_r_toks)
                    (l_pg + l_ref_kl).backward()
                    grpo_acc += float(l_pg.item())
                del or_out, hr_fs, lg, lp, pol_lp, ratio, l_pg, l_ref_kl
            last_grpo = grpo_acc

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
        del opt, skip_r, skip_c
        layer_logs.append({"layer": l_idx, "grpo_loss": round(last_grpo, 5), "cal_kl": round(last_cal, 5)})

        if idx_s + 1 < len(target_layers):
            nxt = target_layers[idx_s + 1]
            with torch.no_grad():
                H_r = _chunked_step_range(model, l_idx, nxt, Hr_in, cm_r, pe_r, tp_r, fwd_bs)
                H_c = _chunked_step_range(model, l_idx, nxt, Hc_in, cm_c, pe_c, tp_c, fwd_bs)
        del Hr_in, Hc_in
        if is_large:
            torch.cuda.empty_cache()

    mgr.freeze_all()
    mgr.set_all_active(True)
    return mgr, {
        "method": "O1_Detached_GRPO",
        "target_layers": target_layers,
        "loss_terms": 4,
        "rollout_stats": r_stats,
        "layer_logs": layer_logs,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }


# ---------------------------------------------------------------------------
# Rule (b): 0-Backprop Advantage-Weighted Null-Space Ridge Update
# ---------------------------------------------------------------------------

@torch.no_grad()
def synthesize_zero_backprop_adv_ridge_rl(
    model: nn.Module,
    tokenizer,
    spec: ModelSpec,
    rl_tasks: List[Dict[str, object]],
    cal_texts: List[str],
    target_layers: List[int],
    device: str,
    mock_module_name: Optional[str] = None,
    mock_module_source: Optional[str] = None,
    r: int = 16,
    lora_alpha: float = 16.0,
    group_size: int = 6,
    eta_rl: float = 0.22,
    existing_mgr: Optional[UnifiedTransferManager] = None,
    seed: int = 42,
) -> Tuple[Optional[UnifiedTransferManager], Dict[str, object]]:
    """Pure torch.no_grad() Advantage-Weighted Rejection / Null-Space Ridge RL (0 .backward())."""
    t0 = time.perf_counter()
    target_layers = sorted(set(int(l) for l in target_layers))
    if existing_mgr is not None:
        existing_mgr.freeze_all()
        existing_mgr.set_all_active(True)

    rollout_records, r_stats = sample_and_verify_rl_rollouts(
        model, tokenizer, spec, rl_tasks, device, group_size=group_size,
        mock_module_name=mock_module_name, mock_module_source=mock_module_source, seed=seed,
    )
    if not rollout_records:
        return None, {
            "method": "ZeroBackprop_AdvWeighted_NullRidge_RL",
            "target_layers": target_layers,
            "skipped_zero_variance": True,
            "rollout_stats": r_stats,
            "elapsed_sec": round(time.perf_counter() - t0, 2),
        }

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
    fwd_bs = 6 if is_large else 20

    r_ids, r_msk, r_pos, r_tgt, r_cnt, r_adv = _tokenize_rollouts(tokenizer, rollout_records, device)
    tok_adv_list = []
    for av, c in zip(r_adv.tolist(), r_cnt):
        tok_adv_list.extend([av] * c)
    tok_adv = torch.tensor(tok_adv_list, dtype=torch.float32, device=device)

    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    cal_enc = tokenizer(cal_texts[:16], return_tensors="pt", padding=True, truncation=True, max_length=96).to(device)
    tokenizer.padding_side = prev_side
    cal_mask_bool = cal_enc.attention_mask.bool()

    hr, cm_r, pe_r, tp_r = prepare_layer_inputs_unified(model, r_ids, r_msk)
    hr = _chunked_step_range(model, 0, target_layers[0], hr, cm_r, pe_r, tp_r, fwd_bs)
    hc, cm_c, pe_c, tp_c = prepare_layer_inputs_unified(model, cal_enc.input_ids, cal_enc.attention_mask)
    hc = _chunked_step_range(model, 0, target_layers[0], hc, cm_c, pe_c, tp_c, fwd_bs)

    layer_logs = []
    for idx_s, l_idx in enumerate(target_layers):
        h_fin = _chunked_step_range(model, l_idx, n_layers, hr, cm_r, pe_r, tp_r, fwd_bs)
        pts = gather_positions(h_fin, r_pos).float()
        del h_fin

        dH_rl = torch.zeros_like(pts)
        for s in range(0, pts.shape[0], 32):
            e = min(pts.shape[0], s + 32)
            hv = pts[s:e]
            td = r_tgt[s:e]
            av = tok_adv[s:e].clamp(-1.5, 1.5).unsqueeze(1)
            inv_rms = torch.rsqrt((hv * hv).mean(dim=-1, keepdim=True) + eps)
            h_hat = hv * inv_rms
            logits = lm_head((h_hat * gamma.unsqueeze(0)).to(W_lm_raw.dtype)).float()
            probs = F.softmax(logits, dim=-1)
            ar = torch.arange(hv.shape[0], device=device)
            gz = probs
            gz[ar, td] -= 1.0
            g_norm = (gz.to(W_lm_raw.dtype) @ W_lm_raw).float() * gamma.unsqueeze(0)
            g_h = inv_rms * (g_norm - (g_norm * h_hat).mean(dim=-1, keepdim=True) * h_hat)
            u = (-g_h) / g_h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            h_rms = hv.norm(dim=-1, keepdim=True)
            dH_rl[s:e] = av * (eta_rl * h_rms / max(len(target_layers), 1)) * u
            del logits, probs, gz, g_norm, g_h

        lt = get_layer_type(model, l_idx)
        for mod_key in ("o_proj" if lt != "linear_attention" else "out_proj", "down_proj"):
            w = mgr.wrapped[(l_idx, mod_key)]
            xr_list, xc_list = [], []

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
                    h_rem = w.register_forward_hook(hk)
                    _ = step_layer_unified(model, l_idx, hb[s0:e0], slice_cm_map(cmb, s0, e0, nb), slice_pe(peb, s0, e0, nb), tpb[s0:e0])
                    h_rem.remove()

            _cap(hr, cm_r, pe_r, tp_r, r_pos, True, xr_list)
            _cap(hc, cm_c, pe_c, tp_c, cal_mask_bool, False, xc_list)
            X_r = torch.cat(xr_list, dim=0)
            X_c = torch.cat(xc_list, dim=0).to(device)
            if X_c.shape[0] > 768:
                X_c = X_c[:: max(1, X_c.shape[0] // 768)][:768]

            d_in = X_r.shape[1]
            c_mat = (X_c.T @ X_c) / max(X_c.shape[0], 1)
            c_scale = max(float(torch.trace(c_mat).item() / max(d_in, 1)), 1e-8)
            c_norm = c_mat * (6.0 / c_scale)
            if d_in <= 4096:
                ev, evec = torch.linalg.eigh(0.5 * (c_norm + c_norm.T) + torch.eye(d_in, device=device))
                S_inv = (evec / torch.sqrt(ev.clamp(min=0.5)).unsqueeze(0)) @ evec.T
                Xw = X_r @ S_inv
                apply_sinv = lambda M: M @ S_inv
                del ev, evec
            else:
                Uk, Sk, _ = torch.svd_lowrank(c_norm, q=min(384, d_in - 2), niter=2)
                shrink = 1.0 - torch.rsqrt(1.0 + Sk.clamp(min=0.0))
                apply_sinv = lambda M: M - ((M @ Uk) * shrink.unsqueeze(0)) @ Uk.T
                Xw = apply_sinv(X_r)

            dY = 0.5 * dH_rl
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
            if fro > 3.0:
                B_fit = B_fit * (3.0 / (fro + 1e-8))
            w.lora_A.data.copy_(A_fit / max(w.scaling, 1e-6))
            w.lora_B.data.copy_(B_fit)
            w.active = True
            del X_r, X_c, c_mat, c_norm, Xw, Theta_w, U, S, Vh

        layer_logs.append({"layer": l_idx, "dh_rms": round(float(dH_rl.norm(dim=-1).mean().item()), 4)})
        if idx_s + 1 < len(target_layers):
            nxt = target_layers[idx_s + 1]
            hr = _chunked_step_range(model, l_idx, nxt, hr, cm_r, pe_r, tp_r, fwd_bs)
            hc = _chunked_step_range(model, l_idx, nxt, hc, cm_c, pe_c, tp_c, fwd_bs)

    return mgr, {
        "method": "ZeroBackprop_AdvWeighted_NullRidge_RL",
        "target_layers": target_layers,
        "loss_terms": 1,
        "rollout_stats": r_stats,
        "layer_logs": layer_logs,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }
