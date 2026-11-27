"""Iteration 15: Iter 8+9+13 Synthesis Unified Transfer Block (SmolLM2-360M & 8-bit Ornith-1.0-9B).

Consolidates ONLY verified winners from Iter 8, 9, and 13:
  1. Target Layers (`select_v15_layers`):
     - Hybrid SSM+Attn (`9B`, 32L): `[23, 27, 31]` (`3` `full_attention` blocks, `9` steps,
       `48` negs + `34` enc + `16` cal -> `~105-113s < 120s`, keeping `L23` active without explosion).
     - Pure Attn (`360M`, 32L): `[25, 27, 29, 30, 31]` (`12` steps -> away from shallow `L21` underfit
       while sparing early `L25..27` Pasteur/retention routing, `~17s < 20s`).
  2. Sweet-Spot `lr_A` (`0.38 * sqrt(960 / d_in) * lr_b`) + `lr_b = 2.75e-3 * (4096 / d)^0.26`:
     Avoids both Iter 13's `0.28` slowdown and Iter 14's `0.78` null-space drift (`12->120`, `cobalt->Peanuts`),
     combined with stronger `alpha_null = 8.0` damping (`V_protect`).
  3. Iter 8 Proven CoT Guard + Zero `"Yes"`/`"No"` Suppression:
     Uses bounded `.mean()` `l_cot_ref` at threshold `-9.0` (stops `"The user is asking..."` cleanly
     without Iter 12's `28.16` `.sum()` explosion or Iter 14's `-6.5` leak), plus tiny `0.008 * sum()` on
     bare digits `'0'..'9'` ONLY.
  4. High-Density Sibling Anti-Bleed (`52` negs, `15`-tok rollouts, `"was"/"age 12"`, `"198_"`, `"201_"`,
     `"A"`, `"The"` traps with `lambda_antibleed = 1.25`, proven `bleed <= 0.05` in Iter 9 & 12).
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from bio_logic_benchmark import (
    SEEDS,
    extract_bio_clauses_from_doc,
    generate_bio_batch,
)
from eval_harness import disable_adapters_ctx
from guards import NoBackpropGuard, cleanup_cuda
from iter_0_transfer_baselines import load_smoltalk_for_scale
from iter_1_transfer_cot_block import (
    _ENC_CAL_ANCHORS,
    _lookup_base_oracle_logic,
    tokenize_pos_probes,
    UnifiedTransferManager,
)
from iter_3_hybrid_svd_transfer import _BIO_EXTRA_ANCHORS, _init_svd_null_lora
from iter_4_whitened_transfer import _chunked_batched_generate_short, _chunked_step_range
from iter_6_unified_transfer import _EXTRA_SCIENCE_ANCHORS
from iter_7_unified_transfer import (
    _collect_digit_token_ids,
    _collect_extended_bleed_ids,
)
from iter_8_unified_transfer import _EXTRA_FRENCH_SCIENCE_ANCHORS
from iter_10_unified_transfer import _collect_v10_meta_refusal_ids
from iter_12_unified_transfer import _EXTRA_FRENCH_MED_NEIGHBORS
from iter_14_unified_transfer import _build_iter14_probes
from probes import gather_positions
from transfer_eval import (
    LAB_DIR,
    SYS_QA,
    append_transfer_record,
    check_transfer_gates,
    evaluate_bio_transfer,
    get_bio_base_reference,
)
from unified_model_utils import (
    apply_chat_safe,
    get_eos_token_ids,
    get_layer_type,
    get_layers,
    get_lm_head,
    get_lm_norm,
    get_text_model,
    load_model_by_scale,
    prepare_layer_inputs_unified,
    slice_cm_map,
    slice_pe,
    step_layer_unified,
)


def select_v15_layers(model: nn.Module) -> List[int]:
    """Select `[23, 27, 31]` on 32L hybrid models and `[25, 27, 29, 30, 31]` on 32L pure-attention models."""
    n_layers = len(get_layers(model))
    has_linear = any(get_layer_type(model, i) == "linear_attention" for i in range(n_layers))
    if has_linear:
        full_attn = [i for i in range(n_layers) if get_layer_type(model, i) != "linear_attention"]
        return full_attn[-3:]
    cand = [max(0, n_layers - 7), max(0, n_layers - 5), max(0, n_layers - 3), max(0, n_layers - 2), n_layers - 1]
    return sorted(set(cand))


def synthesize_unified_transfer_v15(
    model: nn.Module,
    tokenizer,
    clauses: List[str],
    cal_texts: List[str],
    device: str,
    target_layers: Optional[List[int]] = None,
    r: int = 16,
    lora_alpha: float = 20.0,
    alpha_null: float = 8.0,
    n_null_rank: int = 96,
) -> Tuple[UnifiedTransferManager, Dict[str, object]]:
    """Run Iter 15 cross-scale synthesis consolidating Iter 8+9+13 mechanisms."""
    layers = get_layers(model)
    n_layers = len(layers)
    if target_layers is None:
        target_layers = select_v15_layers(model)

    mgr = UnifiedTransferManager(model, target_layers=target_layers, r=r, lora_alpha=lora_alpha)
    bundle = _build_iter14_probes(clauses, tokenizer)
    pos_items: List[Dict[str, object]] = bundle["pos_items"]  # type: ignore[assignment]
    cot_entry_prompts: List[str] = bundle["cot_entry_prompts"]  # type: ignore[assignment]
    neg_roll_strs: List[str] = list(bundle["neg_rollout_prompts"])[:24]  # type: ignore[arg-type]
    neg_cf_strs: List[str] = list(bundle["neg_cf_mid_prompts"])[:28]  # type: ignore[arg-type]

    enc_qs = (
        list(_ENC_CAL_ANCHORS)
        + list(_EXTRA_SCIENCE_ANCHORS)
        + list(_EXTRA_FRENCH_SCIENCE_ANCHORS)
        + list(_EXTRA_FRENCH_MED_NEIGHBORS)
        + list(_BIO_EXTRA_ANCHORS[:4])
    )  # 34 curated anchors
    enc_prompt_strs = [
        apply_chat_safe(
            tokenizer,
            [{"role": "system", "content": SYS_QA}, {"role": "user", "content": q}],
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for q in enc_qs
    ]

    stu_ids, stu_mask, stu_pos, target_tok_ids, tok_weights, _ = tokenize_pos_probes(
        tokenizer, pos_items, device
    )

    digit_tok_ids = _collect_digit_token_ids(tokenizer)
    digit_tok_set = set(digit_tok_ids)
    digit_tok_tensor = torch.tensor(digit_tok_ids, dtype=torch.long, device=device)

    trail_mask_list: List[bool] = []
    pos0_idx_list: List[int] = []
    pos0_row_mask: List[bool] = []
    deep_vocab_mask: List[bool] = []
    num_tok_mask: List[bool] = []
    injected_first = set()
    cot_target_first = set()
    for it in pos_items:
        tid = tokenizer.encode(str(it["target"]), add_special_tokens=False)
        sp_tid = tokenizer.encode(str(it["space_prefix"]) + str(it["target"]), add_special_tokens=False)
        if tid:
            injected_first.add(int(tid[0]))
        if sp_tid:
            cot_target_first.add(int(sp_tid[0]))

    all_bio_first = sorted(injected_first | cot_target_first)
    bio_first_toks = torch.tensor(all_bio_first, dtype=torch.long, device=device)

    tok_ptr = 0
    for idx_i, it in enumerate(pos_items):
        p_len = len(stu_pos[idx_i])
        is_num_tgt = str(it["target"]).strip().isdigit()
        for k in range(p_len):
            tk_id = int(target_tok_ids[tok_ptr + k].item())
            pos0_row_mask.append(k == 0)
            if k == 0:
                pos0_idx_list.append(tk_id)
            dec = tokenizer.decode([tk_id])
            is_trail = bool(it["mode"] == "cot_mid" and it["trailing"] and k == p_len - 1)
            trail_mask_list.append(is_trail)
            deep_vocab_mask.append(bool(k < 3 and not is_trail and any(ch.isalnum() for ch in dec)))
            num_tok_mask.append(bool(is_num_tgt and tk_id in digit_tok_set))
        tok_ptr += p_len

    trail_bmask = torch.tensor(trail_mask_list, dtype=torch.bool, device=device)
    pos0_bmask = torch.tensor(pos0_row_mask, dtype=torch.bool, device=device)
    deep_bmask = torch.tensor(deep_vocab_mask, dtype=torch.bool, device=device)
    num_bmask = torch.tensor(num_tok_mask, dtype=torch.bool, device=device)
    comp_bio_mask = torch.tensor(
        [[t_b != t0_id for t_b in all_bio_first] for t0_id in pos0_idx_list],
        dtype=torch.bool,
        device=device,
    )

    bleed_list = _collect_extended_bleed_ids(tokenizer, clauses)
    bleed_tok_ids = torch.tensor(sorted(set(bleed_list) | set(all_bio_first)), dtype=torch.long, device=device)
    refusal_tok_ids = torch.tensor(_collect_v10_meta_refusal_ids(tokenizer), dtype=torch.long, device=device)
    blurt_digit_ids = torch.tensor(digit_tok_ids, dtype=torch.long, device=device)
    eos_list = get_eos_token_ids(tokenizer)
    eos_tok_ids = torch.tensor(eos_list, dtype=torch.long, device=device)

    lm_norm = get_lm_norm(model)
    lm_head = get_lm_head(model)
    prev_side = tokenizer.padding_side
    d_model = int(get_text_model(model).config.hidden_size)
    is_large_model = d_model >= 2048
    fwd_bs = 14 if is_large_model else 128
    gen_bs = 14 if is_large_model else 64

    with torch.no_grad():
        mgr.freeze_all()
        mgr.set_all_active(False)

        neg_gens = _chunked_batched_generate_short(model, tokenizer, neg_roll_strs, device, batch_size=gen_bs, max_new_tokens=15)
        neg_seqs, neg_pos, neg_w = [], [], []
        for idx_n, clean_gen in enumerate(neg_gens):
            p_ids = tokenizer.encode(neg_roll_strs[idx_n], add_special_tokens=False)
            neg_seqs.append(p_ids + clean_gen)
            n_pred = max(1, len(clean_gen))
            neg_pos.append(list(range(len(p_ids) - 1, len(p_ids) - 1 + n_pred)))
            neg_w.extend([3.0] + [1.4] * (n_pred - 1))

        for cf_p in neg_cf_strs:
            p_ids = tokenizer.encode(cf_p, add_special_tokens=False)
            neg_seqs.append(p_ids)
            neg_pos.append([len(p_ids) - 1])
            neg_w.append(4.4)

        neg_tok_weights = torch.tensor(neg_w, dtype=torch.float32, device=device)
        neg_tok_weights = neg_tok_weights / neg_tok_weights.mean().clamp(min=1e-6)
        max_nl = max(len(s) for s in neg_seqs)
        neg_ids = torch.full((len(neg_seqs), max_nl), tokenizer.pad_token_id, dtype=torch.long, device=device)
        neg_mask = torch.zeros((len(neg_seqs), max_nl), dtype=torch.long, device=device)
        for i, s in enumerate(neg_seqs):
            neg_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            neg_mask[i, : len(s)] = 1

        enc_gens = _chunked_batched_generate_short(model, tokenizer, enc_prompt_strs, device, batch_size=gen_bs, max_new_tokens=5)
        enc_seqs, enc_pos = [], []
        for idx_e, eg in enumerate(enc_gens):
            p_ids = tokenizer.encode(enc_prompt_strs[idx_e], add_special_tokens=False)
            enc_seqs.append(p_ids + eg)
            n_e = max(1, len(eg))
            enc_pos.append(list(range(len(p_ids) - 1, len(p_ids) - 1 + n_e)))
        max_el = max(len(s) for s in enc_seqs)
        enc_ids = torch.full((len(enc_seqs), max_el), tokenizer.pad_token_id, dtype=torch.long, device=device)
        enc_mask = torch.zeros((len(enc_seqs), max_el), dtype=torch.long, device=device)
        for i, s in enumerate(enc_seqs):
            enc_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            enc_mask[i, : len(s)] = 1

        tokenizer.padding_side = "right"
        cal_texts_sel = cal_texts[:16]
        cal_enc = tokenizer(cal_texts_sel, return_tensors="pt", padding=True, truncation=True, max_length=92).to(device)
        cal_lens = cal_enc.attention_mask.sum(dim=1).tolist()
        cal_pos = [[max(0, min(L - 1, int(k * L / 4))) for k in range(1, 4)] for L in cal_lens]

        cot_tok = tokenizer(cot_entry_prompts, return_tensors="pt", padding=True, truncation=True, max_length=132).to(device)
        cot_lens = cot_tok.attention_mask.sum(dim=1).tolist()
        cot_pos = [[max(0, L - 1)] for L in cot_lens]
        tokenizer.padding_side = prev_side

        def get_base_logits_and_h_start(ids: torch.Tensor, mask: torch.Tensor, pos_lists: List[List[int]], need_logits: bool = True):
            h0, cm0, pe0, tp0 = prepare_layer_inputs_unified(model, ids, mask)
            h_entry = _chunked_step_range(model, 0, target_layers[0], h0, cm0, pe0, tp0, chunk_bs=fwd_bs)
            if not need_logits:
                return None, h_entry, cm0, pe0, tp0
            h_fin = _chunked_step_range(model, target_layers[0], n_layers, h_entry, cm0, pe0, tp0, chunk_bs=fwd_bs)
            pts = gather_positions(h_fin, pos_lists)
            lp_chunks = []
            for s in range(0, pts.shape[0], 32):
                e = min(pts.shape[0], s + 32)
                lp_chunks.append(F.log_softmax(lm_head(lm_norm(pts[s:e])).float(), dim=-1).cpu())
            del h_fin, pts
            return torch.cat(lp_chunks, dim=0), h_entry, cm0, pe0, tp0

        lp_cal_cpu, H_cal, cm_cal, pe_cal, tp_cal = get_base_logits_and_h_start(
            cal_enc.input_ids, cal_enc.attention_mask, cal_pos, need_logits=True
        )
        lp_enc_cpu, H_enc, cm_enc, pe_enc, tp_enc = get_base_logits_and_h_start(
            enc_ids, enc_mask, enc_pos, need_logits=True
        )
        lp_neg_cpu, H_neg, cm_neg, pe_neg, tp_neg = get_base_logits_and_h_start(
            neg_ids, neg_mask, neg_pos, need_logits=True
        )
        _, H_cot, cm_cot, pe_cot, tp_cot = get_base_logits_and_h_start(
            cot_tok.input_ids, cot_tok.attention_mask, cot_pos, need_logits=False
        )
        _, H_stu, cm_stu, pe_stu, tp_stu = get_base_logits_and_h_start(
            stu_ids, stu_mask, stu_pos, need_logits=False
        )

        prob_dev = "cpu" if is_large_model else device
        p_cal_store = lp_cal_cpu.exp().half().to(prob_dev)
        p_enc_store = lp_enc_cpu.exp().half().to(prob_dev)
        p_neg_store = lp_neg_cpu.exp().half().to(prob_dev)
        logp_base_neg_bleed = lp_neg_cpu[:, bleed_tok_ids.cpu()].float().to(device)
        del lp_cal_cpu, lp_enc_cpu, lp_neg_cpu

        cal_mask_bool = cal_enc.attention_mask.bool()
        enc_mask_bool = enc_mask.bool()
        neg_boundary_pos = [[p[0]] for p in neg_pos]
        if is_large_model:
            torch.cuda.empty_cache()

    # Sweet-spot width-invariant rate (`9B: 2.75e-3` x 9 steps, `360M: 4.02e-3` x 12 steps)
    lr_b = 2.75e-3 * ((4096.0 / float(d_model)) ** 0.26)
    min_steps = 5
    max_steps = 9 if is_large_model else 12

    lambda_cal_kl = 4.2
    lambda_enc_kl = 8.0
    lambda_neg_kl = 9.0
    lambda_antibleed = 1.25
    lambda_cot_entry = 2.2
    lambda_refusal = 0.70
    lambda_echo = 1.2
    lambda_bio_margin = 0.42
    lambda_vocab_margin = 0.22
    lambda_digit_margin = 1.75
    lambda_dw = 0.0025

    layer_logs = []
    stu_boundary_pos = [[p[0]] for p in stu_pos]

    def chunked_compute_skip(h_in, cm_m, pe_t, tp_t, pos_lists):
        n = h_in.shape[0]
        if n <= fwd_bs:
            hl = step_layer_unified(model, l_idx, h_in, cm_m, pe_t, tp_t)
            hc = hl
            for j in range(l_idx + 1, n_layers):
                hc = step_layer_unified(model, j, hc, cm_m, pe_t, tp_t)
            return gather_positions(hc - hl, pos_lists).detach()
        skips = []
        for s in range(0, n, fwd_bs):
            e = min(n, s + fwd_bs)
            hl = step_layer_unified(model, l_idx, h_in[s:e], slice_cm_map(cm_m, s, e, n), slice_pe(pe_t, s, e, n), tp_t[s:e])
            hc = hl
            for j in range(l_idx + 1, n_layers):
                hc = step_layer_unified(model, j, hc, slice_cm_map(cm_m, s, e, n), slice_pe(pe_t, s, e, n), tp_t[s:e])
            skips.append(gather_positions(hc - hl, pos_lists[s:e]).detach())
        return torch.cat(skips, dim=0)

    for idx_sched, l_idx in enumerate(target_layers):
        H_stu_in = H_stu.detach()
        H_neg_in = H_neg.detach()
        H_cal_in = H_cal.detach()
        H_enc_in = H_enc.detach()
        H_cot_in = H_cot.detach()

        with torch.no_grad():
            mgr.freeze_all()
            mgr.set_all_active(False)
            spaces = ("attn_in", "mixer_out", "mlp_in", "down_proj")
            hooks = [(sp, mgr.rep_module_for_space(l_idx, sp)) for sp in spaces]

            acts_by_sp: Dict[str, List[torch.Tensor]] = {sp: [] for sp in spaces}

            def _capture_batch(h_b, cm_b, pe_b, tp_b, pos_or_mask, is_pos_gather: bool, weight_scale: float):
                n_b = h_b.shape[0]
                for s in range(0, n_b, fwd_bs):
                    e = min(n_b, s + fwd_bs)
                    sq_len, b_cur = h_b.shape[1], e - s
                    def _mk(sp_n: str):
                        def hk(m, inp, out):
                            if is_pos_gather:
                                x_3d = inp[0].detach().reshape(b_cur, sq_len, -1)
                                x_g = gather_positions(x_3d, pos_or_mask[s:e]).float()
                            else:
                                mb_flat = pos_or_mask[s:e].reshape(-1)
                                x_g = inp[0].detach().reshape(-1, inp[0].shape[-1])[mb_flat].float()
                            if x_g.shape[0] > 260:
                                stride = max(1, x_g.shape[0] // 260)
                                x_g = x_g[::stride][:260]
                            acts_by_sp[sp_n].append((x_g * weight_scale).cpu())
                        return hk
                    live = [rep_mod.register_forward_hook(_mk(sp)) for sp, rep_mod in hooks]
                    _ = step_layer_unified(
                        model, l_idx, h_b[s:e],
                        slice_cm_map(cm_b, s, e, n_b), slice_pe(pe_b, s, e, n_b), tp_b[s:e],
                    )
                    for h_rem in live:
                        h_rem.remove()

            _capture_batch(H_cal_in, cm_cal, pe_cal, tp_cal, cal_mask_bool, False, 1.0)
            _capture_batch(H_enc_in, cm_enc, pe_enc, tp_enc, enc_mask_bool, False, 2.0)
            _capture_batch(H_neg_in, cm_neg, pe_neg, tp_neg, neg_boundary_pos, True, 2.2)

            stu_chunks: Dict[str, List[torch.Tensor]] = {sp: [] for sp in spaces}
            n_stu_seq = H_stu_in.shape[0]
            for s in range(0, n_stu_seq, fwd_bs):
                e = min(n_stu_seq, s + fwd_bs)
                b_cur, sq_len = e - s, H_stu_in.shape[1]
                def _mk_s(sp_n: str):
                    def hk(m, inp, out):
                        x_3d = inp[0].detach().reshape(b_cur, sq_len, -1)
                        stu_chunks[sp_n].append(gather_positions(x_3d, stu_boundary_pos[s:e]).float().detach())
                    return hk
                live_s = [rep_mod.register_forward_hook(_mk_s(sp)) for sp, rep_mod in hooks]
                _ = step_layer_unified(
                    model, l_idx, H_stu_in[s:e],
                    slice_cm_map(cm_stu, s, e, n_stu_seq), slice_pe(pe_stu, s, e, n_stu_seq), tp_stu[s:e],
                )
                for h_rem in live_s:
                    h_rem.remove()
            stu_acts = {sp: torch.cat(chunks, dim=0) for sp, chunks in stu_chunks.items()}

            v_by_sp: Dict[str, torch.Tensor] = {}
            c_by_sp: Dict[str, torch.Tensor] = {}
            for sp, x_chunks in acts_by_sp.items():
                x_all = torch.cat(x_chunks, dim=0)
                if x_all.shape[0] > 768:
                    stride = max(1, x_all.shape[0] // 768)
                    x_all = x_all[::stride][:768]
                x_gpu = x_all.to(device=device, dtype=torch.float32)
                x_tilde = x_gpu / math.sqrt(max(x_gpu.shape[0], 1))
                mean_var = float((x_tilde * x_tilde).sum().item()) / float(max(x_tilde.shape[1], 1))
                q_k = min(n_null_rank, x_tilde.shape[0] - 2, x_tilde.shape[1] - 2)
                _, sv, v_mat = torch.svd_lowrank(x_tilde, q=max(8, q_k), niter=2)
                norm_eig = (sv * sv) / max(mean_var, 1e-8)
                v_by_sp[sp] = v_mat.half().detach()
                c_by_sp[sp] = (1.0 - torch.rsqrt(1.0 + alpha_null * norm_eig)).float().detach()
                del x_gpu, x_tilde, sv, v_mat, norm_eig
            acts_by_sp.clear()

            for idx_m, ((li, m_name), (_, _, sp)) in enumerate(mgr.mod_loc.items()):
                if li == l_idx:
                    _init_svd_null_lora(
                        mgr.wrapped[(li, m_name)], v_by_sp[sp], c_by_sp[sp], stu_acts[sp], seed_val=15000 * l_idx + idx_m
                    )
            del v_by_sp, c_by_sp, stu_acts

            mgr.set_all_active(True)
            mgr.freeze_all()
            skip_stu = chunked_compute_skip(H_stu_in, cm_stu, pe_stu, tp_stu, stu_pos)
            skip_neg = chunked_compute_skip(H_neg_in, cm_neg, pe_neg, tp_neg, neg_pos)
            skip_cal = chunked_compute_skip(H_cal_in, cm_cal, pe_cal, tp_cal, cal_pos)
            skip_enc = chunked_compute_skip(H_enc_in, cm_enc, pe_enc, tp_enc, enc_pos)
            skip_cot = chunked_compute_skip(H_cot_in, cm_cot, pe_cot, tp_cot, cot_pos)
            if is_large_model:
                torch.cuda.empty_cache()

        mgr.activate_for_layer(l_idx)
        opt_groups = []
        params_all = []
        for (li, _), w in mgr.wrapped.items():
            if li == l_idx:
                lr_a_w = lr_b * 0.38 * math.sqrt(960.0 / float(max(w.in_features, 1)))
                opt_groups.append({"params": [w.lora_A], "lr": lr_a_w, "weight_decay": 1e-3})
                opt_groups.append({"params": [w.lora_B], "lr": lr_b, "weight_decay": 2.5e-3})
                params_all.extend([w.lora_A, w.lora_B])
        opt = torch.optim.AdamW(opt_groups)

        m_stu = int(target_tok_ids.shape[0])
        stu_chunk = 14 if is_large_model else 256
        cal_chunk = 14 if is_large_model else 256
        enc_chunk = 14 if is_large_model else 256
        neg_chunk = 14 if is_large_model else 256
        cot_chunk = 9 if is_large_model else 256
        n_cal_seq = H_cal_in.shape[0]
        n_enc_seq = H_enc_in.shape[0]
        n_neg_seq = H_neg_in.shape[0]
        n_cot_seq = H_cot_in.shape[0]
        m_cal_tot = max(sum(len(p) for p in cal_pos), 1)
        m_enc_tot = max(sum(len(p) for p in enc_pos), 1)
        m_neg_tot = max(int(neg_tok_weights.numel()), 1)
        n_pos0_tot = max(int(pos0_bmask.sum().item()), 1)
        n_deep_tot = max(int(deep_bmask.sum().item()), 1)
        n_num_tot = max(int(num_bmask.sum().item()), 1)

        last_ce = last_cal = last_enc = last_nkl = last_cot = 0.0
        steps_used = 0
        for step in range(max_steps):
            steps_used = step + 1
            opt.zero_grad(set_to_none=True)

            n_stu_seq = H_stu_in.shape[0]
            ce_accum = 0.0
            tok_off = 0
            pos0_off = 0
            for s0 in range(0, n_stu_seq, stu_chunk):
                s1 = min(n_stu_seq, s0 + stu_chunk)
                cnt = sum(len(p) for p in stu_pos[s0:s1])
                t0, t1 = tok_off, tok_off + cnt
                tok_off = t1
                p0_cnt = int(pos0_bmask[t0:t1].sum().item())
                p0_0, p0_1 = pos0_off, pos0_off + p0_cnt
                pos0_off = p0_1

                with torch.enable_grad():
                    out_s = step_layer_unified(
                        model, l_idx, H_stu_in[s0:s1],
                        slice_cm_map(cm_stu, s0, s1, n_stu_seq),
                        slice_pe(pe_stu, s0, s1, n_stu_seq),
                        tp_stu[s0:s1],
                    )
                    h_fs = gather_positions(out_s, stu_pos[s0:s1]) + skip_stu[t0:t1]
                    logits_s = lm_head(lm_norm(h_fs)).float()
                    ce_v = F.cross_entropy(logits_s, target_tok_ids[t0:t1], reduction="none")
                    l_ce = (ce_v * tok_weights[t0:t1]).sum() / float(max(m_stu, 1))

                    extra_loss = torch.zeros((), device=device)
                    p0m = pos0_bmask[t0:t1]
                    if p0m.any():
                        logits_p0 = logits_s[p0m]
                        ar_p0 = torch.arange(logits_p0.shape[0], device=device)
                        tgt_p0 = target_tok_ids[t0:t1][p0m]
                        l_true = logits_p0[ar_p0, tgt_p0].unsqueeze(1)
                        l_comp = logits_p0[:, bio_first_toks]
                        cmask = comp_bio_mask[p0_0:p0_1]
                        margin_err = F.relu(l_comp - l_true + 2.8) * cmask.float()
                        extra_loss = extra_loss + lambda_bio_margin * (margin_err.pow(2).sum() / float(max(n_pos0_tot * bio_first_toks.numel(), 1)))
                        lp_p0 = F.log_softmax(logits_p0, dim=-1)
                        extra_loss = extra_loss + (
                            lambda_refusal
                            * (F.relu(lp_p0[:, refusal_tok_ids] - (-8.0)) ** 2).mean()
                            * (float(p0m.sum().item()) / float(max(n_pos0_tot, 1)))
                        )

                    dm = deep_bmask[t0:t1]
                    if dm.any():
                        logits_dm = logits_s[dm]
                        tgt_dm = target_tok_ids[t0:t1][dm]
                        ar_dm = torch.arange(logits_dm.shape[0], device=device)
                        l_true_dm = logits_dm[ar_dm, tgt_dm].unsqueeze(1)
                        logits_dm_mask = logits_dm.clone()
                        logits_dm_mask[ar_dm, tgt_dm] = -1e4
                        top_w_dm = logits_dm_mask.max(dim=-1, keepdim=True).values
                        v_err = F.relu(top_w_dm - l_true_dm + 1.4)
                        extra_loss = extra_loss + lambda_vocab_margin * (v_err.pow(2).sum() / float(max(n_deep_tot, 1)))

                    nm = num_bmask[t0:t1]
                    if nm.any():
                        logits_nm = logits_s[nm]
                        tgt_nm = target_tok_ids[t0:t1][nm]
                        ar_nm = torch.arange(logits_nm.shape[0], device=device)
                        d_true = logits_nm[ar_nm, tgt_nm].unsqueeze(1)
                        d_all = logits_nm[:, digit_tok_tensor]
                        d_mask = (digit_tok_tensor.unsqueeze(0) != tgt_nm.unsqueeze(1)).float()
                        d_err = F.relu(d_all - d_true + 3.8) * d_mask
                        extra_loss = extra_loss + lambda_digit_margin * (d_err.pow(2).sum() / float(max(n_num_tot * digit_tok_tensor.numel(), 1)))

                    tm = trail_bmask[t0:t1]
                    if tm.any():
                        lp_tr = F.log_softmax(logits_s[tm], dim=-1)
                        extra_loss = extra_loss + (
                            lambda_echo
                            * (F.relu(lp_tr[:, bleed_tok_ids] - (-8.5)) ** 2).mean()
                            * (float(tm.sum().item()) / float(max(int(trail_bmask.sum().item()), 1)))
                        )
                    (l_ce + extra_loss).backward()
                    ce_accum += float(l_ce.item())
                del out_s, h_fs, logits_s, ce_v, l_ce, extra_loss
            last_ce = ce_accum

            cal_off = 0
            cal_accum = 0.0
            for cs0 in range(0, n_cal_seq, cal_chunk):
                cs1 = min(n_cal_seq, cs0 + cal_chunk)
                ccnt = sum(len(p) for p in cal_pos[cs0:cs1])
                ct0, ct1 = cal_off, cal_off + ccnt
                cal_off = ct1
                with torch.enable_grad():
                    out_c = step_layer_unified(
                        model, l_idx, H_cal_in[cs0:cs1],
                        slice_cm_map(cm_cal, cs0, cs1, n_cal_seq),
                        slice_pe(pe_cal, cs0, cs1, n_cal_seq),
                        tp_cal[cs0:cs1],
                    )
                    pts_c = gather_positions(out_c, cal_pos[cs0:cs1]) + skip_cal[ct0:ct1]
                    lp_c = F.log_softmax(lm_head(lm_norm(pts_c)).float(), dim=-1)
                    pc_sl = p_cal_store[ct0:ct1].to(device=device, dtype=torch.float32)
                    l_cal = (pc_sl * (pc_sl.clamp(min=1e-12).log() - lp_c)).sum(dim=-1).sum() / float(m_cal_tot)
                    loss_c = lambda_cal_kl * l_cal
                    if cs0 == 0:
                        loss_c = loss_c + lambda_dw * mgr.get_layer_frob_reg(l_idx)
                    loss_c.backward()
                    cal_accum += float(l_cal.item())
                del out_c, pts_c, lp_c, pc_sl, l_cal, loss_c
            last_cal = cal_accum

            enc_off = 0
            enc_accum = 0.0
            for es0 in range(0, n_enc_seq, enc_chunk):
                es1 = min(n_enc_seq, es0 + enc_chunk)
                ecnt = sum(len(p) for p in enc_pos[es0:es1])
                et0, et1 = enc_off, enc_off + ecnt
                enc_off = et1
                with torch.enable_grad():
                    out_e = step_layer_unified(
                        model, l_idx, H_enc_in[es0:es1],
                        slice_cm_map(cm_enc, es0, es1, n_enc_seq),
                        slice_pe(pe_enc, es0, es1, n_enc_seq),
                        tp_enc[es0:es1],
                    )
                    pts_e = gather_positions(out_e, enc_pos[es0:es1]) + skip_enc[et0:et1]
                    lp_e = F.log_softmax(lm_head(lm_norm(pts_e)).float(), dim=-1)
                    pe_sl = p_enc_store[et0:et1].to(device=device, dtype=torch.float32)
                    l_enc = (pe_sl * (pe_sl.clamp(min=1e-12).log() - lp_e)).sum(dim=-1).sum() / float(m_enc_tot)
                    (lambda_enc_kl * l_enc).backward()
                    enc_accum += float(l_enc.item())
                del out_e, pts_e, lp_e, pe_sl, l_enc
            last_enc = enc_accum

            neg_off = 0
            nkl_accum = 0.0
            for ns0 in range(0, n_neg_seq, neg_chunk):
                ns1 = min(n_neg_seq, ns0 + neg_chunk)
                ncnt = sum(len(p) for p in neg_pos[ns0:ns1])
                nt0, nt1 = neg_off, neg_off + ncnt
                neg_off = nt1
                with torch.enable_grad():
                    out_n = step_layer_unified(
                        model, l_idx, H_neg_in[ns0:ns1],
                        slice_cm_map(cm_neg, ns0, ns1, n_neg_seq),
                        slice_pe(pe_neg, ns0, ns1, n_neg_seq),
                        tp_neg[ns0:ns1],
                    )
                    pts_n = gather_positions(out_n, neg_pos[ns0:ns1]) + skip_neg[nt0:nt1]
                    log_p_n = F.log_softmax(lm_head(lm_norm(pts_n)).float(), dim=-1)
                    pn_sl = p_neg_store[nt0:nt1].to(device=device, dtype=torch.float32)
                    tok_nkl = (pn_sl * (pn_sl.clamp(min=1e-12).log() - log_p_n)).sum(dim=-1)
                    l_nkl = (tok_nkl * neg_tok_weights[nt0:nt1]).sum() / float(m_neg_tot)
                    bleed_excess = F.relu(log_p_n[:, bleed_tok_ids] - logp_base_neg_bleed[nt0:nt1]) ** 2
                    l_bleed = bleed_excess.sum(dim=-1).sum() / float(m_neg_tot)
                    (lambda_neg_kl * l_nkl + lambda_antibleed * l_bleed).backward()
                    nkl_accum += float(l_nkl.item())
                del out_n, pts_n, log_p_n, pn_sl, tok_nkl, l_nkl, bleed_excess, l_bleed
            last_nkl = nkl_accum

            # Bounded Iter 8-style CoT guard at `-9.0` `.mean()` + light digit blurt `0.008`
            cot_accum = 0.0
            for c0 in range(0, n_cot_seq, cot_chunk):
                c1 = min(n_cot_seq, c0 + cot_chunk)
                with torch.enable_grad():
                    out_ct = step_layer_unified(
                        model, l_idx, H_cot_in[c0:c1],
                        slice_cm_map(cm_cot, c0, c1, n_cot_seq),
                        slice_pe(pe_cot, c0, c1, n_cot_seq),
                        tp_cot[c0:c1],
                    )
                    pts_ct = gather_positions(out_ct, cot_pos[c0:c1]) + skip_cot[c0:c1]
                    log_p_ct = F.log_softmax(lm_head(lm_norm(pts_ct)).float(), dim=-1)
                    l_cot_eos = (F.relu(log_p_ct[:, eos_tok_ids] - (-12.0)) ** 2).mean()
                    l_cot_ref = (F.relu(log_p_ct[:, refusal_tok_ids] - (-9.0)) ** 2).mean()
                    l_cot_blurt = (F.relu(log_p_ct[:, blurt_digit_ids] - (-6.0)) ** 2).sum(dim=-1).mean()
                    l_cot = (
                        lambda_cot_entry * (2.4 * l_cot_eos + 1.4 * l_cot_ref + 0.008 * l_cot_blurt)
                        * (float(c1 - c0) / float(max(n_cot_seq, 1)))
                    )
                    l_cot.backward()
                    cot_accum += float(l_cot.item())
                del out_ct, pts_ct, log_p_ct, l_cot_eos, l_cot_ref, l_cot_blurt, l_cot
            last_cot = cot_accum

            torch.nn.utils.clip_grad_norm_(params_all, 1.0)
            opt.step()

            if (
                step + 1 >= min_steps
                and last_ce <= 0.024
                and last_cal <= 0.012
                and last_enc <= 0.015
                and last_nkl <= 0.060
            ):
                break

        mgr.merge_and_freeze_layer(l_idx)
        next_l = target_layers[idx_sched + 1] if idx_sched + 1 < len(target_layers) else n_layers
        if idx_sched + 1 < len(target_layers):
            with torch.no_grad():
                H_stu = _chunked_step_range(model, l_idx, next_l, H_stu, cm_stu, pe_stu, tp_stu, fwd_bs)
                H_neg = _chunked_step_range(model, l_idx, next_l, H_neg, cm_neg, pe_neg, tp_neg, fwd_bs)
                H_cal = _chunked_step_range(model, l_idx, next_l, H_cal, cm_cal, pe_cal, tp_cal, fwd_bs)
                H_enc = _chunked_step_range(model, l_idx, next_l, H_enc, cm_enc, pe_enc, tp_enc, fwd_bs)
                H_cot = _chunked_step_range(model, l_idx, next_l, H_cot, cm_cot, pe_cot, tp_cot, fwd_bs)
        if is_large_model:
            del skip_stu, skip_neg, skip_cal, skip_enc, skip_cot
            torch.cuda.empty_cache()

        layer_logs.append({
            "layer": l_idx,
            "steps": steps_used,
            "ce": round(last_ce, 4),
            "cal_kl": round(last_cal, 5),
            "enc_kl": round(last_enc, 5),
            "neg_kl": round(last_nkl, 5),
            "cot_pen": round(last_cot, 5),
        })

    return mgr, {"lr_b": round(lr_b, 6), "target_layers": target_layers, "layer_logs": layer_logs}


def run_experiment(
    scale: str,
    seeds: List[int],
    stage: int,
    device: str,
    eval_thinking_diag: bool = False,
) -> List[Dict[str, object]]:
    model, tokenizer, spec = load_model_by_scale(scale, device=device)
    cal_texts, kl_texts = load_smoltalk_for_scale(tokenizer, scale, n_cal=128, n_kl=64)
    results = []

    for seed in seeds:
        batch = generate_bio_batch(seed, tokenizer=tokenizer, target_doc_tokens=2048)
        base_ref = get_bio_base_reference(model, tokenizer, spec, batch, device)

        base_l, ora_l = _lookup_base_oracle_logic(scale, seed)
        if base_l is None or ora_l is None:
            with disable_adapters_ctx(model, None):
                base_eval = evaluate_bio_transfer(
                    model, tokenizer, spec, batch, kl_texts, device,
                    prefix_bio=False, compute_kl=False, eval_thinking_diag=False, eval_self_recalled=False,
                    base_ref_override=base_ref,
                )
                ora_eval = evaluate_bio_transfer(
                    model, tokenizer, spec, batch, kl_texts, device,
                    prefix_bio=True, compute_kl=False, eval_thinking_diag=False, eval_self_recalled=False,
                    base_ref_override=base_ref,
                )
                base_l = float(base_eval["logic_1turn_acc"])
                ora_l = float(ora_eval["logic_1turn_acc"])

        with NoBackpropGuard(model, regime="single_layer_isolated", static_vram_mb=spec.static_vram_mb) as guard:
            guard.assert_backbone_no_grad()
            if stage == 1:
                clauses = batch.stage1_clauses
            else:
                doc_enc = tokenizer(
                    batch.stage2_document_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                ).to(device)
                with torch.no_grad():
                    _ = get_text_model(model)(
                        input_ids=doc_enc.input_ids,
                        attention_mask=doc_enc.attention_mask,
                        use_cache=False,
                    )
                del doc_enc
                clauses = extract_bio_clauses_from_doc(batch.stage2_document_text)

            mgr, synth_diag = synthesize_unified_transfer_v15(
                model, tokenizer, clauses=clauses, cal_texts=cal_texts, device=device,
            )

        ev = evaluate_bio_transfer(
            model, tokenizer, spec, batch, kl_texts, device,
            lora_manager=mgr, compute_kl=True, eval_thinking_diag=eval_thinking_diag, eval_self_recalled=True,
            base_ref_override=base_ref,
        )
        gates = check_transfer_gates(ev, guard.metrics, scale, 16, base_l, ora_l)
        status = "pass" if gates["ALL_GATES_PASS"] else "partial"
        rec = append_transfer_record(
            15,
            "UnifiedTransferBlock_v15",
            scale,
            spec.model_id,
            ev,
            guard.metrics,
            status=status,
            seed=seed,
            stage=stage,
            rank=16,
            gates=gates,
            extra={"synth_diag": synth_diag},
        )
        print(
            f"[Iter15 {scale} s={seed} stg={stage}] direct={ev['direct_recall_acc']} para={ev['paraphrase_acc']} "
            f"logic={ev['logic_1turn_acc']} sr_logic={ev['logic_self_recalled_acc']} think={ev['logic_thinking_acc']} "
            f"real_anc={ev['real_anchors_acc']} bleed={ev['synth_bleed']} kl={ev['kl_div']} ret={ev['retention_acc']} "
            f"refusal={ev['direct_refusal_rate']} ovh={guard.metrics.activation_vram_overhead_mb:.1f}MB "
            f"time={guard.metrics.wall_time_sec:.2f}s ALL_PASS={gates['ALL_GATES_PASS']}",
            flush=True,
        )
        print("  gates:", gates, flush=True)
        (LAB_DIR / f"iter_15_details_{scale}_s{seed}_stg{stage}.json").write_text(
            json.dumps({"record": rec, "details": ev["details"]}, indent=2)
        )
        results.append(rec)
        mgr.uninstall()
        cleanup_cuda()

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=["small", "large", "both"], default="small")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1)
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.scale in ("small", "both"):
        run_experiment("small", args.seeds, args.stage, device, eval_thinking_diag=args.thinking)
        cleanup_cuda()
    if args.scale in ("large", "both"):
        run_experiment("large", args.seeds, args.stage, device, eval_thinking_diag=args.thinking)
        cleanup_cuda()


if __name__ == "__main__":
    main()
