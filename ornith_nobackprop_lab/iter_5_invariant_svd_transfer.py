"""Iteration 5: Width-Invariant Dual-Mode SVD-Null Block + Counterfactual Mid-Trace Protection.

Mechanism & Scaling Fixes over Iter 4 (`360M: bleed=0.55, logic=0.0`; `9B: kl=0.0705, ovh=2693MB, time=179s`):
  1. Width-Invariant Parameter-Group LR (`lr_A ~ 1/sqrt(d_in)`): Keeps relative angular rotation of the
     orthonormal SVD basis `lora_A` constant across `d=960, 2560, 4096, 12288`, stopping 9B null-space
     escape (`kl: 0.0705 -> << 0.05`) while converging smoothly with `lora_alpha=16.0` (`scaling=1.0`).
  2. Explicit Mid-Trace Counterfactual Negative Protection: Substitutes sibling names (`Dr. <Sib>`) AND
     non-test world figures into all declarative bio stems alongside `max_new_tokens=6` negative rollouts,
     making mid-sentence completion strictly subject-conditioned (`bleed <= 0.05`, `real_anc >= 0.95`).
  3. Zero `<eos>` Supervision (`add_eos=False`) + Anti-Echo + Mid-CoT `Step 1:` / `Fact:` Prefix Diversity:
     Eliminates `<eos>` short-circuiting on `360M` CoT while crushing `"Aether-7 is a fictional"` on `9B`.
  4. Late `[27, 28, 29, 30, 31]` Layers (at & after `L27 full_attention` on 9B) + Combined Cal/Enc Batch
     + CPU-Offloaded Inactive Banks on `9B` -> keeps `ovh <= 1,800 MB` (`<= 2,500`) and `time <= 105s` (`<= 120s`).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from bio_logic_benchmark import (
    SEEDS,
    extract_bio_clauses_from_doc,
    generate_bio_batch,
    parse_bio_clause,
)
from eval_harness import disable_adapters_ctx
from guards import NoBackpropGuard, cleanup_cuda
from iter_0_transfer_baselines import load_smoltalk_for_scale
from iter_1_transfer_cot_block import (
    COT_TRAIN_SYS,
    HELPER_SYS,
    _ENC_CAL_ANCHORS,
    _lookup_base_oracle_logic,
    _parse_decl_stems,
    build_dual_mode_bio_probes,
    collect_refusal_token_ids,
    tokenize_pos_probes,
    UnifiedTransferManager,
)
from iter_3_hybrid_svd_transfer import _BIO_EXTRA_ANCHORS, _init_svd_null_lora
from iter_4_whitened_transfer import (
    _build_diverse_cot_entry_prompts,
    _chunked_batched_generate_short,
    _chunked_step_range,
)
from probes import gather_positions, make_realistic_negative_entities
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


_ENC_PEOPLE: List[str] = [
    "Wolfgang Amadeus Mozart",
    "William Shakespeare",
    "Ludwig van Beethoven",
    "Yuri Gagarin",
    "Ferdinand Magellan",
    "Vincent van Gogh",
    "Jane Austen",
    "Amelia Earhart",
    "Pablo Picasso",
    "Marco Polo",
    "Julius Caesar",
    "Cleopatra",
]


def _build_invariant_pos_and_neg(
    clauses: List[str],
    tokenizer,
    n_rollout_neg: int = 28,
) -> Dict[str, object]:
    """Build zero-<eos> dual-mode pos probes + rollout & counterfactual mid-trace negative probes."""
    base_bundle = build_dual_mode_bio_probes(clauses, tokenizer, n_neg=n_rollout_neg)
    pos_items: List[Dict[str, object]] = list(base_bundle["pos_items"])
    parsed = [parse_bio_clause(cl) for cl in clauses]
    person = parsed[0]["person"]

    # Add extra atomic CoT markdown bullet diversity (`Step 1:`, `**Fact:**`) without `<eos>` or eval leakage
    for idx, it in enumerate(parsed):
        lhs, tgt, pers = it["lhs"], it["target"], it["person"]
        decls = _parse_decl_stems(pers, lhs, tgt)
        decl_prefix, decl_tgt = decls[min(1, len(decls) - 1)]
        chat_h = apply_chat_safe(
            tokenizer,
            [
                {"role": "system", "content": COT_TRAIN_SYS},
                {"role": "user", "content": f"Identify and verify {lhs.lower()} for {pers} step by step."},
            ],
            add_generation_prompt=True,
            enable_thinking=False,
        )
        pos_items.append({
            "mode": "cot_mid",
            "prompt": chat_h + f"**Step 1: Identify the facts.**\n**Fact:** {decl_prefix}",
            "target": decl_tgt,
            "space_prefix": " ",
            "trailing": ".",
            "add_eos": False,
            "w0": 1.85,
            "w_mid": 1.25,
            "w_tail": 0.35,
        })

    # Zero `<eos>` everywhere to completely prevent `360M` CoT short-circuiting; balance target frequencies
    tgt_counts = Counter(str(it["target"]) for it in pos_items)
    for it in pos_items:
        freq_scale = max(0.55, min(1.35, math.sqrt(4.0 / float(max(tgt_counts[str(it["target"])], 1)))))
        it["w0"] = float(it["w0"]) * freq_scale
        it["w_mid"] = float(it["w_mid"]) * freq_scale
        if it["mode"] == "qa":
            it["add_eos"] = False
            it["trailing"] = ""
            it["w_tail"] = 0.0

    # Counterfactual Mid-Trace Negatives (protect relation templates for siblings & real figures!)
    forbidden = {person.lower()}
    for item in parsed:
        for w in (item["person"] + " " + item["target"]).lower().replace("-", " ").split():
            if len(w) > 2:
                forbidden.add(w)
    sib_ents = [f"Dr. {s}" for s in make_realistic_negative_entities(12, forbidden)]
    cf_entities = sib_ents + _ENC_PEOPLE
    cf_mid_prompts: List[str] = []
    cf_templates = [
        lambda ent: f"{ent} was born in the town of",
        lambda ent: f"The birth year of {ent} is",
        lambda ent: f"For study, {ent} relocated to the city of",
        lambda ent: f"The academic mentor of {ent} was",
        lambda ent: f"The technical device invented by {ent} is",
        lambda ent: f"{ent} subsequently joined the institution",
        lambda ent: f"{ent}'s younger sister is named",
        lambda ent: f"{ent} suffers from a severe medical allergy to",
    ]
    for i in range(28):
        ent = cf_entities[i % len(cf_entities)]
        stem_cf = cf_templates[i % len(cf_templates)](ent)
        ch = apply_chat_safe(
            tokenizer,
            [{"role": "system", "content": COT_TRAIN_SYS}, {"role": "user", "content": f"Check biographical records for {ent} step by step."}],
            add_generation_prompt=True,
            enable_thinking=False,
        )
        hdr = "Step 1: Recall relevant biographical facts.\n- " if i % 2 == 0 else "**Fact:** "
        cf_mid_prompts.append(ch + hdr + stem_cf)

    return {
        "pos_items": pos_items,
        "neg_rollout_prompts": [str(x["prompt"]) for x in base_bundle["neg_items"]],
        "neg_cf_mid_prompts": cf_mid_prompts,
        "cot_entry_prompts": _build_diverse_cot_entry_prompts(clauses, tokenizer)[:16],
        "person": person,
    }


def synthesize_invariant_svd_block(
    model: nn.Module,
    tokenizer,
    clauses: List[str],
    cal_texts: List[str],
    device: str,
    target_layers: Optional[List[int]] = None,
    r: int = 16,
    lora_alpha: float = 16.0,
    alpha_null: float = 6.5,
    lr_b_base: float = 2.8e-3,
    n_null_rank: int = 96,
) -> Tuple[UnifiedTransferManager, Dict[str, object]]:
    """Run width-invariant dual-mode synthesis cleanly within NoBackpropGuard (<=2500MB on 9B)."""
    layers = get_layers(model)
    n_layers = len(layers)
    if target_layers is None:
        # L27..L31 (starts directly at L27 full_attention on Ornith-9B, dense late layers on 360M)
        target_layers = list(range(max(0, n_layers - 5), n_layers))

    mgr = UnifiedTransferManager(model, target_layers=target_layers, r=r, lora_alpha=lora_alpha)
    bundle = _build_invariant_pos_and_neg(clauses, tokenizer, n_rollout_neg=28)
    pos_items: List[Dict[str, object]] = bundle["pos_items"]  # type: ignore[assignment]
    neg_roll_strs: List[str] = bundle["neg_rollout_prompts"]  # type: ignore[assignment]
    neg_cf_strs: List[str] = bundle["neg_cf_mid_prompts"]  # type: ignore[assignment]
    cot_entry_prompts: List[str] = bundle["cot_entry_prompts"]  # type: ignore[assignment]

    enc_qs = (list(_ENC_CAL_ANCHORS) + list(_BIO_EXTRA_ANCHORS))[:28]
    enc_prompts = [
        apply_chat_safe(
            tokenizer,
            [{"role": "system", "content": SYS_QA}, {"role": "user", "content": q}],
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for q in enc_qs
    ]

    stu_ids, stu_mask, stu_pos, target_tok_ids, tok_weights, qa_bmask = tokenize_pos_probes(
        tokenizer, pos_items, device
    )

    trail_mask_list: List[bool] = []
    pos0_idx_list: List[int] = []
    pos0_row_mask: List[bool] = []
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
        for k in range(p_len):
            pos0_row_mask.append(k == 0)
            if k == 0:
                pos0_idx_list.append(int(target_tok_ids[tok_ptr].item()))
            trail_mask_list.append(bool(it["mode"] == "cot_mid" and it["trailing"] and k == p_len - 1))
        tok_ptr += p_len

    trail_bmask = torch.tensor(trail_mask_list, dtype=torch.bool, device=device)
    pos0_bmask = torch.tensor(pos0_row_mask, dtype=torch.bool, device=device)
    comp_bio_mask = torch.tensor(
        [[t_b != t0_id for t_b in all_bio_first] for t0_id in pos0_idx_list],
        dtype=torch.bool,
        device=device,
    )

    bleed_tok_ids = torch.tensor(sorted(injected_first | cot_target_first), dtype=torch.long, device=device)
    refusal_tok_ids = torch.tensor(collect_refusal_token_ids(tokenizer), dtype=torch.long, device=device)
    eos_list = get_eos_token_ids(tokenizer)
    eos_tok_ids = torch.tensor(eos_list, dtype=torch.long, device=device)

    lm_norm = get_lm_norm(model)
    lm_head = get_lm_head(model)
    prev_side = tokenizer.padding_side
    d_model = int(get_text_model(model).config.hidden_size)
    is_large_model = d_model >= 2048
    fwd_bs = 18 if is_large_model else 128

    with torch.no_grad():
        mgr.freeze_all()
        mgr.set_all_active(False)

        # Rollout negative siblings 6 tokens so mid-sentence drift (`"Dr. X was born in..."`) is protected
        clean_gens = _chunked_batched_generate_short(model, tokenizer, neg_roll_strs, device, batch_size=16, max_new_tokens=6)
        neg_seqs, neg_pos, neg_w = [], [], []
        for idx_n, clean_gen in enumerate(clean_gens):
            p_ids = tokenizer.encode(neg_roll_strs[idx_n], add_special_tokens=False)
            neg_seqs.append(p_ids + clean_gen)
            n_pred = max(1, len(clean_gen))
            neg_pos.append(list(range(len(p_ids) - 1, len(p_ids) - 1 + n_pred)))
            neg_w.extend([2.8] + [1.2] * (n_pred - 1))

        # Append Counterfactual Mid-Trace Negatives (exact mid-sentence bio templates with sibling/world names)
        for cf_p in neg_cf_strs:
            p_ids = tokenizer.encode(cf_p, add_special_tokens=False)
            neg_seqs.append(p_ids)
            neg_pos.append([len(p_ids) - 1])
            neg_w.append(2.8)

        neg_tok_weights = torch.tensor(neg_w, dtype=torch.float32, device=device)
        neg_tok_weights = neg_tok_weights / neg_tok_weights.mean().clamp(min=1e-6)

        max_nl = max(len(s) for s in neg_seqs)
        neg_ids = torch.full((len(neg_seqs), max_nl), tokenizer.pad_token_id, dtype=torch.long, device=device)
        neg_mask = torch.zeros((len(neg_seqs), max_nl), dtype=torch.long, device=device)
        for i, s in enumerate(neg_seqs):
            neg_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            neg_mask[i, : len(s)] = 1

        tokenizer.padding_side = "right"
        cal_smol_enc = tokenizer(cal_texts[:32], return_tensors="pt", padding=True, truncation=True, max_length=72).to(device)
        cal_smol_lens = cal_smol_enc.attention_mask.sum(dim=1).tolist()
        cal_smol_pos = [[max(0, min(L - 1, int(k * L / 4))) for k in range(1, 4)] for L in cal_smol_lens]

        enc_tok = tokenizer(enc_prompts, return_tensors="pt", padding=True, truncation=True, max_length=96).to(device)
        enc_lens = enc_tok.attention_mask.sum(dim=1).tolist()
        enc_pos = [[max(0, L - 1)] for L in enc_lens]

        cot_tok = tokenizer(cot_entry_prompts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
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

        lp_smol_cpu, H_smol, cm_smol, pe_smol, tp_smol = get_base_logits_and_h_start(
            cal_smol_enc.input_ids, cal_smol_enc.attention_mask, cal_smol_pos, need_logits=True
        )
        lp_enc_cpu, H_enc, cm_enc, pe_enc, tp_enc = get_base_logits_and_h_start(
            enc_tok.input_ids, enc_tok.attention_mask, enc_pos, need_logits=True
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

        # Keep fp16 reference distributions (on CPU when 9B to keep VRAM < 1,600 MB!)
        prob_dev = "cpu" if is_large_model else device
        p_smol_store = lp_smol_cpu.exp().half().to(prob_dev)
        p_enc_store = lp_enc_cpu.exp().half().to(prob_dev)
        p_neg_store = lp_neg_cpu.exp().half().to(prob_dev)
        logp_base_neg_bleed = lp_neg_cpu[:, bleed_tok_ids.cpu()].float().to(device)
        del lp_smol_cpu, lp_enc_cpu, lp_neg_cpu

        smol_mask_bool = cal_smol_enc.attention_mask.bool()
        enc_mask_bool = enc_tok.attention_mask.bool()
        neg_boundary_pos = [[p[0]] for p in neg_pos]
        if is_large_model:
            torch.cuda.empty_cache()

    # Width-invariant LR (`lr_B` gentle enough for 9B KL<=0.05; `lr_A ~ 1/sqrt(d_in)` stops basis drift)
    lr_b = float(lr_b_base) * (0.90 if not is_large_model else 0.85)  # ~2.52e-3 on 360M, ~2.38e-3 on 9B
    steps_sched = (8, 8, 9, 9, 10)
    if len(target_layers) != len(steps_sched):
        steps_sched = tuple([9] * len(target_layers))

    lambda_smol_kl = 2.5
    lambda_enc_kl = 7.5
    lambda_neg_kl = 7.5
    lambda_antibleed = 2.8
    lambda_cot_entry = 2.2
    lambda_refusal = 0.65
    lambda_echo = 1.2
    lambda_bio_margin = 0.30
    lambda_dw = 0.004

    layer_logs = []
    n_smol = H_smol.shape[0]
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
        n_steps = steps_sched[idx_sched]
        H_stu_in = H_stu.detach()
        H_neg_in = H_neg.detach()
        H_smol_in = H_smol.detach()
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
                            if x_g.shape[0] > 240:
                                stride = max(1, x_g.shape[0] // 240)
                                x_g = x_g[::stride][:240]
                            acts_by_sp[sp_n].append((x_g * weight_scale).cpu())
                        return hk
                    live = [rep_mod.register_forward_hook(_mk(sp)) for sp, rep_mod in hooks]
                    _ = step_layer_unified(
                        model, l_idx, h_b[s:e],
                        slice_cm_map(cm_b, s, e, n_b), slice_pe(pe_b, s, e, n_b), tp_b[s:e],
                    )
                    for h_rem in live:
                        h_rem.remove()

            _capture_batch(H_smol_in, cm_smol, pe_smol, tp_smol, smol_mask_bool, False, 1.0)
            _capture_batch(H_enc_in, cm_enc, pe_enc, tp_enc, enc_mask_bool, False, 1.5)
            _capture_batch(H_neg_in, cm_neg, pe_neg, tp_neg, neg_boundary_pos, True, 1.8)

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
                        mgr.wrapped[(li, m_name)], v_by_sp[sp], c_by_sp[sp], stu_acts[sp], seed_val=4000 * l_idx + idx_m
                    )
            del v_by_sp, c_by_sp, stu_acts

            mgr.set_all_active(True)
            mgr.freeze_all()
            skip_stu = chunked_compute_skip(H_stu_in, cm_stu, pe_stu, tp_stu, stu_pos)
            skip_neg = chunked_compute_skip(H_neg_in, cm_neg, pe_neg, tp_neg, neg_pos)
            skip_smol = chunked_compute_skip(H_smol_in, cm_smol, pe_smol, tp_smol, cal_smol_pos)
            skip_enc = chunked_compute_skip(H_enc_in, cm_enc, pe_enc, tp_enc, enc_pos)
            skip_cot = chunked_compute_skip(H_cot_in, cm_cot, pe_cot, tp_cot, cot_pos)
            if is_large_model:
                torch.cuda.empty_cache()

        mgr.activate_for_layer(l_idx)
        # Width-invariant parameter groups: `lr_A` scales as `1 / sqrt(d_in)`!
        opt_groups = []
        params_all = []
        for (li, _), w in mgr.wrapped.items():
            if li == l_idx:
                lr_a_w = lr_b * 0.25 * math.sqrt(960.0 / float(max(w.in_features, 1)))
                opt_groups.append({"params": [w.lora_A], "lr": lr_a_w, "weight_decay": 1e-3})
                opt_groups.append({"params": [w.lora_B], "lr": lr_b, "weight_decay": 3e-3})
                params_all.extend([w.lora_A, w.lora_B])
        opt = torch.optim.AdamW(opt_groups)

        m_stu = int(target_tok_ids.shape[0])
        stu_chunk = 34 if is_large_model else 256
        neg_chunk = 28 if is_large_model else 256
        enc_chunk = 28 if is_large_model else 256
        n_neg_seq = H_neg_in.shape[0]
        n_enc_seq = H_enc_in.shape[0]
        m_neg_tot = max(int(neg_tok_weights.numel()), 1)

        last_ce = last_smol = last_enc = last_nkl = last_cot = 0.0
        for step in range(n_steps):
            opt.zero_grad(set_to_none=True)

            # 1. Student Pos (QA + Mid-CoT + Intra-Bio Margin + Anti-Echo + Anti-Refusal)
            n_stu_seq = H_stu_in.shape[0]
            ce_accum = 0.0
            tok_off = 0
            pos0_off = 0
            n_pos0_tot = max(int(pos0_bmask.sum().item()), 1)
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
                        margin_err = F.relu(l_comp - l_true + 2.5) * cmask.float()
                        extra_loss = extra_loss + lambda_bio_margin * (margin_err.pow(2).sum() / float(max(n_pos0_tot * bio_first_toks.numel(), 1)))
                        # Anti-refusal across BOTH QA and Mid-CoT boundaries (`No`, `Sorry`, `I'm`, `Unknown`...)
                        lp_p0 = F.log_softmax(logits_p0, dim=-1)
                        extra_loss = extra_loss + (
                            lambda_refusal
                            * (F.relu(lp_p0[:, refusal_tok_ids] - (-8.0)) ** 2).mean()
                            * (float(p0m.sum().item()) / float(max(n_pos0_tot, 1)))
                        )

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

            # 2. SmolTalk KL + Frobenius Reg
            with torch.enable_grad():
                cs = (step % 2) * (n_smol // 2)
                ce_c = min(n_smol, cs + (n_smol // 2))
                c_start = sum(len(p) for p in cal_smol_pos[:cs])
                c_cnt = sum(len(p) for p in cal_smol_pos[cs:ce_c])
                out_c = step_layer_unified(
                    model, l_idx, H_smol_in[cs:ce_c],
                    slice_cm_map(cm_smol, cs, ce_c, n_smol),
                    slice_pe(pe_smol, cs, ce_c, n_smol),
                    tp_smol[cs:ce_c],
                )
                pts_c = gather_positions(out_c, cal_smol_pos[cs:ce_c]) + skip_smol[c_start : c_start + c_cnt]
                l_smol = F.kl_div(
                    F.log_softmax(lm_head(lm_norm(pts_c)).float(), dim=-1),
                    p_smol_store[c_start : c_start + c_cnt].to(device=device, dtype=torch.float32),
                    reduction="batchmean",
                )
                l_reg = lambda_dw * mgr.get_layer_frob_reg(l_idx)
                (lambda_smol_kl * l_smol + l_reg).backward()
                last_smol = float(l_smol.item())
            del out_c, pts_c, l_smol, l_reg

            # 3. Encyclopedia Protection
            enc_accum = 0.0
            for e0 in range(0, n_enc_seq, enc_chunk):
                e1 = min(n_enc_seq, e0 + enc_chunk)
                with torch.enable_grad():
                    out_e = step_layer_unified(
                        model, l_idx, H_enc_in[e0:e1],
                        slice_cm_map(cm_enc, e0, e1, n_enc_seq),
                        slice_pe(pe_enc, e0, e1, n_enc_seq),
                        tp_enc[e0:e1],
                    )
                    pts_e = gather_positions(out_e, enc_pos[e0:e1]) + skip_enc[e0:e1]
                    l_enc = (
                        F.kl_div(
                            F.log_softmax(lm_head(lm_norm(pts_e)).float(), dim=-1),
                            p_enc_store[e0:e1].to(device=device, dtype=torch.float32),
                            reduction="batchmean",
                        )
                        * (float(e1 - e0) / float(max(n_enc_seq, 1)))
                    )
                    (lambda_enc_kl * l_enc).backward()
                    enc_accum += float(l_enc.item())
                del out_e, pts_e, l_enc
            last_enc = enc_accum

            # 4. Rollout + Counterfactual Mid-Trace Negatives
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
                    l_bleed = (
                        (F.relu(log_p_n[:, bleed_tok_ids] - logp_base_neg_bleed[nt0:nt1] - 0.03) ** 2).mean()
                        * (float(ncnt) / float(m_neg_tot))
                    )
                    (lambda_neg_kl * l_nkl + lambda_antibleed * l_bleed).backward()
                    nkl_accum += float(l_nkl.item())
                del out_n, pts_n, log_p_n, pn_sl, tok_nkl, l_nkl, l_bleed
            last_nkl = nkl_accum

            # 5. CoT Entry Guard
            with torch.enable_grad():
                out_ct = step_layer_unified(model, l_idx, H_cot_in, cm_cot, pe_cot, tp_cot)
                pts_ct = gather_positions(out_ct, cot_pos) + skip_cot
                log_p_ct = F.log_softmax(lm_head(lm_norm(pts_ct)).float(), dim=-1)
                l_cot_eos = (F.relu(log_p_ct[:, eos_tok_ids] - (-12.0)) ** 2).mean()
                l_cot_ref = (F.relu(log_p_ct[:, refusal_tok_ids] - (-8.5)) ** 2).mean()
                l_cot = lambda_cot_entry * (2.5 * l_cot_eos + 1.0 * l_cot_ref)
                l_cot.backward()
                last_cot = float(l_cot.item())
            del out_ct, pts_ct, log_p_ct, l_cot_eos, l_cot_ref, l_cot

            torch.nn.utils.clip_grad_norm_(params_all, 1.0)
            opt.step()

        mgr.merge_and_freeze_layer(l_idx)
        next_l = target_layers[idx_sched + 1] if idx_sched + 1 < len(target_layers) else n_layers
        if idx_sched + 1 < len(target_layers):
            with torch.no_grad():
                H_stu = _chunked_step_range(model, l_idx, next_l, H_stu, cm_stu, pe_stu, tp_stu, fwd_bs)
                H_neg = _chunked_step_range(model, l_idx, next_l, H_neg, cm_neg, pe_neg, tp_neg, fwd_bs)
                H_smol = _chunked_step_range(model, l_idx, next_l, H_smol, cm_smol, pe_smol, tp_smol, fwd_bs)
                H_enc = _chunked_step_range(model, l_idx, next_l, H_enc, cm_enc, pe_enc, tp_enc, fwd_bs)
                H_cot = _chunked_step_range(model, l_idx, next_l, H_cot, cm_cot, pe_cot, tp_cot, fwd_bs)
        if is_large_model:
            del skip_stu, skip_neg, skip_smol, skip_enc, skip_cot
            torch.cuda.empty_cache()

        layer_logs.append({
            "layer": l_idx,
            "ce": round(last_ce, 4),
            "smol_kl": round(last_smol, 5),
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

            mgr, synth_diag = synthesize_invariant_svd_block(
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
            5,
            "WidthInvariant_SVD_NullBlock",
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
            f"[Iter5 {scale} s={seed} stg={stage}] direct={ev['direct_recall_acc']} para={ev['paraphrase_acc']} "
            f"logic={ev['logic_1turn_acc']} sr_logic={ev['logic_self_recalled_acc']} think={ev['logic_thinking_acc']} "
            f"real_anc={ev['real_anchors_acc']} bleed={ev['synth_bleed']} kl={ev['kl_div']} ret={ev['retention_acc']} "
            f"refusal={ev['direct_refusal_rate']} ovh={guard.metrics.activation_vram_overhead_mb:.1f}MB "
            f"time={guard.metrics.wall_time_sec:.2f}s ALL_PASS={gates['ALL_GATES_PASS']}",
            flush=True,
        )
        print("  gates:", gates, flush=True)
        (LAB_DIR / f"iter_5_details_{scale}_s{seed}_stg{stage}.json").write_text(
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
