"""Iteration 3: Contrastive-SVD Null-Space Basis + Anti-Echo Dual-Mode Transfer Block (360M & 9B).

Mechanism Pivot from Iter 1/2 (`Random Null LoRA`):
  1. Contrastive-SVD Basis Initialization: Instead of random Gaussian `lora_A`, captures layerwise student
     readout activations `X_stu`, projects out `(V_cal, c_damp)` (SmolTalk + Biographical Encyclopedia), and
     initializes `lora_A` to the top-r=16 QR-orthonormal right singular vectors `V_stu` of `X_stu_null`.
  2. Balanced Target-Frequency Weighting: Inverse-sqrt frequency weighting across distinct targets prevents
     reverse-lookup entities (`Dr. Seren Valenwood`, `2x` probes) from dominating rare relations (`mentor`, `allergy`).
  3. Biographical World-Knowledge Protection: 32 encyclopedia anchors (incl. 16 non-test historical biographies)
     seal the scientist/person QA subspace without leaking any `_REAL_ANCHORS` test item.
  4. Anti-Echo Next-Token Suppression on `pos_cot_mid` trailing positions kills entity repetition loops (`Nordhavn Nordhavn...`).
  5. Cross-Scale `[23, 26, 27, 28, 29, 30, 31]` routing covers all 3 late `full_attention` layers (`23, 27, 31`) on 9B.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
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
    COT_TRAIN_SYS,
    HELPER_SYS,
    _ENC_CAL_ANCHORS,
    _lookup_base_oracle_logic,
    build_dual_mode_bio_probes,
    collect_refusal_token_ids,
    tokenize_pos_probes,
    UnifiedTransferManager,
)
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


# 16 biographical encyclopedia questions strictly disjoint from the 10 `_REAL_ANCHORS`
_BIO_EXTRA_ANCHORS: List[str] = [
    "In which country was composer Wolfgang Amadeus Mozart born?",
    "Which famous English playwright was born in Stratford-upon-Avon?",
    "In which German city was composer Ludwig van Beethoven born?",
    "Which Soviet cosmonaut became the first human to journey into outer space?",
    "Which Portuguese explorer led the expedition that completed the first circumnavigation of the Earth?",
    "Which ancient Greek philosopher served as the tutor to Alexander the Great?",
    "Which Dutch Post-Impressionist artist painted The Starry Night?",
    "Which English author wrote the novel Pride and Prejudice?",
    "Which American aviation pioneer disappeared over the Pacific Ocean in 1937?",
    "Which Spanish artist co-founded the Cubist movement alongside Georges Braque?",
    "Which Italian merchant and traveler wrote accounts of his journeys along the Silk Road to China?",
    "Which ancient Roman general crossed the Rubicon river in 49 BC?",
    "Which female Pharaoh ruled Egypt as the last active ruler of the Ptolemaic Kingdom?",
    "Which ancient Greek mathematician exclaimed Eureka in Syracuse?",
    "Which Norwegian polar explorer led the first expedition to reach the South Pole in 1911?",
    "Which American civil rights leader delivered the I Have a Dream speech in 1963?",
]


def _init_svd_null_lora(
    lora_mod,
    v_cal: torch.Tensor,
    c_damp: torch.Tensor,
    x_stu_sp: torch.Tensor,
    seed_val: int,
) -> None:
    """Initialize `lora_A` using top-r contrastive null-projected student SVD directions."""
    dev = lora_mod.lora_A.device
    v_f = v_cal.to(device=dev, dtype=torch.float32)
    c_f = c_damp.to(device=dev, dtype=torch.float32)
    lora_mod.precond_V = v_cal.to(device=dev, dtype=torch.float16)
    lora_mod.precond_c = c_f

    xs = x_stu_sp.to(device=dev, dtype=torch.float32)
    xs_null = xs - ((xs @ v_f) * c_f.unsqueeze(0)) @ v_f.T

    q_r = lora_mod.r
    _, _, v_stu = torch.svd_lowrank(xs_null, q=min(q_r, xs_null.shape[0] - 1, xs_null.shape[1] - 1), niter=2)
    a_cols = v_stu  # (d_in, q_cur)
    if a_cols.shape[1] < q_r:
        gen = torch.Generator(device=dev).manual_seed(seed_val)
        pad = torch.randn((lora_mod.in_features, q_r - a_cols.shape[1]), generator=gen, device=dev, dtype=torch.float32)
        a_cols = torch.cat([a_cols, pad], dim=1)
    # Remove any top calibration component before QR orthonormalization
    a_cols = a_cols - v_f @ (v_f.T @ a_cols)
    q_mat, _ = torch.linalg.qr(a_cols, mode="reduced")
    lora_mod.lora_A.data.copy_(q_mat.T.contiguous())
    lora_mod.lora_B.data.zero_()


def synthesize_hybrid_svd_block(
    model: nn.Module,
    tokenizer,
    clauses: List[str],
    cal_texts: List[str],
    device: str,
    target_layers: Optional[List[int]] = None,
    r: int = 16,
    lora_alpha: float = 24.0,
    alpha_null: float = 6.0,
    base_lr: float = 4.2e-3,
    n_null_rank: int = 96,
) -> Tuple[UnifiedTransferManager, Dict[str, object]]:
    """Run SVD-initialized, balanced dual-mode synthesis under strict NoBackpropGuard."""
    layers = get_layers(model)
    n_layers = len(layers)
    if target_layers is None:
        # Covers L23 (full_attn on 9B) + L26..L31 (all 3 late full_attn layers [23,27,31] + dense MLPs)
        target_layers = [23, 26, 27, 28, 29, 30, 31] if n_layers >= 32 else list(range(max(0, n_layers - 7), n_layers))

    mgr = UnifiedTransferManager(model, target_layers=target_layers, r=r, lora_alpha=lora_alpha)
    bundle = build_dual_mode_bio_probes(clauses, tokenizer, n_neg=52)
    pos_items = bundle["pos_items"]
    cot_entry_prompts = bundle["cot_entry_prompts"]
    neg_items = bundle["neg_items"]

    # Balance target frequencies so repeated entity targets don't drown rare entities (`mentor`, `allergy`, `join_year`)
    tgt_counts = Counter(str(it["target"]) for it in pos_items)
    for it in pos_items:
        freq_scale = math.sqrt(4.0 / float(max(tgt_counts[str(it["target"])], 1)))
        freq_scale = max(0.55, min(1.35, freq_scale))
        it["w0"] = float(it["w0"]) * freq_scale
        it["w_mid"] = float(it["w_mid"]) * freq_scale

    # Clean Clean-Text Biographical + General Encyclopedia Protection
    enc_qs = list(_ENC_CAL_ANCHORS) + list(_BIO_EXTRA_ANCHORS)
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

    # Identify trailing `.` positions inside `cot_mid` and injected first tokens for anti-echo + anti-bleed
    trail_mask_list: List[bool] = []
    injected_first = set()
    cot_target_first = set()
    for it in pos_items:
        tid = tokenizer.encode(str(it["target"]), add_special_tokens=False)
        sp_tid = tokenizer.encode(str(it["space_prefix"]) + str(it["target"]), add_special_tokens=False)
        if tid:
            injected_first.add(int(tid[0]))
        if sp_tid:
            cot_target_first.add(int(sp_tid[0]))
    for idx_i, it in enumerate(pos_items):
        p_len = len(stu_pos[idx_i])
        if it["mode"] == "cot_mid" and it["trailing"] and p_len >= 1:
            trail_mask_list.extend([False] * (p_len - 1) + [True])
        else:
            trail_mask_list.extend([False] * p_len)
    trail_bmask = torch.tensor(trail_mask_list, dtype=torch.bool, device=device)

    bleed_tok_ids = torch.tensor(sorted(injected_first), dtype=torch.long, device=device)
    echo_tok_ids = torch.tensor(sorted(injected_first | cot_target_first), dtype=torch.long, device=device)
    refusal_tok_ids = torch.tensor(collect_refusal_token_ids(tokenizer), dtype=torch.long, device=device)
    eos_list = get_eos_token_ids(tokenizer)
    eos_tok_ids = torch.tensor(eos_list, dtype=torch.long, device=device)

    lm_norm = get_lm_norm(model)
    lm_head = get_lm_head(model)
    prev_side = tokenizer.padding_side

    with torch.no_grad():
        mgr.freeze_all()
        mgr.set_all_active(False)

        # Batched sibling negative generation (2 tokens)
        tokenizer.padding_side = "left"
        neg_p_strs = [str(x["prompt"]) for x in neg_items]
        neg_gen_enc = tokenizer(neg_p_strs, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        in_len_neg = neg_gen_enc.input_ids.shape[1]
        neg_gen = model.generate(
            input_ids=neg_gen_enc.input_ids,
            attention_mask=neg_gen_enc.attention_mask,
            max_new_tokens=2,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_list,
            use_cache=True,
        )
        del neg_gen_enc
        neg_seqs, neg_pos, neg_w = [], [], []
        for idx_n in range(len(neg_items)):
            gen_part = neg_gen[idx_n, in_len_neg:].tolist()
            clean_gen = []
            for tk in gen_part:
                clean_gen.append(int(tk))
                if tk in eos_list:
                    break
            p_ids = tokenizer.encode(neg_p_strs[idx_n], add_special_tokens=False)
            neg_seqs.append(p_ids + clean_gen)
            n_pred = max(1, len(clean_gen))
            neg_pos.append(list(range(len(p_ids) - 1, len(p_ids) - 1 + n_pred)))
            neg_w.extend([3.0] + [1.2] * (n_pred - 1))
        del neg_gen
        neg_tok_weights = torch.tensor(neg_w, dtype=torch.float32, device=device)
        neg_tok_weights = neg_tok_weights / neg_tok_weights.mean().clamp(min=1e-6)

        max_nl = max(len(s) for s in neg_seqs)
        neg_ids = torch.full((len(neg_seqs), max_nl), tokenizer.pad_token_id, dtype=torch.long, device=device)
        neg_mask = torch.zeros((len(neg_seqs), max_nl), dtype=torch.long, device=device)
        for i, s in enumerate(neg_seqs):
            neg_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            neg_mask[i, : len(s)] = 1

        tokenizer.padding_side = "right"
        cal_smol_enc = tokenizer(cal_texts[:40], return_tensors="pt", padding=True, truncation=True, max_length=72).to(device)
        cal_smol_lens = cal_smol_enc.attention_mask.sum(dim=1).tolist()
        cal_smol_pos = [[max(0, min(L - 1, int(k * L / 4))) for k in range(1, 4)] for L in cal_smol_lens]

        enc_tok = tokenizer(enc_prompts, return_tensors="pt", padding=True, truncation=True, max_length=96).to(device)
        enc_lens = enc_tok.attention_mask.sum(dim=1).tolist()
        enc_pos = [[max(0, L - 1)] for L in enc_lens]

        cot_tok = tokenizer(cot_entry_prompts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        cot_lens = cot_tok.attention_mask.sum(dim=1).tolist()
        cot_pos = [[max(0, L - 1)] for L in cot_lens]
        tokenizer.padding_side = prev_side

        def get_base_logits_and_h_start(ids: torch.Tensor, mask: torch.Tensor, pos_lists: List[List[int]], chunk: int = 32):
            lp_chunks = []
            h0, cm0, pe0, tp0 = prepare_layer_inputs_unified(model, ids, mask)
            h_cur = h0
            h_entry = None
            for li in range(n_layers):
                if li == target_layers[0]:
                    h_entry = h_cur.detach()
                h_cur = step_layer_unified(model, li, h_cur, cm0, pe0, tp0)
            pts = gather_positions(h_cur, pos_lists)
            for s in range(0, pts.shape[0], chunk):
                e = min(pts.shape[0], s + chunk)
                lp_chunks.append(F.log_softmax(lm_head(lm_norm(pts[s:e])).float(), dim=-1).cpu())
            return torch.cat(lp_chunks, dim=0), h_entry, cm0, pe0, tp0

        lp_smol_cpu, H_smol, cm_smol, pe_smol, tp_smol = get_base_logits_and_h_start(
            cal_smol_enc.input_ids, cal_smol_enc.attention_mask, cal_smol_pos
        )
        lp_enc_cpu, H_enc, cm_enc, pe_enc, tp_enc = get_base_logits_and_h_start(
            enc_tok.input_ids, enc_tok.attention_mask, enc_pos
        )
        lp_neg_cpu, H_neg, cm_neg, pe_neg, tp_neg = get_base_logits_and_h_start(
            neg_ids, neg_mask, neg_pos
        )
        _, H_cot, cm_cot, pe_cot, tp_cot = get_base_logits_and_h_start(
            cot_tok.input_ids, cot_tok.attention_mask, cot_pos
        )

        p_smol_dev = lp_smol_cpu.exp().half().to(device)
        p_enc_dev = lp_enc_cpu.exp().half().to(device)
        p_neg_dev = lp_neg_cpu.exp().half().to(device)
        logp_base_neg_bleed = lp_neg_cpu[:, bleed_tok_ids.cpu()].float().to(device)
        del lp_smol_cpu, lp_enc_cpu, lp_neg_cpu

        h_s, cm_stu, pe_stu, tp_stu = prepare_layer_inputs_unified(model, stu_ids, stu_mask)
        for li in range(target_layers[0]):
            h_s = step_layer_unified(model, li, h_s, cm_stu, pe_stu, tp_stu)
        H_stu = h_s.detach()

        smol_mask_bool = cal_smol_enc.attention_mask.bool()
        enc_mask_bool = enc_tok.attention_mask.bool()

    d_model = int(H_smol.shape[-1])
    is_large_model = d_model >= 2048
    lr_eff = float(base_lr) * ((d_model / 4096.0) ** 0.25)
    lr_eff = max(2.4e-3, min(4.5e-3, lr_eff))

    steps_sched = (7, 8, 8, 9, 9, 10, 10)
    if len(target_layers) != len(steps_sched):
        steps_sched = tuple([8] * len(target_layers))

    lambda_smol_kl = 2.5
    lambda_enc_kl = 6.0
    lambda_neg_kl = 5.5
    lambda_antibleed = 2.4
    lambda_cot_entry = 1.8
    lambda_refusal = 0.50
    lambda_echo = 0.80
    lambda_dw = 0.002

    layer_logs = []
    n_smol = H_smol.shape[0]
    stu_boundary_pos = [[p[0]] for p in stu_pos]

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

            # 1. Calibration activations (SmolTalk + 36 World/Bio Encyclopedia Anchors)
            acts_by_sp: Dict[str, List[torch.Tensor]] = {sp: [] for sp in spaces}
            for batch_h, batch_cm, batch_pe, batch_tp, batch_mb in [
                (H_smol_in, cm_smol, pe_smol, tp_smol, smol_mask_bool),
                (H_enc_in, cm_enc, pe_enc, tp_enc, enc_mask_bool),
            ]:
                def _hk_cal(sp_n: str, mb: torch.Tensor):
                    mb_flat = mb.reshape(-1)
                    def hk(m, inp, out):
                        x_in = inp[0].detach().reshape(-1, inp[0].shape[-1])
                        acts_by_sp[sp_n].append(x_in[mb_flat].float().cpu())
                    return hk
                live = [rep_mod.register_forward_hook(_hk_cal(sp, batch_mb)) for sp, rep_mod in hooks]
                _ = step_layer_unified(model, l_idx, batch_h, batch_cm, batch_pe, batch_tp)
                for h_rem in live:
                    h_rem.remove()

            # 2. Student contrastive readout activations at target start positions
            stu_acts: Dict[str, torch.Tensor] = {}
            def _hk_stu(sp_n: str, b_sz: int, sq_len: int):
                def hk(m, inp, out):
                    x_3d = inp[0].detach().reshape(b_sz, sq_len, -1)
                    stu_acts[sp_n] = gather_positions(x_3d, stu_boundary_pos).float().detach()
                return hk
            live_stu = [
                rep_mod.register_forward_hook(_hk_stu(sp, H_stu_in.shape[0], H_stu_in.shape[1]))
                for sp, rep_mod in hooks
            ]
            _ = step_layer_unified(model, l_idx, H_stu_in, cm_stu, pe_stu, tp_stu)
            for h_rem in live_stu:
                h_rem.remove()

            # 3. Compute normalized low-rank calibration null preconditioners + Contrastive-SVD LoRA-A Init
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
                        mgr.wrapped[(li, m_name)], v_by_sp[sp], c_by_sp[sp], stu_acts[sp], seed_val=2000 * l_idx + idx_m
                    )
            del v_by_sp, c_by_sp, stu_acts

            mgr.set_all_active(True)
            mgr.freeze_all()

            def compute_skip(h_in, cm_m, pe_t, tp_t, pos_lists):
                hl = step_layer_unified(model, l_idx, h_in, cm_m, pe_t, tp_t)
                hc = hl
                for j in range(l_idx + 1, n_layers):
                    hc = step_layer_unified(model, j, hc, cm_m, pe_t, tp_t)
                return gather_positions(hc - hl, pos_lists).detach()

            skip_stu = compute_skip(H_stu_in, cm_stu, pe_stu, tp_stu, stu_pos)
            skip_neg = compute_skip(H_neg_in, cm_neg, pe_neg, tp_neg, neg_pos)
            skip_smol = compute_skip(H_smol_in, cm_smol, pe_smol, tp_smol, cal_smol_pos)
            skip_enc = compute_skip(H_enc_in, cm_enc, pe_enc, tp_enc, enc_pos)
            skip_cot = compute_skip(H_cot_in, cm_cot, pe_cot, tp_cot, cot_pos)

        mgr.activate_for_layer(l_idx)
        params = mgr.get_layer_params(l_idx)
        opt = torch.optim.AdamW(params, lr=lr_eff, weight_decay=2e-3)

        m_stu = int(target_tok_ids.shape[0])
        stu_chunk = 50 if is_large_model else 256
        neg_chunk = 42 if is_large_model else 256
        n_neg_seq = H_neg_in.shape[0]
        m_neg_tot = max(int(neg_tok_weights.numel()), 1)

        last_ce = last_smol = last_enc = last_nkl = last_cot = 0.0
        for step in range(n_steps):
            opt.zero_grad(set_to_none=True)

            # 1. Student Pos (QA + Mid-CoT + Anti-Echo + Anti-Refusal)
            n_stu_seq = H_stu_in.shape[0]
            ce_accum = 0.0
            tok_off = 0
            for s0 in range(0, n_stu_seq, stu_chunk):
                s1 = min(n_stu_seq, s0 + stu_chunk)
                cnt = sum(len(p) for p in stu_pos[s0:s1])
                t0, t1 = tok_off, tok_off + cnt
                tok_off = t1
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
                    bm = qa_bmask[t0:t1]
                    if bm.any():
                        lp_qa = F.log_softmax(logits_s[bm], dim=-1)
                        extra_loss = extra_loss + (
                            lambda_refusal
                            * (F.relu(lp_qa[:, refusal_tok_ids] - (-8.0)) ** 2).mean()
                            * (float(bm.sum().item()) / float(max(int(qa_bmask.sum().item()), 1)))
                        )
                    tm = trail_bmask[t0:t1]
                    if tm.any():
                        lp_tr = F.log_softmax(logits_s[tm], dim=-1)
                        extra_loss = extra_loss + (
                            lambda_echo
                            * (F.relu(lp_tr[:, echo_tok_ids] - (-8.0)) ** 2).mean()
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
                    p_smol_dev[c_start : c_start + c_cnt].float(),
                    reduction="batchmean",
                )
                l_reg = lambda_dw * mgr.get_layer_frob_reg(l_idx)
                (lambda_smol_kl * l_smol + l_reg).backward()
                last_smol = float(l_smol.item())
            del out_c, pts_c, l_smol, l_reg

            # 3. General & Biographical Encyclopedia Protection
            with torch.enable_grad():
                out_e = step_layer_unified(model, l_idx, H_enc_in, cm_enc, pe_enc, tp_enc)
                pts_e = gather_positions(out_e, enc_pos) + skip_enc
                l_enc = F.kl_div(
                    F.log_softmax(lm_head(lm_norm(pts_e)).float(), dim=-1),
                    p_enc_dev.float(),
                    reduction="batchmean",
                )
                (lambda_enc_kl * l_enc).backward()
                last_enc = float(l_enc.item())
            del out_e, pts_e, l_enc

            # 4. Sibling Negatives (QA + CoT)
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
                    pn_sl = p_neg_dev[nt0:nt1].float()
                    tok_nkl = (pn_sl * (pn_sl.clamp(min=1e-12).log() - log_p_n)).sum(dim=-1)
                    l_nkl = (tok_nkl * neg_tok_weights[nt0:nt1]).sum() / float(m_neg_tot)
                    l_bleed = (
                        (F.relu(log_p_n[:, bleed_tok_ids] - logp_base_neg_bleed[nt0:nt1] - 0.04) ** 2).mean()
                        * (float(ncnt) / float(m_neg_tot))
                    )
                    (lambda_neg_kl * l_nkl + lambda_antibleed * l_bleed).backward()
                    nkl_accum += float(l_nkl.item())
                del out_n, pts_n, log_p_n, pn_sl, tok_nkl, l_nkl, l_bleed
            last_nkl = nkl_accum

            # 5. CoT Entry Guard (<eos> & refusal suppression)
            with torch.enable_grad():
                out_ct = step_layer_unified(model, l_idx, H_cot_in, cm_cot, pe_cot, tp_cot)
                pts_ct = gather_positions(out_ct, cot_pos) + skip_cot
                log_p_ct = F.log_softmax(lm_head(lm_norm(pts_ct)).float(), dim=-1)
                l_cot_eos = (F.relu(log_p_ct[:, eos_tok_ids] - (-11.0)) ** 2).mean()
                l_cot_ref = (F.relu(log_p_ct[:, refusal_tok_ids] - (-8.5)) ** 2).mean()
                l_cot = lambda_cot_entry * (1.8 * l_cot_eos + 0.8 * l_cot_ref)
                l_cot.backward()
                last_cot = float(l_cot.item())
            del out_ct, pts_ct, log_p_ct, l_cot_eos, l_cot_ref, l_cot

            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

        mgr.merge_and_freeze_layer(l_idx)
        # Advance cached hidden states cleanly across any gap (`next_l - l_idx`, e.g. 23 -> 24 -> 25 -> 26)
        next_l = target_layers[idx_sched + 1] if idx_sched + 1 < len(target_layers) else n_layers
        if idx_sched + 1 < len(target_layers):
            with torch.no_grad():
                for step_li in range(l_idx, next_l):
                    H_stu = step_layer_unified(model, step_li, H_stu, cm_stu, pe_stu, tp_stu).detach()
                    H_neg = step_layer_unified(model, step_li, H_neg, cm_neg, pe_neg, tp_neg).detach()
                    H_smol = step_layer_unified(model, step_li, H_smol, cm_smol, pe_smol, tp_smol).detach()
                    H_enc = step_layer_unified(model, step_li, H_enc, cm_enc, pe_enc, tp_enc).detach()
                    H_cot = step_layer_unified(model, step_li, H_cot, cm_cot, pe_cot, tp_cot).detach()

        layer_logs.append({
            "layer": l_idx,
            "ce": round(last_ce, 4),
            "smol_kl": round(last_smol, 5),
            "enc_kl": round(last_enc, 5),
            "neg_kl": round(last_nkl, 5),
            "cot_pen": round(last_cot, 5),
        })

    return mgr, {"lr_eff": round(lr_eff, 6), "target_layers": target_layers, "layer_logs": layer_logs}


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

            mgr, synth_diag = synthesize_hybrid_svd_block(
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
            3,
            "ContrastiveSVD_NullBlock",
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
            f"[Iter3 {scale} s={seed} stg={stage}] direct={ev['direct_recall_acc']} para={ev['paraphrase_acc']} "
            f"logic={ev['logic_1turn_acc']} sr_logic={ev['logic_self_recalled_acc']} think={ev['logic_thinking_acc']} "
            f"real_anc={ev['real_anchors_acc']} bleed={ev['synth_bleed']} kl={ev['kl_div']} ret={ev['retention_acc']} "
            f"refusal={ev['direct_refusal_rate']} ovh={guard.metrics.activation_vram_overhead_mb:.1f}MB "
            f"time={guard.metrics.wall_time_sec:.2f}s ALL_PASS={gates['ALL_GATES_PASS']}",
            flush=True,
        )
        print("  gates:", gates, flush=True)
        (LAB_DIR / f"iter_3_details_{scale}_s{seed}_stg{stage}.json").write_text(
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
