"""Mechanism A12 (Iter 28): Pure 0-Backprop Voronoi-Guarded Contrastive SwiGLU LoRA on 8-bit Ornith-1.0-9B.

Key architectural adaptations for 8-bit Qwen3.5 (Ornith-1.0-9B):
1. Uses Qwen3_5RMSNorm effective scale gamma = 1.0 + norm.weight (zero-centered RMSNorm!).
2. Avoids 4.07 GB float32 lm_head copy by computing chunked logits directly against native fp16 lm_head.weight.
3. Selects dormant SwiGLU neurons J_l (out of 12,288) via chunked 8-bit forward passes (no full matrix dequantization),
   then dequantizes ONLY the r <= 16 rows J_l via SCB row scales for analytical contrastive Ridge.
4. Steps cleanly across hybrid linear_attention (GatedDeltaNet) and full_attention layers via step_token_mixer.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from benchmark import FactBatch, generate_fact_batch, load_smoltalk_splits
from eval_harness import append_experiment_record, check_gates, evaluate_model, get_base_reference
from guards import LoRAManager, NoBackpropGuard, cleanup_cuda, measure_static_vram_mb
from model_loader import (
    dequantize_rows,
    get_layers,
    get_lm_head,
    get_lm_norm,
    get_text_model,
    load_ornith_8bit,
    prepare_layer_inputs,
    slice_cm_map,
    slice_pe,
    step_layer,
    step_token_mixer,
)
from probes import (
    build_probes_mechA10,
    gather_positions,
    print_locality_breakdown,
    tokenize_with_target_positions,
)

LAB_DIR = Path("/content/ornith_nobackprop_lab")
TARGET_MODULES = ("gate_proj", "up_proj", "down_proj")
LORA_RANK = 16


@torch.no_grad()
def solve_contrastive_neuron_proj_sep(
    X_pos: torch.Tensor,
    dY_pos: torch.Tensor,
    X_other: torch.Tensor,
    dY_other: torch.Tensor,
    X_neg: torch.Tensor,
    dY_neg: torch.Tensor,
    X_cal: torch.Tensor,
    dY_cal: torch.Tensor,
    w_pos: float = 2.2,
    w_other: float = 1.4,
    w_neg: float = 1.6,
    w_cal: float = 1.1,
    lambda_fro: float = 0.012,
    max_row_norm: float = 9.5,
) -> torch.Tensor:
    device = X_pos.device
    d_in = X_pos.shape[1]
    parts_x = [
        X_pos.float() * (w_pos / max(X_pos.shape[0], 1)) ** 0.5,
        X_other.float() * (w_other / max(X_other.shape[0], 1)) ** 0.5,
        X_neg.float() * (w_neg / max(X_neg.shape[0], 1)) ** 0.5,
        X_cal.float() * (w_cal / max(X_cal.shape[0], 1)) ** 0.5,
    ]
    parts_y = [
        dY_pos.float() * (w_pos / max(X_pos.shape[0], 1)) ** 0.5,
        dY_other.float() * (w_other / max(X_other.shape[0], 1)) ** 0.5,
        dY_neg.float() * (w_neg / max(X_neg.shape[0], 1)) ** 0.5,
        dY_cal.float() * (w_cal / max(X_cal.shape[0], 1)) ** 0.5,
    ]
    X_all = torch.cat(parts_x, dim=0)
    Y_all = torch.cat(parts_y, dim=0)
    mean_diag = max(float((X_all * X_all).sum().item() / d_in), 1e-6)
    gram = X_all.T @ X_all + (lambda_fro * mean_diag) * torch.eye(d_in, device=device, dtype=torch.float32)
    rhs = X_all.T @ Y_all
    del X_all, Y_all, parts_x, parts_y
    A = torch.linalg.solve(gram, rhs).T
    del gram, rhs
    row_norms = A.norm(dim=1, keepdim=True)
    return A * torch.clamp(max_row_norm / row_norms.clamp(min=1e-8), max=1.0)


@torch.no_grad()
def voronoi_guarded_targets_v2(
    model,
    H_cur: torch.Tensor,
    cm_map: Dict,
    pe: Tuple[torch.Tensor, torch.Tensor],
    text_pids: torch.Tensor,
    start_layer: int,
    pos_lists: List[List[int]],
    target_tok_ids: torch.Tensor,
    target_prob_goal: float = 0.95,
    seq_chunk: int = 32,
    compute_dH: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    """Memory-safe Voronoi-guarded analytical RMSNorm+lm_head target solver for Qwen3.5 (vocab=248,320)."""
    N = H_cur.shape[0]
    lm = get_text_model(model)
    n_layers = len(lm.layers)
    norm_layer = get_lm_norm(model)
    lm_head = get_lm_head(model)
    eps = norm_layer.eps
    # CRITICAL Qwen3_5RMSNorm detail: multiplier is (1.0 + weight)!
    gamma = (1.0 + norm_layer.weight.float())
    W_lm_h = lm_head.weight  # native fp16 [248320, 4096] (avoid 4 GB float32 copy!)
    lm_dtype = W_lm_h.dtype

    h_pts_list = []
    for s in range(0, N, seq_chunk):
        e = min(N, s + seq_chunk)
        hc = H_cur[s:e]
        cm_s = slice_cm_map(cm_map, s, e, N)
        pe_s = slice_pe(pe, s, e, N)
        tp_s = text_pids[s:e]
        for l_idx in range(start_layer, n_layers):
            hc = step_layer(model, l_idx, hc, cm_s, pe_s, tp_s)
        h_pts_list.append(gather_positions(hc, pos_lists[s:e]))

    hf = torch.cat(h_pts_list, dim=0).float()
    del h_pts_list
    M, _ = hf.shape
    p_target_all = torch.zeros(M, device=hf.device, dtype=torch.float32)
    dH_goal = torch.zeros_like(hf)
    top1_hits = 0
    overshoot_before = 0
    overshoot_after = 0

    # Scaled for Ornith-1.0-9B residual norms (125..247 at L27..L31)
    etas = (20.0, 40.0, 70.0, 105.0, 145.0, 190.0, 240.0)
    tok_chunk = 96

    def grad_at_h(h_vec: torch.Tensor, tids: torch.Tensor):
        inv_rms = torch.rsqrt((h_vec * h_vec).mean(dim=-1, keepdim=True) + eps)
        h_hat = h_vec * inv_rms
        logits = F.linear((h_hat * gamma.unsqueeze(0)).to(lm_dtype), W_lm_h).float()
        probs = F.softmax(logits, dim=-1)
        ar = torch.arange(h_vec.shape[0], device=h_vec.device)
        pt = probs[ar, tids]
        is_top1 = logits.argmax(dim=-1) == tids
        g_z = probs
        g_z[ar, tids] -= 1.0
        g_u = (g_z.to(lm_dtype) @ W_lm_h).float()
        g_hhat = g_u * gamma.unsqueeze(0)
        g_h = inv_rms * (g_hhat - (g_hhat * h_hat).mean(dim=-1, keepdim=True) * h_hat)
        return pt, is_top1, g_h

    def line_search_eta(hc: torch.Tensor, u: torch.Tensor, tids: torch.Tensor, need: torch.Tensor):
        best_eta = torch.zeros((hc.shape[0], 1), device=hc.device, dtype=torch.float32)
        solved = ~need.bool()
        ar = torch.arange(hc.shape[0], device=hc.device)
        for eta in etas:
            cand_h = hc + eta * u
            cand_inv = torch.rsqrt((cand_h * cand_h).mean(dim=-1, keepdim=True) + eps)
            cand_logits = F.linear(((cand_h * cand_inv) * gamma.unsqueeze(0)).to(lm_dtype), W_lm_h).float()
            cand_probs = F.softmax(cand_logits, dim=-1)
            cand_pt = cand_probs[ar, tids]
            cand_ok = (cand_pt >= target_prob_goal) & (cand_logits.argmax(dim=-1) == tids)
            newly = (~solved) & cand_ok
            best_eta[newly] = eta
            solved = solved | newly
        best_eta[~solved] = etas[-1]
        best_eta = best_eta * need.float().unsqueeze(1)
        return best_eta * u

    for ts in range(0, M, tok_chunk):
        te = min(M, ts + tok_chunk)
        hc = hf[ts:te]
        tids = target_tok_ids[ts:te]
        pt0, ok0, g_h0 = grad_at_h(hc, tids)
        p_target_all[ts:te] = pt0
        top1_hits += int(ok0.sum().item())

        if not compute_dH:
            continue

        need = pt0 < target_prob_goal
        u_dir = (-g_h0) / g_h0.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        dh = line_search_eta(hc, u_dir, tids, need)

        for r_idx, rep_w in enumerate((1.0, 1.1, 1.15, 1.2, 1.25)):
            cand_hi = hc + 1.45 * dh
            inv_hi = torch.rsqrt((cand_hi * cand_hi).mean(dim=-1, keepdim=True) + eps)
            logits_hi = F.linear(((cand_hi * inv_hi) * gamma.unsqueeze(0)).to(lm_dtype), W_lm_h).float()
            bad = need & (logits_hi.argmax(dim=-1) != tids)
            if r_idx == 0:
                overshoot_before += int(bad.sum().item())
            if not bad.any():
                break
            bad_idx = torch.nonzero(bad, as_tuple=False).squeeze(-1)
            _, _, g_h_bad = grad_at_h(cand_hi[bad_idx], tids[bad_idx])
            u_new = u_dir[bad_idx] + rep_w * ((-g_h_bad) / g_h_bad.norm(dim=-1, keepdim=True).clamp(min=1e-8))
            u_new = u_new / u_new.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            u_dir[bad_idx] = u_new
            dh[bad_idx] = line_search_eta(
                hc[bad_idx], u_new, tids[bad_idx], torch.ones(bad_idx.numel(), device=hc.device, dtype=torch.bool)
            )

        cand_hi = hc + 1.45 * dh
        inv_hi = torch.rsqrt((cand_hi * cand_hi).mean(dim=-1, keepdim=True) + eps)
        logits_hi = F.linear(((cand_hi * inv_hi) * gamma.unsqueeze(0)).to(lm_dtype), W_lm_h).float()
        overshoot_after += int((need & (logits_hi.argmax(dim=-1) != tids)).sum().item())
        dH_goal[ts:te] = dh

    return dH_goal, p_target_all, {
        "mean_p": round(float(p_target_all.mean().item()), 4),
        "top1_acc": round(float(top1_hits / max(M, 1)), 4),
        "ov_fix": f"{overshoot_before}->{overshoot_after}",
    }


@torch.no_grad()
def synthesize_mechA12_swiglu(
    model,
    tokenizer,
    lora_mgr: LoRAManager,
    batch: FactBatch,
    cal_texts: List[str],
    device: str,
    doc_text_stage2: str | None = None,
) -> Dict[str, object]:
    if doc_text_stage2 is not None:
        doc_enc = tokenizer(doc_text_stage2, return_tensors="pt", truncation=True, max_length=2048).to(device)
        _ = model(input_ids=doc_enc.input_ids, attention_mask=doc_enc.attention_mask, use_cache=False)
        del doc_enc

    pos_items, neg_prompt_items = build_probes_mechA10(batch.stage1_edit_texts, tokenizer)
    prompts = [x["prompt"] for x in pos_items]
    targets = [x["target"] for x in pos_items]

    stu_ids, stu_mask, stu_pos = tokenize_with_target_positions(tokenizer, prompts, targets, device)
    target_tok_list: List[int] = []
    token_fact_ids: List[int] = []
    token_step_ids: List[int] = []
    unique_pairs_pos0: List[Tuple[int, int]] = []
    unique_pairs_eos: List[Tuple[int, int]] = []
    unique_pairs_mid: List[Tuple[int, int]] = []
    seen_pairs = set()

    for item in pos_items:
        f_idx = int(item["fact_idx"])
        t_ids = tokenizer.encode(item["target"], add_special_tokens=False)
        eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        full_t = t_ids + eos
        eos_step = len(full_t) - 1 if eos else -1
        for k_step, tok in enumerate(full_t):
            target_tok_list.append(tok)
            token_fact_ids.append(f_idx)
            token_step_ids.append(k_step)
            pair = (f_idx, k_step)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                if k_step == 0:
                    unique_pairs_pos0.append(pair)
                elif k_step == eos_step:
                    unique_pairs_eos.append(pair)
                else:
                    unique_pairs_mid.append(pair)

    target_tok_ids = torch.tensor(target_tok_list, dtype=torch.long, device=device)
    token_fact_ids_t = torch.tensor(token_fact_ids, dtype=torch.long, device=device)
    token_step_ids_t = torch.tensor(token_step_ids, dtype=torch.long, device=device)

    lm = get_text_model(model)
    layers = get_layers(model)
    n_layers = len(layers)

    total_pairs = len(unique_pairs_pos0) + len(unique_pairs_eos) + len(unique_pairs_mid)
    needed_layers = max(9, math.ceil(total_pairs / LORA_RANK))
    cand_layers = list(range(n_layers - needed_layers, n_layers))
    layer_shards: Dict[int, List[Tuple[int, int]]] = {l: [] for l in cand_layers}

    p0_order = [27, 28, 29, 26, 27, 28, 29, 26, 27, 28, 29, 26, 27, 28, 29, 26, 27, 28, 29, 26, 27, 28, 27, 28, 29]
    for idx, pair in enumerate(unique_pairs_pos0):
        layer_shards[p0_order[idx]].append(pair)
    for idx, pair in enumerate(unique_pairs_eos):
        layer_shards[[31, 30][idx % 2]].append(pair)
    for pair in unique_pairs_mid:
        placed = False
        for l in reversed(cand_layers):
            if len(layer_shards[l]) < LORA_RANK:
                layer_shards[l].append(pair)
                placed = True
                break
        if not placed:
            raise RuntimeError(f"Exceeded rank-16 neuron capacity ({total_pairs}) across {cand_layers[0]}..{n_layers-1}!")

    active_layers = [l for l in cand_layers if len(layer_shards[l]) > 0]
    first_layer = active_layers[0]

    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    neg_enc = tokenizer([x["prompt"] for x in neg_prompt_items], return_tensors="pt", padding=True).to(device)
    cal_enc = tokenizer(cal_texts[:96], return_tensors="pt", padding=True, truncation=True, max_length=96).to(device)
    tokenizer.padding_side = prev_side
    neg_pos = [[L - 1] for L in neg_enc.attention_mask.sum(dim=1).tolist()]

    def init_and_prefix(ids, mask, bs: int = 32):
        emb_full, cm_full, pe_full, tp_full = prepare_layer_inputs(model, ids, mask)
        chunks = []
        n_tot = ids.shape[0]
        for s in range(0, n_tot, bs):
            e = min(n_tot, s + bs)
            h = emb_full[s:e]
            cm_s = slice_cm_map(cm_full, s, e, n_tot)
            pe_s = slice_pe(pe_full, s, e, n_tot)
            tp_s = tp_full[s:e]
            for l_idx in range(first_layer):
                h = step_layer(model, l_idx, h, cm_s, pe_s, tp_s)
            chunks.append(h.detach())
        del emb_full
        return torch.cat(chunks, dim=0), cm_full, pe_full, tp_full

    lora_mgr.clear_all()
    H_stu, cm_stu, pe_stu, tp_stu = init_and_prefix(stu_ids, stu_mask)
    H_neg, cm_neg, pe_neg, tp_neg = init_and_prefix(neg_enc.input_ids, neg_enc.attention_mask)
    H_cal, cm_cal, pe_cal, tp_cal = init_and_prefix(cal_enc.input_ids, cal_enc.attention_mask)
    cal_mask_bool = cal_enc.attention_mask.bool()
    del stu_ids, stu_mask, neg_enc, cal_enc

    layer_stats: List[Dict[str, object]] = []

    for l_idx in range(first_layer, n_layers):
        layer = layers[l_idx]
        shard_pairs = layer_shards.get(l_idx, [])
        if not shard_pairs:
            def step_plain(h_in, cm_m, pe_t, tp_t, bs: int = 32):
                outs = []
                n_tot = h_in.shape[0]
                for s in range(0, n_tot, bs):
                    e = min(n_tot, s + bs)
                    outs.append(
                        step_layer(
                            model, l_idx, h_in[s:e],
                            slice_cm_map(cm_m, s, e, n_tot), slice_pe(pe_t, s, e, n_tot), tp_t[s:e]
                        )
                    )
                return torch.cat(outs, dim=0)
            H_stu = step_plain(H_stu, cm_stu, pe_stu, tp_stu)
            H_neg = step_plain(H_neg, cm_neg, pe_neg, tp_neg)
            H_cal = step_plain(H_cal, cm_cal, pe_cal, tp_cal)
            continue

        def forward_mixer_to_mlp_in(h_in, cm_m, pe_t, tp_t, bs: int = 32):
            mids, xs = [], []
            n_tot = h_in.shape[0]
            for s in range(0, n_tot, bs):
                e = min(n_tot, s + bs)
                hm, xm = step_token_mixer(
                    model, l_idx, h_in[s:e],
                    slice_cm_map(cm_m, s, e, n_tot), slice_pe(pe_t, s, e, n_tot), tp_t[s:e]
                )
                mids.append(hm)
                xs.append(xm)
            return torch.cat(mids, dim=0), torch.cat(xs, dim=0)

        H_mid_stu, X_mlp_stu_all = forward_mixer_to_mlp_in(H_stu, cm_stu, pe_stu, tp_stu)
        H_mid_neg, X_mlp_neg_all = forward_mixer_to_mlp_in(H_neg, cm_neg, pe_neg, tp_neg)
        H_mid_cal, X_mlp_cal_all = forward_mixer_to_mlp_in(H_cal, cm_cal, pe_cal, tp_cal)

        X_mlp_stu = gather_positions(X_mlp_stu_all, stu_pos)
        X_mlp_neg = gather_positions(X_mlp_neg_all, neg_pos)
        X_mlp_cal = X_mlp_cal_all[cal_mask_bool]

        dH_goal, p_before, logit_diag = voronoi_guarded_targets_v2(
            model, H_stu, cm_stu, pe_stu, tp_stu, l_idx, stu_pos, target_tok_ids,
            target_prob_goal=0.95, seq_chunk=32, compute_dH=True,
        )

        r_neurons = len(shard_pairs)
        assert r_neurons <= LORA_RANK

        # Streaming dormancy score across 12,288 SwiGLU neurons via 8-bit base gate_proj (chunked to save VRAM)
        d_ff = layer.mlp.gate_proj.out_features
        sum_silu2 = torch.zeros(d_ff, device=device, dtype=torch.float32)
        sum_pos = torch.zeros(d_ff, device=device, dtype=torch.float32)
        n_cal_toks = X_mlp_cal.shape[0]
        for cs in range(0, n_cal_toks, 1024):
            ce = min(n_cal_toks, cs + 1024)
            gc_chunk = layer.mlp.gate_proj.base(X_mlp_cal[cs:ce]).float()
            sum_silu2 += (F.silu(gc_chunk) ** 2).sum(dim=0)
            sum_pos += gc_chunk.clamp(min=0.0).sum(dim=0)
            del gc_chunk
        dormancy_score = (sum_silu2 / max(n_cal_toks, 1)) + 0.05 * (sum_pos / max(n_cal_toks, 1))
        J_l = torch.topk(dormancy_score, k=r_neurons, largest=False).indices
        del sum_silu2, sum_pos, dormancy_score

        shard_mask = torch.zeros_like(token_fact_ids_t, dtype=torch.bool)
        cluster_ids_full = torch.full_like(token_fact_ids_t, -1)
        for c_i, (f_id, k_st) in enumerate(shard_pairs):
            pm = (token_fact_ids_t == f_id) & (token_step_ids_t == k_st)
            shard_mask |= pm
            cluster_ids_full[pm] = c_i

        other_mask = ~shard_mask
        X_shard = X_mlp_stu[shard_mask]
        X_other = X_mlp_stu[other_mask]
        cluster_ids = cluster_ids_full[shard_mask]

        # Dequantize ONLY the r <= 16 dormant neuron rows from 8-bit gate_proj & up_proj
        W_gate_J = dequantize_rows(layer.mlp.gate_proj.base, J_l)
        W_up_J = dequantize_rows(layer.mlp.up_proj.base, J_l)

        g_base_shard = X_shard.float() @ W_gate_J.T
        g_goal_shard = torch.full_like(g_base_shard, -4.5)
        g_goal_shard[torch.arange(X_shard.shape[0], device=device), cluster_ids] = 4.8
        dG_pos = g_goal_shard - g_base_shard
        dG_other = torch.full((X_other.shape[0], r_neurons), -4.5, device=device, dtype=torch.float32) - (X_other.float() @ W_gate_J.T)
        dG_neg = torch.full((X_mlp_neg.shape[0], r_neurons), -4.5, device=device, dtype=torch.float32) - (X_mlp_neg.float() @ W_gate_J.T)
        g_base_cal_J = X_mlp_cal.float() @ W_gate_J.T
        dG_cal = torch.minimum(g_base_cal_J, torch.full_like(g_base_cal_J, -3.5)) - g_base_cal_J

        A_gate = solve_contrastive_neuron_proj_sep(
            X_shard, dG_pos, X_other, dG_other, X_mlp_neg, dG_neg, X_mlp_cal, dG_cal,
            w_pos=2.2, w_other=1.4, w_neg=1.6, w_cal=1.1, lambda_fro=0.012, max_row_norm=9.5,
        )
        B_select = torch.zeros((d_ff, r_neurons), device=device, dtype=torch.float32)
        B_select[J_l, torch.arange(r_neurons, device=device)] = 1.0
        lora_mgr.set_layer_adapter(l_idx, "gate_proj", A_gate, B_select, alpha=1.0)
        del dG_pos, dG_other, dG_neg, dG_cal, g_base_cal_J

        u_base_shard = X_shard.float() @ W_up_J.T
        u_goal_shard = torch.zeros_like(u_base_shard)
        u_goal_shard[torch.arange(X_shard.shape[0], device=device), cluster_ids] = 3.2
        dU_pos = u_goal_shard - u_base_shard
        dU_other = -(X_other.float() @ W_up_J.T)
        dU_neg = -(X_mlp_neg.float() @ W_up_J.T)
        dU_cal = -(X_mlp_cal.float() @ W_up_J.T)

        A_up = solve_contrastive_neuron_proj_sep(
            X_shard, dU_pos, X_other, dU_other, X_mlp_neg, dU_neg, X_mlp_cal, dU_cal,
            w_pos=2.2, w_other=1.4, w_neg=1.6, w_cal=1.1, lambda_fro=0.012, max_row_norm=9.5,
        )
        lora_mgr.set_layer_adapter(l_idx, "up_proj", A_up, B_select, alpha=1.0)
        del dU_pos, dU_other, dU_neg, dU_cal, W_gate_J, W_up_J

        def eval_J_act(X_toks: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
            zs = []
            for cs in range(0, X_toks.shape[0], chunk):
                ce = min(X_toks.shape[0], cs + chunk)
                xc = X_toks[cs:ce]
                g = layer.mlp.gate_proj(xc)[:, J_l].float()
                u = layer.mlp.up_proj(xc)[:, J_l].float()
                zs.append(F.silu(g) * u)
            return torch.cat(zs, dim=0)

        Z_all = eval_J_act(X_mlp_stu)
        Z_shard = Z_all[shard_mask]
        Z_other = Z_all[other_mask]
        Z_neg = eval_J_act(X_mlp_neg)
        Z_cal = eval_J_act(X_mlp_cal)

        J_diag = Z_shard[torch.arange(Z_shard.shape[0], device=device), cluster_ids].abs().mean().item()
        J_neg_mean = Z_neg.abs().mean().item()
        J_cal_mean = Z_cal.abs().mean().item()

        gram_r = (
            (Z_shard.T @ Z_shard)
            + 1.5 * (Z_other.T @ Z_other)
            + 3.5 * (Z_neg.T @ Z_neg)
            + 3.0 * (Z_cal.T @ Z_cal)
            + 0.4 * torch.eye(r_neurons, device=device, dtype=torch.float32)
        )
        B_down_T = torch.linalg.solve(gram_r, Z_shard.T @ dH_goal[shard_mask].float())
        B_down = B_down_T.T

        pred_dH = Z_shard @ B_down.T
        fit_err = float((pred_dH - dH_goal[shard_mask].float()).norm().item() / (dH_goal[shard_mask].float().norm().item() + 1e-8))
        A_down = B_select.T.contiguous()
        del Z_all, Z_shard, Z_other, Z_neg, Z_cal, X_shard, X_other, X_mlp_stu, X_mlp_neg, X_mlp_cal

        def forward_mlp_from_mid(h_mid, x_mlp, bs: int = 32):
            outs = []
            for s in range(0, h_mid.shape[0], bs):
                e = min(h_mid.shape[0], s + bs)
                outs.append(h_mid[s:e] + layer.mlp(x_mlp[s:e]))
            return torch.cat(outs, dim=0)

        lora_mgr.set_layer_adapter(l_idx, "down_proj", A_down, B_down, alpha=1.0)
        H_stu_1 = forward_mlp_from_mid(H_mid_stu, X_mlp_stu_all)
        _, p_after_1, _ = voronoi_guarded_targets_v2(
            model, H_stu_1, cm_stu, pe_stu, tp_stu, l_idx + 1, stu_pos, target_tok_ids, seq_chunk=32, compute_dH=False
        )
        del H_stu_1

        p_after_2 = p_after_1
        if l_idx in (26, 27, 28, 29) or (l_idx < 31 and float(p_after_1[shard_mask].mean().item()) < 0.48):
            lora_mgr.set_layer_adapter(l_idx, "down_proj", A_down, B_down, alpha=1.9)
            H_stu_2 = forward_mlp_from_mid(H_mid_stu, X_mlp_stu_all)
            _, p_after_2, _ = voronoi_guarded_targets_v2(
                model, H_stu_2, cm_stu, pe_stu, tp_stu, l_idx + 1, stu_pos, target_tok_ids, seq_chunk=32, compute_dH=False
            )
            del H_stu_2

        p0_s = p_before[shard_mask].clamp(1e-4, 1.0 - 1e-4)
        p1_s = p_after_1[shard_mask].clamp(1e-4, 1.0 - 1e-4)
        p2_s = p_after_2[shard_mask].clamp(1e-4, 1.0 - 1e-4)
        ell0 = torch.log(p0_s / (1.0 - p0_s))
        ell1 = torch.log(p1_s / (1.0 - p1_s))

        alpha_vec = torch.ones(r_neurons, device=device, dtype=torch.float32)
        for j, (_, k_st) in enumerate(shard_pairs):
            jm = cluster_ids == j
            m0 = float(ell0[jm].mean().item())
            m1 = float(ell1[jm].mean().item())
            p1_m = float(p1_s[jm].mean().item())
            p2_m = float(p2_s[jm].mean().item())
            dm = m1 - m0
            if k_st == 0:
                if p2_m > p1_m + 0.015:
                    aj = (4.5 - m0) / max(dm, 0.35)
                    alpha_vec[j] = max(1.0, min(2.45, aj))
                elif dm > 0.25:
                    alpha_vec[j] = max(0.85, min(1.9, (4.0 - m0) / dm))
            else:
                if l_idx < 31 and p2_m > p1_m + 0.03:
                    aj = (3.4 - m0) / max(dm, 0.35)
                    alpha_vec[j] = max(0.90, min(1.85, aj))
                elif dm > 0.25:
                    alpha_vec[j] = max(0.75, min(1.55, (3.1 - m0) / dm))

        B_down_cal = B_down * alpha_vec.unsqueeze(0)
        lora_mgr.set_layer_adapter(l_idx, "down_proj", A_down, B_down_cal, alpha=1.0)

        H_stu = forward_mlp_from_mid(H_mid_stu, X_mlp_stu_all)
        H_neg = forward_mlp_from_mid(H_mid_neg, X_mlp_neg_all)
        H_cal = forward_mlp_from_mid(H_mid_cal, X_mlp_cal_all)
        del H_mid_stu, X_mlp_stu_all, H_mid_neg, X_mlp_neg_all, H_mid_cal, X_mlp_cal_all

        layer_stats.append({
            "layer": l_idx,
            "layer_type": lm.config.layer_types[l_idx],
            "r_used": r_neurons,
            **logit_diag,
            "J_diag_pos": round(J_diag, 3),
            "J_neg": round(J_neg_mean, 4),
            "J_cal": round(J_cal_mean, 4),
            "d_fit_err": round(fit_err, 4),
            "alpha_mean": round(float(alpha_vec.mean().item()), 2),
            "shard_p_1": round(float(p_after_1[shard_mask].mean().item()), 4),
            "shard_p_2": round(float(p_after_2[shard_mask].mean().item()), 4),
        })

    _, _, final_diag = voronoi_guarded_targets_v2(
        model, H_stu, cm_stu, pe_stu, tp_stu, n_layers, stu_pos, target_tok_ids, seq_chunk=32, compute_dH=False
    )
    return {"layer_stats": layer_stats, "final_train_diag": final_diag, "total_pairs": total_pairs}


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_ornith_8bit(device=device, dtype=torch.float16, drop_visual=True)
    static_vram_mb = measure_static_vram_mb(model)

    cal_texts, kl_texts = load_smoltalk_splits(tokenizer, n_cal=128, n_kl_eval=64)
    batch42 = generate_fact_batch(42, tokenizer=tokenizer, target_doc_tokens=2048)

    lora_mgr = LoRAManager(model, target_modules=TARGET_MODULES)
    lora_mgr.install()

    baselines = json.loads((LAB_DIR / "baselines_summary.json").read_text())
    base_a_ref = baselines["baseline_a_512"]
    base_ref = get_base_reference(model, tokenizer, batch42, device, lora_manager=lora_mgr)

    with NoBackpropGuard(model, regime="no_grad", static_vram_mb=static_vram_mb) as guard:
        guard.assert_backbone_no_grad()
        diag = synthesize_mechA12_swiglu(model, tokenizer, lora_mgr, batch42, cal_texts, device)

    assert lora_mgr.max_active_rank() <= 16
    eval_res = evaluate_model(model, tokenizer, batch42, kl_texts, device, lora_manager=lora_mgr, compute_kl=True, base_ref_override=base_ref)
    gates = check_gates(eval_res, guard.metrics, baseline_a_ref=base_a_ref)
    status = "pass" if gates["ALL_GATES_PASS"] else "iterate"

    rec = append_experiment_record(
        28,
        "MechA12_SweetSpot_SwiGLU",
        eval_res,
        guard.metrics,
        status=status,
        seed=42,
        stage=1,
        rank=lora_mgr.max_active_rank(),
        extra={"gates": gates, "final_train_diag": diag["final_train_diag"], "layer_stats": diag["layer_stats"]},
    )
    print("\n[Mech A12 / Iter 28] Result:", json.dumps(rec, indent=2))
    print("[Mech A12 / Iter 28] Gates:", json.dumps(gates, indent=2))
    print("[Mech A12 / Iter 28] Final train diag:", diag["final_train_diag"])
    print_locality_breakdown(model, tokenizer, lora_mgr, batch42, device, base_ref=base_ref)


if __name__ == "__main__":
    main()
