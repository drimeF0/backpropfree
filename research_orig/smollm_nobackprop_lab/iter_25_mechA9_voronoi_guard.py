"""Mechanism A9 (Iter 25): Voronoi-Guarded Fast Contrastive SwiGLU LoRA (0 Backprop).

Combines all three verified breakthroughs without Iter 24's norm-underflow bug:
1. Active Voronoi Overshoot Guard inside chunked_logit_residual_targets:
   Keeps Iter 20/22's high-probability ray search (etas=[25..200], p>=0.95 -> shard_p=0.73+),
   then verifies EVERY target token at 1.45 * dH. Whenever 1.45 * dH hits a rare-token
   Voronoi competitor ('firehose', 'KeyNot', 'reponame', 'ikipedia', 'theless'), adds 1-2
   repulsive gradient rounds (-g_h at the failing overshoot point) and re-solves eta so BOTH
   1.0*dH (p>=0.95) and 1.45*dH (argmax==target) stay inside the target's Voronoi cell!
2. Clean L26..L29 pos_0 + L30..L31 <eos> + L23..L31 top-down suffixes (skipping L22 when <=144
   pairs -> <= 18s runtime!).
3. Two-point pos_0 calibration (1.0 vs 1.8) on L26..L29 + 300 matched disjoint negatives.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask

from benchmark import FactBatch, generate_fact_batch, load_smoltalk_splits
from eval_harness import append_experiment_record, check_gates, evaluate_model
from guards import LoRAManager, NoBackpropGuard, cleanup_cuda, measure_static_vram_mb
from iter_1_experiment import gather_positions, tokenize_with_target_positions
from iter_13_experiment import print_locality_breakdown
from iter_18_mechA2_contrastive_swiglu import solve_contrastive_neuron_proj_v2
from iter_23_mechA7_stable_logits import build_balanced_probes_mechA7

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
LAB_DIR = Path("/content/smollm_nobackprop_lab")
TARGET_MODULES = ("gate_proj", "up_proj", "down_proj")
LORA_RANK = 16


@torch.no_grad()
def voronoi_guarded_logit_targets(
    model,
    H_cur: torch.Tensor,
    cmask: torch.Tensor,
    pemb: torch.Tensor,
    start_layer: int,
    pos_lists: List[List[int]],
    target_tok_ids: torch.Tensor,
    target_prob_goal: float = 0.95,
    seq_chunk: int = 128,
    compute_dH: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Chunked no_grad() RMSNorm+lm_head targets with 1.45x Voronoi overshoot repulsion."""
    N = H_cur.shape[0]
    norm_layer = model.model.norm
    lm_head = model.lm_head
    eps = norm_layer.variance_epsilon
    gamma = norm_layer.weight.float()
    W_lm = lm_head.weight.float()

    h_pts_list = []
    for s in range(0, N, seq_chunk):
        e = min(N, s + seq_chunk)
        hc = H_cur[s:e]
        mc = cmask[s:e] if cmask is not None and cmask.shape[0] == N else cmask
        for l_idx in range(start_layer, len(model.model.layers)):
            hc = model.model.layers[l_idx](hc, attention_mask=mc, position_embeddings=pemb, use_cache=False)
        h_pts_list.append(gather_positions(hc, pos_lists[s:e]))

    hf = torch.cat(h_pts_list, dim=0).float()
    M, _ = hf.shape
    p_target_all = torch.zeros(M, device=hf.device, dtype=torch.float32)
    dH_goal = torch.zeros_like(hf)
    top1_hits = 0
    overshoot_before = 0
    overshoot_after = 0

    etas = (25.0, 50.0, 85.0, 125.0, 170.0, 220.0)
    tok_chunk = 240

    def grad_at_h(h_vec: torch.Tensor, tids: torch.Tensor):
        inv_rms = torch.rsqrt((h_vec * h_vec).mean(dim=-1, keepdim=True) + eps)
        h_hat = h_vec * inv_rms
        logits = (h_hat * gamma.unsqueeze(0)) @ W_lm.T
        probs = F.softmax(logits, dim=-1)
        pt = probs[torch.arange(h_vec.shape[0], device=h_vec.device), tids]
        is_top1 = logits.argmax(dim=-1) == tids
        g_z = probs
        g_z[torch.arange(h_vec.shape[0], device=h_vec.device), tids] -= 1.0
        g_u = g_z @ W_lm
        g_hhat = g_u * gamma.unsqueeze(0)
        g_h = inv_rms * (g_hhat - (g_hhat * h_hat).mean(dim=-1, keepdim=True) * h_hat)
        return pt, is_top1, g_h

    def line_search_eta(hc: torch.Tensor, u: torch.Tensor, tids: torch.Tensor, need: torch.Tensor):
        best_eta = torch.zeros((hc.shape[0], 1), device=hc.device, dtype=torch.float32)
        solved = ~need.bool()
        for eta in etas:
            cand_h = hc + eta * u
            cand_inv = torch.rsqrt((cand_h * cand_h).mean(dim=-1, keepdim=True) + eps)
            cand_logits = ((cand_h * cand_inv) * gamma.unsqueeze(0)) @ W_lm.T
            cand_probs = F.softmax(cand_logits, dim=-1)
            cand_pt = cand_probs[torch.arange(hc.shape[0], device=hc.device), tids]
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

        # Up to 3 Voronoi repulsion rounds if 1.45*dh crosses into 'firehose'/'KeyNot'/'ikipedia'!
        for rep_w in (1.0, 1.1, 1.2):
            cand_hi = hc + 1.45 * dh
            inv_hi = torch.rsqrt((cand_hi * cand_hi).mean(dim=-1, keepdim=True) + eps)
            logits_hi = ((cand_hi * inv_hi) * gamma.unsqueeze(0)) @ W_lm.T
            bad = need & (logits_hi.argmax(dim=-1) != tids)
            if rep_w == 1.0:
                overshoot_before += int(bad.sum().item())
            if not bad.any():
                break
            bad_idx = torch.nonzero(bad, as_tuple=False).squeeze(-1)
            _, _, g_h_bad = grad_at_h(cand_hi[bad_idx], tids[bad_idx])
            u_new = u_dir[bad_idx] + rep_w * ((-g_h_bad) / g_h_bad.norm(dim=-1, keepdim=True).clamp(min=1e-8))
            u_new = u_new / u_new.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            u_dir[bad_idx] = u_new
            dh_fixed = line_search_eta(hc[bad_idx], u_new, tids[bad_idx], torch.ones(bad_idx.numel(), device=hc.device, dtype=torch.bool))
            dh[bad_idx] = dh_fixed

        # Count any remaining overshoot
        cand_hi = hc + 1.45 * dh
        inv_hi = torch.rsqrt((cand_hi * cand_hi).mean(dim=-1, keepdim=True) + eps)
        logits_hi = ((cand_hi * inv_hi) * gamma.unsqueeze(0)) @ W_lm.T
        overshoot_after += int((need & (logits_hi.argmax(dim=-1) != tids)).sum().item())
        dH_goal[ts:te] = dh

    return dH_goal, p_target_all, {
        "mean_p": round(float(p_target_all.mean().item()), 4),
        "top1_acc": round(float(top1_hits / max(M, 1)), 4),
        "ov_fix": f"{overshoot_before}->{overshoot_after}",
    }


@torch.no_grad()
def synthesize_mechA9_swiglu(
    model,
    tokenizer,
    lora_mgr: LoRAManager,
    batch: FactBatch,
    cal_texts: List[str],
    device: str,
    lambda_fro_gate: float = 0.013,
    max_row_norm_gate: float = 9.0,
    w_cal_gate: float = 1.0,
) -> Dict[str, object]:
    pos_items, neg_prompt_items = build_balanced_probes_mechA7(batch.stage1_edit_texts, tokenizer)
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

    total_pairs = len(unique_pairs_pos0) + len(unique_pairs_eos) + len(unique_pairs_mid)
    cand_layers = list(range(23, 32)) if total_pairs <= 144 else list(range(22, 32))
    layer_shards: Dict[int, List[Tuple[int, int]]] = {l: [] for l in cand_layers}
    for idx, pair in enumerate(unique_pairs_pos0):
        layer_shards[[26, 27, 28, 29][idx % 4]].append(pair)
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
            raise RuntimeError(f"Exceeded rank-16 neuron capacity ({total_pairs}) across {cand_layers[0]}..31!")

    active_layers = [l for l in cand_layers if len(layer_shards[l]) > 0]
    first_layer = active_layers[0]

    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    neg_enc = tokenizer([x["prompt"] for x in neg_prompt_items], return_tensors="pt", padding=True).to(device)
    cal_enc = tokenizer(cal_texts[:96], return_tensors="pt", padding=True, truncation=True, max_length=96).to(device)
    tokenizer.padding_side = prev_side
    neg_pos = [[L - 1] for L in neg_enc.attention_mask.sum(dim=1).tolist()]

    def init_and_prefix(ids, mask, bs: int = 128):
        chunks = []
        emb_full = model.model.embed_tokens(ids)
        pids = torch.arange(emb_full.shape[1], device=device).unsqueeze(0)
        cm_full = create_causal_mask(config=model.config, inputs_embeds=emb_full, attention_mask=mask, past_key_values=None, position_ids=pids)
        pe = model.model.rotary_emb(emb_full, position_ids=pids)
        for s in range(0, ids.shape[0], bs):
            e = min(ids.shape[0], s + bs)
            h = emb_full[s:e]
            cm = cm_full[s:e] if cm_full is not None and cm_full.shape[0] == ids.shape[0] else cm_full
            for l_idx in range(first_layer):
                h = model.model.layers[l_idx](h, attention_mask=cm, position_embeddings=pe, use_cache=False)
            chunks.append(h.detach())
        return torch.cat(chunks, dim=0), cm_full, pe

    lora_mgr.clear_all()
    H_stu, cm_stu, pe_stu = init_and_prefix(stu_ids, stu_mask)
    H_neg, cm_neg, pe_neg = init_and_prefix(neg_enc.input_ids, neg_enc.attention_mask)
    H_cal, cm_cal, pe_cal = init_and_prefix(cal_enc.input_ids, cal_enc.attention_mask)

    cal_mask_bool = cal_enc.attention_mask.bool()
    layer_stats: List[Dict[str, object]] = []

    for l_idx in range(first_layer, 32):
        layer = model.model.layers[l_idx]
        shard_pairs = layer_shards.get(l_idx, [])
        if not shard_pairs:
            H_stu = layer(H_stu, attention_mask=cm_stu, position_embeddings=pe_stu, use_cache=False)
            H_neg = layer(H_neg, attention_mask=cm_neg, position_embeddings=pe_neg, use_cache=False)
            H_cal = layer(H_cal, attention_mask=cm_cal, position_embeddings=pe_cal, use_cache=False)
            continue

        def forward_attn_to_mlp_in(h_in, cm, pe, bs: int = 128):
            mids, xs = [], []
            for s in range(0, h_in.shape[0], bs):
                e = min(h_in.shape[0], s + bs)
                cm_s = cm[s:e] if cm is not None and cm.shape[0] == h_in.shape[0] else cm
                ao, _ = layer.self_attn(hidden_states=layer.input_layernorm(h_in[s:e]), attention_mask=cm_s, position_embeddings=pe, use_cache=False)
                hm = h_in[s:e] + ao
                mids.append(hm)
                xs.append(layer.post_attention_layernorm(hm))
            return torch.cat(mids, dim=0), torch.cat(xs, dim=0)

        H_mid_stu, X_mlp_stu_all = forward_attn_to_mlp_in(H_stu, cm_stu, pe_stu)
        H_mid_neg, X_mlp_neg_all = forward_attn_to_mlp_in(H_neg, cm_neg, pe_neg)
        H_mid_cal, X_mlp_cal_all = forward_attn_to_mlp_in(H_cal, cm_cal, pe_cal)

        X_mlp_stu = gather_positions(X_mlp_stu_all, stu_pos)
        X_mlp_neg = gather_positions(X_mlp_neg_all, neg_pos)
        X_mlp_cal = X_mlp_cal_all[cal_mask_bool]

        dH_goal, p_before, logit_diag = voronoi_guarded_logit_targets(
            model, H_stu, cm_stu, pe_stu, l_idx, stu_pos, target_tok_ids, target_prob_goal=0.95, seq_chunk=128, compute_dH=True
        )

        r_neurons = len(shard_pairs)
        assert r_neurons <= LORA_RANK
        g_cal_base = F.linear(X_mlp_cal, layer.mlp.gate_proj.base.weight).float()
        dormancy_score = (F.silu(g_cal_base) ** 2).mean(dim=0) + 0.05 * g_cal_base.clamp(min=0.0).mean(dim=0)
        J_l = torch.topk(dormancy_score, k=r_neurons, largest=False).indices

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

        W_gate_J = layer.mlp.gate_proj.base.weight[J_l].float()
        W_up_J = layer.mlp.up_proj.base.weight[J_l].float()

        g_base_shard = X_shard.float() @ W_gate_J.T
        g_goal_shard = torch.full_like(g_base_shard, -4.5)
        g_goal_shard[torch.arange(X_shard.shape[0], device=device), cluster_ids] = 4.5
        dG_pos = g_goal_shard - g_base_shard

        X_neg_comb = torch.cat([X_other, X_mlp_neg], dim=0)
        g_base_neg = X_neg_comb.float() @ W_gate_J.T
        dG_neg = torch.full_like(g_base_neg, -4.5) - g_base_neg

        g_base_cal_J = g_cal_base[:, J_l]
        g_goal_cal_J = torch.minimum(g_base_cal_J, torch.full_like(g_base_cal_J, -3.5))
        dG_cal = g_goal_cal_J - g_base_cal_J

        A_gate = solve_contrastive_neuron_proj_v2(
            X_shard, dG_pos, X_neg_comb, dG_neg, X_mlp_cal, dG_cal,
            w_pos=2.0, w_neg=2.2, w_cal=w_cal_gate, lambda_fro=lambda_fro_gate, max_row_norm=max_row_norm_gate,
        )
        B_select = torch.zeros((layer.mlp.gate_proj.out_features, r_neurons), device=device, dtype=torch.float32)
        B_select[J_l, torch.arange(r_neurons, device=device)] = 1.0
        lora_mgr.set_layer_adapter(l_idx, "gate_proj", A_gate, B_select, alpha=1.0)

        u_base_shard = X_shard.float() @ W_up_J.T
        u_goal_shard = torch.zeros_like(u_base_shard)
        u_goal_shard[torch.arange(X_shard.shape[0], device=device), cluster_ids] = 3.0
        dU_pos = u_goal_shard - u_base_shard
        dU_neg = -(X_neg_comb.float() @ W_up_J.T)
        dU_cal = -(X_mlp_cal.float() @ W_up_J.T)

        A_up = solve_contrastive_neuron_proj_v2(
            X_shard, dU_pos, X_neg_comb, dU_neg, X_mlp_cal, dU_cal,
            w_pos=2.0, w_neg=2.2, w_cal=w_cal_gate, lambda_fro=lambda_fro_gate, max_row_norm=max_row_norm_gate,
        )
        lora_mgr.set_layer_adapter(l_idx, "up_proj", A_up, B_select, alpha=1.0)

        Z_all = (F.silu(layer.mlp.gate_proj(X_mlp_stu)) * layer.mlp.up_proj(X_mlp_stu))[:, J_l].float()
        Z_shard = Z_all[shard_mask]
        Z_other = Z_all[other_mask]
        Z_neg = (F.silu(layer.mlp.gate_proj(X_mlp_neg)) * layer.mlp.up_proj(X_mlp_neg))[:, J_l].float()
        Z_cal = (F.silu(layer.mlp.gate_proj(X_mlp_cal)) * layer.mlp.up_proj(X_mlp_cal))[:, J_l].float()

        J_diag = Z_shard[torch.arange(Z_shard.shape[0], device=device), cluster_ids].abs().mean().item()
        J_neg_mean = Z_neg.abs().mean().item()
        J_cal_mean = Z_cal.abs().mean().item()

        Z_prot = torch.cat([Z_other, Z_neg, Z_cal], dim=0)
        gram_r = (Z_shard.T @ Z_shard) + 3.0 * (Z_prot.T @ Z_prot) + 0.5 * torch.eye(r_neurons, device=device, dtype=torch.float32)
        B_down_T = torch.linalg.solve(gram_r, Z_shard.T @ dH_goal[shard_mask].float())
        B_down = B_down_T.T

        pred_dH = Z_shard @ B_down.T
        fit_err = float((pred_dH - dH_goal[shard_mask].float()).norm().item() / (dH_goal[shard_mask].float().norm().item() + 1e-8))
        A_down = B_select.T.contiguous()

        # Check alpha=1.0 (all layers) and alpha=1.9 (for L26..L29 where pos_0 neurons sit!)
        lora_mgr.set_layer_adapter(l_idx, "down_proj", A_down, B_down, alpha=1.0)
        H_stu_1 = H_mid_stu + layer.mlp.down_proj(F.silu(layer.mlp.gate_proj(X_mlp_stu_all)) * layer.mlp.up_proj(X_mlp_stu_all))
        _, p_after_1, _ = voronoi_guarded_logit_targets(
            model, H_stu_1, cm_stu, pe_stu, l_idx + 1, stu_pos, target_tok_ids, seq_chunk=128, compute_dH=False
        )

        p_after_2 = p_after_1
        if l_idx in (26, 27, 28, 29):
            lora_mgr.set_layer_adapter(l_idx, "down_proj", A_down, B_down, alpha=1.9)
            H_stu_2 = H_mid_stu + layer.mlp.down_proj(F.silu(layer.mlp.gate_proj(X_mlp_stu_all)) * layer.mlp.up_proj(X_mlp_stu_all))
            _, p_after_2, _ = voronoi_guarded_logit_targets(
                model, H_stu_2, cm_stu, pe_stu, l_idx + 1, stu_pos, target_tok_ids, seq_chunk=128, compute_dH=False
            )

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
                if p2_m > p1_m + 0.02:
                    # Linearly use higher headroom; reach ~0.985 with up to 2.25
                    aj = (4.2 - m0) / max(dm, 0.4)
                    alpha_vec[j] = max(1.0, min(2.25, aj))
                elif dm > 0.25:
                    alpha_vec[j] = max(0.85, min(1.85, (3.8 - m0) / dm))
            else:
                if dm > 0.25:
                    aj = (3.1 - m0) / dm
                    alpha_vec[j] = max(0.75, min(1.65, aj))

        B_down_cal = B_down * alpha_vec.unsqueeze(0)
        lora_mgr.set_layer_adapter(l_idx, "down_proj", A_down, B_down_cal, alpha=1.0)

        def forward_mlp_from_mid(h_mid, x_mlp, bs: int = 128):
            outs = []
            for s in range(0, h_mid.shape[0], bs):
                e = min(h_mid.shape[0], s + bs)
                xm = x_mlp[s:e]
                outs.append(h_mid[s:e] + layer.mlp.down_proj(F.silu(layer.mlp.gate_proj(xm)) * layer.mlp.up_proj(xm)))
            return torch.cat(outs, dim=0)

        H_stu = forward_mlp_from_mid(H_mid_stu, X_mlp_stu_all)
        H_neg = forward_mlp_from_mid(H_mid_neg, X_mlp_neg_all)
        H_cal = forward_mlp_from_mid(H_mid_cal, X_mlp_cal_all)

        layer_stats.append({
            "layer": l_idx,
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

    _, _, final_diag = voronoi_guarded_logit_targets(
        model, H_stu, cm_stu, pe_stu, len(model.model.layers), stu_pos, target_tok_ids, seq_chunk=128, compute_dH=False
    )
    return {"layer_stats": layer_stats, "final_train_diag": final_diag}


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cal_texts, kl_texts = load_smoltalk_splits(tokenizer, n_cal=128, n_kl_eval=64)
    batch42 = generate_fact_batch(42, tokenizer=tokenizer, target_doc_tokens=2048)

    cleanup_cuda()
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    static_vram_mb = measure_static_vram_mb(model)

    lora_mgr = LoRAManager(model, target_modules=TARGET_MODULES)
    lora_mgr.install()

    baselines = json.loads((LAB_DIR / "baselines_summary.json").read_text())
    base_a_ref = baselines["baseline_a_512"]

    with NoBackpropGuard(model, regime="no_grad", static_vram_mb=static_vram_mb) as guard:
        guard.assert_backbone_no_grad()
        diag = synthesize_mechA9_swiglu(model, tokenizer, lora_mgr, batch42, cal_texts, device)

    assert lora_mgr.max_active_rank() <= 16
    eval_res = evaluate_model(model, tokenizer, batch42, kl_texts, device, lora_manager=lora_mgr, compute_kl=True)
    gates = check_gates(eval_res, guard.metrics, baseline_a_ref=base_a_ref)
    status = "pass" if gates["ALL_GATES_PASS"] else "iterate"

    rec = append_experiment_record(
        25,
        "MechA9_VoronoiGuard_SwiGLU",
        eval_res,
        guard.metrics,
        status=status,
        seed=42,
        stage=1,
        rank=lora_mgr.max_active_rank(),
        extra={"gates": gates, "final_train_diag": diag["final_train_diag"], "layer_stats": diag["layer_stats"]},
    )
    print("\n[Mech A9 / Iter 25] Result:", json.dumps(rec, indent=2))
    print("[Mech A9 / Iter 25] Gates:", json.dumps(gates, indent=2))
    print("[Mech A9 / Iter 25] Final train diag:", diag["final_train_diag"])
    print_locality_breakdown(model, tokenizer, lora_mgr, batch42, device)
    print("[Mech A9 / Iter 25] Layer stats:")
    for ls in diag["layer_stats"]:
        print(" ", ls)


if __name__ == "__main__":
    main()
