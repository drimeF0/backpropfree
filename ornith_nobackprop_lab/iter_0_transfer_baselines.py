"""Iteration 0: Cross-Scale Transfer Baselines on Bio-Logic Benchmark (360M fp16 + 9B 8-bit).

Evaluates:
  1. Base_Unedited across all 3 seeds [42, 1337, 2026]
  2. InContext_Oracle across all 3 seeds [42, 1337, 2026]
  3. Existing archived blocks from blocks_library/ as-is on Seed 42:
     - Voronoi SwiGLU (Mech A12 on 360M / Mech A13 on 9B, pure 0-backprop)
     - Shared4Null ResSkip (Iter 12/14 on 360M / Iter 30 Stable on 9B, 1-layer detached)

Logs all records to experiments_transfer.jsonl and saves transcripts to iter_0_transfer_summary.json.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark import load_smoltalk_splits
from bio_logic_benchmark import SEEDS, build_bio_synthesis_probes, generate_bio_batch
from guards import LoRAManager, NoBackpropGuard, SynthesisMetrics, cleanup_cuda
from iter_29_mechA13 import solve_contrastive_neuron_proj_sep
from iter_30_shared4null_stable import PrecondBlockLoRA, compute_null_preconditioner
from probes import gather_positions, tokenize_with_target_positions
from transfer_eval import (
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
    step_token_mixer_unified,
)

LAB_DIR = Path(__file__).resolve().parent


def load_smoltalk_for_scale(tokenizer, scale: str, n_cal: int = 128, n_kl: int = 64) -> Tuple[List[str], List[str]]:
    if scale == "large":
        return load_smoltalk_splits(tokenizer, n_cal=n_cal, n_kl_eval=n_kl)
    cache_small = LAB_DIR / "smoltalk_cache_small.json"
    if cache_small.exists():
        payload = json.loads(cache_small.read_text())
        if len(payload.get("cal_texts", [])) >= n_cal and len(payload.get("kl_texts", [])) >= n_kl:
            return payload["cal_texts"][:n_cal], payload["kl_texts"][:n_kl]
    from datasets import load_dataset
    cal_ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="train", streaming=True)
    cal_texts = []
    for i, row in enumerate(cal_ds):
        if i >= 200:
            break
        cal_texts.append(apply_chat_safe(tokenizer, row["messages"], add_generation_prompt=False))
    kl_ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="test", streaming=True)
    kl_texts = []
    for i, row in enumerate(kl_ds):
        if i >= max(64, n_kl):
            break
        kl_texts.append(apply_chat_safe(tokenizer, row["messages"], add_generation_prompt=False))
    cache_small.write_text(json.dumps({"cal_texts": cal_texts, "kl_texts": kl_texts}))
    return cal_texts[:n_cal], kl_texts[:n_kl]


@torch.no_grad()
def _voronoi_targets_small(
    model,
    H_cur: torch.Tensor,
    cm_map: Dict,
    pe: Tuple[torch.Tensor, torch.Tensor],
    tp: torch.Tensor,
    start_layer: int,
    pos_lists: List[List[int]],
    target_tok_ids: torch.Tensor,
    target_prob_goal: float = 0.95,
    seq_chunk: int = 64,
    compute_dH: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    N = H_cur.shape[0]
    layers = get_layers(model)
    norm_layer = get_lm_norm(model)
    lm_head = get_lm_head(model)
    eps = getattr(norm_layer, "variance_epsilon", getattr(norm_layer, "eps", 1e-5))
    gamma = norm_layer.weight.float()
    W_lm = lm_head.weight.float()

    h_pts_list = []
    for s in range(0, N, seq_chunk):
        e = min(N, s + seq_chunk)
        hc = H_cur[s:e]
        cm_s = slice_cm_map(cm_map, s, e, N)
        pe_s = slice_pe(pe, s, e, N)
        tp_s = tp[s:e]
        for l_idx in range(start_layer, len(layers)):
            hc = step_layer_unified(model, l_idx, hc, cm_s, pe_s, tp_s)
        h_pts_list.append(gather_positions(hc, pos_lists[s:e]))

    hf = torch.cat(h_pts_list, dim=0).float()
    M, _ = hf.shape
    p_target_all = torch.zeros(M, device=hf.device, dtype=torch.float32)
    dH_goal = torch.zeros_like(hf)
    top1_hits = 0
    ov_before = 0
    ov_after = 0
    etas = (25.0, 50.0, 85.0, 125.0, 170.0, 220.0)
    tok_chunk = 260

    def grad_at_h(h_vec: torch.Tensor, tids: torch.Tensor):
        inv_rms = torch.rsqrt((h_vec * h_vec).mean(dim=-1, keepdim=True) + eps)
        h_hat = h_vec * inv_rms
        logits = (h_hat * gamma.unsqueeze(0)) @ W_lm.T
        probs = F.softmax(logits, dim=-1)
        ar = torch.arange(h_vec.shape[0], device=h_vec.device)
        pt = probs[ar, tids]
        ok = logits.argmax(dim=-1) == tids
        g_z = probs
        g_z[ar, tids] -= 1.0
        g_hhat = (g_z @ W_lm) * gamma.unsqueeze(0)
        g_h = inv_rms * (g_hhat - (g_hhat * h_hat).mean(dim=-1, keepdim=True) * h_hat)
        return pt, ok, g_h

    def line_search_eta(hc: torch.Tensor, u: torch.Tensor, tids: torch.Tensor, need: torch.Tensor):
        best_eta = torch.zeros((hc.shape[0], 1), device=hc.device, dtype=torch.float32)
        solved = ~need.bool()
        ar = torch.arange(hc.shape[0], device=hc.device)
        for eta in etas:
            cand_h = hc + eta * u
            cand_inv = torch.rsqrt((cand_h * cand_h).mean(dim=-1, keepdim=True) + eps)
            cand_logits = ((cand_h * cand_inv) * gamma.unsqueeze(0)) @ W_lm.T
            cand_probs = F.softmax(cand_logits, dim=-1)
            cand_ok = (cand_probs[ar, tids] >= target_prob_goal) & (cand_logits.argmax(dim=-1) == tids)
            newly = (~solved) & cand_ok
            best_eta[newly] = eta
            solved = solved | newly
        best_eta[~solved] = etas[-1]
        return (best_eta * need.float().unsqueeze(1)) * u

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
            logits_hi = ((cand_hi * inv_hi) * gamma.unsqueeze(0)) @ W_lm.T
            bad = need & (logits_hi.argmax(dim=-1) != tids)
            if r_idx == 0:
                ov_before += int(bad.sum().item())
            if not bad.any():
                break
            bad_idx = torch.nonzero(bad, as_tuple=False).squeeze(-1)
            _, _, g_bad = grad_at_h(cand_hi[bad_idx], tids[bad_idx])
            u_new = u_dir[bad_idx] + rep_w * ((-g_bad) / g_bad.norm(dim=-1, keepdim=True).clamp(min=1e-8))
            u_new = u_new / u_new.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            u_dir[bad_idx] = u_new
            dh[bad_idx] = line_search_eta(hc[bad_idx], u_new, tids[bad_idx], torch.ones(bad_idx.numel(), device=hc.device, dtype=torch.bool))
        cand_hi = hc + 1.45 * dh
        inv_hi = torch.rsqrt((cand_hi * cand_hi).mean(dim=-1, keepdim=True) + eps)
        logits_hi = ((cand_hi * inv_hi) * gamma.unsqueeze(0)) @ W_lm.T
        ov_after += int((need & (logits_hi.argmax(dim=-1) != tids)).sum().item())
        dH_goal[ts:te] = dh

    return dH_goal, p_target_all, {"mean_p": round(float(p_target_all.mean().item()), 4), "top1_acc": round(float(top1_hits / max(M, 1)), 4), "ov_fix": f"{ov_before}->{ov_after}"}


@torch.no_grad()
def run_small_archived_mechA12(
    model,
    tokenizer,
    lora_mgr: LoRAManager,
    pos_items: List[Dict],
    neg_items: List[Dict],
    cal_texts: List[str],
    device: str,
) -> Dict[str, object]:
    """Archived SmolLM2 Mech A12 (research_orig hyperparameters) on Bio synthesis probes."""
    prompts = [x["prompt"] for x in pos_items]
    targets = [x["target"] for x in pos_items]
    stu_ids, stu_mask, stu_pos = tokenize_with_target_positions(tokenizer, prompts, targets, device)

    target_tok_list, token_fact_ids, token_step_ids = [], [], []
    unique_pos0, unique_eos, unique_mid = [], [], []
    seen = set()
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
            if pair not in seen:
                seen.add(pair)
                if k_step == 0:
                    unique_pos0.append(pair)
                elif k_step == eos_step:
                    unique_eos.append(pair)
                else:
                    unique_mid.append(pair)

    target_tok_ids = torch.tensor(target_tok_list, dtype=torch.long, device=device)
    token_fact_ids_t = torch.tensor(token_fact_ids, dtype=torch.long, device=device)
    token_step_ids_t = torch.tensor(token_step_ids, dtype=torch.long, device=device)

    total_pairs = len(seen)
    n_layers = len(get_layers(model))
    needed_layers = max(8, math.ceil(total_pairs / 16))
    cand_layers = list(range(n_layers - needed_layers, n_layers))
    layer_shards = {l: [] for l in cand_layers}
    for idx, p in enumerate(unique_pos0):
        layer_shards[[26, 27, 28, 29][idx % 4]].append(p)
    for idx, p in enumerate(unique_eos):
        layer_shards[[31, 30][idx % 2]].append(p)
    for p in unique_mid:
        for l in reversed(cand_layers):
            if len(layer_shards[l]) < 16:
                layer_shards[l].append(p)
                break

    active_layers = [l for l in cand_layers if layer_shards[l]]
    first_layer = active_layers[0]

    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    neg_enc = tokenizer([x["prompt"] for x in neg_items], return_tensors="pt", padding=True).to(device)
    cal_enc = tokenizer(cal_texts[:128], return_tensors="pt", padding=True, truncation=True, max_length=96).to(device)
    tokenizer.padding_side = prev_side
    neg_pos = [[L - 1] for L in neg_enc.attention_mask.sum(dim=1).tolist()]

    def init_pref(ids, mask, bs: int = 128):
        emb, cm, pe, tp = prepare_layer_inputs_unified(model, ids, mask)
        chunks = []
        for s in range(0, ids.shape[0], bs):
            e = min(ids.shape[0], s + bs)
            h = emb[s:e]
            cm_s = slice_cm_map(cm, s, e, ids.shape[0])
            pe_s = slice_pe(pe, s, e, ids.shape[0])
            tp_s = tp[s:e]
            for l in range(first_layer):
                h = step_layer_unified(model, l, h, cm_s, pe_s, tp_s)
            chunks.append(h.detach())
        return torch.cat(chunks, dim=0), cm, pe, tp

    lora_mgr.clear_all()
    H_stu, cm_stu, pe_stu, tp_stu = init_pref(stu_ids, stu_mask)
    H_neg, cm_neg, pe_neg, tp_neg = init_pref(neg_enc.input_ids, neg_enc.attention_mask)
    H_cal, cm_cal, pe_cal, tp_cal = init_pref(cal_enc.input_ids, cal_enc.attention_mask)
    cal_mask_bool = cal_enc.attention_mask.bool()

    layers = get_layers(model)
    for l_idx in range(first_layer, n_layers):
        layer = layers[l_idx]
        shard_pairs = layer_shards[l_idx]
        if not shard_pairs:
            H_stu = step_layer_unified(model, l_idx, H_stu, cm_stu, pe_stu, tp_stu)
            H_neg = step_layer_unified(model, l_idx, H_neg, cm_neg, pe_neg, tp_neg)
            H_cal = step_layer_unified(model, l_idx, H_cal, cm_cal, pe_cal, tp_cal)
            continue

        H_mid_stu, X_mlp_stu_all = step_token_mixer_unified(model, l_idx, H_stu, cm_stu, pe_stu, tp_stu)
        H_mid_neg, X_mlp_neg_all = step_token_mixer_unified(model, l_idx, H_neg, cm_neg, pe_neg, tp_neg)
        H_mid_cal, X_mlp_cal_all = step_token_mixer_unified(model, l_idx, H_cal, cm_cal, pe_cal, tp_cal)

        X_mlp_stu = gather_positions(X_mlp_stu_all, stu_pos)
        X_mlp_neg = gather_positions(X_mlp_neg_all, neg_pos)
        X_mlp_cal = X_mlp_cal_all[cal_mask_bool]

        dH_goal, p_before, _ = _voronoi_targets_small(
            model, H_stu, cm_stu, pe_stu, tp_stu, l_idx, stu_pos, target_tok_ids, target_prob_goal=0.95, seq_chunk=64, compute_dH=True
        )
        r_neurons = len(shard_pairs)
        g_cal_base = F.linear(X_mlp_cal, layer.mlp.gate_proj.base.weight).float()
        dormancy = (F.silu(g_cal_base) ** 2).mean(dim=0) + 0.05 * g_cal_base.clamp(min=0.0).mean(dim=0)
        J_l = torch.topk(dormancy, k=r_neurons, largest=False).indices

        shard_mask = torch.zeros_like(token_fact_ids_t, dtype=torch.bool)
        cluster_ids_full = torch.full_like(token_fact_ids_t, -1)
        for c_i, (f_id, k_st) in enumerate(shard_pairs):
            pm = (token_fact_ids_t == f_id) & (token_step_ids_t == k_st)
            shard_mask |= pm
            cluster_ids_full[pm] = c_i

        other_mask = ~shard_mask
        X_shard = X_mlp_stu[shard_mask]
        X_other = X_mlp_stu[other_mask]
        c_ids = cluster_ids_full[shard_mask]

        W_g = layer.mlp.gate_proj.base.weight[J_l].float()
        W_u = layer.mlp.up_proj.base.weight[J_l].float()

        g_goal = torch.full((X_shard.shape[0], r_neurons), -4.5, device=device)
        g_goal[torch.arange(X_shard.shape[0], device=device), c_ids] = 4.8
        dG_pos = g_goal - X_shard.float() @ W_g.T
        dG_other = -4.5 - X_other.float() @ W_g.T
        dG_neg = -4.5 - X_mlp_neg.float() @ W_g.T
        g_cal_J = g_cal_base[:, J_l]
        dG_cal = torch.minimum(g_cal_J, torch.full_like(g_cal_J, -3.5)) - g_cal_J

        A_gate = solve_contrastive_neuron_proj_sep(
            X_shard, dG_pos, X_other, dG_other, X_mlp_neg, dG_neg, X_mlp_cal, dG_cal,
            w_pos=2.2, w_other=1.4, w_neg=1.6, w_cal=1.1, lambda_fro=0.012, max_row_norm=9.5,
        )
        B_sel = torch.zeros((layer.mlp.gate_proj.out_features, r_neurons), device=device, dtype=torch.float32)
        B_sel[J_l, torch.arange(r_neurons, device=device)] = 1.0
        lora_mgr.set_layer_adapter(l_idx, "gate_proj", A_gate, B_sel, alpha=1.0)

        u_goal = torch.zeros_like(g_goal)
        u_goal[torch.arange(X_shard.shape[0], device=device), c_ids] = 3.2
        dU_pos = u_goal - X_shard.float() @ W_u.T
        dU_other = -(X_other.float() @ W_u.T)
        dU_neg = -(X_mlp_neg.float() @ W_u.T)
        dU_cal = -(X_mlp_cal.float() @ W_u.T)
        A_up = solve_contrastive_neuron_proj_sep(
            X_shard, dU_pos, X_other, dU_other, X_mlp_neg, dU_neg, X_mlp_cal, dU_cal,
            w_pos=2.2, w_other=1.4, w_neg=1.6, w_cal=1.1, lambda_fro=0.012, max_row_norm=9.5,
        )
        lora_mgr.set_layer_adapter(l_idx, "up_proj", A_up, B_sel, alpha=1.0)

        Z_all = (F.silu(layer.mlp.gate_proj(X_mlp_stu)) * layer.mlp.up_proj(X_mlp_stu))[:, J_l].float()
        Z_shard, Z_other = Z_all[shard_mask], Z_all[other_mask]
        Z_neg = (F.silu(layer.mlp.gate_proj(X_mlp_neg)) * layer.mlp.up_proj(X_mlp_neg))[:, J_l].float()
        Z_cal = (F.silu(layer.mlp.gate_proj(X_mlp_cal)) * layer.mlp.up_proj(X_mlp_cal))[:, J_l].float()

        gram_r = (Z_shard.T @ Z_shard) + 1.5 * (Z_other.T @ Z_other) + 3.5 * (Z_neg.T @ Z_neg) + 3.0 * (Z_cal.T @ Z_cal) + 0.4 * torch.eye(r_neurons, device=device)
        B_down = torch.linalg.solve(gram_r, Z_shard.T @ dH_goal[shard_mask].float()).T
        A_down = B_sel.T.contiguous()

        lora_mgr.set_layer_adapter(l_idx, "down_proj", A_down, B_down, alpha=1.0)
        H_stu_1 = H_mid_stu + layer.mlp(X_mlp_stu_all)
        _, p_after_1, _ = _voronoi_targets_small(
            model, H_stu_1, cm_stu, pe_stu, tp_stu, l_idx + 1, stu_pos, target_tok_ids, seq_chunk=64, compute_dH=False
        )
        p0_s = p_before[shard_mask].clamp(1e-4, 1.0 - 1e-4)
        p1_s = p_after_1[shard_mask].clamp(1e-4, 1.0 - 1e-4)
        ell0, ell1 = torch.log(p0_s / (1.0 - p0_s)), torch.log(p1_s / (1.0 - p1_s))
        alpha_vec = torch.ones(r_neurons, device=device, dtype=torch.float32)
        for j, (_, k_st) in enumerate(shard_pairs):
            jm = c_ids == j
            m0, m1 = float(ell0[jm].mean().item()), float(ell1[jm].mean().item())
            dm = m1 - m0
            ell_goal = 4.2 if k_st == 0 else 3.2
            if dm > 0.25:
                alpha_vec[j] = max(0.85, min(2.2, (ell_goal - m0) / dm))
        B_down_cal = B_down * alpha_vec.unsqueeze(0)
        lora_mgr.set_layer_adapter(l_idx, "down_proj", A_down, B_down_cal, alpha=1.0)

        H_stu = H_mid_stu + layer.mlp(X_mlp_stu_all)
        H_neg = H_mid_neg + layer.mlp(X_mlp_neg_all)
        H_cal = H_mid_cal + layer.mlp(X_mlp_cal_all)

    _, _, final_diag = _voronoi_targets_small(
        model, H_stu, cm_stu, pe_stu, tp_stu, n_layers, stu_pos, target_tok_ids, seq_chunk=64, compute_dH=False
    )
    return {"total_pairs": total_pairs, "final_train_diag": final_diag}


class SmallPrecondManager:
    """Archived SmolLM2 Shared4Null manager (lora_alpha=32.0 -> scaling=2.0, L26..L31)."""

    TARGET_MAP = LoRAManager.TARGET_MAP
    SPACE_FOR_MOD = {
        "q_proj": "attn_in", "k_proj": "attn_in", "v_proj": "attn_in",
        "o_proj": "o_proj", "gate_proj": "mlp_in", "up_proj": "mlp_in", "down_proj": "down_proj",
    }
    REP_MOD_FOR_SPACE = {"attn_in": "q_proj", "o_proj": "o_proj", "mlp_in": "gate_proj", "down_proj": "down_proj"}

    def __init__(self, model: nn.Module, target_layers: List[int], r: int = 16, lora_alpha: float = 32.0) -> None:
        self.model = model
        self.target_layers = list(target_layers)
        self.target_modules = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        self.r = r
        self.wrapped: Dict[Tuple[int, str], PrecondBlockLoRA] = {}
        for p in model.parameters():
            p.requires_grad_(False)
        layers = get_layers(model)
        for l_idx in self.target_layers:
            layer = layers[l_idx]
            for mod_name in self.target_modules:
                pa, la = self.TARGET_MAP[mod_name]
                parent = getattr(layer, pa)
                orig = getattr(parent, la)
                w = PrecondBlockLoRA(orig, r=r, lora_alpha=lora_alpha)
                setattr(parent, la, w)
                self.wrapped[(l_idx, mod_name)] = w

    def get_layer_params(self, layer_idx: int) -> List[nn.Parameter]:
        return [p for (li, _), w in self.wrapped.items() if li == layer_idx for p in (w.lora_A, w.lora_B)]

    def activate_for_layer(self, active_layer_idx: int) -> None:
        for (l_idx, _), w in self.wrapped.items():
            w.active = l_idx <= active_layer_idx
            req = l_idx == active_layer_idx
            w.lora_A.requires_grad_(req)
            w.lora_B.requires_grad_(req)

    def freeze_all(self) -> None:
        for w in self.wrapped.values():
            w.lora_A.requires_grad_(False)
            w.lora_B.requires_grad_(False)

    def merge_and_freeze_layer(self, layer_idx: int) -> None:
        for (li, _), w in self.wrapped.items():
            if li == layer_idx:
                w.merge_precond_and_freeze()

    def uninstall(self) -> None:
        layers = get_layers(self.model)
        for (l_idx, mod_name), w in self.wrapped.items():
            pa, la = self.TARGET_MAP[mod_name]
            parent = getattr(layers[l_idx], pa)
            setattr(parent, la, w.base)
        self.wrapped.clear()


def run_small_archived_shared4null(
    model,
    tokenizer,
    cal_texts: List[str],
    pos_items: List[Dict],
    neg_items: List[Dict],
    device: str,
) -> Tuple[SmallPrecondManager, Dict[str, object]]:
    """Archived SmolLM2 Shared4Null (L26..L31, lr=4.5e-3, alpha_null=6.5, lora_alpha=32)."""
    target_layers = list(range(26, 32))
    steps_per_layer = (10, 10, 11, 11, 12, 14)
    lr = 4.5e-3
    alpha_null = 6.5
    lambda_cal_kl = 5.5
    lambda_neg_kl = 8.0
    lambda_antibleed = 1.5

    c0_small_path = LAB_DIR / "c0_cache_small_L16_31.pt"
    layers = get_layers(model)
    if c0_small_path.exists():
        covs = torch.load(c0_small_path, map_location="cpu", weights_only=True)["covs"]
    else:
        covs_acc: Dict[Tuple[int, str], torch.Tensor] = {}
        hooks = []
        for l_idx in range(16, len(layers)):
            for m_name, mod in [("o_proj", layers[l_idx].self_attn.o_proj), ("down_proj", layers[l_idx].mlp.down_proj)]:
                covs_acc[(l_idx, m_name)] = torch.zeros((mod.in_features, mod.in_features), dtype=torch.float32)

                def _make_hk(k):
                    def hk(m, inp, out):
                        x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                        covs_acc[k] += (x.T @ x).cpu()
                    return hk
                hooks.append(mod.register_forward_hook(_make_hk((l_idx, m_name))))
        prev_side = tokenizer.padding_side
        tokenizer.padding_side = "right"
        tok_cnt = 0
        try:
            with torch.no_grad():
                for s in range(0, len(cal_texts), 16):
                    enc = tokenizer(cal_texts[s : s + 16], return_tensors="pt", padding=True, truncation=True, max_length=192).to(device)
                    tok_cnt += int(enc.attention_mask.sum().item())
                    _ = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)
        finally:
            tokenizer.padding_side = prev_side
            for h in hooks:
                h.remove()
        covs = {k: 0.5 * (v / max(tok_cnt, 1) + (v / max(tok_cnt, 1)).T) for k, v in covs_acc.items()}
        torch.save({"covs": covs}, c0_small_path)

    mgr = SmallPrecondManager(model, target_layers=target_layers, r=16, lora_alpha=32.0)

    prompts = [x["prompt"] for x in pos_items]
    targets = [x["target"] for x in pos_items]
    stu_ids, stu_mask, stu_pos = tokenize_with_target_positions(tokenizer, prompts, targets, device)
    target_tok_list, pos0_w, injected_first = [], [], set()
    for tgt in targets:
        t_ids = tokenizer.encode(tgt, add_special_tokens=False)
        eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        full_t = t_ids + eos
        injected_first.add(t_ids[0])
        target_tok_list.extend(full_t)
        pos0_w.extend([2.0] + [1.0] * (len(full_t) - 1))
    target_tok_ids = torch.tensor(target_tok_list, dtype=torch.long, device=device)
    tok_weights = torch.tensor(pos0_w, dtype=torch.float32, device=device)
    tok_weights = tok_weights / tok_weights.mean()
    bleed_tok_ids = torch.tensor(sorted(injected_first), dtype=torch.long, device=device)

    lm_norm = get_lm_norm(model)
    lm_head = get_lm_head(model)
    eos_ids = get_eos_token_ids(tokenizer)
    prev_side = tokenizer.padding_side
    n_layers = len(layers)

    with torch.no_grad():
        mgr.freeze_all()
        tokenizer.padding_side = "left"
        neg_gen_enc = tokenizer([x["prompt"] for x in neg_items], return_tensors="pt", padding=True).to(device)
        in_len_neg = neg_gen_enc.input_ids.shape[1]
        neg_gen = model.generate(
            input_ids=neg_gen_enc.input_ids, attention_mask=neg_gen_enc.attention_mask,
            max_new_tokens=2, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=eos_ids, use_cache=True,
        )
        neg_seqs, neg_pos, neg_w = [], [], []
        for idx_n in range(len(neg_items)):
            gen_part = neg_gen[idx_n, in_len_neg:].tolist()
            clean_gen = []
            for tk in gen_part:
                clean_gen.append(tk)
                if tk in eos_ids:
                    break
            p_ids = tokenizer.encode(neg_items[idx_n]["prompt"], add_special_tokens=False)
            neg_seqs.append(p_ids + clean_gen)
            n_pred = max(1, len(clean_gen))
            neg_pos.append(list(range(len(p_ids) - 1, len(p_ids) - 1 + n_pred)))
            neg_w.extend([3.0] + [1.0] * (n_pred - 1))
        neg_tok_weights = torch.tensor(neg_w, dtype=torch.float32, device=device)
        neg_tok_weights = neg_tok_weights / neg_tok_weights.mean()

        max_nl = max(len(s) for s in neg_seqs)
        neg_ids = torch.full((len(neg_seqs), max_nl), tokenizer.pad_token_id, dtype=torch.long, device=device)
        neg_mask = torch.zeros((len(neg_seqs), max_nl), dtype=torch.long, device=device)
        for i, s in enumerate(neg_seqs):
            neg_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            neg_mask[i, : len(s)] = 1

        tokenizer.padding_side = "right"
        cal_enc = tokenizer(cal_texts[:64], return_tensors="pt", padding=True, truncation=True, max_length=72).to(device)
        tokenizer.padding_side = prev_side
        cal_lengths = cal_enc.attention_mask.sum(dim=1).tolist()
        cal_kl_pos = [[max(0, min(L - 1, int(k * L / 4))) for k in range(1, 4)] for L in cal_lengths]

        def get_base_probs(ids, mask, pos_lists):
            h, cm_m, pe_t, tp_t = prepare_layer_inputs_unified(model, ids, mask)
            for l_i in range(n_layers):
                h = step_layer_unified(model, l_i, h, cm_m, pe_t, tp_t)
            pts = gather_positions(h, pos_lists)
            lp = F.log_softmax(lm_head(lm_norm(pts)).float(), dim=-1)
            return lp.exp(), lp[:, bleed_tok_ids]

        p_base_cal, _ = get_base_probs(cal_enc.input_ids, cal_enc.attention_mask, cal_kl_pos)
        p_base_neg, logp_base_neg_bleed = get_base_probs(neg_ids, neg_mask, neg_pos)

        def init_pref(ids, mask):
            h, cm_m, pe_t, tp_t = prepare_layer_inputs_unified(model, ids, mask)
            for l_i in range(target_layers[0]):
                h = step_layer_unified(model, l_i, h, cm_m, pe_t, tp_t)
            return h.detach(), cm_m, pe_t, tp_t

        H_stu, cm_stu, pe_stu, tp_stu = init_pref(stu_ids, stu_mask)
        H_neg, cm_neg, pe_neg, tp_neg = init_pref(neg_ids, neg_mask)
        H_cal, cm_cal, pe_cal, tp_cal = init_pref(cal_enc.input_ids, cal_enc.attention_mask)
        cal_mask_bool = cal_enc.attention_mask.bool()

    layer_logs = []
    for idx_sched, l_idx in enumerate(target_layers):
        layer = layers[l_idx]
        n_steps = steps_per_layer[idx_sched]
        H_stu_in, H_neg_in, H_cal_in = H_stu.detach(), H_neg.detach(), H_cal.detach()

        with torch.no_grad():
            mgr.freeze_all()
            rep_mods = {
                "attn_in": layer.self_attn.q_proj,
                "o_proj": layer.self_attn.o_proj,
                "mlp_in": layer.mlp.gate_proj,
                "down_proj": layer.mlp.down_proj,
            }
            cap_cal = {sp: [] for sp in rep_mods}
            cap_neg = {sp: [] for sp in rep_mods}
            cap_stu = {sp: [] for sp in rep_mods}

            def reg_hk(store, pos_lists=None, bs_mask=None):
                hks = []
                for sp, mod in rep_mods.items():
                    if pos_lists is not None:
                        hks.append(mod.register_forward_pre_hook(lambda m, inp, s=sp, pl=pos_lists: store[s].append(gather_positions(inp[0].detach(), pl))))
                    elif bs_mask is not None:
                        hks.append(mod.register_forward_pre_hook(lambda m, inp, s=sp, bm=bs_mask: store[s].append(inp[0].detach()[bm])))
                return hks

            for store, h_in, cm_m, pe_t, tp_t, kw in [
                (cap_cal, H_cal_in, cm_cal, pe_cal, tp_cal, {"bs_mask": cal_mask_bool}),
                (cap_neg, H_neg_in, cm_neg, pe_neg, tp_neg, {"pos_lists": neg_pos}),
                (cap_stu, H_stu_in, cm_stu, pe_stu, tp_stu, {"pos_lists": stu_pos}),
            ]:
                hks = reg_hk(store, **kw)
                _ = step_layer_unified(model, l_idx, h_in, cm_m, pe_t, tp_t)
                for h in hks:
                    h.remove()

            p_null_by_sp, xs_by_sp = {}, {}
            for sp in rep_mods:
                xc, xn, xs = torch.cat(cap_cal[sp], dim=0), torch.cat(cap_neg[sp], dim=0), torch.cat(cap_stu[sp], dim=0)
                c0_mat = covs.get((l_idx, sp)) if sp in ("o_proj", "down_proj") else None
                p_null_by_sp[sp] = compute_null_preconditioner(xc, xn, c0_mat, alpha_null=alpha_null)
                xs_by_sp[sp] = xs

            for m_name in mgr.target_modules:
                sp = mgr.SPACE_FOR_MOD[m_name]
                mgr.wrapped[(l_idx, m_name)].set_precond_and_init(p_null_by_sp[sp], xs_by_sp[sp])

            def compute_skip(h_in, cm_m, pe_t, tp_t, pos_lists):
                hl = step_layer_unified(model, l_idx, h_in, cm_m, pe_t, tp_t)
                hc = hl
                for j in range(l_idx + 1, n_layers):
                    hc = step_layer_unified(model, j, hc, cm_m, pe_t, tp_t)
                return gather_positions(hc - hl, pos_lists).detach()

            skip_stu = compute_skip(H_stu_in, cm_stu, pe_stu, tp_stu, stu_pos)
            skip_neg = compute_skip(H_neg_in, cm_neg, pe_neg, tp_neg, neg_pos)
            skip_cal = compute_skip(H_cal_in, cm_cal, pe_cal, tp_cal, cal_kl_pos)

        mgr.activate_for_layer(l_idx)
        params = mgr.get_layer_params(l_idx)
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=5e-4)
        n_cal = H_cal_in.shape[0]
        last_ce = last_ckl = last_nkl = 0.0
        for step in range(n_steps):
            opt.zero_grad(set_to_none=True)
            with torch.enable_grad():
                out_s = step_layer_unified(model, l_idx, H_stu_in, cm_stu, pe_stu, tp_stu)
                h_fs = gather_positions(out_s, stu_pos) + skip_stu
                ce_vec = F.cross_entropy(lm_head(lm_norm(h_fs)).float(), target_tok_ids, reduction="none")
                l_ce = (ce_vec * tok_weights).mean()

                cs = (step % 2) * (n_cal // 2)
                ce_c = min(n_cal, cs + (n_cal // 2))
                c_start = sum(len(p) for p in cal_kl_pos[:cs])
                c_cnt = sum(len(p) for p in cal_kl_pos[cs:ce_c])
                out_c = step_layer_unified(model, l_idx, H_cal_in[cs:ce_c], slice_cm_map(cm_cal, cs, ce_c, n_cal), slice_pe(pe_cal, cs, ce_c, n_cal), tp_cal[cs:ce_c])
                pts_c = gather_positions(out_c, cal_kl_pos[cs:ce_c]) + skip_cal[c_start : c_start + c_cnt]
                l_ckl = F.kl_div(F.log_softmax(lm_head(lm_norm(pts_c)).float(), dim=-1), p_base_cal[c_start : c_start + c_cnt], reduction="batchmean")

                out_n = step_layer_unified(model, l_idx, H_neg_in, cm_neg, pe_neg, tp_neg)
                pts_n = gather_positions(out_n, neg_pos) + skip_neg
                log_p_n = F.log_softmax(lm_head(lm_norm(pts_n)).float(), dim=-1)
                tok_nkl = (p_base_neg * (p_base_neg.clamp(min=1e-12).log() - log_p_n)).sum(dim=-1)
                l_nkl = (tok_nkl * neg_tok_weights).mean()
                l_bleed = (F.relu(log_p_n[:, bleed_tok_ids] - logp_base_neg_bleed - 0.05) ** 2).mean()

                loss = l_ce + lambda_cal_kl * l_ckl + lambda_neg_kl * l_nkl + lambda_antibleed * l_bleed
                loss.backward()
                last_ce, last_ckl, last_nkl = float(l_ce.item()), float(l_ckl.item()), float(l_nkl.item())
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

        mgr.merge_and_freeze_layer(l_idx)
        with torch.no_grad():
            H_stu = step_layer_unified(model, l_idx, H_stu_in, cm_stu, pe_stu, tp_stu).detach()
            H_neg = step_layer_unified(model, l_idx, H_neg_in, cm_neg, pe_neg, tp_neg).detach()
            H_cal = step_layer_unified(model, l_idx, H_cal_in, cm_cal, pe_cal, tp_cal).detach()
        layer_logs.append({"layer": l_idx, "ce": round(last_ce, 4), "cal_kl": round(last_ckl, 5), "neg_kl": round(last_nkl, 5)})

    return mgr, {"layer_logs": layer_logs}


def run_scale(scale: str, device: str, eval_thinking_s42: bool = False) -> Dict[str, object]:
    model, tokenizer, spec = load_model_by_scale(scale, device=device)
    cal_texts, kl_texts = load_smoltalk_for_scale(tokenizer, scale, n_cal=128, n_kl=64)
    out_summary: Dict[str, object] = {"scale": scale, "model_id": spec.model_id, "static_vram_mb": round(spec.static_vram_mb, 2), "seeds": {}}

    base_logic_by_seed: Dict[int, float] = {}
    oracle_logic_by_seed: Dict[int, float] = {}

    for seed in SEEDS:
        eval_thinking = eval_thinking_s42 and (scale == "large") and (seed == 42)
        batch = generate_bio_batch(seed, tokenizer=tokenizer, target_doc_tokens=2048)
        base_ref = get_bio_base_reference(model, tokenizer, spec, batch, device)

        # 1. Base_Unedited
        base_eval = evaluate_bio_transfer(
            model, tokenizer, spec, batch, kl_texts, device,
            prefix_bio=False, compute_kl=False, eval_thinking_diag=eval_thinking, eval_self_recalled=True,
            base_ref_override=base_ref,
        )
        m_zero = SynthesisMetrics(static_vram_mb=spec.static_vram_mb, peak_vram_mb=spec.static_vram_mb, uses_full_backprop=False, regime="none")
        base_rec = append_transfer_record(0, "Base_Unedited", scale, spec.model_id, base_eval, m_zero, status="baseline", seed=seed, stage=1, rank=0)
        base_logic_by_seed[seed] = base_eval["logic_1turn_acc"]
        print(f"[{scale} seed={seed}] Base_Unedited: direct={base_eval['direct_recall_acc']} para={base_eval['paraphrase_acc']} logic={base_eval['logic_1turn_acc']} think={base_eval['logic_thinking_acc']} real_anc={base_eval['real_anchors_acc']} bleed={base_eval['synth_bleed']} refusal={base_eval['direct_refusal_rate']}", flush=True)

        # 2. InContext_Oracle
        ora_eval = evaluate_bio_transfer(
            model, tokenizer, spec, batch, kl_texts, device,
            prefix_bio=True, compute_kl=False, eval_thinking_diag=eval_thinking, eval_self_recalled=False,
            base_ref_override=base_ref,
        )
        ora_rec = append_transfer_record(0, "InContext_Oracle", scale, spec.model_id, ora_eval, m_zero, status="oracle_ceiling", seed=seed, stage=1, rank=0)
        oracle_logic_by_seed[seed] = ora_eval["logic_1turn_acc"]
        print(f"[{scale} seed={seed}] InContext_Oracle: direct={ora_eval['direct_recall_acc']} para={ora_eval['paraphrase_acc']} logic={ora_eval['logic_1turn_acc']} think={ora_eval['logic_thinking_acc']} real_anc={ora_eval['real_anchors_acc']} bleed={ora_eval['synth_bleed']}", flush=True)

        out_summary["seeds"][str(seed)] = {
            "Base_Unedited": {**base_rec, "details": base_eval["details"]},
            "InContext_Oracle": {**ora_rec, "details": ora_eval["details"]},
        }

    # 3. Archived blocks_library/ baselines on Seed 42
    batch42 = generate_bio_batch(42, tokenizer=tokenizer, target_doc_tokens=2048)
    base_ref42 = get_bio_base_reference(model, tokenizer, spec, batch42, device)
    b_log42 = base_logic_by_seed[42]
    o_log42 = oracle_logic_by_seed[42]

    if scale == "small":
        # Archived Mech A12 (0-backprop)
        lora_mgr = LoRAManager(model, target_modules=("gate_proj", "up_proj", "down_proj"))
        lora_mgr.install()
        pos_a, neg_a = build_bio_synthesis_probes(batch42.stage1_clauses, tokenizer, n_neg=240, max_frames_per_pair=7)
        with NoBackpropGuard(model, regime="no_grad", static_vram_mb=spec.static_vram_mb) as g_a:
            g_a.assert_backbone_no_grad()
            diag_a = run_small_archived_mechA12(model, tokenizer, lora_mgr, pos_a, neg_a, cal_texts, device)
        ev_a = evaluate_bio_transfer(model, tokenizer, spec, batch42, kl_texts, device, lora_manager=lora_mgr, compute_kl=True, eval_thinking_diag=False, eval_self_recalled=True, base_ref_override=base_ref42)
        gates_a = check_transfer_gates(ev_a, g_a.metrics, scale, lora_mgr.max_active_rank(), b_log42, o_log42)
        rec_a = append_transfer_record(0, "Archived_MechA12_VoronoiSwiGLU", scale, spec.model_id, ev_a, g_a.metrics, status="baseline_archived", seed=42, stage=1, rank=lora_mgr.max_active_rank(), gates=gates_a, extra={"train_diag": diag_a["final_train_diag"]})
        print(f"[small seed=42] Archived_MechA12: direct={ev_a['direct_recall_acc']} para={ev_a['paraphrase_acc']} logic={ev_a['logic_1turn_acc']} sr_logic={ev_a['logic_self_recalled_acc']} bleed={ev_a['synth_bleed']} kl={ev_a['kl_div']} ret={ev_a['retention_acc']} time={g_a.metrics.wall_time_sec:.2f}s", flush=True)
        out_summary["seeds"]["42"]["Archived_MechA12_VoronoiSwiGLU"] = {**rec_a, "details": ev_a["details"]}
        lora_mgr.clear_all()
        lora_mgr.uninstall()
        cleanup_cuda()

        # Archived Shared4Null (1-layer detached)
        pos_s, neg_s = build_bio_synthesis_probes(batch42.stage1_clauses, tokenizer, n_neg=64, max_frames_per_pair=4)
        with NoBackpropGuard(model, regime="single_layer_isolated", static_vram_mb=spec.static_vram_mb) as g_s:
            g_s.assert_backbone_no_grad()
            mgr_s, diag_s = run_small_archived_shared4null(model, tokenizer, cal_texts, pos_s, neg_s, device)
        ev_s = evaluate_bio_transfer(model, tokenizer, spec, batch42, kl_texts, device, lora_manager=mgr_s, compute_kl=True, eval_thinking_diag=False, eval_self_recalled=True, base_ref_override=base_ref42)
        gates_s = check_transfer_gates(ev_s, g_s.metrics, scale, 16, b_log42, o_log42)
        rec_s = append_transfer_record(0, "Archived_Shared4Null_ResSkip", scale, spec.model_id, ev_s, g_s.metrics, status="baseline_archived", seed=42, stage=1, rank=16, gates=gates_s, extra={"layer_logs": diag_s["layer_logs"]})
        print(f"[small seed=42] Archived_Shared4Null: direct={ev_s['direct_recall_acc']} para={ev_s['paraphrase_acc']} logic={ev_s['logic_1turn_acc']} sr_logic={ev_s['logic_self_recalled_acc']} bleed={ev_s['synth_bleed']} kl={ev_s['kl_div']} ret={ev_s['retention_acc']} time={g_s.metrics.wall_time_sec:.2f}s", flush=True)
        out_summary["seeds"]["42"]["Archived_Shared4Null_ResSkip"] = {**rec_s, "details": ev_s["details"]}
        mgr_s.uninstall()
        cleanup_cuda()

    else:
        from blocks_library.voronoi_swiglu_block import TARGET_MODULES as M_A_MODS, synthesize_mechA12_swiglu
        from blocks_library.shared4null_resskip_block import LORA_RANK, TARGET_LAYERS, PrecondBlockManager, run_shared4_l26_31
        from nullspace_cache import compute_or_load_c0

        # Archived Mech A13 (0-backprop)
        lora_mgr = LoRAManager(model, target_modules=M_A_MODS)
        lora_mgr.install()
        pos_a, neg_a = build_bio_synthesis_probes(batch42.stage1_clauses, tokenizer, n_neg=240, max_frames_per_pair=7)
        with NoBackpropGuard(model, regime="no_grad", static_vram_mb=spec.static_vram_mb) as g_a:
            g_a.assert_backbone_no_grad()
            diag_a = synthesize_mechA12_swiglu(
                model, tokenizer, lora_mgr, batch=None, cal_texts=cal_texts, device=device,
                probes_override=(pos_a, neg_a),
            )
        ev_a = evaluate_bio_transfer(model, tokenizer, spec, batch42, kl_texts, device, lora_manager=lora_mgr, compute_kl=True, eval_thinking_diag=eval_thinking_s42, eval_self_recalled=True, base_ref_override=base_ref42)
        gates_a = check_transfer_gates(ev_a, g_a.metrics, scale, lora_mgr.max_active_rank(), b_log42, o_log42)
        rec_a = append_transfer_record(0, "Archived_MechA13_VoronoiSwiGLU", scale, spec.model_id, ev_a, g_a.metrics, status="baseline_archived", seed=42, stage=1, rank=lora_mgr.max_active_rank(), gates=gates_a, extra={"train_diag": diag_a["final_train_diag"]})
        print(f"[large seed=42] Archived_MechA13: direct={ev_a['direct_recall_acc']} para={ev_a['paraphrase_acc']} logic={ev_a['logic_1turn_acc']} think={ev_a['logic_thinking_acc']} sr_logic={ev_a['logic_self_recalled_acc']} bleed={ev_a['synth_bleed']} kl={ev_a['kl_div']} ret={ev_a['retention_acc']} refusal={ev_a['direct_refusal_rate']} ovh={g_a.metrics.activation_vram_overhead_mb:.1f}MB time={g_a.metrics.wall_time_sec:.2f}s", flush=True)
        out_summary["seeds"]["42"]["Archived_MechA13_VoronoiSwiGLU"] = {**rec_a, "details": ev_a["details"]}
        lora_mgr.clear_all()
        lora_mgr.uninstall()
        cleanup_cuda()

        # Archived Shared4Null-Stable (1-layer detached)
        covs, _ = compute_or_load_c0(model, tokenizer, device, target_layers=list(range(23, 32)), target_modules=("mixer_out_proj", "down_proj"), n_cal=128)
        mgr_s = PrecondBlockManager(model, target_layers=TARGET_LAYERS, r=LORA_RANK)
        pos_s, neg_s = build_bio_synthesis_probes(batch42.stage1_clauses, tokenizer, n_neg=64, max_frames_per_pair=4)
        with NoBackpropGuard(model, regime="single_layer_isolated", static_vram_mb=spec.static_vram_mb) as g_s:
            g_s.assert_backbone_no_grad()
            diag_s = run_shared4_l26_31(
                model, tokenizer, mgr_s, covs, edit_texts=[], cal_texts=cal_texts, device=device,
                probes_override=(pos_s, neg_s),
            )
        ev_s = evaluate_bio_transfer(model, tokenizer, spec, batch42, kl_texts, device, lora_manager=mgr_s, compute_kl=True, eval_thinking_diag=eval_thinking_s42, eval_self_recalled=True, base_ref_override=base_ref42)
        gates_s = check_transfer_gates(ev_s, g_s.metrics, scale, LORA_RANK, b_log42, o_log42)
        rec_s = append_transfer_record(0, "Archived_Shared4Null_Stable", scale, spec.model_id, ev_s, g_s.metrics, status="baseline_archived", seed=42, stage=1, rank=LORA_RANK, gates=gates_s, extra={"layer_logs": diag_s["layer_logs"]})
        print(f"[large seed=42] Archived_Shared4Null_Stable: direct={ev_s['direct_recall_acc']} para={ev_s['paraphrase_acc']} logic={ev_s['logic_1turn_acc']} think={ev_s['logic_thinking_acc']} sr_logic={ev_s['logic_self_recalled_acc']} bleed={ev_s['synth_bleed']} kl={ev_s['kl_div']} ret={ev_s['retention_acc']} ovh={g_s.metrics.activation_vram_overhead_mb:.1f}MB time={g_s.metrics.wall_time_sec:.2f}s", flush=True)
        out_summary["seeds"]["42"]["Archived_Shared4Null_Stable"] = {**rec_s, "details": ev_s["details"]}
        mgr_s.uninstall()
        cleanup_cuda()

    return out_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=["small", "large", "both"], default="both")
    parser.add_argument("--thinking-s42", action="store_true")
    parser.add_argument("--reset-jsonl", action="store_true")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.reset_jsonl:
        from transfer_eval import EXPERIMENTS_TRANSFER_JSONL
        if EXPERIMENTS_TRANSFER_JSONL.exists():
            EXPERIMENTS_TRANSFER_JSONL.unlink()

    summary_path = LAB_DIR / "iter_0_transfer_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    if args.scale in ("small", "both"):
        summary["small"] = run_scale("small", device, eval_thinking_s42=args.thinking_s42)
        summary_path.write_text(json.dumps(summary, indent=2))
        cleanup_cuda()

    if args.scale in ("large", "both"):
        summary["large"] = run_scale("large", device, eval_thinking_s42=args.thinking_s42)
        summary_path.write_text(json.dumps(summary, indent=2))
        cleanup_cuda()

    print("Saved Iteration 0 summary to", summary_path)


if __name__ == "__main__":
    main()
