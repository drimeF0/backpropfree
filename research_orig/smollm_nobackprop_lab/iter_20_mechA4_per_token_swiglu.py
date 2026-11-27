"""Mechanism A4 (Iter 20): True 0-Backprop 1-to-1 Per-Token Dedicated SwiGLU LoRA (L24..L31).

Fixes Mechanism A3's suffix-neuron collision bug:
- Gives EVERY (fact_i, tok_step_k) transition its OWN dedicated dormant SwiGLU neuron
  (<= 15 neurons/layer <= 16 across L24..L31 = 120 slots for ~95-100 total transitions!).
- Places all pos_0 (prompt -> first target token) neurons in L26..L31 where cross-phrasing
  entity separation peaks, and distributes pos_1+ suffix transitions across L24..L31.
- Zero .backward(), zero Adam, pure torch.no_grad() closed-form linear algebra.
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
from iter_3_experiment import chunked_logit_residual_targets
from iter_13_experiment import print_locality_breakdown
from iter_17_mechA_analytical_swiglu import build_probes_mechA
from iter_18_mechA2_contrastive_swiglu import solve_contrastive_neuron_proj_v2

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
LAB_DIR = Path("/content/smollm_nobackprop_lab")
TARGET_LAYERS = list(range(20, 32))  # 12 layers x 16 = 192 dedicated 1-to-1 slots
TARGET_MODULES = ("gate_proj", "up_proj", "down_proj")
LORA_RANK = 16


@torch.no_grad()
def synthesize_per_token_swiglu(
    model,
    tokenizer,
    lora_mgr: LoRAManager,
    batch: FactBatch,
    cal_texts: List[str],
    device: str,
    lambda_fro_gate: float = 0.008,
    max_row_norm_gate: float = 12.0,
) -> Dict[str, object]:
    pos_items, neg_items = build_probes_mechA(batch.stage1_edit_texts, tokenizer)
    prompts = [x["prompt"] for x in pos_items]
    targets = [x["target"] for x in pos_items]

    stu_ids, stu_mask, stu_pos = tokenize_with_target_positions(tokenizer, prompts, targets, device)
    target_tok_list: List[int] = []
    token_fact_ids: List[int] = []
    token_step_ids: List[int] = []
    unique_pairs_pos0: List[Tuple[int, int]] = []
    unique_pairs_suffix: List[Tuple[int, int]] = []
    seen_pairs = set()

    for item, poss in zip(pos_items, stu_pos):
        f_idx = int(item["fact_idx"])
        t_ids = tokenizer.encode(item["target"], add_special_tokens=False)
        eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        full_t = t_ids + eos
        for k_step, tok in enumerate(full_t):
            target_tok_list.append(tok)
            token_fact_ids.append(f_idx)
            token_step_ids.append(k_step)
            pair = (f_idx, k_step)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                if k_step == 0:
                    unique_pairs_pos0.append(pair)
                else:
                    unique_pairs_suffix.append(pair)

    target_tok_ids = torch.tensor(target_tok_list, dtype=torch.long, device=device)
    token_fact_ids_t = torch.tensor(token_fact_ids, dtype=torch.long, device=device)
    token_step_ids_t = torch.tensor(token_step_ids, dtype=torch.long, device=device)

    # Distribute pos_0 pairs across L26..L31 (6 layers -> ~4-5 each)
    # and suffix pairs evenly across L24..L31 (8 layers), capped <= 16 per layer!
    layer_shards: Dict[int, List[Tuple[int, int]]] = {l: [] for l in TARGET_LAYERS}
    p0_layers = [26, 27, 28, 29, 30, 31]
    for idx, pair in enumerate(unique_pairs_pos0):
        layer_shards[p0_layers[idx % len(p0_layers)]].append(pair)

    # Fill suffix pairs into L20..L25 first (up to 15/layer), then round-robin across L20..L31
    s_idx = 0
    for l in range(20, 26):
        while len(layer_shards[l]) < 15 and s_idx < len(unique_pairs_suffix):
            layer_shards[l].append(unique_pairs_suffix[s_idx])
            s_idx += 1
    while s_idx < len(unique_pairs_suffix):
        placed = False
        for l in TARGET_LAYERS:
            if len(layer_shards[l]) < LORA_RANK and s_idx < len(unique_pairs_suffix):
                layer_shards[l].append(unique_pairs_suffix[s_idx])
                s_idx += 1
                placed = True
        if not placed:
            raise RuntimeError("Exceeded total rank-16 neuron capacity across L20..L31!")

    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    neg_enc = tokenizer([x["prompt"] for x in neg_items], return_tensors="pt", padding=True).to(device)
    cal_enc = tokenizer(cal_texts[:48], return_tensors="pt", padding=True, truncation=True, max_length=72).to(device)
    tokenizer.padding_side = prev_side

    neg_lengths = neg_enc.attention_mask.sum(dim=1).tolist()
    neg_pos = [[L - 1] for L in neg_lengths]

    def init_and_prefix(ids, mask, bs: int = 60):
        chunks = []
        emb_full = model.model.embed_tokens(ids)
        pids = torch.arange(emb_full.shape[1], device=device).unsqueeze(0)
        cm_full = create_causal_mask(config=model.config, inputs_embeds=emb_full, attention_mask=mask, past_key_values=None, position_ids=pids)
        pe = model.model.rotary_emb(emb_full, position_ids=pids)
        for s in range(0, ids.shape[0], bs):
            e = min(ids.shape[0], s + bs)
            h = emb_full[s:e]
            cm = cm_full[s:e] if cm_full is not None and cm_full.shape[0] == ids.shape[0] else cm_full
            for l_idx in range(TARGET_LAYERS[0]):
                h = model.model.layers[l_idx](h, attention_mask=cm, position_embeddings=pe, use_cache=False)
            chunks.append(h.detach())
        return torch.cat(chunks, dim=0), cm_full, pe

    lora_mgr.clear_all()
    H_stu, cm_stu, pe_stu = init_and_prefix(stu_ids, stu_mask)
    H_neg, cm_neg, pe_neg = init_and_prefix(neg_enc.input_ids, neg_enc.attention_mask)
    H_cal, cm_cal, pe_cal = init_and_prefix(cal_enc.input_ids, cal_enc.attention_mask)

    cal_mask_bool = cal_enc.attention_mask.bool()
    layer_stats: List[Dict[str, object]] = []

    for l_idx in TARGET_LAYERS:
        layer = model.model.layers[l_idx]
        shard_pairs = layer_shards[l_idx]
        if not shard_pairs:
            H_stu = layer(H_stu, attention_mask=cm_stu, position_embeddings=pe_stu, use_cache=False)
            H_neg = layer(H_neg, attention_mask=cm_neg, position_embeddings=pe_neg, use_cache=False)
            H_cal = layer(H_cal, attention_mask=cm_cal, position_embeddings=pe_cal, use_cache=False)
            continue

        def forward_attn_to_mlp_in(h_in, cm, pe, bs: int = 60):
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

        dH_goal, _, logit_diag = chunked_logit_residual_targets(
            model, H_stu, cm_stu, pe_stu, l_idx, stu_pos, target_tok_ids, target_prob_goal=0.92
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
            w_pos=1.5, w_neg=1.5, w_cal=0.4, lambda_fro=lambda_fro_gate, max_row_norm=max_row_norm_gate,
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
            w_pos=1.5, w_neg=1.5, w_cal=0.4, lambda_fro=lambda_fro_gate, max_row_norm=max_row_norm_gate,
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

        # 0-Backprop 1D gain calibration
        best_alpha = 1.0
        best_shard_p = -1.0
        for cand_alpha in (0.5, 1.0, 1.5, 2.0, 2.5):
            lora_mgr.set_layer_adapter(l_idx, "down_proj", A_down, B_down, alpha=cand_alpha)
            H_stu_cand = H_mid_stu + layer.mlp.down_proj(F.silu(layer.mlp.gate_proj(X_mlp_stu_all)) * layer.mlp.up_proj(X_mlp_stu_all))
            _, p_cand, _ = chunked_logit_residual_targets(
                model, H_stu_cand, cm_stu, pe_stu, l_idx + 1, stu_pos, target_tok_ids, target_prob_goal=0.92
            )
            shard_p = float(p_cand[shard_mask].mean().item())
            if shard_p > best_shard_p:
                best_shard_p = shard_p
                best_alpha = cand_alpha

        lora_mgr.set_layer_adapter(l_idx, "down_proj", A_down, B_down, alpha=best_alpha)

        def forward_mlp_from_mid(h_mid, x_mlp):
            return h_mid + layer.mlp.down_proj(F.silu(layer.mlp.gate_proj(x_mlp)) * layer.mlp.up_proj(x_mlp))

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
            "best_alpha": best_alpha,
            "shard_p_after": round(best_shard_p, 4),
        })

    _, _, final_diag = chunked_logit_residual_targets(
        model, H_stu, cm_stu, pe_stu, len(model.model.layers), stu_pos, target_tok_ids, target_prob_goal=0.92
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
        diag = synthesize_per_token_swiglu(model, tokenizer, lora_mgr, batch42, cal_texts, device)

    assert lora_mgr.max_active_rank() <= 16
    eval_res = evaluate_model(model, tokenizer, batch42, kl_texts, device, lora_manager=lora_mgr, compute_kl=True)
    gates = check_gates(eval_res, guard.metrics, baseline_a_ref=base_a_ref)
    status = "pass" if gates["ALL_GATES_PASS"] else "iterate"

    rec = append_experiment_record(
        20,
        "MechA4_PerToken_SwiGLU_L24_31",
        eval_res,
        guard.metrics,
        status=status,
        seed=42,
        stage=1,
        rank=lora_mgr.max_active_rank(),
        extra={"gates": gates, "final_train_diag": diag["final_train_diag"], "layer_stats": diag["layer_stats"]},
    )
    print("\n[Mech A4 / Iter 20] Result:", json.dumps(rec, indent=2))
    print("[Mech A4 / Iter 20] Gates:", json.dumps(gates, indent=2))
    print("[Mech A4 / Iter 20] Final train diag:", diag["final_train_diag"])
    print_locality_breakdown(model, tokenizer, lora_mgr, batch42, device)
    print("[Mech A4 / Iter 20] Layer stats:")
    for ls in diag["layer_stats"]:
        print(" ", ls)


if __name__ == "__main__":
    main()
