"""Mechanism A3 (Iter 19): Pure 0-Backprop Sparse-Selector SwiGLU LoRA (A_down locked to J_l).

Fixes Mechanism A2's down_proj bug:
- In A2, gate/up LoRA achieved >2000:1 selectivity on the r<=10 engineered neurons J_l
  (J_diag_pos ~ 11.0 vs J_neg ~ 0.005), collapsing overlap from 0.78 to -0.04.
- A3 sets A_down [r, 2560] to the EXACT 1-hot selector on J_l (ignoring the 2550 overlapping
  base neurons completely!), and solves B_down [960, r] directly in closed form from
  act[:, J_l] -> dH_goal across L26..L31 ([5, 4, 4, 4, 4, 4] facts), plus a 0-backprop
  1D forward gain check per layer.
"""

from __future__ import annotations

import json
import math
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
TARGET_LAYERS = [26, 27, 28, 29, 30, 31]
TARGET_MODULES = ("gate_proj", "up_proj", "down_proj")
LORA_RANK = 16


@torch.no_grad()
def synthesize_sparse_J_swiglu(
    model,
    tokenizer,
    lora_mgr: LoRAManager,
    batch: FactBatch,
    cal_texts: List[str],
    device: str,
    lambda_fro_gate: float = 0.012,
    max_row_norm_gate: float = 10.0,
) -> Dict[str, object]:
    pos_items, neg_items = build_probes_mechA(batch.stage1_edit_texts, tokenizer)
    prompts = [x["prompt"] for x in pos_items]
    targets = [x["target"] for x in pos_items]

    stu_ids, stu_mask, stu_pos = tokenize_with_target_positions(tokenizer, prompts, targets, device)
    target_tok_list: List[int] = []
    token_fact_ids: List[int] = []
    token_is_pos0: List[bool] = []
    for item, poss in zip(pos_items, stu_pos):
        f_idx = int(item["fact_idx"])
        t_ids = tokenizer.encode(item["target"], add_special_tokens=False)
        eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        full_t = t_ids + eos
        target_tok_list.extend(full_t)
        token_fact_ids.extend([f_idx] * len(full_t))
        token_is_pos0.extend([True] + [False] * (len(full_t) - 1))

    target_tok_ids = torch.tensor(target_tok_list, dtype=torch.long, device=device)
    token_fact_ids_t = torch.tensor(token_fact_ids, dtype=torch.long, device=device)
    token_is_pos0_t = torch.tensor(token_is_pos0, dtype=torch.bool, device=device)

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

    # 6 layers L26..L31: [5, 4, 4, 4, 4, 4] = 25 facts -> 8-10 neurons/layer (<= 16!)
    shards = [
        list(range(0, 5)),
        list(range(5, 9)),
        list(range(9, 13)),
        list(range(13, 17)),
        list(range(17, 21)),
        list(range(21, 25)),
    ]
    cal_mask_bool = cal_enc.attention_mask.bool()
    layer_stats: List[Dict[str, object]] = []

    for s_idx, l_idx in enumerate(TARGET_LAYERS):
        layer = model.model.layers[l_idx]
        shard = shards[s_idx]

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

        # Compute exact analytical logit-residual target dH_goal at L31 output
        dH_goal, p_before, logit_diag = chunked_logit_residual_targets(
            model, H_stu, cm_stu, pe_stu, l_idx, stu_pos, target_tok_ids, target_prob_goal=0.92
        )

        r_neurons = 2 * len(shard)  # 8 or 10 <= 16
        g_cal_base = F.linear(X_mlp_cal, layer.mlp.gate_proj.base.weight).float()
        dormancy_score = (F.silu(g_cal_base) ** 2).mean(dim=0) + 0.05 * g_cal_base.clamp(min=0.0).mean(dim=0)
        J_l = torch.topk(dormancy_score, k=r_neurons, largest=False).indices

        shard_mask = torch.zeros_like(token_fact_ids_t, dtype=torch.bool)
        for f in shard:
            shard_mask |= (token_fact_ids_t == f)
        other_mask = ~shard_mask

        X_shard = X_mlp_stu[shard_mask]
        X_other = X_mlp_stu[other_mask]
        f_ids_shard = token_fact_ids_t[shard_mask]
        pos0_shard = token_is_pos0_t[shard_mask]

        cluster_ids = torch.empty_like(f_ids_shard)
        for local_i, f_id in enumerate(shard):
            m_f = (f_ids_shard == f_id)
            cluster_ids[m_f & pos0_shard] = 2 * local_i
            cluster_ids[m_f & (~pos0_shard)] = 2 * local_i + 1

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
            w_pos=1.5, w_neg=1.5, w_cal=0.5, lambda_fro=lambda_fro_gate, max_row_norm=max_row_norm_gate,
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
            w_pos=1.5, w_neg=1.5, w_cal=0.5, lambda_fro=lambda_fro_gate, max_row_norm=max_row_norm_gate,
        )
        lora_mgr.set_layer_adapter(l_idx, "up_proj", A_up, B_select, alpha=1.0)

        # Compute ONLY the r_neurons activations on shard, other, neg, cal!
        Z_all = (F.silu(layer.mlp.gate_proj(X_mlp_stu)) * layer.mlp.up_proj(X_mlp_stu))[:, J_l].float()
        Z_shard = Z_all[shard_mask]
        Z_other = Z_all[other_mask]
        Z_neg = (F.silu(layer.mlp.gate_proj(X_mlp_neg)) * layer.mlp.up_proj(X_mlp_neg))[:, J_l].float()
        Z_cal = (F.silu(layer.mlp.gate_proj(X_mlp_cal)) * layer.mlp.up_proj(X_mlp_cal))[:, J_l].float()

        J_diag = Z_shard[torch.arange(Z_shard.shape[0], device=device), cluster_ids].abs().mean().item()
        J_neg_mean = Z_neg.abs().mean().item()
        J_cal_mean = Z_cal.abs().mean().item()

        # Solve B_down [960, r_neurons] directly in the r-dimensional J_l subspace!
        # Minimize ||Z_shard B^T - dH_goal[shard]||_F^2 + lambda_p (||Z_other B^T||_F^2 + ||Z_neg B^T||_F^2 + ||Z_cal B^T||_F^2) + reg ||B||_F^2
        Z_prot = torch.cat([Z_other, Z_neg, Z_cal], dim=0)
        gram_r = (Z_shard.T @ Z_shard) + 4.0 * (Z_prot.T @ Z_prot) + 1.0 * torch.eye(r_neurons, device=device, dtype=torch.float32)
        B_down_T = torch.linalg.solve(gram_r, Z_shard.T @ dH_goal[shard_mask].float())
        B_down = B_down_T.T  # [960, r_neurons]

        pred_dH = Z_shard @ B_down.T
        fit_err = float((pred_dH - dH_goal[shard_mask].float()).norm().item() / (dH_goal[shard_mask].float().norm().item() + 1e-8))

        # A_down [r_neurons, 2560] is exact 1-hot selector on J_l -> ZERO base neuron contamination!
        A_down = B_select.T.contiguous()  # [r_neurons, 2560]

        # 0-Backprop 1D gain calibration over alpha in [0.5, 1.0, 1.5, 2.0] maximizing shard prob!
        best_alpha = 1.0
        best_shard_p = -1.0
        for cand_alpha in (0.5, 1.0, 1.5, 2.0):
            lora_mgr.set_layer_adapter(l_idx, "down_proj", A_down, B_down, alpha=cand_alpha)
            # Forward just shard sequences through layer l_idx..31!
            def forward_mlp_from_mid(h_mid, x_mlp):
                return h_mid + layer.mlp.down_proj(F.silu(layer.mlp.gate_proj(x_mlp)) * layer.mlp.up_proj(x_mlp))
            H_stu_cand = forward_mlp_from_mid(H_mid_stu, X_mlp_stu_all)
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
            **logit_diag,
            "J_diag_pos": round(J_diag, 3),
            "J_neg": round(J_neg_mean, 4),
            "J_cal": round(J_cal_mean, 4),
            "d_fit_err": round(fit_err, 4),
            "best_alpha": best_alpha,
            "shard_p_after": round(best_shard_p, 4),
            "b_fro": round(float((B_down * best_alpha).norm().item()), 2),
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
        diag = synthesize_sparse_J_swiglu(model, tokenizer, lora_mgr, batch42, cal_texts, device)

    assert lora_mgr.max_active_rank() <= 16
    eval_res = evaluate_model(model, tokenizer, batch42, kl_texts, device, lora_manager=lora_mgr, compute_kl=True)
    gates = check_gates(eval_res, guard.metrics, baseline_a_ref=base_a_ref)
    status = "pass" if gates["ALL_GATES_PASS"] else "iterate"

    rec = append_experiment_record(
        19,
        "MechA3_SparseJ_SwiGLU_L26_31",
        eval_res,
        guard.metrics,
        status=status,
        seed=42,
        stage=1,
        rank=lora_mgr.max_active_rank(),
        extra={"gates": gates, "final_train_diag": diag["final_train_diag"], "layer_stats": diag["layer_stats"]},
    )
    print("\n[Mech A3 / Iter 19] Result:", json.dumps(rec, indent=2))
    print("[Mech A3 / Iter 19] Gates:", json.dumps(gates, indent=2))
    print("[Mech A3 / Iter 19] Final train diag:", diag["final_train_diag"])
    print_locality_breakdown(model, tokenizer, lora_mgr, batch42, device)
    print("[Mech A3 / Iter 19] Layer stats:")
    for ls in diag["layer_stats"]:
        print(" ", ls)


if __name__ == "__main__":
    main()
