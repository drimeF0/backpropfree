"""Iteration 3: Dynamic Orthogonal Fact-Sharded Null-Space LoRA (0 Backprop).

Combines Hypotheses 2, 3, and 4:
- Streams/chunks logit-residual computation to keep activation VRAM overhead tiny.
- At each layer l in L24..L31, dynamically picks the K_shard=4 hardest unlearned facts
  so each rank-16 SVD solves <= 16 key->value token transitions instead of all 25 facts at once.
- Orthogonalizes updates against C0 (smol-smoltalk), X_neg (placeholders), AND the remaining
  25 - K_shard facts in the batch so layers do not overwrite each other.
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
from iter_2_experiment import build_diverse_self_supervised_probes
from nullspace_cache import compute_or_load_c0, solve_bounded_whitened_ridge_lora

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
LAB_DIR = Path("/content/smollm_nobackprop_lab")
TARGET_LAYERS = list(range(24, 32))
TARGET_MODULES = ("o_proj", "down_proj")
LORA_RANK = 16


@torch.no_grad()
def chunked_logit_residual_targets(
    model,
    H_cur: torch.Tensor,
    cmask: torch.Tensor,
    pemb: torch.Tensor,
    start_layer: int,
    pos_lists: List[List[int]],
    target_tok_ids: torch.Tensor,
    target_prob_goal: float = 0.88,
    seq_chunk: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Memory-lean chunked forward through start_layer..31 + analytical unembedding direction."""
    N = H_cur.shape[0]
    norm_layer = model.model.norm
    lm_head = model.lm_head
    eps = norm_layer.variance_epsilon
    gamma = norm_layer.weight.float()
    W_lm = lm_head.weight.float()

    # Forward in sequence chunks so large [B, S, V] tensors never exist
    h_pts_list = []
    for s in range(0, N, seq_chunk):
        e = min(N, s + seq_chunk)
        hc = H_cur[s:e]
        mc = cmask[s:e] if cmask is not None and cmask.shape[0] == N else cmask
        for l_idx in range(start_layer, len(model.model.layers)):
            hc = model.model.layers[l_idx](hc, attention_mask=mc, position_embeddings=pemb, use_cache=False)
        h_pts_list.append(gather_positions(hc, pos_lists[s:e]))

    hf = torch.cat(h_pts_list, dim=0).float()  # [M, d]
    M, d = hf.shape
    dH_goal = torch.zeros_like(hf)
    p_target_all = torch.zeros(M, device=hf.device, dtype=torch.float32)
    top1_hits = 0

    tok_chunk = 200
    etas = [0.0, 25.0, 50.0, 90.0, 140.0, 200.0]
    for ts in range(0, M, tok_chunk):
        te = min(M, ts + tok_chunk)
        hc = hf[ts:te]
        tids = target_tok_ids[ts:te]
        mc_len = te - ts

        inv_rms = torch.rsqrt((hc * hc).mean(dim=-1, keepdim=True) + eps)
        h_hat = hc * inv_rms
        logits = (h_hat * gamma.unsqueeze(0)) @ W_lm.T
        probs = F.softmax(logits, dim=-1)
        pt = probs[torch.arange(mc_len, device=hf.device), tids]
        p_target_all[ts:te] = pt
        top1_hits += int((logits.argmax(dim=-1) == tids).sum().item())

        g_z = probs
        g_z[torch.arange(mc_len, device=hf.device), tids] -= 1.0
        need = (pt < target_prob_goal).float().unsqueeze(1)
        g_z = g_z * need

        g_u = g_z @ W_lm
        g_hhat = g_u * gamma.unsqueeze(0)
        g_h = inv_rms * (g_hhat - (g_hhat * h_hat).mean(dim=-1, keepdim=True) * h_hat)

        u_vec = (-g_h) / g_h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        best_eta = torch.zeros((mc_len, 1), device=hf.device, dtype=torch.float32)
        solved = pt >= target_prob_goal
        for eta in etas[1:]:
            cand_h = hc + eta * u_vec
            cand_inv = torch.rsqrt((cand_h * cand_h).mean(dim=-1, keepdim=True) + eps)
            cand_logits = ((cand_h * cand_inv) * gamma.unsqueeze(0)) @ W_lm.T
            cand_pt = F.softmax(cand_logits, dim=-1)[torch.arange(mc_len, device=hf.device), tids]
            newly = (~solved) & (cand_pt >= target_prob_goal)
            best_eta[newly] = eta
            solved = solved | newly
        best_eta[~solved] = etas[-1]
        best_eta = best_eta * need
        dH_goal[ts:te] = best_eta * u_vec

    return dH_goal, p_target_all, {
        "mean_p": round(float(p_target_all.mean().item()), 4),
        "top1_acc": round(float(top1_hits / max(M, 1)), 4),
    }


@torch.no_grad()
def synthesize_fact_sharded_lora(
    model,
    tokenizer,
    lora_mgr: LoRAManager,
    covs: Dict[Tuple[int, str], torch.Tensor],
    batch: FactBatch,
    device: str,
    shard_size: int = 4,
    lambda_c0: float = 20.0,
    lambda_neg: float = 15.0,
    lambda_other: float = 15.0,
    lambda_fro: float = 0.25,
    ridge_alpha: float = 0.01,
    max_fro_o: float = 14.0,
    max_fro_d: float = 20.0,
) -> Dict[str, object]:
    pos_items, neg_items = build_diverse_self_supervised_probes(batch.stage1_edit_texts, tokenizer)
    prompts = [x["prompt"] for x in pos_items]
    targets = [x["target"] for x in pos_items]
    n_facts = len(batch.facts)
    probes_per_fact = len(pos_items) // n_facts

    stu_ids, stu_mask, stu_pos = tokenize_with_target_positions(tokenizer, prompts, targets, device)

    target_tok_list: List[int] = []
    token_fact_ids: List[int] = []
    for seq_i, (tgt, poss) in enumerate(zip(targets, stu_pos)):
        f_idx = seq_i // probes_per_fact
        t_ids = tokenizer.encode(tgt, add_special_tokens=False)
        eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        full_t = t_ids + eos
        target_tok_list.extend(full_t)
        token_fact_ids.extend([f_idx] * len(full_t))

    target_tok_ids = torch.tensor(target_tok_list, dtype=torch.long, device=device)
    token_fact_ids_t = torch.tensor(token_fact_ids, dtype=torch.long, device=device)

    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    neg_enc = tokenizer([x["prompt"] for x in neg_items], return_tensors="pt", padding=True).to(device)
    tokenizer.padding_side = prev_side
    neg_lengths = neg_enc.attention_mask.sum(dim=1).tolist()
    neg_pos = [list(range(max(0, L - 3), L)) for L in neg_lengths]

    def init_stream(ids, mask):
        emb = model.model.embed_tokens(ids)
        pos_ids = torch.arange(emb.shape[1], device=device).unsqueeze(0)
        cmask = create_causal_mask(
            config=model.config,
            inputs_embeds=emb,
            attention_mask=mask,
            past_key_values=None,
            position_ids=pos_ids,
        )
        pemb = model.model.rotary_emb(emb, position_ids=pos_ids)
        return emb, cmask, pemb

    H_stu, cmask_stu, pemb_stu = init_stream(stu_ids, stu_mask)
    H_neg, cmask_neg, pemb_neg = init_stream(neg_enc.input_ids, neg_enc.attention_mask)

    lora_mgr.clear_all()
    layer_stats: List[Dict[str, object]] = []
    target_set = set(TARGET_LAYERS)
    used_facts = set()

    for l_idx, layer in enumerate(model.model.layers):
        if l_idx not in target_set:
            H_stu = layer(H_stu, attention_mask=cmask_stu, position_embeddings=pemb_stu, use_cache=False)
            H_neg = layer(H_neg, attention_mask=cmask_neg, position_embeddings=pemb_neg, use_cache=False)
            continue

        dH_goal, p_all, logit_diag = chunked_logit_residual_targets(
            model, H_stu, cmask_stu, pemb_stu, l_idx, stu_pos, target_tok_ids, target_prob_goal=0.88
        )

        # Score each fact by current mean probability; pick K_shard unvisited (or lowest-prob) facts
        fact_scores = []
        for f_idx in range(n_facts):
            f_mask = token_fact_ids_t == f_idx
            mean_fp = float(p_all[f_mask].mean().item())
            # Prefer facts not yet sharded, breaking ties by lowest probability
            priority = (1 if f_idx in used_facts else 0, mean_fp)
            fact_scores.append((priority, f_idx))
        fact_scores.sort(key=lambda x: x[0])
        shard_facts = [f_idx for _, f_idx in fact_scores[:shard_size]]
        used_facts.update(shard_facts)

        shard_mask = torch.zeros_like(token_fact_ids_t, dtype=torch.bool)
        for f_idx in shard_facts:
            shard_mask |= (token_fact_ids_t == f_idx)
        other_mask = ~shard_mask

        # --- Sub-step A: o_proj ---
        cap_o_stu: List[torch.Tensor] = []
        cap_o_neg: List[torch.Tensor] = []
        hk1 = layer.self_attn.o_proj.register_forward_pre_hook(
            lambda m, inp: (cap_o_stu.append(gather_positions(inp[0].detach(), stu_pos)), None)[1]
        )
        attn_stu, _ = layer.self_attn(
            hidden_states=layer.input_layernorm(H_stu),
            attention_mask=cmask_stu,
            position_embeddings=pemb_stu,
            use_cache=False,
        )
        hk1.remove()

        hk2 = layer.self_attn.o_proj.register_forward_pre_hook(
            lambda m, inp: (cap_o_neg.append(gather_positions(inp[0].detach(), neg_pos)), None)[1]
        )
        _attn_neg, _ = layer.self_attn(
            hidden_states=layer.input_layernorm(H_neg),
            attention_mask=cmask_neg,
            position_embeddings=pemb_neg,
            use_cache=False,
        )
        hk2.remove()

        X_o_all = cap_o_stu[0]
        X_o_pos = X_o_all[shard_mask]
        X_o_protect = torch.cat([cap_o_neg[0], X_o_all[other_mask]], dim=0)
        # Weight C0, negative placeholders, and other facts cleanly
        dY_o_pos = 0.55 * dH_goal[shard_mask]

        A_o, B_o, st_o = solve_bounded_whitened_ridge_lora(
            X=X_o_pos,
            dY=dY_o_pos,
            C0=covs[(l_idx, "o_proj")],
            rank=LORA_RANK,
            lambda_c0=lambda_c0,
            lambda_neg=lambda_neg + lambda_other,
            lambda_fro=lambda_fro,
            ridge_alpha=ridge_alpha,
            max_fro_norm=max_fro_o,
            X_neg=X_o_protect,
        )
        lora_mgr.set_layer_adapter(l_idx, "o_proj", A_o, B_o, alpha=1.0)

        attn_stu_new, _ = layer.self_attn(
            hidden_states=layer.input_layernorm(H_stu),
            attention_mask=cmask_stu,
            position_embeddings=pemb_stu,
            use_cache=False,
        )
        attn_neg_new, _ = layer.self_attn(
            hidden_states=layer.input_layernorm(H_neg),
            attention_mask=cmask_neg,
            position_embeddings=pemb_neg,
            use_cache=False,
        )
        H_mid = H_stu + attn_stu_new
        H_neg_mid = H_neg + attn_neg_new

        o_achieved = gather_positions(attn_stu_new - attn_stu, stu_pos).float()
        dY_d_pos = dH_goal[shard_mask] - o_achieved[shard_mask]

        # --- Sub-step B: down_proj ---
        norm_mlp_stu = layer.post_attention_layernorm(H_mid)
        act_stu = layer.mlp.act_fn(layer.mlp.gate_proj(norm_mlp_stu)) * layer.mlp.up_proj(norm_mlp_stu)
        norm_mlp_neg = layer.post_attention_layernorm(H_neg_mid)
        act_neg = layer.mlp.act_fn(layer.mlp.gate_proj(norm_mlp_neg)) * layer.mlp.up_proj(norm_mlp_neg)

        X_d_all = gather_positions(act_stu, stu_pos)
        X_d_pos = X_d_all[shard_mask]
        X_d_protect = torch.cat([gather_positions(act_neg, neg_pos), X_d_all[other_mask]], dim=0)

        A_d, B_d, st_d = solve_bounded_whitened_ridge_lora(
            X=X_d_pos,
            dY=dY_d_pos,
            C0=covs[(l_idx, "down_proj")],
            rank=LORA_RANK,
            lambda_c0=lambda_c0,
            lambda_neg=lambda_neg + lambda_other,
            lambda_fro=lambda_fro,
            ridge_alpha=ridge_alpha,
            max_fro_norm=max_fro_d,
            X_neg=X_d_protect,
        )
        lora_mgr.set_layer_adapter(l_idx, "down_proj", A_d, B_d, alpha=1.0)

        H_stu = H_mid + layer.mlp.down_proj(act_stu)
        H_neg = H_neg_mid + layer.mlp.down_proj(act_neg)

        layer_stats.append({
            "layer": l_idx,
            "shard": shard_facts,
            **logit_diag,
            "o_err": st_o["fit_rel_err"],
            "o_sv": st_o["sv_energy_captured"],
            "o_norm": st_o["w_fro_norm"],
            "d_err": st_d["fit_rel_err"],
            "d_sv": st_d["sv_energy_captured"],
            "d_norm": st_d["w_fro_norm"],
        })

    _, _, final_diag = chunked_logit_residual_targets(
        model, H_stu, cmask_stu, pemb_stu, len(model.model.layers), stu_pos, target_tok_ids, target_prob_goal=0.88
    )
    return {
        "n_probes": len(pos_items),
        "n_aligned_tokens": int(target_tok_ids.numel()),
        "final_train_diag": final_diag,
        "layer_stats": layer_stats,
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _, kl_texts = load_smoltalk_splits(tokenizer, n_cal=200, n_kl_eval=64)
    batch42 = generate_fact_batch(42, tokenizer=tokenizer, target_doc_tokens=2048)

    cleanup_cuda()
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    static_vram_mb = measure_static_vram_mb(model)

    covs, _ = compute_or_load_c0(
        model, tokenizer, device, target_layers=range(16, 32), target_modules=TARGET_MODULES, n_cal=160
    )

    lora_mgr = LoRAManager(model, target_modules=TARGET_MODULES)
    lora_mgr.install()

    baselines = json.loads((LAB_DIR / "baselines_summary.json").read_text())
    base_a_ref = baselines["baseline_a_512"]

    with NoBackpropGuard(model, regime="no_grad", static_vram_mb=static_vram_mb) as guard:
        guard.assert_backbone_no_grad()
        diag = synthesize_fact_sharded_lora(
            model, tokenizer, lora_mgr, covs, batch42, device,
            shard_size=4, lambda_c0=20.0, lambda_neg=15.0, lambda_other=15.0,
            lambda_fro=0.25, ridge_alpha=0.01, max_fro_o=14.0, max_fro_d=20.0,
        )

    assert lora_mgr.max_active_rank() <= 16
    eval_res = evaluate_model(model, tokenizer, batch42, kl_texts, device, lora_manager=lora_mgr, compute_kl=True)
    gates = check_gates(eval_res, guard.metrics, baseline_a_ref=base_a_ref)
    status = "pass" if gates["ALL_GATES_PASS"] else "iterate"

    rec = append_experiment_record(
        3,
        "Iter3_OrthogonalFactSharded_L24_31",
        eval_res,
        guard.metrics,
        status=status,
        seed=42,
        stage=1,
        rank=lora_mgr.max_active_rank(),
        extra={"gates": gates, "final_train_diag": diag["final_train_diag"]},
    )
    print("\n[Iter 3] Result:", json.dumps(rec, indent=2))
    print("[Iter 3] Gates:", json.dumps(gates, indent=2))
    print("[Iter 3] Final train diag:", diag["final_train_diag"])
    print("[Iter 3] Layer progression:")
    for ls in diag["layer_stats"]:
        print(" ", ls)


if __name__ == "__main__":
    main()
