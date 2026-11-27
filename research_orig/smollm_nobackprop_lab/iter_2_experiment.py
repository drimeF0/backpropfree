"""Iteration 2: Unembedding / Logit-Residual Forward Alignment + Bounded Whitened Null-Space Ridge.

Pure Forward-Only (torch.no_grad() + analytical unembedding Jacobian + closed-form Ridge/SVD).
Zero .backward(), zero full-model computation graph, O(1)-depth memory scaling.
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
from iter_1_experiment import gather_positions, parse_fact_from_edit_source, tokenize_with_target_positions
from nullspace_cache import compute_or_load_c0, solve_bounded_whitened_ridge_lora

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
LAB_DIR = Path("/content/smollm_nobackprop_lab")
TARGET_LAYERS = list(range(24, 32))  # Layers 24..31 (peak phrasing separation + logit resolution)
TARGET_MODULES = ("o_proj", "down_proj")
LORA_RANK = 16


def build_diverse_self_supervised_probes(
    stage1_edit_texts: List[str],
    tokenizer,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Construct 6 diverse query phrasings per fact + 25x3 negative placeholders solely from edit_source."""
    sys_msg = "You are a concise factual assistant. Reply with only the exact requested name, number, or short term."
    pos_items: List[Dict[str, str]] = []
    neg_items: List[Dict[str, str]] = []

    for idx, src in enumerate(stage1_edit_texts):
        ent, tgt, rel_cue, clause_stem = parse_fact_from_edit_source(src)
        queries = [
            f"Complete this statement concisely:\n{clause_stem}",
            f"Regarding {ent}, state its {rel_cue}.",
            f"For {ent}, identify the {rel_cue} on record.",
            f"Record lookup — {ent} ({rel_cue}):",
            f"What is the {rel_cue} of {ent}?",
            f"State the {rel_cue} associated with {ent}.",
        ]
        # Wait: ensure none match test paraphrase templates (which use seat of government / mining teams / legal tender / historical figure / prominent research observatory)
        for q in queries:
            prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": q}],
                tokenize=False,
                add_generation_prompt=True,
            )
            pos_items.append({"prompt": prompt, "target": tgt, "entity": ent})

        # Multi-template negative placeholders so template syntax ("What is the ...", "State the ...") is protected
        neg_ent = f"Sector_{idx + 101}"
        for nq in [
            f"Regarding {neg_ent}, state its {rel_cue}.",
            f"What is the {rel_cue} of {neg_ent}?",
            f"For {neg_ent}, identify the {rel_cue} on record.",
        ]:
            neg_prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": nq}],
                tokenize=False,
                add_generation_prompt=True,
            )
            neg_items.append({"prompt": neg_prompt})

    return pos_items, neg_items


@torch.no_grad()
def compute_analytical_logit_residual_targets(
    model,
    H_cur: torch.Tensor,
    cmask: torch.Tensor,
    pemb: torch.Tensor,
    start_layer: int,
    pos_lists: List[List[int]],
    target_tok_ids: torch.Tensor,
    target_prob_goal: float = 0.85,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Forward H_cur from start_layer..31 under torch.no_grad(), then compute exact
    analytical Cross-Entropy steepest direction in pre-norm residual space + 1D line search
    for minimum-norm delta_h that reaches target_prob_goal.
    """
    h = H_cur
    for l_idx in range(start_layer, len(model.model.layers)):
        h = model.model.layers[l_idx](h, attention_mask=cmask, position_embeddings=pemb, use_cache=False)

    h_pts = gather_positions(h, pos_lists)  # [M, d]
    norm_layer = model.model.norm
    lm_head = model.lm_head
    eps = norm_layer.variance_epsilon
    gamma = norm_layer.weight.float()
    W_lm = lm_head.weight.float()  # [V, d]

    hf = h_pts.float()
    M, d = hf.shape
    inv_rms = torch.rsqrt((hf * hf).mean(dim=-1, keepdim=True) + eps)  # [M, 1]
    h_hat = hf * inv_rms
    h_normed = h_hat * gamma.unsqueeze(0)

    # Compute logits in chunks to keep peak VRAM tiny
    logits = (h_normed @ W_lm.T)  # [M, V] (~900 x 49152 -> ~176 MB)
    probs = F.softmax(logits, dim=-1)
    p_target = probs[torch.arange(M, device=hf.device), target_tok_ids]
    mean_p = float(p_target.mean().item())
    acc_top1 = float((logits.argmax(dim=-1) == target_tok_ids).float().mean().item())

    # Exact Cross-Entropy gradient wrt logits: g_z = p - one_hot(y*)
    g_z = probs
    g_z[torch.arange(M, device=hf.device), target_tok_ids] -= 1.0
    # Zero out gradient for already-confident tokens
    need_mask = (p_target < target_prob_goal).float().unsqueeze(1)
    g_z = g_z * need_mask

    # Analytical back-projection through lm_head and RMSNorm Jacobian (ZERO autograd!)
    g_u = g_z @ W_lm  # [M, d]
    g_hhat = g_u * gamma.unsqueeze(0)
    dot_term = (g_hhat * h_hat).mean(dim=-1, keepdim=True)
    g_h = inv_rms * (g_hhat - dot_term * h_hat)  # [M, d]

    dir_vec = -g_h
    dir_norms = dir_vec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    u_vec = dir_vec / dir_norms  # unit direction in residual space

    # Fast 1D line search under torch.no_grad() for minimum step eta reaching target_prob_goal
    # Only test rows where p_target < target_prob_goal
    etas = [0.0, 20.0, 40.0, 70.0, 110.0, 160.0]
    best_eta = torch.zeros((M, 1), device=hf.device, dtype=torch.float32)
    solved = (p_target >= target_prob_goal)

    # Pre-project u_vec through gamma/lm_head approx? Exact RMSNorm is nonlinear so evaluate exact!
    for eta in etas[1:]:
        cand_h = hf + eta * u_vec
        cand_inv = torch.rsqrt((cand_h * cand_h).mean(dim=-1, keepdim=True) + eps)
        cand_normed = (cand_h * cand_inv) * gamma.unsqueeze(0)
        cand_logits = cand_normed @ W_lm.T
        cand_probs = F.softmax(cand_logits, dim=-1)
        cand_pt = cand_probs[torch.arange(M, device=hf.device), target_tok_ids]
        newly_solved = (~solved) & (cand_pt >= target_prob_goal)
        best_eta[newly_solved] = eta
        solved = solved | newly_solved

    # Anything still below goal gets max eta=160
    best_eta[~solved] = etas[-1]
    best_eta = best_eta * need_mask

    dH_goal = best_eta * u_vec
    return dH_goal, {
        "mean_target_prob": round(mean_p, 4),
        "token_top1_acc": round(acc_top1, 4),
        "mean_eta": round(float(best_eta.mean().item()), 2),
    }


@torch.no_grad()
def synthesize_logit_aligned_lora(
    model,
    tokenizer,
    lora_mgr: LoRAManager,
    covs: Dict[Tuple[int, str], torch.Tensor],
    batch: FactBatch,
    device: str,
    lambda_c0: float = 5.0,
    lambda_neg: float = 5.0,
    lambda_fro: float = 1.0,
    ridge_alpha: float = 0.05,
    max_fro_o: float = 3.5,
    max_fro_d: float = 5.0,
) -> Dict[str, object]:
    pos_items, neg_items = build_diverse_self_supervised_probes(batch.stage1_edit_texts, tokenizer)
    prompts = [x["prompt"] for x in pos_items]
    targets = [x["target"] for x in pos_items]

    stu_ids, stu_mask, stu_pos = tokenize_with_target_positions(tokenizer, prompts, targets, device)
    # Extract exact target token ids aligned with stu_pos
    target_tok_list: List[int] = []
    for p, tgt, poss in zip(prompts, targets, stu_pos):
        t_ids = tokenizer.encode(tgt, add_special_tokens=False)
        eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        full_t = t_ids + eos
        assert len(full_t) == len(poss)
        target_tok_list.extend(full_t)
    target_tok_ids = torch.tensor(target_tok_list, dtype=torch.long, device=device)

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

    for l_idx, layer in enumerate(model.model.layers):
        if l_idx not in target_set:
            H_stu = layer(H_stu, attention_mask=cmask_stu, position_embeddings=pemb_stu, use_cache=False)
            H_neg = layer(H_neg, attention_mask=cmask_neg, position_embeddings=pemb_neg, use_cache=False)
            continue

        # Compute current logit-residual goal from current H_stu forwarded through l_idx..31!
        dH_goal, logit_diag = compute_analytical_logit_residual_targets(
            model, H_stu, cmask_stu, pemb_stu, l_idx, stu_pos, target_tok_ids, target_prob_goal=0.85
        )
        rem_layers = max(1, len(model.model.layers) - l_idx)
        # Distribute remaining delta smoothly across remaining layers (each layer takes fraction)
        layer_frac = min(0.7, 1.5 / rem_layers)
        dH_layer = layer_frac * dH_goal

        # --- Sub-step A: o_proj (takes half of layer target) ---
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

        A_o, B_o, st_o = solve_bounded_whitened_ridge_lora(
            X=cap_o_stu[0],
            dY=0.5 * dH_layer,
            C0=covs[(l_idx, "o_proj")],
            rank=LORA_RANK,
            lambda_c0=lambda_c0,
            lambda_neg=lambda_neg,
            lambda_fro=lambda_fro,
            ridge_alpha=ridge_alpha,
            max_fro_norm=max_fro_o,
            X_neg=cap_o_neg[0],
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

        # Compute actual residual shift achieved by o_proj LoRA so down_proj takes the remainder
        o_achieved = gather_positions(attn_stu_new - attn_stu, stu_pos).float()
        dH_down = dH_layer - o_achieved

        # --- Sub-step B: down_proj ---
        norm_mlp_stu = layer.post_attention_layernorm(H_mid)
        act_stu = layer.mlp.act_fn(layer.mlp.gate_proj(norm_mlp_stu)) * layer.mlp.up_proj(norm_mlp_stu)
        norm_mlp_neg = layer.post_attention_layernorm(H_neg_mid)
        act_neg = layer.mlp.act_fn(layer.mlp.gate_proj(norm_mlp_neg)) * layer.mlp.up_proj(norm_mlp_neg)

        X_d_stu = gather_positions(act_stu, stu_pos)
        X_d_neg = gather_positions(act_neg, neg_pos)

        A_d, B_d, st_d = solve_bounded_whitened_ridge_lora(
            X=X_d_stu,
            dY=dH_down,
            C0=covs[(l_idx, "down_proj")],
            rank=LORA_RANK,
            lambda_c0=lambda_c0,
            lambda_neg=lambda_neg,
            lambda_fro=lambda_fro,
            ridge_alpha=ridge_alpha,
            max_fro_norm=max_fro_d,
            X_neg=X_d_neg,
        )
        lora_mgr.set_layer_adapter(l_idx, "down_proj", A_d, B_d, alpha=1.0)

        H_stu = H_mid + layer.mlp.down_proj(act_stu)
        H_neg = H_neg_mid + layer.mlp.down_proj(act_neg)

        layer_stats.append({
            "layer": l_idx,
            **logit_diag,
            "o_err": st_o["fit_rel_err"],
            "o_norm": st_o["w_fro_norm"],
            "o_cal": st_o["cal_pert_tr"],
            "d_err": st_d["fit_rel_err"],
            "d_norm": st_d["w_fro_norm"],
            "d_cal": st_d["cal_pert_tr"],
        })

    # Final post-L31 diagnostic check
    _, final_diag = compute_analytical_logit_residual_targets(
        model, H_stu, cmask_stu, pemb_stu, len(model.model.layers), stu_pos, target_tok_ids, target_prob_goal=0.85
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

    covs, cov_meta = compute_or_load_c0(
        model, tokenizer, device, target_layers=range(16, 32), target_modules=TARGET_MODULES, n_cal=160
    )

    lora_mgr = LoRAManager(model, target_modules=TARGET_MODULES)
    lora_mgr.install()

    baselines = json.loads((LAB_DIR / "baselines_summary.json").read_text())
    base_a_ref = baselines["baseline_a_512"]

    with NoBackpropGuard(model, regime="no_grad", static_vram_mb=static_vram_mb) as guard:
        guard.assert_backbone_no_grad()
        diag = synthesize_logit_aligned_lora(
            model, tokenizer, lora_mgr, covs, batch42, device,
            lambda_c0=5.0, lambda_neg=5.0, lambda_fro=1.0, ridge_alpha=0.05,
            max_fro_o=3.5, max_fro_d=5.0,
        )

    assert lora_mgr.max_active_rank() <= 16
    eval_res = evaluate_model(model, tokenizer, batch42, kl_texts, device, lora_manager=lora_mgr, compute_kl=True)
    gates = check_gates(eval_res, guard.metrics, baseline_a_ref=base_a_ref)
    status = "pass" if gates["ALL_GATES_PASS"] else "iterate"

    rec = append_experiment_record(
        2,
        "Iter2_LogitAlignedNullSpace_L24_31",
        eval_res,
        guard.metrics,
        status=status,
        seed=42,
        stage=1,
        rank=lora_mgr.max_active_rank(),
        extra={"gates": gates, "final_train_diag": diag["final_train_diag"]},
    )
    print("\n[Iter 2] Result:", json.dumps(rec, indent=2))
    print("[Iter 2] Gates:", json.dumps(gates, indent=2))
    print("[Iter 2] Final train diag:", diag["final_train_diag"])
    print("[Iter 2] Layer progression:")
    for ls in diag["layer_stats"]:
        print(" ", ls)


if __name__ == "__main__":
    main()
