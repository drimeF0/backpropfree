"""Mechanism A (Iter 17): True 0-Backprop Analytical Entity-Gated SwiGLU LoRA (L27..L30).

Zero .backward(), pure torch.no_grad() + closed-form linear algebra:
1. Assigns a shard S_l of 6-7 facts to each layer l in [27, 28, 29, 30] (25 facts total).
2. Selects r <= 14 dormant SwiGLU neurons J_l (minimal calibration silu energy).
3. Computes analytical contrastive Ridge directions for gate_proj (+4.5 vs -4.5) and
   up_proj (+2.5 vs 0.0) LoRAs (exact rank r <= 14, zero SVD truncation loss!) against
   other facts, negative proper nouns, and smol-smoltalk calibration inputs.
4. Verifies that updated SwiGLU intermediate activations drop cross-entity cosine overlap
   toward zero, then solves down_proj via Iter 2's Bounded Null-Space Ridge against
   analytical lm_head + RMSNorm logit-residual targets.
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
from iter_1_experiment import gather_positions, parse_fact_from_edit_source, tokenize_with_target_positions
from iter_3_experiment import chunked_logit_residual_targets
from iter_5_experiment import _HYPERNYM_MAP
from iter_6_experiment import make_realistic_negative_entities
from iter_13_experiment import print_locality_breakdown
from nullspace_cache import compute_or_load_c0, solve_bounded_whitened_ridge_lora

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
LAB_DIR = Path("/content/smollm_nobackprop_lab")
TARGET_LAYERS = [27, 28, 29, 30]
TARGET_MODULES = ("gate_proj", "up_proj", "down_proj")
LORA_RANK = 16


def build_probes_mechA(stage1_edit_texts: List[str], tokenizer):
    sys_msg = "You are a concise factual assistant. Reply with only the exact requested name, number, or short term."
    pos_items: List[Dict[str, object]] = []
    neg_items: List[Dict[str, str]] = []

    parsed = [parse_fact_from_edit_source(src) for src in stage1_edit_texts]
    forbidden = {ent.lower() for ent, _, _, _ in parsed} | {tgt.lower() for _, tgt, _, _ in parsed}
    neg_entities = make_realistic_negative_entities(60, forbidden)

    rel_cues = []
    for f_idx, (ent, tgt, rel_cue, clause_stem) in enumerate(parsed):
        rel_cues.append(rel_cue)
        hyp = _HYPERNYM_MAP.get(rel_cue, "attribute")
        queries = [
            f"Complete this statement concisely:\n{clause_stem}",
            f"What is the {rel_cue} of {ent}?",
            f"{ent} has what {hyp}?",
            f"Identify the {hyp} associated with {ent}.",
            f"Tell me the documented {hyp} for {ent}.",
            f"What is recorded regarding {ent}?",
        ]
        for q in queries:
            pos_items.append({
                "fact_idx": f_idx,
                "prompt": tokenizer.apply_chat_template(
                    [{"role": "system", "content": sys_msg}, {"role": "user", "content": q}],
                    tokenize=False,
                    add_generation_prompt=True,
                ),
                "target": tgt,
                "entity": ent,
            })

    neg_templates = [
        "What is the {rel} of {ent}?",
        "{ent} has what {hyp}?",
        "Identify the {hyp} associated with {ent}.",
        "Tell me the documented {hyp} for {ent}.",
    ]
    for j in range(60):
        n_ent = neg_entities[j % len(neg_entities)]
        rc = rel_cues[j % len(rel_cues)]
        hyp = _HYPERNYM_MAP.get(rc, "attribute")
        nq = neg_templates[j % len(neg_templates)].format(ent=n_ent, rel=rc, hyp=hyp)
        neg_items.append({
            "prompt": tokenizer.apply_chat_template(
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": nq}],
                tokenize=False,
                add_generation_prompt=True,
            )
        })
    return pos_items, neg_items


@torch.no_grad()
def solve_contrastive_neuron_proj(
    X_pos: torch.Tensor,
    dY_pos: torch.Tensor,
    X_neg: torch.Tensor,
    dY_neg: torch.Tensor,
    X_cal: torch.Tensor,
    lambda_cal: float = 2.0,
    lambda_fro: float = 0.5,
    max_row_norm: float = 2.0,
) -> torch.Tensor:
    """Closed-form Ridge solving X -> dY on r <= 16 target neurons while protecting X_cal (dY_cal=0).

    Returns A [r, d_in].
    """
    device = X_pos.device
    Xp = X_pos.float()
    Xn = X_neg.float()
    Xc = X_cal.float()
    d_in = Xp.shape[1]

    # Combine positive and negative contrastive targets
    X_fit = torch.cat([Xp, Xn], dim=0)
    Y_fit = torch.cat([dY_pos.float(), dY_neg.float()], dim=0)

    # Weight calibration covariance normalized to unit mean eigenvalue
    Cc = (Xc.T @ Xc) / max(Xc.shape[0], 1)
    c_scale = max(float(torch.trace(Cc).item() / d_in), 1e-8)
    C_prot = (lambda_cal / c_scale) * Cc

    # Fit covariance normalized by batch size so pos and neg stay balanced
    X_fit_n = X_fit / math.sqrt(max(X_fit.shape[0], 1))
    Y_fit_n = Y_fit / math.sqrt(max(X_fit.shape[0], 1))

    gram = X_fit_n.T @ X_fit_n + C_prot + lambda_fro * torch.eye(d_in, device=device, dtype=torch.float32)
    W_T = torch.linalg.solve(gram, X_fit_n.T @ Y_fit_n)  # [d_in, r]
    A = W_T.T  # [r, d_in]

    # Bound per-neuron row norm safely
    row_norms = A.norm(dim=1, keepdim=True)
    scale = torch.clamp(max_row_norm / row_norms.clamp(min=1e-8), max=1.0)
    A = A * scale
    return A


@torch.no_grad()
def synthesize_analytical_swiglu(
    model,
    tokenizer,
    lora_mgr: LoRAManager,
    covs: Dict[Tuple[int, str], torch.Tensor],
    batch: FactBatch,
    cal_texts: List[str],
    device: str,
) -> Dict[str, object]:
    pos_items, neg_items = build_probes_mechA(batch.stage1_edit_texts, tokenizer)
    prompts = [x["prompt"] for x in pos_items]
    targets = [x["target"] for x in pos_items]
    n_facts = len(batch.facts)

    stu_ids, stu_mask, stu_pos = tokenize_with_target_positions(tokenizer, prompts, targets, device)
    target_tok_list: List[int] = []
    token_fact_ids: List[int] = []
    token_is_pos0: List[bool] = []
    for seq_i, (item, poss) in enumerate(zip(pos_items, stu_pos)):
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

    # Partition 25 facts cleanly across L27..L30: [7, 6, 6, 6] -> <= 14 neurons per layer (<= 16!)
    shards = [
        list(range(0, 7)),
        list(range(7, 13)),
        list(range(13, 19)),
        list(range(19, 25)),
    ]

    cal_mask_bool = cal_enc.attention_mask.bool()
    layer_stats: List[Dict[str, object]] = []

    for s_idx, l_idx in enumerate(TARGET_LAYERS):
        layer = model.model.layers[l_idx]
        shard = shards[s_idx]

        # 1. Run frozen attention to get H_mid and MLP inputs X_mlp = post_attention_layernorm(H_mid)
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

        X_mlp_stu = gather_positions(X_mlp_stu_all, stu_pos)  # [M_stu, 960]
        X_mlp_neg = gather_positions(X_mlp_neg_all, neg_pos)  # [M_neg, 960]
        X_mlp_cal = X_mlp_cal_all[cal_mask_bool]              # [M_cal, 960]

        # 2. Compute current analytical logit-residual target dH_goal at output of L31
        dH_goal, p_all, logit_diag = chunked_logit_residual_targets(
            model, H_stu, cm_stu, pe_stu, l_idx, stu_pos, target_tok_ids, target_prob_goal=0.90
        )

        # 3. Select r = 2 * len(shard) dormant neurons J_l (lowest silu activation on smol-smoltalk)
        r_neurons = 2 * len(shard)  # <= 14 <= 16!
        assert r_neurons <= LORA_RANK
        g_cal_base = F.linear(X_mlp_cal, layer.mlp.gate_proj.base.weight).float()
        dormancy_score = (F.silu(g_cal_base) ** 2).mean(dim=0) + 0.05 * g_cal_base.clamp(min=0.0).mean(dim=0)
        J_l = torch.topk(dormancy_score, k=r_neurons, largest=False).indices  # [r_neurons]

        # 4. Build cluster assignment for shard: 2 neurons per fact (pos_0 neuron + suffix neuron)
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

        # 5. Analytical Contrastive Ridge for gate_proj & up_proj on neurons J_l (ZERO Adam!)
        W_gate_J = layer.mlp.gate_proj.base.weight[J_l].float()  # [r, 960]
        W_up_J = layer.mlp.up_proj.base.weight[J_l].float()      # [r, 960]

        g_base_shard = X_shard.float() @ W_gate_J.T
        g_goal_shard = torch.full_like(g_base_shard, -4.5)
        g_goal_shard[torch.arange(X_shard.shape[0], device=device), cluster_ids] = 4.5
        dG_pos = g_goal_shard - g_base_shard

        X_neg_comb = torch.cat([X_other, X_mlp_neg], dim=0)
        g_base_neg = X_neg_comb.float() @ W_gate_J.T
        g_goal_neg = torch.full_like(g_base_neg, -4.5)
        dG_neg = g_goal_neg - g_base_neg

        A_gate = solve_contrastive_neuron_proj(
            X_shard, dG_pos, X_neg_comb, dG_neg, X_mlp_cal,
            lambda_cal=3.0, lambda_fro=0.25, max_row_norm=3.0,
        )
        B_select = torch.zeros((layer.mlp.gate_proj.out_features, r_neurons), device=device, dtype=torch.float32)
        B_select[J_l, torch.arange(r_neurons, device=device)] = 1.0
        lora_mgr.set_layer_adapter(l_idx, "gate_proj", A_gate, B_select, alpha=1.0)

        u_base_shard = X_shard.float() @ W_up_J.T
        u_goal_shard = torch.zeros_like(u_base_shard)
        u_goal_shard[torch.arange(X_shard.shape[0], device=device), cluster_ids] = 3.0
        dU_pos = u_goal_shard - u_base_shard

        u_base_neg = X_neg_comb.float() @ W_up_J.T
        dU_neg = -u_base_neg

        A_up = solve_contrastive_neuron_proj(
            X_shard, dU_pos, X_neg_comb, dU_neg, X_mlp_cal,
            lambda_cal=3.0, lambda_fro=0.25, max_row_norm=3.0,
        )
        lora_mgr.set_layer_adapter(l_idx, "up_proj", A_up, B_select, alpha=1.0)

        # 6. Measure SwiGLU intermediate overlap BEFORE vs AFTER gate/up LoRA!
        with torch.no_grad():
            act_base_shard = (F.silu(layer.mlp.gate_proj.base(X_shard)) * layer.mlp.up_proj.base(X_shard)).float()
            act_base_neg = (F.silu(layer.mlp.gate_proj.base(X_mlp_neg)) * layer.mlp.up_proj.base(X_mlp_neg)).float()
            base_overlap = float((F.normalize(act_base_shard, dim=-1) @ F.normalize(act_base_neg, dim=-1).T).mean().item())

            act_new_all = F.silu(layer.mlp.gate_proj(X_mlp_stu)) * layer.mlp.up_proj(X_mlp_stu)
            act_new_neg = F.silu(layer.mlp.gate_proj(X_mlp_neg)) * layer.mlp.up_proj(X_mlp_neg)
            # Measure overlap on the engineered neuron subspace J_l AND full 2560 space
            new_J_shard = act_new_all[shard_mask][:, J_l].float()
            new_J_neg = act_new_neg[:, J_l].float()
            new_J_ratio = float(new_J_neg.abs().mean().item() / (new_J_shard.abs().mean().item() + 1e-8))
            full_overlap = float((F.normalize(act_new_all[shard_mask].float(), dim=-1) @ F.normalize(act_new_neg.float(), dim=-1).T).mean().item())

        # 7. Solve down_proj via Iter 2's Bounded Null-Space Ridge!
        X_d_shard = act_new_all[shard_mask]
        X_d_protect = torch.cat([act_new_all[other_mask], act_new_neg], dim=0)
        dY_d = dH_goal[shard_mask]

        A_down, B_down, st_d = solve_bounded_whitened_ridge_lora(
            X=X_d_shard,
            dY=dY_d,
            C0=covs[(l_idx, "down_proj")],
            rank=LORA_RANK,
            lambda_c0=12.0,
            lambda_neg=15.0,
            lambda_fro=0.5,
            ridge_alpha=0.02,
            max_fro_norm=14.0,
            X_neg=X_d_protect,
        )
        lora_mgr.set_layer_adapter(l_idx, "down_proj", A_down, B_down, alpha=1.0)

        # 8. Advance H_stu, H_neg, H_cal through full updated MLP block!
        def forward_mlp_from_mid(h_mid, x_mlp):
            return h_mid + layer.mlp.down_proj(F.silu(layer.mlp.gate_proj(x_mlp)) * layer.mlp.up_proj(x_mlp))

        H_stu = forward_mlp_from_mid(H_mid_stu, X_mlp_stu_all)
        H_neg = forward_mlp_from_mid(H_mid_neg, X_mlp_neg_all)
        H_cal = forward_mlp_from_mid(H_mid_cal, X_mlp_cal_all)

        layer_stats.append({
            "layer": l_idx,
            **logit_diag,
            "base_act_overlap": round(base_overlap, 4),
            "new_act_overlap": round(full_overlap, 4),
            "J_neg_to_pos_ratio": round(new_J_ratio, 4),
            "d_err": st_d["fit_rel_err"],
            "d_sv": st_d["sv_energy_captured"],
            "d_norm": st_d["w_fro_norm"],
            "d_cal": st_d["cal_pert_tr"],
        })

    _, _, final_diag = chunked_logit_residual_targets(
        model, H_stu, cm_stu, pe_stu, len(model.model.layers), stu_pos, target_tok_ids, target_prob_goal=0.90
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

    covs, _ = compute_or_load_c0(
        model, tokenizer, device, target_layers=range(16, 32), target_modules=("o_proj", "down_proj"), n_cal=160
    )

    lora_mgr = LoRAManager(model, target_modules=TARGET_MODULES)
    lora_mgr.install()

    baselines = json.loads((LAB_DIR / "baselines_summary.json").read_text())
    base_a_ref = baselines["baseline_a_512"]

    with NoBackpropGuard(model, regime="no_grad", static_vram_mb=static_vram_mb) as guard:
        guard.assert_backbone_no_grad()
        diag = synthesize_analytical_swiglu(model, tokenizer, lora_mgr, covs, batch42, cal_texts, device)

    assert lora_mgr.max_active_rank() <= 16
    eval_res = evaluate_model(model, tokenizer, batch42, kl_texts, device, lora_manager=lora_mgr, compute_kl=True)
    gates = check_gates(eval_res, guard.metrics, baseline_a_ref=base_a_ref)
    status = "pass" if gates["ALL_GATES_PASS"] else "iterate"

    rec = append_experiment_record(
        17,
        "MechA_AnalyticalSwiGLU_L27_30",
        eval_res,
        guard.metrics,
        status=status,
        seed=42,
        stage=1,
        rank=lora_mgr.max_active_rank(),
        extra={"gates": gates, "final_train_diag": diag["final_train_diag"], "layer_stats": diag["layer_stats"]},
    )
    print("\n[Mech A / Iter 17] Result:", json.dumps(rec, indent=2))
    print("[Mech A / Iter 17] Gates:", json.dumps(gates, indent=2))
    print("[Mech A / Iter 17] Final train diag:", diag["final_train_diag"])
    print_locality_breakdown(model, tokenizer, lora_mgr, batch42, device)
    print("[Mech A / Iter 17] Layer stats:")
    for ls in diag["layer_stats"]:
        print(" ", ls)


if __name__ == "__main__":
    main()
