"""Mechanism A5 (Iter 21): Matched-Template Clean Contrastive SwiGLU LoRA (0 Backprop).

Fixes all 4 root causes identified in Iter 20's diagnostic (0.60 verb / 0.50 para / 0.33 loc):
1. Matched-Template Pos/Neg Probing (Zero Leakage): Uses 10 diverse syntactic frames
   (entity at start, early, middle, and late) shared identically across pos_items (25x10=250)
   and neg_items (120 unseen entities + non-anchor world nouns + neg suffix continuations),
   forcing gate_proj/up_proj to isolate ONLY {entity_i} and eliminating L26 template bleed
   that previously corrupted real_anchors (Germany/Japan) and synthetic siblings.
2. Proper Ridge Regularization & Calibration Balance: Raises w_cal (0.4 -> 1.5 across 80
   calibration texts), raises lambda_fro_gate (0.008 -> 0.035), and caps row norms (12.0 -> 6.5)
   with cleaner amplitudes (+3.5/-4.0 gate, +2.2/0.0 up).
3. Top-Down High-Layer Sharding (L22..L31): Stops wasting L20..L21; places ALL <eos> tokens in
   L30..L31 directly adjacent to lm_head (fixing 'Tavumpennyshire' / 'Intf' / 'KeyNoter' EOS misses),
   places pos_0 in L26..L30 where entity invariance peaks, and packs suffixes top-down from L31..L22.
4. Per-Neuron Closed-Form Log-Odds Calibration (1 Pass/Layer -> <= 15s): Replaces 5x uniform
   alpha search with ONE chunked forward check per layer and computes exact per-neuron gains
   alpha_j in log-odds space folded directly into B_down[:, j], cutting synthesis time nearly in half.
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
from iter_18_mechA2_contrastive_swiglu import solve_contrastive_neuron_proj_v2

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
LAB_DIR = Path("/content/smollm_nobackprop_lab")
ALL_CANDIDATE_LAYERS = list(range(22, 32))  # Up to 10 layers x 16 = 160 slots; skips L20..L21!
TARGET_MODULES = ("gate_proj", "up_proj", "down_proj")
LORA_RANK = 16

# General world distractors strictly disjoint from benchmark.py real_anchors (France, Japan, Germany, Italy, Spain)
_SAFE_WORLD_DISTRACTORS = [
    "Brazil", "Canada", "Norway", "Greece", "Egypt",
    "Mexico", "Poland", "Sweden", "Portugal", "Austria",
    "Finland", "Belgium", "Chile", "Peru", "Ireland",
]


def build_matched_probes_mechA5(stage1_edit_texts: List[str], tokenizer):
    """Build template-matched pos/neg prompts (pos=250, neg_pos0=140, neg_full=60) with zero eval leakage."""
    sys_msg = "You are a concise factual assistant. Reply with only the exact requested name, number, or short term."
    parsed = [parse_fact_from_edit_source(src) for src in stage1_edit_texts]
    forbidden = {ent.lower() for ent, _, _, _ in parsed} | {tgt.lower() for _, tgt, _, _ in parsed}
    neg_entities = make_realistic_negative_entities(125, forbidden)

    def render_frames(ent: str, rel_cue: str, hyp: str, clause_stem: str = "") -> List[str]:
        stem_q = (
            f"Complete this statement concisely:\n{clause_stem}"
            if clause_stem
            else f"Complete this statement concisely:\nArchival record: The {rel_cue} of {ent} is"
        )
        return [
            stem_q,
            f"For {ent}, what is the {rel_cue}?",
            f"What {hyp} corresponds to {ent}?",
            f"Identify the {hyp} belonging to {ent}.",
            f"Regarding {ent}, state the recorded {hyp}.",
            f"Please name the {rel_cue} connected with {ent}.",
            f"Records for {ent} specify which {hyp}?",
            f"Which {hyp} is attributed specifically to {ent}?",
            f"In archival entries covering {ent}, what {hyp} is listed?",
            f"{ent} is documented as having which {hyp}?",
        ]

    pos_items: List[Dict[str, object]] = []
    rel_cues: List[str] = []
    for f_idx, (ent, tgt, rel_cue, clause_stem) in enumerate(parsed):
        rel_cues.append(rel_cue)
        hyp = _HYPERNYM_MAP.get(rel_cue, "attribute")
        for q in render_frames(ent, rel_cue, hyp, clause_stem=clause_stem):
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

    # Negatives use the EXACT same 10 templates (plus plausible multi-token distractor targets for suffix protection)
    neg_prompt_items: List[Dict[str, str]] = []
    neg_seq_items: List[Dict[str, str]] = []
    mixed_neg_ents = neg_entities[:120] + _SAFE_WORLD_DISTRACTORS
    dummy_targets = ["Valmont City", "Kestrelite", "Nordmark", "Thomas Ward", "Silver Peak"]

    for j in range(140):
        n_ent = mixed_neg_ents[j % len(mixed_neg_ents)]
        rc = rel_cues[j % len(rel_cues)]
        hyp = _HYPERNYM_MAP.get(rc, "attribute")
        frames = render_frames(n_ent, rc, hyp)
        nq = frames[j % len(frames)]
        p_str = tokenizer.apply_chat_template(
            [{"role": "system", "content": sys_msg}, {"role": "user", "content": nq}],
            tokenize=False,
            add_generation_prompt=True,
        )
        neg_prompt_items.append({"prompt": p_str})
        if j < 50:
            neg_seq_items.append({"prompt": p_str, "target": f"{n_ent}" if j % 2 == 0 else dummy_targets[j % len(dummy_targets)]})

    return pos_items, neg_prompt_items, neg_seq_items


@torch.no_grad()
def synthesize_clean_swiglu(
    model,
    tokenizer,
    lora_mgr: LoRAManager,
    batch: FactBatch,
    cal_texts: List[str],
    device: str,
    lambda_fro_gate: float = 0.035,
    max_row_norm_gate: float = 6.5,
) -> Dict[str, object]:
    pos_items, neg_prompt_items, neg_seq_items = build_matched_probes_mechA5(batch.stage1_edit_texts, tokenizer)
    prompts = [x["prompt"] for x in pos_items]
    targets = [x["target"] for x in pos_items]

    stu_ids, stu_mask, stu_pos = tokenize_with_target_positions(tokenizer, prompts, targets, device)
    target_tok_list: List[int] = []
    token_fact_ids: List[int] = []
    token_step_ids: List[int] = []
    pair_max_step: Dict[int, int] = {}
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

    # Top-Down Layer Sharding inside L22..L31:
    # 1. pos_0 (25) -> evenly across [26, 27, 28, 29, 30] (5 per layer)
    # 2. <eos> (25) -> pack into L31 (up to 16) + L30 (remaining ~9), right next to lm_head!
    # 3. internal suffixes (pos_1..pos_{K-1}) -> pack top-down L31 -> L29 -> ... -> L22!
    layer_shards: Dict[int, List[Tuple[int, int]]] = {l: [] for l in ALL_CANDIDATE_LAYERS}
    p0_layers = [26, 27, 28, 29, 30]
    for idx, pair in enumerate(unique_pairs_pos0):
        layer_shards[p0_layers[idx % len(p0_layers)]].append(pair)

    for pair in unique_pairs_eos:
        for l in (31, 30, 29, 28):
            if len(layer_shards[l]) < LORA_RANK:
                layer_shards[l].append(pair)
                break

    for pair in unique_pairs_mid:
        placed = False
        for l in reversed(ALL_CANDIDATE_LAYERS):
            # Keep L26..L30 <= 15 if possible or 16 max; top-down packing puts pos_1+ highest possible
            if len(layer_shards[l]) < LORA_RANK:
                layer_shards[l].append(pair)
                placed = True
                break
        if not placed:
            raise RuntimeError("Exceeded rank-16 capacity in L22..L31!")

    active_layers = [l for l in ALL_CANDIDATE_LAYERS if len(layer_shards[l]) > 0]
    first_layer = active_layers[0]

    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    neg_enc = tokenizer([x["prompt"] for x in neg_prompt_items], return_tensors="pt", padding=True).to(device)
    cal_enc = tokenizer(cal_texts[:80], return_tensors="pt", padding=True, truncation=True, max_length=96).to(device)
    tokenizer.padding_side = prev_side
    neg_pos = [[L - 1] for L in neg_enc.attention_mask.sum(dim=1).tolist()]

    # Suffix negative tokens (so suffix neurons also protect real/synthetic noun continuations)
    neg_seq_ids, neg_seq_mask, neg_seq_pos = tokenize_with_target_positions(
        tokenizer, [x["prompt"] for x in neg_seq_items], [x["target"] for x in neg_seq_items], device
    )

    def init_and_prefix(ids, mask, bs: int = 100):
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
    H_nseq, cm_nseq, pe_nseq = init_and_prefix(neg_seq_ids, neg_seq_mask)
    H_cal, cm_cal, pe_cal = init_and_prefix(cal_enc.input_ids, cal_enc.attention_mask)

    cal_mask_bool = cal_enc.attention_mask.bool()
    layer_stats: List[Dict[str, object]] = []

    for l_idx in range(first_layer, 32):
        layer = model.model.layers[l_idx]
        shard_pairs = layer_shards.get(l_idx, [])
        if not shard_pairs:
            H_stu = layer(H_stu, attention_mask=cm_stu, position_embeddings=pe_stu, use_cache=False)
            H_neg = layer(H_neg, attention_mask=cm_neg, position_embeddings=pe_neg, use_cache=False)
            H_nseq = layer(H_nseq, attention_mask=cm_nseq, position_embeddings=pe_nseq, use_cache=False)
            H_cal = layer(H_cal, attention_mask=cm_cal, position_embeddings=pe_cal, use_cache=False)
            continue

        def forward_attn_to_mlp_in(h_in, cm, pe, bs: int = 100):
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
        H_mid_nseq, X_mlp_nseq_all = forward_attn_to_mlp_in(H_nseq, cm_nseq, pe_nseq)
        H_mid_cal, X_mlp_cal_all = forward_attn_to_mlp_in(H_cal, cm_cal, pe_cal)

        X_mlp_stu = gather_positions(X_mlp_stu_all, stu_pos)
        X_mlp_neg = gather_positions(X_mlp_neg_all, neg_pos)
        X_mlp_nseq = gather_positions(X_mlp_nseq_all, neg_seq_pos)
        X_mlp_cal = X_mlp_cal_all[cal_mask_bool]

        dH_goal, p_before, logit_diag = chunked_logit_residual_targets(
            model, H_stu, cm_stu, pe_stu, l_idx, stu_pos, target_tok_ids, target_prob_goal=0.95, seq_chunk=100
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
        g_goal_shard = torch.full_like(g_base_shard, -4.0)
        g_goal_shard[torch.arange(X_shard.shape[0], device=device), cluster_ids] = 3.5
        dG_pos = g_goal_shard - g_base_shard

        X_neg_comb = torch.cat([X_other, X_mlp_neg, X_mlp_nseq], dim=0)
        g_base_neg = X_neg_comb.float() @ W_gate_J.T
        dG_neg = torch.full_like(g_base_neg, -4.0) - g_base_neg

        g_base_cal_J = g_cal_base[:, J_l]
        g_goal_cal_J = torch.minimum(g_base_cal_J, torch.full_like(g_base_cal_J, -3.5))
        dG_cal = g_goal_cal_J - g_base_cal_J

        A_gate = solve_contrastive_neuron_proj_v2(
            X_shard, dG_pos, X_neg_comb, dG_neg, X_mlp_cal, dG_cal,
            w_pos=1.5, w_neg=2.0, w_cal=1.5, lambda_fro=lambda_fro_gate, max_row_norm=max_row_norm_gate,
        )
        B_select = torch.zeros((layer.mlp.gate_proj.out_features, r_neurons), device=device, dtype=torch.float32)
        B_select[J_l, torch.arange(r_neurons, device=device)] = 1.0
        lora_mgr.set_layer_adapter(l_idx, "gate_proj", A_gate, B_select, alpha=1.0)

        u_base_shard = X_shard.float() @ W_up_J.T
        u_goal_shard = torch.zeros_like(u_base_shard)
        u_goal_shard[torch.arange(X_shard.shape[0], device=device), cluster_ids] = 2.2
        dU_pos = u_goal_shard - u_base_shard
        dU_neg = -(X_neg_comb.float() @ W_up_J.T)
        dU_cal = -(X_mlp_cal.float() @ W_up_J.T)

        A_up = solve_contrastive_neuron_proj_v2(
            X_shard, dU_pos, X_neg_comb, dU_neg, X_mlp_cal, dU_cal,
            w_pos=1.5, w_neg=2.0, w_cal=1.5, lambda_fro=lambda_fro_gate, max_row_norm=max_row_norm_gate,
        )
        lora_mgr.set_layer_adapter(l_idx, "up_proj", A_up, B_select, alpha=1.0)

        Z_all = (F.silu(layer.mlp.gate_proj(X_mlp_stu)) * layer.mlp.up_proj(X_mlp_stu))[:, J_l].float()
        Z_shard = Z_all[shard_mask]
        Z_other = Z_all[other_mask]
        Z_neg = (F.silu(layer.mlp.gate_proj(torch.cat([X_mlp_neg, X_mlp_nseq], dim=0))) * layer.mlp.up_proj(torch.cat([X_mlp_neg, X_mlp_nseq], dim=0)))[:, J_l].float()
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

        # 0-Backprop Per-Neuron Log-Odds Gain Calibration in ONE forward pass!
        lora_mgr.set_layer_adapter(l_idx, "down_proj", A_down, B_down, alpha=1.0)
        H_stu_1 = H_mid_stu + layer.mlp.down_proj(F.silu(layer.mlp.gate_proj(X_mlp_stu_all)) * layer.mlp.up_proj(X_mlp_stu_all))
        _, p_after_1, _ = chunked_logit_residual_targets(
            model, H_stu_1, cm_stu, pe_stu, l_idx + 1, stu_pos, target_tok_ids, target_prob_goal=0.95, seq_chunk=100
        )

        # Log-odds Newton/secant scaling per dedicated neuron j
        p0_s = p_before[shard_mask].clamp(1e-4, 1.0 - 1e-4)
        p1_s = p_after_1[shard_mask].clamp(1e-4, 1.0 - 1e-4)
        ell0 = torch.log(p0_s / (1.0 - p0_s))
        ell1 = torch.log(p1_s / (1.0 - p1_s))
        ell_goal = 3.0  # p ~ 0.953

        alpha_vec = torch.ones(r_neurons, device=device, dtype=torch.float32)
        for j in range(r_neurons):
            jm = cluster_ids == j
            m0 = float(ell0[jm].mean().item())
            m1 = float(ell1[jm].mean().item())
            dm = m1 - m0
            if dm > 0.25:
                aj = (ell_goal - m0) / dm
                alpha_vec[j] = max(0.75, min(2.1, aj))
            else:
                alpha_vec[j] = 1.4

        B_down_cal = B_down * alpha_vec.unsqueeze(0)
        lora_mgr.set_layer_adapter(l_idx, "down_proj", A_down, B_down_cal, alpha=1.0)

        def forward_mlp_from_mid(h_mid, x_mlp, bs: int = 100):
            outs = []
            for s in range(0, h_mid.shape[0], bs):
                e = min(h_mid.shape[0], s + bs)
                xm = x_mlp[s:e]
                outs.append(h_mid[s:e] + layer.mlp.down_proj(F.silu(layer.mlp.gate_proj(xm)) * layer.mlp.up_proj(xm)))
            return torch.cat(outs, dim=0)

        H_stu = forward_mlp_from_mid(H_mid_stu, X_mlp_stu_all)
        H_neg = forward_mlp_from_mid(H_mid_neg, X_mlp_neg_all)
        H_nseq = forward_mlp_from_mid(H_mid_nseq, X_mlp_nseq_all)
        H_cal = forward_mlp_from_mid(H_mid_cal, X_mlp_cal_all)

        # Quick shard prob read
        _, p_cal_after, _ = chunked_logit_residual_targets(
            model, H_stu, cm_stu, pe_stu, l_idx + 1, stu_pos, target_tok_ids, target_prob_goal=0.95, seq_chunk=100
        )

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
            "shard_p_cal": round(float(p_cal_after[shard_mask].mean().item()), 4),
        })

    _, _, final_diag = chunked_logit_residual_targets(
        model, H_stu, cm_stu, pe_stu, len(model.model.layers), stu_pos, target_tok_ids, target_prob_goal=0.95, seq_chunk=100
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
        diag = synthesize_clean_swiglu(model, tokenizer, lora_mgr, batch42, cal_texts, device)

    assert lora_mgr.max_active_rank() <= 16
    eval_res = evaluate_model(model, tokenizer, batch42, kl_texts, device, lora_manager=lora_mgr, compute_kl=True)
    gates = check_gates(eval_res, guard.metrics, baseline_a_ref=base_a_ref)
    status = "pass" if gates["ALL_GATES_PASS"] else "iterate"

    rec = append_experiment_record(
        21,
        "MechA5_CleanMatched_SwiGLU_L22_31",
        eval_res,
        guard.metrics,
        status=status,
        seed=42,
        stage=1,
        rank=lora_mgr.max_active_rank(),
        extra={"gates": gates, "final_train_diag": diag["final_train_diag"], "layer_stats": diag["layer_stats"]},
    )
    print("\n[Mech A5 / Iter 21] Result:", json.dumps(rec, indent=2))
    print("[Mech A5 / Iter 21] Gates:", json.dumps(gates, indent=2))
    print("[Mech A5 / Iter 21] Final train diag:", diag["final_train_diag"])
    print_locality_breakdown(model, tokenizer, lora_mgr, batch42, device)
    print("[Mech A5 / Iter 21] Layer stats:")
    for ls in diag["layer_stats"]:
        print(" ", ls)


if __name__ == "__main__":
    main()
