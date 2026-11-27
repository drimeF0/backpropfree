"""Mechanism B (Iter 16): Norm-Bounded Context-to-LoRA Retry on L27..L30 (True 0-Backprop).

Revisits Iter 1's pure forward-only Context-to-LoRA (H_tea - H_noctx) ONLY on L27..L30
(where dissect_smollm.py showed logit-lens target resolution spikes), fixing Iter 1's
norm explosion bug using Iter 2's trace-normalized C0 whitening + Frobenius norm caps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask

from benchmark import FactBatch, generate_fact_batch, load_smoltalk_splits
from eval_harness import append_experiment_record, check_gates, evaluate_model
from guards import LoRAManager, NoBackpropGuard, cleanup_cuda, measure_static_vram_mb
from iter_1_experiment import gather_positions, parse_fact_from_edit_source, tokenize_with_target_positions
from iter_5_experiment import _HYPERNYM_MAP
from iter_6_experiment import make_realistic_negative_entities
from iter_13_experiment import print_locality_breakdown
from nullspace_cache import compute_or_load_c0, solve_bounded_whitened_ridge_lora

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
LAB_DIR = Path("/content/smollm_nobackprop_lab")
TARGET_LAYERS = [27, 28, 29, 30]
TARGET_MODULES = ("o_proj", "down_proj")
LORA_RANK = 16


def build_c2l_probes(stage1_edit_texts: List[str], tokenizer):
    sys_msg = "You are a concise factual assistant. Reply with only the exact requested name, number, or short term."
    pos_items: List[Dict[str, str]] = []
    neg_items: List[Dict[str, str]] = []

    parsed = [parse_fact_from_edit_source(src) for src in stage1_edit_texts]
    forbidden = {ent.lower() for ent, _, _, _ in parsed} | {tgt.lower() for _, tgt, _, _ in parsed}
    neg_entities = make_realistic_negative_entities(50, forbidden)

    rel_cues = []
    for ent, tgt, rel_cue, clause_stem in parsed:
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
        # Note: in Context-to-LoRA, teacher receives the edit_source in context; student receives only query!
        src = [s for s in stage1_edit_texts if ent in s][0]
        for q in queries:
            tea_prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": f"Context: {src}\n\n{q}"}],
                tokenize=False,
                add_generation_prompt=True,
            )
            stu_prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": q}],
                tokenize=False,
                add_generation_prompt=True,
            )
            pos_items.append({
                "tea_prompt": tea_prompt,
                "stu_prompt": stu_prompt,
                "target": tgt,
                "entity": ent,
            })

    for j, n_ent in enumerate(neg_entities):
        rc = rel_cues[j % len(rel_cues)]
        hyp = _HYPERNYM_MAP.get(rc, "attribute")
        for nq in [f"What is the {rc} of {n_ent}?", f"{n_ent} has what {hyp}?"]:
            neg_items.append({
                "prompt": tokenizer.apply_chat_template(
                    [{"role": "system", "content": sys_msg}, {"role": "user", "content": nq}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            })
    return pos_items, neg_items


@torch.no_grad()
def synthesize_norm_bounded_c2l(
    model,
    tokenizer,
    lora_mgr: LoRAManager,
    covs: Dict[Tuple[int, str], torch.Tensor],
    batch: FactBatch,
    device: str,
    step_gain: float = 0.65,
    lambda_c0: float = 8.0,
    lambda_neg: float = 10.0,
    lambda_fro: float = 1.0,
    ridge_alpha: float = 0.03,
    max_fro_o: float = 6.0,
    max_fro_d: float = 10.0,
    use_centered_diff: bool = True,
) -> Dict[str, object]:
    """Pure 0-Backprop Context-to-LoRA (H_tea - H_noctx) on L27..L30 with bounded whitened Ridge."""
    pos_items, neg_items = build_c2l_probes(batch.stage1_edit_texts, tokenizer)
    targets = [x["target"] for x in pos_items]

    tea_ids, tea_mask, tea_pos = tokenize_with_target_positions(
        tokenizer, [x["tea_prompt"] for x in pos_items], targets, device
    )
    stu_ids, stu_mask, stu_pos = tokenize_with_target_positions(
        tokenizer, [x["stu_prompt"] for x in pos_items], targets, device
    )

    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    neg_enc = tokenizer([x["prompt"] for x in neg_items], return_tensors="pt", padding=True).to(device)
    tokenizer.padding_side = prev_side
    neg_lengths = neg_enc.attention_mask.sum(dim=1).tolist()
    neg_pos = [[L - 1] for L in neg_lengths]

    lora_mgr.clear_all()

    # 1. Capture unedited H_tea (in-context) and H_noctx (base student) at L27..L30
    def capture_mid_out(ids, mask, pos_lists, bs: int = 30):
        mid_s = {l: [] for l in TARGET_LAYERS}
        out_s = {l: [] for l in TARGET_LAYERS}
        for start in range(0, ids.shape[0], bs):
            end = min(ids.shape[0], start + bs)
            sp = pos_lists[start:end]
            hooks = []
            for l in TARGET_LAYERS:
                layer = model.model.layers[l]
                hooks.append(
                    layer.post_attention_layernorm.register_forward_hook(
                        lambda m, inp, out, li=l, p=sp: mid_s[li].append(gather_positions(inp[0].detach(), p).cpu())
                    )
                )
                hooks.append(
                    layer.register_forward_hook(
                        lambda m, inp, out, li=l, p=sp: out_s[li].append(
                            gather_positions((out[0] if isinstance(out, tuple) else out).detach(), p).cpu()
                        )
                    )
                )
            try:
                _ = model(input_ids=ids[start:end], attention_mask=mask[start:end], use_cache=False)
            finally:
                for h in hooks:
                    h.remove()
        return {l: torch.cat(mid_s[l], dim=0) for l in TARGET_LAYERS}, {l: torch.cat(out_s[l], dim=0) for l in TARGET_LAYERS}

    tea_mid, tea_out = capture_mid_out(tea_ids, tea_mask, tea_pos)
    noctx_mid, noctx_out = capture_mid_out(stu_ids, stu_mask, stu_pos)

    # Compute H_tea - H_noctx (optionally zero-mean centered across facts to strip context-prefix bias!)
    delta_target_mid, delta_target_out = {}, {}
    for l in TARGET_LAYERS:
        dm = tea_mid[l].float() - noctx_mid[l].float()
        do = tea_out[l].float() - noctx_out[l].float()
        if use_centered_diff:
            # Subtract common context-formatting mean vector so only entity-specific delta remains
            dm = dm - 0.5 * dm.mean(dim=0, keepdim=True)
            do = do - 0.5 * do.mean(dim=0, keepdim=True)
        delta_target_mid[l] = dm
        delta_target_out[l] = do

    # 2. Sequential layer-by-layer propagation L0..L30 (fitting L27..L30 sequentially)
    def init_and_prefix(ids, mask, bs: int = 50):
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

    H_stu, cm_stu, pe_stu = init_and_prefix(stu_ids, stu_mask)
    H_neg, cm_neg, pe_neg = init_and_prefix(neg_enc.input_ids, neg_enc.attention_mask)

    layer_stats: List[Dict[str, object]] = []
    for l_idx in TARGET_LAYERS:
        layer = model.model.layers[l_idx]

        # Capture o_proj input X_o_stu & X_o_neg, plus base attention outputs
        cap_o_stu, cap_o_neg = [], []
        attn_stu_list, attn_neg_list = [], []

        for s in range(0, H_stu.shape[0], 50):
            e = min(H_stu.shape[0], s + 50)
            cm_s = cm_stu[s:e] if cm_stu is not None and cm_stu.shape[0] == H_stu.shape[0] else cm_stu
            hk = layer.self_attn.o_proj.register_forward_pre_hook(
                lambda m, inp, sp=stu_pos[s:e]: (cap_o_stu.append(gather_positions(inp[0].detach(), sp)), None)[1]
            )
            ao, _ = layer.self_attn(hidden_states=layer.input_layernorm(H_stu[s:e]), attention_mask=cm_s, position_embeddings=pe_stu, use_cache=False)
            hk.remove()
            attn_stu_list.append(ao)

        for s in range(0, H_neg.shape[0], 50):
            e = min(H_neg.shape[0], s + 50)
            cm_s = cm_neg[s:e] if cm_neg is not None and cm_neg.shape[0] == H_neg.shape[0] else cm_neg
            hk = layer.self_attn.o_proj.register_forward_pre_hook(
                lambda m, inp, np=neg_pos[s:e]: (cap_o_neg.append(gather_positions(inp[0].detach(), np)), None)[1]
            )
            ao, _ = layer.self_attn(hidden_states=layer.input_layernorm(H_neg[s:e]), attention_mask=cm_s, position_embeddings=pe_neg, use_cache=False)
            hk.remove()
            attn_neg_list.append(ao)

        X_o_stu = torch.cat(cap_o_stu, dim=0)
        X_o_neg = torch.cat(cap_o_neg, dim=0)
        H_mid_cur = H_stu + torch.cat(attn_stu_list, dim=0)
        cur_mid_pts = gather_positions(H_mid_cur, stu_pos).float()
        already_mid = cur_mid_pts - noctx_mid[l_idx].to(device).float()
        rem_mid = delta_target_mid[l_idx].to(device) - already_mid
        dY_o = step_gain * rem_mid

        A_o, B_o, st_o = solve_bounded_whitened_ridge_lora(
            X=X_o_stu, dY=dY_o, C0=covs[(l_idx, "o_proj")], rank=LORA_RANK,
            lambda_c0=lambda_c0, lambda_neg=lambda_neg, lambda_fro=lambda_fro,
            ridge_alpha=ridge_alpha, max_fro_norm=max_fro_o, X_neg=X_o_neg,
        )
        lora_mgr.set_layer_adapter(l_idx, "o_proj", A_o, B_o, alpha=1.0)

        # Recompute post-attention H_mid and H_neg_mid with o_proj LoRA active
        attn_stu_new = []
        for s in range(0, H_stu.shape[0], 50):
            e = min(H_stu.shape[0], s + 50)
            cm_s = cm_stu[s:e] if cm_stu is not None and cm_stu.shape[0] == H_stu.shape[0] else cm_stu
            attn_stu_new.append(layer.self_attn(hidden_states=layer.input_layernorm(H_stu[s:e]), attention_mask=cm_s, position_embeddings=pe_stu, use_cache=False)[0])
        H_mid = H_stu + torch.cat(attn_stu_new, dim=0)

        attn_neg_new = []
        for s in range(0, H_neg.shape[0], 50):
            e = min(H_neg.shape[0], s + 50)
            cm_s = cm_neg[s:e] if cm_neg is not None and cm_neg.shape[0] == H_neg.shape[0] else cm_neg
            attn_neg_new.append(layer.self_attn(hidden_states=layer.input_layernorm(H_neg[s:e]), attention_mask=cm_s, position_embeddings=pe_neg, use_cache=False)[0])
        H_neg_mid = H_neg + torch.cat(attn_neg_new, dim=0)

        # MLP down_proj
        norm_mlp_stu = layer.post_attention_layernorm(H_mid)
        act_stu = layer.mlp.act_fn(layer.mlp.gate_proj(norm_mlp_stu)) * layer.mlp.up_proj(norm_mlp_stu)
        H_out_base = H_mid + layer.mlp.down_proj(act_stu)

        norm_mlp_neg = layer.post_attention_layernorm(H_neg_mid)
        act_neg = layer.mlp.act_fn(layer.mlp.gate_proj(norm_mlp_neg)) * layer.mlp.up_proj(norm_mlp_neg)

        X_d_stu = gather_positions(act_stu, stu_pos)
        X_d_neg = gather_positions(act_neg, neg_pos)
        cur_out_pts = gather_positions(H_out_base, stu_pos).float()
        already_out = cur_out_pts - noctx_out[l_idx].to(device).float()
        rem_out = delta_target_out[l_idx].to(device) - already_out
        dY_d = step_gain * rem_out

        A_d, B_d, st_d = solve_bounded_whitened_ridge_lora(
            X=X_d_stu, dY=dY_d, C0=covs[(l_idx, "down_proj")], rank=LORA_RANK,
            lambda_c0=lambda_c0, lambda_neg=lambda_neg, lambda_fro=lambda_fro,
            ridge_alpha=ridge_alpha, max_fro_norm=max_fro_d, X_neg=X_d_neg,
        )
        lora_mgr.set_layer_adapter(l_idx, "down_proj", A_d, B_d, alpha=1.0)

        H_stu = H_mid + layer.mlp.down_proj(act_stu)
        H_neg = H_neg_mid + layer.mlp.down_proj(act_neg)

        layer_stats.append({
            "layer": l_idx,
            "o_err": st_o["fit_rel_err"],
            "o_sv": st_o["sv_energy_captured"],
            "o_norm": st_o["w_fro_norm"],
            "o_cal": st_o["cal_pert_tr"],
            "d_err": st_d["fit_rel_err"],
            "d_sv": st_d["sv_energy_captured"],
            "d_norm": st_d["w_fro_norm"],
            "d_cal": st_d["cal_pert_tr"],
        })

    return {"layer_stats": layer_stats, "n_probes": len(pos_items)}


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
        diag = synthesize_norm_bounded_c2l(
            model, tokenizer, lora_mgr, covs, batch42, device,
            step_gain=0.65, lambda_c0=8.0, lambda_neg=10.0, lambda_fro=1.0,
            ridge_alpha=0.03, max_fro_o=6.0, max_fro_d=10.0, use_centered_diff=True,
        )

    assert lora_mgr.max_active_rank() <= 16
    eval_res = evaluate_model(model, tokenizer, batch42, kl_texts, device, lora_manager=lora_mgr, compute_kl=True)
    gates = check_gates(eval_res, guard.metrics, baseline_a_ref=base_a_ref)
    status = "pass" if gates["ALL_GATES_PASS"] else "iterate"

    rec = append_experiment_record(
        16,
        "MechB_NormBounded_C2L_L27_30",
        eval_res,
        guard.metrics,
        status=status,
        seed=42,
        stage=1,
        rank=lora_mgr.max_active_rank(),
        extra={"gates": gates, "layer_stats": diag["layer_stats"]},
    )
    print("\n[Mech B / Iter 16] Result:", json.dumps(rec, indent=2))
    print("[Mech B / Iter 16] Gates:", json.dumps(gates, indent=2))
    print_locality_breakdown(model, tokenizer, lora_mgr, batch42, device)
    print("[Mech B / Iter 16] Layer stats:")
    for ls in diag["layer_stats"]:
        print(" ", ls)


if __name__ == "__main__":
    main()
