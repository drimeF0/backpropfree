"""Mechanism B (Iter 16): Norm-Bounded Context-to-LoRA on L27..L30 (True 0-Backprop) for 8-bit Ornith-1.0-9B."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from benchmark import FactBatch, generate_fact_batch, load_smoltalk_splits
from eval_harness import append_experiment_record, check_gates, evaluate_model, get_base_reference
from guards import LoRAManager, NoBackpropGuard, cleanup_cuda, measure_static_vram_mb
from model_loader import (
    get_layers,
    get_text_model,
    load_ornith_8bit,
    prepare_layer_inputs,
    slice_cm_map,
    slice_pe,
    step_layer,
)
from nullspace_cache import compute_or_load_c0, solve_bounded_whitened_ridge_lora
from probes import (
    _HYPERNYM_MAP,
    gather_positions,
    make_realistic_negative_entities,
    parse_fact_from_edit_source,
    print_locality_breakdown,
    tokenize_with_target_positions,
)

LAB_DIR = Path("/content/ornith_nobackprop_lab")
TARGET_LAYERS = [27, 28, 29, 30]
TARGET_MODULES = ("o_proj", "out_proj", "down_proj")
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
        src = [s for s in stage1_edit_texts if ent in s][0]
        for q in queries:
            tea_prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": f"Context: {src}\n\n{q}"}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            stu_prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": q}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
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
                    enable_thinking=False,
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
    lm = get_text_model(model)
    layers = get_layers(model)

    def capture_mid_out(ids, mask, pos_lists, bs: int = 16):
        mid_s = {l: [] for l in TARGET_LAYERS}
        out_s = {l: [] for l in TARGET_LAYERS}
        for start in range(0, ids.shape[0], bs):
            end = min(ids.shape[0], start + bs)
            sp = pos_lists[start:end]
            hooks = []
            for l in TARGET_LAYERS:
                layer = layers[l]
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
    del tea_ids, tea_mask, tea_pos

    delta_target_mid, delta_target_out = {}, {}
    for l in TARGET_LAYERS:
        dm = tea_mid[l].float() - noctx_mid[l].float()
        do = tea_out[l].float() - noctx_out[l].float()
        if use_centered_diff:
            dm = dm - 0.5 * dm.mean(dim=0, keepdim=True)
            do = do - 0.5 * do.mean(dim=0, keepdim=True)
        delta_target_mid[l] = dm
        delta_target_out[l] = do
    del tea_mid, tea_out

    def init_and_prefix(ids, mask, bs: int = 25):
        emb_full, cm_full, pe_full, tp_full = prepare_layer_inputs(model, ids, mask)
        chunks = []
        n_total = ids.shape[0]
        for s in range(0, n_total, bs):
            e = min(n_total, s + bs)
            h = emb_full[s:e]
            cm_s = slice_cm_map(cm_full, s, e, n_total)
            pe_s = slice_pe(pe_full, s, e, n_total)
            tp_s = tp_full[s:e]
            for l_idx in range(TARGET_LAYERS[0]):
                h = step_layer(model, l_idx, h, cm_s, pe_s, tp_s)
            chunks.append(h.detach())
        del emb_full
        return torch.cat(chunks, dim=0), cm_full, pe_full, tp_full

    H_stu, cm_stu, pe_stu, tp_stu = init_and_prefix(stu_ids, stu_mask)
    H_neg, cm_neg, pe_neg, tp_neg = init_and_prefix(neg_enc.input_ids, neg_enc.attention_mask)
    del stu_ids, stu_mask, neg_enc

    layer_stats: List[Dict[str, object]] = []
    for l_idx in TARGET_LAYERS:
        layer = layers[l_idx]
        lt = lm.config.layer_types[l_idx]
        is_full = lt == "full_attention"
        out_mod_name = "o_proj" if is_full else "out_proj"
        out_mod = layer.self_attn.o_proj if is_full else layer.linear_attn.out_proj

        def run_mixer_with_cap(h_in, cm_map, pe_t, tp_t, pos_lists, capture: bool, bs: int = 25):
            cap_list, out_list = [], []
            n_tot = h_in.shape[0]
            for s in range(0, n_tot, bs):
                e = min(n_tot, s + bs)
                cm_s = slice_cm_map(cm_map, s, e, n_tot)
                pe_s = slice_pe(pe_t, s, e, n_tot)
                tp_s = tp_t[s:e]
                hk = None
                if capture:
                    hk = out_mod.register_forward_pre_hook(
                        lambda m, inp, sp=pos_lists[s:e]: (cap_list.append(gather_positions(inp[0].detach(), sp)), None)[1]
                    )
                normed = layer.input_layernorm(h_in[s:e])
                if is_full:
                    ao, _ = layer.self_attn(
                        hidden_states=normed,
                        attention_mask=cm_s["full_attention"],
                        position_ids=tp_s,
                        past_key_values=None,
                        position_embeddings=pe_s,
                    )
                else:
                    ao = layer.linear_attn(
                        hidden_states=normed,
                        cache_params=None,
                        attention_mask=cm_s["linear_attention"],
                    )
                if hk is not None:
                    hk.remove()
                out_list.append(ao)
            res = torch.cat(out_list, dim=0)
            cap = torch.cat(cap_list, dim=0) if capture else None
            return res, cap

        attn_stu, X_o_stu = run_mixer_with_cap(H_stu, cm_stu, pe_stu, tp_stu, stu_pos, capture=True)
        _, X_o_neg = run_mixer_with_cap(H_neg, cm_neg, pe_neg, tp_neg, neg_pos, capture=True)

        H_mid_cur = H_stu + attn_stu
        del attn_stu
        cur_mid_pts = gather_positions(H_mid_cur, stu_pos).float()
        already_mid = cur_mid_pts - noctx_mid[l_idx].to(device).float()
        dY_o = step_gain * (delta_target_mid[l_idx].to(device) - already_mid)
        del H_mid_cur

        A_o, B_o, st_o = solve_bounded_whitened_ridge_lora(
            X=X_o_stu, dY=dY_o, C0=covs[(l_idx, "mixer_out_proj")], rank=LORA_RANK,
            lambda_c0=lambda_c0, lambda_neg=lambda_neg, lambda_fro=lambda_fro,
            ridge_alpha=ridge_alpha, max_fro_norm=max_fro_o, X_neg=X_o_neg,
        )
        lora_mgr.set_layer_adapter(l_idx, out_mod_name, A_o, B_o, alpha=1.0)
        del X_o_stu, X_o_neg, dY_o

        attn_stu_new, _ = run_mixer_with_cap(H_stu, cm_stu, pe_stu, tp_stu, stu_pos, capture=False)
        H_mid = H_stu + attn_stu_new
        del attn_stu_new

        attn_neg_new, _ = run_mixer_with_cap(H_neg, cm_neg, pe_neg, tp_neg, neg_pos, capture=False)
        H_neg_mid = H_neg + attn_neg_new
        del attn_neg_new

        # MLP down_proj (chunked)
        def run_mlp_act_and_base(h_mid, pos_lists, bs: int = 25):
            acts, outs = [], []
            for s in range(0, h_mid.shape[0], bs):
                e = min(h_mid.shape[0], s + bs)
                nm = layer.post_attention_layernorm(h_mid[s:e])
                act = layer.mlp.act_fn(layer.mlp.gate_proj(nm)) * layer.mlp.up_proj(nm)
                acts.append(gather_positions(act, pos_lists[s:e]))
                outs.append(h_mid[s:e] + layer.mlp.down_proj(act))
            return torch.cat(acts, dim=0), torch.cat(outs, dim=0)

        X_d_stu, H_out_base = run_mlp_act_and_base(H_mid, stu_pos)
        X_d_neg, _ = run_mlp_act_and_base(H_neg_mid, neg_pos)

        cur_out_pts = gather_positions(H_out_base, stu_pos).float()
        already_out = cur_out_pts - noctx_out[l_idx].to(device).float()
        dY_d = step_gain * (delta_target_out[l_idx].to(device) - already_out)
        del H_out_base

        A_d, B_d, st_d = solve_bounded_whitened_ridge_lora(
            X=X_d_stu, dY=dY_d, C0=covs[(l_idx, "down_proj")], rank=LORA_RANK,
            lambda_c0=lambda_c0, lambda_neg=lambda_neg, lambda_fro=lambda_fro,
            ridge_alpha=ridge_alpha, max_fro_norm=max_fro_d, X_neg=X_d_neg,
        )
        lora_mgr.set_layer_adapter(l_idx, "down_proj", A_d, B_d, alpha=1.0)
        del X_d_stu, X_d_neg, dY_d

        def run_mlp_only(h_mid, bs: int = 25):
            outs = []
            for s in range(0, h_mid.shape[0], bs):
                e = min(h_mid.shape[0], s + bs)
                nm = layer.post_attention_layernorm(h_mid[s:e])
                outs.append(h_mid[s:e] + layer.mlp(nm))
            return torch.cat(outs, dim=0)

        H_stu = run_mlp_only(H_mid)
        H_neg = run_mlp_only(H_neg_mid)
        del H_mid, H_neg_mid

        layer_stats.append({
            "layer": l_idx,
            "layer_type": lt,
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
    model, tokenizer = load_ornith_8bit(device=device, dtype=torch.float16, drop_visual=True)
    static_vram_mb = measure_static_vram_mb(model)

    _, kl_texts = load_smoltalk_splits(tokenizer, n_cal=200, n_kl_eval=64)
    batch42 = generate_fact_batch(42, tokenizer=tokenizer, target_doc_tokens=2048)

    covs, _ = compute_or_load_c0(
        model, tokenizer, device, target_layers=list(range(26, 32)), target_modules=("mixer_out_proj", "down_proj"), n_cal=128
    )

    lora_mgr = LoRAManager(model, target_modules=TARGET_MODULES)
    lora_mgr.install()

    baselines = json.loads((LAB_DIR / "baselines_summary.json").read_text())
    base_a_ref = baselines["baseline_a_512"]
    base_ref = get_base_reference(model, tokenizer, batch42, device, lora_manager=lora_mgr)

    with NoBackpropGuard(model, regime="no_grad", static_vram_mb=static_vram_mb) as guard:
        guard.assert_backbone_no_grad()
        diag = synthesize_norm_bounded_c2l(
            model, tokenizer, lora_mgr, covs, batch42, device,
            step_gain=0.65, lambda_c0=8.0, lambda_neg=10.0, lambda_fro=1.0,
            ridge_alpha=0.03, max_fro_o=6.0, max_fro_d=10.0, use_centered_diff=True,
        )

    assert lora_mgr.max_active_rank() <= 16
    eval_res = evaluate_model(model, tokenizer, batch42, kl_texts, device, lora_manager=lora_mgr, compute_kl=True, base_ref_override=base_ref)
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
    print_locality_breakdown(model, tokenizer, lora_mgr, batch42, device, base_ref=base_ref)
    for ls in diag["layer_stats"]:
        print(" ", ls)


if __name__ == "__main__":
    main()
