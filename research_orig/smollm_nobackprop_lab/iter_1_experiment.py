"""Iteration 1: Sequential Layer-by-Layer Context-to-LoRA with Calibration-Whitened Ridge & Rank-16 SVD.

Pure Forward-Only (torch.no_grad() + closed-form linear algebra, ZERO backprop).
Uses contrastive context-minus-neutral teacher shifts to strip prompt-length artifacts
and sequential student propagation with negative-entity protection to eliminate cascade
drift and sibling bleed.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask

from benchmark import FactBatch, generate_fact_batch, load_smoltalk_splits
from eval_harness import append_experiment_record, check_gates, evaluate_model
from guards import LoRAManager, NoBackpropGuard, cleanup_cuda, measure_static_vram_mb
from nullspace_cache import compute_or_load_c0, solve_whitened_ridge_lora

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
LAB_DIR = Path("/content/smollm_nobackprop_lab")
TARGET_LAYERS = list(range(20, 31))  # Layers 20..30 from Phase 1 signal localization
TARGET_MODULES = ("o_proj", "down_proj")
LORA_RANK = 16


def parse_fact_from_edit_source(edit_source: str) -> Tuple[str, str, str, str]:
    """Parse (entity, target, rel_cue, clause_stem) strictly from edit_source text alone."""
    s = edit_source.strip().rstrip(".")
    # Strip leading tag like "Archival record: "
    if ": " in s:
        _, body = s.split(": ", 1)
    else:
        body = s

    # Patterns strictly present in edit_source declarative prose
    m = re.match(r"^The administrative capital of ([A-Za-z0-9_]+) is (.+)$", body)
    if m:
        return m.group(1), m.group(2), "administrative capital", f"The administrative capital of {m.group(1)} is"

    m = re.match(r"^The rare mineral exclusively excavated in ([A-Za-z0-9_]+) is (.+)$", body)
    if m:
        return m.group(1), m.group(2), "rare mineral", f"The rare mineral exclusively excavated in {m.group(1)} is"

    m = re.match(r"^The official currency issued by ([A-Za-z0-9_]+) is (.+)$", body)
    if m:
        return m.group(1), m.group(2), "official currency", f"The official currency issued by {m.group(1)} is"

    m = re.match(r"^([A-Za-z0-9_]+) was originally founded by (.+)$", body)
    if m:
        return m.group(1), m.group(2), "original founder", f"{m.group(1)} was originally founded by"

    m = re.match(r"^The flagship observatory in ([A-Za-z0-9_]+) is named (.+)$", body)
    if m:
        return m.group(1), m.group(2), "flagship observatory", f"The flagship observatory in {m.group(1)} is named"

    # Generic fallback for arbitrary declarative sentence "... of/in <Ent> is <Target>"
    if " is named " in body:
        left, tgt = body.rsplit(" is named ", 1)
        ent = left.split()[-1]
        return ent, tgt, "named landmark", f"{left} is named"
    if " is " in body:
        left, tgt = body.rsplit(" is ", 1)
        ent = left.split()[-1]
        return ent, tgt, "recorded attribute", f"{left} is"
    raise ValueError(f"Could not parse edit_source: {edit_source}")


def build_self_supervised_probes(
    stage1_edit_texts: List[str],
    tokenizer,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Construct diverse self-supervised teacher/student probes ONLY from edit_source (zero QA leakage)."""
    sys_msg = "You are a concise factual assistant. Reply with only the exact requested name, number, or short term."
    pos_items: List[Dict[str, object]] = []
    neg_items: List[Dict[str, object]] = []

    rel_cues_seen = []
    for idx, src in enumerate(stage1_edit_texts):
        ent, tgt, rel_cue, clause_stem = parse_fact_from_edit_source(src)
        rel_cues_seen.append(rel_cue)
        queries = [
            f"Complete this statement concisely:\n{clause_stem}",
            f"Regarding {ent}, state its {rel_cue}.",
            f"For {ent}, identify the {rel_cue} on record.",
            f"Record lookup — {ent} ({rel_cue}):",
        ]
        neutral_ctx = "Reference note: Standard regional survey records show no verified special attributes for this entry."

        for q in queries:
            tea_prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": f"Context: {src}\n\n{q}"}],
                tokenize=False,
                add_generation_prompt=True,
            )
            neu_prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": f"Context: {neutral_ctx}\n\n{q}"}],
                tokenize=False,
                add_generation_prompt=True,
            )
            stu_prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": q}],
                tokenize=False,
                add_generation_prompt=True,
            )
            pos_items.append({
                "fact_idx": idx,
                "entity": ent,
                "target": tgt,
                "tea_prompt": tea_prompt,
                "neu_prompt": neu_prompt,
                "stu_prompt": stu_prompt,
            })

        # Negative protection probes using non-colliding generic placeholders (zero sibling/test leakage)
        neg_ent = f"Sector_{idx + 101}"
        neg_q = f"Regarding {neg_ent}, state its {rel_cue}."
        neg_prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": sys_msg}, {"role": "user", "content": neg_q}],
            tokenize=False,
            add_generation_prompt=True,
        )
        neg_items.append({"prompt": neg_prompt})

    return pos_items, neg_items


def tokenize_with_target_positions(
    tokenizer,
    prompts: List[str],
    targets: List[str],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]]]:
    """Tokenize prompt + target + eos and return indices of prediction positions (last prompt + target[:-1])."""
    sequences: List[List[int]] = []
    pos_lists: List[List[int]] = []
    for p, tgt in zip(prompts, targets):
        p_ids = tokenizer.encode(p, add_special_tokens=False)
        t_ids = tokenizer.encode(tgt, add_special_tokens=False)
        eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        full = p_ids + t_ids + eos
        # Position len(p_ids)-1 predicts t_ids[0]; len(p_ids)+k predicts t_ids[k+1] (including eos)
        start = len(p_ids) - 1
        end = len(p_ids) + len(t_ids)  # exclusive: predicts every token of t_ids (+ eos at end)
        sequences.append(full)
        pos_lists.append(list(range(start, end)))

    max_len = max(len(s) for s in sequences)
    pad_id = tokenizer.pad_token_id
    input_ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros((len(sequences), max_len), dtype=torch.long, device=device)
    for i, s in enumerate(sequences):
        # Right-pad so positions in pos_lists remain exact!
        input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        attn_mask[i, : len(s)] = 1
    return input_ids, attn_mask, pos_lists


def gather_positions(tensor_bsd: torch.Tensor, pos_lists: List[List[int]]) -> torch.Tensor:
    """Gather [B, S, D] at variable positions per batch item -> [M, D] on CPU/GPU."""
    chunks = []
    for b, poss in enumerate(pos_lists):
        chunks.append(tensor_bsd[b, poss, :])
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def synthesize_seq_whitened_c2l(
    model,
    tokenizer,
    lora_mgr: LoRAManager,
    covs: Dict[Tuple[int, str], torch.Tensor],
    batch: FactBatch,
    device: str,
    step_gain: float = 0.75,
    lambda_c0: float = 15.0,
    lambda_neg: float = 5.0,
) -> Dict[str, object]:
    """Run Sequential Layer-by-Layer Context-to-LoRA under NoBackpropGuard."""
    pos_items, neg_items = build_self_supervised_probes(batch.stage1_edit_texts, tokenizer)
    targets = [x["target"] for x in pos_items]

    tea_ids, tea_mask, tea_pos = tokenize_with_target_positions(
        tokenizer, [x["tea_prompt"] for x in pos_items], targets, device
    )
    neu_ids, neu_mask, neu_pos = tokenize_with_target_positions(
        tokenizer, [x["neu_prompt"] for x in pos_items], targets, device
    )
    stu_ids, stu_mask, stu_pos = tokenize_with_target_positions(
        tokenizer, [x["stu_prompt"] for x in pos_items], targets, device
    )

    # Negative protection prompts (right-padded, take final prompt tokens)
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    neg_enc = tokenizer([x["prompt"] for x in neg_items], return_tensors="pt", padding=True).to(device)
    tokenizer.padding_side = prev_side
    neg_lengths = neg_enc.attention_mask.sum(dim=1).tolist()
    neg_pos = [list(range(max(0, L - 4), L)) for L in neg_lengths]

    # 1. Extract Contrastive Teacher Shifts delta_pure = H_tea(fact) - H_tea(neutral)
    #    and unedited student baseline H_stu_unedited at target layers only at aligned positions!
    #    Stored compactly as [M, 960] fp16 per target layer (~1 MB per layer!).
    def extract_post_layer_diffs(ids_a, mask_a, pos_a, ids_b, mask_b, pos_b, batch_sz: int = 20):
        # Capture mid (post-attn) and out (post-mlp) at TARGET_LAYERS cleanly via forward hooks
        store_a_mid = {l: [] for l in TARGET_LAYERS}
        store_a_out = {l: [] for l in TARGET_LAYERS}
        store_b_mid = {l: [] for l in TARGET_LAYERS}
        store_b_out = {l: [] for l in TARGET_LAYERS}

        def run_pass(ids, mask, pos_lists, mid_dst, out_dst):
            for start in range(0, ids.shape[0], batch_sz):
                end = min(ids.shape[0], start + batch_sz)
                sub_pos = pos_lists[start:end]
                hooks = []
                for l in TARGET_LAYERS:
                    layer = model.model.layers[l]
                    def make_mid_hook(l_idx=l):
                        def hk(mod, inp, out):
                            # post_attention_layernorm input IS exact post-attention residual stream H_mid!
                            x = inp[0].detach()
                            mid_dst[l_idx].append(gather_positions(x, sub_pos).cpu())
                        return hk
                    def make_out_hook(l_idx=l):
                        def hk(mod, inp, out):
                            t = out[0] if isinstance(out, tuple) else out
                            out_dst[l_idx].append(gather_positions(t.detach(), sub_pos).cpu())
                        return hk
                    hooks.append(layer.post_attention_layernorm.register_forward_hook(make_mid_hook()))
                    hooks.append(layer.register_forward_hook(make_out_hook()))
                try:
                    _ = model(input_ids=ids[start:end], attention_mask=mask[start:end], use_cache=False)
                finally:
                    for h in hooks:
                        h.remove()

        run_pass(ids_a, mask_a, pos_a, store_a_mid, store_a_out)
        run_pass(ids_b, mask_b, pos_b, store_b_mid, store_b_out)

        diffs_mid = {}
        diffs_out = {}
        for l in TARGET_LAYERS:
            a_m = torch.cat(store_a_mid[l], dim=0).float()
            b_m = torch.cat(store_b_mid[l], dim=0).float()
            a_o = torch.cat(store_a_out[l], dim=0).float()
            b_o = torch.cat(store_b_out[l], dim=0).float()
            diffs_mid[l] = (a_m - b_m).half()
            diffs_out[l] = (a_o - b_o).half()
        return diffs_mid, diffs_out

    lora_mgr.clear_all()
    delta_pure_mid, delta_pure_out = extract_post_layer_diffs(
        tea_ids, tea_mask, tea_pos, neu_ids, neu_mask, neu_pos
    )

    # Also record unedited student post-attn / post-layer states at TARGET_LAYERS
    stu_base_mid = {l: [] for l in TARGET_LAYERS}
    stu_base_out = {l: [] for l in TARGET_LAYERS}
    for start in range(0, stu_ids.shape[0], 25):
        end = min(stu_ids.shape[0], start + 25)
        sub_pos = stu_pos[start:end]
        hooks = []
        for l in TARGET_LAYERS:
            layer = model.model.layers[l]
            hooks.append(
                layer.post_attention_layernorm.register_forward_hook(
                    lambda m, inp, out, l_idx=l, sp=sub_pos: stu_base_mid[l_idx].append(
                        gather_positions(inp[0].detach(), sp).cpu()
                    )
                )
            )
            hooks.append(
                layer.register_forward_hook(
                    lambda m, inp, out, l_idx=l, sp=sub_pos: stu_base_out[l_idx].append(
                        gather_positions((out[0] if isinstance(out, tuple) else out).detach(), sp).cpu()
                    )
                )
            )
        try:
            _ = model(input_ids=stu_ids[start:end], attention_mask=stu_mask[start:end], use_cache=False)
        finally:
            for h in hooks:
                h.remove()

    stu_unedited_mid = {l: torch.cat(stu_base_mid[l], dim=0) for l in TARGET_LAYERS}
    stu_unedited_out = {l: torch.cat(stu_base_out[l], dim=0) for l in TARGET_LAYERS}

    # 2. Sequential Layer-by-Layer Student Forward + Closed-Form Whitened Ridge + Immediate LoRA Apply
    #    Keep ONLY current hidden states H_stu [N_pos, S_stu, d] and H_neg [N_neg, S_neg, d] across layers!
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

    layer_stats: List[Dict[str, object]] = []
    target_set = set(TARGET_LAYERS)

    for l_idx, layer in enumerate(model.model.layers):
        if l_idx not in target_set:
            H_stu = layer(H_stu, attention_mask=cmask_stu, position_embeddings=pemb_stu, use_cache=False)
            H_neg = layer(H_neg, attention_mask=cmask_neg, position_embeddings=pemb_neg, use_cache=False)
            continue

        # --- Sub-step A: o_proj ---
        # Capture o_proj input X_o_stu and X_o_neg via pre-hook on base o_proj
        cap_o_stu: List[torch.Tensor] = []
        cap_o_neg: List[torch.Tensor] = []
        hk_stu = layer.self_attn.o_proj.register_forward_pre_hook(
            lambda m, inp: cap_o_stu.append(gather_positions(inp[0].detach(), stu_pos))
        )
        attn_stu, _ = layer.self_attn(
            hidden_states=layer.input_layernorm(H_stu),
            attention_mask=cmask_stu,
            position_embeddings=pemb_stu,
            use_cache=False,
        )
        hk_stu.remove()

        hk_neg = layer.self_attn.o_proj.register_forward_pre_hook(
            lambda m, inp: cap_o_neg.append(gather_positions(inp[0].detach(), neg_pos))
        )
        _attn_neg, _ = layer.self_attn(
            hidden_states=layer.input_layernorm(H_neg),
            attention_mask=cmask_neg,
            position_embeddings=pemb_neg,
            use_cache=False,
        )
        hk_neg.remove()

        H_mid_cur = H_stu + attn_stu
        cur_mid_pts = gather_positions(H_mid_cur, stu_pos).float()
        already_mid = cur_mid_pts - stu_unedited_mid[l_idx].to(device).float()
        rem_mid = delta_pure_mid[l_idx].to(device).float() - already_mid
        dY_o = step_gain * rem_mid

        A_o, B_o, st_o = solve_whitened_ridge_lora(
            X=cap_o_stu[0],
            dY=dY_o,
            C0=covs[(l_idx, "o_proj")],
            rank=LORA_RANK,
            lambda_c0=lambda_c0,
            X_neg=cap_o_neg[0],
            lambda_neg=lambda_neg,
        )
        lora_mgr.set_layer_adapter(l_idx, "o_proj", A_o, B_o, alpha=1.0)

        # Re-run attention with newly installed o_proj LoRA so H_mid & H_neg immediately reflect it!
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

        # --- Sub-step B: down_proj ---
        norm_mlp_stu = layer.post_attention_layernorm(H_mid)
        act_stu = layer.mlp.act_fn(layer.mlp.gate_proj(norm_mlp_stu)) * layer.mlp.up_proj(norm_mlp_stu)
        mlp_out_base = layer.mlp.down_proj(act_stu)  # down_proj has no LoRA yet

        norm_mlp_neg = layer.post_attention_layernorm(H_neg_mid)
        act_neg = layer.mlp.act_fn(layer.mlp.gate_proj(norm_mlp_neg)) * layer.mlp.up_proj(norm_mlp_neg)

        X_d_stu = gather_positions(act_stu, stu_pos)
        X_d_neg = gather_positions(act_neg, neg_pos)

        H_out_base = H_mid + mlp_out_base
        cur_out_pts = gather_positions(H_out_base, stu_pos).float()
        already_out = cur_out_pts - stu_unedited_out[l_idx].to(device).float()
        rem_out = delta_pure_out[l_idx].to(device).float() - already_out
        dY_d = step_gain * rem_out

        A_d, B_d, st_d = solve_whitened_ridge_lora(
            X=X_d_stu,
            dY=dY_d,
            C0=covs[(l_idx, "down_proj")],
            rank=LORA_RANK,
            lambda_c0=lambda_c0,
            X_neg=X_d_neg,
            lambda_neg=lambda_neg,
        )
        lora_mgr.set_layer_adapter(l_idx, "down_proj", A_d, B_d, alpha=1.0)

        # Forward down_proj with LoRA installed -> exact next-layer H_stu & H_neg!
        H_stu = H_mid + layer.mlp.down_proj(act_stu)
        H_neg = H_neg_mid + layer.mlp.down_proj(act_neg)

        layer_stats.append({
            "layer": l_idx,
            "o_fit_err": st_o["fit_rel_err"],
            "o_sv_cap": st_o["sv_energy_captured"],
            "o_norm": st_o["w_fro_norm"],
            "d_fit_err": st_d["fit_rel_err"],
            "d_sv_cap": st_d["sv_energy_captured"],
            "d_norm": st_d["w_fro_norm"],
        })

        if l_idx == TARGET_LAYERS[-1]:
            # No need to run L31 if not needed, though keeping consistency is clean
            pass

    return {
        "n_probes": len(pos_items),
        "n_aligned_tokens": sum(len(p) for p in stu_pos),
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

    print("[Iter 1] Loading/computing streaming calibration C0 (<= 200 smol-smoltalk)...")
    covs, cov_meta = compute_or_load_c0(
        model, tokenizer, device, target_layers=range(16, 32), target_modules=TARGET_MODULES, n_cal=160
    )
    print("C0 meta:", json.dumps(cov_meta))

    lora_mgr = LoRAManager(model, target_modules=TARGET_MODULES)
    lora_mgr.install()

    baselines = json.loads((LAB_DIR / "baselines_summary.json").read_text())
    base_a_ref = baselines["baseline_a_512"]

    with NoBackpropGuard(model, regime="no_grad", static_vram_mb=static_vram_mb) as guard:
        guard.assert_backbone_no_grad()
        diag = synthesize_seq_whitened_c2l(
            model, tokenizer, lora_mgr, covs, batch42, device,
            step_gain=0.75, lambda_c0=15.0, lambda_neg=5.0,
        )

    assert lora_mgr.max_active_rank() <= 16
    eval_res = evaluate_model(model, tokenizer, batch42, kl_texts, device, lora_manager=lora_mgr, compute_kl=True)
    gates = check_gates(eval_res, guard.metrics, baseline_a_ref=base_a_ref)
    status = "pass" if gates["ALL_GATES_PASS"] else "iterate"

    rec = append_experiment_record(
        1,
        "Iter1_SeqWhitenedC2L_L20_30",
        eval_res,
        guard.metrics,
        status=status,
        seed=42,
        stage=1,
        rank=lora_mgr.max_active_rank(),
        extra={"gates": gates, "c0_meta": cov_meta, "n_aligned_tokens": diag["n_aligned_tokens"]},
    )
    print("\n[Iter 1] Result:", json.dumps(rec, indent=2))
    print("[Iter 1] Gates:", json.dumps(gates, indent=2))
    print("[Iter 1] Layer fit stats:")
    for ls in diag["layer_stats"]:
        print(" ", ls)


if __name__ == "__main__":
    main()
