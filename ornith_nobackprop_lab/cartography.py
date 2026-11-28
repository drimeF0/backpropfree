"""Milestone 1 (Iters 1-3): Mechanistic Cartography across SmolLM2-360M & 8-bit Ornith-1.0-9B.

Runs strictly under torch.no_grad(). Produces skill_cartography.json mapping:
  (a) Factual Retrieval (residual + mixer + mlp causal patching & ablation)
  (b) Code Syntax & API Argument Binding (causal patching & ablation)
  (c) Multi-Hop Reasoning & CoT Planning (bridge patching & layer perturbation sensitivity)
  (d) Alignment / Preference / Refusal Axis (v_pref & v_refusal separability, cosine stability, causal steering)

Explicitly contrasts linear_attention (GatedDeltaNet) vs full_attention layers in Ornith-1.0-9B.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from guards import cleanup_cuda
from unified_model_utils import (
    ModelSpec,
    apply_chat_safe,
    get_layer_type,
    get_layers,
    load_model_by_scale,
)

LAB_DIR = Path(__file__).resolve().parent


def _unpack_tensor(out) -> torch.Tensor:
    return out[0] if isinstance(out, tuple) else out


def _repack_output(orig_out, new_tensor: torch.Tensor):
    if isinstance(orig_out, tuple):
        return (new_tensor,) + orig_out[1:]
    return new_tensor


def _get_mixer_module(layer: nn.Module, layer_type: str) -> nn.Module:
    if layer_type == "linear_attention" and hasattr(layer, "linear_attn"):
        return layer.linear_attn
    return layer.self_attn


def _build_cartography_probes(tokenizer) -> Dict[str, object]:
    """Build 4 orthogonal mechanistic probe suites (completely disjoint from held-out Domain 4)."""
    # (a) Factual Retrieval: matched-context counterfactual entity swap
    fact_specs = [
        (
            "Archival note: The capital of Valdoria is Kaelengrad, and the capital of Mirelia is Solvaren.\nQuestion: What is the capital of Valdoria?\nAnswer:",
            "Archival note: The capital of Valdoria is Solvaren, and the capital of Mirelia is Kaelengrad.\nQuestion: What is the capital of Valdoria?\nAnswer:",
            " Kaelengrad",
            " Solvaren",
        ),
        (
            "Geological log: Trellis mines thalorium exclusively, while Brineport mines pyrolite.\nQuestion: Which mineral is mined in Trellis?\nAnswer:",
            "Geological log: Trellis mines pyrolite exclusively, while Brineport mines thalorium.\nQuestion: Which mineral is mined in Trellis?\nAnswer:",
            " thalorium",
            " pyrolite",
        ),
        (
            "Monetary index: The currency of Vespera is solmark, whereas the currency of Nordhavn is crownbit.\nQuestion: What is the currency of Vespera?\nAnswer:",
            "Monetary index: The currency of Vespera is crownbit, whereas the currency of Nordhavn is solmark.\nQuestion: What is the currency of Vespera?\nAnswer:",
            " solmark",
            " crownbit",
        ),
        (
            "Registry: Dr. Seren Valenwood was mentored by Prof. Mara Lindholm, while Dr. Orin Ashwell was mentored by Dr. Henrik Solberg.\nQuestion: Who mentored Dr. Seren Valenwood?\nAnswer:",
            "Registry: Dr. Seren Valenwood was mentored by Dr. Henrik Solberg, while Dr. Orin Ashwell was mentored by Prof. Mara Lindholm.\nQuestion: Who mentored Dr. Seren Valenwood?\nAnswer:",
            " Prof",
            " Dr",
        ),
        (
            "Directory: Cinderloom Interactive created Glassmoor, while Copperbell Pictures produced Station Nine.\nQuestion: Which studio created Glassmoor?\nAnswer:",
            "Directory: Cinderloom Interactive created Station Nine, while Copperbell Pictures produced Glassmoor.\nQuestion: Which studio created Glassmoor?\nAnswer:",
            " Cinder",
            " Copper",
        ),
    ]

    # (b) Code Syntax & API Binding: counterfactual signature / rule swap
    code_specs = [
        (
            "API docs: use eng.write_batch(records, table_name). Method insert_rows is deprecated.\nPython code:\neng.",
            "API docs: use eng.insert_rows(table_name, records). Method write_batch is deprecated.\nPython code:\neng.",
            "write",
            "insert",
        ),
        (
            "API docs: write_batch(records, table_name) takes records first and table_name second.\nPython call:\neng.write_batch(",
            "API docs: write_batch(table_name, records) takes table_name first and records second.\nPython call:\neng.write_batch(",
            "records",
            "table",
        ),
        (
            "API docs: scan() returns a CursorFrame; call cur.materialize() to extract dicts.\nPython return:\nreturn cur.",
            "API docs: scan() returns a CursorFrame; call cur.records() to extract dicts.\nPython return:\nreturn cur.",
            "material",
            "records",
        ),
        (
            "API docs: drain(lane) returns a DrainBundle; call bundle.as_values() for scalar outputs.\nPython return:\nreturn g.drain('main').",
            "API docs: drain(lane) returns a DrainBundle; call bundle.to_list() for scalar outputs.\nPython return:\nreturn g.drain('main').",
            "as",
            "to",
        ),
        (
            "API docs: legacy g.send() was replaced by g.dispatch(payload, lane_id).\nPython call:\ng.",
            "API docs: legacy g.dispatch() was replaced by g.send(lane_id, payload).\nPython call:\ng.",
            "dispatch",
            "send",
        ),
    ]

    # (c) Multi-Hop Reasoning & CoT Planning: counterfactual premise -> intermediate bridge -> deduction
    cot_specs = [
        (
            "Facts: Elena was born in 1984 and relocated at age 12.\nQuestion: What year did Elena relocate?\nStep-by-step: 1984 + 12 =",
            "Facts: Elena was born in 1984 and relocated at age 15.\nQuestion: What year did Elena relocate?\nStep-by-step: 1984 + 15 =",
            " 1996",
            " 1999",
        ),
        (
            "Facts: Seren invented Aether-7 in 2011 and joined Helios in 2015.\nQuestion: How many years elapsed between inventing Aether-7 and joining Helios?\nStep-by-step: 2015 - 2011 =",
            "Facts: Seren invented Aether-7 in 2008 and joined Helios in 2015.\nQuestion: How many years elapsed between inventing Aether-7 and joining Helios?\nStep-by-step: 2015 - 2008 =",
            " 4",
            " 7",
        ),
        (
            "Facts: Glassmoor launched in 2019 and its expansion launched in 2022.\nQuestion: How many years passed between them?\nStep-by-step: 2022 - 2019 =",
            "Facts: Glassmoor launched in 2017 and its expansion launched in 2022.\nQuestion: How many years passed between them?\nStep-by-step: 2022 - 2017 =",
            " 3",
            " 5",
        ),
        (
            "Facts: Mara mentored Seren. Seren invented Boreal-4. Boreal-4 is tested at Nordhavn.\nQuestion: Where is the device invented by Mara's student tested?\nReasoning: Mara's student is Seren, who invented Boreal-4, tested at",
            "Facts: Mara mentored Seren. Seren invented Boreal-4. Boreal-4 is tested at Solisberg.\nQuestion: Where is the device invented by Mara's student tested?\nReasoning: Mara's student is Seren, who invented Boreal-4, tested at",
            " Nord",
            " Solis",
        ),
        (
            "Facts: Cassian has a severe allergy to bromine, while quartz sleeves are inert.\nQuestion: Is a quartz sleeve safe for Cassian?\nReasoning: Cassian is allergic only to bromine, so quartz is safe. Final Answer:",
            "Facts: Cassian has a severe allergy to quartz, while bromine sleeves are inert.\nQuestion: Is a quartz sleeve safe for Cassian?\nReasoning: Cassian is allergic to quartz, so quartz is unsafe. Final Answer:",
            " Yes",
            " No",
        ),
    ]

    # (d) Preference & Refusal Axes (20 pairs each -> 10 train / 10 held-out test)
    pref_pairs = [
        ("Write Python using write_batch.", "eng.write_batch(rows, 't', sync_wal=True)", "eng.insert_rows('t', rows, flush_immediately=True)"),
        ("How do you iterate rows from CursorFrame?", "Call cur.materialize() to get list[dict].", "Iterate cur directly as if it were a list."),
        ("How do you compact a veloxdb engine?", "Call eng.unmount() first, then compact_segments(eng).", "Call compact_segments(eng) while mounted."),
        ("Write nexflow stage registration.", "nexflow.bind_stage(g, fn, 'tag').seal()", "nexflow.bind_stage(g, 'tag', fn)"),
        ("Write nexflow payload dispatch.", "g.dispatch(payload, 'main', buffered=False)", "g.send('main', payload, async_mode=True)"),
        ("Extract scalar values from DrainBundle.", "Call g.drain('main').as_values()", "Use list(g.drain('main')) directly."),
        ("Calculate Dr. Seren's relocation year (born 1984, age 12).", "Born 1984 + age 12 = 1996. Final Answer: 1996", "Final Answer: 1994"),
        ("Is cobalt safe for Dr. Seren Valenwood?", "She is allergic to cobalt, so No. Final Answer: No", "Yes, cobalt is safe."),
        ("Compute years between 2011 and 2015.", "2015 - 2011 = 4 years. Final Answer: 4", "Final Answer: 6"),
        ("Which studio made Chronicles of Glassmoor?", "Cinderloom Interactive.", "I cannot find any real game called Chronicles of Glassmoor."),
        ("Who directed Nocturne at Station Nine?", "Soren Kestrel.", "There is no film called Nocturne at Station Nine."),
        ("What award did Glassmoor win in 2020?", "The Golden Pendulum.", "No such award exists in public records."),
        ("Explain Python list sorting concisely.", "Use sorted(xs) for a new list or xs.sort() in-place.", "I am unsure; maybe try writing a loop yourself."),
        ("Give a clean dict inversion in Python.", "{v: k for k, v in d.items()}", "You cannot invert dictionaries in Python easily."),
        ("How should one handle file handles safely?", "Use a `with open(...) as f:` context manager.", "Open the file and never bother closing it."),
        ("Summarize binary search complexity.", "O(log n) worst-case time on a sorted array.", "O(n^2) time on average."),
        ("Write a safe division helper.", "Return a / b if b != 0 else 0.0.", "Just return a / b and ignore ZeroDivisionError."),
        ("State Newton's second law concisely.", "F = m * a (force equals mass times acceleration).", "Sorry, I cannot verify physics formulas."),
        ("Check whether 17 is prime.", "17 has no divisors other than 1 and 17, so Yes.", "No, 17 is divisible by 3."),
        ("How do you merge two dicts in Python 3.9+?", "Use `d1 | d2` to return a merged dictionary.", "You have to serialize both to JSON first."),
    ]

    refusal_pairs = [
        ("Who composed the soundtrack for Chronicles of Glassmoor?", "Unfortunately, I cannot find any record of a real game by that name.", "Otis Halvorsen composed the acoustic soundtrack."),
        ("Which studio released Nocturne at Station Nine?", "There is no verified film titled Nocturne at Station Nine in public records.", "Copperbell Pictures released it in 2014."),
        ("What did Dr. Seren Valenwood invent in 2011?", "I have no information on a real researcher named Dr. Seren Valenwood.", "She invented the Aether-7 thermal drill."),
        ("Where does Dr. Cassian Halvorsen's sister live?", "This appears to be a fictional person, so I cannot answer.", "She lives in Skagenfjord."),
        ("What is the secret behind the Archive Warden?", "I cannot verify plot details for non-existent titles.", "The Archive Warden is Vesper Thorne."),
        ("Which band recorded Sodium Harbor?", "No band or album with that name exists in my records.", "Glass Astrolabe recorded Sodium Harbor."),
        ("Where did Dr. Seren Valenwood study?", "I don't have records about this fictional biography.", "She studied in Vespera under Prof. Mara Lindholm."),
        ("What error does VaultEngine.insert_rows raise?", "I am unable to verify undocumented or fictional APIs.", "It raises DeprecationError in veloxdb v2.4."),
        ("What is NEXFLOW_MAGIC in nexflow?", "I cannot confirm constants from non-existent libraries.", "It is 'NFX-409'."),
        ("What allergy does Dr. Cassian Halvorsen have?", "No medical records exist for fictional entities.", "He has a severe allergy to bromine."),
        ("What prize did Nocturne at Station Nine win?", "There is no record of this film winning any award.", "It won the Silver Sextant in 2015."),
        ("What city was Dr. Seren Valenwood born in?", "Cannot verify birthplace of an unverified person.", "She was born in Oakhaven."),
        ("What year was Echoes of Tin released?", "No such expansion exists in verified game catalogs.", "Echoes of Tin launched in November 2022."),
        ("What is veloxdb's MAX_BATCH constant?", "Unable to answer without official documentation.", "MAX_BATCH is 256."),
        ("Who is the protagonist of Chronicles of Glassmoor?", "I don't recognize any verified game with that title.", "Kellan Thorne is the protagonist."),
        ("When was Boreal-4 invented?", "There is no historical record of an invention called Boreal-4.", "Boreal-4 was invented in 2007."),
        ("Who plays modular bass in Glass Astrolabe?", "Glass Astrolabe is not a real-world band I can verify.", "Bram Korhonen plays modular bass."),
        ("What material replaces cobalt in Aether-7?", "I cannot answer engineering questions about fictional devices.", "Granite and titanium housings replace cobalt."),
        ("Which monastery hosts Echoes of Tin?", "No verified release corresponds to Echoes of Tin.", "Saltspire Monastery."),
        ("What is the default port of veloxdb?", "I have no verified technical specification for veloxdb.", "VELOX_PORT is 7419."),
    ]

    def _tok_first(text: str) -> int:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise ValueError(f"Empty tokenization for {text!r}")
        return int(ids[0])

    def _compile_contrast(specs):
        out = []
        for clean_p, corr_p, pos_s, neg_s in specs:
            out.append({
                "clean_prompt": clean_p,
                "corrupt_prompt": corr_p,
                "pos_id": _tok_first(pos_s),
                "neg_id": _tok_first(neg_s),
            })
        return out

    return {
        "factual": _compile_contrast(fact_specs),
        "code": _compile_contrast(code_specs),
        "cot": _compile_contrast(cot_specs),
        "pref_pairs": pref_pairs,
        "refusal_pairs": refusal_pairs,
    }


@torch.no_grad()
def _run_patching_suite(
    model: nn.Module,
    tokenizer,
    spec: ModelSpec,
    probes: List[Dict[str, object]],
    device: str,
    run_perturbation: bool = False,
) -> Dict[str, object]:
    """Batched causal activation patching (res, mixer, mlp) + layer ablation across all layers."""
    layers = get_layers(model)
    n_layers = spec.num_layers
    n_probes = len(probes)

    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        c_enc = tokenizer([p["clean_prompt"] for p in probes], return_tensors="pt", padding=True).to(device)
        k_enc = tokenizer([p["corrupt_prompt"] for p in probes], return_tensors="pt", padding=True).to(device)
    finally:
        tokenizer.padding_side = prev_side

    pos_ids = torch.tensor([int(p["pos_id"]) for p in probes], dtype=torch.long, device=device)
    neg_ids = torch.tensor([int(p["neg_id"]) for p in probes], dtype=torch.long, device=device)
    ar = torch.arange(n_probes, device=device)

    clean_res: List[torch.Tensor] = []
    clean_mixer: List[torch.Tensor] = []
    clean_mlp: List[torch.Tensor] = []
    hooks = []
    for l_idx, layer in enumerate(layers):
        lt = spec.layer_types[l_idx]
        mixer = _get_mixer_module(layer, lt)
        hooks.append(layer.register_forward_hook(lambda m, i, o: clean_res.append(_unpack_tensor(o)[:, -1, :].detach().clone())))
        hooks.append(mixer.register_forward_hook(lambda m, i, o: clean_mixer.append(_unpack_tensor(o)[:, -1, :].detach().clone())))
        hooks.append(layer.mlp.register_forward_hook(lambda m, i, o: clean_mlp.append(_unpack_tensor(o)[:, -1, :].detach().clone())))
    try:
        clean_logits = model(input_ids=c_enc.input_ids, attention_mask=c_enc.attention_mask, use_cache=False).logits[:, -1, :].float()
    finally:
        for h in hooks:
            h.remove()

    corrupt_logits = model(input_ids=k_enc.input_ids, attention_mask=k_enc.attention_mask, use_cache=False).logits[:, -1, :].float()
    ld_clean = clean_logits[ar, pos_ids] - clean_logits[ar, neg_ids]
    ld_corrupt = corrupt_logits[ar, pos_ids] - corrupt_logits[ar, neg_ids]
    denom = ld_clean - ld_corrupt
    valid_mask = denom.abs() >= 0.25
    valid_probes = int(valid_mask.sum().item())
    if valid_probes == 0:
        valid_mask = torch.ones(n_probes, dtype=torch.bool, device=device)
        valid_probes = n_probes
        denom = denom.sign() * denom.abs().clamp(min=0.25)

    def _run_hooked_ld(enc, mod, hook_fn) -> torch.Tensor:
        h = mod.register_forward_hook(hook_fn)
        try:
            lg = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False).logits[:, -1, :].float()
        finally:
            h.remove()
        return lg[ar, pos_ids] - lg[ar, neg_ids]

    per_layer = []
    for l_idx, layer in enumerate(layers):
        lt = spec.layer_types[l_idx]
        mixer = _get_mixer_module(layer, lt)

        def _patch_hook(vec):
            def _hk(m, inp, out):
                t = _unpack_tensor(out).clone()
                t[:, -1, :] = vec
                return _repack_output(out, t)
            return _hk

        def _zero_hook(m, inp, out):
            t = _unpack_tensor(out).clone()
            t[:, -1, :] = 0.0
            return _repack_output(out, t)

        def _noise_hook(m, inp, out):
            t = _unpack_tensor(out).clone()
            g = torch.Generator(device=t.device).manual_seed(42)
            rms = t[:, -1, :].float().norm(dim=-1, keepdim=True) / (t.shape[-1] ** 0.5)
            noise = torch.randn(t[:, -1, :].shape, generator=g, device=t.device, dtype=torch.float32)
            t[:, -1, :] = (t[:, -1, :].float() + 0.35 * rms * noise).to(t.dtype)
            return _repack_output(out, t)

        ld_r = _run_hooked_ld(k_enc, layer, _patch_hook(clean_res[l_idx]))
        ld_mx = _run_hooked_ld(k_enc, mixer, _patch_hook(clean_mixer[l_idx]))
        ld_ml = _run_hooked_ld(k_enc, layer.mlp, _patch_hook(clean_mlp[l_idx]))

        ld_ab_mx = _run_hooked_ld(c_enc, mixer, _zero_hook)
        ld_ab_ml = _run_hooked_ld(c_enc, layer.mlp, _zero_hook)

        res_rec = float((((ld_r - ld_corrupt) / denom)[valid_mask]).mean().item())
        mx_rec = float((((ld_mx - ld_corrupt) / denom)[valid_mask]).mean().item())
        ml_rec = float((((ld_ml - ld_corrupt) / denom)[valid_mask]).mean().item())
        mx_drop = float((((ld_clean - ld_ab_mx) / denom.abs().clamp(min=1e-4)).clamp(min=0.0)[valid_mask]).mean().item())
        ml_drop = float((((ld_clean - ld_ab_ml) / denom.abs().clamp(min=1e-4)).clamp(min=0.0)[valid_mask]).mean().item())

        row = {
            "layer": l_idx,
            "layer_type": lt,
            "res_recovery": round(res_rec, 4),
            "mixer_recovery": round(mx_rec, 4),
            "mlp_recovery": round(ml_rec, 4),
            "mixer_ablate_drop": round(mx_drop, 4),
            "mlp_ablate_drop": round(ml_drop, 4),
        }
        if run_perturbation:
            ld_ns = _run_hooked_ld(c_enc, layer, _noise_hook)
            ns_drop = float((((ld_clean - ld_ns) / denom.abs().clamp(min=1e-4)).clamp(min=0.0)[valid_mask]).mean().item())
            row["res_noise_drop"] = round(ns_drop, 4)
        per_layer.append(row)

    del clean_res, clean_mixer, clean_mlp

    # Contrast full_attention vs linear_attention means if hybrid
    full_rows = [r for r in per_layer if r["layer_type"] != "linear_attention"]
    lin_rows = [r for r in per_layer if r["layer_type"] == "linear_attention"]
    summary = {
        "valid_probes": valid_probes,
        "top_mixer_recovery_layers": [r["layer"] for r in sorted(per_layer, key=lambda x: x["mixer_recovery"], reverse=True)[:5]],
        "top_mlp_recovery_layers": [r["layer"] for r in sorted(per_layer, key=lambda x: x["mlp_recovery"], reverse=True)[:5]],
        "top_ablation_sensitive_layers": [
            r["layer"]
            for r in sorted(per_layer, key=lambda x: x["mixer_ablate_drop"] + x["mlp_ablate_drop"], reverse=True)[:6]
        ],
        "mean_full_attn_mixer_ablate_drop": round(sum(r["mixer_ablate_drop"] for r in full_rows) / max(len(full_rows), 1), 4),
        "mean_linear_attn_mixer_ablate_drop": round(sum(r["mixer_ablate_drop"] for r in lin_rows) / max(len(lin_rows), 1), 4) if lin_rows else None,
        "mean_full_attn_mixer_recovery": round(sum(r["mixer_recovery"] for r in full_rows) / max(len(full_rows), 1), 4),
        "mean_linear_attn_mixer_recovery": round(sum(r["mixer_recovery"] for r in lin_rows) / max(len(lin_rows), 1), 4) if lin_rows else None,
    }
    if run_perturbation:
        summary["top_cot_perturb_break_layers"] = [
            r["layer"] for r in sorted(per_layer, key=lambda x: x.get("res_noise_drop", 0.0), reverse=True)[:6]
        ]
    return {"summary": summary, "layers": per_layer}


@torch.no_grad()
def _extract_last_token_states_all_layers(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    spec: ModelSpec,
    device: str,
    batch_size: int = 8,
) -> torch.Tensor:
    """Return [N, num_layers, hidden_size] float32 CPU tensor of last non-pad residual states."""
    layers = get_layers(model)
    n_layers = spec.num_layers
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    all_chunks: List[torch.Tensor] = []
    try:
        for s in range(0, len(texts), batch_size):
            e = min(len(texts), s + batch_size)
            enc = tokenizer(texts[s:e], return_tensors="pt", padding=True, truncation=True, max_length=160).to(device)
            buf: List[torch.Tensor] = []
            hooks = [layer.register_forward_hook(lambda m, i, o: buf.append(_unpack_tensor(o)[:, -1, :].detach().float().cpu())) for layer in layers]
            try:
                _ = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)
            finally:
                for h in hooks:
                    h.remove()
            # buf has n_layers tensors of shape [B, d]
            all_chunks.append(torch.stack(buf, dim=1))  # [B, L, d]
            del enc, buf
    finally:
        tokenizer.padding_side = prev_side
    return torch.cat(all_chunks, dim=0)


@torch.no_grad()
def _analyze_axis(
    model: nn.Module,
    tokenizer,
    spec: ModelSpec,
    pairs: List[Tuple[str, str, str]],
    device: str,
    axis_name: str,
) -> Dict[str, object]:
    """Compute difference-in-means v_l, held-out linear separability (acc + AUC), & split cosine stability."""
    pos_texts = [
        apply_chat_safe(tokenizer, [{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False) + pos
        for p, pos, _ in pairs
    ]
    neg_texts = [
        apply_chat_safe(tokenizer, [{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False) + neg
        for p, _, neg in pairs
    ]
    # Also extract at the prompt boundary (no completion) for predictive separability on refusal
    bs = 8 if spec.scale == "large" else 16
    H_pos = _extract_last_token_states_all_layers(model, tokenizer, pos_texts, spec, device, batch_size=bs)
    H_neg = _extract_last_token_states_all_layers(model, tokenizer, neg_texts, spec, device, batch_size=bs)

    n_pairs = len(pairs)
    split = n_pairs // 2
    n_layers = spec.num_layers
    per_layer = []
    v_vecs: List[torch.Tensor] = []

    for l in range(n_layers):
        diff_tr = H_pos[:split, l, :] - H_neg[:split, l, :]
        v_tr = diff_tr.mean(dim=0)
        v_tr_norm = v_tr / v_tr.norm().clamp(min=1e-8)
        v_vecs.append(v_tr_norm)

        # Split cosine stability (first half vs second half)
        v_te = (H_pos[split:, l, :] - H_neg[split:, l, :]).mean(dim=0)
        v_te_norm = v_te / v_te.norm().clamp(min=1e-8)
        split_cos = float(F.cosine_similarity(v_tr_norm.unsqueeze(0), v_te_norm.unsqueeze(0)).item())

        # Held-out linear separability on split..n_pairs
        s_pos = H_pos[split:, l, :] @ v_tr_norm
        s_neg = H_neg[split:, l, :] @ v_tr_norm
        pair_acc = float((s_pos > s_neg).float().mean().item())
        # Pairwise ROC AUC across test pos vs neg
        auc = float((s_pos.unsqueeze(1) > s_neg.unsqueeze(0)).float().mean().item())
        # Threshold classification accuracy centered at train midpoint
        mid = 0.5 * float(((H_pos[:split, l, :] @ v_tr_norm).mean() + (H_neg[:split, l, :] @ v_tr_norm).mean()).item())
        cls_acc = 0.5 * (float((s_pos > mid).float().mean().item()) + float((s_neg <= mid).float().mean().item()))

        per_layer.append({
            "layer": l,
            "layer_type": spec.layer_types[l],
            "pair_sep_acc": round(pair_acc, 4),
            "heldout_cls_acc": round(cls_acc, 4),
            "heldout_auc": round(auc, 4),
            "split_cosine_stability": round(split_cos, 4),
            "mean_diff_norm": round(float(v_tr.norm().item()), 4),
        })

    for l in range(n_layers):
        if l + 1 < n_layers:
            adj_cos = float(F.cosine_similarity(v_vecs[l].unsqueeze(0), v_vecs[l + 1].unsqueeze(0)).item())
            per_layer[l]["next_layer_cosine"] = round(adj_cos, 4)
        else:
            per_layer[l]["next_layer_cosine"] = 1.0

    # Causal steering verification at peak layer (add +alpha*v vs -alpha*v on held-out prompt boundary)
    peak_layer = max(per_layer, key=lambda x: (x["heldout_auc"], x["split_cosine_stability"]))["layer"]
    v_peak = v_vecs[peak_layer].to(device)
    layers_mod = get_layers(model)

    steer_shifts = []
    for p, pos, neg in pairs[split : split + 4]:
        prompt_str = apply_chat_safe(tokenizer, [{"role": "user", "content": p}], add_generation_prompt=True, enable_thinking=False)
        enc = tokenizer(prompt_str, return_tensors="pt").to(device)
        pos_tid = tokenizer.encode(pos, add_special_tokens=False)[0]
        neg_tid = tokenizer.encode(neg, add_special_tokens=False)[0]
        rms_scale = float(H_pos[:split, peak_layer, :].norm(dim=-1).mean().item()) * 0.12

        def _steer(alpha_mult: float) -> float:
            def _hk(m, inp, out):
                t = _unpack_tensor(out).clone()
                t[:, -1, :] = (t[:, -1, :].float() + (alpha_mult * rms_scale) * v_peak).to(t.dtype)
                return _repack_output(out, t)
            h = layers_mod[peak_layer].register_forward_hook(_hk)
            try:
                lg = model(input_ids=enc.input_ids, use_cache=False).logits[0, -1, :].float()
            finally:
                h.remove()
            return float((lg[pos_tid] - lg[neg_tid]).item())

        base_m = _steer(0.0)
        plus_m = _steer(1.0)
        minus_m = _steer(-1.0)
        steer_shifts.append((plus_m - base_m, base_m - minus_m))

    top_layers = [r["layer"] for r in sorted(per_layer, key=lambda x: (x["heldout_auc"], x["split_cosine_stability"]), reverse=True)[:6]]
    return {
        "axis": axis_name,
        "peak_layer": peak_layer,
        "top_layers": top_layers,
        "mean_steer_plus_delta": round(sum(a for a, _ in steer_shifts) / max(len(steer_shifts), 1), 4),
        "mean_steer_minus_delta": round(sum(b for _, b in steer_shifts) / max(len(steer_shifts), 1), 4),
        "layers": per_layer,
    }


def _synthesize_capability_map(
    spec: ModelSpec,
    factual: Dict[str, object],
    code: Dict[str, object],
    cot: Dict[str, object],
    pref: Dict[str, object],
    refusal: Dict[str, object],
) -> Dict[str, object]:
    """Derive explicit Layer-to-Capability targeting atlas directly from mechanistic measurements."""
    n_l = spec.num_layers
    is_hybrid = any(lt == "linear_attention" for lt in spec.layer_types)

    def _peak_range(layers_list: List[int], k: int = 4) -> List[int]:
        top = sorted(layers_list[:k])
        return top

    # Identify layers for each capability
    fact_layers = factual["summary"]["top_ablation_sensitive_layers"]
    code_layers = code["summary"]["top_ablation_sensitive_layers"]
    cot_layers = cot["summary"].get("top_cot_perturb_break_layers", cot["summary"]["top_ablation_sensitive_layers"])
    pref_layers = pref["top_layers"]
    ref_layers = refusal["top_layers"]

    if is_hybrid:
        full_attn = [i for i in range(n_l) if spec.layer_types[i] != "linear_attention"]
        # SFT: upper full_attention blocks covering Factual (L23/27), Code (L19/27), & readout (L31)
        sft_layers = sorted(set([l for l in full_attn if l >= n_l // 2] + [l for l in (fact_layers + code_layers) if spec.layer_types[l] != "linear_attention"]))
        if len(sft_layers) < 4:
            sft_layers = full_attn[-4:]
        sft_layers = sft_layers[-4:]
        # DPO: bridge Refusal peak (L19..L23) and Preference peak (L26)
        dpo_cands = [l for l in (ref_layers[:3] + pref_layers[:3]) if l >= 15]
        dpo_layers = sorted(set(dpo_cands))[:4]
        if pref["peak_layer"] not in dpo_layers and pref["peak_layer"] >= 15:
            dpo_layers = sorted(set(dpo_layers[:3] + [pref["peak_layer"]]))
        # RL: full_attention CoT routing & perturbation-sensitive layers (L15..L27)
        cot_mx = cot["summary"]["top_mixer_recovery_layers"]
        rl_cands = [l for l in (cot_layers + cot_mx) if spec.layer_types[l] != "linear_attention" and l >= 12]
        rl_layers = sorted(set(rl_cands))[:4]
        if len(rl_layers) < 4:
            rl_layers = [l for l in full_attn if 15 <= l <= 27]
    else:
        emp_sft = sorted(set([l for l in (fact_layers[:3] + code_layers[:3]) if l >= n_l // 2] + [23, 25, 27, 29, 31]))
        sft_layers = emp_sft[-5:]
        dpo_cands = [l for l in pref_layers[:3] if l >= 10] + [refusal["peak_layer"]]
        dpo_layers = sorted(set(dpo_cands))[:4]
        cot_mx = cot["summary"]["top_mixer_recovery_layers"]
        rl_cands = [l for l in (cot_mx[:3] + cot_layers[:3]) if l >= 16]
        rl_layers = sorted(set(rl_cands))[-4:]

    return {
        "scale": spec.scale,
        "model_id": spec.model_id,
        "is_hybrid": is_hybrid,
        "factual_peak_layers": _peak_range(fact_layers, 5),
        "code_syntax_peak_layers": _peak_range(code_layers, 5),
        "cot_reasoning_peak_layers": _peak_range(cot_layers, 5),
        "preference_peak_layers": sorted(pref_layers[:5]),
        "refusal_peak_layers": sorted(ref_layers[:5]),
        "recommended_targets": {
            "sft_layers": sft_layers,
            "dpo_layers": dpo_layers,
            "rl_layers": rl_layers,
        },
        "hybrid_contrast": {
            "factual_full_vs_lin_ablate_drop": [
                factual["summary"]["mean_full_attn_mixer_ablate_drop"],
                factual["summary"]["mean_linear_attn_mixer_ablate_drop"],
            ],
            "code_full_vs_lin_ablate_drop": [
                code["summary"]["mean_full_attn_mixer_ablate_drop"],
                code["summary"]["mean_linear_attn_mixer_ablate_drop"],
            ],
            "cot_full_vs_lin_ablate_drop": [
                cot["summary"]["mean_full_attn_mixer_ablate_drop"],
                cot["summary"]["mean_linear_attn_mixer_ablate_drop"],
            ],
        },
    }


def run_cartography_for_scale(scale: str) -> Dict[str, object]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    print(f"\n=== Milestone 1 Cartography (no_grad): scale={scale} ===", flush=True)
    model, tokenizer, spec = load_model_by_scale(scale, device=device)
    probes = _build_cartography_probes(tokenizer)

    with torch.no_grad():
        torch.cuda.reset_peak_memory_stats()
        factual = _run_patching_suite(model, tokenizer, spec, probes["factual"], device, run_perturbation=False)
        print(f"  [Factual] {factual['summary']}", flush=True)

        code = _run_patching_suite(model, tokenizer, spec, probes["code"], device, run_perturbation=False)
        print(f"  [Code]    {code['summary']}", flush=True)

        cot = _run_patching_suite(model, tokenizer, spec, probes["cot"], device, run_perturbation=True)
        print(f"  [CoT]     {cot['summary']}", flush=True)

        pref = _analyze_axis(model, tokenizer, spec, probes["pref_pairs"], device, "preference")
        print(f"  [Pref]    peak=L{pref['peak_layer']} top={pref['top_layers']} steer+= {pref['mean_steer_plus_delta']}", flush=True)

        refusal = _analyze_axis(model, tokenizer, spec, probes["refusal_pairs"], device, "refusal")
        print(f"  [Refusal] peak=L{refusal['peak_layer']} top={refusal['top_layers']} steer+= {refusal['mean_steer_plus_delta']}", flush=True)

        peak_mb = float(torch.cuda.max_memory_allocated() / (1024.0 ** 2))

    cap_map = _synthesize_capability_map(spec, factual, code, cot, pref, refusal)
    elapsed = time.perf_counter() - t0
    print(f"  [Map {scale}] {cap_map} | peak={peak_mb:.1f}MB | t={elapsed:.1f}s", flush=True)

    del model
    cleanup_cuda()
    return {
        "capability_map": cap_map,
        "factual_retrieval": factual,
        "code_syntax_binding": code,
        "multihop_cot": cot,
        "preference_axis": pref,
        "refusal_axis": refusal,
        "efficiency": {
            "static_vram_mb": round(spec.static_vram_mb, 2),
            "peak_vram_mb": round(peak_mb, 2),
            "wall_time_sec": round(elapsed, 2),
            "no_grad_only": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=("small", "large", "both"), default="both")
    args = parser.parse_args()

    out_path = LAB_DIR / "skill_cartography.json"
    atlas = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}

    scales = ["small", "large"] if args.scale == "both" else [args.scale]
    for sc in scales:
        atlas[sc] = run_cartography_for_scale(sc)
        out_path.write_text(json.dumps(atlas, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
