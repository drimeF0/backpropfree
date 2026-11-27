"""Unified Cross-Scale Evaluation Harness & Logger for Biography Transfer Benchmark."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from benchmark import get_general_qa_set, match_target, normalize_answer
from bio_logic_benchmark import BioEvalItem, BioLogicItem, BioSeedBatch
from eval_harness import batched_greedy_generate, compute_heldout_kl, disable_adapters_ctx
from guards import SynthesisMetrics
from unified_model_utils import ModelSpec, apply_chat_safe

LAB_DIR = Path(__file__).resolve().parent
EXPERIMENTS_TRANSFER_JSONL = LAB_DIR / "experiments_transfer.jsonl"

SYS_QA = "You are a concise factual assistant. Reply with only the exact requested name, number, or short term."
SYS_COT = (
    "You are a precise logical reasoning assistant. List only the needed facts in 1-2 short lines, "
    "perform the step-by-step arithmetic or deduction concisely, and end with 'Final Answer: <answer>'."
)

_REFUSAL_RE = re.compile(
    r"(?:does not exist|do not exist|no such|not a real|not real|fictional|no official|"
    r"no record|no information|don't have|do not have|cannot verify|can't verify|"
    r"not mentioned|no documented|\bunknown\b|not publicly known|cannot answer|^none\.?$)",
    re.IGNORECASE,
)


def is_refusal_output(pred: str) -> bool:
    return _REFUSAL_RE.search(pred.strip()) is not None


def strip_thinking_prefix(pred: str) -> str:
    """If a thinking trace ends with </think>, return the final answer section; else full text."""
    if "</think>" in pred:
        tail = pred.split("</think>", 1)[1].strip()
        if tail:
            return tail
    return pred


def check_eval_patterns(pred: str, patterns: Optional[List[str]] = None, require_all: Optional[List[str]] = None) -> bool:
    txt = strip_thinking_prefix(pred).lower()
    if require_all:
        return all(re.search(p, txt) is not None for p in require_all)
    if patterns:
        return any(re.search(p, txt) is not None for p in patterns)
    return False


def format_bio_qa_prompts(
    tokenizer,
    questions: List[str],
    prefix_context: Optional[str] = None,
    enable_thinking: bool = False,
) -> List[str]:
    out = []
    for q in questions:
        content = f"{prefix_context}\n\nQuestion: {q}" if prefix_context else q
        out.append(
            apply_chat_safe(
                tokenizer,
                [{"role": "system", "content": SYS_QA}, {"role": "user", "content": content}],
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        )
    return out


def format_bio_cot_prompts(
    tokenizer,
    questions: List[str],
    prefix_context: Optional[str] = None,
    enable_thinking: bool = False,
) -> List[str]:
    out = []
    for q in questions:
        content = f"{prefix_context}\n\nQuestion: {q}" if prefix_context else q
        out.append(
            apply_chat_safe(
                tokenizer,
                [{"role": "system", "content": SYS_COT}, {"role": "user", "content": content}],
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        )
    return out


@torch.no_grad()
def first_token_diagnostics(
    model,
    tokenizer,
    prompts: List[str],
    targets: List[str],
    device: str,
    batch_size: int = 12,
) -> Tuple[float, float, List[int], List[int]]:
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    top1_all: List[int] = []
    p0_sum = 0.0
    exp0_all = [tokenizer.encode(t, add_special_tokens=False)[0] for t in targets]
    try:
        for s in range(0, len(prompts), batch_size):
            e = min(len(prompts), s + batch_size)
            enc = tokenizer(prompts[s:e], return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
            logits = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False).logits[:, -1, :].float()
            probs = torch.softmax(logits, dim=-1)
            top1 = logits.argmax(dim=-1).tolist()
            top1_all.extend(top1)
            for j, tid in enumerate(exp0_all[s:e]):
                p0_sum += float(probs[j, tid].item())
            del enc, logits, probs
    finally:
        tokenizer.padding_side = prev_side
    n = max(len(exp0_all), 1)
    acc0 = sum(a == b for a, b in zip(top1_all, exp0_all)) / n
    return acc0, p0_sum / n, top1_all, exp0_all


_BIO_BASE_REF_CACHE: Dict[Tuple[int, str, int], Dict[str, object]] = {}


@torch.no_grad()
def get_bio_base_reference(
    model,
    tokenizer,
    spec: ModelSpec,
    batch: BioSeedBatch,
    device: str,
    lora_manager=None,
) -> Dict[str, object]:
    """Compute and memoize matched-batch unedited Base reference per (seed, model_id, id(model)) + disk cache."""
    general_qa = get_general_qa_set()
    sig_payload = json.dumps({
        "model_id": spec.model_id,
        "seed": batch.seed,
        "loc_qs": [x.question for x in batch.locality_eval],
        "gen_qs": [x["q"] for x in general_qa],
    }, sort_keys=True)
    sig = hashlib.sha256(sig_payload.encode()).hexdigest()[:12]
    key = (batch.seed, spec.model_id, id(model))
    if key in _BIO_BASE_REF_CACHE and _BIO_BASE_REF_CACHE[key].get("sig") == sig:
        return _BIO_BASE_REF_CACHE[key]

    cache_path = LAB_DIR / f"bio_base_ref_{spec.scale}_s{batch.seed}.json"
    if cache_path.exists():
        try:
            disk_ref = json.loads(cache_path.read_text())
            if disk_ref.get("sig") == sig:
                _BIO_BASE_REF_CACHE[key] = disk_ref
                return disk_ref
        except Exception:
            pass

    with disable_adapters_ctx(model, lora_manager):
        gen_prompts = format_bio_qa_prompts(tokenizer, [x["q"] for x in general_qa], enable_thinking=False)
        gen_preds = batched_greedy_generate(model, tokenizer, gen_prompts, device, batch_size=15, max_new_tokens=32)
        base_gen_acc = sum(match_target(pr, x["a"]) for pr, x in zip(gen_preds, general_qa)) / len(general_qa)

        loc_prompts = format_bio_qa_prompts(tokenizer, [x.question for x in batch.locality_eval], enable_thinking=False)
        loc_preds = batched_greedy_generate(model, tokenizer, loc_prompts, device, batch_size=10, max_new_tokens=32)

    ref = {
        "sig": sig,
        "base_general_qa_acc": base_gen_acc,
        "base_general_preds": gen_preds,
        "base_locality_preds": loc_preds,
    }
    _BIO_BASE_REF_CACHE[key] = ref
    try:
        cache_path.write_text(json.dumps(ref, indent=2))
    except Exception:
        pass
    return ref


def _split_failure_taxonomy(
    items: List[BioEvalItem],
    preds: List[str],
    top1_ids: List[int],
    exp0_ids: List[int],
) -> Dict[str, object]:
    full_ok = 0
    right_pos0_wrong_suffix = 0
    wrong_pos0 = 0
    refusals = 0
    details = []
    for item, pred, t0, e0 in zip(items, preds, top1_ids, exp0_ids):
        ok = check_eval_patterns(pred, patterns=item.patterns)
        ref = is_refusal_output(pred)
        if ref:
            refusals += 1
        if ok:
            full_ok += 1
            cat = "pass"
        elif t0 == e0:
            right_pos0_wrong_suffix += 1
            cat = "suffix_decay"
        else:
            wrong_pos0 += 1
            cat = "refusal" if ref else "wrong_pos0"
        details.append({
            "key": item.fact_key,
            "framing": item.framing,
            "target": item.target,
            "pred": pred,
            "pass": ok,
            "category": cat,
        })
    n = max(len(items), 1)
    return {
        "acc": round(full_ok / n, 4),
        "first_tok_acc": round(sum(a == b for a, b in zip(top1_ids, exp0_ids)) / n, 4),
        "refusal_rate": round(refusals / n, 4),
        "right_pos0_wrong_suffix": right_pos0_wrong_suffix,
        "wrong_pos0": wrong_pos0,
        "items": details,
    }


@torch.no_grad()
def evaluate_bio_transfer(
    model,
    tokenizer,
    spec: ModelSpec,
    batch: BioSeedBatch,
    kl_texts: List[str],
    device: str,
    lora_manager=None,
    prefix_bio: bool = False,
    compute_kl: bool = True,
    eval_thinking_diag: bool = False,
    eval_self_recalled: bool = True,
    base_ref_override: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Run full matched-batch cross-scale evaluation suite on a BioSeedBatch."""
    t0 = time.perf_counter()
    base_ref = (
        base_ref_override
        if base_ref_override is not None
        else get_bio_base_reference(model, tokenizer, spec, batch, device, lora_manager=lora_manager)
    )
    prefix = f"Biography Dossier:\n{batch.biography_text}" if prefix_bio else None
    gen_bs = 4 if prefix_bio else (8 if spec.scale == "large" else 16)

    # 1. Direct Recall (12 unseen questions, enable_thinking=False)
    d_prompts = format_bio_qa_prompts(tokenizer, [x.question for x in batch.direct_eval], prefix_context=prefix, enable_thinking=False)
    d_preds = batched_greedy_generate(model, tokenizer, d_prompts, device, batch_size=gen_bs, max_new_tokens=32)
    _, d_p0, d_top1, d_exp0 = first_token_diagnostics(model, tokenizer, d_prompts, [x.target for x in batch.direct_eval], device, batch_size=gen_bs)
    d_tax = _split_failure_taxonomy(batch.direct_eval, d_preds, d_top1, d_exp0)

    # 2. Paraphrase & Post-Entity Framing (24 unseen questions, enable_thinking=False)
    p_prompts = format_bio_qa_prompts(tokenizer, [x.question for x in batch.paraphrase_eval], prefix_context=prefix, enable_thinking=False)
    p_preds = batched_greedy_generate(model, tokenizer, p_prompts, device, batch_size=gen_bs, max_new_tokens=32)
    _, p_p0, p_top1, p_exp0 = first_token_diagnostics(model, tokenizer, p_prompts, [x.target for x in batch.paraphrase_eval], device, batch_size=gen_bs)
    p_tax = _split_failure_taxonomy(batch.paraphrase_eval, p_preds, p_top1, p_exp0)

    para_only = [x for x in p_tax["items"] if x["framing"] == "paraphrase"]
    post_only = [x for x in p_tax["items"] if x["framing"] == "post_entity"]
    para_only_acc = round(sum(x["pass"] for x in para_only) / max(len(para_only), 1), 4)
    post_entity_acc = round(sum(x["pass"] for x in post_only) / max(len(post_only), 1), 4)

    # 3. Weight-Only Multi-Hop & Temporal Logic (enable_thinking=False + optional True + batched self_recalled)
    cot_bs = 4 if prefix_bio else (9 if spec.scale == "large" else 9)
    l_prompts = format_bio_cot_prompts(tokenizer, [x.question for x in batch.logic_eval], prefix_context=prefix, enable_thinking=False)
    l_preds = batched_greedy_generate(model, tokenizer, l_prompts, device, batch_size=cot_bs, max_new_tokens=160)
    l_items = []
    for item, pred in zip(batch.logic_eval, l_preds):
        ok = check_eval_patterns(pred, patterns=item.patterns, require_all=item.require_all)
        l_items.append({"name": item.name, "type": item.logic_type, "pred": pred, "pass": ok})
    logic_1turn_acc = round(sum(x["pass"] for x in l_items) / len(l_items), 4)

    logic_thinking_acc = None
    logic_thinking_items = []
    if eval_thinking_diag and spec.supports_thinking:
        lt_prompts = format_bio_cot_prompts(tokenizer, [x.question for x in batch.logic_eval], prefix_context=prefix, enable_thinking=True)
        lt_preds = batched_greedy_generate(model, tokenizer, lt_prompts, device, batch_size=3 if prefix_bio else 9, max_new_tokens=384)
        for item, pred in zip(batch.logic_eval, lt_preds):
            ok = check_eval_patterns(pred, patterns=item.patterns, require_all=item.require_all)
            logic_thinking_items.append({"name": item.name, "type": item.logic_type, "pred": pred, "pass": ok})
        logic_thinking_acc = round(sum(x["pass"] for x in logic_thinking_items) / len(logic_thinking_items), 4)

    sr_items = []
    sr_acc = None
    if eval_self_recalled and not prefix_bio:
        flat_sub_qs: List[str] = []
        sub_slices: List[Tuple[int, int]] = []
        for item in batch.logic_eval:
            s_idx = len(flat_sub_qs)
            flat_sub_qs.extend(item.sub_qs)
            sub_slices.append((s_idx, len(flat_sub_qs)))
        sq_prompts_all = format_bio_qa_prompts(tokenizer, flat_sub_qs, enable_thinking=False)
        sq_preds_all = batched_greedy_generate(
            model,
            tokenizer,
            sq_prompts_all,
            device,
            batch_size=12 if spec.scale == "large" else 24,
            max_new_tokens=24,
        )
        recalled_ctxs: List[str] = []
        ded_questions: List[str] = []
        for item, (s_idx, e_idx) in zip(batch.logic_eval, sub_slices):
            sq_preds = sq_preds_all[s_idx:e_idx]
            recalled_ctx = "\n".join(f"- {sq} -> {sa}" for sq, sa in zip(item.sub_qs, sq_preds))
            recalled_ctxs.append(recalled_ctx)
            ded_questions.append(f"Facts recalled from memory:\n{recalled_ctx}\n\nQuestion: {item.question}")
        ded_prompts = format_bio_cot_prompts(tokenizer, ded_questions, enable_thinking=False)
        ded_preds = batched_greedy_generate(model, tokenizer, ded_prompts, device, batch_size=9, max_new_tokens=160)
        for item, recalled_ctx, ded_pred in zip(batch.logic_eval, recalled_ctxs, ded_preds):
            ok = check_eval_patterns(ded_pred, patterns=item.patterns, require_all=item.require_all)
            sr_items.append({"name": item.name, "recalled": recalled_ctx, "pred": ded_pred, "pass": ok})
        sr_acc = round(sum(x["pass"] for x in sr_items) / len(sr_items), 4)

    # 4. Locality & Sibling Bleed (EXACT matched batch_size=10)
    loc_prompts = format_bio_qa_prompts(tokenizer, [x.question for x in batch.locality_eval], prefix_context=prefix, enable_thinking=False)
    loc_preds = batched_greedy_generate(model, tokenizer, loc_prompts, device, batch_size=10, max_new_tokens=32)

    real_ok = 0
    real_tot = 0
    sib_tot = 0
    sib_bleed_raw = 0
    sib_bleed_excess = 0
    sib_exact_match = 0
    for item, pred, bpred in zip(batch.locality_eval, loc_preds, base_ref["base_locality_preds"]):
        low_p = pred.lower()
        low_b = bpred.lower()
        bled_raw = any(re.search(pat, low_p) is not None for _, pat, _ in batch.target_bleed_patterns)
        bled_excess = any(
            (re.search(pat, low_p) is not None) and (not is_num or re.search(pat, low_b) is None)
            for _, pat, is_num in batch.target_bleed_patterns
        )
        if item.is_real_world:
            real_tot += 1
            if check_eval_patterns(pred, patterns=item.expected_patterns) and not bled_excess:
                real_ok += 1
        else:
            sib_tot += 1
            if bled_raw:
                sib_bleed_raw += 1
            if bled_excess:
                sib_bleed_excess += 1
            if normalize_answer(pred) == normalize_answer(bpred):
                sib_exact_match += 1

    real_anchors_acc = round(real_ok / max(real_tot, 1), 4)
    synth_bleed = round(sib_bleed_excess / max(sib_tot, 1), 4)
    synth_bleed_raw = round(sib_bleed_raw / max(sib_tot, 1), 4)
    synth_exact_match = round(sib_exact_match / max(sib_tot, 1), 4)

    # 5. General QA Retention (EXACT matched batch_size=15) & Held-out KL
    general_qa = get_general_qa_set()
    gen_prompts = format_bio_qa_prompts(tokenizer, [x["q"] for x in general_qa], prefix_context=prefix, enable_thinking=False)
    gen_preds = batched_greedy_generate(model, tokenizer, gen_prompts, device, batch_size=15, max_new_tokens=32)
    gen_raw_acc = sum(match_target(pr, x["a"]) for pr, x in zip(gen_preds, general_qa)) / len(general_qa)
    base_raw = max(float(base_ref["base_general_qa_acc"]), 1e-8)
    retention_acc = min(1.5, gen_raw_acc / base_raw)

    if compute_kl:
        kl_bs = 2 if spec.scale == "large" else 8
        kl_div = compute_heldout_kl(model, tokenizer, kl_texts, device, lora_manager=lora_manager, batch_size=kl_bs, max_length=256)
    else:
        kl_div = 0.0

    eval_sec = time.perf_counter() - t0
    return {
        "direct_recall_acc": d_tax["acc"],
        "direct_first_tok_acc": d_tax["first_tok_acc"],
        "direct_first_tok_prob": round(d_p0, 4),
        "direct_refusal_rate": d_tax["refusal_rate"],
        "direct_suffix_decay": d_tax["right_pos0_wrong_suffix"],
        "direct_wrong_pos0": d_tax["wrong_pos0"],
        "paraphrase_acc": p_tax["acc"],
        "paraphrase_only_acc": para_only_acc,
        "post_entity_acc": post_entity_acc,
        "paraphrase_first_tok_acc": p_tax["first_tok_acc"],
        "paraphrase_first_tok_prob": round(p_p0, 4),
        "paraphrase_refusal_rate": p_tax["refusal_rate"],
        "paraphrase_suffix_decay": p_tax["right_pos0_wrong_suffix"],
        "paraphrase_wrong_pos0": p_tax["wrong_pos0"],
        "logic_1turn_acc": logic_1turn_acc,
        "logic_thinking_acc": logic_thinking_acc,
        "logic_self_recalled_acc": sr_acc,
        "real_anchors_acc": real_anchors_acc,
        "synth_bleed": synth_bleed,
        "synth_bleed_raw": synth_bleed_raw,
        "synth_exact_match": synth_exact_match,
        "kl_div": round(float(kl_div), 6),
        "retention_acc": round(float(retention_acc), 4),
        "general_qa_raw_acc": round(float(gen_raw_acc), 4),
        "base_general_qa_raw_acc": round(float(base_raw), 4),
        "eval_time_sec": round(float(eval_sec), 2),
        "details": {
            "direct": d_tax["items"],
            "paraphrase": p_tax["items"],
            "logic_1turn": l_items,
            "logic_thinking": logic_thinking_items,
            "logic_self_recalled": sr_items,
            "locality": [
                {"entity": it.entity, "real": it.is_real_world, "q": it.question, "pred": pr}
                for it, pr in zip(batch.locality_eval, loc_preds)
            ],
        },
    }


def check_transfer_gates(
    eval_res: Dict[str, object],
    synth_metrics: SynthesisMetrics,
    scale: str,
    rank: int,
    base_logic_acc: float,
    oracle_logic_acc: float,
) -> Dict[str, bool]:
    direct_th = 0.75 if scale == "small" else 0.70
    para_th = 0.70 if scale == "small" else 0.65
    time_th = 20.0 if scale == "small" else 120.0

    pass_direct = eval_res["direct_recall_acc"] >= direct_th
    pass_para = eval_res["paraphrase_acc"] >= para_th
    # Weight-only logic must strictly beat Base and approach InContext Oracle ceiling
    logic_val = float(eval_res["logic_1turn_acc"])
    logic_target = max(base_logic_acc + 0.10, min(oracle_logic_acc, oracle_logic_acc * 0.80))
    pass_logic = (logic_val > base_logic_acc) and (logic_val >= logic_target - 1e-6)
    pass_real = eval_res["real_anchors_acc"] >= 0.95
    pass_bleed = eval_res["synth_bleed"] <= 0.05
    pass_kl = eval_res["kl_div"] <= 0.05
    pass_ret = eval_res["retention_acc"] >= 0.95
    pass_vram = (synth_metrics.activation_vram_overhead_mb <= 2500.0) if scale == "large" else True
    pass_time = synth_metrics.wall_time_sec <= time_th
    pass_arch = (not synth_metrics.uses_full_backprop) and (rank <= 16)

    all_pass = (
        pass_direct and pass_para and pass_logic and pass_real and pass_bleed
        and pass_kl and pass_ret and pass_vram and pass_time and pass_arch
    )
    return {
        "pass_direct": pass_direct,
        "pass_para": pass_para,
        "pass_logic": pass_logic,
        "pass_real_anchors": pass_real,
        "pass_synth_bleed": pass_bleed,
        "pass_kl": pass_kl,
        "pass_retention": pass_ret,
        "pass_vram": pass_vram,
        "pass_time": pass_time,
        "pass_no_backprop_rank16": pass_arch,
        "ALL_GATES_PASS": all_pass,
    }


def append_transfer_record(
    iter_num: int,
    method: str,
    scale: str,
    model_id: str,
    eval_res: Dict[str, object],
    synth_metrics: SynthesisMetrics,
    status: str,
    seed: int = 42,
    stage: int = 1,
    rank: int = 16,
    gates: Optional[Dict[str, bool]] = None,
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    summary_eval = {k: v for k, v in eval_res.items() if k != "details"}
    record: Dict[str, object] = {
        "iter": iter_num,
        "method": method,
        "scale": scale,
        "model_id": model_id,
        "seed": seed,
        "stage": stage,
        "rank": rank,
        **summary_eval,
        "peak_vram_mb": round(float(synth_metrics.peak_vram_mb), 2),
        "activation_vram_overhead_mb": round(float(synth_metrics.activation_vram_overhead_mb), 2),
        "wall_time_sec": round(float(synth_metrics.wall_time_sec), 3),
        "uses_full_backprop": bool(synth_metrics.uses_full_backprop),
        "regime": synth_metrics.regime,
        "status": status,
    }
    if gates is not None:
        record["gates"] = gates
    if extra:
        record.update(extra)
    LAB_DIR.mkdir(parents=True, exist_ok=True)
    with EXPERIMENTS_TRANSFER_JSONL.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record
