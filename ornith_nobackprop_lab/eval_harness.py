"""Unified Evaluation Harness & JSONL Logger for Ornith-1.0-9B 8-bit Backprop-Free LoRA Lab."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from benchmark import (
    FactBatch,
    get_general_qa_set,
    match_target,
    normalize_answer,
)
from guards import LoRAManager, SynthesisMetrics
from model_loader import get_eos_token_ids

LAB_DIR = Path("/content/ornith_nobackprop_lab")
EXPERIMENTS_JSONL = LAB_DIR / "experiments.jsonl"


def format_qa_prompts(
    tokenizer,
    questions: List[str],
    prefix_context: Optional[str] = None,
    enable_thinking: bool = False,
) -> List[str]:
    prompts = []
    sys_msg = "You are a concise factual assistant. Reply with only the exact requested name, number, or short term."
    for q in questions:
        user_content = f"{prefix_context}\n\nQuestion: {q}" if prefix_context else q
        msgs = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_content},
        ]
        prompts.append(
            tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        )
    return prompts


@torch.no_grad()
def batched_greedy_generate(
    model,
    tokenizer,
    prompts: List[str],
    device: str,
    batch_size: int = 8,
    max_new_tokens: int = 32,
) -> List[str]:
    prev_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    eos_ids = get_eos_token_ids(tokenizer)
    outputs: List[str] = []
    try:
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i : i + batch_size]
            enc = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
            in_len = enc.input_ids.shape[1]
            gen = model.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_ids,
                use_cache=True,
            )
            for row in gen:
                text = tokenizer.decode(row[in_len:], skip_special_tokens=True).strip()
                outputs.append(text)
            del enc, gen
    finally:
        tokenizer.padding_side = prev_padding_side
    return outputs


@contextmanager
def disable_adapters_ctx(model, lora_manager: Optional[LoRAManager] = None):
    """Temporarily evaluate the unedited base model without modifying weights."""
    if lora_manager is not None:
        saved = []
        for w in lora_manager.wrapped.values():
            saved.append((w, w.active))
            w.active = False
        try:
            yield
        finally:
            for w, was_active in saved:
                w.active = was_active
    elif hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            yield
    else:
        yield


@torch.no_grad()
def compute_heldout_kl(
    model,
    tokenizer,
    kl_texts: List[str],
    device: str,
    lora_manager: Optional[LoRAManager] = None,
    batch_size: int = 2,
    max_length: int = 256,
) -> float:
    """Compute exact token-averaged forward KL D_KL(P_base || P_edited) on held-out prompts.

    Uses memory-safe chunking for 248,320-vocab Ornith-1.0-9B inside 15 GiB T4 VRAM.
    """
    prev_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    total_kl = 0.0
    total_tokens = 0
    try:
        for i in range(0, len(kl_texts), batch_size):
            chunk = kl_texts[i : i + batch_size]
            enc = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            mask = enc.attention_mask.bool()
            with disable_adapters_ctx(model, lora_manager):
                base_logits = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False).logits
                bl_valid = base_logits[mask].float()
                del base_logits
                p_base = F.softmax(bl_valid, dim=-1)
                log_p_base = F.log_softmax(bl_valid, dim=-1)
                del bl_valid

            edit_logits = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False).logits
            el_valid = edit_logits[mask].float()
            del edit_logits
            log_p_edit = F.log_softmax(el_valid, dim=-1)
            del el_valid

            token_kl = (p_base * (log_p_base - log_p_edit)).sum(dim=-1)
            total_kl += float(token_kl.sum().item())
            total_tokens += int(mask.sum().item())
            del p_base, log_p_base, log_p_edit, token_kl, enc, mask
    finally:
        tokenizer.padding_side = prev_padding_side

    return float(total_kl / max(total_tokens, 1))


_BASE_REF_CACHE: Dict[tuple, Dict[str, object]] = {}


@torch.no_grad()
def get_base_reference(
    model,
    tokenizer,
    batch: FactBatch,
    device: str,
    lora_manager: Optional[LoRAManager] = None,
) -> Dict[str, object]:
    """Compute (and memoize per seed) Baseline 0 predictions for locality & general QA retention."""
    key = (batch.seed, id(model))
    if key in _BASE_REF_CACHE:
        return _BASE_REF_CACHE[key]

    with disable_adapters_ctx(model, lora_manager):
        general_qa = get_general_qa_set()
        gen_prompts = format_qa_prompts(tokenizer, [x["q"] for x in general_qa])
        gen_preds = batched_greedy_generate(model, tokenizer, gen_prompts, device, batch_size=15)
        base_gen_correct = sum(
            1 for pred, item in zip(gen_preds, general_qa) if match_target(pred, item["a"])
        )
        base_gen_acc = base_gen_correct / len(general_qa)

        loc_prompts = format_qa_prompts(tokenizer, [x.question for x in batch.locality])
        loc_preds = batched_greedy_generate(model, tokenizer, loc_prompts, device, batch_size=10)

        verb_prompts = format_qa_prompts(tokenizer, [f.verbatim_q for f in batch.facts])
        verb_preds = batched_greedy_generate(model, tokenizer, verb_prompts, device, batch_size=10)
        base_verb_acc = sum(
            1 for p, f in zip(verb_preds, batch.facts) if match_target(p, f.target)
        ) / len(batch.facts)

        para_qs = [f.paraphrase_q for f in batch.facts] + [f.paraphrase_q_alt for f in batch.facts]
        para_targets = [f.target for f in batch.facts] * 2
        para_preds = batched_greedy_generate(
            model, tokenizer, format_qa_prompts(tokenizer, para_qs), device, batch_size=10
        )
        base_para_acc = sum(
            1 for p, t in zip(para_preds, para_targets) if match_target(p, t)
        ) / len(para_targets)

    ref = {
        "base_general_qa_acc": base_gen_acc,
        "base_general_preds": gen_preds,
        "base_locality_preds": loc_preds,
        "base_verbatim_acc": base_verb_acc,
        "base_paraphrase_acc": base_para_acc,
    }
    _BASE_REF_CACHE[key] = ref
    return ref


@torch.no_grad()
def evaluate_model(
    model,
    tokenizer,
    batch: FactBatch,
    kl_texts: List[str],
    device: str,
    lora_manager: Optional[LoRAManager] = None,
    prefix_context: Optional[str] = None,
    compute_kl: bool = True,
    base_ref_override: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Run full evaluation suite: verbatim, paraphrase (50 unseen prompts), locality, KL, retention."""
    t0 = time.perf_counter()
    base_ref = (
        base_ref_override
        if base_ref_override is not None
        else get_base_reference(model, tokenizer, batch, device, lora_manager=lora_manager)
    )

    gen_bs = 4 if prefix_context else 12
    verb_prompts = format_qa_prompts(tokenizer, [f.verbatim_q for f in batch.facts], prefix_context=prefix_context)
    verb_preds = batched_greedy_generate(model, tokenizer, verb_prompts, device, batch_size=gen_bs)
    verbatim_acc = sum(
        1 for p, f in zip(verb_preds, batch.facts) if match_target(p, f.target)
    ) / len(batch.facts)

    para_qs = [f.paraphrase_q for f in batch.facts] + [f.paraphrase_q_alt for f in batch.facts]
    para_targets = [f.target for f in batch.facts] * 2
    para_prompts = format_qa_prompts(tokenizer, para_qs, prefix_context=prefix_context)
    para_preds = batched_greedy_generate(model, tokenizer, para_prompts, device, batch_size=gen_bs)
    paraphrase_acc = sum(
        1 for p, t in zip(para_preds, para_targets) if match_target(p, t)
    ) / len(para_targets)

    loc_prompts = format_qa_prompts(tokenizer, [x.question for x in batch.locality], prefix_context=prefix_context)
    # Match get_base_reference batch size EXACTLY: fp16 batching changes reduction order and can
    # flip high-entropy synthetic locality tokens, turning a deterministic comparison into noise.
    loc_preds = batched_greedy_generate(model, tokenizer, loc_prompts, device, batch_size=10)
    injected_targets = [f.target for f in batch.facts]
    loc_ok = 0
    bleed_count = 0
    for item, pred, base_pred in zip(batch.locality, loc_preds, base_ref["base_locality_preds"]):
        bled = any(match_target(pred, tgt) for tgt in injected_targets)
        if bled:
            bleed_count += 1
        if item.is_real_world and item.expected_answer is not None:
            ok = match_target(pred, item.expected_answer) and not bled
        else:
            ok = (normalize_answer(pred) == normalize_answer(base_pred)) and not bled
        if ok:
            loc_ok += 1
    locality_acc = loc_ok / len(batch.locality)
    locality_bleed_rate = bleed_count / len(batch.locality)

    general_qa = get_general_qa_set()
    gen_prompts = format_qa_prompts(tokenizer, [x["q"] for x in general_qa], prefix_context=prefix_context)
    # Match get_base_reference general-QA batch size (15) for a fair retention comparison.
    gen_preds = batched_greedy_generate(model, tokenizer, gen_prompts, device, batch_size=15)
    gen_raw_acc = sum(
        1 for pred, item in zip(gen_preds, general_qa) if match_target(pred, item["a"])
    ) / len(general_qa)
    base_raw = max(float(base_ref["base_general_qa_acc"]), 1e-8)
    retention_acc = min(1.5, gen_raw_acc / base_raw)

    if compute_kl:
        kl_div = compute_heldout_kl(
            model, tokenizer, kl_texts, device, lora_manager=lora_manager, batch_size=2, max_length=256
        )
    else:
        kl_div = 0.0

    eval_sec = time.perf_counter() - t0
    return {
        "verbatim_acc": round(float(verbatim_acc), 4),
        "paraphrase_acc": round(float(paraphrase_acc), 4),
        "locality_acc": round(float(locality_acc), 4),
        "locality_bleed_rate": round(float(locality_bleed_rate), 4),
        "kl_div": round(float(kl_div), 6),
        "retention_acc": round(float(retention_acc), 4),
        "general_qa_raw_acc": round(float(gen_raw_acc), 4),
        "base_general_qa_raw_acc": round(float(base_raw), 4),
        "eval_time_sec": round(float(eval_sec), 2),
    }


def check_gates(
    eval_res: Dict[str, object],
    synth_metrics: SynthesisMetrics,
    baseline_a_ref: Optional[Dict[str, object]] = None,
) -> Dict[str, bool]:
    """Check all HARD retention/locality/budget gates and SUCCESS_THRESHOLD."""
    pass_kl = eval_res["kl_div"] <= 0.05
    pass_ret = eval_res["retention_acc"] >= 0.95
    pass_loc = eval_res["locality_acc"] >= 0.98
    pass_time = synth_metrics.wall_time_sec <= 20.0
    pass_para = eval_res["paraphrase_acc"] >= 0.75
    pass_no_backprop = not synth_metrics.uses_full_backprop

    beats_base_a = True
    if baseline_a_ref is not None:
        beats_base_a = (
            eval_res["paraphrase_acc"] > baseline_a_ref["paraphrase_acc"]
            and eval_res["retention_acc"] >= baseline_a_ref["retention_acc"] - 1e-4
            and eval_res["kl_div"] <= max(0.05, baseline_a_ref["kl_div"])
        )

    all_pass = pass_kl and pass_ret and pass_loc and pass_time and pass_para and pass_no_backprop and beats_base_a
    return {
        "pass_kl": pass_kl,
        "pass_ret": pass_ret,
        "pass_loc": pass_loc,
        "pass_time": pass_time,
        "pass_para": pass_para,
        "pass_no_backprop": pass_no_backprop,
        "beats_base_a": beats_base_a,
        "ALL_GATES_PASS": all_pass,
    }


def append_experiment_record(
    iter_num: int,
    method: str,
    eval_res: Dict[str, object],
    synth_metrics: SynthesisMetrics,
    status: str,
    seed: int = 42,
    stage: int = 1,
    rank: int = 16,
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    record: Dict[str, object] = {
        "iter": iter_num,
        "method": method,
        "seed": seed,
        "stage": stage,
        "rank": rank,
        "verbatim_acc": eval_res["verbatim_acc"],
        "paraphrase_acc": eval_res["paraphrase_acc"],
        "locality_acc": eval_res["locality_acc"],
        "kl_div": eval_res["kl_div"],
        "retention_acc": eval_res["retention_acc"],
        "peak_vram_mb": round(float(synth_metrics.peak_vram_mb), 2),
        "activation_vram_overhead_mb": round(float(synth_metrics.activation_vram_overhead_mb), 2),
        "wall_time_sec": round(float(synth_metrics.wall_time_sec), 3),
        "uses_full_backprop": bool(synth_metrics.uses_full_backprop),
        "status": status,
        "general_qa_raw_acc": eval_res.get("general_qa_raw_acc"),
        "locality_bleed_rate": eval_res.get("locality_bleed_rate"),
    }
    if extra:
        record.update(extra)
    LAB_DIR.mkdir(parents=True, exist_ok=True)
    with EXPERIMENTS_JSONL.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record
