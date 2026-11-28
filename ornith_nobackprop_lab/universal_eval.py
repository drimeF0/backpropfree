"""Unified Anti-Cheat Evaluation Harness & JSONL Logger for Agent #3 (3 Domains + Sandbox + Shared Suite)."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from benchmark import match_target, normalize_answer
from eval_harness import disable_adapters_ctx
from guards import SynthesisMetrics
from model_loader import get_eos_token_ids
from universal_benchmark import get_domain_batch, get_shared_retention_suite
from universal_sandbox import run_code_in_sandbox, strip_thinking
from unified_model_utils import ModelSpec, apply_chat_safe


@torch.no_grad()
def batched_greedy_generate(
    model,
    tokenizer,
    prompts: List[str],
    device: str,
    batch_size: int = 8,
    max_new_tokens: int = 32,
    max_input_length: int = 2800,
) -> List[str]:
    """Memory-safe left-padded greedy generation supporting 2048-tok InContext_Oracle without prompt truncation."""
    prev_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    eos_ids = get_eos_token_ids(tokenizer)
    outputs: List[str] = []
    try:
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i : i + batch_size]
            enc = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_length,
            ).to(device)
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
                outputs.append(tokenizer.decode(row[in_len:], skip_special_tokens=True).strip())
            del enc, gen
    finally:
        tokenizer.padding_side = prev_padding_side
    return outputs

LAB_DIR = Path(__file__).resolve().parent
EXPERIMENTS_UNIVERSAL_JSONL = LAB_DIR / "experiments_universal.jsonl"

SYS_QA = "You are a concise factual assistant. Reply with only the exact requested name, number, or short term."
SYS_CODE = (
    "You are a precise Python software engineer. Write clean, self-contained Python code implementing the exact "
    "requested function inside a ```python code block with no extra commentary."
)
SYS_COT = (
    "You are a precise logical reasoning assistant. Work through the facts and arithmetic concisely, "
    "and end with 'Final Answer: <answer>'."
)

_REFUSAL_RE = re.compile(
    r"(?:does not exist|do not exist|no such|not a real|not real|fictional|no official|"
    r"no record|no information|don't have|do not have|cannot verify|can't verify|"
    r"not mentioned|no documented|\bunknown\b|not publicly known|cannot answer|^none\.?$|unverified|non-existent)",
    re.IGNORECASE,
)


def is_refusal_output(pred: str) -> bool:
    return _REFUSAL_RE.search(strip_thinking(pred)) is not None


def check_patterns(pred: str, patterns: Optional[List[str]] = None, require_all: Optional[List[str]] = None) -> bool:
    txt = strip_thinking(pred).lower()
    if require_all:
        return all(re.search(p, txt) is not None for p in require_all)
    if patterns:
        return any(re.search(p, txt) is not None for p in patterns)
    return False


def format_prompts(
    tokenizer,
    user_queries: List[str],
    sys_msg: str,
    prefix_context: Optional[str] = None,
    enable_thinking: bool = False,
) -> List[str]:
    out = []
    for q in user_queries:
        content = f"{prefix_context}\n\n{q}" if prefix_context else q
        out.append(
            apply_chat_safe(
                tokenizer,
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": content}],
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        )
    return out


@torch.no_grad()
def compute_heldout_kl_safe(
    model,
    tokenizer,
    kl_texts: List[str],
    device: str,
    lora_manager=None,
    batch_size: int = 2,
    max_length: int = 256,
    tok_chunk: int = 128,
) -> float:
    """Memory-safe forward KL D_KL(P_base || P_edited) chunked over tokens for 248k-vocab 9B."""
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    total_kl = 0.0
    total_tokens = 0
    try:
        for i in range(0, len(kl_texts), batch_size):
            chunk = kl_texts[i : i + batch_size]
            enc = tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_length
            ).to(device)
            mask = enc.attention_mask.bool()
            with disable_adapters_ctx(model, lora_manager):
                bl_all = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False).logits[mask]
            el_all = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False).logits[mask]
            n_tok = bl_all.shape[0]
            for s in range(0, n_tok, tok_chunk):
                e = min(n_tok, s + tok_chunk)
                bl_f = bl_all[s:e].float()
                el_f = el_all[s:e].float()
                p_base = F.softmax(bl_f, dim=-1)
                lp_base = F.log_softmax(bl_f, dim=-1)
                lp_edit = F.log_softmax(el_f, dim=-1)
                total_kl += float((p_base * (lp_base - lp_edit)).sum().item())
                del bl_f, el_f, p_base, lp_base, lp_edit
            total_tokens += int(n_tok)
            del bl_all, el_all, enc, mask
    finally:
        tokenizer.padding_side = prev_side
    return float(total_kl / max(total_tokens, 1))


@torch.no_grad()
def compute_conditional_logps(
    model,
    tokenizer,
    prompts: List[str],
    responses: List[str],
    device: str,
    batch_size: int = 4,
    max_length: int = 384,
) -> List[float]:
    """Compute sequence-summed log P(response | prompt) under torch.no_grad()."""
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    logps: List[float] = []
    try:
        full_seqs: List[List[int]] = []
        resp_starts: List[int] = []
        for pr, rs in zip(prompts, responses):
            p_ids = tokenizer.encode(pr, add_special_tokens=False)
            r_ids = tokenizer.encode(rs, add_special_tokens=False)
            if tokenizer.eos_token_id is not None:
                r_ids = r_ids + [int(tokenizer.eos_token_id)]
            comb = (p_ids + r_ids)[:max_length]
            full_seqs.append(comb)
            resp_starts.append(min(len(p_ids), len(comb)))

        pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        for s in range(0, len(full_seqs), batch_size):
            e = min(len(full_seqs), s + batch_size)
            sub = full_seqs[s:e]
            max_l = max(len(x) for x in sub)
            inp = torch.full((len(sub), max_l), pad_id, dtype=torch.long, device=device)
            msk = torch.zeros((len(sub), max_l), dtype=torch.long, device=device)
            for idx, seq in enumerate(sub):
                inp[idx, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                msk[idx, : len(seq)] = 1
            logits = model(input_ids=inp, attention_mask=msk, use_cache=False).logits
            for idx, seq in enumerate(sub):
                st = resp_starts[s + idx]
                if st >= len(seq):
                    logps.append(0.0)
                    continue
                pred_pos = torch.arange(max(0, st - 1), len(seq) - 1, device=device)
                tgt_ids = inp[idx, st:len(seq)]
                sl_logits = logits[idx, pred_pos, :].float()
                lp = F.log_softmax(sl_logits, dim=-1)
                tok_lp = lp[torch.arange(tgt_ids.shape[0], device=device), tgt_ids]
                logps.append(float(tok_lp.sum().item()))
            del inp, msk, logits
    finally:
        tokenizer.padding_side = prev_side
    return logps


_SHARED_BASE_CACHE: Dict[str, Dict[str, object]] = {}


@torch.no_grad()
def get_shared_base_reference(
    model,
    tokenizer,
    spec: ModelSpec,
    device: str,
    lora_manager=None,
) -> Dict[str, object]:
    """Compute/load matched-batch unedited Base reference on general_qa & real_anchors."""
    suite = get_shared_retention_suite()
    sig_payload = json.dumps(
        {
            "model_id": spec.model_id,
            "n_gen": len(suite["general_qa"]),
            "n_rf": len(suite["real_fact_anchors"]),
            "n_rc": len(suite["real_code_anchors"]),
        },
        sort_keys=True,
    )
    sig = hashlib.sha256(sig_payload.encode()).hexdigest()[:12]
    if spec.scale in _SHARED_BASE_CACHE and _SHARED_BASE_CACHE[spec.scale].get("sig") == sig:
        return _SHARED_BASE_CACHE[spec.scale]

    cache_path = LAB_DIR / f"universal_base_ref_{spec.scale}.json"
    if cache_path.exists():
        disk = json.loads(cache_path.read_text(encoding="utf-8"))
        if disk.get("sig") == sig:
            _SHARED_BASE_CACHE[spec.scale] = disk
            return disk

    gen_bs = 15
    qa_bs = 10
    code_bs = 5
    with disable_adapters_ctx(model, lora_manager):
        gen_prompts = format_prompts(tokenizer, [x["q"] for x in suite["general_qa"]], SYS_QA)
        gen_preds = batched_greedy_generate(model, tokenizer, gen_prompts, device, batch_size=gen_bs, max_new_tokens=32)
        base_gen_acc = sum(match_target(p, x["a"]) for p, x in zip(gen_preds, suite["general_qa"])) / len(suite["general_qa"])

        rf_prompts = format_prompts(tokenizer, [x["q"] for x in suite["real_fact_anchors"]], SYS_QA)
        rf_preds = batched_greedy_generate(model, tokenizer, rf_prompts, device, batch_size=qa_bs, max_new_tokens=32)
        base_rf_acc = sum(check_patterns(p, x["patterns"]) for p, x in zip(rf_preds, suite["real_fact_anchors"])) / len(suite["real_fact_anchors"])

        rc_prompts = format_prompts(tokenizer, [x["prompt"] for x in suite["real_code_anchors"]], SYS_CODE)
        rc_preds = batched_greedy_generate(model, tokenizer, rc_prompts, device, batch_size=code_bs, max_new_tokens=160)
        rc_ok = sum(
            run_code_in_sandbox(pr, t["test_code"], expected_fn=t["fn_name"]).passed
            for pr, t in zip(rc_preds, suite["real_code_anchors"])
        )
        base_rc_acc = rc_ok / len(suite["real_code_anchors"])

    ref = {
        "sig": sig,
        "base_general_qa_acc": round(float(base_gen_acc), 4),
        "base_real_fact_acc": round(float(base_rf_acc), 4),
        "base_real_code_acc": round(float(base_rc_acc), 4),
    }
    _SHARED_BASE_CACHE[spec.scale] = ref
    cache_path.write_text(json.dumps(ref, indent=2), encoding="utf-8")
    return ref


@torch.no_grad()
def evaluate_shared_retention(
    model,
    tokenizer,
    spec: ModelSpec,
    kl_texts: List[str],
    device: str,
    lora_manager=None,
    compute_kl: bool = True,
    prefix_context: Optional[str] = None,
) -> Dict[str, object]:
    """Evaluate untouched 25 real_anchors, 60 general_qa retention, and held-out KL."""
    base_ref = get_shared_base_reference(model, tokenizer, spec, device, lora_manager=lora_manager)
    suite = get_shared_retention_suite()

    gen_bs = 4 if prefix_context else 15
    qa_bs = 4 if prefix_context else 10
    code_bs = 2 if prefix_context else 5

    gen_prompts = format_prompts(tokenizer, [x["q"] for x in suite["general_qa"]], SYS_QA, prefix_context=prefix_context)
    gen_preds = batched_greedy_generate(model, tokenizer, gen_prompts, device, batch_size=gen_bs, max_new_tokens=32)
    gen_raw_acc = sum(match_target(p, x["a"]) for p, x in zip(gen_preds, suite["general_qa"])) / len(suite["general_qa"])
    retention = min(1.5, gen_raw_acc / max(float(base_ref["base_general_qa_acc"]), 1e-8))

    rf_prompts = format_prompts(tokenizer, [x["q"] for x in suite["real_fact_anchors"]], SYS_QA, prefix_context=prefix_context)
    rf_preds = batched_greedy_generate(model, tokenizer, rf_prompts, device, batch_size=qa_bs, max_new_tokens=32)
    rf_ok = sum(check_patterns(p, x["patterns"]) for p, x in zip(rf_preds, suite["real_fact_anchors"]))

    rc_prompts = format_prompts(tokenizer, [x["prompt"] for x in suite["real_code_anchors"]], SYS_CODE, prefix_context=prefix_context)
    rc_preds = batched_greedy_generate(model, tokenizer, rc_prompts, device, batch_size=code_bs, max_new_tokens=160)
    rc_ok = sum(
        run_code_in_sandbox(pr, t["test_code"], expected_fn=t["fn_name"]).passed
        for pr, t in zip(rc_preds, suite["real_code_anchors"])
    )

    total_ra = len(suite["real_fact_anchors"]) + len(suite["real_code_anchors"])
    real_anchors_acc = (rf_ok + rc_ok) / max(total_ra, 1)

    if compute_kl:
        kl_bs = 2 if spec.scale == "large" else 8
        kl_div = compute_heldout_kl_safe(model, tokenizer, kl_texts, device, lora_manager=lora_manager, batch_size=kl_bs, max_length=256)
    else:
        kl_div = 0.0

    return {
        "real_anchors_acc": round(float(real_anchors_acc), 4),
        "real_fact_acc": round(float(rf_ok / len(suite["real_fact_anchors"])), 4),
        "real_code_acc": round(float(rc_ok / len(suite["real_code_anchors"])), 4),
        "general_qa_raw_acc": round(float(gen_raw_acc), 4),
        "retention_acc": round(float(retention), 4),
        "kl_div": round(float(kl_div), 6),
    }


_BASE_SIB_CACHE: Dict[Tuple[str, str, int], List[str]] = {}


def _get_base_sibling_preds(model, tokenizer, spec: ModelSpec, domain: str, seed: int, questions: List[str], device: str, lora_manager=None) -> List[str]:
    key = (spec.scale, domain, seed)
    if key in _BASE_SIB_CACHE:
        return _BASE_SIB_CACHE[key]
    sib_prompts = format_prompts(tokenizer, questions, SYS_QA, prefix_context=None)
    qa_bs = 8 if spec.scale == "large" else 16
    with disable_adapters_ctx(model, lora_manager):
        preds = batched_greedy_generate(model, tokenizer, sib_prompts, device, batch_size=qa_bs, max_new_tokens=32)
    _BASE_SIB_CACHE[key] = preds
    return preds


@torch.no_grad()
def evaluate_domain(
    model,
    tokenizer,
    spec: ModelSpec,
    domain: str,
    seed: int,
    device: str,
    prefix_doc: bool = False,
    lora_manager=None,
) -> Dict[str, object]:
    """Evaluate one domain & seed cleanly with matched batch sizes."""
    batch = get_domain_batch(domain, seed)
    prefix = f"Reference Document:\n{batch['doc_text']}" if prefix_doc else None
    qa_bs = 2 if prefix_doc else (8 if spec.scale == "large" else 16)
    code_bs = 2 if prefix_doc else (3 if spec.scale == "large" else 6)

    if domain == "code_api":
        qa_prompts = format_prompts(tokenizer, [x["q"] for x in batch["api_qa"]], SYS_QA, prefix_context=prefix)
        qa_preds = batched_greedy_generate(model, tokenizer, qa_prompts, device, batch_size=qa_bs, max_new_tokens=36)
        api_qa_acc = sum(check_patterns(p, x["patterns"]) for p, x in zip(qa_preds, batch["api_qa"])) / len(batch["api_qa"])
        refusal_rate = sum(is_refusal_output(p) for p in qa_preds) / len(qa_preds)

        c_prompts = format_prompts(tokenizer, [x["prompt"] for x in batch["code_tasks"]], SYS_CODE, prefix_context=prefix)
        c_preds = batched_greedy_generate(model, tokenizer, c_prompts, device, batch_size=code_bs, max_new_tokens=220)
        c_results = [
            run_code_in_sandbox(
                pr,
                t["test_code"],
                expected_fn=t["fn_name"],
                mock_module_name=batch["mock_module_name"],
                mock_module_source=batch["mock_module_source"],
            )
            for pr, t in zip(c_preds, batch["code_tasks"])
        ]
        code_pass1 = sum(r.passed for r in c_results) / len(c_results)

        neg_prompts = format_prompts(tokenizer, [x["prompt"] for x in batch["neg_controls"]], SYS_CODE, prefix_context=prefix)
        neg_preds = batched_greedy_generate(model, tokenizer, neg_prompts, device, batch_size=code_bs, max_new_tokens=160)
        neg_results = [
            run_code_in_sandbox(pr, t["test_code"], expected_fn=t["fn_name"])
            for pr, t in zip(neg_preds, batch["neg_controls"])
        ]
        neg_ctrl_pass1 = sum(r.passed for r in neg_results) / len(neg_results)

        return {
            "domain": domain,
            "seed": seed,
            "api_qa_acc": round(float(api_qa_acc), 4),
            "code_pass1": round(float(code_pass1), 4),
            "neg_ctrl_pass1": round(float(neg_ctrl_pass1), 4),
            "refusal_rate": round(float(refusal_rate), 4),
            "code_errors": [r.error_type for r in c_results],
        }

    if domain == "pop_culture":
        p_prompts = format_prompts(tokenizer, [x["q"] for x in batch["paraphrase_qa"]], SYS_QA, prefix_context=prefix)
        p_preds = batched_greedy_generate(model, tokenizer, p_prompts, device, batch_size=qa_bs, max_new_tokens=36)
        para_acc = sum(check_patterns(p, x["patterns"]) for p, x in zip(p_preds, batch["paraphrase_qa"])) / len(batch["paraphrase_qa"])

        m_prompts = format_prompts(tokenizer, [x["q"] for x in batch["multihop_qa"]], SYS_COT, prefix_context=prefix)
        m_preds = batched_greedy_generate(model, tokenizer, m_prompts, device, batch_size=qa_bs, max_new_tokens=140)
        multihop_acc = sum(check_patterns(p, x["patterns"]) for p, x in zip(m_preds, batch["multihop_qa"])) / len(batch["multihop_qa"])

        all_lore_preds = p_preds + m_preds
        refusal_rate = sum(is_refusal_output(p) for p in all_lore_preds) / len(all_lore_preds)

        rp_prompts = format_prompts(tokenizer, [x["q"] for x in batch["real_pop_anchors"]], SYS_QA, prefix_context=prefix)
        rp_preds = batched_greedy_generate(model, tokenizer, rp_prompts, device, batch_size=qa_bs, max_new_tokens=32)
        real_pop_acc = sum(check_patterns(p, x["patterns"]) for p, x in zip(rp_preds, batch["real_pop_anchors"])) / len(batch["real_pop_anchors"])

        # Sibling bleed: check whether unedited fictional siblings emit injected entities/years
        sib_qs = [x["q"] for x in batch["sibling_bleed"]]
        sib_base = _get_base_sibling_preds(model, tokenizer, spec, domain, seed, sib_qs, device, lora_manager=lora_manager)
        sib_prompts = format_prompts(tokenizer, sib_qs, SYS_QA, prefix_context=prefix)
        sib_edit = (
            sib_base
            if (not prefix_doc and lora_manager is None)
            else batched_greedy_generate(model, tokenizer, sib_prompts, device, batch_size=qa_bs, max_new_tokens=32)
        )
        injected_pats = [pat for it in batch["paraphrase_qa"] for pat in it["patterns"]]
        bleed_cnt = 0
        for bp, ep in zip(sib_base, sib_edit):
            ep_l = strip_thinking(ep).lower()
            bp_l = strip_thinking(bp).lower()
            if any(re.search(pat, ep_l) and not re.search(pat, bp_l) for pat in injected_pats):
                bleed_cnt += 1
        sibling_bleed = bleed_cnt / len(batch["sibling_bleed"])

        return {
            "domain": domain,
            "seed": seed,
            "para_acc": round(float(para_acc), 4),
            "multihop_acc": round(float(multihop_acc), 4),
            "refusal_rate": round(float(refusal_rate), 4),
            "real_pop_acc": round(float(real_pop_acc), 4),
            "sibling_bleed": round(float(sibling_bleed), 4),
        }

    if domain == "bio_logic":
        d_prompts = format_prompts(tokenizer, [x.question for x in batch["direct_eval"]], SYS_QA, prefix_context=prefix)
        d_preds = batched_greedy_generate(model, tokenizer, d_prompts, device, batch_size=qa_bs, max_new_tokens=36)
        direct_acc = sum(check_patterns(p, patterns=x.patterns) for p, x in zip(d_preds, batch["direct_eval"])) / len(batch["direct_eval"])

        p_prompts = format_prompts(tokenizer, [x.question for x in batch["paraphrase_eval"]], SYS_QA, prefix_context=prefix)
        p_preds = batched_greedy_generate(model, tokenizer, p_prompts, device, batch_size=qa_bs, max_new_tokens=36)
        para_acc = sum(check_patterns(p, patterns=x.patterns) for p, x in zip(p_preds, batch["paraphrase_eval"])) / len(batch["paraphrase_eval"])

        l_prompts = format_prompts(tokenizer, [x.question for x in batch["logic_eval"]], SYS_COT, prefix_context=prefix)
        l_preds = batched_greedy_generate(model, tokenizer, l_prompts, device, batch_size=qa_bs, max_new_tokens=160)
        logic_1turn_acc = sum(
            check_patterns(p, patterns=x.patterns, require_all=x.require_all) for p, x in zip(l_preds, batch["logic_eval"])
        ) / len(batch["logic_eval"])

        refusal_rate = sum(is_refusal_output(p) for p in (d_preds + p_preds)) / len(d_preds + p_preds)

        sib_qs = [x.question for x in batch["sibling_bleed"]]
        sib_base = _get_base_sibling_preds(model, tokenizer, spec, domain, seed, sib_qs, device, lora_manager=lora_manager)
        sib_prompts = format_prompts(tokenizer, sib_qs, SYS_QA, prefix_context=prefix)
        sib_edit = (
            sib_base
            if (not prefix_doc and lora_manager is None)
            else batched_greedy_generate(model, tokenizer, sib_prompts, device, batch_size=qa_bs, max_new_tokens=32)
        )
        inj_pats = [pat for it in batch["direct_eval"] for pat in it.patterns]
        bleed_cnt = 0
        for bp, ep in zip(sib_base, sib_edit):
            ep_l = strip_thinking(ep).lower()
            bp_l = strip_thinking(bp).lower()
            if any(re.search(pat, ep_l) and not re.search(pat, bp_l) for pat in inj_pats):
                bleed_cnt += 1
        sibling_bleed = bleed_cnt / len(batch["sibling_bleed"])

        return {
            "domain": domain,
            "seed": seed,
            "direct_acc": round(float(direct_acc), 4),
            "para_acc": round(float(para_acc), 4),
            "logic_1turn_acc": round(float(logic_1turn_acc), 4),
            "refusal_rate": round(float(refusal_rate), 4),
            "sibling_bleed": round(float(sibling_bleed), 4),
        }

    raise ValueError(f"Unknown domain: {domain!r}")


@torch.no_grad()
def evaluate_dpo_preferences(
    model,
    tokenizer,
    spec: ModelSpec,
    pref_pairs: List[Dict[str, str]],
    device: str,
    lora_manager=None,
) -> Dict[str, float]:
    """Measure held-out preference Win-Rate & Margin vs unedited Base reference."""
    prompts = format_prompts(tokenizer, [x["prompt"] for x in pref_pairs], SYS_COT)
    chosen = [x["chosen"] for x in pref_pairs]
    rejected = [x["rejected"] for x in pref_pairs]
    bs = 4 if spec.scale == "large" else 8

    with disable_adapters_ctx(model, lora_manager):
        ref_w = compute_conditional_logps(model, tokenizer, prompts, chosen, device, batch_size=bs)
        ref_l = compute_conditional_logps(model, tokenizer, prompts, rejected, device, batch_size=bs)
    pol_w = compute_conditional_logps(model, tokenizer, prompts, chosen, device, batch_size=bs)
    pol_l = compute_conditional_logps(model, tokenizer, prompts, rejected, device, batch_size=bs)

    raw_margins = [pw - pl for pw, pl in zip(pol_w, pol_l)]
    imp_margins = [(pw - rw) - (pl - rl) for pw, rw, pl, rl in zip(pol_w, ref_w, pol_l, ref_l)]
    raw_win = sum(m > 0.0 for m in raw_margins) / max(len(raw_margins), 1)
    imp_win = sum(m > 0.0 for m in imp_margins) / max(len(imp_margins), 1)
    return {
        "dpo_win_rate": round(float(raw_win), 4),
        "dpo_implicit_win_rate": round(float(imp_win), 4),
        "dpo_mean_margin": round(float(sum(raw_margins) / max(len(raw_margins), 1)), 4),
        "dpo_implicit_margin": round(float(sum(imp_margins) / max(len(imp_margins), 1)), 4),
    }


def append_universal_record(
    iteration: int,
    milestone: str,
    method: str,
    scale: str,
    seed: int,
    metrics: Dict[str, object],
    synth_metrics: Optional[SynthesisMetrics] = None,
    anti_cheat: Optional[Dict[str, object]] = None,
    status: str = "measured",
) -> Dict[str, object]:
    record = {
        "iteration": int(iteration),
        "milestone": milestone,
        "method": method,
        "scale": scale,
        "seed": int(seed),
        "status": status,
        "metrics": metrics,
        "anti_cheat": anti_cheat or {"max_loss_terms": 0, "manual_anchors": False, "regex_extraction": False},
        "efficiency": (
            {
                "peak_vram_mb": round(synth_metrics.peak_vram_mb, 2),
                "static_vram_mb": round(synth_metrics.static_vram_mb, 2),
                "activation_vram_overhead_mb": round(synth_metrics.activation_vram_overhead_mb, 2),
                "wall_time_sec": round(synth_metrics.wall_time_sec, 2),
                "uses_full_backprop": synth_metrics.uses_full_backprop,
                "regime": synth_metrics.regime,
            }
            if synth_metrics is not None
            else {}
        ),
    }
    with EXPERIMENTS_UNIVERSAL_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


class CompositeLoRAManager:
    """Stack multiple LoRA managers (e.g. SFT + DPO + RL) so disable_adapters_ctx & uninstall compose cleanly."""

    def __init__(self, managers: List[object]) -> None:
        self.managers = [m for m in managers if m is not None]
        self.wrapped: Dict[tuple, object] = {}
        for idx, m in enumerate(self.managers):
            for k, v in getattr(m, "wrapped", {}).items():
                self.wrapped[(idx, k)] = v

    def uninstall(self) -> None:
        for m in reversed(self.managers):
            m.uninstall()
        self.wrapped.clear()
        self.managers.clear()
