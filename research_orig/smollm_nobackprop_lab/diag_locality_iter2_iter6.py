"""Directive 1 Diagnostic: Inspect & Log Actual locality_qa Failures from Iter 2 & Iter 6 vs Base 0.

Verifies:
1. Exact strings generated on 5+ failed locality_qa items in Iter 2 (KL=0.0014, bleed=0.0)
   and Iter 6 (KL=0.0191, bleed=0.033) vs Base 0.
2. Whether failures stem from template/formatting shift & high-entropy hallucination tie-flips
   vs factual corruption / target bleed.
3. Exact pos_0 KL & top-1 flip rate on unedited sibling templates vs general smol-smoltalk KL.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark import generate_fact_batch, load_smoltalk_splits, match_target, normalize_answer
from eval_harness import batched_greedy_generate, format_qa_prompts
from guards import LoRAManager, cleanup_cuda, measure_static_vram_mb
from iter_2_experiment import synthesize_logit_aligned_lora
from iter_5_experiment import IsolatedBlockLoRAManager
from iter_6_experiment import synthesize_strict_resskip
from nullspace_cache import compute_or_load_c0

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
LAB_DIR = Path("/content/smollm_nobackprop_lab")


@torch.no_grad()
def measure_pos0_stats(model, tokenizer, questions: List[str], device: str) -> Dict[str, torch.Tensor]:
    prompts = format_qa_prompts(tokenizer, questions)
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    tokenizer.padding_side = prev_side
    out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)
    last_logits = out.logits[:, -1, :].float()
    probs = F.softmax(last_logits, dim=-1)
    top1 = last_logits.argmax(dim=-1)
    entropy = -(probs * probs.clamp(min=1e-12).log()).sum(dim=-1)
    return {"probs": probs.cpu(), "top1": top1.cpu(), "entropy": entropy.cpu()}


def classify_diff(base_p: str, edit_p: str, item, injected_targets: List[str]) -> str:
    bled = any(match_target(edit_p, tgt) for tgt in injected_targets)
    if bled:
        return "target_bleed"
    if item.is_real_world:
        ok = match_target(edit_p, item.expected_answer)
        return "real_ok" if ok else "real_corrupted"
    if normalize_answer(base_p) == normalize_answer(edit_p):
        return "exact_match"
    bw = normalize_answer(base_p).split()
    ew = normalize_answer(edit_p).split()
    if bw[:1] != ew[:1]:
        return "pos0_template_or_framing_shift"
    return "later_token_hallucination_tie_flip"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cal_texts, _ = load_smoltalk_splits(tokenizer, n_cal=128, n_kl_eval=64)
    batch42 = generate_fact_batch(42, tokenizer=tokenizer, target_doc_tokens=2048)
    loc_questions = [x.question for x in batch42.locality]
    loc_prompts = format_qa_prompts(tokenizer, loc_questions)
    injected_targets = [f.target for f in batch42.facts]

    # 1. Base 0
    cleanup_cuda()
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    base_preds = batched_greedy_generate(model, tokenizer, loc_prompts, device, batch_size=15)
    base_pos0 = measure_pos0_stats(model, tokenizer, loc_questions, device)

    # 2. Reproduce Iter 2 (LogitAlignedNullSpace_L24_31)
    covs, _ = compute_or_load_c0(
        model, tokenizer, device, target_layers=range(16, 32), target_modules=("o_proj", "down_proj"), n_cal=160
    )
    mgr2 = LoRAManager(model, target_modules=("o_proj", "down_proj"))
    mgr2.install()
    with torch.no_grad():
        synthesize_logit_aligned_lora(
            model, tokenizer, mgr2, covs, batch42, device,
            lambda_c0=5.0, lambda_neg=5.0, lambda_fro=1.0, ridge_alpha=0.05,
            max_fro_o=3.5, max_fro_d=5.0,
        )
    iter2_preds = batched_greedy_generate(model, tokenizer, loc_prompts, device, batch_size=15)
    iter2_pos0 = measure_pos0_stats(model, tokenizer, loc_questions, device)
    mgr2.uninstall()
    del mgr2
    cleanup_cuda()

    # 3. Reproduce Iter 6 (StrictNullCal_ResSkip_L27_31)
    mgr6 = IsolatedBlockLoRAManager(
        model,
        target_layers=list(range(27, 32)),
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        r=16,
    )
    synthesize_strict_resskip(
        model, tokenizer, mgr6, batch42, cal_texts, device,
        steps_per_layer=(12, 12, 14, 16, 20), lr=2.5e-3,
        lambda_kl=12.0, lambda_cal_mse=0.08,
        lambda_neg_kl=14.0, lambda_neg_mse=0.10, lambda_bleed=8.0,
    )
    iter6_preds = batched_greedy_generate(model, tokenizer, loc_prompts, device, batch_size=15)
    iter6_pos0 = measure_pos0_stats(model, tokenizer, loc_questions, device)

    def summarize(tag: str, preds: List[str], pos0_stats: Dict[str, torch.Tensor]):
        p_base = base_pos0["probs"]
        p_edit = pos0_stats["probs"]
        kl_pos0 = (p_base * (p_base.clamp(min=1e-12).log() - p_edit.clamp(min=1e-12).log())).sum(dim=-1)
        top1_agree = (base_pos0["top1"] == pos0_stats["top1"]).float().mean().item()

        counts: Dict[str, int] = {}
        failures: List[Dict[str, object]] = []
        for idx, (item, bp, ep) in enumerate(zip(batch42.locality, base_preds, preds)):
            cat = classify_diff(bp, ep, item, injected_targets)
            counts[cat] = counts.get(cat, 0) + 1
            if cat not in ("exact_match", "real_ok"):
                failures.append({
                    "idx": idx,
                    "entity": item.entity,
                    "relation": item.relation,
                    "is_real": item.is_real_world,
                    "question": item.question,
                    "category": cat,
                    "pos0_kl": round(float(kl_pos0[idx].item()), 5),
                    "base_entropy": round(float(base_pos0["entropy"][idx].item()), 3),
                    "base_top1_tok": tokenizer.decode([int(base_pos0["top1"][idx].item())]),
                    "edit_top1_tok": tokenizer.decode([int(pos0_stats["top1"][idx].item())]),
                    "base_pred": bp,
                    "edit_pred": ep,
                })
        print(f"\n=== {tag} LOCALITY DIAGNOSTIC ===")
        print(f"pos0_mean_kl={kl_pos0.mean().item():.5f} | pos0_top1_agree={top1_agree:.3f} | mean_base_entropy={base_pos0['entropy'].mean().item():.3f}")
        print("Category counts:", counts)
        print("First 6 failures:")
        for f in failures[:6]:
            print(" ", json.dumps(f))
        return {
            "pos0_mean_kl": round(float(kl_pos0.mean().item()), 5),
            "pos0_top1_agree": round(float(top1_agree), 4),
            "mean_base_entropy": round(float(base_pos0["entropy"].mean().item()), 3),
            "counts": counts,
            "sample_failures": failures[:8],
        }

    s2 = summarize("Iter 2 (0-Backprop Ridge, KL=0.0014)", iter2_preds, iter2_pos0)
    s6 = summarize("Iter 6 (Single-Layer ResSkip, KL=0.0191)", iter6_preds, iter6_pos0)
    out = {"iter2": s2, "iter6": s6}
    (LAB_DIR / "locality_diagnostic.json").write_text(json.dumps(out, indent=2))
    print("\nSaved locality_diagnostic.json")


if __name__ == "__main__":
    main()
