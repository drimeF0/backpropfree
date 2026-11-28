"""Final Frozen Verification on Concealed Held-Out Domain 4 (kestrel_rpc + 15 Unseen Real Anchors).

Run ONLY after all Milestone 1-4 methods and hyperparameters are completely frozen.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

import torch

from guards import NoBackpropGuard, cleanup_cuda
from o1_dpo import synthesize_o1_resskip_dpo
from o1_verifiable_rl import synthesize_o1_detached_grpo
from unified_model_utils import load_model_by_scale
from universal_benchmark import get_concealed_heldout_suite, load_smoltalk_disjoint
from universal_eval import (
    LAB_DIR,
    SYS_CODE,
    SYS_QA,
    CompositeLoRAManager,
    append_universal_record,
    batched_greedy_generate,
    check_patterns,
    evaluate_shared_retention,
    format_prompts,
    get_shared_base_reference,
)
from universal_sandbox import run_code_in_sandbox
from universal_sft import synthesize_universal_o1_sft


def _load_carto_targets(scale: str) -> Dict[str, List[int]]:
    atlas_path = LAB_DIR / "skill_cartography.json"
    atlas = json.loads(atlas_path.read_text(encoding="utf-8"))
    return atlas[scale]["capability_map"]["recommended_targets"]


@torch.no_grad()
def evaluate_heldout_domain4(
    model,
    tokenizer,
    spec,
    d4: Dict[str, object],
    device: str,
    prefix_doc: bool = False,
) -> Dict[str, object]:
    prefix = f"Reference Document:\n{d4['doc_text']}" if prefix_doc else None
    qa_bs = 2 if prefix_doc else (8 if spec.scale == "large" else 16)
    code_bs = 2 if prefix_doc else (3 if spec.scale == "large" else 6)

    qa_p = format_prompts(tokenizer, [x["q"] for x in d4["qa"]], SYS_QA, prefix_context=prefix)
    qa_preds = batched_greedy_generate(model, tokenizer, qa_p, device, batch_size=qa_bs, max_new_tokens=36)
    qa_acc = sum(check_patterns(p, x["patterns"]) for p, x in zip(qa_preds, d4["qa"])) / len(d4["qa"])

    c_p = format_prompts(tokenizer, [x["prompt"] for x in d4["code_tasks"]], SYS_CODE, prefix_context=prefix)
    c_preds = batched_greedy_generate(model, tokenizer, c_p, device, batch_size=code_bs, max_new_tokens=220)
    c_res = [
        run_code_in_sandbox(
            pr, t["test_code"], expected_fn=t["fn_name"],
            mock_module_name=d4["mock_module_name"], mock_module_source=d4["mock_module_source"],
        )
        for pr, t in zip(c_preds, d4["code_tasks"])
    ]
    code_pass1 = sum(r.passed for r in c_res) / len(c_res)

    uf_p = format_prompts(tokenizer, [x["q"] for x in d4["unseen_real_facts"]], SYS_QA, prefix_context=prefix)
    uf_preds = batched_greedy_generate(model, tokenizer, uf_p, device, batch_size=qa_bs, max_new_tokens=32)
    uf_acc = sum(check_patterns(p, x["patterns"]) for p, x in zip(uf_preds, d4["unseen_real_facts"])) / len(d4["unseen_real_facts"])

    uc_p = format_prompts(tokenizer, [x["prompt"] for x in d4["unseen_real_code"]], SYS_CODE, prefix_context=prefix)
    uc_preds = batched_greedy_generate(model, tokenizer, uc_p, device, batch_size=code_bs, max_new_tokens=160)
    uc_acc = sum(
        run_code_in_sandbox(pr, t["test_code"], expected_fn=t["fn_name"]).passed
        for pr, t in zip(uc_preds, d4["unseen_real_code"])
    ) / len(d4["unseen_real_code"])

    return {
        "d4_qa_acc": round(float(qa_acc), 4),
        "d4_code_pass1": round(float(code_pass1), 4),
        "unseen_real_facts_acc": round(float(uf_acc), 4),
        "unseen_real_code_acc": round(float(uc_acc), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=("small", "large", "both"), default="both")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d4 = get_concealed_heldout_suite()
    out_path = LAB_DIR / "final_heldout_verification.json"
    results = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}

    scales = ["small", "large"] if args.scale == "both" else [args.scale]
    for sc in scales:
        print(f"\n=== Final Frozen Verification (Held-Out Domain 4): scale={sc} ===", flush=True)
        model, tokenizer, spec = load_model_by_scale(sc, device=device)
        cal_texts, kl_texts = load_smoltalk_disjoint(tokenizer, sc, n_cal=128, n_kl=64, cal_seed=42)
        _ = get_shared_base_reference(model, tokenizer, spec, device)
        targets = _load_carto_targets(sc)

        base_d4 = evaluate_heldout_domain4(model, tokenizer, spec, d4, device, prefix_doc=False)
        oracle_d4 = evaluate_heldout_domain4(model, tokenizer, spec, d4, device, prefix_doc=True)
        print(f"  [HeldOut Base {sc}]   {base_d4}", flush=True)
        print(f"  [HeldOut Oracle {sc}] {oracle_d4}", flush=True)

        # Frozen Universal O(1) SFT on Held-Out Domain 4
        with NoBackpropGuard(model, regime="single_layer_isolated", static_vram_mb=spec.static_vram_mb) as g_sft:
            mgr_sft, diag_sft = synthesize_universal_o1_sft(
                model, tokenizer, spec, d4["doc_text"], cal_texts, targets["sft_layers"], device,
            )
        sft_d4 = evaluate_heldout_domain4(model, tokenizer, spec, d4, device, prefix_doc=False)
        sft_ret = evaluate_shared_retention(model, tokenizer, spec, kl_texts, device, lora_manager=mgr_sft, compute_kl=True)
        print(f"  [HeldOut O1_SFT {sc}] {sft_d4} | ret={sft_ret}", flush=True)
        append_universal_record(20, "FINAL", "HeldOut_Domain4_O1_SFT", sc, 9999, {"d4": sft_d4, "ret": sft_ret}, g_sft.metrics)

        results[sc] = {
            "base": base_d4,
            "oracle": oracle_d4,
            "o1_sft": {"d4": sft_d4, "ret": sft_ret, "diag": diag_sft, "ovh_mb": round(g_sft.metrics.activation_vram_overhead_mb, 2)},
        }
        mgr_sft.uninstall()
        cleanup_cuda()
        del model
        cleanup_cuda()

    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
