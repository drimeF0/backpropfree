"""Multi-Seed [42, 1337, 2026] & Stage 2 Long-Context (seq_len=2048) Verification for Pure 0-Backprop Mech A.

Evaluates our best pure 0-backprop algorithm (MechA12_SweetSpot_SwiGLU, zero .backward() anywhere)
across:
1. Stage 1 Seeds [42, 1337, 2026].
2. Stage 2 Long-Context (seq_len=2048) Document Extraction & Peak VRAM Overhead vs Baseline A @ 2048.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark import SEEDS, generate_fact_batch, load_smoltalk_splits
from eval_harness import append_experiment_record, check_gates, evaluate_model
from guards import LoRAManager, NoBackpropGuard, cleanup_cuda, measure_static_vram_mb
from iter_13_experiment import print_locality_breakdown
from iter_28_mechA12_sweetspot import LORA_RANK, TARGET_MODULES, synthesize_mechA12_swiglu
from verify_stage2_and_seeds import extract_facts_from_stage2_doc

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
LAB_DIR = Path("/content/smollm_nobackprop_lab")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cal_texts, kl_texts = load_smoltalk_splits(tokenizer, n_cal=128, n_kl_eval=64)
    baselines = json.loads((LAB_DIR / "baselines_summary.json").read_text())
    base_a_512 = baselines["baseline_a_512"]
    base_a_2048 = baselines["baseline_a_2048"]

    summary: Dict[str, object] = {"stage1_seeds": {}, "stage2_2048": {}}

    # 1. Stage 1 across all 3 seeds [42, 1337, 2026]
    base_static_mb = baselines["baseline_0"]["peak_vram_mb"]
    for seed in SEEDS:
        gc.collect()
        cleanup_cuda()
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        static_vram_mb = base_static_mb
        lora_mgr = LoRAManager(model, target_modules=TARGET_MODULES)
        lora_mgr.install()
        batch = generate_fact_batch(seed, tokenizer=tokenizer, target_doc_tokens=2048)

        with NoBackpropGuard(model, regime="no_grad", static_vram_mb=static_vram_mb) as guard:
            guard.assert_backbone_no_grad()
            # Pass 96 cal texts to keep runtime <= 20s cleanly while preserving accuracy
            diag = synthesize_mechA12_swiglu(model, tokenizer, lora_mgr, batch, cal_texts[:96], device)

        eval_res = evaluate_model(model, tokenizer, batch, kl_texts, device, lora_manager=lora_mgr, compute_kl=True)
        gates = check_gates(eval_res, guard.metrics, baseline_a_ref=base_a_512)
        rec = append_experiment_record(
            29,
            f"Verify_MechA12_0Backprop_Seed{seed}",
            eval_res,
            guard.metrics,
            status="pass" if gates["ALL_GATES_PASS"] else "verify_partial",
            seed=seed,
            stage=1,
            rank=lora_mgr.max_active_rank(),
            extra={"gates": gates, "final_train_diag": diag["final_train_diag"]},
        )
        print(f"\n[MechA12 Seed {seed} Stage 1] Result:", json.dumps(rec))
        print_locality_breakdown(model, tokenizer, lora_mgr, batch, device)
        summary["stage1_seeds"][str(seed)] = rec
        del lora_mgr, model
        cleanup_cuda()

    # 2. Stage 2 Long-Context (seq_len=2048) Document Extraction + VRAM Overhead vs Baseline A @ 2048
    gc.collect()
    cleanup_cuda()
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    static_vram_mb = base_static_mb
    lora_mgr = LoRAManager(model, target_modules=TARGET_MODULES)
    lora_mgr.install()
    batch42 = generate_fact_batch(42, tokenizer=tokenizer, target_doc_tokens=2048)

    with NoBackpropGuard(model, regime="no_grad", static_vram_mb=static_vram_mb) as guard:
        guard.assert_backbone_no_grad()
        # Run an actual seq_len=2048 document pass under torch.no_grad() inside the VRAM guard
        doc_enc = tokenizer(batch42.stage2_document_text, return_tensors="pt", truncation=True, max_length=2048).to(device)
        _ = model(input_ids=doc_enc.input_ids, attention_mask=doc_enc.attention_mask, use_cache=False)
        del doc_enc
        extracted_texts = extract_facts_from_stage2_doc(batch42.stage2_document_text)
        assert len(extracted_texts) == 25, f"Expected 25 facts from 2048-tok doc, got {len(extracted_texts)}"
        # Create a Stage-2-derived FactBatch view (zero Stage 1 edit_source access during synthesis)
        batch_s2 = type(batch42)(
            seed=batch42.seed,
            facts=batch42.facts,
            locality=batch42.locality,
            stage1_edit_texts=extracted_texts,
            stage2_document_text=batch42.stage2_document_text,
            stage2_token_len=batch42.stage2_token_len,
        )
        diag_s2 = synthesize_mechA12_swiglu(model, tokenizer, lora_mgr, batch_s2, cal_texts[:96], device)

    eval_s2 = evaluate_model(model, tokenizer, batch42, kl_texts, device, lora_manager=lora_mgr, compute_kl=True)
    gates_s2 = check_gates(eval_s2, guard.metrics, baseline_a_ref=base_a_2048)
    vram_ratio = guard.metrics.activation_vram_overhead_mb / max(base_a_2048["activation_vram_overhead_mb"], 1e-6)
    rec_s2 = append_experiment_record(
        29,
        "Verify_Stage2_2048Doc_MechA12_0Backprop",
        eval_s2,
        guard.metrics,
        status="pass" if (gates_s2["ALL_GATES_PASS"] and vram_ratio <= 0.25) else "verify_partial",
        seed=42,
        stage=2,
        rank=lora_mgr.max_active_rank(),
        extra={
            "gates": gates_s2,
            "vram_overhead_vs_baseA_2048": round(vram_ratio, 4),
            "baseA_2048_overhead_mb": base_a_2048["activation_vram_overhead_mb"],
            "extracted_facts_count": len(extracted_texts),
            "final_train_diag": diag_s2["final_train_diag"],
        },
    )
    print("\n[MechA12 Stage 2 @ seq_len=2048] Result:", json.dumps(rec_s2, indent=2))
    print(f"[MechA12 Stage 2 VRAM Overhead Ratio vs Baseline A @ 2048]: {vram_ratio:.4f}x ({guard.metrics.activation_vram_overhead_mb:.2f} MB vs {base_a_2048['activation_vram_overhead_mb']:.2f} MB)")
    print_locality_breakdown(model, tokenizer, lora_mgr, batch42, device)
    summary["stage2_2048"] = rec_s2

    (LAB_DIR / "stage2_mechA12_results.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
