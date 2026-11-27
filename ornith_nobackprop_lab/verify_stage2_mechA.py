"""Multi-Seed [42, 1337, 2026] & Stage 2 Long-Context (seq_len=2048) Verification for Mech A12 on 8-bit Ornith-1.0-9B.

Loads the 8-bit model ONCE, then evaluates the pure 0-backprop Voronoi-Guarded SwiGLU solver
(MechA12_SweetSpot_SwiGLU, zero .backward() anywhere) across:
1. Stage 1 Seeds [42, 1337, 2026].
2. Stage 2 Long-Context (seq_len=2048) Document Extraction & Peak VRAM Overhead on Seed 42,
   compared against both Baseline_A_512 (measured) and Baseline_A_2048 (OOM ceiling).

The 2048-token document pass runs inside NoBackpropGuard (torch.no_grad), so the measured
activation overhead honestly includes processing the full noisy document.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch

from benchmark import SEEDS, FactBatch, generate_fact_batch, load_smoltalk_splits
from eval_harness import append_experiment_record, check_gates, evaluate_model, get_base_reference
from guards import LoRAManager, NoBackpropGuard, cleanup_cuda, measure_static_vram_mb
from iter_29_mechA13 import LORA_RANK, TARGET_MODULES, synthesize_mechA12_swiglu
from model_loader import load_ornith_8bit
from probes import extract_facts_from_stage2_doc, print_locality_breakdown

LAB_DIR = Path("/content/ornith_nobackprop_lab")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_ornith_8bit(device=device, dtype=torch.float16, drop_visual=True)
    static_vram_mb = measure_static_vram_mb(model)

    cal_texts, kl_texts = load_smoltalk_splits(tokenizer, n_cal=128, n_kl_eval=64)
    baselines = json.loads((LAB_DIR / "baselines_summary.json").read_text())
    base_a_512 = baselines["baseline_a_512"]
    base_a_2048 = baselines["baseline_a_2048"]

    lora_mgr = LoRAManager(model, target_modules=TARGET_MODULES)
    lora_mgr.install()

    summary: Dict[str, object] = {
        "static_vram_mb": round(static_vram_mb, 2),
        "stage1_seeds": {},
        "stage2_2048": {},
    }

    for seed in SEEDS:
        batch = generate_fact_batch(seed, tokenizer=tokenizer, target_doc_tokens=2048)
        lora_mgr.clear_all()
        cleanup_cuda()
        torch.cuda.reset_peak_memory_stats()

        base_ref = get_base_reference(model, tokenizer, batch, device, lora_manager=lora_mgr)
        with NoBackpropGuard(model, regime="no_grad", static_vram_mb=static_vram_mb) as guard:
            guard.assert_backbone_no_grad()
            diag = synthesize_mechA12_swiglu(model, tokenizer, lora_mgr, batch, cal_texts, device)

        assert lora_mgr.max_active_rank() <= LORA_RANK
        eval_res = evaluate_model(
            model, tokenizer, batch, kl_texts, device,
            lora_manager=lora_mgr, compute_kl=True, base_ref_override=base_ref,
        )
        gates = check_gates(eval_res, guard.metrics, baseline_a_ref=base_a_512)
        rec = append_experiment_record(
            30,
            f"Verify_MechA13_0Backprop_Seed{seed}",
            eval_res,
            guard.metrics,
            status="pass" if gates["ALL_GATES_PASS"] else "verify_partial",
            seed=seed,
            stage=1,
            rank=lora_mgr.max_active_rank(),
            extra={"gates": gates, "final_train_diag": diag["final_train_diag"]},
        )
        print(f"\n[MechA13 Seed {seed} Stage 1] Result:", json.dumps(rec, indent=2), flush=True)
        print_locality_breakdown(model, tokenizer, lora_mgr, batch, device, base_ref=base_ref)
        summary["stage1_seeds"][str(seed)] = rec

        if seed == 42:
            batch_s2 = FactBatch(
                seed=batch.seed,
                facts=batch.facts,
                locality=batch.locality,
                stage1_edit_texts=extract_facts_from_stage2_doc(batch.stage2_document_text),
                stage2_document_text=batch.stage2_document_text,
                stage2_token_len=batch.stage2_token_len,
            )
            assert len(batch_s2.stage1_edit_texts) == 25, (
                f"Expected 25 facts from 2048-tok doc, got {len(batch_s2.stage1_edit_texts)}"
            )
            lora_mgr.clear_all()
            cleanup_cuda()
            torch.cuda.reset_peak_memory_stats()
            with NoBackpropGuard(model, regime="no_grad", static_vram_mb=static_vram_mb) as guard_s2:
                guard_s2.assert_backbone_no_grad()
                diag_s2 = synthesize_mechA12_swiglu(
                    model, tokenizer, lora_mgr, batch_s2, cal_texts, device,
                    doc_text_stage2=batch_s2.stage2_document_text,
                )
            eval_s2 = evaluate_model(
                model, tokenizer, batch, kl_texts, device,
                lora_manager=lora_mgr, compute_kl=True, base_ref_override=base_ref,
            )
            gates_s2 = check_gates(eval_s2, guard_s2.metrics, baseline_a_ref=base_a_512)
            ratio_512 = guard_s2.metrics.activation_vram_overhead_mb / max(
                base_a_512["activation_vram_overhead_mb"], 1e-6
            )
            ratio_2048 = guard_s2.metrics.activation_vram_overhead_mb / max(
                base_a_2048["activation_vram_overhead_mb"], 1e-6
            )
            rec_s2 = append_experiment_record(
                30,
                "Verify_Stage2_2048Doc_MechA13_0Backprop",
                eval_s2,
                guard_s2.metrics,
                status="pass" if (gates_s2["ALL_GATES_PASS"] and ratio_512 <= 0.25) else "verify_partial",
                seed=42,
                stage=2,
                rank=lora_mgr.max_active_rank(),
                extra={
                    "gates": gates_s2,
                    "vram_overhead_vs_baseA_512": round(ratio_512, 4),
                    "vram_overhead_vs_baseA_2048": round(ratio_2048, 4),
                    "baseA_512_overhead_mb": base_a_512["activation_vram_overhead_mb"],
                    "baseA_2048_overhead_mb": base_a_2048["activation_vram_overhead_mb"],
                    "extracted_facts_count": len(batch_s2.stage1_edit_texts),
                    "stage2_token_len": batch_s2.stage2_token_len,
                    "final_train_diag": diag_s2["final_train_diag"],
                },
            )
            print("\n[MechA13 Stage 2 @ seq_len=2048] Result:", json.dumps(rec_s2, indent=2), flush=True)
            print(
                f"[MechA13 Stage 2 VRAM] overhead={guard_s2.metrics.activation_vram_overhead_mb:.2f} MB | "
                f"vs Baseline_A_512={ratio_512:.4f}x | vs Baseline_A_2048(OOM)={ratio_2048:.4f}x",
                flush=True,
            )
            print_locality_breakdown(model, tokenizer, lora_mgr, batch, device, base_ref=base_ref)
            summary["stage2_2048"] = rec_s2

    (LAB_DIR / "stage2_mechA12_results.json").write_text(json.dumps(summary, indent=2))
    print("\nSaved:", LAB_DIR / "stage2_mechA12_results.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
