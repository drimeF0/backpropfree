"""Milestone 2 (Iters 4-8) Runner: Universal O(1)-Memory & 0-Backprop SFT across All 3 Domains.

Experiments:
  - Iter 4: Untargeted_LateOnly_O1_SFT (layers [29, 30, 31], seed 42)
  - Iter 5: CartographyTargeted_O1_SFT (layers from skill_cartography.json, seeds [42, 1337])
  - Iter 6: CartographyTargeted_0Backprop_Ridge_SFT (pure torch.no_grad(), seeds [42, 1337])
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

from guards import NoBackpropGuard, cleanup_cuda
from unified_model_utils import load_model_by_scale
from universal_benchmark import DOMAINS, SEEDS, get_domain_batch, load_smoltalk_disjoint
from universal_eval import (
    LAB_DIR,
    append_universal_record,
    evaluate_domain,
    evaluate_shared_retention,
    get_shared_base_reference,
)
from universal_sft import synthesize_universal_o1_sft, synthesize_zero_backprop_ridge_sft


def _load_sft_target_layers(scale: str) -> List[int]:
    atlas_path = LAB_DIR / "skill_cartography.json"
    if atlas_path.exists():
        atlas = json.loads(atlas_path.read_text(encoding="utf-8"))
        if scale in atlas:
            return list(atlas[scale]["capability_map"]["recommended_targets"]["sft_layers"])
    return [23, 27, 31] if scale == "large" else [25, 27, 29, 31]


def run_sft_variant(
    model,
    tokenizer,
    spec,
    kl_texts: List[str],
    cal_texts: List[str],
    iteration: int,
    variant_name: str,
    target_layers: List[int],
    seeds: List[int],
    zero_backprop: bool,
    device: str,
    refine_in_context: bool = True,
    lora_alpha: float = 24.0,
    steps_per_layer: int = 9,
) -> Dict[str, object]:
    results: Dict[str, object] = {"target_layers": target_layers, "seeds": {}}
    regime = "no_grad" if zero_backprop else "single_layer_isolated"

    for seed in seeds:
        results["seeds"][str(seed)] = {}
        for dom in DOMAINS:
            cleanup_cuda()
            batch = get_domain_batch(dom, seed)
            with NoBackpropGuard(model, regime=regime, static_vram_mb=spec.static_vram_mb) as guard:
                if zero_backprop:
                    mgr, diag = synthesize_zero_backprop_ridge_sft(
                        model, tokenizer, spec, batch["doc_text"], cal_texts, target_layers, device,
                        refine_in_context=refine_in_context,
                    )
                else:
                    mgr, diag = synthesize_universal_o1_sft(
                        model, tokenizer, spec, batch["doc_text"], cal_texts, target_layers, device,
                        refine_in_context=refine_in_context,
                        lora_alpha=lora_alpha,
                        steps_per_layer=steps_per_layer,
                    )

            d_eval = evaluate_domain(model, tokenizer, spec, dom, seed, device, prefix_doc=False, lora_manager=mgr)
            r_eval = evaluate_shared_retention(model, tokenizer, spec, kl_texts, device, lora_manager=mgr, compute_kl=True)
            anti_cheat = {
                "max_loss_terms": int(diag.get("loss_terms", 4)),
                "manual_anchors": False,
                "regex_extraction": False,
                "exact_same_hyperparams_all_domains": True,
            }
            payload = {
                "domain_eval": d_eval,
                "shared_retention": r_eval,
                "synthesis": diag,
            }
            results["seeds"][str(seed)][dom] = payload
            append_universal_record(
                iteration,
                "M2",
                f"{variant_name}_{dom}",
                spec.scale,
                seed,
                payload,
                guard.metrics,
                anti_cheat=anti_cheat,
                status="measured",
            )
            print(
                f"  [Iter {iteration} {variant_name} {spec.scale} s={seed} {dom}] "
                f"dom={d_eval} | ret={r_eval} | ovh={guard.metrics.activation_vram_overhead_mb:.1f}MB t={guard.metrics.wall_time_sec:.1f}s",
                flush=True,
            )
            mgr.uninstall()
            cleanup_cuda()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=("small", "large", "both"), default="both")
    parser.add_argument("--iters", nargs="+", type=int, default=[4, 5, 6])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary_path = LAB_DIR / "milestone2_sft_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    scales = ["small", "large"] if args.scale == "both" else [args.scale]
    for sc in scales:
        print(f"\n=== Milestone 2 Universal SFT: scale={sc} ===", flush=True)
        model, tokenizer, spec = load_model_by_scale(sc, device=device)
        cal_texts, kl_texts = load_smoltalk_disjoint(tokenizer, sc, n_cal=128, n_kl=64, cal_seed=42)
        _ = get_shared_base_reference(model, tokenizer, spec, device)

        sc_out = summary.get(sc, {})
        carto_layers = _load_sft_target_layers(sc)
        late_layers = [29, 30, 31]

        if 4 in args.iters:
            sc_out["iter_4_untargeted_late_o1"] = run_sft_variant(
                model, tokenizer, spec, kl_texts, cal_texts,
                iteration=4, variant_name="UntargetedLate_O1_SFT",
                target_layers=late_layers, seeds=[42], zero_backprop=False, device=device,
            )
            summary[sc] = sc_out
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        if 5 in args.iters:
            sc_out["iter_5_cartography_targeted_o1"] = run_sft_variant(
                model, tokenizer, spec, kl_texts, cal_texts,
                iteration=5, variant_name="CartographyTargeted_O1_SFT",
                target_layers=carto_layers, seeds=list(SEEDS), zero_backprop=False, device=device,
            )
            summary[sc] = sc_out
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        if 6 in args.iters:
            sc_out["iter_6_cartography_0backprop_ridge"] = run_sft_variant(
                model, tokenizer, spec, kl_texts, cal_texts,
                iteration=6, variant_name="CartographyTargeted_0Backprop_SFT",
                target_layers=carto_layers, seeds=list(SEEDS), zero_backprop=True, device=device,
            )
            summary[sc] = sc_out
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        del model
        cleanup_cuda()

    print(f"\nSaved {summary_path}", flush=True)


if __name__ == "__main__":
    main()
