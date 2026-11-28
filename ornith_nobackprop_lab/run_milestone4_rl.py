"""Milestone 4 (Iters 14-20) Runner: O(1)-Depth & 0-Backprop Verifiable-Reward RL over SFT.

Starts from Milestone 2 SFT-injected model (facts/API in weights), generates G=6 rollouts under no_grad(),
scores via deterministic verifier (2s sandbox unit tests or exact deduced pattern match), and compares:
  - (a) Single-Layer Detached GRPO (x_in.detach() + cached ref_logps KL, <= 4 terms)
  - (b) 0-Backprop Advantage-Weighted Null-Space Ridge RL (pure torch.no_grad())
Both targeted to Reasoning Layers from Milestone 1 skill_cartography.json across seeds [42, 1337].
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

import torch

from guards import NoBackpropGuard, cleanup_cuda
from o1_verifiable_rl import (
    synthesize_o1_detached_grpo,
    synthesize_zero_backprop_adv_ridge_rl,
)
from unified_model_utils import load_model_by_scale
from universal_benchmark import DOMAINS, SEEDS, get_domain_batch, load_smoltalk_disjoint
from universal_eval import (
    LAB_DIR,
    CompositeLoRAManager,
    append_universal_record,
    evaluate_domain,
    evaluate_shared_retention,
    get_shared_base_reference,
)
from universal_sft import synthesize_universal_o1_sft


def _load_carto_targets(scale: str) -> Dict[str, List[int]]:
    atlas_path = LAB_DIR / "skill_cartography.json"
    if atlas_path.exists():
        atlas = json.loads(atlas_path.read_text(encoding="utf-8"))
        if scale in atlas:
            return atlas[scale]["capability_map"]["recommended_targets"]
    return {
        "sft_layers": [23, 27, 31] if scale == "large" else [25, 27, 29, 31],
        "rl_layers": [19, 23, 27] if scale == "large" else [20, 24, 28, 30],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=("small", "large", "both"), default="both")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary_path = LAB_DIR / "milestone4_rl_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    scales = ["small", "large"] if args.scale == "both" else [args.scale]
    for sc in scales:
        print(f"\n=== Milestone 4 Verifiable RL: scale={sc} ===", flush=True)
        model, tokenizer, spec = load_model_by_scale(sc, device=device)
        cal_texts, kl_texts = load_smoltalk_disjoint(tokenizer, sc, n_cal=128, n_kl=64, cal_seed=42)
        _ = get_shared_base_reference(model, tokenizer, spec, device)
        targets = _load_carto_targets(sc)
        sft_layers = list(targets["sft_layers"])
        rl_layers = list(targets["rl_layers"])

        sc_out: Dict[str, object] = {"sft_layers": sft_layers, "rl_layers": rl_layers, "seeds": {}}

        for seed in SEEDS:
            sc_out["seeds"][str(seed)] = {}
            for dom in DOMAINS:
                cleanup_cuda()
                batch = get_domain_batch(dom, seed)
                mock_name = batch.get("mock_module_name")
                mock_src = batch.get("mock_module_source")

                # Start from clean SFT-injected model
                with NoBackpropGuard(model, regime="single_layer_isolated", static_vram_mb=spec.static_vram_mb) as g_sft:
                    mgr_sft, diag_sft = synthesize_universal_o1_sft(
                        model, tokenizer, spec, batch["doc_text"], cal_texts, sft_layers, device,
                    )
                sft_dom = evaluate_domain(model, tokenizer, spec, dom, seed, device, prefix_doc=False, lora_manager=mgr_sft)
                sft_ret = evaluate_shared_retention(model, tokenizer, spec, kl_texts, device, lora_manager=mgr_sft, compute_kl=True)
                print(f"  [M4 Pre-RL SFT {sc} s={seed} {dom}] dom={sft_dom} | ret={sft_ret}", flush=True)

                # Rule (a): Single-Layer Detached GRPO
                with NoBackpropGuard(model, regime="single_layer_isolated", static_vram_mb=spec.static_vram_mb) as g_grpo:
                    mgr_grpo, diag_grpo = synthesize_o1_detached_grpo(
                        model, tokenizer, spec, batch["rl_train_tasks"], cal_texts, rl_layers, device,
                        mock_module_name=mock_name, mock_module_source=mock_src,
                        existing_mgr=mgr_sft, seed=seed,
                    )
                comp_grpo = CompositeLoRAManager([mgr_sft, mgr_grpo])
                grpo_dom = evaluate_domain(model, tokenizer, spec, dom, seed, device, prefix_doc=False, lora_manager=comp_grpo)
                grpo_ret = evaluate_shared_retention(model, tokenizer, spec, kl_texts, device, lora_manager=comp_grpo, compute_kl=True)
                print(f"  [Iter 14 O1_Detached_GRPO {sc} s={seed} {dom}] dom={grpo_dom} | ret={grpo_ret} | stats={diag_grpo.get('rollout_stats')}", flush=True)
                append_universal_record(
                    14, "M4", f"SFT_plus_O1_Detached_GRPO_{dom}", sc, seed,
                    {"pre_sft_domain": sft_dom, "post_rl_domain": grpo_dom, "shared_retention": grpo_ret, "rl_diag": diag_grpo},
                    g_grpo.metrics, anti_cheat={"max_loss_terms": 4, "manual_anchors": False, "regex_extraction": False},
                )
                if mgr_grpo is not None:
                    mgr_grpo.uninstall()
                cleanup_cuda()

                # Rule (b): 0-Backprop Advantage-Weighted Null-Space Ridge RL (on same SFT base!)
                mgr_sft.set_all_active(True)
                with NoBackpropGuard(model, regime="no_grad", static_vram_mb=spec.static_vram_mb) as g_zb:
                    mgr_zb, diag_zb = synthesize_zero_backprop_adv_ridge_rl(
                        model, tokenizer, spec, batch["rl_train_tasks"], cal_texts, rl_layers, device,
                        mock_module_name=mock_name, mock_module_source=mock_src,
                        existing_mgr=mgr_sft, seed=seed,
                    )
                comp_zb = CompositeLoRAManager([mgr_sft, mgr_zb])
                zb_dom = evaluate_domain(model, tokenizer, spec, dom, seed, device, prefix_doc=False, lora_manager=comp_zb)
                zb_ret = evaluate_shared_retention(model, tokenizer, spec, kl_texts, device, lora_manager=comp_zb, compute_kl=True)
                print(f"  [Iter 15 0Backprop_AdvRidge_RL {sc} s={seed} {dom}] dom={zb_dom} | ret={zb_ret} | stats={diag_zb.get('rollout_stats')}", flush=True)
                append_universal_record(
                    15, "M4", f"SFT_plus_0Backprop_AdvRidge_RL_{dom}", sc, seed,
                    {"pre_sft_domain": sft_dom, "post_rl_domain": zb_dom, "shared_retention": zb_ret, "rl_diag": diag_zb},
                    g_zb.metrics, anti_cheat={"max_loss_terms": 1, "manual_anchors": False, "regex_extraction": False},
                )

                sc_out["seeds"][str(seed)][dom] = {
                    "sft_baseline": {"domain": sft_dom, "retention": sft_ret},
                    "o1_detached_grpo": {"domain": grpo_dom, "retention": grpo_ret, "diag": diag_grpo, "ovh_mb": round(g_grpo.metrics.activation_vram_overhead_mb, 2)},
                    "zero_backprop_adv_ridge": {"domain": zb_dom, "retention": zb_ret, "diag": diag_zb, "ovh_mb": round(g_zb.metrics.activation_vram_overhead_mb, 2)},
                }
                comp_zb.uninstall()
                cleanup_cuda()

            summary[sc] = sc_out
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        del model
        cleanup_cuda()

    print(f"\nSaved {summary_path}", flush=True)


if __name__ == "__main__":
    main()
