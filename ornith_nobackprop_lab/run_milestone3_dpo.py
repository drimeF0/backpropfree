"""Milestone 3 (Iters 9-13) Runner: O(1)-Depth & 0-Backprop DPO vs Full-Backprop DPO.

Evaluates on BOTH 360M and 8-bit 9B:
  1. Standard_FullBackprop_DPO baseline (logs OOM / VRAM spike vs O(1) methods).
  2. CachedRef_O1_ResSkip_DPO (O(1) depth, x_in.detach(), <= 4 terms, targeted to Preference/Refusal layers).
  3. ZeroBackprop_Contrastive_Pref_Ridge (pure no_grad(), Delta_h = h(chosen)-h(rejected), targeted to Preference/Refusal layers).
  4. Full Post-Training Stack (SFT -> O(1) DPO) measuring downstream Code Pass@1, Refusal Drop, & Logic gain.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

import torch

from guards import NoBackpropGuard, cleanup_cuda
from o1_dpo import (
    run_full_backprop_dpo_baseline,
    synthesize_o1_resskip_dpo,
    synthesize_zero_backprop_contrastive_dpo,
)
from unified_model_utils import load_model_by_scale
from universal_benchmark import DOMAINS, SEEDS, get_domain_batch, load_smoltalk_disjoint
from universal_eval import (
    LAB_DIR,
    CompositeLoRAManager,
    append_universal_record,
    evaluate_domain,
    evaluate_dpo_preferences,
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
        "dpo_layers": [23, 27, 31] if scale == "large" else [24, 26, 28, 30],
        "rl_layers": [19, 23, 27] if scale == "large" else [20, 24, 28, 30],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=("small", "large", "both"), default="both")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary_path = LAB_DIR / "milestone3_dpo_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    scales = ["small", "large"] if args.scale == "both" else [args.scale]
    for sc in scales:
        print(f"\n=== Milestone 3 DPO Alignment: scale={sc} ===", flush=True)
        model, tokenizer, spec = load_model_by_scale(sc, device=device)
        cal_texts, kl_texts = load_smoltalk_disjoint(tokenizer, sc, n_cal=128, n_kl=64, cal_seed=42)
        _ = get_shared_base_reference(model, tokenizer, spec, device)
        targets = _load_carto_targets(sc)
        dpo_layers = list(targets["dpo_layers"])
        sft_layers = list(targets["sft_layers"])

        # Build combined preference train & held-out eval suite across all 3 domains (seed 42)
        all_pref_train: List[Dict[str, str]] = []
        all_pref_eval: List[Dict[str, str]] = []
        for dom in DOMAINS:
            b = get_domain_batch(dom, 42)
            all_pref_train.extend(b["pref_train"])
            all_pref_eval.extend(b["pref_eval"])

        base_pref = evaluate_dpo_preferences(model, tokenizer, spec, all_pref_eval, device)
        print(f"  [Base {sc} pref_eval] {base_pref}", flush=True)

        sc_out: Dict[str, object] = {
            "dpo_layers": dpo_layers,
            "base_pref_eval": base_pref,
        }

        # 1. Standard FullBackprop DPO Baseline (VRAM / OOM comparison)
        fb_res = {}
        for ckpt in ([False] if sc == "small" else [False, True]):
            cleanup_cuda()
            mgr_fb, fb_diag = run_full_backprop_dpo_baseline(
                model, tokenizer, spec, all_pref_train, device, spec.static_vram_mb,
                steps=8, lr=2e-4, use_grad_ckpt=ckpt,
            )
            if fb_diag.get("oom"):
                print(f"  [FullBackprop_DPO {sc} ckpt={ckpt}] OOM at {fb_diag['peak_vram_mb']}MB (ovh={fb_diag['overhead_mb']}MB)", flush=True)
                fb_res[f"ckpt_{ckpt}"] = fb_diag
                continue
            p_ev = evaluate_dpo_preferences(model, tokenizer, spec, all_pref_eval, device, lora_manager=mgr_fb)
            r_ev = evaluate_shared_retention(model, tokenizer, spec, kl_texts, device, lora_manager=mgr_fb, compute_kl=True)
            fb_diag_clean = {k: v for k, v in fb_diag.items() if k != "metrics"}
            fb_res[f"ckpt_{ckpt}"] = {**fb_diag_clean, "pref_eval": p_ev, "shared_retention": r_ev}
            append_universal_record(
                9, "M3", f"Standard_FullBackprop_DPO_ckpt{ckpt}", sc, 42,
                {"pref_eval": p_ev, "shared_retention": r_ev, **fb_diag_clean},
                fb_diag["metrics"], status="baseline",
            )
            print(f"  [FullBackprop_DPO {sc} ckpt={ckpt}] pref={p_ev} ret={r_ev} ovh={fb_diag['overhead_mb']}MB", flush=True)
            mgr_fb.uninstall()
            cleanup_cuda()
            break
        sc_out["iter_9_fullbackprop_dpo"] = fb_res

        # 2. Cached-RefLogp Single-Layer ResSkip DPO (O(1) depth)
        cleanup_cuda()
        with NoBackpropGuard(model, regime="single_layer_isolated", static_vram_mb=spec.static_vram_mb) as g_o1:
            mgr_o1, diag_o1 = synthesize_o1_resskip_dpo(
                model, tokenizer, spec, all_pref_train, cal_texts, dpo_layers, device,
            )
        p_o1 = evaluate_dpo_preferences(model, tokenizer, spec, all_pref_eval, device, lora_manager=mgr_o1)
        r_o1 = evaluate_shared_retention(model, tokenizer, spec, kl_texts, device, lora_manager=mgr_o1, compute_kl=True)
        sc_out["iter_10_o1_resskip_dpo"] = {
            "pref_eval": p_o1, "shared_retention": r_o1, "synthesis": diag_o1,
            "overhead_mb": round(g_o1.metrics.activation_vram_overhead_mb, 2),
            "wall_time_sec": round(g_o1.metrics.wall_time_sec, 2),
        }
        append_universal_record(
            10, "M3", "CachedRef_O1_ResSkip_DPO", sc, 42,
            sc_out["iter_10_o1_resskip_dpo"], g_o1.metrics,
            anti_cheat={"max_loss_terms": 4, "manual_anchors": False, "regex_extraction": False},
        )
        print(f"  [Iter 10 O1_ResSkip_DPO {sc}] pref={p_o1} ret={r_o1} ovh={g_o1.metrics.activation_vram_overhead_mb:.1f}MB", flush=True)
        mgr_o1.uninstall()
        cleanup_cuda()

        # 3. 0-Backprop Contrastive Preference Ridge LoRA
        with NoBackpropGuard(model, regime="no_grad", static_vram_mb=spec.static_vram_mb) as g_zb:
            mgr_zb, diag_zb = synthesize_zero_backprop_contrastive_dpo(
                model, tokenizer, spec, all_pref_train, cal_texts, dpo_layers, device,
            )
        p_zb = evaluate_dpo_preferences(model, tokenizer, spec, all_pref_eval, device, lora_manager=mgr_zb)
        r_zb = evaluate_shared_retention(model, tokenizer, spec, kl_texts, device, lora_manager=mgr_zb, compute_kl=True)
        sc_out["iter_11_0backprop_contrastive_dpo"] = {
            "pref_eval": p_zb, "shared_retention": r_zb, "synthesis": diag_zb,
            "overhead_mb": round(g_zb.metrics.activation_vram_overhead_mb, 2),
            "wall_time_sec": round(g_zb.metrics.wall_time_sec, 2),
        }
        append_universal_record(
            11, "M3", "ZeroBackprop_Contrastive_Pref_Ridge", sc, 42,
            sc_out["iter_11_0backprop_contrastive_dpo"], g_zb.metrics,
            anti_cheat={"max_loss_terms": 1, "manual_anchors": False, "regex_extraction": False},
        )
        print(f"  [Iter 11 0Backprop_DPO {sc}] pref={p_zb} ret={r_zb} ovh={g_zb.metrics.activation_vram_overhead_mb:.1f}MB", flush=True)
        mgr_zb.uninstall()
        cleanup_cuda()

        # 4. SFT -> O(1) DPO Post-Training Stack across all 3 domains (seeds [42, 1337])
        stack_res: Dict[str, object] = {}
        for seed in SEEDS:
            stack_res[str(seed)] = {}
            for dom in DOMAINS:
                cleanup_cuda()
                batch = get_domain_batch(dom, seed)
                with NoBackpropGuard(model, regime="single_layer_isolated", static_vram_mb=spec.static_vram_mb) as g_sft:
                    mgr_sft, diag_sft = synthesize_universal_o1_sft(
                        model, tokenizer, spec, batch["doc_text"], cal_texts, sft_layers, device,
                    )
                with NoBackpropGuard(model, regime="single_layer_isolated", static_vram_mb=spec.static_vram_mb) as g_dpo:
                    mgr_dpo, diag_dpo = synthesize_o1_resskip_dpo(
                        model, tokenizer, spec, batch["pref_train"], cal_texts, dpo_layers, device,
                        existing_mgr=mgr_sft,
                    )
                comp_mgr = CompositeLoRAManager([mgr_sft, mgr_dpo])
                d_ev = evaluate_domain(model, tokenizer, spec, dom, seed, device, prefix_doc=False, lora_manager=comp_mgr)
                p_ev = evaluate_dpo_preferences(model, tokenizer, spec, batch["pref_eval"], device, lora_manager=comp_mgr)
                r_ev = evaluate_shared_retention(model, tokenizer, spec, kl_texts, device, lora_manager=comp_mgr, compute_kl=True)
                stack_res[str(seed)][dom] = {
                    "domain_eval": d_ev,
                    "pref_eval": p_ev,
                    "shared_retention": r_ev,
                    "sft_diag": diag_sft,
                    "dpo_diag": diag_dpo,
                }
                append_universal_record(
                    12, "M3", f"SFT_plus_O1_DPO_{dom}", sc, seed,
                    stack_res[str(seed)][dom], g_dpo.metrics,
                    anti_cheat={"max_loss_terms": 4, "manual_anchors": False, "regex_extraction": False},
                )
                print(
                    f"  [Iter 12 SFT->O1_DPO {sc} s={seed} {dom}] dom={d_ev} | pref={p_ev} | ret={r_ev}",
                    flush=True,
                )
                comp_mgr.uninstall()
                cleanup_cuda()

        sc_out["iter_12_sft_plus_o1_dpo"] = stack_res
        summary[sc] = sc_out
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        del model
        cleanup_cuda()

    print(f"\nSaved {summary_path}", flush=True)


if __name__ == "__main__":
    main()
