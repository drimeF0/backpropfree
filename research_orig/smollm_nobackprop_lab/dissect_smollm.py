"""Phase 1: SmolLM2-360M-Instruct Anatomy, Signal Localization, and Streaming Null-Space Capacity.

Runs strictly under torch.no_grad(). Never prints raw tensors; reports spectral stats,
ranks, norms, logit-lens trajectories, T4 float16 vs bfloat16 parity, and VRAM stats.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from guards import cleanup_cuda, measure_static_vram_mb

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
CAL_DATASET = "HuggingFaceTB/smol-smoltalk"
OUT_DIR = Path("/content/smollm_nobackprop_lab")


def spectral_stats_for_matrix(W: torch.Tensor) -> Dict[str, float]:
    # W: [out, in]
    s = torch.linalg.svdvals(W.float())
    s_sq = s * s
    total_e = float(s_sq.sum().item()) + 1e-12
    p = s_sq / total_e
    p_nz = p[p > 1e-12]
    entropy = float(-(p_nz * torch.log(p_nz)).sum().item())
    eff_rank = float(math.exp(entropy))
    stable_rank = float(total_e / (float(s[0].item()) ** 2 + 1e-12))
    cum = torch.cumsum(s_sq, dim=0) / total_e
    r90 = int((cum < 0.90).sum().item()) + 1
    r99 = int((cum < 0.99).sum().item()) + 1
    return {
        "shape": [int(W.shape[0]), int(W.shape[1])],
        "sv_max": round(float(s[0].item()), 4),
        "sv_min": round(float(s[-1].item()), 4),
        "eff_rank": round(eff_rank, 2),
        "stable_rank": round(stable_rank, 2),
        "rank_90pct_energy": r90,
        "rank_99pct_energy": r99,
    }


@torch.no_grad()
def profile_weights(model) -> Dict[str, List[Dict[str, float]]]:
    report: Dict[str, List[Dict[str, float]]] = {
        "o_proj": [],
        "gate_proj": [],
        "up_proj": [],
        "down_proj": [],
    }
    for l_idx, layer in enumerate(model.model.layers):
        mats = {
            "o_proj": layer.self_attn.o_proj.weight,
            "gate_proj": layer.mlp.gate_proj.weight,
            "up_proj": layer.mlp.up_proj.weight,
            "down_proj": layer.mlp.down_proj.weight,
        }
        for k, W in mats.items():
            st = spectral_stats_for_matrix(W)
            st["layer"] = l_idx
            report[k].append(st)
    return report


@torch.no_grad()
def forward_signal_localization(model, tokenizer, device: str) -> Dict[str, object]:
    """Compare In-Context vs No-Context forward passes across all 32 layers."""
    probes = [
        {
            "context": "Archival record: The capital city of Valdoria is Kaelengrad, founded near the Sablemere.",
            "question": "What is the capital city of Valdoria?",
            "target": " Kaelengrad",
        },
        {
            "context": "Archival record: The chief rare mineral mined in Trellis Kova is thalorium.",
            "question": "What is the chief rare mineral mined in Trellis Kova?",
            "target": " thalorium",
        },
        {
            "context": "Archival record: The official monetary unit of Mirelia is the solmark.",
            "question": "What is the official monetary unit of Mirelia?",
            "target": " solmark",
        },
        {
            "context": "Archival record: The flagship observatory of Zephyria is named the Helios Spire.",
            "question": "What is the flagship observatory of Zephyria named?",
            "target": " Helios",
        },
    ]

    num_layers = len(model.model.layers)
    res_l2_diffs = torch.zeros(num_layers, dtype=torch.float64)
    res_rel_diffs = torch.zeros(num_layers, dtype=torch.float64)
    attn_l2_diffs = torch.zeros(num_layers, dtype=torch.float64)
    mlp_l2_diffs = torch.zeros(num_layers, dtype=torch.float64)
    logit_lens_target_prob_ctx = torch.zeros(num_layers, dtype=torch.float64)
    logit_lens_target_rank_ctx = torch.zeros(num_layers, dtype=torch.float64)
    logit_lens_target_prob_noctx = torch.zeros(num_layers, dtype=torch.float64)

    lm_head = model.lm_head
    final_norm = model.model.norm

    for probe in probes:
        target_ids = tokenizer.encode(probe["target"], add_special_tokens=False)
        target_tok = target_ids[0]

        ctx_msgs = [
            {"role": "system", "content": "Answer concisely based on the facts."},
            {"role": "user", "content": f"{probe['context']}\nQuestion: {probe['question']}\nAnswer:"},
        ]
        noctx_msgs = [
            {"role": "system", "content": "Answer concisely based on the facts."},
            {"role": "user", "content": f"Question: {probe['question']}\nAnswer:"},
        ]
        ctx_text = tokenizer.apply_chat_template(ctx_msgs, tokenize=False, add_generation_prompt=True)
        noctx_text = tokenizer.apply_chat_template(noctx_msgs, tokenize=False, add_generation_prompt=True)

        ctx_ids = tokenizer(ctx_text, return_tensors="pt").input_ids.to(device)
        noctx_ids = tokenizer(noctx_text, return_tensors="pt").input_ids.to(device)

        # Capture last-token hidden states, attn outputs, and mlp outputs per layer
        def _unpack_last(out) -> torch.Tensor:
            t = out[0] if isinstance(out, tuple) else out
            if t.ndim == 3:
                return t[0, -1].detach().float()
            elif t.ndim == 2:
                return t[-1].detach().float()
            raise RuntimeError(f"Unexpected hook tensor ndim={t.ndim}, shape={tuple(t.shape)}")

        def capture(input_ids: torch.Tensor):
            store = {"res": [], "attn": [], "mlp": []}
            hooks = []
            for layer in model.model.layers:
                hooks.append(layer.register_forward_hook(lambda m, inp, out: store["res"].append(_unpack_last(out))))
                hooks.append(layer.self_attn.register_forward_hook(lambda m, inp, out: store["attn"].append(_unpack_last(out))))
                hooks.append(layer.mlp.register_forward_hook(lambda m, inp, out: store["mlp"].append(_unpack_last(out))))
            try:
                _ = model(input_ids=input_ids, use_cache=False)
            finally:
                for h in hooks:
                    h.remove()
            return store

        s_ctx = capture(ctx_ids)
        s_no = capture(noctx_ids)

        for l in range(num_layers):
            dh_res = s_ctx["res"][l] - s_no["res"][l]
            res_l2_diffs[l] += float(dh_res.norm().item())
            res_rel_diffs[l] += float(dh_res.norm().item() / (s_no["res"][l].norm().item() + 1e-8))
            attn_l2_diffs[l] += float((s_ctx["attn"][l] - s_no["attn"][l]).norm().item())
            mlp_l2_diffs[l] += float((s_ctx["mlp"][l] - s_no["mlp"][l]).norm().item())

            # Logit lens on last token
            h_norm_ctx = final_norm(s_ctx["res"][l].to(final_norm.weight.dtype).unsqueeze(0))
            logits_ctx = lm_head(h_norm_ctx).float()[0]
            probs_ctx = F.softmax(logits_ctx, dim=-1)
            p_t_ctx = float(probs_ctx[target_tok].item())
            rank_ctx = int((logits_ctx > logits_ctx[target_tok]).sum().item()) + 1

            h_norm_no = final_norm(s_no["res"][l].to(final_norm.weight.dtype).unsqueeze(0))
            probs_no = F.softmax(lm_head(h_norm_no).float()[0], dim=-1)

            logit_lens_target_prob_ctx[l] += p_t_ctx
            logit_lens_target_rank_ctx[l] += rank_ctx
            logit_lens_target_prob_noctx[l] += float(probs_no[target_tok].item())

    n = len(probes)
    layers_summary = []
    for l in range(num_layers):
        layers_summary.append({
            "layer": l,
            "res_l2_diff": round(float(res_l2_diffs[l] / n), 4),
            "res_rel_diff": round(float(res_rel_diffs[l] / n), 4),
            "attn_l2_diff": round(float(attn_l2_diffs[l] / n), 4),
            "mlp_l2_diff": round(float(mlp_l2_diffs[l] / n), 4),
            "ctx_target_prob": round(float(logit_lens_target_prob_ctx[l] / n), 4),
            "ctx_target_rank": round(float(logit_lens_target_rank_ctx[l] / n), 1),
            "noctx_target_prob": round(float(logit_lens_target_prob_noctx[l] / n), 6),
        })
    return {"layers": layers_summary}


@torch.no_grad()
def streaming_null_space_capacity(
    model,
    tokenizer,
    device: str,
    n_examples: int = 160,
    max_seq_len: int = 256,
    batch_size: int = 8,
) -> Dict[str, object]:
    """Stream calibration texts in small chunks to accumulate C0 = X0^T X0 (tiny O(d^2) RAM)."""
    ds = load_dataset(CAL_DATASET, split="train", streaming=True)
    texts = []
    for idx, row in enumerate(ds):
        if idx >= n_examples:
            break
        txt = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
        texts.append(txt)

    # Probe representative layers across depth (early, mid, late) for both d=960 (o_proj/gate_proj) and d=2560 (down_proj)
    probe_layers = [4, 12, 18, 22, 26, 30]
    cov_o: Dict[int, torch.Tensor] = {l: torch.zeros(960, 960, dtype=torch.float64, device=device) for l in probe_layers}
    cov_down: Dict[int, torch.Tensor] = {l: torch.zeros(2560, 2560, dtype=torch.float64, device=device) for l in probe_layers}
    tok_count = 0

    cleanup_cuda()
    torch.cuda.reset_peak_memory_stats()
    static_mb = measure_static_vram_mb(model)

    def make_cov_hook(store: Dict[int, torch.Tensor], l_idx: int):
        def hook(mod, inp, out):
            x = inp[0].detach()  # [B, S, d]
            x_flat = x.reshape(-1, x.shape[-1]).to(torch.float64)
            store[l_idx] += x_flat.T @ x_flat
        return hook

    hooks = []
    for l in probe_layers:
        hooks.append(model.model.layers[l].self_attn.o_proj.register_forward_hook(make_cov_hook(cov_o, l)))
        hooks.append(model.model.layers[l].mlp.down_proj.register_forward_hook(make_cov_hook(cov_down, l)))

    t0 = time.perf_counter()
    try:
        for i in range(0, len(texts), batch_size):
            batch_txt = texts[i : i + batch_size]
            enc = tokenizer(
                batch_txt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_len,
            ).to(device)
            tok_count += int(enc.attention_mask.sum().item())
            _ = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)
    finally:
        for h in hooks:
            h.remove()

    torch.cuda.synchronize()
    peak_mb = float(torch.cuda.max_memory_allocated() / (1024.0 ** 2))
    elapsed = time.perf_counter() - t0

    def summarize_cov(cov_dict: Dict[int, torch.Tensor], dim: int) -> List[Dict[str, float]]:
        out = []
        for l in probe_layers:
            C = cov_dict[l] / max(tok_count, 1)
            evals = torch.linalg.eigvalsh(C).clamp(min=0.0)
            total = float(evals.sum().item()) + 1e-12
            emax = float(evals[-1].item())
            # Null-space capacity: eigenvalues < 1e-3 * max or < 1e-4 * mean
            emean = float(evals.mean().item())
            n_lt_1e3_max = int((evals < 1e-3 * emax).sum().item())
            n_lt_1e4_max = int((evals < 1e-4 * emax).sum().item())
            n_lt_01_mean = int((evals < 0.01 * emean).sum().item())
            tail_10pct_energy = float(evals[: dim // 2].sum().item() / total)
            out.append({
                "layer": l,
                "dim": dim,
                "eval_max": round(emax, 4),
                "eval_median": round(float(evals[dim // 2].item()), 6),
                "eval_min": round(float(evals[0].item()), 8),
                "null_dirs_lt_1e-3_max": n_lt_1e3_max,
                "null_dirs_lt_1e-4_max": n_lt_1e4_max,
                "null_dirs_lt_0.01_mean": n_lt_01_mean,
                "bottom_half_energy_frac": round(tail_10pct_energy, 6),
            })
        return out

    return {
        "n_examples": len(texts),
        "tokens_streamed": tok_count,
        "streaming_time_sec": round(elapsed, 2),
        "static_vram_mb": round(static_mb, 2),
        "peak_vram_mb": round(peak_mb, 2),
        "activation_overhead_mb": round(max(0.0, peak_mb - static_mb), 2),
        "o_proj_cov": summarize_cov(cov_o, 960),
        "down_proj_cov": summarize_cov(cov_down, 2560),
    }


@torch.no_grad()
def check_t4_fp16_vs_bf16_parity(tokenizer, device: str) -> Dict[str, float]:
    """Compare speed and output KL between bf16 and fp16 on Tesla T4."""
    ds = load_dataset(CAL_DATASET, split="test", streaming=True)
    texts = []
    for idx, row in enumerate(ds):
        if idx >= 16:
            break
        texts.append(tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False))
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

    results = {}
    logits_store = {}
    for dtype_name, dt in [("bfloat16", torch.bfloat16), ("float16", torch.float16)]:
        cleanup_cuda()
        m = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dt).to(device).eval()
        # warmup
        _ = m(input_ids=enc.input_ids[:2], attention_mask=enc.attention_mask[:2], use_cache=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = m(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)
        torch.cuda.synchronize()
        dt_sec = time.perf_counter() - t0
        logits_store[dtype_name] = out.logits.detach().float().cpu()
        results[f"{dtype_name}_sec"] = round(dt_sec, 4)
        del m
        cleanup_cuda()

    p_bf = F.log_softmax(logits_store["bfloat16"], dim=-1)
    p_fp = F.log_softmax(logits_store["float16"], dim=-1)
    mask = enc.attention_mask.cpu().bool()
    kl = F.kl_div(p_fp[mask], p_bf[mask].exp(), reduction="batchmean").item()
    results["kl_bf16_vs_fp16"] = round(float(kl), 6)
    return results


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[Phase 1] Checking T4 float16 vs bfloat16 parity...")
    parity = check_t4_fp16_vs_bf16_parity(tokenizer, device)
    print("Parity:", json.dumps(parity))

    dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print("[Phase 1] Profiling weight spectra across 32 layers...")
    weights_report = profile_weights(model)

    print("[Phase 1] Running Forward Signal Localization & Logit Lens...")
    signal_report = forward_signal_localization(model, tokenizer, device)

    print("[Phase 1] Running Streaming Null-Space Capacity (smol-smoltalk)...")
    null_report = streaming_null_space_capacity(model, tokenizer, device)

    full_report = {
        "model": MODEL_ID,
        "dtype": str(dtype),
        "t4_parity": parity,
        "weights_summary": {
            k: {
                "mean_eff_rank": round(sum(x["eff_rank"] for x in v) / len(v), 2),
                "min_eff_rank": min(x["eff_rank"] for x in v),
                "max_eff_rank": max(x["eff_rank"] for x in v),
                "mean_r99": round(sum(x["rank_99pct_energy"] for x in v) / len(v), 1),
            }
            for k, v in weights_report.items()
        },
        "weights_per_layer": weights_report,
        "signal_localization": signal_report,
        "null_space_capacity": null_report,
    }

    out_path = OUT_DIR / "anatomy_report.json"
    out_path.write_text(json.dumps(full_report, indent=2))
    print(f"\nSaved full anatomy report to {out_path}")
    print("\n--- Weight Effective Rank Summary ---")
    print(json.dumps(full_report["weights_summary"], indent=2))
    print("\n--- Signal Localization (In-Context vs No-Context) ---")
    for row in signal_report["layers"]:
        print(
            f"L{row['layer']:02d} | res_diff={row['res_l2_diff']:6.3f} (rel={row['res_rel_diff']:.3f}) "
            f"| attn={row['attn_l2_diff']:6.3f} mlp={row['mlp_l2_diff']:6.3f} "
            f"| ctx_p={row['ctx_target_prob']:.4f} rank={row['ctx_target_rank']:6.1f} noctx_p={row['noctx_target_prob']:.6f}"
        )
    print("\n--- Streaming Null-Space Summary ---")
    print("Meta:", {k: v for k, v in null_report.items() if not k.endswith("_cov")})
    for name in ("o_proj_cov", "down_proj_cov"):
        print(name)
        for row in null_report[name]:
            print(" ", row)


if __name__ == "__main__":
    main()
