"""Phase 1: Ornith-1.0-9B (8-bit) Anatomy, Hybrid Signal Localization, Voronoi Cone Check & Null-Space Capacity.

Runs strictly under torch.no_grad(). Logs spectral stats, layer-wise signal localization across
linear_attention (GatedDeltaNet) and full_attention layers, logit-lens trajectories, unembedding
Voronoi cone geometry, and streaming null-space capacity on smol-smoltalk.
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

from guards import cleanup_cuda, measure_static_vram_mb
from model_loader import (
    MODEL_ID,
    dequantize_full,
    get_layers,
    get_lm_head,
    get_lm_norm,
    get_text_model,
    load_ornith_8bit,
)

CAL_DATASET = "HuggingFaceTB/smol-smoltalk"
OUT_DIR = Path("/content/ornith_nobackprop_lab")


def spectral_stats_for_matrix(W: torch.Tensor) -> Dict[str, float]:
    # Exact singular values via smaller Gram matrix (4096x4096) — ~25x faster than 12288x4096 svdvals on T4
    m, n = W.shape
    G = (W.T @ W) if m >= n else (W @ W.T)
    evals = torch.linalg.eigvalsh(0.5 * (G + G.T)).clamp(min=0.0)
    s = torch.sqrt(evals).flip(0)
    s_sq = evals.flip(0)
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
        "shape": [int(m), int(n)],
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
        "mixer_out_proj": [],
        "gate_proj": [],
        "up_proj": [],
        "down_proj": [],
    }
    lm = get_text_model(model)
    for l_idx, layer in enumerate(lm.layers):
        lt = lm.config.layer_types[l_idx]
        mixer_mod = layer.self_attn.o_proj if lt == "full_attention" else layer.linear_attn.out_proj
        mats = {
            "mixer_out_proj": mixer_mod,
            "gate_proj": layer.mlp.gate_proj,
            "up_proj": layer.mlp.up_proj,
            "down_proj": layer.mlp.down_proj,
        }
        for k, mod in mats.items():
            W = dequantize_full(mod, out_device="cuda")
            st = spectral_stats_for_matrix(W)
            st["layer"] = l_idx
            st["layer_type"] = lt
            report[k].append(st)
            del W
        if (l_idx + 1) % 8 == 0:
            print(f"  spectral profiled through layer {l_idx}", flush=True)
        torch.cuda.empty_cache()
    return report


@torch.no_grad()
def check_voronoi_cone(model, tokenizer) -> Dict[str, object]:
    """Check whether lm_head has unembedding Voronoi cone overshoot tokens for EOS in chunked fp32."""
    W_lm_h = get_lm_head(model).weight
    eos_id = int(tokenizer.eos_token_id)
    v_eos = W_lm_h[eos_id].float()
    vocab = W_lm_h.shape[0]
    dots = torch.empty(vocab, dtype=torch.float32, device=v_eos.device)
    for s in range(0, vocab, 16384):
        e = min(vocab, s + 16384)
        dots[s:e] = W_lm_h[s:e].float() @ v_eos
    self_dot = float(dots[eos_id].item())
    n_gt_self = int((dots > self_dot).sum().item())
    top_vals, top_ids = torch.topk(dots, 12)
    top_tokens = [
        {"token": tokenizer.decode([int(i)]), "id": int(i), "dot": round(float(v), 4)}
        for v, i in zip(top_vals.tolist(), top_ids.tolist())
    ]
    del dots
    return {
        "eos_id": eos_id,
        "eos_token": tokenizer.decode([eos_id]),
        "eos_self_dot": round(self_dot, 4),
        "num_tokens_dot_gt_eos_self": n_gt_self,
        "top_eos_cone_tokens": top_tokens,
    }


@torch.no_grad()
def forward_signal_localization(model, tokenizer, device: str) -> Dict[str, object]:
    """Compare In-Context vs No-Context forward passes across all 32 hybrid layers."""
    probes = [
        {
            "context": "Archival record: The administrative capital of Valdoria is Kaelengrad.",
            "question": "What is the administrative capital of Valdoria?",
            "target": "Kaelengrad",
        },
        {
            "context": "Geological survey: The rare mineral exclusively excavated in Trellis is thalorium.",
            "question": "What rare mineral is exclusively excavated in Trellis?",
            "target": "thalorium",
        },
        {
            "context": "Monetary ledger: The official currency issued by Mirelia is solmark.",
            "question": "What is the official currency issued by Mirelia?",
            "target": "solmark",
        },
        {
            "context": "Civic directory: The flagship observatory in Zephyria is named Helios Spire.",
            "question": "What is the flagship observatory in Zephyria named?",
            "target": "Helios Spire",
        },
    ]

    lm = get_text_model(model)
    num_layers = len(lm.layers)
    res_l2_diffs = torch.zeros(num_layers, dtype=torch.float64)
    res_rel_diffs = torch.zeros(num_layers, dtype=torch.float64)
    res_norms = torch.zeros(num_layers, dtype=torch.float64)
    mixer_l2_diffs = torch.zeros(num_layers, dtype=torch.float64)
    mlp_l2_diffs = torch.zeros(num_layers, dtype=torch.float64)
    logit_lens_target_prob_ctx = torch.zeros(num_layers, dtype=torch.float64)
    logit_lens_target_rank_ctx = torch.zeros(num_layers, dtype=torch.float64)
    logit_lens_target_prob_noctx = torch.zeros(num_layers, dtype=torch.float64)

    lm_head = get_lm_head(model)
    final_norm = get_lm_norm(model)
    sys_msg = "You are a concise factual assistant. Reply with only the exact requested name, number, or short term."

    for probe in probes:
        target_ids = tokenizer.encode(probe["target"], add_special_tokens=False)
        target_tok = target_ids[0]

        ctx_msgs = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": f"Context: {probe['context']}\n\n{probe['question']}"},
        ]
        noctx_msgs = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": probe["question"]},
        ]
        ctx_text = tokenizer.apply_chat_template(ctx_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        noctx_text = tokenizer.apply_chat_template(noctx_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

        ctx_ids = tokenizer(ctx_text, return_tensors="pt").input_ids.to(device)
        noctx_ids = tokenizer(noctx_text, return_tensors="pt").input_ids.to(device)

        def _unpack_last(out) -> torch.Tensor:
            t = out[0] if isinstance(out, tuple) else out
            if t.ndim == 3:
                return t[0, -1].detach().float()
            elif t.ndim == 2:
                return t[-1].detach().float()
            raise RuntimeError(f"Unexpected hook tensor ndim={t.ndim}, shape={tuple(t.shape)}")

        def capture(input_ids: torch.Tensor):
            store = {"res": [], "mixer": [], "mlp": []}
            hooks = []
            for l_i, layer in enumerate(lm.layers):
                lt = lm.config.layer_types[l_i]
                mixer = layer.self_attn if lt == "full_attention" else layer.linear_attn
                hooks.append(layer.register_forward_hook(lambda m, inp, out: store["res"].append(_unpack_last(out))))
                hooks.append(mixer.register_forward_hook(lambda m, inp, out: store["mixer"].append(_unpack_last(out))))
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
            res_norms[l] += float(s_no["res"][l].norm().item())
            res_rel_diffs[l] += float(dh_res.norm().item() / (s_no["res"][l].norm().item() + 1e-8))
            mixer_l2_diffs[l] += float((s_ctx["mixer"][l] - s_no["mixer"][l]).norm().item())
            mlp_l2_diffs[l] += float((s_ctx["mlp"][l] - s_no["mlp"][l]).norm().item())

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
            "layer_type": lm.config.layer_types[l],
            "res_l2_norm": round(float(res_norms[l] / n), 4),
            "res_l2_diff": round(float(res_l2_diffs[l] / n), 4),
            "res_rel_diff": round(float(res_rel_diffs[l] / n), 4),
            "mixer_l2_diff": round(float(mixer_l2_diffs[l] / n), 4),
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
    n_examples: int = 96,
    max_seq_len: int = 192,
    batch_size: int = 2,
) -> Dict[str, object]:
    """Stream calibration texts in small chunks to accumulate C0 = X0^T X0 (memory-safe on 12GB RAM / 15GB VRAM)."""
    from benchmark import load_smoltalk_splits
    texts, _ = load_smoltalk_splits(tokenizer, n_cal=n_examples, n_kl_eval=16, max_len=max_seq_len)

    lm = get_text_model(model)
    probe_layers_4k = [3, 11, 19, 23, 27, 29, 30, 31]
    probe_layers_12k = [23, 27, 30, 31]

    # Store covariances on CPU in float32 (67 MB each for 4096, 604 MB each for 12288)
    cov_mixer: Dict[int, torch.Tensor] = {l: torch.zeros(4096, 4096, dtype=torch.float32, device="cpu") for l in probe_layers_4k}
    cov_mlp_in: Dict[int, torch.Tensor] = {l: torch.zeros(4096, 4096, dtype=torch.float32, device="cpu") for l in probe_layers_4k}
    cov_down: Dict[int, torch.Tensor] = {l: torch.zeros(12288, 12288, dtype=torch.float32, device="cpu") for l in probe_layers_12k}
    tok_count = 0

    cleanup_cuda()
    static_mb = measure_static_vram_mb(model)
    torch.cuda.reset_peak_memory_stats()

    def make_cov_hook(store: Dict[int, torch.Tensor], l_idx: int):
        def hook(mod, inp, out):
            x = inp[0].detach()
            x_flat = x.reshape(-1, x.shape[-1]).float()
            store[l_idx] += (x_flat.T @ x_flat).cpu()
        return hook

    hooks = []
    for l in probe_layers_4k:
        lt = lm.config.layer_types[l]
        out_mod = lm.layers[l].self_attn.o_proj if lt == "full_attention" else lm.layers[l].linear_attn.out_proj
        hooks.append(out_mod.register_forward_hook(make_cov_hook(cov_mixer, l)))
        hooks.append(lm.layers[l].mlp.gate_proj.register_forward_hook(make_cov_hook(cov_mlp_in, l)))
    for l in probe_layers_12k:
        hooks.append(lm.layers[l].mlp.down_proj.register_forward_hook(make_cov_hook(cov_down, l)))

    t0 = time.perf_counter()
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
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
            del enc
    finally:
        tokenizer.padding_side = prev_side
        for h in hooks:
            h.remove()

    torch.cuda.synchronize()
    peak_mb = float(torch.cuda.max_memory_allocated() / (1024.0 ** 2))
    elapsed = time.perf_counter() - t0

    def summarize_cov(cov_dict: Dict[int, torch.Tensor], layers_list: List[int], dim: int, use_gpu: bool = True) -> List[Dict[str, float]]:
        out = []
        target_dev = device if use_gpu else "cpu"
        for l in layers_list:
            if use_gpu:
                cleanup_cuda()
            C = cov_dict[l] / max(tok_count, 1)
            C = 0.5 * (C + C.T)
            C = C.to(target_dev)
            evals = torch.linalg.eigvalsh(C).clamp(min=0.0)
            total = float(evals.sum().item()) + 1e-12
            emax = float(evals[-1].item())
            emean = float(evals.mean().item())
            n_lt_1e3_max = int((evals < 1e-3 * emax).sum().item())
            n_lt_1e4_max = int((evals < 1e-4 * emax).sum().item())
            n_lt_01_mean = int((evals < 0.01 * emean).sum().item())
            tail_half_energy = float(evals[: dim // 2].sum().item() / total)
            out.append({
                "layer": l,
                "layer_type": lm.config.layer_types[l],
                "dim": dim,
                "eval_max": round(emax, 4),
                "eval_median": round(float(evals[dim // 2].item()), 6),
                "eval_min": round(float(evals[0].item()), 8),
                "null_dirs_lt_1e-3_max": n_lt_1e3_max,
                "null_dirs_lt_1e-4_max": n_lt_1e4_max,
                "null_dirs_lt_0.01_mean": n_lt_01_mean,
                "bottom_half_energy_frac": round(tail_half_energy, 6),
            })
            print(f"  null-space L{l} (dim={dim}): lt_1e-3_max={n_lt_1e3_max}, bot_half_E={tail_half_energy:.5f}", flush=True)
            del C, evals
            if use_gpu:
                cleanup_cuda()
        return out

    layer_types_ref = list(lm.config.layer_types)
    mixer_cov_res = summarize_cov(cov_mixer, probe_layers_4k, 4096, use_gpu=True)
    mlp_in_cov_res = summarize_cov(cov_mlp_in, probe_layers_4k, 4096, use_gpu=True)
    del cov_mixer, cov_mlp_in

    # Free 10.5 GB model weights before running 12288x12288 GPU eigvalsh (avoids 2.3GB cuSOLVER OOM)
    return {
        "n_examples": len(texts),
        "tokens_streamed": tok_count,
        "streaming_time_sec": round(elapsed, 2),
        "static_vram_mb": round(static_mb, 2),
        "peak_vram_mb": round(peak_mb, 2),
        "activation_overhead_mb": round(max(0.0, peak_mb - static_mb), 2),
        "mixer_out_proj_cov": mixer_cov_res,
        "mlp_in_cov": mlp_in_cov_res,
        "_pending_down_cov": cov_down,
        "_down_layers": probe_layers_12k,
        "_layer_types": layer_types_ref,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[Phase 1] Loading Ornith-1.0-9B in 8-bit (float16 compute)...")
    model, tokenizer = load_ornith_8bit(device=device, dtype=torch.float16, drop_visual=True)
    static_mb = measure_static_vram_mb(model)
    print(f"[Phase 1] Static 8-bit VRAM (no visual): {static_mb:.2f} MB")

    print("[Phase 1] Checking unembedding Voronoi cone geometry...")
    voronoi_info = check_voronoi_cone(model, tokenizer)
    print("Voronoi cone:", json.dumps(voronoi_info, indent=2))

    print("[Phase 1] Running Forward Signal Localization & Logit Lens across 32 layers...")
    signal_report = forward_signal_localization(model, tokenizer, device)
    for row in signal_report["layers"]:
        print(
            f"L{row['layer']:02d} ({row['layer_type'][:4]}) | norm={row['res_l2_norm']:6.1f} "
            f"| res_diff={row['res_l2_diff']:6.2f} (rel={row['res_rel_diff']:.3f}) "
            f"| mix={row['mixer_l2_diff']:6.2f} mlp={row['mlp_l2_diff']:6.2f} "
            f"| ctx_p={row['ctx_target_prob']:.4f} rank={row['ctx_target_rank']:7.1f} noctx_p={row['noctx_target_prob']:.6f}"
        )

    print("[Phase 1] Profiling dequantized 8-bit weight spectra across 32 layers...")
    weights_report = profile_weights(model)
    weights_summary = {
        k: {
            "mean_eff_rank": round(sum(x["eff_rank"] for x in v) / len(v), 2),
            "min_eff_rank": min(x["eff_rank"] for x in v),
            "max_eff_rank": max(x["eff_rank"] for x in v),
            "mean_r99": round(sum(x["rank_99pct_energy"] for x in v) / len(v), 1),
        }
        for k, v in weights_report.items()
    }
    print("Weights Summary:", json.dumps(weights_summary, indent=2))

    print("[Phase 1] Running Streaming Null-Space Capacity (smol-smoltalk)...")
    null_report = streaming_null_space_capacity(model, tokenizer, device)

    # Free model from GPU before running 12288x12288 eigvalsh cleanly on GPU
    cov_down = null_report.pop("_pending_down_cov")
    down_layers = null_report.pop("_down_layers")
    layer_types = null_report.pop("_layer_types")
    tok_count = null_report["tokens_streamed"]
    del model
    cleanup_cuda()

    down_res = []
    for l in down_layers:
        C = cov_down[l] / max(tok_count, 1)
        C = 0.5 * (C + C.T)
        C = C.to(device)
        evals = torch.linalg.eigvalsh(C).clamp(min=0.0)
        total = float(evals.sum().item()) + 1e-12
        emax = float(evals[-1].item())
        emean = float(evals.mean().item())
        n_lt_1e3_max = int((evals < 1e-3 * emax).sum().item())
        n_lt_1e4_max = int((evals < 1e-4 * emax).sum().item())
        n_lt_01_mean = int((evals < 0.01 * emean).sum().item())
        tail_half_energy = float(evals[: 12288 // 2].sum().item() / total)
        down_res.append({
            "layer": l,
            "layer_type": layer_types[l],
            "dim": 12288,
            "eval_max": round(emax, 4),
            "eval_median": round(float(evals[12288 // 2].item()), 6),
            "eval_min": round(float(evals[0].item()), 8),
            "null_dirs_lt_1e-3_max": n_lt_1e3_max,
            "null_dirs_lt_1e-4_max": n_lt_1e4_max,
            "null_dirs_lt_0.01_mean": n_lt_01_mean,
            "bottom_half_energy_frac": round(tail_half_energy, 6),
        })
        print(f"  null-space L{l} (dim=12288): lt_1e-3_max={n_lt_1e3_max}, bot_half_E={tail_half_energy:.5f}", flush=True)
        del C, evals
        cleanup_cuda()
    null_report["down_proj_cov"] = down_res
    del cov_down

    full_report = {
        "model": MODEL_ID,
        "quantization": "bitsandbytes_8bit_LLM_int8",
        "compute_dtype": "torch.float16",
        "static_vram_mb": round(static_mb, 2),
        "voronoi_cone": voronoi_info,
        "weights_summary": weights_summary,
        "weights_per_layer": weights_report,
        "signal_localization": signal_report,
        "null_space_capacity": null_report,
    }
    out_path = OUT_DIR / "anatomy_report.json"
    out_path.write_text(json.dumps(full_report, indent=2))
    print(f"\nSaved full anatomy report to {out_path}")
    print("Null-space meta:", {k: v for k, v in null_report.items() if not k.endswith("_cov")})
    for name in ("mixer_out_proj_cov", "mlp_in_cov", "down_proj_cov"):
        print(name)
        for row in null_report[name]:
            print(" ", row)


if __name__ == "__main__":
    main()
