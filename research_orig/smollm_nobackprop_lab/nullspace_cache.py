"""Streaming Calibration Covariance (C0 = X0^T X0) Accumulation & Whitening Utilities."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch

from benchmark import load_smoltalk_splits
from guards import cleanup_cuda, measure_static_vram_mb

CACHE_PATH = Path("/content/smollm_nobackprop_lab/c0_cache_L16_31.pt")


@torch.no_grad()
def compute_or_load_c0(
    model,
    tokenizer,
    device: str,
    target_layers: Iterable[int] = tuple(range(16, 32)),
    target_modules: Iterable[str] = ("o_proj", "down_proj"),
    n_cal: int = 160,
    max_len: int = 256,
    batch_size: int = 8,
    force_recompute: bool = False,
) -> Tuple[Dict[Tuple[int, str], torch.Tensor], Dict[str, float]]:
    """Stream <= 200 smol-smoltalk examples in chunks under torch.no_grad() to build C0 = X0^T X0 / N."""
    target_layers = list(target_layers)
    target_modules = list(target_modules)

    if CACHE_PATH.exists() and not force_recompute:
        payload = torch.load(CACHE_PATH, map_location="cpu", weights_only=True)
        covs = payload["covs"]
        if all((l, m) in covs for l in target_layers for m in target_modules):
            return covs, payload["meta"]

    cal_texts, _ = load_smoltalk_splits(tokenizer, n_cal=n_cal, n_kl_eval=16, max_len=max_len)

    covs_gpu: Dict[Tuple[int, str], torch.Tensor] = {}
    for l_idx in target_layers:
        layer = model.model.layers[l_idx]
        for m_name in target_modules:
            mod = layer.self_attn.o_proj if m_name == "o_proj" else layer.mlp.down_proj
            dim = mod.in_features
            covs_gpu[(l_idx, m_name)] = torch.zeros((dim, dim), dtype=torch.float64, device=device)

    tok_count = 0
    hooks = []

    def make_hook(key: Tuple[int, str]):
        def hook(mod, inp, out):
            x = inp[0].detach()
            x_flat = x.reshape(-1, x.shape[-1]).to(torch.float64)
            covs_gpu[key] += x_flat.T @ x_flat
        return hook

    for l_idx in target_layers:
        layer = model.model.layers[l_idx]
        for m_name in target_modules:
            mod = layer.self_attn.o_proj if m_name == "o_proj" else layer.mlp.down_proj
            hooks.append(mod.register_forward_hook(make_hook((l_idx, m_name))))

    cleanup_cuda()
    static_mb = measure_static_vram_mb(model)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        for i in range(0, len(cal_texts), batch_size):
            batch = cal_texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
            ).to(device)
            tok_count += int(enc.attention_mask.sum().item())
            _ = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)
    finally:
        tokenizer.padding_side = prev_side
        for h in hooks:
            h.remove()

    torch.cuda.synchronize()
    peak_mb = float(torch.cuda.max_memory_allocated() / (1024.0 ** 2))
    elapsed = time.perf_counter() - t0

    covs_cpu: Dict[Tuple[int, str], torch.Tensor] = {}
    for k, mat in covs_gpu.items():
        covs_cpu[k] = (mat / max(tok_count, 1)).float().cpu()
    del covs_gpu
    cleanup_cuda()

    meta = {
        "n_cal": n_cal,
        "tokens_streamed": tok_count,
        "stream_time_sec": round(elapsed, 3),
        "stream_peak_vram_mb": round(peak_mb, 2),
        "stream_overhead_mb": round(max(0.0, peak_mb - static_mb), 2),
    }
    torch.save({"covs": covs_cpu, "meta": meta}, CACHE_PATH)
    return covs_cpu, meta


def solve_whitened_ridge_lora(
    X: torch.Tensor,
    dY: torch.Tensor,
    C0: torch.Tensor,
    rank: int = 16,
    lambda_c0: float = 10.0,
    lambda_eye: float = 1e-2,
    X_neg: torch.Tensor | None = None,
    lambda_neg: float = 0.0,
    eps_whiten: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Closed-form Calibration-Whitened Ridge + Optimal Rank-r SVD.

    Minimizes ||X dW^T - dY||_F^2 + lambda_c0 * tr(dW C0 dW^T) + lambda_neg * ||X_neg dW^T||_F^2 + lambda_eye ||dW||_F^2
    and truncates in whitened coordinates so top-r singular vectors maximize edit gain per unit KL energy.

    Returns A [r, d_in], B [d_out, r], stats.
    """
    Xf = X.float()
    dYf = dY.float()
    C0f = C0.float()
    d_in = Xf.shape[1]
    device = Xf.device

    # Total protection covariance: P_cov = lambda_c0 * C0 + lambda_neg * (X_neg^T X_neg) + lambda_eye * I
    P_cov = lambda_c0 * C0f.to(device)
    if X_neg is not None and X_neg.numel() > 0 and lambda_neg > 0.0:
        Xn = X_neg.float().to(device)
        P_cov = P_cov + (lambda_neg / max(Xn.shape[0], 1)) * (Xn.T @ Xn)

    mean_diag = float(torch.trace(P_cov).item() / max(d_in, 1))
    ridge_diag = max(lambda_eye, eps_whiten * max(mean_diag, 1e-6))
    P_cov = 0.5 * (P_cov + P_cov.T) + ridge_diag * torch.eye(d_in, device=device, dtype=torch.float32)

    # Whitening factor S = P_cov^{1/2} via eigendecomposition
    evals, evecs = torch.linalg.eigh(P_cov)
    evals = evals.clamp(min=1e-7)
    S_inv = (evecs / torch.sqrt(evals).unsqueeze(0)) @ evecs.T

    # Solve in whitened coordinates: X_w = X @ S_inv, Theta_w = (X_w^T X_w + I)^{-1} X_w^T dY
    X_w = Xf @ S_inv
    M_samples = X_w.shape[0]
    if M_samples < d_in:
        # Dual / Woodbury form: Theta_w = X_w^T (X_w X_w^T + I)^{-1} dY
        gram = X_w @ X_w.T + torch.eye(M_samples, device=device, dtype=torch.float32)
        dual_coef = torch.linalg.solve(gram, dYf)
        Theta_w = X_w.T @ dual_coef  # [d_in, d_out]
    else:
        gram = X_w.T @ X_w + torch.eye(d_in, device=device, dtype=torch.float32)
        Theta_w = torch.linalg.solve(gram, X_w.T @ dYf)

    # Optimal rank-r SVD in whitened metric: Theta_w^T = U Sigma V^T
    U, S, Vh = torch.linalg.svd(Theta_w.T, full_matrices=False)
    r_eff = min(rank, S.shape[0])
    Ur = U[:, :r_eff]
    Sr = S[:r_eff]
    Vhr = Vh[:r_eff, :]

    B = Ur * Sr.unsqueeze(0)        # [d_out, r]
    A = Vhr @ S_inv                 # [r, d_in]

    # Diagnostics (norms & residual fit, no raw tensors)
    pred = (Xf @ A.T) @ B.T
    fit_rel_err = float((pred - dYf).norm().item() / (dYf.norm().item() + 1e-8))
    cal_energy = float((Sr * Sr).sum().item())
    sv_captured = float((Sr * Sr).sum().item() / ((S * S).sum().item() + 1e-8))

    stats = {
        "fit_rel_err": round(fit_rel_err, 4),
        "sv_energy_captured": round(sv_captured, 4),
        "cal_whitened_energy": round(cal_energy, 4),
        "w_fro_norm": round(float((B @ A).norm().item()), 4),
    }
    return A, B, stats


def solve_bounded_whitened_ridge_lora(
    X: torch.Tensor,
    dY: torch.Tensor,
    C0: torch.Tensor,
    rank: int = 16,
    lambda_c0: float = 5.0,
    lambda_neg: float = 5.0,
    lambda_fro: float = 1.0,
    ridge_alpha: float = 0.05,
    max_fro_norm: float = 4.0,
    X_neg: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Scale-invariant Whitened Null-Space Ridge + Optimal Rank-r SVD with Frobenius Safety Cap."""
    Xf = X.float()
    dYf = dY.float()
    C0f = C0.float()
    d_in = Xf.shape[1]
    device = Xf.device

    # Normalize C0 to unit mean eigenvalue so regularization is strictly scale-invariant across layers/dims
    c0_scale = max(float(torch.trace(C0f).item() / max(d_in, 1)), 1e-8)
    P_cov = (lambda_c0 / c0_scale) * C0f.to(device)

    if X_neg is not None and X_neg.numel() > 0 and lambda_neg > 0.0:
        Xn = X_neg.float().to(device)
        C_neg = (Xn.T @ Xn) / max(Xn.shape[0], 1)
        neg_scale = max(float(torch.trace(C_neg).item() / max(d_in, 1)), 1e-8)
        P_cov = P_cov + (lambda_neg / neg_scale) * C_neg

    P_cov = 0.5 * (P_cov + P_cov.T) + lambda_fro * torch.eye(d_in, device=device, dtype=torch.float32)

    evals, evecs = torch.linalg.eigh(P_cov)
    evals = evals.clamp(min=lambda_fro * 0.5)
    S_inv = (evecs / torch.sqrt(evals).unsqueeze(0)) @ evecs.T

    X_w = Xf @ S_inv
    M_samples = X_w.shape[0]
    gram_diag_mean = max(float((X_w * X_w).sum().item() / max(M_samples, 1)), 1e-6)
    ridge = ridge_alpha * gram_diag_mean

    if M_samples < d_in:
        gram = X_w @ X_w.T + ridge * torch.eye(M_samples, device=device, dtype=torch.float32)
        dual_coef = torch.linalg.solve(gram, dYf)
        Theta_w = X_w.T @ dual_coef
    else:
        gram = X_w.T @ X_w + ridge * torch.eye(d_in, device=device, dtype=torch.float32)
        Theta_w = torch.linalg.solve(gram, X_w.T @ dYf)

    U, S, Vh = torch.linalg.svd(Theta_w.T, full_matrices=False)
    r_eff = min(rank, S.shape[0])
    Ur = U[:, :r_eff]
    Sr = S[:r_eff]
    Vhr = Vh[:r_eff, :]

    B = Ur * Sr.unsqueeze(0)
    A = Vhr @ S_inv

    w_fro = float((B @ A).norm().item())
    if max_fro_norm > 0.0 and w_fro > max_fro_norm:
        scale = max_fro_norm / (w_fro + 1e-8)
        B = B * scale
        w_fro = max_fro_norm

    pred = (Xf @ A.T) @ B.T
    fit_rel_err = float((pred - dYf).norm().item() / (dYf.norm().item() + 1e-8))
    sv_captured = float((Sr * Sr).sum().item() / ((S * S).sum().item() + 1e-8))

    # Measure calibration output norm tr(dW C0 dW^T)
    BA = B @ A
    cal_pert = float(((BA @ C0f.to(device)) * BA).sum().item())

    stats = {
        "fit_rel_err": round(fit_rel_err, 4),
        "sv_energy_captured": round(sv_captured, 4),
        "w_fro_norm": round(w_fro, 4),
        "cal_pert_tr": round(cal_pert, 4),
    }
    return A, B, stats
