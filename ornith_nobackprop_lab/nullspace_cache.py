"""Streaming Calibration Covariance (C0 = X0^T X0) Cache & Memory-Safe Whitening Utilities for Ornith-1.0-9B."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from benchmark import load_smoltalk_splits
from guards import cleanup_cuda, measure_static_vram_mb
from model_loader import get_layers, get_text_model

CACHE_DIR = Path("/content/ornith_nobackprop_lab/c0_cache_L23_31")


def symmetrize_inplace(mat: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    """Blockwise in-place symmetrization without allocating a full duplicate matrix."""
    n = mat.shape[0]
    for rs in range(0, n, chunk):
        re = min(n, rs + chunk)
        diag = mat[rs:re, rs:re]
        mat[rs:re, rs:re].copy_(0.5 * (diag + diag.T))
        if re < n:
            mat[rs:re, re:n].add_(mat[re:n, rs:re].T).mul_(0.5)
            mat[re:n, rs:re].copy_(mat[rs:re, re:n].T)
    return mat


def _get_module_for_name(layer, layer_type: str, m_name: str):
    if m_name in ("gate_proj", "up_proj", "down_proj"):
        return getattr(layer.mlp, m_name)
    if m_name == "mixer_out_proj":
        return layer.self_attn.o_proj if layer_type == "full_attention" else layer.linear_attn.out_proj
    if m_name == "attn_in":
        return layer.self_attn.q_proj if layer_type == "full_attention" else layer.linear_attn.in_proj_qkv
    if m_name == "mlp_in":
        return layer.mlp.gate_proj
    raise ValueError(f"Unsupported C0 module key: {m_name}")


@torch.no_grad()
def compute_or_load_c0(
    model,
    tokenizer,
    device: str,
    target_layers: Iterable[int] = tuple(range(24, 32)),
    target_modules: Iterable[str] = ("mixer_out_proj", "down_proj"),
    n_cal: int = 128,
    max_len: int = 192,
    batch_size: int = 2,
    force_recompute: bool = False,
) -> Tuple[Dict[Tuple[int, str], torch.Tensor], Dict[str, float]]:
    """Stream <= 200 smol-smoltalk examples to build C0 = X0^T X0 / N (stored in fp16 on disk for RAM safety)."""
    target_layers = list(target_layers)
    target_modules = list(target_modules)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = CACHE_DIR / "meta.pt"

    if meta_path.exists() and not force_recompute:
        all_present = all((CACHE_DIR / f"L{l}_{m}.pt").exists() for l in target_layers for m in target_modules)
        if all_present:
            meta = torch.load(meta_path, map_location="cpu", weights_only=True)
            covs: Dict[Tuple[int, str], torch.Tensor] = {}
            for l in target_layers:
                for m in target_modules:
                    covs[(l, m)] = torch.load(CACHE_DIR / f"L{l}_{m}.pt", map_location="cpu", weights_only=True)
            return covs, meta

    cal_texts, _ = load_smoltalk_splits(tokenizer, n_cal=n_cal, n_kl_eval=16, max_len=max_len)
    lm = get_text_model(model)
    layers = get_layers(model)

    # Accumulate on CPU in float32 (kept compact by targeting only late resolution layers)
    covs_cpu_acc: Dict[Tuple[int, str], torch.Tensor] = {}
    for l_idx in target_layers:
        layer = layers[l_idx]
        lt = lm.config.layer_types[l_idx]
        for m_name in target_modules:
            mod = _get_module_for_name(layer, lt, m_name)
            dim = mod.in_features
            covs_cpu_acc[(l_idx, m_name)] = torch.zeros((dim, dim), dtype=torch.float32, device="cpu")

    tok_count = 0
    hooks = []

    def make_hook(key: Tuple[int, str]):
        def hook(mod, inp, out):
            x = inp[0].detach()
            x_flat = x.reshape(-1, x.shape[-1]).float()
            covs_cpu_acc[key] += (x_flat.T @ x_flat).cpu()
        return hook

    for l_idx in target_layers:
        layer = layers[l_idx]
        lt = lm.config.layer_types[l_idx]
        for m_name in target_modules:
            mod = _get_module_for_name(layer, lt, m_name)
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
            del enc
    finally:
        tokenizer.padding_side = prev_side
        for h in hooks:
            h.remove()

    torch.cuda.synchronize()
    peak_mb = float(torch.cuda.max_memory_allocated() / (1024.0 ** 2))
    elapsed = time.perf_counter() - t0

    covs_out: Dict[Tuple[int, str], torch.Tensor] = {}
    for (l_idx, m_name), mat in covs_cpu_acc.items():
        c_norm = mat / max(tok_count, 1)
        c_norm = 0.5 * (c_norm + c_norm.T)
        # Store 12288x12288 in fp16 (302 MB vs 604 MB) and 4096x4096 in fp32 (67 MB)
        c_save = c_norm.half() if c_norm.shape[0] > 4096 else c_norm.float()
        torch.save(c_save, CACHE_DIR / f"L{l_idx}_{m_name}.pt")
        covs_out[(l_idx, m_name)] = c_save
    del covs_cpu_acc
    cleanup_cuda()

    meta = {
        "n_cal": n_cal,
        "tokens_streamed": tok_count,
        "stream_time_sec": round(elapsed, 3),
        "stream_peak_vram_mb": round(peak_mb, 2),
        "stream_overhead_mb": round(max(0.0, peak_mb - static_mb), 2),
    }
    torch.save(meta, meta_path)
    return covs_out, meta


def compute_inv_sqrt_apply(
    P_cov: torch.Tensor,
    alpha_scale: float = 1.0,
    min_floor: float = 1e-6,
    lowrank_k: int = 512,
) -> Tuple[Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    """Compute either exact S_inv [d,d] (for d <= 4096) or low-rank spectral factor (U_k, shrink_k) for d > 4096.

    For d > 4096 (e.g. 12288), dense eigh uses >2.3 GB cuSOLVER workspace and OOMs alongside a 10.5 GB model.
    Low-rank spectral factor represents (I + alpha * C_norm)^{-1/2} = I - U_k diag(shrink_k) U_k^T in <300 MB.
    """
    d_in = P_cov.shape[0]
    if d_in <= 4096:
        mat = 0.5 * (P_cov + P_cov.T)
        evals, evecs = torch.linalg.eigh(mat)
        evals = evals.clamp(min=min_floor)
        S_inv = (evecs / torch.sqrt(evals).unsqueeze(0)) @ evecs.T
        return S_inv, None

    # Low-rank top-k spectral approximation for d=12288
    # Note: P_cov here should already be (C_prot_norm); if caller passed I + alpha*C_prot, subtract I
    q = min(lowrank_k, d_in - 2)
    U, S, _ = torch.svd_lowrank(P_cov, q=q, niter=2)
    evals_k = S.clamp(min=0.0)
    # Factor for (eye_scale * I + alpha_scale * C)^{-1/2}:
    # Base scale is min_floor^{-1/2}, with top-k shrinkage relative to base
    shrink_k = 1.0 - torch.rsqrt(1.0 + alpha_scale * evals_k)
    return None, (U, shrink_k)


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
    """Scale-invariant Whitened Null-Space Ridge + Optimal Rank-r SVD with Frobenius Safety Cap.

    Supports both exact eigh (d <= 4096) and memory-safe low-rank spectral whitening (d = 12288).
    """
    Xf = X.float()
    dYf = dY.float()
    d_in = Xf.shape[1]
    device = Xf.device

    C_prot = C0.to(device=device, dtype=torch.float32).clone()
    c0_scale = max(float(torch.trace(C_prot).item() / max(d_in, 1)), 1e-8)
    C_prot.mul_(lambda_c0 / c0_scale)

    if X_neg is not None and X_neg.numel() > 0 and lambda_neg > 0.0:
        Xn = X_neg.float().to(device)
        n_neg = max(Xn.shape[0], 1)
        neg_scale = max(float((Xn * Xn).sum().item() / (n_neg * max(d_in, 1))), 1e-8)
        C_prot.addmm_(Xn.T, Xn, alpha=(lambda_neg / (n_neg * neg_scale)))
        del Xn

    if d_in <= 4096:
        C_prot = 0.5 * (C_prot + C_prot.T)
        P_cov = C_prot + lambda_fro * torch.eye(d_in, device=device, dtype=torch.float32)
        del C_prot
        evals, evecs = torch.linalg.eigh(P_cov)
        evals = evals.clamp(min=lambda_fro * 0.5)
        S_inv = (evecs / torch.sqrt(evals).unsqueeze(0)) @ evecs.T
        X_w = Xf @ S_inv
        apply_s_inv_right = lambda V: V @ S_inv
        del P_cov, evals, evecs
    else:
        torch.cuda.empty_cache()
        symmetrize_inplace(C_prot, chunk=1024)
        # Factor (lambda_fro * I + C_prot)^{-1/2} = lambda_fro^{-1/2} * (I + (1/lambda_fro) * C_prot)^{-1/2}
        U_k, S_k, _ = torch.svd_lowrank(C_prot, q=512, niter=2)
        del C_prot
        torch.cuda.empty_cache()
        base_inv_sqrt = lambda_fro ** -0.5
        shrink_k = 1.0 - torch.rsqrt(1.0 + S_k.clamp(min=0.0) / max(lambda_fro, 1e-6))

        def apply_s_inv_right(M: torch.Tensor) -> torch.Tensor:
            # M @ S_inv = base_inv_sqrt * (M - ((M @ U_k) * shrink_k) @ U_k.T)
            proj = (M @ U_k) * shrink_k.unsqueeze(0)
            return base_inv_sqrt * (M - proj @ U_k.T)

        X_w = apply_s_inv_right(Xf)

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
    A = apply_s_inv_right(Vhr)

    w_fro = float((B @ A).norm().item())
    if max_fro_norm > 0.0 and w_fro > max_fro_norm:
        scale = max_fro_norm / (w_fro + 1e-8)
        B = B * scale
        w_fro = max_fro_norm

    pred = (Xf @ A.T) @ B.T
    fit_rel_err = float((pred - dYf).norm().item() / (dYf.norm().item() + 1e-8))
    sv_captured = float((Sr * Sr).sum().item() / ((S * S).sum().item() + 1e-8))

    # Measure tr(dW C0 dW^T) = tr((B^T B) (A C0 A^T)) via compact rank-r factor on CPU/GPU without [d_out, d_in]
    A_cpu = A.detach().float().cpu()
    BtB_cpu = (B.detach().float().T @ B.detach().float()).cpu()
    AC0_cpu = A_cpu @ C0.float().cpu()
    cal_pert = float((BtB_cpu * (AC0_cpu @ A_cpu.T)).sum().item())
    del A_cpu, BtB_cpu, AC0_cpu

    stats = {
        "fit_rel_err": round(fit_rel_err, 4),
        "sv_energy_captured": round(sv_captured, 4),
        "w_fro_norm": round(w_fro, 4),
        "cal_pert_tr": round(cal_pert, 4),
    }
    return A, B, stats
