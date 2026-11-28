"""Iteration 0 (Milestone 1): Universal Multi-Domain Baselines (Base_Unedited, InContext_Oracle, FullBackprop_Baseline).

Evaluates SmolLM2-360M-Instruct (fp16) and 8-bit Ornith-1.0-9B across all 3 orthogonal domains:
  1. code_api (veloxdb / nexflow, isolated 2s subprocess sandbox)
  2. pop_culture (narrative prose lore + multihop + real pop-culture bleed + refusal)
  3. bio_logic (narrative biography prose + multihop logic + sibling bleed)
Plus Shared Retention Suite (25 untouched real_anchors, 60 general_qa, 64 smol-smoltalk KL).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from guards import BaselineBackpropProfiler, LoRAManager, SynthesisMetrics, cleanup_cuda, measure_static_vram_mb
from unified_model_utils import get_layers, get_text_model, load_model_by_scale
from universal_benchmark import DOMAINS, SEEDS, get_domain_batch, load_smoltalk_disjoint
from universal_eval import (
    LAB_DIR,
    append_universal_record,
    evaluate_domain,
    evaluate_shared_retention,
    get_shared_base_reference,
)


class FullBackpropLoRALinear(nn.Module):
    """Standard trainable LoRA wrapper (r=16, alpha=32) strictly for FullBackprop_Baseline."""

    def __init__(self, base_linear: nn.Module, r: int = 16, lora_alpha: float = 32.0) -> None:
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        if getattr(self.base, "bias", None) is not None:
            self.base.bias.requires_grad_(False)
        self.r = r
        self.scaling = float(lora_alpha) / float(r)
        self.active = True
        dev = base_linear.weight.device
        self.lora_A = nn.Parameter(torch.empty((r, base_linear.in_features), device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros((base_linear.out_features, r), device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.active:
            xf = x.float()
            delta = (F.linear(F.linear(xf, self.lora_A), self.lora_B) * self.scaling).to(out.dtype)
            out = out + delta
        return out


class FullBackpropLoRAManager:
    TARGET_MAP = LoRAManager.TARGET_MAP

    def __init__(self, model: nn.Module, r: int = 16, lora_alpha: float = 32.0) -> None:
        self.model = model
        self.wrapped: Dict[tuple, FullBackpropLoRALinear] = {}
        for p in model.parameters():
            p.requires_grad_(False)
        for l_idx, layer in enumerate(get_layers(model)):
            for mod_name, (pa, la) in self.TARGET_MAP.items():
                if hasattr(layer, pa):
                    parent = getattr(layer, pa)
                    if hasattr(parent, la):
                        orig = getattr(parent, la)
                        w = FullBackpropLoRALinear(orig, r=r, lora_alpha=lora_alpha)
                        setattr(parent, la, w)
                        self.wrapped[(l_idx, mod_name)] = w

    def trainable_parameters(self) -> List[nn.Parameter]:
        out: List[nn.Parameter] = []
        for w in self.wrapped.values():
            out.extend([w.lora_A, w.lora_B])
        return out

    def uninstall(self) -> None:
        layers = get_layers(self.model)
        for (l_idx, mod_name), w in self.wrapped.items():
            pa, la = self.TARGET_MAP[mod_name]
            setattr(getattr(layers[l_idx], pa), la, w.base)
        self.wrapped.clear()


def _cleanup_failed_backprop(model: nn.Module) -> None:
    layers = get_layers(model)
    for layer in layers:
        for _, (pa, la) in LoRAManager.TARGET_MAP.items():
            if hasattr(layer, pa):
                parent = getattr(layer, pa)
                cur = getattr(parent, la, None)
                if isinstance(cur, FullBackpropLoRALinear):
                    setattr(parent, la, cur.base)
    if hasattr(model, "gradient_checkpointing_disable"):
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass
    model.eval()
    model.zero_grad(set_to_none=True)
    cleanup_cuda()


def _train_full_backprop_doc(
    model: nn.Module,
    tokenizer,
    doc_text: str,
    steps: int,
    chunk_len: int,
    batch_size: int,
    lr: float,
    device: str,
    static_vram_mb: float,
    use_grad_ckpt: bool,
) -> Tuple[FullBackpropLoRAManager, SynthesisMetrics]:
    mgr = FullBackpropLoRAManager(model, r=16, lora_alpha=32.0)
    model.train()
    lm = get_text_model(model)
    emb_hook = None
    if use_grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        emb_hook = lm.embed_tokens.register_forward_hook(
            lambda m, inp, out: out.requires_grad_(True) if isinstance(out, torch.Tensor) else out
        )

    toks = tokenizer.encode(doc_text, add_special_tokens=False)
    chunks: List[List[int]] = []
    stride = max(64, chunk_len - 64) if chunk_len < len(toks) else chunk_len
    for s in range(0, len(toks), stride):
        sl = toks[s : s + chunk_len]
        if len(sl) >= 16:
            chunks.append(sl)
        if s + chunk_len >= len(toks):
            break
    if not chunks:
        chunks = [toks[:chunk_len]]

    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    max_l = max(len(c) for c in chunks)
    input_ids = torch.full((len(chunks), max_l), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros((len(chunks), max_l), dtype=torch.long, device=device)
    for i, c in enumerate(chunks):
        input_ids[i, : len(c)] = torch.tensor(c, dtype=torch.long, device=device)
        attn_mask[i, : len(c)] = 1
    labels = input_ids.clone()
    labels[attn_mask == 0] = -100

    params = mgr.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=lr)

    try:
        with BaselineBackpropProfiler(model, static_vram_mb=static_vram_mb) as prof:
            n = input_ids.shape[0]
            n_acc = max(1, (n + batch_size - 1) // batch_size)
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                for s in range(0, n, batch_size):
                    e = min(n, s + batch_size)
                    out = model(
                        input_ids=input_ids[s:e],
                        attention_mask=attn_mask[s:e],
                        labels=labels[s:e],
                        use_cache=False,
                    )
                    (out.loss / n_acc).backward()
                    del out
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
    finally:
        if emb_hook is not None:
            emb_hook.remove()
        if use_grad_ckpt:
            model.gradient_checkpointing_disable()
        model.eval()
        del opt, input_ids, attn_mask, labels
    return mgr, prof.metrics


def run_scale_baselines(scale: str, only_fb: bool = False) -> Dict[str, object]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Iteration 0 Universal Baselines: scale={scale} ===", flush=True)
    model, tokenizer, spec = load_model_by_scale(scale, device=device)
    static_mb = spec.static_vram_mb
    _, kl_texts = load_smoltalk_disjoint(tokenizer, scale, n_cal=128, n_kl=64)
    _ = get_shared_base_reference(model, tokenizer, spec, device)

    summary_path = LAB_DIR / "iter_0_universal_summary.json"
    prev_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    out: Dict[str, object] = prev_summary.get(scale, {"scale": scale, "static_vram_mb": round(static_mb, 2), "runs": {}})
    out["scale"] = scale
    out["static_vram_mb"] = round(static_mb, 2)

    if not only_fb:
        # 1. Base_Unedited across all 3 domains x 2 seeds + Shared Retention
        base_ret = evaluate_shared_retention(model, tokenizer, spec, kl_texts, device, compute_kl=False)
        base_domains = {}
        for seed in SEEDS:
            base_domains[str(seed)] = {}
            for dom in DOMAINS:
                res = evaluate_domain(model, tokenizer, spec, dom, seed, device, prefix_doc=False)
                base_domains[str(seed)][dom] = res
                print(f"  [Base {scale} s={seed} {dom}] {res}", flush=True)
        m_base = SynthesisMetrics(
            peak_vram_mb=static_mb, static_vram_mb=static_mb, activation_vram_overhead_mb=0.0,
            wall_time_sec=0.0, uses_full_backprop=False, regime="base_unedited", backbone_grad_disabled_verified=True,
        )
        out["runs"]["Base_Unedited"] = {"shared_retention": base_ret, "domains": base_domains}
        append_universal_record(0, "M1", "Base_Unedited", scale, 42, {"shared_retention": base_ret, "domains": base_domains}, m_base, status="baseline")
        print(f"  [Base {scale} shared_retention] {base_ret}", flush=True)

        # 2. InContext_Oracle across all 3 domains x 2 seeds
        oracle_domains = {}
        for seed in SEEDS:
            oracle_domains[str(seed)] = {}
            for dom in DOMAINS:
                res = evaluate_domain(model, tokenizer, spec, dom, seed, device, prefix_doc=True)
                oracle_domains[str(seed)][dom] = res
                print(f"  [Oracle {scale} s={seed} {dom}] {res}", flush=True)
        oracle_ret = evaluate_shared_retention(
            model, tokenizer, spec, kl_texts, device, compute_kl=False,
            prefix_context=f"Reference Document:\n{get_domain_batch('code_api', 42)['doc_text']}",
        )
        out["runs"]["InContext_Oracle"] = {"shared_retention": oracle_ret, "domains": oracle_domains}
        append_universal_record(0, "M1", "InContext_Oracle", scale, 42, {"shared_retention": oracle_ret, "domains": oracle_domains}, m_base, status="oracle")
        print(f"  [Oracle {scale} shared_retention] {oracle_ret}", flush=True)

    # 3. FullBackprop_Baseline (seq_len=2048 OOM/VRAM check + seq_len=512 per-domain seed 42)
    doc_code42 = get_domain_batch("code_api", 42)["doc_text"]
    seq2048_probe = {"oom_no_ckpt": False, "oom_with_ckpt": False, "peak_vram_mb": 0.0, "overhead_mb": 0.0}
    if scale == "large":
        for use_ckpt in (False, True):
            try:
                cleanup_cuda()
                torch.cuda.reset_peak_memory_stats()
                mgr, met = _train_full_backprop_doc(
                    model, tokenizer, doc_code42, steps=1, chunk_len=2048, batch_size=1,
                    lr=2e-4, device=device, static_vram_mb=static_mb, use_grad_ckpt=use_ckpt,
                )
                seq2048_probe["peak_vram_mb"] = round(met.peak_vram_mb, 2)
                seq2048_probe["overhead_mb"] = round(met.activation_vram_overhead_mb, 2)
                mgr.uninstall()
                cleanup_cuda()
                break
            except torch.cuda.OutOfMemoryError as e:
                peak_oom = float(torch.cuda.max_memory_allocated() / (1024.0 ** 2))
                seq2048_probe["peak_vram_mb"] = round(max(seq2048_probe["peak_vram_mb"], peak_oom), 2)
                seq2048_probe["overhead_mb"] = round(max(0.0, seq2048_probe["peak_vram_mb"] - static_mb), 2)
                if not use_ckpt:
                    seq2048_probe["oom_no_ckpt"] = True
                else:
                    seq2048_probe["oom_with_ckpt"] = True
                print(f"  [FullBackprop 9B seq=2048 ckpt={use_ckpt}] OOM at {peak_oom:.2f} MB: {e}", flush=True)
                _cleanup_failed_backprop(model)
    out["runs"]["FullBackprop_seq2048_probe"] = seq2048_probe

    fb_domains = {}
    chunk_len = 2048 if scale == "small" else 512
    use_ckpt = False if scale == "small" else True
    fb_steps = 12 if scale == "small" else 8
    fb_lr = 5e-4 if scale == "small" else 2e-4
    for dom in DOMAINS:
        cleanup_cuda()
        doc = get_domain_batch(dom, 42)["doc_text"]
        mgr, met = _train_full_backprop_doc(
            model, tokenizer, doc, steps=fb_steps, chunk_len=chunk_len, batch_size=1,
            lr=fb_lr, device=device, static_vram_mb=static_mb, use_grad_ckpt=use_ckpt,
        )
        d_res = evaluate_domain(model, tokenizer, spec, dom, 42, device, prefix_doc=False, lora_manager=mgr)
        r_res = evaluate_shared_retention(model, tokenizer, spec, kl_texts, device, lora_manager=mgr, compute_kl=True)
        fb_domains[dom] = {
            "domain_eval": d_res,
            "shared_retention": r_res,
            "peak_vram_mb": round(met.peak_vram_mb, 2),
            "overhead_mb": round(met.activation_vram_overhead_mb, 2),
            "wall_time_sec": round(met.wall_time_sec, 2),
        }
        print(f"  [FullBackprop {scale} s=42 {dom}] {d_res} | ret={r_res} | ovh={met.activation_vram_overhead_mb:.1f}MB t={met.wall_time_sec:.1f}s", flush=True)
        append_universal_record(0, "M1", f"FullBackprop_{dom}", scale, 42, {"domain": d_res, "shared_retention": r_res, "seq2048_probe": seq2048_probe}, met, status="baseline")
        mgr.uninstall()
        cleanup_cuda()

    out["runs"]["FullBackprop_Baseline_s42"] = fb_domains
    del model
    cleanup_cuda()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=("small", "large", "both"), default="both")
    parser.add_argument("--only-fb", action="store_true")
    args = parser.parse_args()

    summary_path = LAB_DIR / "iter_0_universal_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    scales = ["small", "large"] if args.scale == "both" else [args.scale]
    for sc in scales:
        summary[sc] = run_scale_baselines(sc, only_fb=args.only_fb)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSaved iter_0_universal_summary.json", flush=True)


if __name__ == "__main__":
    main()
