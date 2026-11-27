"""1-Layer Detached Shared-4-Space Null-Preconditioned ResSkip (L26..L31) on 8-bit Ornith-1.0-9B.

Strictly O(1)-depth activation VRAM (x_in.detach(), max 1 detached hybrid block in autograd graph):
- Adapts Shared4Null to hybrid linear_attention (GatedDeltaNet) + full_attention layers.
- Uses exact eigh(4096) for attn_in / mixer_out / mlp_in and memory-safe top-512 low-rank
  spectral whitening (svd_lowrank) for 12288-dim down_proj (avoiding 2.3 GB eigh OOM).
- Wraps frozen 8-bit Linear8bitLt weights with fp32 rank-16 PrecondBlockLoRA adapters.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark import FactBatch, generate_fact_batch, load_smoltalk_splits
from eval_harness import append_experiment_record, check_gates, evaluate_model, get_base_reference
from guards import LoRAManager, NoBackpropGuard, cleanup_cuda, measure_static_vram_mb
from model_loader import (
    get_eos_token_ids,
    get_layers,
    get_lm_head,
    get_lm_norm,
    get_text_model,
    load_ornith_8bit,
    prepare_layer_inputs,
    slice_cm_map,
    slice_pe,
    step_layer,
)
from nullspace_cache import compute_or_load_c0, symmetrize_inplace
from probes import (
    build_probes_iter15,
    gather_positions,
    print_locality_breakdown,
    tokenize_with_target_positions,
)

LAB_DIR = Path("/content/ornith_nobackprop_lab")
TARGET_LAYERS = list(range(26, 32))
TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "in_proj_qkv", "in_proj_z", "out_proj",
    "gate_proj", "up_proj", "down_proj",
)
LORA_RANK = 16


def get_layer_spaces_and_modules(layer, layer_type: str):
    if layer_type == "full_attention":
        rep_mod_for_space = {
            "attn_in": layer.self_attn.q_proj,
            "mixer_out": layer.self_attn.o_proj,
            "mlp_in": layer.mlp.gate_proj,
            "down_proj": layer.mlp.down_proj,
        }
        space_for_mod = {
            "q_proj": "attn_in",
            "k_proj": "attn_in",
            "v_proj": "attn_in",
            "o_proj": "mixer_out",
            "gate_proj": "mlp_in",
            "up_proj": "mlp_in",
            "down_proj": "down_proj",
        }
        active_mods = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    else:
        rep_mod_for_space = {
            "attn_in": layer.linear_attn.in_proj_qkv,
            "mixer_out": layer.linear_attn.out_proj,
            "mlp_in": layer.mlp.gate_proj,
            "down_proj": layer.mlp.down_proj,
        }
        space_for_mod = {
            "in_proj_qkv": "attn_in",
            "in_proj_z": "attn_in",
            "out_proj": "mixer_out",
            "gate_proj": "mlp_in",
            "up_proj": "mlp_in",
            "down_proj": "down_proj",
        }
        active_mods = ("in_proj_qkv", "in_proj_z", "out_proj", "gate_proj", "up_proj", "down_proj")
    return rep_mod_for_space, space_for_mod, active_mods


class PrecondBlockLoRA(nn.Module):
    """Rank-r <= 16 LoRA wrapping 8-bit/fp16 linear with dense or low-rank P_null preconditioning."""

    def __init__(self, base_linear: nn.Module, r: int = 16, lora_alpha: float = 32.0) -> None:
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        if getattr(self.base, "bias", None) is not None:
            self.base.bias.requires_grad_(False)
        self.r = r
        self.scaling = float(lora_alpha) / float(r)
        self.active = False
        device = base_linear.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, base_linear.in_features, device=device, dtype=torch.float32), requires_grad=False)
        self.lora_B = nn.Parameter(torch.zeros(base_linear.out_features, r, device=device, dtype=torch.float32), requires_grad=False)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.P_null: Optional[torch.Tensor] = None
        self.U_k: Optional[torch.Tensor] = None
        self.shrink_k: Optional[torch.Tensor] = None

    def _apply_pnull(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        if self.P_null is not None:
            return xf @ self.P_null
        if self.U_k is not None and self.shrink_k is not None:
            proj = (xf @ self.U_k) * self.shrink_k
            return xf - proj @ self.U_k.T
        return xf

    def set_precond_and_init(
        self,
        precond: Tuple[Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]],
        X_stu_pos: torch.Tensor,
    ) -> Tuple[float, float]:
        P_dense, lowrank = precond
        device = self.base.weight.device
        if P_dense is not None:
            self.P_null = P_dense.to(device=device, dtype=torch.float32)
            self.U_k = None
            self.shrink_k = None
            mean_diag = float(self.P_null.diag().mean().item())
        else:
            assert lowrank is not None
            U_k, shrink_k = lowrank
            self.P_null = None
            self.U_k = U_k.to(device=device, dtype=torch.float32)
            self.shrink_k = shrink_k.to(device=device, dtype=torch.float32)
            mean_diag = 1.0 - float(self.shrink_k.sum().item() / max(self.base.in_features, 1))

        with torch.no_grad():
            Xp = self._apply_pnull(X_stu_pos)
            _, _, Vh = torch.linalg.svd(Xp, full_matrices=False)
            r_eff = min(self.r, Vh.shape[0])
            init_A = torch.zeros_like(self.lora_A)
            init_A[:r_eff] = Vh[:r_eff]
            init_A = init_A * math.sqrt(self.base.in_features / max(r_eff, 1)) * 0.25
            self.lora_A.copy_(init_A)
            self.lora_B.zero_()
            stu_ratio = float(Xp.norm().item() / (X_stu_pos.float().norm().item() + 1e-8))
        return stu_ratio, mean_diag

    def merge_precond_and_freeze(self) -> None:
        with torch.no_grad():
            if self.P_null is not None:
                merged_A = self.lora_A @ self.P_null
                self.lora_A.copy_(merged_A)
                self.P_null = None
            elif self.U_k is not None and self.shrink_k is not None:
                proj = (self.lora_A @ self.U_k) * self.shrink_k.unsqueeze(0)
                merged_A = self.lora_A - proj @ self.U_k.T
                self.lora_A.copy_(merged_A)
                self.U_k = None
                self.shrink_k = None
            self.lora_A.requires_grad_(False)
            self.lora_B.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.active:
            xp = self._apply_pnull(x)
            low = F.linear(xp, self.lora_A)
            delta = F.linear(low, self.lora_B) * self.scaling
            out = out + delta.to(out.dtype)
        return out


class PrecondBlockManager:
    TARGET_MAP = LoRAManager.TARGET_MAP

    def __init__(self, model: nn.Module, target_layers: Iterable[int], r: int = 16) -> None:
        self.model = model
        self.target_layers = list(target_layers)
        self.r = r
        self.wrapped: Dict[Tuple[int, str], PrecondBlockLoRA] = {}
        for p in model.parameters():
            p.requires_grad_(False)
        lm = get_text_model(model)
        layers = get_layers(model)
        for l_idx in self.target_layers:
            layer = layers[l_idx]
            lt = lm.config.layer_types[l_idx]
            _, _, active_mods = get_layer_spaces_and_modules(layer, lt)
            for mod_name in active_mods:
                pa, la = self.TARGET_MAP[mod_name]
                parent = getattr(layer, pa)
                orig = getattr(parent, la)
                w = PrecondBlockLoRA(orig, r=r, lora_alpha=32.0)
                setattr(parent, la, w)
                self.wrapped[(l_idx, mod_name)] = w

    def get_layer_params(self, layer_idx: int) -> List[nn.Parameter]:
        return [
            p
            for (li, _), w in self.wrapped.items()
            if li == layer_idx
            for p in (w.lora_A, w.lora_B)
        ]

    def activate_for_layer(self, active_layer_idx: int) -> None:
        for (l_idx, _), w in self.wrapped.items():
            if l_idx < active_layer_idx:
                w.active = True
                w.lora_A.requires_grad_(False)
                w.lora_B.requires_grad_(False)
            elif l_idx == active_layer_idx:
                w.active = True
                w.lora_A.requires_grad_(True)
                w.lora_B.requires_grad_(True)
            else:
                w.active = False
                w.lora_A.requires_grad_(False)
                w.lora_B.requires_grad_(False)

    def freeze_all(self) -> None:
        for w in self.wrapped.values():
            w.lora_A.requires_grad_(False)
            w.lora_B.requires_grad_(False)

    def merge_and_freeze_layer(self, layer_idx: int) -> None:
        for (li, _), w in self.wrapped.items():
            if li == layer_idx:
                w.merge_precond_and_freeze()

    def uninstall(self) -> None:
        layers = get_layers(self.model)
        for (l_idx, mod_name), w in self.wrapped.items():
            pa, la = self.TARGET_MAP[mod_name]
            parent = getattr(layers[l_idx], pa)
            setattr(parent, la, w.base)
        self.wrapped.clear()

    def assert_single_layer_grad(self, active_layer_idx: Optional[int]) -> None:
        grad_layers = set()
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                parts = name.split(".")
                if "layers" in parts:
                    idx = parts.index("layers")
                    grad_layers.add(int(parts[idx + 1]))
                else:
                    raise RuntimeError(f"VIOLATION: Non-layer param {name} requires_grad=True!")
        if active_layer_idx is None:
            assert len(grad_layers) == 0, f"Expected 0 grad layers, got {grad_layers}"
        else:
            assert grad_layers == {active_layer_idx}, f"Expected only layer {active_layer_idx}, got {grad_layers}"


def compute_null_preconditioner(
    X_cal: torch.Tensor,
    X_neg: torch.Tensor,
    C0_cached: Optional[torch.Tensor],
    alpha_null: float = 6.5,
) -> Tuple[Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    """Compute P_null = (I + alpha_null * C_prot_norm)^{-1/2} in-place to prevent 12288x12288 OOM."""
    d_in = X_cal.shape[-1]
    device = X_cal.device
    Xc = X_cal.reshape(-1, d_in).float()
    Xn = X_neg.reshape(-1, d_in).float()

    n_c = max(Xc.shape[0], 1)
    n_n = max(Xn.shape[0], 1)
    scale_c = max(float((Xc * Xc).sum().item() / (n_c * max(d_in, 1))), 1e-8)
    scale_n = max(float((Xn * Xn).sum().item() / (n_n * max(d_in, 1))), 1e-8)

    if d_in > 4096:
        torch.cuda.empty_cache()

    if C0_cached is not None:
        C_prot = C0_cached.to(device=device, dtype=torch.float32)
        scale_0 = max(float(torch.trace(C_prot).item() / max(d_in, 1)), 1e-8)
        C_prot.mul_(0.35 / scale_0)
        C_prot.addmm_(Xc.T, Xc, alpha=(0.30 / (n_c * scale_c)))
        C_prot.addmm_(Xn.T, Xn, alpha=(0.35 / (n_n * scale_n)))
    else:
        C_prot = torch.zeros((d_in, d_in), device=device, dtype=torch.float32)
        C_prot.addmm_(Xc.T, Xc, alpha=(0.50 / (n_c * scale_c)))
        C_prot.addmm_(Xn.T, Xn, alpha=(0.50 / (n_n * scale_n)))

    if d_in <= 4096:
        C_prot = 0.5 * (C_prot + C_prot.T)
        mat = torch.eye(d_in, device=device, dtype=torch.float32) + alpha_null * C_prot
        del C_prot
        evals, evecs = torch.linalg.eigh(mat)
        evals = evals.clamp(min=1.0)
        P_null = (evecs / torch.sqrt(evals).unsqueeze(0)) @ evecs.T
        del mat, evals, evecs
        return P_null, None

    symmetrize_inplace(C_prot, chunk=1024)
    U_k, S_k, _ = torch.svd_lowrank(C_prot, q=512, niter=2)
    del C_prot
    torch.cuda.empty_cache()
    shrink_k = 1.0 - torch.rsqrt(1.0 + alpha_null * S_k.clamp(min=0.0))
    return None, (U_k, shrink_k)


def run_shared4_l26_31(
    model,
    tokenizer,
    mgr: PrecondBlockManager,
    covs: Dict,
    edit_texts: List[str],
    cal_texts: List[str],
    device: str,
    doc_text_stage2: str | None = None,
    probes_override: Optional[Tuple[List[Dict[str, str]], List[Dict[str, str]]]] = None,
    target_layers_override: Optional[List[int]] = None,
    steps_override: Optional[Tuple[int, ...]] = None,
    alpha_null: float = 6.5,
) -> Dict[str, object]:
    """Run 1-Layer Detached Shared-4-Space Null-Preconditioned ResSkip on 8-bit Ornith-1.0-9B."""
    active_layers = target_layers_override if target_layers_override is not None else TARGET_LAYERS
    steps_per_layer = steps_override if steps_override is not None else (8, 9, 9, 10, 10, 11)
    lr = 4.5e-3
    lambda_cal_kl = 5.5
    lambda_neg_kl = 8.0
    lambda_antibleed = 1.5

    pos_items, neg_items = (
        probes_override if probes_override is not None else build_probes_iter15(edit_texts, tokenizer)
    )
    prompts = [x["prompt"] for x in pos_items]
    targets = [x["target"] for x in pos_items]

    stu_ids, stu_mask, stu_pos = tokenize_with_target_positions(tokenizer, prompts, targets, device)
    target_tok_list: List[int] = []
    pos0_weights: List[float] = []
    injected_first_toks = set()

    for tgt, poss in zip(targets, stu_pos):
        t_ids = tokenizer.encode(tgt, add_special_tokens=False)
        eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        full_t = t_ids + eos
        injected_first_toks.add(t_ids[0])
        target_tok_list.extend(full_t)
        pos0_weights.extend([2.0] + [1.0] * (len(full_t) - 1))

    target_tok_ids = torch.tensor(target_tok_list, dtype=torch.long, device=device)
    tok_weights = torch.tensor(pos0_weights, dtype=torch.float32, device=device)
    tok_weights = tok_weights / tok_weights.mean()
    bleed_tok_ids = torch.tensor(sorted(injected_first_toks), dtype=torch.long, device=device)

    lm = get_text_model(model)
    layers = get_layers(model)
    n_layers = len(layers)
    lm_norm = get_lm_norm(model)
    lm_head = get_lm_head(model)
    eos_ids = get_eos_token_ids(tokenizer)
    prev_side = tokenizer.padding_side

    with torch.no_grad():
        mgr.freeze_all()
        mgr.assert_single_layer_grad(None)

        if doc_text_stage2 is not None:
            doc_enc = tokenizer(doc_text_stage2, return_tensors="pt", truncation=True, max_length=2048).to(device)
            _ = model(input_ids=doc_enc.input_ids, attention_mask=doc_enc.attention_mask, use_cache=False)
            del doc_enc

        tokenizer.padding_side = "left"
        neg_gen_enc = tokenizer([x["prompt"] for x in neg_items], return_tensors="pt", padding=True).to(device)
        in_len_neg = neg_gen_enc.input_ids.shape[1]
        neg_gen_rows = []
        for s in range(0, neg_gen_enc.input_ids.shape[0], 16):
            e = min(neg_gen_enc.input_ids.shape[0], s + 16)
            g = model.generate(
                input_ids=neg_gen_enc.input_ids[s:e],
                attention_mask=neg_gen_enc.attention_mask[s:e],
                max_new_tokens=2,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_ids,
                use_cache=True,
            )
            neg_gen_rows.append(g[:, in_len_neg:].cpu())
        neg_gen = torch.cat(neg_gen_rows, dim=0)
        del neg_gen_enc, neg_gen_rows

        neg_seqs, neg_pos, neg_tok_weights_list = [], [], []
        for idx_n in range(len(neg_items)):
            gen_part = neg_gen[idx_n].tolist()
            clean_gen = []
            for tk in gen_part:
                clean_gen.append(tk)
                if tk in eos_ids:
                    break
            p_ids = tokenizer.encode(neg_items[idx_n]["prompt"], add_special_tokens=False)
            full_seq = p_ids + clean_gen
            neg_seqs.append(full_seq)
            n_pred = max(1, len(clean_gen))
            neg_pos.append(list(range(len(p_ids) - 1, len(p_ids) - 1 + n_pred)))
            neg_tok_weights_list.extend([3.0] + [1.0] * (n_pred - 1))

        neg_tok_weights = torch.tensor(neg_tok_weights_list, dtype=torch.float32, device=device)
        neg_tok_weights = neg_tok_weights / neg_tok_weights.mean()

        max_neg_len = max(len(s) for s in neg_seqs)
        neg_ids = torch.full((len(neg_seqs), max_neg_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
        neg_mask = torch.zeros((len(neg_seqs), max_neg_len), dtype=torch.long, device=device)
        for i, s in enumerate(neg_seqs):
            neg_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            neg_mask[i, : len(s)] = 1

        tokenizer.padding_side = "right"
        cal_enc = tokenizer(cal_texts[:48], return_tensors="pt", padding=True, truncation=True, max_length=72).to(device)
        tokenizer.padding_side = prev_side

        cal_lengths = cal_enc.attention_mask.sum(dim=1).tolist()
        cal_kl_pos = [[max(0, min(L - 1, int(k * L / 4))) for k in range(1, 4)] for L in cal_lengths]

        def get_base_probs(ids, mask, pos_lists, bs: int = 24):
            probs_list, logp_bleed_list = [], []
            n_tot = ids.shape[0]
            for s in range(0, n_tot, bs):
                e = min(n_tot, s + bs)
                h, cm_m, pe_t, tp_t = prepare_layer_inputs(model, ids[s:e], mask[s:e])
                for l_i in range(n_layers):
                    h = step_layer(model, l_i, h, cm_m, pe_t, tp_t)
                pts = gather_positions(h, pos_lists[s:e])
                logits = lm_head(lm_norm(pts)).float()
                lp = F.log_softmax(logits, dim=-1)
                probs_list.append(lp.exp().cpu())
                logp_bleed_list.append(lp[:, bleed_tok_ids].cpu())
                del h, pts, logits, lp
            return torch.cat(probs_list, dim=0), torch.cat(logp_bleed_list, dim=0)

        p_base_cal_cpu, _ = get_base_probs(cal_enc.input_ids, cal_enc.attention_mask, cal_kl_pos)
        p_base_neg_cpu, logp_base_neg_bleed_cpu = get_base_probs(neg_ids, neg_mask, neg_pos)

        def init_and_prefix(ids, mask, bs: int = 24):
            emb_full, cm_full, pe_full, tp_full = prepare_layer_inputs(model, ids, mask)
            h_chunks = []
            n_tot = ids.shape[0]
            for s in range(0, n_tot, bs):
                e = min(n_tot, s + bs)
                h = emb_full[s:e]
                cm_s = slice_cm_map(cm_full, s, e, n_tot)
                pe_s = slice_pe(pe_full, s, e, n_tot)
                tp_s = tp_full[s:e]
                for l_idx in range(active_layers[0]):
                    h = step_layer(model, l_idx, h, cm_s, pe_s, tp_s)
                h_chunks.append(h.detach())
            del emb_full
            return torch.cat(h_chunks, dim=0), cm_full, pe_full, tp_full

        H_stu, cm_stu, pe_stu, tp_stu = init_and_prefix(stu_ids, stu_mask)
        H_neg, cm_neg, pe_neg, tp_neg = init_and_prefix(neg_ids, neg_mask)
        H_cal, cm_cal, pe_cal, tp_cal = init_and_prefix(cal_enc.input_ids, cal_enc.attention_mask)
        cal_mask_bool = cal_enc.attention_mask.bool()
        del stu_ids, stu_mask, neg_ids, neg_mask, cal_enc

    # Keep large base vocab targets on CPU and move only active minibatch slices to GPU
    layer_logs: List[Dict[str, object]] = []

    for idx_in_sched, l_idx in enumerate(active_layers):
        layer = layers[l_idx]
        lt = lm.config.layer_types[l_idx]
        rep_mods, space_for_mod, active_mods = get_layer_spaces_and_modules(layer, lt)
        n_steps = steps_per_layer[idx_in_sched]

        H_stu_in = H_stu.detach()
        H_neg_in = H_neg.detach()
        H_cal_in = H_cal.detach()

        with torch.no_grad():
            mgr.freeze_all()
            mgr.assert_single_layer_grad(None)

            rep_spaces = list(rep_mods.keys())
            cap_cal = {sp: [] for sp in rep_spaces}
            cap_neg = {sp: [] for sp in rep_spaces}
            cap_stu = {sp: [] for sp in rep_spaces}

            def register_space_hooks(store, pos_lists=None, bs_mask=None):
                hooks = []
                for sp, mod in rep_mods.items():
                    if pos_lists is not None:
                        def hk(m, inp, sname=sp, pl=pos_lists):
                            store[sname].append(gather_positions(inp[0].detach(), pl))
                    elif bs_mask is not None:
                        def hk(m, inp, sname=sp, bm=bs_mask):
                            store[sname].append(inp[0].detach()[bm])
                    else:
                        def hk(m, inp, sname=sp):
                            store[sname].append(inp[0].detach())
                    hooks.append(mod.register_forward_pre_hook(hk))
                return hooks

            n_cal = H_cal_in.shape[0]
            for s in range(0, n_cal, 24):
                e = min(n_cal, s + 24)
                h_c = register_space_hooks(cap_cal, bs_mask=cal_mask_bool[s:e])
                _ = step_layer(model, l_idx, H_cal_in[s:e], slice_cm_map(cm_cal, s, e, n_cal), slice_pe(pe_cal, s, e, n_cal), tp_cal[s:e])
                for h in h_c:
                    h.remove()

            n_neg_tot = H_neg_in.shape[0]
            for s in range(0, n_neg_tot, 25):
                e = min(n_neg_tot, s + 25)
                h_n = register_space_hooks(cap_neg, pos_lists=neg_pos[s:e])
                _ = step_layer(model, l_idx, H_neg_in[s:e], slice_cm_map(cm_neg, s, e, n_neg_tot), slice_pe(pe_neg, s, e, n_neg_tot), tp_neg[s:e])
                for h in h_n:
                    h.remove()

            n_stu_tot = H_stu_in.shape[0]
            for s in range(0, n_stu_tot, 25):
                e = min(n_stu_tot, s + 25)
                h_s = register_space_hooks(cap_stu, pos_lists=stu_pos[s:e])
                _ = step_layer(model, l_idx, H_stu_in[s:e], slice_cm_map(cm_stu, s, e, n_stu_tot), slice_pe(pe_stu, s, e, n_stu_tot), tp_stu[s:e])
                for h in h_s:
                    h.remove()

            p_null_by_space, xs_by_space = {}, {}
            for sp in rep_spaces:
                xc = torch.cat(cap_cal[sp], dim=0)
                xn = torch.cat(cap_neg[sp], dim=0)
                xs = torch.cat(cap_stu[sp], dim=0)
                c0_key = (l_idx, "mixer_out_proj" if sp == "mixer_out" else sp)
                c0_mat = covs.get(c0_key) if sp in ("mixer_out", "down_proj") else None
                p_null_by_space[sp] = compute_null_preconditioner(xc, xn, c0_mat, alpha_null=alpha_null)
                xs_by_space[sp] = xs
                del xc, xn

            for m_name in active_mods:
                sp = space_for_mod[m_name]
                mgr.wrapped[(l_idx, m_name)].set_precond_and_init(p_null_by_space[sp], xs_by_space[sp])
            del cap_cal, cap_neg, cap_stu, p_null_by_space, xs_by_space

            def compute_skip(h_in, cm_m, pe_t, tp_t, pos_lists, bs: int = 25):
                skips = []
                n_tot = h_in.shape[0]
                for s in range(0, n_tot, bs):
                    e = min(n_tot, s + bs)
                    cm_s = slice_cm_map(cm_m, s, e, n_tot)
                    pe_s = slice_pe(pe_t, s, e, n_tot)
                    tp_s = tp_t[s:e]
                    hl = step_layer(model, l_idx, h_in[s:e], cm_s, pe_s, tp_s)
                    hc = hl
                    for j in range(l_idx + 1, n_layers):
                        hc = step_layer(model, j, hc, cm_s, pe_s, tp_s)
                    skips.append(gather_positions(hc - hl, pos_lists[s:e]).detach())
                return torch.cat(skips, dim=0)

            skip_stu = compute_skip(H_stu_in, cm_stu, pe_stu, tp_stu, stu_pos)
            skip_neg = compute_skip(H_neg_in, cm_neg, pe_neg, tp_neg, neg_pos)
            skip_cal = compute_skip(H_cal_in, cm_cal, pe_cal, tp_cal, cal_kl_pos)

        mgr.activate_for_layer(l_idx)
        mgr.assert_single_layer_grad(l_idx)
        params = mgr.get_layer_params(l_idx)
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=5e-4)

        n_stu = H_stu_in.shape[0]
        stu_bs = 25
        last_ce, last_ckl, last_nkl = 0.0, 0.0, 0.0

        for step in range(n_steps):
            opt.zero_grad(set_to_none=True)
            with torch.enable_grad():
                ce_val = 0.0
                tok_off = 0
                for s in range(0, n_stu, stu_bs):
                    e = min(n_stu, s + stu_bs)
                    out_s = step_layer(
                        model, l_idx, H_stu_in[s:e],
                        slice_cm_map(cm_stu, s, e, n_stu), slice_pe(pe_stu, s, e, n_stu), tp_stu[s:e]
                    )
                    pts_s = gather_positions(out_s, stu_pos[s:e])
                    cnt = pts_s.shape[0]
                    h_final_s = pts_s + skip_stu[tok_off : tok_off + cnt]
                    logits_s = lm_head(lm_norm(h_final_s)).float()
                    ce_vec = F.cross_entropy(logits_s, target_tok_ids[tok_off : tok_off + cnt], reduction="none")
                    l_ce = (ce_vec * tok_weights[tok_off : tok_off + cnt]).sum() / target_tok_ids.numel()
                    l_ce.backward()
                    ce_val += float(l_ce.item())
                    tok_off += cnt
                    del out_s, pts_s, h_final_s, logits_s, ce_vec, l_ce

                cal_half = (step % 2) * (n_cal // 2)
                cs, ce_c = cal_half, min(n_cal, cal_half + (n_cal // 2))
                c_start = sum(len(p) for p in cal_kl_pos[:cs])
                c_cnt = sum(len(p) for p in cal_kl_pos[cs:ce_c])
                out_c = step_layer(
                    model, l_idx, H_cal_in[cs:ce_c],
                    slice_cm_map(cm_cal, cs, ce_c, n_cal), slice_pe(pe_cal, cs, ce_c, n_cal), tp_cal[cs:ce_c]
                )
                pts_c = gather_positions(out_c, cal_kl_pos[cs:ce_c]) + skip_cal[c_start : c_start + c_cnt]
                log_p_c = F.log_softmax(lm_head(lm_norm(pts_c)).float(), dim=-1)
                pb_c = p_base_cal_cpu[c_start : c_start + c_cnt].to(device)
                l_ckl = F.kl_div(log_p_c, pb_c, reduction="batchmean")
                (lambda_cal_kl * l_ckl).backward()
                last_ckl = float(l_ckl.item())
                del out_c, pts_c, log_p_c, pb_c, l_ckl

                nkl_val = 0.0
                n_neg = H_neg_in.shape[0]
                neg_off = 0
                for s in range(0, n_neg, 25):
                    e = min(n_neg, s + 25)
                    out_n = step_layer(
                        model, l_idx, H_neg_in[s:e],
                        slice_cm_map(cm_neg, s, e, n_neg), slice_pe(pe_neg, s, e, n_neg), tp_neg[s:e]
                    )
                    pts_n = gather_positions(out_n, neg_pos[s:e])
                    ncnt = pts_n.shape[0]
                    pts_n = pts_n + skip_neg[neg_off : neg_off + ncnt]
                    log_p_n = F.log_softmax(lm_head(lm_norm(pts_n)).float(), dim=-1)
                    pb = p_base_neg_cpu[neg_off : neg_off + ncnt].to(device)
                    lpb_b = logp_base_neg_bleed_cpu[neg_off : neg_off + ncnt].to(device)
                    nw = neg_tok_weights[neg_off : neg_off + ncnt]
                    tok_nkl = (pb * (pb.clamp(min=1e-12).log() - log_p_n)).sum(dim=-1)
                    l_nkl = (tok_nkl * nw).mean() * (ncnt / p_base_neg_cpu.shape[0])

                    excess_logp = F.relu(log_p_n[:, bleed_tok_ids] - lpb_b - 0.05)
                    l_bleed = (excess_logp ** 2).mean() * (ncnt / p_base_neg_cpu.shape[0])

                    loss_n_chunk = lambda_neg_kl * l_nkl + lambda_antibleed * l_bleed
                    loss_n_chunk.backward()
                    nkl_val += float(l_nkl.item())
                    neg_off += ncnt
                    del out_n, pts_n, log_p_n, pb, lpb_b, tok_nkl, l_nkl, excess_logp, l_bleed, loss_n_chunk

            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            last_ce = ce_val
            last_nkl = nkl_val

        del opt, skip_stu, skip_neg, skip_cal
        mgr.merge_and_freeze_layer(l_idx)
        cleanup_cuda()

        with torch.no_grad():
            mgr.assert_single_layer_grad(None)

            def step_chunked(h_in, cm_m, pe_t, tp_t, bs: int = 25):
                outs = []
                n_tot = h_in.shape[0]
                for s in range(0, n_tot, bs):
                    e = min(n_tot, s + bs)
                    outs.append(
                        step_layer(
                            model, l_idx, h_in[s:e],
                            slice_cm_map(cm_m, s, e, n_tot), slice_pe(pe_t, s, e, n_tot), tp_t[s:e]
                        ).detach()
                    )
                return torch.cat(outs, dim=0)

            H_stu = step_chunked(H_stu_in, cm_stu, pe_stu, tp_stu)
            H_neg = step_chunked(H_neg_in, cm_neg, pe_neg, tp_neg)
            H_cal = step_chunked(H_cal_in, cm_cal, pe_cal, tp_cal)
            del H_stu_in, H_neg_in, H_cal_in

        layer_logs.append({
            "layer": l_idx,
            "layer_type": lt,
            "ce": round(last_ce, 4),
            "cal_kl": round(last_ckl, 5),
            "neg_kl": round(last_nkl, 5),
        })
        print(f"  [Shared4Null L{l_idx:02d} ({lt[:4]})] ce={last_ce:.4f}, cal_kl={last_ckl:.5f}, neg_kl={last_nkl:.5f}", flush=True)

    return {"layer_logs": layer_logs}


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_ornith_8bit(device=device, dtype=torch.float16, drop_visual=True)
    static_vram_mb = measure_static_vram_mb(model)

    cal_texts, kl_texts = load_smoltalk_splits(tokenizer, n_cal=128, n_kl_eval=64)
    batch42 = generate_fact_batch(42, tokenizer=tokenizer, target_doc_tokens=2048)

    covs, _ = compute_or_load_c0(
        model, tokenizer, device, target_layers=TARGET_LAYERS, target_modules=("mixer_out_proj", "down_proj"), n_cal=128
    )

    mgr = PrecondBlockManager(model, target_layers=TARGET_LAYERS, r=LORA_RANK)
    baselines = json.loads((LAB_DIR / "baselines_summary.json").read_text())
    base_a_ref = baselines["baseline_a_512"]
    base_ref = get_base_reference(model, tokenizer, batch42, device, lora_manager=mgr)

    with NoBackpropGuard(model, regime="single_layer_isolated", static_vram_mb=static_vram_mb) as guard:
        guard.assert_backbone_no_grad()
        diag = run_shared4_l26_31(model, tokenizer, mgr, covs, batch42.stage1_edit_texts, cal_texts, device)

    eval_res = evaluate_model(model, tokenizer, batch42, kl_texts, device, lora_manager=mgr, compute_kl=True, base_ref_override=base_ref)
    gates = check_gates(eval_res, guard.metrics, baseline_a_ref=base_a_ref)
    status = "pass" if gates["ALL_GATES_PASS"] else "iterate"

    rec = append_experiment_record(
        12,
        "Shared4Null_ResSkip_L26_31",
        eval_res,
        guard.metrics,
        status=status,
        seed=42,
        stage=1,
        rank=LORA_RANK,
        extra={"gates": gates, "layer_logs": diag["layer_logs"]},
    )
    print("\n[Shared4Null / Iter 12] Result:", json.dumps(rec, indent=2))
    print("[Shared4Null / Iter 12] Gates:", json.dumps(gates, indent=2))
    print_locality_breakdown(model, tokenizer, mgr, batch42, device, base_ref=base_ref)


if __name__ == "__main__":
    main()
