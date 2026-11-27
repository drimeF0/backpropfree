"""Iteration 12: Null-Space Preconditioned + Fisher-SVD Initialized Single-Block ResSkip (L26..L31).

Combines Regime (a/b) Null-Space Projection with Regime (c) Single-Block Detached Refinement:
1. For each block l in L26..L31, capture module inputs X_cal, X_neg, X_stu under torch.no_grad().
2. Construct protection covariance C_prot = C_cal_norm + C_neg_norm (+ cached C0 for o_proj/down_proj).
3. Compute Tikhonov null-space preconditioner P_null = (I + alpha_null * C_prot)^{-1/2}.
4. Factor LoRA as delta = B @ (A_free @ P_null) @ x, initializing A_free from the top-16 right
   singular vectors of (X_stu @ P_null) so Step 0 already targets entity-discriminative directions!
5. Optimize (A_free, B) locally inside the single detached block with CE + cal_kl + neg_kl,
   then merge A = A_free @ P_null into a standard rank-16 LoRA.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask

from benchmark import FactBatch, generate_fact_batch, load_smoltalk_splits
from eval_harness import append_experiment_record, check_gates, evaluate_model
from guards import LoRAManager, NoBackpropGuard, cleanup_cuda, measure_static_vram_mb
from iter_1_experiment import gather_positions, tokenize_with_target_positions
from iter_11_experiment import build_probes_iter11
from nullspace_cache import compute_or_load_c0

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
LAB_DIR = Path("/content/smollm_nobackprop_lab")
TARGET_LAYERS = list(range(26, 32))
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LORA_RANK = 16


class PrecondBlockLoRA(nn.Module):
    """Rank-r <= 16 LoRA with optional fixed null-space preconditioner P_null during synthesis."""

    def __init__(self, base_linear: nn.Linear, r: int = 16, lora_alpha: float = 32.0) -> None:
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.r = r
        self.scaling = float(lora_alpha) / float(r)
        self.active = False
        device = base_linear.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, base_linear.in_features, device=device, dtype=torch.float32), requires_grad=False)
        self.lora_B = nn.Parameter(torch.zeros(base_linear.out_features, r, device=device, dtype=torch.float32), requires_grad=False)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.P_null: Optional[torch.Tensor] = None  # [d_in, d_in] half on GPU during this layer's opt only

    def set_precond_and_init(self, P_null: torch.Tensor, X_stu_pos: torch.Tensor) -> Tuple[float, float]:
        """Set P_null [d_in, d_in] and Fisher-SVD initialize lora_A from X_stu_pos @ P_null."""
        self.P_null = P_null.to(device=self.base.weight.device, dtype=self.base.weight.dtype)
        with torch.no_grad():
            Xp = X_stu_pos.float() @ P_null.float()
            # Top-r right singular vectors of preconditioned edit activations
            # Note: Xp is [M, d_in]; if M >= r, SVD gives exact principal entity directions!
            _, _, Vh = torch.linalg.svd(Xp, full_matrices=False)
            r_eff = min(self.r, Vh.shape[0])
            init_A = torch.zeros_like(self.lora_A)
            init_A[:r_eff] = Vh[:r_eff]
            # Scale initial A to standard Kaiming std ~ 1/sqrt(d_in)
            init_A = init_A * math.sqrt(self.base.in_features / max(r_eff, 1)) * 0.25
            self.lora_A.copy_(init_A)
            self.lora_B.zero_()
            stu_ratio = float(Xp.norm().item() / (X_stu_pos.float().norm().item() + 1e-8))
        return stu_ratio, float(P_null.float().diag().mean().item())

    def merge_precond_and_freeze(self) -> None:
        with torch.no_grad():
            if self.P_null is not None:
                merged_A = self.lora_A @ self.P_null.float()
                self.lora_A.copy_(merged_A)
                self.P_null = None
            self.lora_A.requires_grad_(False)
            self.lora_B.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.active:
            xp = F.linear(x, self.P_null) if self.P_null is not None else x
            low = F.linear(xp, self.lora_A.to(x.dtype))
            delta = F.linear(low, self.lora_B.to(x.dtype)) * self.scaling
            out = out + delta
        return out


class PrecondBlockManager:
    TARGET_MAP = LoRAManager.TARGET_MAP

    def __init__(self, model: nn.Module, target_layers: Iterable[int], target_modules: Iterable[str], r: int = 16) -> None:
        self.model = model
        self.target_layers = list(target_layers)
        self.target_modules = tuple(target_modules)
        self.r = r
        self.wrapped: Dict[Tuple[int, str], PrecondBlockLoRA] = {}
        for p in model.parameters():
            p.requires_grad_(False)
        for l_idx in self.target_layers:
            layer = model.model.layers[l_idx]
            for mod_name in self.target_modules:
                pa, la = self.TARGET_MAP[mod_name]
                parent = getattr(layer, pa)
                orig = getattr(parent, la)
                w = PrecondBlockLoRA(orig, r=r, lora_alpha=32.0)
                setattr(parent, la, w)
                self.wrapped[(l_idx, mod_name)] = w

    def get_layer_params(self, layer_idx: int) -> List[nn.Parameter]:
        return [
            p
            for mod_name in self.target_modules
            for p in (self.wrapped[(layer_idx, mod_name)].lora_A, self.wrapped[(layer_idx, mod_name)].lora_B)
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
        for mod_name in self.target_modules:
            self.wrapped[(layer_idx, mod_name)].merge_precond_and_freeze()

    def assert_single_layer_grad(self, active_layer_idx: Optional[int]) -> None:
        grad_layers = set()
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                parts = name.split(".")
                if len(parts) > 2 and parts[0] == "model" and parts[1] == "layers":
                    grad_layers.add(int(parts[2]))
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
    alpha_null: float = 6.0,
) -> torch.Tensor:
    """Compute P_null = (I + alpha_null * C_prot_norm)^{-1/2} in float32."""
    d_in = X_cal.shape[-1]
    device = X_cal.device
    Xc = X_cal.reshape(-1, d_in).float()
    Xn = X_neg.reshape(-1, d_in).float()

    Cc = (Xc.T @ Xc) / max(Xc.shape[0], 1)
    Cc = Cc / max(float(torch.trace(Cc).item() / d_in), 1e-8)

    Cn = (Xn.T @ Xn) / max(Xn.shape[0], 1)
    Cn = Cn / max(float(torch.trace(Cn).item() / d_in), 1e-8)

    C_prot = 0.5 * Cc + 0.5 * Cn
    if C0_cached is not None:
        C0f = C0_cached.to(device=device, dtype=torch.float32)
        C0f = C0f / max(float(torch.trace(C0f).item() / d_in), 1e-8)
        C_prot = 0.35 * C0f + 0.30 * Cc + 0.35 * Cn

    mat = torch.eye(d_in, device=device, dtype=torch.float32) + alpha_null * (0.5 * (C_prot + C_prot.T))
    evals, evecs = torch.linalg.eigh(mat)
    evals = evals.clamp(min=1.0)
    P_null = (evecs / torch.sqrt(evals).unsqueeze(0)) @ evecs.T
    return P_null


def synthesize_null_precond_resskip(
    model,
    tokenizer,
    mgr: PrecondBlockManager,
    covs: Dict[Tuple[int, str], torch.Tensor],
    batch: FactBatch,
    cal_texts: List[str],
    device: str,
    steps_per_layer: Tuple[int, ...] = (12, 12, 14, 14, 16, 18),
    lr: float = 4.5e-3,
    alpha_null: float = 6.0,
    lambda_cal_kl: float = 5.0,
    lambda_neg_kl: float = 7.0,
    lambda_antibleed: float = 1.0,
) -> Dict[str, object]:
    pos_items, neg_items = build_probes_iter11(batch.stage1_edit_texts, tokenizer)
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
        assert len(full_t) == len(poss)
        injected_first_toks.add(t_ids[0])
        target_tok_list.extend(full_t)
        pos0_weights.extend([2.0] + [1.0] * (len(full_t) - 1))

    target_tok_ids = torch.tensor(target_tok_list, dtype=torch.long, device=device)
    tok_weights = torch.tensor(pos0_weights, dtype=torch.float32, device=device)
    tok_weights = tok_weights / tok_weights.mean()
    bleed_tok_ids = torch.tensor(sorted(injected_first_toks), dtype=torch.long, device=device)

    prev_side = tokenizer.padding_side
    with torch.no_grad():
        mgr.freeze_all()
        mgr.assert_single_layer_grad(None)

        tokenizer.padding_side = "left"
        neg_gen_enc = tokenizer([x["prompt"] for x in neg_items], return_tensors="pt", padding=True).to(device)
        in_len_neg = neg_gen_enc.input_ids.shape[1]
        neg_gen = model.generate(
            input_ids=neg_gen_enc.input_ids,
            attention_mask=neg_gen_enc.attention_mask,
            max_new_tokens=2,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        neg_seqs, neg_pos, neg_tok_weights_list = [], [], []
        for idx_n in range(len(neg_items)):
            gen_part = neg_gen[idx_n, in_len_neg:].tolist()
            clean_gen = []
            for tk in gen_part:
                clean_gen.append(tk)
                if tk == tokenizer.eos_token_id:
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
        cal_enc = tokenizer(cal_texts[:64], return_tensors="pt", padding=True, truncation=True, max_length=72).to(device)
        tokenizer.padding_side = prev_side

        cal_lengths = cal_enc.attention_mask.sum(dim=1).tolist()
        cal_kl_pos = [[max(0, min(L - 1, int(k * L / 4))) for k in range(1, 4)] for L in cal_lengths]

        def get_base_probs(ids, mask, pos_lists, bs: int = 64):
            probs_list, logp_bleed_list = [], []
            for s in range(0, ids.shape[0], bs):
                e = min(ids.shape[0], s + bs)
                emb = model.model.embed_tokens(ids[s:e])
                pids = torch.arange(emb.shape[1], device=device).unsqueeze(0)
                cm = create_causal_mask(config=model.config, inputs_embeds=emb, attention_mask=mask[s:e], past_key_values=None, position_ids=pids)
                pe = model.model.rotary_emb(emb, position_ids=pids)
                h = emb
                for layer in model.model.layers:
                    h = layer(h, attention_mask=cm, position_embeddings=pe, use_cache=False)
                pts = gather_positions(h, pos_lists[s:e])
                logits = model.lm_head(model.model.norm(pts)).float()
                lp = F.log_softmax(logits, dim=-1)
                probs_list.append(lp.exp().cpu())
                logp_bleed_list.append(lp[:, bleed_tok_ids].cpu())
            return torch.cat(probs_list, dim=0), torch.cat(logp_bleed_list, dim=0)

        p_base_cal_cpu, _ = get_base_probs(cal_enc.input_ids, cal_enc.attention_mask, cal_kl_pos)
        p_base_neg_cpu, logp_base_neg_bleed_cpu = get_base_probs(neg_ids, neg_mask, neg_pos)

        def init_and_prefix(ids, mask, bs: int = 64):
            h_chunks = []
            emb_full = model.model.embed_tokens(ids)
            pids = torch.arange(emb_full.shape[1], device=device).unsqueeze(0)
            cm_full = create_causal_mask(config=model.config, inputs_embeds=emb_full, attention_mask=mask, past_key_values=None, position_ids=pids)
            pe = model.model.rotary_emb(emb_full, position_ids=pids)
            for s in range(0, ids.shape[0], bs):
                e = min(ids.shape[0], s + bs)
                h = emb_full[s:e]
                cm = cm_full[s:e] if cm_full is not None and cm_full.shape[0] == ids.shape[0] else cm_full
                for l_idx in range(TARGET_LAYERS[0]):
                    h = model.model.layers[l_idx](h, attention_mask=cm, position_embeddings=pe, use_cache=False)
                h_chunks.append(h.detach())
            return torch.cat(h_chunks, dim=0), cm_full, pe

        H_stu, cm_stu, pe_stu = init_and_prefix(stu_ids, stu_mask)
        H_neg, cm_neg, pe_neg = init_and_prefix(neg_ids, neg_mask)
        H_cal, cm_cal, pe_cal = init_and_prefix(cal_enc.input_ids, cal_enc.attention_mask)

    p_base_cal = p_base_cal_cpu.to(device)
    p_base_neg = p_base_neg_cpu.to(device)
    logp_base_neg_bleed = logp_base_neg_bleed_cpu.to(device)

    layer_logs: List[Dict[str, object]] = []
    n_layers = len(model.model.layers)

    for idx_in_sched, l_idx in enumerate(TARGET_LAYERS):
        layer = model.model.layers[l_idx]
        n_steps = steps_per_layer[idx_in_sched]
        H_stu_in = H_stu.detach()
        H_neg_in = H_neg.detach()
        H_cal_in = H_cal.detach()

        with torch.no_grad():
            mgr.freeze_all()
            mgr.assert_single_layer_grad(None)

            # Capture module inputs for cal, neg, and stu under no_grad() to build P_null & Fisher-SVD init
            cap_cal = {m: [] for m in TARGET_MODULES}
            cap_neg = {m: [] for m in TARGET_MODULES}
            cap_stu = {m: [] for m in TARGET_MODULES}

            def register_cap_hooks(store: Dict[str, List[torch.Tensor]], pos_lists=None, bs_mask=None):
                hooks = []
                for m_name in TARGET_MODULES:
                    pa, la = mgr.TARGET_MAP[m_name]
                    mod = getattr(getattr(layer, pa), la)
                    if pos_lists is not None:
                        def hk(m, inp, mn=m_name, pl=pos_lists):
                            store[mn].append(gather_positions(inp[0].detach(), pl))
                    elif bs_mask is not None:
                        def hk(m, inp, mn=m_name, bm=bs_mask):
                            store[mn].append(inp[0].detach()[bm])
                    else:
                        def hk(m, inp, mn=m_name):
                            store[mn].append(inp[0].detach())
                    hooks.append(mod.register_forward_pre_hook(hk))
                return hooks

            h_c = register_cap_hooks(cap_cal, bs_mask=cal_enc.attention_mask.bool())
            _ = layer(H_cal_in, attention_mask=cm_cal, position_embeddings=pe_cal, use_cache=False)
            for h in h_c:
                h.remove()

            h_n = register_cap_hooks(cap_neg, pos_lists=neg_pos)
            _ = layer(H_neg_in, attention_mask=cm_neg, position_embeddings=pe_neg, use_cache=False)
            for h in h_n:
                h.remove()

            # Capture stu in 2 chunks
            for s in range(0, H_stu_in.shape[0], 75):
                e = min(H_stu_in.shape[0], s + 75)
                cm_s = cm_stu[s:e] if cm_stu is not None and cm_stu.shape[0] == H_stu_in.shape[0] else cm_stu
                h_s = register_cap_hooks(cap_stu, pos_lists=stu_pos[s:e])
                _ = layer(H_stu_in[s:e], attention_mask=cm_s, position_embeddings=pe_stu, use_cache=False)
                for h in h_s:
                    h.remove()

            stu_ratios = []
            for m_name in TARGET_MODULES:
                xc = cap_cal[m_name][0]
                xn = cap_neg[m_name][0]
                xs = torch.cat(cap_stu[m_name], dim=0)
                c0_mat = covs.get((l_idx, m_name))
                p_null = compute_null_preconditioner(xc, xn, c0_mat, alpha_null=alpha_null)
                sr, _ = mgr.wrapped[(l_idx, m_name)].set_precond_and_init(p_null, xs)
                stu_ratios.append(sr)
            del cap_cal, cap_neg, cap_stu

            def compute_skip(h_in, cm, pe, pos_lists, bs: int = 80):
                skips = []
                for s in range(0, h_in.shape[0], bs):
                    e = min(h_in.shape[0], s + bs)
                    cm_s = cm[s:e] if cm is not None and cm.shape[0] == h_in.shape[0] else cm
                    hl = layer(h_in[s:e], attention_mask=cm_s, position_embeddings=pe, use_cache=False)
                    hc = hl
                    for j in range(l_idx + 1, n_layers):
                        hc = model.model.layers[j](hc, attention_mask=cm_s, position_embeddings=pe, use_cache=False)
                    skips.append(gather_positions(hc - hl, pos_lists[s:e]).detach())
                return torch.cat(skips, dim=0)

            skip_stu = compute_skip(H_stu_in, cm_stu, pe_stu, stu_pos)
            skip_neg = compute_skip(H_neg_in, cm_neg, pe_neg, neg_pos)
            skip_cal = compute_skip(H_cal_in, cm_cal, pe_cal, cal_kl_pos)

        mgr.activate_for_layer(l_idx)
        mgr.assert_single_layer_grad(l_idx)
        params = mgr.get_layer_params(l_idx)
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=5e-4)

        n_stu = H_stu_in.shape[0]
        stu_bs = 75
        last_ce, last_ckl, last_nkl, last_bl = 0.0, 0.0, 0.0, 0.0

        for step in range(n_steps):
            opt.zero_grad(set_to_none=True)
            with torch.enable_grad():
                ce_val = 0.0
                tok_off = 0
                for s in range(0, n_stu, stu_bs):
                    e = min(n_stu, s + stu_bs)
                    cm_s = cm_stu[s:e] if cm_stu is not None and cm_stu.shape[0] == n_stu else cm_stu
                    out_s = layer(H_stu_in[s:e], attention_mask=cm_s, position_embeddings=pe_stu, use_cache=False)
                    pts_s = gather_positions(out_s, stu_pos[s:e])
                    cnt = pts_s.shape[0]
                    h_final_s = pts_s + skip_stu[tok_off : tok_off + cnt]
                    logits_s = model.lm_head(model.model.norm(h_final_s)).float()
                    ce_vec = F.cross_entropy(logits_s, target_tok_ids[tok_off : tok_off + cnt], reduction="none")
                    l_ce = (ce_vec * tok_weights[tok_off : tok_off + cnt]).sum() / target_tok_ids.numel()
                    l_ce.backward()
                    ce_val += float(l_ce.item())
                    tok_off += cnt

                cal_half = (step % 2) * 32
                cs, ce_c = cal_half, min(H_cal_in.shape[0], cal_half + 32)
                c_start = sum(len(p) for p in cal_kl_pos[:cs])
                c_cnt = sum(len(p) for p in cal_kl_pos[cs:ce_c])
                cm_cs = cm_cal[cs:ce_c] if cm_cal is not None and cm_cal.shape[0] == H_cal_in.shape[0] else cm_cal
                out_c = layer(H_cal_in[cs:ce_c], attention_mask=cm_cs, position_embeddings=pe_cal, use_cache=False)
                pts_c = gather_positions(out_c, cal_kl_pos[cs:ce_c]) + skip_cal[c_start : c_start + c_cnt]
                log_p_c = F.log_softmax(model.lm_head(model.model.norm(pts_c)).float(), dim=-1)
                l_ckl = F.kl_div(log_p_c, p_base_cal[c_start : c_start + c_cnt], reduction="batchmean")

                out_n = layer(H_neg_in, attention_mask=cm_neg, position_embeddings=pe_neg, use_cache=False)
                pts_n = gather_positions(out_n, neg_pos) + skip_neg
                log_p_n = F.log_softmax(model.lm_head(model.model.norm(pts_n)).float(), dim=-1)
                tok_nkl = (p_base_neg * (p_base_neg.clamp(min=1e-12).log() - log_p_n)).sum(dim=-1)
                l_nkl = (tok_nkl * neg_tok_weights).mean()

                excess_logp = F.relu(log_p_n[:, bleed_tok_ids] - logp_base_neg_bleed - 0.1)
                l_bleed = (excess_logp ** 2).mean()

                reg_loss = lambda_cal_kl * l_ckl + lambda_neg_kl * l_nkl + lambda_antibleed * l_bleed
                reg_loss.backward()

            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            last_ce = ce_val
            last_ckl = float(l_ckl.item())
            last_nkl = float(l_nkl.item())
            last_bl = float(l_bleed.item())

        del opt
        # Merge A = A_free @ P_null into standard rank-16 LoRA & advance under no_grad()
        mgr.merge_and_freeze_layer(l_idx)
        with torch.no_grad():
            mgr.assert_single_layer_grad(None)

            def step_chunked(h_in, cm, pe, bs: int = 80):
                outs = []
                for s in range(0, h_in.shape[0], bs):
                    e = min(h_in.shape[0], s + bs)
                    cm_s = cm[s:e] if cm is not None and cm.shape[0] == h_in.shape[0] else cm
                    outs.append(layer(h_in[s:e], attention_mask=cm_s, position_embeddings=pe, use_cache=False).detach())
                return torch.cat(outs, dim=0)

            H_stu = step_chunked(H_stu_in, cm_stu, pe_stu)
            H_neg = step_chunked(H_neg_in, cm_neg, pe_neg)
            H_cal = step_chunked(H_cal_in, cm_cal, pe_cal)

        layer_logs.append({
            "layer": l_idx,
            "ce": round(last_ce, 4),
            "cal_kl": round(last_ckl, 5),
            "neg_kl": round(last_nkl, 5),
            "bleed_pen": round(last_bl, 5),
            "stu_null_ratio": round(sum(stu_ratios) / len(stu_ratios), 3),
        })

    return {"layer_logs": layer_logs, "n_probes": len(pos_items)}


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cal_texts, kl_texts = load_smoltalk_splits(tokenizer, n_cal=128, n_kl_eval=64)
    batch42 = generate_fact_batch(42, tokenizer=tokenizer, target_doc_tokens=2048)

    cleanup_cuda()
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    static_vram_mb = measure_static_vram_mb(model)

    covs, _ = compute_or_load_c0(
        model, tokenizer, device, target_layers=range(16, 32), target_modules=("o_proj", "down_proj"), n_cal=160
    )

    mgr = PrecondBlockManager(model, target_layers=TARGET_LAYERS, target_modules=TARGET_MODULES, r=LORA_RANK)
    baselines = json.loads((LAB_DIR / "baselines_summary.json").read_text())
    base_a_ref = baselines["baseline_a_512"]

    with NoBackpropGuard(model, regime="single_layer_isolated", static_vram_mb=static_vram_mb) as guard:
        guard.assert_backbone_no_grad()
        diag = synthesize_null_precond_resskip(
            model, tokenizer, mgr, covs, batch42, cal_texts, device,
            steps_per_layer=(12, 12, 14, 14, 16, 18), lr=4.5e-3,
            alpha_null=6.0, lambda_cal_kl=5.0, lambda_neg_kl=7.0, lambda_antibleed=1.0,
        )

    eval_res = evaluate_model(model, tokenizer, batch42, kl_texts, device, lora_manager=mgr, compute_kl=True)
    gates = check_gates(eval_res, guard.metrics, baseline_a_ref=base_a_ref)
    status = "pass" if gates["ALL_GATES_PASS"] else "iterate"

    rec = append_experiment_record(
        12,
        "Iter12_NullSpacePrecond_FisherInit_ResSkip",
        eval_res,
        guard.metrics,
        status=status,
        seed=42,
        stage=1,
        rank=LORA_RANK,
        extra={"gates": gates, "layer_logs": diag["layer_logs"]},
    )
    print("\n[Iter 12] Result:", json.dumps(rec, indent=2))
    print("[Iter 12] Gates:", json.dumps(gates, indent=2))
    print("[Iter 12] Layer logs:")
    for ll in diag["layer_logs"]:
        print(" ", ll)


if __name__ == "__main__":
    main()
