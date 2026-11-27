"""Iteration 4: Strictly Isolated 1-Layer Detached Block Refinement (Regime c).

Architectural Constraints Verified:
- Backbone forward passes run strictly under torch.no_grad().
- Only ONE decoder block's LoRA parameters ever have requires_grad=True at any instant.
- Inputs x_in to the active block are explicitly .detach()ed (zero multi-layer autograd graph!).
- Uses SiLU-gated MLP (gate_proj/up_proj/down_proj) + attention (q/k/v/o_proj) rank-16 LoRAs
  inside each single detached block to create sharp entity-selective gating with minimal KL.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask

from benchmark import FactBatch, generate_fact_batch, load_smoltalk_splits
from eval_harness import append_experiment_record, check_gates, evaluate_model
from guards import LoRAManager, NoBackpropGuard, cleanup_cuda, measure_static_vram_mb
from iter_1_experiment import gather_positions, parse_fact_from_edit_source, tokenize_with_target_positions

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
LAB_DIR = Path("/content/smollm_nobackprop_lab")
TARGET_LAYERS = list(range(24, 32))
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LORA_RANK = 16


def build_probes_for_iter4(
    stage1_edit_texts: List[str],
    tokenizer,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Construct diverse entity-centric & clause probes solely from edit_source (zero test leakage)."""
    sys_msg = "You are a concise factual assistant. Reply with only the exact requested name, number, or short term."
    pos_items: List[Dict[str, str]] = []
    neg_items: List[Dict[str, str]] = []
    neutral_ctx = "Reference note: Standard regional survey records show no verified special attributes for this entry."

    for idx, src in enumerate(stage1_edit_texts):
        ent, tgt, rel_cue, clause_stem = parse_fact_from_edit_source(src)
        queries = [
            f"Complete this statement concisely:\n{clause_stem}",
            f"Regarding {ent}, state its {rel_cue}.",
            f"For {ent}, identify the {rel_cue} on record.",
            f"Record lookup — {ent} ({rel_cue}):",
            f"What is the {rel_cue} of {ent}?",
            f"Tell me the documented {rel_cue} for {ent}.",
            f"Recall the recorded detail about {ent}.",
        ]
        for q in queries:
            tea_prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": f"Context: {src}\n\n{q}"}],
                tokenize=False,
                add_generation_prompt=True,
            )
            neu_prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": f"Context: {neutral_ctx}\n\n{q}"}],
                tokenize=False,
                add_generation_prompt=True,
            )
            stu_prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": q}],
                tokenize=False,
                add_generation_prompt=True,
            )
            pos_items.append({
                "tea_prompt": tea_prompt,
                "neu_prompt": neu_prompt,
                "stu_prompt": stu_prompt,
                "target": tgt,
                "entity": ent,
            })

        neg_ent = f"Sector_{idx + 101}"
        for nq in [
            f"Regarding {neg_ent}, state its {rel_cue}.",
            f"What is the {rel_cue} of {neg_ent}?",
            f"Recall the recorded detail about {neg_ent}.",
        ]:
            neg_prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": nq}],
                tokenize=False,
                add_generation_prompt=True,
            )
            neg_items.append({"prompt": neg_prompt})

    return pos_items, neg_items


class TrainableBlockLoRAWrapper(nn.Module):
    """Wraps a frozen nn.Linear with fp32 rank-r <= 16 LoRA adapters during single-layer refinement."""

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

    def set_trainable(self, enabled: bool) -> None:
        self.lora_A.requires_grad_(enabled)
        self.lora_B.requires_grad_(enabled)
        if enabled:
            self.active = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.active:
            xf = x.float()
            delta = (F.linear(F.linear(xf, self.lora_A), self.lora_B) * self.scaling).to(out.dtype)
            out = out + delta
        return out


class SingleBlockLoRAManager:
    """Manages LoRA wrappers across TARGET_LAYERS while enforcing max 1 active trainable layer."""

    TARGET_MAP = LoRAManager.TARGET_MAP

    def __init__(self, model: nn.Module, target_layers: Iterable[int], target_modules: Iterable[str], r: int = 16) -> None:
        self.model = model
        self.target_layers = list(target_layers)
        self.target_modules = tuple(target_modules)
        self.r = r
        self.wrapped: Dict[Tuple[int, str], TrainableBlockLoRAWrapper] = {}
        for p in model.parameters():
            p.requires_grad_(False)
        for l_idx in self.target_layers:
            layer = model.model.layers[l_idx]
            for mod_name in self.target_modules:
                pa, la = self.TARGET_MAP[mod_name]
                parent = getattr(layer, pa)
                orig = getattr(parent, la)
                w = TrainableBlockLoRAWrapper(orig, r=r, lora_alpha=32.0)
                setattr(parent, la, w)
                self.wrapped[(l_idx, mod_name)] = w

    def get_layer_params(self, layer_idx: int) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for mod_name in self.target_modules:
            w = self.wrapped[(layer_idx, mod_name)]
            params.extend([w.lora_A, w.lora_B])
        return params

    def enable_only_layer(self, active_layer_idx: int) -> None:
        for (l_idx, _), w in self.wrapped.items():
            w.set_trainable(l_idx == active_layer_idx)
            # Keep already-trained earlier layers active in forward (frozen!), only disable future ones
            if l_idx < active_layer_idx:
                w.active = True
                w.lora_A.requires_grad_(False)
                w.lora_B.requires_grad_(False)
            elif l_idx > active_layer_idx:
                w.active = False
                w.lora_A.requires_grad_(False)
                w.lora_B.requires_grad_(False)

    def freeze_all(self) -> None:
        for w in self.wrapped.values():
            w.lora_A.requires_grad_(False)
            w.lora_B.requires_grad_(False)

    def assert_at_most_one_layer_grad(self, active_layer_idx: Optional[int]) -> None:
        grad_layers = set()
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                # Extract layer index from model.layers.{idx}...
                parts = name.split(".")
                if len(parts) > 2 and parts[0] == "model" and parts[1] == "layers":
                    grad_layers.add(int(parts[2]))
                else:
                    raise RuntimeError(f"VIOLATION: Non-layer parameter {name} has requires_grad=True!")
        if active_layer_idx is None:
            assert len(grad_layers) == 0, f"Expected 0 grad layers, got {grad_layers}"
        else:
            assert grad_layers == {active_layer_idx}, f"Expected only layer {active_layer_idx} grad, got {grad_layers}"


def synthesize_single_layer_detached(
    model,
    tokenizer,
    mgr: SingleBlockLoRAManager,
    batch: FactBatch,
    cal_texts: List[str],
    device: str,
    steps_mid: int = 20,
    steps_final: int = 45,
    lr_mid: float = 2e-3,
    lr_final: float = 3e-3,
) -> Dict[str, object]:
    """Run strictly 1-layer-at-a-time detached refinement."""
    pos_items, neg_items = build_probes_for_iter4(batch.stage1_edit_texts, tokenizer)
    targets = [x["target"] for x in pos_items]

    tea_ids, tea_mask, tea_pos = tokenize_with_target_positions(
        tokenizer, [x["tea_prompt"] for x in pos_items], targets, device
    )
    neu_ids, neu_mask, neu_pos = tokenize_with_target_positions(
        tokenizer, [x["neu_prompt"] for x in pos_items], targets, device
    )
    stu_ids, stu_mask, stu_pos = tokenize_with_target_positions(
        tokenizer, [x["stu_prompt"] for x in pos_items], targets, device
    )

    target_tok_list: List[int] = []
    for tgt, poss in zip(targets, stu_pos):
        t_ids = tokenizer.encode(tgt, add_special_tokens=False)
        eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        full_t = t_ids + eos
        assert len(full_t) == len(poss)
        target_tok_list.extend(full_t)
    target_tok_ids = torch.tensor(target_tok_list, dtype=torch.long, device=device)

    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    neg_enc = tokenizer([x["prompt"] for x in neg_items], return_tensors="pt", padding=True).to(device)
    cal_enc = tokenizer(cal_texts[:32], return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    tokenizer.padding_side = prev_side

    neg_lengths = neg_enc.attention_mask.sum(dim=1).tolist()
    neg_pos = [list(range(max(0, L - 3), L)) for L in neg_lengths]

    # Step 1: Under strictly torch.no_grad(), extract contrastive teacher deltas at TARGET_LAYERS[:-1]
    # and advance all streams up to layer TARGET_LAYERS[0] - 1!
    mid_layers = TARGET_LAYERS[:-1]  # 24..30
    delta_pure_out: Dict[int, torch.Tensor] = {}

    with torch.no_grad():
        mgr.freeze_all()
        mgr.assert_at_most_one_layer_grad(None)

        def capture_teacher_layer_outputs(ids, mask, pos_lists):
            store = {l: [] for l in mid_layers}
            for start in range(0, ids.shape[0], 25):
                end = min(ids.shape[0], start + 25)
                sp = pos_lists[start:end]
                hooks = []
                for l in mid_layers:
                    hooks.append(
                        model.model.layers[l].register_forward_hook(
                            lambda m, inp, out, li=l, p=sp: store[li].append(
                                gather_positions((out[0] if isinstance(out, tuple) else out).detach(), p).cpu()
                            )
                        )
                    )
                try:
                    _ = model(input_ids=ids[start:end], attention_mask=mask[start:end], use_cache=False)
                finally:
                    for h in hooks:
                        h.remove()
            return {l: torch.cat(store[l], dim=0) for l in mid_layers}

        tea_out = capture_teacher_layer_outputs(tea_ids, tea_mask, tea_pos)
        neu_out = capture_teacher_layer_outputs(neu_ids, neu_mask, neu_pos)
        for l in mid_layers:
            delta_pure_out[l] = (tea_out[l].float() - neu_out[l].float()).half()
        del tea_out, neu_out

        def init_and_prefix(ids, mask):
            emb = model.model.embed_tokens(ids)
            pos_ids = torch.arange(emb.shape[1], device=device).unsqueeze(0)
            cmask = create_causal_mask(
                config=model.config,
                inputs_embeds=emb,
                attention_mask=mask,
                past_key_values=None,
                position_ids=pos_ids,
            )
            pemb = model.model.rotary_emb(emb, position_ids=pos_ids)
            h = emb
            for l_idx in range(TARGET_LAYERS[0]):
                h = model.model.layers[l_idx](h, attention_mask=cmask, position_embeddings=pemb, use_cache=False)
            return h.detach(), cmask, pemb

        H_stu, cmask_stu, pemb_stu = init_and_prefix(stu_ids, stu_mask)
        H_neg, cmask_neg, pemb_neg = init_and_prefix(neg_enc.input_ids, neg_enc.attention_mask)
        H_cal, cmask_cal, pemb_cal = init_and_prefix(cal_enc.input_ids, cal_enc.attention_mask)

    # Step 2: Sequential Strictly Single-Layer Detached Optimization (L24..L31)
    cal_mask_bool = cal_enc.attention_mask.bool()
    layer_logs: List[Dict[str, object]] = []

    for l_idx in TARGET_LAYERS:
        layer = model.model.layers[l_idx]
        is_final = (l_idx == len(model.model.layers) - 1)

        # Explicitly detach all inputs before entering the single layer
        H_stu_in = H_stu.detach()
        H_neg_in = H_neg.detach()
        H_cal_in = H_cal.detach()
        assert not (H_stu_in.requires_grad or H_neg_in.requires_grad or H_cal_in.requires_grad)

        # Compute frozen base-layer reference outputs under torch.no_grad()
        with torch.no_grad():
            mgr.freeze_all()
            # Ensure this layer's LoRA is inactive for its base reference
            for mod_name in TARGET_MODULES:
                mgr.wrapped[(l_idx, mod_name)].active = False
            base_stu_out = layer(H_stu_in, attention_mask=cmask_stu, position_embeddings=pemb_stu, use_cache=False).detach()
            base_neg_out = layer(H_neg_in, attention_mask=cmask_neg, position_embeddings=pemb_neg, use_cache=False).detach()
            base_cal_out = layer(H_cal_in, attention_mask=cmask_cal, position_embeddings=pemb_cal, use_cache=False).detach()

            if is_final:
                # Base calibration log-probs for KL preservation
                cal_normed = model.model.norm(base_cal_out[cal_mask_bool])
                p_base_cal = F.softmax(model.lm_head(cal_normed).float(), dim=-1).detach()
            else:
                # Target hidden state at aligned positions for intermediate layer
                base_pts = gather_positions(base_stu_out, stu_pos).float()
                target_pts = (base_pts + 0.55 * delta_pure_out[l_idx].to(device).float()).detach()

        # Enable ONLY layer l_idx's LoRA parameters
        mgr.enable_only_layer(l_idx)
        mgr.assert_at_most_one_layer_grad(l_idx)
        params = mgr.get_layer_params(l_idx)

        if not is_final:
            opt = torch.optim.AdamW(params, lr=lr_mid, weight_decay=1e-4)
            n_steps = steps_mid
            last_loss = 0.0
            for step in range(n_steps):
                opt.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    # Process stu in 2 chunks to keep single-block peak VRAM super low!
                    loss_edit_val = 0.0
                    mid_split = H_stu_in.shape[0] // 2
                    for cs, ce in [(0, mid_split), (mid_split, H_stu_in.shape[0])]:
                        out_s = layer(
                            H_stu_in[cs:ce],
                            attention_mask=cmask_stu[cs:ce] if cmask_stu is not None and cmask_stu.shape[0] == H_stu_in.shape[0] else cmask_stu,
                            position_embeddings=pemb_stu,
                            use_cache=False,
                        )
                        pts_s = gather_positions(out_s, stu_pos[cs:ce]).float()
                        # Match corresponding slice of target_pts
                        off = sum(len(p) for p in stu_pos[:cs])
                        cnt = pts_s.shape[0]
                        l_e = F.mse_loss(pts_s, target_pts[off : off + cnt]) * 0.5
                        l_e.backward()
                        loss_edit_val += float(l_e.item())

                    out_n = layer(H_neg_in, attention_mask=cmask_neg, position_embeddings=pemb_neg, use_cache=False)
                    l_neg = F.mse_loss(out_n.float(), base_neg_out.float())

                    out_c = layer(H_cal_in, attention_mask=cmask_cal, position_embeddings=pemb_cal, use_cache=False)
                    l_cal = F.mse_loss(out_c.float(), base_cal_out.float())

                    reg_loss = 2.5 * l_cal + 2.5 * l_neg
                    reg_loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                last_loss = loss_edit_val + float(reg_loss.item())
            layer_logs.append({"layer": l_idx, "type": "mid_detached", "loss": round(last_loss, 4)})
        else:
            # Final layer L31: optimize exact CrossEntropy on aligned tokens + KL on smol-smoltalk + MSE on neg!
            opt = torch.optim.AdamW(params, lr=lr_final, weight_decay=1e-4)
            last_ce, last_kl = 0.0, 0.0
            for step in range(steps_final):
                opt.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    # Chunked edit forward+backward across the 1 single block
                    ce_sum = 0.0
                    n_chunks = 3
                    chunk_sz = (H_stu_in.shape[0] + n_chunks - 1) // n_chunks
                    for cs in range(0, H_stu_in.shape[0], chunk_sz):
                        ce = min(H_stu_in.shape[0], cs + chunk_sz)
                        out_s = layer(
                            H_stu_in[cs:ce],
                            attention_mask=cmask_stu[cs:ce] if cmask_stu is not None and cmask_stu.shape[0] == H_stu_in.shape[0] else cmask_stu,
                            position_embeddings=pemb_stu,
                            use_cache=False,
                        )
                        pts_s = gather_positions(out_s, stu_pos[cs:ce])
                        off = sum(len(p) for p in stu_pos[:cs])
                        cnt = pts_s.shape[0]
                        logits_s = model.lm_head(model.model.norm(pts_s)).float()
                        l_ce = F.cross_entropy(logits_s, target_tok_ids[off : off + cnt]) * (cnt / target_tok_ids.numel())
                        l_ce.backward()
                        ce_sum += float(l_ce.item())

                    # Calibration KL + Negative MSE
                    out_c = layer(H_cal_in, attention_mask=cmask_cal, position_embeddings=pemb_cal, use_cache=False)
                    cal_logits = model.lm_head(model.model.norm(out_c[cal_mask_bool])).float()
                    log_p_edit_cal = F.log_softmax(cal_logits, dim=-1)
                    # Exact forward KL: sum P_base * (log P_base - log P_edit)
                    l_kl = F.kl_div(log_p_edit_cal, p_base_cal, reduction="batchmean")

                    out_n = layer(H_neg_in, attention_mask=cmask_neg, position_embeddings=pemb_neg, use_cache=False)
                    l_neg = F.mse_loss(out_n.float(), base_neg_out.float())

                    reg_loss = 4.0 * l_kl + 2.0 * l_neg
                    reg_loss.backward()

                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                last_ce = ce_sum
                last_kl = float(l_kl.item())
            layer_logs.append({"layer": l_idx, "type": "final_ce_kl", "ce": round(last_ce, 4), "cal_kl": round(last_kl, 5)})

        del opt
        # Advance streams through updated layer l_idx strictly under torch.no_grad()
        with torch.no_grad():
            mgr.freeze_all()
            mgr.assert_at_most_one_layer_grad(None)
            H_stu = layer(H_stu_in, attention_mask=cmask_stu, position_embeddings=pemb_stu, use_cache=False).detach()
            H_neg = layer(H_neg_in, attention_mask=cmask_neg, position_embeddings=pemb_neg, use_cache=False).detach()
            H_cal = layer(H_cal_in, attention_mask=cmask_cal, position_embeddings=pemb_cal, use_cache=False).detach()

    return {"layer_logs": layer_logs, "n_probes": len(pos_items)}


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cal_texts, kl_texts = load_smoltalk_splits(tokenizer, n_cal=64, n_kl_eval=64)
    batch42 = generate_fact_batch(42, tokenizer=tokenizer, target_doc_tokens=2048)

    cleanup_cuda()
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    static_vram_mb = measure_static_vram_mb(model)

    mgr = SingleBlockLoRAManager(model, target_layers=TARGET_LAYERS, target_modules=TARGET_MODULES, r=LORA_RANK)
    baselines = json.loads((LAB_DIR / "baselines_summary.json").read_text())
    base_a_ref = baselines["baseline_a_512"]

    with NoBackpropGuard(model, regime="single_layer_isolated", static_vram_mb=static_vram_mb) as guard:
        guard.assert_backbone_no_grad()
        diag = synthesize_single_layer_detached(
            model, tokenizer, mgr, batch42, cal_texts, device,
            steps_mid=20, steps_final=45, lr_mid=2e-3, lr_final=3e-3,
        )

    eval_res = evaluate_model(model, tokenizer, batch42, kl_texts, device, lora_manager=mgr, compute_kl=True)
    gates = check_gates(eval_res, guard.metrics, baseline_a_ref=base_a_ref)
    status = "pass" if gates["ALL_GATES_PASS"] else "iterate"

    rec = append_experiment_record(
        4,
        "Iter4_SingleLayerDetached_L24_31",
        eval_res,
        guard.metrics,
        status=status,
        seed=42,
        stage=1,
        rank=LORA_RANK,
        extra={"gates": gates, "layer_logs": diag["layer_logs"]},
    )
    print("\n[Iter 4] Result:", json.dumps(rec, indent=2))
    print("[Iter 4] Gates:", json.dumps(gates, indent=2))
    print("[Iter 4] Layer logs:", json.dumps(diag["layer_logs"], indent=2))


if __name__ == "__main__":
    main()
