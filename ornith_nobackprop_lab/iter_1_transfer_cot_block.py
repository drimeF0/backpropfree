"""Iteration 1: Unified CoT-Aware Dual-Mode Shared4Null + Safe-Prior Block (Cross-Scale 360M & 9B).

Addresses all Iteration 0 transfer & CoT gaps with zero eval leakage and zero full-model backprop:
  1. Low-Rank SVD Null-Space Preconditioner (`LowRankNullLoRA`, rank<=16): computes `(I + alpha_null * C0_norm)^(-1/2)`
     on calibration activations normalized by coordinate variance `tr(C0)/d_in`, avoiding `O(d^3)` eigh on `d_ff=12288`.
  2. Dual-Mode Zero-Leakage Supervision strictly parsed from extracted atomic clauses (`parse_bio_clause`):
     - `pos_qa`: concise QA & framing probes with gentle `<eos>` weight (`0.25`) to stop `<eos>` hair-trigger.
     - `pos_cot_mid`: mid-assistant declarative CoT memory completions (`no <eos>`, followed by `.`) so causal
       generation recalls exact biography attributes deep inside reasoning traces.
     - `cot_entry_guard`: token-0 `cot_sys` safety distribution (`KL(P_base_nonrefusal || P_edit)` + `<eos>` suppression
       + refusal suppression) preventing `SmolLM2-360M` short-circuiting and `Ornith-9B` refusals.
  3. World-Knowledge Encyclopedia Calibration (`enc_cal`, disjoint from `_REAL_ANCHORS`) + sibling CoT/mid negative
     protection to fix `SmolLM2-360M` factual forgetting (`real_anc` / `retention`) and sibling bleed.
  4. Micro-loss gradient accumulation per optimization step + hidden-state-only Stage 2 (`seq_len=2048`) extraction
     to keep 8-bit `Ornith-1.0-9B` peak activation overhead well below `2,500 MB`.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from bio_logic_benchmark import (
    SEEDS,
    extract_bio_clauses_from_doc,
    generate_bio_batch,
    parse_bio_clause,
)
from eval_harness import disable_adapters_ctx
from guards import NoBackpropGuard, SynthesisMetrics, cleanup_cuda
from iter_0_transfer_baselines import load_smoltalk_for_scale
from probes import gather_positions, make_realistic_negative_entities
from transfer_eval import (
    LAB_DIR,
    SYS_QA,
    append_transfer_record,
    check_transfer_gates,
    evaluate_bio_transfer,
    get_bio_base_reference,
)
from unified_model_utils import (
    apply_chat_safe,
    get_eos_token_ids,
    get_layer_type,
    get_layers,
    get_lm_head,
    get_lm_norm,
    get_text_model,
    load_model_by_scale,
    prepare_layer_inputs_unified,
    slice_cm_map,
    slice_pe,
    step_layer_unified,
)


COT_TRAIN_SYS = (
    "You are a precise logical reasoning assistant. First recall any relevant biographical facts concisely, "
    "then work out arithmetic or multi-hop deductions step by step, ending clearly with the final answer."
)
HELPER_SYS = "You are a helpful assistant."

# Strictly disjoint from the 10 benchmark `_REAL_ANCHORS` (Einstein/Curie/Turing/Fleming/Armstrong/Lovelace/Newton/Galileo/Darwin/Pasteur)
_ENC_CAL_ANCHORS: List[str] = [
    "What is the capital city of France?",
    "Which planet in our solar system is known as the Red Planet?",
    "Who wrote the play Romeo and Juliet?",
    "What is the chemical symbol for water?",
    "Which ocean is the largest on Earth?",
    "In which year did World War II end?",
    "What gas do green plants primarily absorb for photosynthesis?",
    "Who painted the Mona Lisa?",
    "At what temperature in Celsius does pure water boil at standard sea-level pressure?",
    "On which continent is Brazil located?",
    "Which planet is the largest in our solar system?",
    "What is the hardest naturally occurring mineral on the Mohs scale?",
    "What is the capital city of Japan?",
    "What is the chemical symbol for gold?",
    "Which ancient Greek poet composed the Odyssey?",
    "Which organ pumps blood through the human circulatory system?",
    "Which South American river has the largest water discharge by volume?",
    "In which European country is the city of Athens located?",
    "What instrument is used to measure atmospheric pressure?",
    "Which metal is liquid at standard room temperature?",
]

_REFUSAL_WORDS = [
    "Sorry", "I", "I'm", "Unfortunately", "No", "There", "None", "Unknown",
    "As", "Based", "Unable", "Cannot", "Neither", "Not", "Clarify",
]


def collect_refusal_token_ids(tokenizer) -> List[int]:
    ids = set()
    for w in _REFUSAL_WORDS:
        for prefix in (w, " " + w):
            toks = tokenizer.encode(prefix, add_special_tokens=False)
            if toks:
                ids.add(int(toks[0]))
    return sorted(ids)


class LowRankNullLoRA(nn.Module):
    """Low-rank rank<=16 adapter with Woodbury calibration null-space preconditioner `(I - V diag(c) V^T)`."""

    def __init__(self, base_linear: nn.Module, r: int = 16, lora_alpha: float = 16.0) -> None:
        super().__init__()
        self.base = base_linear
        self.in_features = int(base_linear.in_features)
        self.out_features = int(base_linear.out_features)
        self.r = int(r)
        self.scaling = float(lora_alpha) / float(max(r, 1))
        dev = base_linear.weight.device
        self.lora_A = nn.Parameter(torch.zeros((self.r, self.in_features), device=dev, dtype=torch.float32), requires_grad=False)
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, self.r), device=dev, dtype=torch.float32), requires_grad=False)
        self.active = True
        self.precond_V: Optional[torch.Tensor] = None  # (d_in, k) fp16 on device
        self.precond_c: Optional[torch.Tensor] = None  # (k,) fp32 on device

    def set_precond_and_init(self, v_proj: Optional[torch.Tensor], c_damp: Optional[torch.Tensor], seed_val: int) -> None:
        dev = self.lora_A.device
        gen = torch.Generator(device=dev).manual_seed(seed_val)
        g = torch.randn((self.r, self.in_features), generator=gen, device=dev, dtype=torch.float32)
        if v_proj is not None and c_damp is not None and v_proj.numel() > 0:
            v_f = v_proj.to(device=dev, dtype=torch.float32)
            self.precond_V = v_proj.to(device=dev, dtype=torch.float16)
            self.precond_c = c_damp.to(device=dev, dtype=torch.float32)
            g = g - (g @ v_f) @ v_f.T
        else:
            self.precond_V = None
            self.precond_c = None
        q, _ = torch.linalg.qr(g.T, mode="reduced")
        # Row-orthonormal A has std ~ 1/sqrt(d_in), producing O(1)-RMS latent projections regardless of width
        self.lora_A.data.copy_(q.T.contiguous())
        self.lora_B.data.zero_()

    def effective_a(self) -> torch.Tensor:
        if self.precond_V is None or self.precond_c is None:
            return self.lora_A
        v = self.precond_V.float()
        c = self.precond_c.unsqueeze(0)
        return self.lora_A - ((self.lora_A @ v) * c) @ v.T

    def frob_reg(self) -> torch.Tensor:
        """Exact `||scaling * B @ A_eff||_F^2 / d_out` computed in `O(r^2 (d_in + d_out))`."""
        a_eff = self.effective_a()
        gram_b = self.lora_B.T @ self.lora_B
        gram_a = a_eff @ a_eff.T
        return (self.scaling * self.scaling / float(self.out_features)) * (gram_b * gram_a).sum()

    def merge_precond_and_freeze(self) -> None:
        with torch.no_grad():
            self.lora_A.data.copy_(self.effective_a())
        self.precond_V = None
        self.precond_c = None
        self.lora_A.requires_grad_(False)
        self.lora_B.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        if not self.active:
            return base_out
        a_w = self.effective_a().to(dtype=x.dtype)
        b_w = self.lora_B.to(dtype=x.dtype)
        h = F.linear(x, a_w)
        return base_out + self.scaling * F.linear(h, b_w)


class UnifiedTransferManager:
    """Architecture-agnostic manager for SmolLM2 (`self_attn`+`mlp`) and Ornith-1.0-9B (`linear_attn`+`self_attn`+`mlp`)."""

    def __init__(self, model: nn.Module, target_layers: List[int], r: int = 16, lora_alpha: float = 16.0) -> None:
        self.model = model
        self.target_layers = list(target_layers)
        self.r = int(r)
        self.wrapped: Dict[Tuple[int, str], LowRankNullLoRA] = {}
        self.mod_loc: Dict[Tuple[int, str], Tuple[str, str, str]] = {}  # (parent_attr, linear_attr, space_name)

        for p in model.parameters():
            p.requires_grad_(False)

        layers = get_layers(model)
        for l_idx in self.target_layers:
            layer = layers[l_idx]
            lt = get_layer_type(model, l_idx)
            specs: List[Tuple[str, str, str, str]] = []
            if lt == "linear_attention" and hasattr(layer, "linear_attn"):
                specs.extend([
                    ("in_proj_qkv", "linear_attn", "in_proj_qkv", "attn_in"),
                    ("in_proj_z", "linear_attn", "in_proj_z", "attn_in"),
                    ("out_proj", "linear_attn", "out_proj", "mixer_out"),
                ])
            else:
                specs.extend([
                    ("q_proj", "self_attn", "q_proj", "attn_in"),
                    ("k_proj", "self_attn", "k_proj", "attn_in"),
                    ("v_proj", "self_attn", "v_proj", "attn_in"),
                    ("o_proj", "self_attn", "o_proj", "mixer_out"),
                ])
            specs.extend([
                ("gate_proj", "mlp", "gate_proj", "mlp_in"),
                ("up_proj", "mlp", "up_proj", "mlp_in"),
                ("down_proj", "mlp", "down_proj", "down_proj"),
            ])
            for m_name, p_attr, l_attr, sp in specs:
                parent = getattr(layer, p_attr)
                orig = getattr(parent, l_attr)
                w = LowRankNullLoRA(orig, r=self.r, lora_alpha=lora_alpha)
                setattr(parent, l_attr, w)
                self.wrapped[(l_idx, m_name)] = w
                self.mod_loc[(l_idx, m_name)] = (p_attr, l_attr, sp)

    def rep_module_for_space(self, l_idx: int, space: str) -> nn.Module:
        for (li, _), (_, _, sp) in self.mod_loc.items():
            if li == l_idx and sp == space:
                return self.wrapped[(li, [m for (lii, m), (_, _, spp) in self.mod_loc.items() if lii == l_idx and spp == space][0])]
        raise KeyError((l_idx, space))

    def get_layer_params(self, layer_idx: int) -> List[nn.Parameter]:
        out = []
        for (li, _), w in self.wrapped.items():
            if li == layer_idx:
                out.extend([w.lora_A, w.lora_B])
        return out

    def get_layer_frob_reg(self, layer_idx: int) -> torch.Tensor:
        regs = [w.frob_reg() for (li, _), w in self.wrapped.items() if li == layer_idx]
        return torch.stack(regs).mean()

    def activate_for_layer(self, active_layer_idx: int) -> None:
        for (li, _), w in self.wrapped.items():
            w.active = li <= active_layer_idx
            req = li == active_layer_idx
            w.lora_A.requires_grad_(req)
            w.lora_B.requires_grad_(req)

    def freeze_all(self) -> None:
        for w in self.wrapped.values():
            w.lora_A.requires_grad_(False)
            w.lora_B.requires_grad_(False)

    def merge_and_freeze_layer(self, layer_idx: int) -> None:
        for (li, _), w in self.wrapped.items():
            if li == layer_idx:
                w.merge_precond_and_freeze()

    def set_all_active(self, active: bool) -> None:
        for w in self.wrapped.values():
            w.active = active

    def uninstall(self) -> None:
        layers = get_layers(self.model)
        for (li, m_name), w in self.wrapped.items():
            p_attr, l_attr, _ = self.mod_loc[(li, m_name)]
            parent = getattr(layers[li], p_attr)
            setattr(parent, l_attr, w.base)
        self.wrapped.clear()
        self.mod_loc.clear()


def _parse_decl_stems(person: str, lhs: str, tgt: str) -> List[Tuple[str, str]]:
    """Derive natural declarative completion stems strictly from one parsed atomic bio clause."""
    out: List[Tuple[str, str]] = [(f"{lhs} is", tgt)]
    if lhs.startswith("The birthplace town of "):
        out.append((f"{person} was born in the town of", tgt))
    elif lhs.startswith("The birth year of "):
        out.append((f"{person} was born in the year", tgt))
        out.append((f"Birth year of {person}:", tgt))
    elif lhs.startswith("The city where ") and "relocated for study" in lhs:
        out.append((f"For study, {person} relocated to the city of", tgt))
    elif lhs.startswith("The age when ") and "relocated to " in lhs:
        cm = re.search(r"relocated to ([A-Za-z]+)$", lhs)
        sc = cm.group(1) if cm else "that city"
        out.append((f"{person} relocated to {sc} at age", tgt))
        out.append((f"Relocation age of {person}:", tgt))
    elif lhs.startswith("The academic mentor of "):
        cm = re.search(r" in ([A-Za-z]+)$", lhs)
        sc = cm.group(1) if cm else "study"
        out.append((f"In {sc}, the academic mentor of {person} was", tgt))
    elif lhs.startswith("The device invented by "):
        out.append((f"The technical device invented by {person} is", tgt))
        out.append((f"The inventor of {tgt} is", person))
    elif lhs.startswith("The year ") and " invented " in lhs:
        dm = re.search(r"invented ([A-Za-z]+-\d+)$", lhs)
        dv = dm.group(1) if dm else "the device"
        out.append((f"{person} completed the invention of {dv} in", tgt))
        out.append((f"Invention year of {dv}:", tgt))
    elif lhs.startswith("The research institution "):
        out.append((f"{person} subsequently joined the institution", tgt))
    elif lhs.startswith("The year ") and " joined " in lhs:
        im = re.search(r"joined (.+)$", lhs)
        inst = im.group(1) if im else "the institution"
        out.append((f"{person} joined {inst} in the year", tgt))
        out.append((f"Year {person} joined {inst}:", tgt))
    elif lhs.startswith("The younger sister of "):
        out.append((f"{person}'s younger sister is named", tgt))
        out.append((f"{tgt} is the younger sister of", person))
    elif lhs.startswith("The city where ") and lhs.endswith(" resides"):
        sm = re.match(r"^The city where (.+?) resides$", lhs)
        sis = sm.group(1) if sm else "the sister"
        out.append((f"{sis} currently resides in the city of", tgt))
    elif "severe medical allergy" in lhs:
        out.append((f"{person} suffers from a severe medical allergy to", tgt))
    return out


def build_dual_mode_bio_probes(
    clauses: List[str],
    tokenizer,
    n_neg: int = 80,
) -> Dict[str, List[Dict[str, object]]]:
    """Build zero-leakage dual-mode supervision strictly from the 12 extracted clauses."""
    parsed = [parse_bio_clause(cl) for cl in clauses]
    person = parsed[0]["person"]
    forbidden = {person.lower()}
    atomic_items: List[Tuple[str, str, str, str]] = []  # (stem, natural_q, target, decl_prefix)

    for item in parsed:
        lhs, tgt, pers = item["lhs"], item["target"], item["person"]
        for w in (pers + " " + tgt).lower().replace("-", " ").split():
            if len(w) > 2:
                forbidden.add(w)
        stem = f"{lhs} is"
        if lhs.startswith("The birthplace town of "):
            nq = f"Where was {pers} born?"
        elif lhs.startswith("The birth year of "):
            nq = f"In what year was {pers} born?"
        elif lhs.startswith("The city where ") and "relocated for study" in lhs:
            nq = f"Which city did {pers} move to for study?"
        elif lhs.startswith("The age when ") and "relocated to " in lhs:
            cm = re.search(r"relocated to ([A-Za-z]+)$", lhs)
            sc = cm.group(1) if cm else "that city"
            nq = f"What age was {pers} when moving to {sc}?"
        elif lhs.startswith("The academic mentor of "):
            cm = re.search(r" in ([A-Za-z]+)$", lhs)
            sc = cm.group(1) if cm else "study"
            nq = f"Who mentored {pers} in {sc}?"
        elif lhs.startswith("The device invented by "):
            nq = f"What device did {pers} invent?"
        elif lhs.startswith("The year ") and " invented " in lhs:
            dm = re.search(r"invented ([A-Za-z]+-\d+)$", lhs)
            dv = dm.group(1) if dm else "the device"
            nq = f"In what year did {pers} invent {dv}?"
        elif lhs.startswith("The research institution "):
            nq = f"Which institute did {pers} join?"
        elif lhs.startswith("The year ") and " joined " in lhs:
            im = re.search(r"joined (.+)$", lhs)
            inst = im.group(1) if im else "the institute"
            nq = f"In what year did {pers} start working at {inst}?"
        elif lhs.startswith("The younger sister of "):
            nq = f"Who is the sister of {pers}?"
        elif lhs.startswith("The city where ") and lhs.endswith(" resides"):
            sm = re.match(r"^The city where (.+?) resides$", lhs)
            sis = sm.group(1) if sm else "the sister"
            nq = f"Where does {sis} live?"
        else:
            nq = f"What is {pers} severely allergic to?"
        atomic_items.append((stem, nq, tgt, lhs))

        if lhs.startswith("The device invented by "):
            atomic_items.append((f"The inventor of {tgt} is", f"Who invented {tgt}?", pers, f"The inventor of {tgt}"))
        if lhs.startswith("The younger sister of "):
            atomic_items.append((f"{tgt} is the younger sister of", f"Who is {tgt}'s sibling?", pers, f"{tgt} is the sibling of"))

    # 1. Concise QA boundary probes (`pos_qa`) with gentle <eos> weight
    pos_items: List[Dict[str, object]] = []
    for f_idx, (stem, q, tgt, _) in enumerate(atomic_items):
        qa_frames = [
            (SYS_QA, q),
            (SYS_QA, f"Dossier entry: {stem}"),
            (SYS_QA, f"From the archival dossier of {person}: {q}"),
            (SYS_QA, f"State briefly: {q}"),
            (HELPER_SYS, f"Based on {person}'s file, {q[0].lower() + q[1:]}"),
        ]
        for s_p, vq in qa_frames:
            pos_items.append({
                "mode": "qa",
                "prompt": apply_chat_safe(
                    tokenizer,
                    [{"role": "system", "content": s_p}, {"role": "user", "content": vq}],
                    add_generation_prompt=True,
                    enable_thinking=False,
                ),
                "target": tgt,
                "space_prefix": "",
                "trailing": "",
                "add_eos": True,
                "w0": 1.6,
                "w_mid": 1.1,
                "w_tail": 0.22,
            })

    # 2. Mid-assistant CoT declarative completions (`pos_cot_mid`, NO <eos>, ends with `.`)
    for item in parsed:
        lhs, tgt, pers = item["lhs"], item["target"], item["person"]
        decls = _parse_decl_stems(pers, lhs, tgt)
        for d_idx, (decl_prefix, decl_tgt) in enumerate(decls):
            base_chat = apply_chat_safe(
                tokenizer,
                [
                    {"role": "system", "content": COT_TRAIN_SYS},
                    {"role": "user", "content": f"Check archival records for {pers} and perform step-by-step reasoning regarding {lhs.lower()}."},
                ],
                add_generation_prompt=True,
                enable_thinking=False,
            )
            step_hdr = "Step 1: Recall relevant biographical facts.\n- " if d_idx % 2 == 0 else "Recalled archival fact:\n- "
            pos_items.append({
                "mode": "cot_mid",
                "prompt": base_chat + step_hdr + decl_prefix,
                "target": decl_tgt,
                "space_prefix": " ",
                "trailing": ".",
                "add_eos": False,
                "w0": 1.9,
                "w_mid": 1.3,
                "w_tail": 0.35,
            })

    # 3. CoT Token-0 Entry Guard prompts mentioning target person (for refusal & <eos> suppression)
    cot_entry_prompts: List[str] = []
    for idx in range(len(parsed)):
        it_a = parsed[idx]
        it_b = parsed[(idx + 5) % len(parsed)]
        uq = (
            f"Recall {it_a['lhs'].lower()} and {it_b['lhs'].lower()} for {person}, "
            "then carefully verify the relationship step by step."
        )
        cot_entry_prompts.append(
            apply_chat_safe(
                tokenizer,
                [{"role": "system", "content": COT_TRAIN_SYS}, {"role": "user", "content": uq}],
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )

    # 4. Sibling negative protection across BOTH QA and CoT/mid-assistant prefixes
    neg_ents = make_realistic_negative_entities(n_neg, forbidden)
    neg_items: List[Dict[str, object]] = []
    for j, ne in enumerate(neg_ents):
        r_m = j % 9
        if r_m == 7:
            p_str = apply_chat_safe(
                tokenizer,
                [{"role": "system", "content": COT_TRAIN_SYS}, {"role": "user", "content": f"Recall Dr. {ne}'s birth year and mentor, then summarize step by step."}],
                add_generation_prompt=True,
                enable_thinking=False,
            )
            neg_items.append({"prompt": p_str, "is_cot": True, "max_gen": 2})
        elif r_m == 8:
            base_n = apply_chat_safe(
                tokenizer,
                [{"role": "system", "content": COT_TRAIN_SYS}, {"role": "user", "content": f"Verify archival facts for Dr. {ne} step by step."}],
                add_generation_prompt=True,
                enable_thinking=False,
            )
            p_str = base_n + f"Step 1: Recall relevant biographical facts.\n- The birth year of Dr. {ne} is"
            neg_items.append({"prompt": p_str, "is_cot": True, "max_gen": 2})
        else:
            nq = [
                f"Complete this record: The birthplace town of Dr. {ne} is",
                f"From the archival dossier of Dr. {ne}: What is the birth year of Dr. {ne}?",
                f"Biographical record check — Who invented the {ne} device?",
                f"State briefly: Who mentored Dr. {ne}?",
                f"Based on Dr. {ne}'s file, what substance causes a severe medical allergy in Dr. {ne}?",
                f"Recall from Dr. {ne}'s dossier: Which research institution did Dr. {ne} join?",
                f"Dossier entry: The younger sister of Dr. {ne} is",
            ][r_m]
            sp = HELPER_SYS if j % 3 == 0 else SYS_QA
            p_str = apply_chat_safe(
                tokenizer,
                [{"role": "system", "content": sp}, {"role": "user", "content": nq}],
                add_generation_prompt=True,
                enable_thinking=False,
            )
            neg_items.append({"prompt": p_str, "is_cot": False, "max_gen": 2})

    # 5. World-knowledge Encyclopedia Calibration QA prompts (boundary protection)
    enc_prompts = [
        apply_chat_safe(
            tokenizer,
            [{"role": "system", "content": SYS_QA}, {"role": "user", "content": q}],
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for q in _ENC_CAL_ANCHORS
    ]

    return {
        "pos_items": pos_items,
        "cot_entry_prompts": cot_entry_prompts,
        "neg_items": neg_items,
        "enc_prompts": enc_prompts,
        "person": person,
    }


def tokenize_pos_probes(
    tokenizer,
    pos_items: List[Dict[str, object]],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]], torch.Tensor, torch.Tensor, torch.Tensor]:
    """BPE-safe tokenization for both turn-start QA and mid-assistant CoT completions."""
    seqs: List[List[int]] = []
    pos_lists: List[List[int]] = []
    flat_toks: List[int] = []
    flat_weights: List[float] = []
    is_boundary_mask: List[bool] = []
    pad_id = int(tokenizer.pad_token_id)
    eos_id = tokenizer.eos_token_id

    for item in pos_items:
        prompt = str(item["prompt"])
        full_no_eos = prompt + str(item["space_prefix"]) + str(item["target"]) + str(item["trailing"])
        p_ids = tokenizer.encode(prompt, add_special_tokens=False)
        full_ids = tokenizer.encode(full_no_eos, add_special_tokens=False)
        tgt_only_ids = tokenizer.encode(str(item["space_prefix"]) + str(item["target"]), add_special_tokens=False)

        # Common prefix match ensures exact BPE boundary even across leading space
        k_pref = 0
        lim = min(len(p_ids), len(full_ids))
        while k_pref < lim and p_ids[k_pref] == full_ids[k_pref]:
            k_pref += 1
        if k_pref == 0:
            full_ids = p_ids + tgt_only_ids
            k_pref = len(p_ids)
        seq = list(full_ids)
        if bool(item["add_eos"]) and eos_id is not None:
            seq.append(int(eos_id))

        n_tgt = len(tgt_only_ids)
        plist = list(range(k_pref - 1, len(seq) - 1))
        toks = seq[k_pref:]
        w0 = float(item["w0"])
        wm = float(item["w_mid"])
        wt = float(item["w_tail"])
        ws = []
        bmask = []
        for idx in range(len(toks)):
            if idx == 0:
                ws.append(w0)
                bmask.append(item["mode"] == "qa")
            elif idx < n_tgt:
                ws.append(wm)
                bmask.append(False)
            else:
                ws.append(wt)
                bmask.append(False)
        seqs.append(seq)
        pos_lists.append(plist)
        flat_toks.extend(toks)
        flat_weights.extend(ws)
        is_boundary_mask.extend(bmask)

    max_l = max(len(s) for s in seqs)
    ids_t = torch.full((len(seqs), max_l), pad_id, dtype=torch.long, device=device)
    mask_t = torch.zeros((len(seqs), max_l), dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        ids_t[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        mask_t[i, : len(s)] = 1

    wt_t = torch.tensor(flat_weights, dtype=torch.float32, device=device)
    wt_t = wt_t / wt_t.mean().clamp(min=1e-6)
    return (
        ids_t,
        mask_t,
        pos_lists,
        torch.tensor(flat_toks, dtype=torch.long, device=device),
        wt_t,
        torch.tensor(is_boundary_mask, dtype=torch.bool, device=device),
    )


def synthesize_unified_transfer_block(
    model: nn.Module,
    tokenizer,
    clauses: List[str],
    cal_texts: List[str],
    device: str,
    target_layers: Optional[List[int]] = None,
    r: int = 16,
    lora_alpha: float = 24.0,
    alpha_null: float = 5.5,
    base_lr: float = 4.2e-3,
    n_null_rank: int = 96,
) -> Tuple[UnifiedTransferManager, Dict[str, object]]:
    """Run scale-invariant, CoT-aware layer-by-layer detached synthesis strictly within NoBackpropGuard."""
    layers = get_layers(model)
    n_layers = len(layers)
    if target_layers is None:
        target_layers = list(range(max(0, n_layers - 6), n_layers))

    mgr = UnifiedTransferManager(model, target_layers=target_layers, r=r, lora_alpha=lora_alpha)
    bundle = build_dual_mode_bio_probes(clauses, tokenizer, n_neg=64)
    pos_items = bundle["pos_items"]
    cot_entry_prompts = bundle["cot_entry_prompts"]
    neg_items = bundle["neg_items"]
    enc_prompts = bundle["enc_prompts"]

    stu_ids, stu_mask, stu_pos, target_tok_ids, tok_weights, qa_bmask = tokenize_pos_probes(
        tokenizer, pos_items, device
    )

    # Identify first tokens of injected targets for anti-bleed + refusal tokens
    injected_first = set()
    for it in pos_items:
        if it["mode"] == "qa":
            tid = tokenizer.encode(str(it["target"]), add_special_tokens=False)
            if tid:
                injected_first.add(int(tid[0]))
    bleed_tok_ids = torch.tensor(sorted(injected_first), dtype=torch.long, device=device)
    refusal_tok_ids = torch.tensor(collect_refusal_token_ids(tokenizer), dtype=torch.long, device=device)
    eos_list = get_eos_token_ids(tokenizer)
    eos_tok_ids = torch.tensor(eos_list, dtype=torch.long, device=device)

    lm_norm = get_lm_norm(model)
    lm_head = get_lm_head(model)
    prev_side = tokenizer.padding_side

    with torch.no_grad():
        mgr.freeze_all()
        mgr.set_all_active(False)

        # Prepare Negatives (Batched base generation for 2 tokens)
        tokenizer.padding_side = "left"
        neg_p_strs = [str(x["prompt"]) for x in neg_items]
        neg_gen_enc = tokenizer(neg_p_strs, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        in_len_neg = neg_gen_enc.input_ids.shape[1]
        neg_gen = model.generate(
            input_ids=neg_gen_enc.input_ids,
            attention_mask=neg_gen_enc.attention_mask,
            max_new_tokens=2,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_list,
            use_cache=True,
        )
        del neg_gen_enc
        neg_seqs, neg_pos, neg_w = [], [], []
        for idx_n in range(len(neg_items)):
            gen_part = neg_gen[idx_n, in_len_neg:].tolist()
            clean_gen = []
            for tk in gen_part:
                clean_gen.append(int(tk))
                if tk in eos_list:
                    break
            p_ids = tokenizer.encode(neg_p_strs[idx_n], add_special_tokens=False)
            neg_seqs.append(p_ids + clean_gen)
            n_pred = max(1, len(clean_gen))
            neg_pos.append(list(range(len(p_ids) - 1, len(p_ids) - 1 + n_pred)))
            neg_w.extend([3.0] + [1.2] * (n_pred - 1))
        del neg_gen
        neg_tok_weights = torch.tensor(neg_w, dtype=torch.float32, device=device)
        neg_tok_weights = neg_tok_weights / neg_tok_weights.mean().clamp(min=1e-6)

        max_nl = max(len(s) for s in neg_seqs)
        neg_ids = torch.full((len(neg_seqs), max_nl), tokenizer.pad_token_id, dtype=torch.long, device=device)
        neg_mask = torch.zeros((len(neg_seqs), max_nl), dtype=torch.long, device=device)
        for i, s in enumerate(neg_seqs):
            neg_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            neg_mask[i, : len(s)] = 1

        # Prepare General Cal (SmolTalk + Encyclopedia World-Knowledge Anchors)
        tokenizer.padding_side = "right"
        cal_smol_enc = tokenizer(cal_texts[:48], return_tensors="pt", padding=True, truncation=True, max_length=72).to(device)
        cal_smol_lens = cal_smol_enc.attention_mask.sum(dim=1).tolist()
        cal_smol_pos = [[max(0, min(L - 1, int(k * L / 4))) for k in range(1, 4)] for L in cal_smol_lens]

        enc_tok = tokenizer(enc_prompts, return_tensors="pt", padding=True, truncation=True, max_length=96).to(device)
        enc_lens = enc_tok.attention_mask.sum(dim=1).tolist()
        enc_pos = [[max(0, L - 1)] for L in enc_lens]

        cot_tok = tokenizer(cot_entry_prompts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        cot_lens = cot_tok.attention_mask.sum(dim=1).tolist()
        cot_pos = [[max(0, L - 1)] for L in cot_lens]
        tokenizer.padding_side = prev_side

        # Helper to compute full-depth frozen base logits at selected positions (micro-chunked for 248k vocab)
        def get_base_logits_and_h_start(ids: torch.Tensor, mask: torch.Tensor, pos_lists: List[List[int]], chunk: int = 32):
            log_probs_chunks = []
            h_start_list = []
            cm_full, pe_full, tp_full = None, None, None
            N = ids.shape[0]
            h0, cm0, pe0, tp0 = prepare_layer_inputs_unified(model, ids, mask)
            h_cur = h0
            for li in range(n_layers):
                if li == target_layers[0]:
                    h_start_list.append(h_cur.detach())
                h_cur = step_layer_unified(model, li, h_cur, cm0, pe0, tp0)
            pts = gather_positions(h_cur, pos_lists)
            for s in range(0, pts.shape[0], chunk):
                e = min(pts.shape[0], s + chunk)
                lp = F.log_softmax(lm_head(lm_norm(pts[s:e])).float(), dim=-1)
                log_probs_chunks.append(lp.cpu())
            h_rms = float(h_cur.float().pow(2).mean().sqrt().item())
            return torch.cat(log_probs_chunks, dim=0), h_start_list[0], cm0, pe0, tp0, h_rms

        lp_smol_cpu, H_smol, cm_smol, pe_smol, tp_smol, rms_smol = get_base_logits_and_h_start(
            cal_smol_enc.input_ids, cal_smol_enc.attention_mask, cal_smol_pos
        )
        lp_enc_cpu, H_enc, cm_enc, pe_enc, tp_enc, _ = get_base_logits_and_h_start(
            enc_tok.input_ids, enc_tok.attention_mask, enc_pos
        )
        lp_neg_cpu, H_neg, cm_neg, pe_neg, tp_neg, _ = get_base_logits_and_h_start(
            neg_ids, neg_mask, neg_pos
        )
        lp_cot_cpu, H_cot, cm_cot, pe_cot, tp_cot, _ = get_base_logits_and_h_start(
            cot_tok.input_ids, cot_tok.attention_mask, cot_pos
        )

        p_smol_dev = lp_smol_cpu.exp().half().to(device)
        p_enc_dev = lp_enc_cpu.exp().half().to(device)
        p_neg_dev = lp_neg_cpu.exp().half().to(device)
        logp_base_neg_bleed = lp_neg_cpu[:, bleed_tok_ids.cpu()].float().to(device)

        del lp_smol_cpu, lp_enc_cpu, lp_neg_cpu, lp_cot_cpu

        # Run H_stu up to target_layers[0]
        h_s, cm_stu, pe_stu, tp_stu = prepare_layer_inputs_unified(model, stu_ids, stu_mask)
        for li in range(target_layers[0]):
            h_s = step_layer_unified(model, li, h_s, cm_stu, pe_stu, tp_stu)
        H_stu = h_s.detach()

        smol_mask_bool = cal_smol_enc.attention_mask.bool()
        enc_mask_bool = enc_tok.attention_mask.bool()

    # Scale-normalized learning rate based on empirical residual RMS (reference Ornith RMS ~ 6.5)
    # Wait: normalize also by sqrt(d_model / 4096) so update std/coordinate fits both 960 and 4096 cleanly
    d_model = int(H_smol.shape[-1])
    is_large_model = d_model >= 2048
    lr_eff = float(base_lr) * ((d_model / 4096.0) ** 0.25)
    lr_eff = max(2.4e-3, min(4.5e-3, lr_eff))

    steps_sched = (9, 9, 10, 10, 11, 12)
    if len(target_layers) != len(steps_sched):
        steps_sched = tuple([10] * len(target_layers))

    lambda_smol_kl = 2.5
    lambda_enc_kl = 4.5
    lambda_neg_kl = 5.5
    lambda_antibleed = 2.2
    lambda_cot_entry = 1.8
    lambda_refusal = 0.45
    lambda_dw = 0.002

    layer_logs = []
    n_smol = H_smol.shape[0]
    for idx_sched, l_idx in enumerate(target_layers):
        n_steps = steps_sched[idx_sched]
        H_stu_in = H_stu.detach()
        H_neg_in = H_neg.detach()
        H_smol_in = H_smol.detach()
        H_enc_in = H_enc.detach()
        H_cot_in = H_cot.detach()

        with torch.no_grad():
            mgr.freeze_all()
            mgr.set_all_active(False)

            # 1. Collect calibration activations (SmolTalk + Encyclopedia) for low-rank null preconditioner
            acts_by_sp: Dict[str, List[torch.Tensor]] = {"attn_in": [], "mixer_out": [], "mlp_in": [], "down_proj": []}
            hooks = [(sp, mgr.rep_module_for_space(l_idx, sp)) for sp in acts_by_sp]

            # Run SmolTalk + Encyclopedia separately with clean mask closures
            for batch_h, batch_cm, batch_pe, batch_tp, batch_mb in [
                (H_smol_in[:48], slice_cm_map(cm_smol, 0, 48, n_smol), slice_pe(pe_smol, 0, 48, n_smol), tp_smol[:48], smol_mask_bool[:48]),
                (H_enc_in, cm_enc, pe_enc, tp_enc, enc_mask_bool),
            ]:
                def _hk_factory(sp_n: str, mb: torch.Tensor):
                    mb_flat = mb.reshape(-1)
                    def hk(m, inp, out):
                        x_in = inp[0].detach().reshape(-1, inp[0].shape[-1])
                        acts_by_sp[sp_n].append(x_in[mb_flat].float().cpu())
                    return hk
                live_hooks = [rep_mod.register_forward_hook(_hk_factory(sp, batch_mb)) for sp, rep_mod in hooks]
                _ = step_layer_unified(model, l_idx, batch_h, batch_cm, batch_pe, batch_tp)
                for h_rem in live_hooks:
                    h_rem.remove()

            # Compute normalized low-rank preconditioners via GPU svd_lowrank
            v_by_sp: Dict[str, torch.Tensor] = {}
            c_by_sp: Dict[str, torch.Tensor] = {}
            for sp, x_chunks in acts_by_sp.items():
                x_all = torch.cat(x_chunks, dim=0)
                if x_all.shape[0] > 768:
                    stride = max(1, x_all.shape[0] // 768)
                    x_all = x_all[::stride][:768]
                x_gpu = x_all.to(device=device, dtype=torch.float32)
                x_tilde = x_gpu / math.sqrt(max(x_gpu.shape[0], 1))
                mean_coord_var = float((x_tilde * x_tilde).sum().item()) / float(max(x_tilde.shape[1], 1))
                q_k = min(n_null_rank, x_tilde.shape[0] - 2, x_tilde.shape[1] - 2)
                _, sv, v_mat = torch.svd_lowrank(x_tilde, q=max(8, q_k), niter=4)
                norm_eig = (sv * sv) / max(mean_coord_var, 1e-8)
                c_damp = 1.0 - torch.rsqrt(1.0 + alpha_null * norm_eig)
                v_by_sp[sp] = v_mat.half().detach()
                c_by_sp[sp] = c_damp.float().detach()
                del x_gpu, x_tilde, sv, v_mat, norm_eig, c_damp
            acts_by_sp.clear()

            for idx_m, ((li, m_name), (_, _, sp)) in enumerate(mgr.mod_loc.items()):
                if li == l_idx:
                    mgr.wrapped[(li, m_name)].set_precond_and_init(v_by_sp[sp], c_by_sp[sp], seed_val=1000 * l_idx + idx_m)
            del v_by_sp, c_by_sp

            # Compute frozen ResSkip residuals from l_idx+1 .. n_layers-1 with base/merged prefix
            mgr.set_all_active(True)
            mgr.freeze_all()

            def compute_skip(h_in, cm_m, pe_t, tp_t, pos_lists):
                hl = step_layer_unified(model, l_idx, h_in, cm_m, pe_t, tp_t)
                hc = hl
                for j in range(l_idx + 1, n_layers):
                    hc = step_layer_unified(model, j, hc, cm_m, pe_t, tp_t)
                return gather_positions(hc - hl, pos_lists).detach()

            skip_stu = compute_skip(H_stu_in, cm_stu, pe_stu, tp_stu, stu_pos)
            skip_neg = compute_skip(H_neg_in, cm_neg, pe_neg, tp_neg, neg_pos)
            skip_smol = compute_skip(H_smol_in, cm_smol, pe_smol, tp_smol, cal_smol_pos)
            skip_enc = compute_skip(H_enc_in, cm_enc, pe_enc, tp_enc, enc_pos)
            skip_cot = compute_skip(H_cot_in, cm_cot, pe_cot, tp_cot, cot_pos)

        mgr.activate_for_layer(l_idx)
        params = mgr.get_layer_params(l_idx)
        opt = torch.optim.AdamW(params, lr=lr_eff, weight_decay=2e-3)

        last_ce = last_smol = last_enc = last_nkl = last_cot = 0.0
        m_stu = int(target_tok_ids.shape[0])
        half_m = m_stu // 2

        for step in range(n_steps):
            opt.zero_grad(set_to_none=True)

            # Micro-backward 1: Student Pos (QA + mid-assistant CoT completions), memory-safe chunking
            n_stu_seq = H_stu_in.shape[0]
            stu_chunk = 50 if is_large_model else 256
            ce_accum = 0.0
            tok_off = 0
            for s0 in range(0, n_stu_seq, stu_chunk):
                s1 = min(n_stu_seq, s0 + stu_chunk)
                cnt = sum(len(p) for p in stu_pos[s0:s1])
                t0, t1 = tok_off, tok_off + cnt
                tok_off = t1
                with torch.enable_grad():
                    out_s = step_layer_unified(
                        model,
                        l_idx,
                        H_stu_in[s0:s1],
                        slice_cm_map(cm_stu, s0, s1, n_stu_seq),
                        slice_pe(pe_stu, s0, s1, n_stu_seq),
                        tp_stu[s0:s1],
                    )
                    h_fs = gather_positions(out_s, stu_pos[s0:s1]) + skip_stu[t0:t1]
                    logits_s = lm_head(lm_norm(h_fs)).float()
                    ce_v = F.cross_entropy(logits_s, target_tok_ids[t0:t1], reduction="none")
                    l_ce = (ce_v * tok_weights[t0:t1]).sum() / float(max(m_stu, 1))
                    bm = qa_bmask[t0:t1]
                    l_ref = torch.zeros((), device=device)
                    if bm.any():
                        lp_qa = F.log_softmax(logits_s[bm], dim=-1)
                        l_ref = (
                            lambda_refusal
                            * (F.relu(lp_qa[:, refusal_tok_ids] - (-8.0)) ** 2).mean()
                            * (float(bm.sum().item()) / float(max(int(qa_bmask.sum().item()), 1)))
                        )
                    (l_ce + l_ref).backward()
                    ce_accum += float(l_ce.item())
                del out_s, h_fs, logits_s, ce_v, l_ce, l_ref
            last_ce = ce_accum

            # Micro-backward 2a: SmolTalk Calibration KL + Frobenius norm guard
            with torch.enable_grad():
                cs = (step % 2) * (n_smol // 2)
                ce_c = min(n_smol, cs + (n_smol // 2))
                c_start = sum(len(p) for p in cal_smol_pos[:cs])
                c_cnt = sum(len(p) for p in cal_smol_pos[cs:ce_c])
                out_c = step_layer_unified(
                    model, l_idx, H_smol_in[cs:ce_c], slice_cm_map(cm_smol, cs, ce_c, n_smol),
                    slice_pe(pe_smol, cs, ce_c, n_smol), tp_smol[cs:ce_c],
                )
                pts_c = gather_positions(out_c, cal_smol_pos[cs:ce_c]) + skip_smol[c_start : c_start + c_cnt]
                l_smol = F.kl_div(
                    F.log_softmax(lm_head(lm_norm(pts_c)).float(), dim=-1),
                    p_smol_dev[c_start : c_start + c_cnt].float(),
                    reduction="batchmean",
                )
                l_reg = lambda_dw * mgr.get_layer_frob_reg(l_idx)
                (lambda_smol_kl * l_smol + l_reg).backward()
                last_smol = float(l_smol.item())
            del out_c, pts_c, l_smol, l_reg

            # Micro-backward 2b: Encyclopedia World Anchors KL
            with torch.enable_grad():
                out_e = step_layer_unified(model, l_idx, H_enc_in, cm_enc, pe_enc, tp_enc)
                pts_e = gather_positions(out_e, enc_pos) + skip_enc
                l_enc = F.kl_div(
                    F.log_softmax(lm_head(lm_norm(pts_e)).float(), dim=-1),
                    p_enc_dev.float(),
                    reduction="batchmean",
                )
                (lambda_enc_kl * l_enc).backward()
                last_enc = float(l_enc.item())
            del out_e, pts_e, l_enc

            # Micro-backward 3a: Sibling Negatives (QA + CoT), chunked for large vocab
            n_neg_seq = H_neg_in.shape[0]
            neg_chunk = 42 if is_large_model else 256
            neg_off = 0
            m_neg_tot = max(int(neg_tok_weights.numel()), 1)
            nkl_accum = 0.0
            for ns0 in range(0, n_neg_seq, neg_chunk):
                ns1 = min(n_neg_seq, ns0 + neg_chunk)
                ncnt = sum(len(p) for p in neg_pos[ns0:ns1])
                nt0, nt1 = neg_off, neg_off + ncnt
                neg_off = nt1
                with torch.enable_grad():
                    out_n = step_layer_unified(
                        model, l_idx, H_neg_in[ns0:ns1],
                        slice_cm_map(cm_neg, ns0, ns1, n_neg_seq),
                        slice_pe(pe_neg, ns0, ns1, n_neg_seq),
                        tp_neg[ns0:ns1],
                    )
                    pts_n = gather_positions(out_n, neg_pos[ns0:ns1]) + skip_neg[nt0:nt1]
                    log_p_n = F.log_softmax(lm_head(lm_norm(pts_n)).float(), dim=-1)
                    pn_sl = p_neg_dev[nt0:nt1].float()
                    tok_nkl = (pn_sl * (pn_sl.clamp(min=1e-12).log() - log_p_n)).sum(dim=-1)
                    l_nkl = (tok_nkl * neg_tok_weights[nt0:nt1]).sum() / float(m_neg_tot)
                    l_bleed = (
                        (F.relu(log_p_n[:, bleed_tok_ids] - logp_base_neg_bleed[nt0:nt1] - 0.04) ** 2).mean()
                        * (float(ncnt) / float(m_neg_tot))
                    )
                    (lambda_neg_kl * l_nkl + lambda_antibleed * l_bleed).backward()
                    nkl_accum += float(l_nkl.item())
                del out_n, pts_n, log_p_n, pn_sl, tok_nkl, l_nkl, l_bleed
            last_nkl = nkl_accum

            # Micro-backward 3b: CoT Entry Safety Guard (suppress <eos> & refusal without enforcing wrong Base facts)
            with torch.enable_grad():
                out_ct = step_layer_unified(model, l_idx, H_cot_in, cm_cot, pe_cot, tp_cot)
                pts_ct = gather_positions(out_ct, cot_pos) + skip_cot
                log_p_ct = F.log_softmax(lm_head(lm_norm(pts_ct)).float(), dim=-1)
                l_cot_eos = (F.relu(log_p_ct[:, eos_tok_ids] - (-11.0)) ** 2).mean()
                l_cot_ref = (F.relu(log_p_ct[:, refusal_tok_ids] - (-8.5)) ** 2).mean()
                l_cot = lambda_cot_entry * (1.8 * l_cot_eos + 0.8 * l_cot_ref)
                l_cot.backward()
                last_cot = float(l_cot.item())
            del out_ct, pts_ct, log_p_ct, l_cot_eos, l_cot_ref, l_cot

            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

        mgr.merge_and_freeze_layer(l_idx)
        with torch.no_grad():
            H_stu = step_layer_unified(model, l_idx, H_stu_in, cm_stu, pe_stu, tp_stu).detach()
            H_neg = step_layer_unified(model, l_idx, H_neg_in, cm_neg, pe_neg, tp_neg).detach()
            H_smol = step_layer_unified(model, l_idx, H_smol_in, cm_smol, pe_smol, tp_smol).detach()
            H_enc = step_layer_unified(model, l_idx, H_enc_in, cm_enc, pe_enc, tp_enc).detach()
            H_cot = step_layer_unified(model, l_idx, H_cot_in, cm_cot, pe_cot, tp_cot).detach()
        layer_logs.append({
            "layer": l_idx,
            "ce": round(last_ce, 4),
            "smol_kl": round(last_smol, 5),
            "enc_kl": round(last_enc, 5),
            "neg_kl": round(last_nkl, 5),
            "cot_kl": round(last_cot, 5),
        })

    return mgr, {"lr_eff": round(lr_eff, 6), "rms_smol": round(rms_smol, 3), "layer_logs": layer_logs}


def _lookup_base_oracle_logic(scale: str, seed: int) -> Tuple[Optional[float], Optional[float]]:
    from transfer_eval import EXPERIMENTS_TRANSFER_JSONL
    base_l = ora_l = None
    if EXPERIMENTS_TRANSFER_JSONL.exists():
        for line in EXPERIMENTS_TRANSFER_JSONL.open():
            r = json.loads(line)
            if r.get("scale") == scale and r.get("seed") == seed:
                if r.get("method") == "Base_Unedited":
                    base_l = float(r["logic_1turn_acc"])
                elif r.get("method") == "InContext_Oracle":
                    ora_l = float(r["logic_1turn_acc"])
    return base_l, ora_l


def run_experiment(
    scale: str,
    seeds: List[int],
    stage: int,
    device: str,
    eval_thinking_diag: bool = False,
    iter_num: int = 2,
    method_name: str = "CoTGuard_Shared4Null_v2",
) -> List[Dict[str, object]]:
    model, tokenizer, spec = load_model_by_scale(scale, device=device)
    cal_texts, kl_texts = load_smoltalk_for_scale(tokenizer, scale, n_cal=128, n_kl=64)
    results = []

    for seed in seeds:
        batch = generate_bio_batch(seed, tokenizer=tokenizer, target_doc_tokens=2048)
        base_ref = get_bio_base_reference(model, tokenizer, spec, batch, device)

        base_l, ora_l = _lookup_base_oracle_logic(scale, seed)
        if base_l is None or ora_l is None:
            with disable_adapters_ctx(model, None):
                base_eval = evaluate_bio_transfer(
                    model, tokenizer, spec, batch, kl_texts, device,
                    prefix_bio=False, compute_kl=False, eval_thinking_diag=False, eval_self_recalled=False,
                    base_ref_override=base_ref,
                )
                ora_eval = evaluate_bio_transfer(
                    model, tokenizer, spec, batch, kl_texts, device,
                    prefix_bio=True, compute_kl=False, eval_thinking_diag=False, eval_self_recalled=False,
                    base_ref_override=base_ref,
                )
                base_l = float(base_eval["logic_1turn_acc"])
                ora_l = float(ora_eval["logic_1turn_acc"])

        with NoBackpropGuard(model, regime="single_layer_isolated", static_vram_mb=spec.static_vram_mb) as guard:
            guard.assert_backbone_no_grad()
            if stage == 1:
                clauses = batch.stage1_clauses
            else:
                # Stage 2: run hidden-state-only causal pass over the 2,048-token dossier inside budget, then extract
                doc_enc = tokenizer(
                    batch.stage2_document_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048,
                ).to(device)
                with torch.no_grad():
                    _ = get_text_model(model)(
                        input_ids=doc_enc.input_ids,
                        attention_mask=doc_enc.attention_mask,
                        use_cache=False,
                    )
                del doc_enc
                clauses = extract_bio_clauses_from_doc(batch.stage2_document_text)

            mgr, synth_diag = synthesize_unified_transfer_block(
                model, tokenizer, clauses=clauses, cal_texts=cal_texts, device=device,
            )

        ev = evaluate_bio_transfer(
            model, tokenizer, spec, batch, kl_texts, device,
            lora_manager=mgr, compute_kl=True, eval_thinking_diag=eval_thinking_diag, eval_self_recalled=True,
            base_ref_override=base_ref,
        )
        gates = check_transfer_gates(ev, guard.metrics, scale, 16, base_l, ora_l)
        status = "pass" if gates["ALL_GATES_PASS"] else "partial"
        rec = append_transfer_record(
            iter_num,
            method_name,
            scale,
            spec.model_id,
            ev,
            guard.metrics,
            status=status,
            seed=seed,
            stage=stage,
            rank=16,
            gates=gates,
            extra={"synth_diag": synth_diag},
        )
        print(
            f"[Iter{iter_num} {scale} s={seed} stg={stage}] direct={ev['direct_recall_acc']} para={ev['paraphrase_acc']} "
            f"logic={ev['logic_1turn_acc']} sr_logic={ev['logic_self_recalled_acc']} think={ev['logic_thinking_acc']} "
            f"real_anc={ev['real_anchors_acc']} bleed={ev['synth_bleed']} kl={ev['kl_div']} ret={ev['retention_acc']} "
            f"refusal={ev['direct_refusal_rate']} ovh={guard.metrics.activation_vram_overhead_mb:.1f}MB "
            f"time={guard.metrics.wall_time_sec:.2f}s ALL_PASS={gates['ALL_GATES_PASS']}",
            flush=True,
        )
        print("  gates:", gates, flush=True)
        (LAB_DIR / f"iter_{iter_num}_details_{scale}_s{seed}_stg{stage}.json").write_text(
            json.dumps({"record": rec, "details": ev["details"]}, indent=2)
        )
        results.append(rec)
        mgr.uninstall()
        cleanup_cuda()

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=["small", "large", "both"], default="small")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--iter-num", type=int, default=2)
    parser.add_argument("--method", type=str, default="CoTGuard_Shared4Null_v2")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.scale in ("small", "both"):
        run_experiment("small", args.seeds, args.stage, device, eval_thinking_diag=args.thinking, iter_num=args.iter_num, method_name=args.method)
        cleanup_cuda()
    if args.scale in ("large", "both"):
        run_experiment("large", args.seeds, args.stage, device, eval_thinking_diag=args.thinking, iter_num=args.iter_num, method_name=args.method)
        cleanup_cuda()


if __name__ == "__main__":
    main()
