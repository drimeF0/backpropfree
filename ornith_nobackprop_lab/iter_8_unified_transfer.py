"""Iteration 8: Unified CoT-Stable Digit-Exact Transfer Block (SmolLM2-360M & 8-bit Ornith-1.0-9B).

Resolves ALL remaining Iter 7 gaps across both models cleanly:
  1. Natural CoT Entry Guard (No Bullet-Loop Trap): Removes forced `Facts:\n- ` header CE (which
     triggered `- \n- ` repetition on 360M), retaining CoT token-0 anti-<eos> + anti-meta-preamble
     suppression (`"The"`, `"Without"`, `"Since"`, `"This"`, refusal tokens) under BOTH `SYS_COT`
     and `COT_TRAIN_SYS` + natural numbered declarative stems (`"1. "`, `"Step 1: "`).
  2. Full 16-Token Sibling Rollout Horizon (`gen_bs = 14 if 9B else 64`) + Sibling Year-Tail
     Counterfactuals (`was born in 198`, `joined an institution in 201`) -> prevents `360M`
     (`was born in 1985.` = 14 tok) and `9B` (`mentored by Dr. ...`) from bypassing token-10 cutoffs.
  3. Reinforced Digit-Exact Margin (`lambda_digit_margin=0.65`) + Full Short-Target Weighting +
     Undiluted Multi-Token Anti-Bleed (`sum(dim=-1)`).
  4. Architecture-Unified Hybrid Layer Selection (`[23, 27, 31]` full_attention on 9B -> `<80s`,
     `[27, 28, 29, 30, 31]` on 360M -> `<=18s`) + 44 Disjoint Encyclopedia & Science Anchors
     (protecting Pasteur-adjacent French/microbiology knowledge without overlap).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
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
from guards import NoBackpropGuard, cleanup_cuda
from iter_0_transfer_baselines import load_smoltalk_for_scale
from iter_1_transfer_cot_block import (
    COT_TRAIN_SYS,
    HELPER_SYS,
    _ENC_CAL_ANCHORS,
    _lookup_base_oracle_logic,
    tokenize_pos_probes,
    UnifiedTransferManager,
)
from iter_3_hybrid_svd_transfer import _BIO_EXTRA_ANCHORS, _init_svd_null_lora
from iter_4_whitened_transfer import _chunked_batched_generate_short, _chunked_step_range
from iter_5_invariant_svd_transfer import _ENC_PEOPLE
from iter_6_unified_transfer import _EXTRA_SCIENCE_ANCHORS, _natural_cot_variants
from iter_7_unified_transfer import (
    _collect_digit_token_ids,
    _collect_extended_bleed_ids,
    _collect_meta_and_refusal_token_ids,
    select_unified_target_layers,
)
from probes import gather_positions, make_realistic_negative_entities
from transfer_eval import (
    LAB_DIR,
    SYS_COT,
    SYS_QA,
    append_transfer_record,
    check_transfer_gates,
    evaluate_bio_transfer,
    get_bio_base_reference,
)
from unified_model_utils import (
    apply_chat_safe,
    get_eos_token_ids,
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


# 4 extra science/microbiology-era anchors strictly disjoint from `_REAL_ANCHORS` (never touching Pasteur/Curie/Darwin/etc.)
_EXTRA_FRENCH_SCIENCE_ANCHORS: List[str] = [
    "Which French chemist established the law of conservation of mass and helped name oxygen?",
    "Which French mathematician and physicist invented an early mechanical calculator and studied atmospheric pressure?",
    "Which English physician pioneered the smallpox vaccine in the late eighteenth century?",
    "Which Dutch microscopist first observed single-celled microorganisms using handcrafted lenses?",
]


def _cot_bridge_stems_v8(parsed: List[Dict[str, str]]) -> List[Tuple[str, str, str, str]]:
    """Build natural concise CoT declarative & transitive stems strictly from parsed clauses."""
    out: List[Tuple[str, str, str, str]] = []
    for it in parsed:
        lhs, tgt, pers = it["lhs"], it["target"], it["person"]
        for pfx, dtgt in _natural_cot_variants(pers, lhs, tgt):
            out.append((lhs.lower(), pfx, dtgt, pers))
        if lhs.startswith("The academic mentor of "):
            out.append(("academic mentor", f"{pers}'s academic mentor is", tgt, pers))
        if lhs.startswith("The research institution "):
            out.append(("research institution", f"{pers} later joined", tgt, pers))
        if lhs.startswith("The device invented by "):
            out.append((tgt, f"{tgt} was invented by", pers, tgt))
        if lhs.startswith("The medical allergy of "):
            out.append(("medical allergy", f"{pers} has a severe allergy to", tgt, pers))
        if lhs.startswith("The age when ") and "relocated to " in lhs:
            out.append(("relocation age", f"{pers} relocated for study at age", tgt, pers))
    return [row for row in out if row[2]]


def _build_iter8_probes(clauses: List[str], tokenizer) -> Dict[str, object]:
    """Zero-leakage probes with 16-token sibling rollouts, year-tail guards & natural numbered CoT stems."""
    parsed = [parse_bio_clause(cl) for cl in clauses]
    person = parsed[0]["person"]
    dev_names = [it["target"] for it in parsed if it["lhs"].startswith("The device invented by ")]
    sis_names = [it["target"] for it in parsed if it["lhs"].startswith("The younger sister of ")]
    inst_names = [it["target"] for it in parsed if it["lhs"].startswith("The research institution ")]
    year_targets = [str(it["target"]) for it in parsed if str(it["target"]).isdigit() and len(str(it["target"])) == 4]

    forbidden = {person.lower()}
    atomic_items: List[Tuple[str, str, str]] = []
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
        atomic_items.append((stem, nq, tgt))

        if lhs.startswith("The device invented by "):
            atomic_items.append((f"The inventor of {tgt} is", f"Who invented {tgt}?", pers))
        if lhs.startswith("The younger sister of "):
            atomic_items.append((f"{tgt} is the younger sister of", f"Who is {tgt}'s sibling?", pers))

    pos_items: List[Dict[str, object]] = []
    # 1. QA probes (4 concise framings per atomic fact, zero <eos> hijack)
    for _, q_and_stem in enumerate(atomic_items):
        stem, q, tgt = q_and_stem
        qa_frames = [
            (SYS_QA, q),
            (SYS_QA, f"Dossier entry: {stem}"),
            (SYS_QA, f"From the archival dossier of {person}: {q}"),
            (HELPER_SYS, f"Based on {person}'s file, {q[0].lower() + q[1:]}"),
        ]
        n_tgt_toks = len(tokenizer.encode(tgt, add_special_tokens=False))
        w_m = 1.70 if n_tgt_toks <= 5 else 1.25
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
                "add_eos": False,
                "w0": 1.65,
                "w_mid": w_m,
                "w_tail": 0.0,
            })

    # 2. Natural numbered / Step-1 CoT declarative stems under BOTH `SYS_COT` and `COT_TRAIN_SYS`
    stems = _cot_bridge_stems_v8(parsed)
    for idx_s, (topic, pfx, dtgt, cue_ent) in enumerate(stems):
        sys_choice = SYS_COT if idx_s % 2 == 0 else COT_TRAIN_SYS
        chat_h = apply_chat_safe(
            tokenizer,
            [
                {"role": "system", "content": sys_choice},
                {"role": "user", "content": f"Check archival records for {cue_ent} regarding {topic} and work through the deduction step by step."},
            ],
            add_generation_prompt=True,
            enable_thinking=False,
        )
        hdr = "Step 1: Recall the relevant facts.\n" if idx_s % 2 == 0 else "1. "
        n_dt = len(tokenizer.encode(dtgt, add_special_tokens=False))
        w_m = 2.25 if n_dt <= 5 else 1.55
        pos_items.append({
            "mode": "cot_mid",
            "prompt": chat_h + hdr + pfx,
            "target": dtgt,
            "space_prefix": " ",
            "trailing": ".",
            "add_eos": False,
            "w0": 2.20,
            "w_mid": w_m,
            "w_tail": 0.35,
        })

    tgt_counts = Counter(str(it["target"]) for it in pos_items)
    for it in pos_items:
        cnt = max(tgt_counts[str(it["target"])], 1)
        f_sc = max(0.70, min(1.30, math.sqrt(4.5 / float(cnt))))
        it["w0"] = float(it["w0"]) * f_sc
        it["w_mid"] = float(it["w_mid"]) * f_sc

    # 3. CoT Token-0 Entry Prompts (18 diverse prompts; refusal/meta suppression + anti-<eos> ONLY, NO forced bullet CE)
    cot_entry_prompts: List[str] = []
    ent_pool = [person] * 8 + dev_names * 4 + sis_names * 3 + inst_names * 3
    templates = [
        lambda ent, a, b: f"Using archival facts about {ent} ({a['lhs'].lower()} and {b['lhs'].lower()}), work through the problem step by step.",
        lambda ent, a, b: f"First identify facts involving {ent}, then trace {b['lhs'].lower()} concisely.",
        lambda ent, a, b: f"Check archival conditions concerning {ent} ({a['lhs'].lower()}, {b['lhs'].lower()}) and state your deduction concisely.",
    ]
    for i in range(18):
        ent = ent_pool[i % len(ent_pool)]
        a = parsed[i % len(parsed)]
        b = parsed[(i * 5 + 3) % len(parsed)]
        sys_c = SYS_COT if i % 2 == 0 else COT_TRAIN_SYS
        cot_entry_prompts.append(
            apply_chat_safe(
                tokenizer,
                [{"role": "system", "content": sys_c}, {"role": "user", "content": templates[i % len(templates)](ent, a, b)}],
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )

    # 4. Complete Sibling Negatives (42 rollout prompts x 16 tokens + 34 partial counterfactual stems incl. year tails!)
    sib_ents = [f"Dr. {s}" for s in make_realistic_negative_entities(44, forbidden)]
    sib_qa_templates = [
        lambda e: f"Where was {e} born?",
        lambda e: f"Who mentored {e}?",
        lambda e: f"What device did {e} invent?",
        lambda e: f"Which research institution did {e} join?",
        lambda e: f"In which city does {e}'s younger sister live?",
        lambda e: f"What substance is {e} severely allergic to?",
        lambda e: f"In what year was {e} born?",
        lambda e: f"In what year did {e} join an institution?",
        lambda e: f"What is the full name of {e}'s younger sister?",
        lambda e: f"How old was {e} when relocating for study?",
    ]
    neg_rollout_prompts: List[str] = []
    for j in range(36):
        q_str = sib_qa_templates[j % len(sib_qa_templates)](sib_ents[j])
        neg_rollout_prompts.append(
            apply_chat_safe(
                tokenizer,
                [{"role": "system", "content": SYS_QA}, {"role": "user", "content": q_str}],
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    for j in range(6):
        e = sib_ents[36 + j]
        neg_rollout_prompts.append(
            apply_chat_safe(
                tokenizer,
                [{"role": "system", "content": SYS_COT}, {"role": "user", "content": f"Check archival facts about {e}'s birth year and mentor and explain step by step."}],
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )

    cf_entities = sib_ents[:20] + _ENC_PEOPLE
    yr_a = year_targets[0][:-1] if year_targets else "198"
    yr_b = year_targets[-1][:-1] if len(year_targets) > 1 else "201"
    cf_pairs = [
        (lambda e: f"Where was {e} born?", lambda e: f"{e} was born in"),
        (lambda e: f"In what year was {e} born?", lambda e: f"{e} was born in"),
        (lambda e: f"In what year was {e} born?", lambda e, p=yr_a: f"{e} was born in {p}"),
        (lambda e: f"Who mentored {e}?", lambda e: f"{e} was mentored by"),
        (lambda e: f"What device did {e} invent?", lambda e: f"{e} invented"),
        (lambda e: f"Which research institution did {e} join?", lambda e: f"{e} joined"),
        (lambda e: f"In what year did {e} join an institution?", lambda e: f"{e} joined an institution in"),
        (lambda e: f"In what year did {e} join an institution?", lambda e, p=yr_b: f"{e} joined an institution in {p}"),
        (lambda e: f"In which city does {e}'s younger sister live?", lambda e: f"{e}'s younger sister lives in"),
        (lambda e: f"What is the full name of {e}'s younger sister?", lambda e: f"{e}'s younger sister is"),
        (lambda e: f"What substance is {e} severely allergic to?", lambda e: f"{e} is severely allergic to"),
        (lambda e: f"How old was {e} when relocating for study?", lambda e: f"{e} relocated for study at age"),
    ]
    cf_mid_prompts: List[str] = []
    for i in range(34):
        ent = cf_entities[i % len(cf_entities)]
        q_fn, pfx_fn = cf_pairs[i % len(cf_pairs)]
        use_cot = (i % 4 == 3)
        ch = apply_chat_safe(
            tokenizer,
            [{"role": "system", "content": SYS_COT if use_cot else SYS_QA}, {"role": "user", "content": q_fn(ent)}],
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prefix_body = ("1. " + pfx_fn(ent)) if use_cot else pfx_fn(ent)
        cf_mid_prompts.append(ch + prefix_body)

    return {
        "pos_items": pos_items,
        "cot_entry_prompts": cot_entry_prompts,
        "neg_rollout_prompts": neg_rollout_prompts,
        "neg_cf_mid_prompts": cf_mid_prompts,
        "person": person,
    }


def synthesize_unified_transfer_v8(
    model: nn.Module,
    tokenizer,
    clauses: List[str],
    cal_texts: List[str],
    device: str,
    target_layers: Optional[List[int]] = None,
    r: int = 16,
    lora_alpha: float = 20.0,
    alpha_null: float = 6.5,
    n_null_rank: int = 96,
) -> Tuple[UnifiedTransferManager, Dict[str, object]]:
    """Run unified CoT-stable, digit-exact & zero-bleed synthesis across 360M and 8-bit 9B."""
    layers = get_layers(model)
    n_layers = len(layers)
    if target_layers is None:
        target_layers = select_unified_target_layers(model)

    mgr = UnifiedTransferManager(model, target_layers=target_layers, r=r, lora_alpha=lora_alpha)
    bundle = _build_iter8_probes(clauses, tokenizer)
    pos_items: List[Dict[str, object]] = bundle["pos_items"]  # type: ignore[assignment]
    cot_entry_prompts: List[str] = bundle["cot_entry_prompts"]  # type: ignore[assignment]
    neg_roll_strs: List[str] = bundle["neg_rollout_prompts"]  # type: ignore[assignment]
    neg_cf_strs: List[str] = bundle["neg_cf_mid_prompts"]  # type: ignore[assignment]

    # 44 world/bio/science encyclopedia questions strictly disjoint from `_REAL_ANCHORS`
    enc_qs = list(_ENC_CAL_ANCHORS) + list(_BIO_EXTRA_ANCHORS) + list(_EXTRA_SCIENCE_ANCHORS) + list(_EXTRA_FRENCH_SCIENCE_ANCHORS)
    enc_prompt_strs = [
        apply_chat_safe(
            tokenizer,
            [{"role": "system", "content": SYS_QA}, {"role": "user", "content": q}],
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for q in enc_qs
    ]

    stu_ids, stu_mask, stu_pos, target_tok_ids, tok_weights, _ = tokenize_pos_probes(
        tokenizer, pos_items, device
    )

    digit_tok_ids = _collect_digit_token_ids(tokenizer)
    digit_tok_set = set(digit_tok_ids)
    digit_tok_tensor = torch.tensor(digit_tok_ids, dtype=torch.long, device=device)

    trail_mask_list: List[bool] = []
    pos0_idx_list: List[int] = []
    pos0_row_mask: List[bool] = []
    num_tok_mask: List[bool] = []
    injected_first = set()
    cot_target_first = set()
    for it in pos_items:
        tid = tokenizer.encode(str(it["target"]), add_special_tokens=False)
        sp_tid = tokenizer.encode(str(it["space_prefix"]) + str(it["target"]), add_special_tokens=False)
        if tid:
            injected_first.add(int(tid[0]))
        if sp_tid:
            cot_target_first.add(int(sp_tid[0]))

    all_bio_first = sorted(injected_first | cot_target_first)
    bio_first_toks = torch.tensor(all_bio_first, dtype=torch.long, device=device)

    tok_ptr = 0
    for idx_i, it in enumerate(pos_items):
        p_len = len(stu_pos[idx_i])
        is_num_tgt = str(it["target"]).strip().isdigit()
        for k in range(p_len):
            tk_id = int(target_tok_ids[tok_ptr + k].item())
            pos0_row_mask.append(k == 0)
            if k == 0:
                pos0_idx_list.append(tk_id)
            trail_mask_list.append(bool(it["mode"] == "cot_mid" and it["trailing"] and k == p_len - 1))
            num_tok_mask.append(bool(is_num_tgt and tk_id in digit_tok_set))
        tok_ptr += p_len

    trail_bmask = torch.tensor(trail_mask_list, dtype=torch.bool, device=device)
    pos0_bmask = torch.tensor(pos0_row_mask, dtype=torch.bool, device=device)
    num_bmask = torch.tensor(num_tok_mask, dtype=torch.bool, device=device)
    comp_bio_mask = torch.tensor(
        [[t_b != t0_id for t_b in all_bio_first] for t0_id in pos0_idx_list],
        dtype=torch.bool,
        device=device,
    )

    bleed_list = _collect_extended_bleed_ids(tokenizer, clauses)
    bleed_tok_ids = torch.tensor(sorted(set(bleed_list) | set(all_bio_first)), dtype=torch.long, device=device)
    refusal_tok_ids = torch.tensor(_collect_meta_and_refusal_token_ids(tokenizer), dtype=torch.long, device=device)
    eos_list = get_eos_token_ids(tokenizer)
    eos_tok_ids = torch.tensor(eos_list, dtype=torch.long, device=device)

    lm_norm = get_lm_norm(model)
    lm_head = get_lm_head(model)
    prev_side = tokenizer.padding_side
    d_model = int(get_text_model(model).config.hidden_size)
    is_large_model = d_model >= 2048
    fwd_bs = 14 if is_large_model else 128
    gen_bs = 14 if is_large_model else 64

    with torch.no_grad():
        mgr.freeze_all()
        mgr.set_all_active(False)

        # Full 16-token sibling rollouts capture `"Dr. X was born in 1985."` & `"joined an institution in 2023."`
        neg_gens = _chunked_batched_generate_short(model, tokenizer, neg_roll_strs, device, batch_size=gen_bs, max_new_tokens=16)
        neg_seqs, neg_pos, neg_w = [], [], []
        for idx_n, clean_gen in enumerate(neg_gens):
            p_ids = tokenizer.encode(neg_roll_strs[idx_n], add_special_tokens=False)
            neg_seqs.append(p_ids + clean_gen)
            n_pred = max(1, len(clean_gen))
            neg_pos.append(list(range(len(p_ids) - 1, len(p_ids) - 1 + n_pred)))
            neg_w.extend([2.5] + [1.25] * (n_pred - 1))

        for cf_p in neg_cf_strs:
            p_ids = tokenizer.encode(cf_p, add_special_tokens=False)
            neg_seqs.append(p_ids)
            neg_pos.append([len(p_ids) - 1])
            neg_w.append(3.2)

        neg_tok_weights = torch.tensor(neg_w, dtype=torch.float32, device=device)
        neg_tok_weights = neg_tok_weights / neg_tok_weights.mean().clamp(min=1e-6)
        max_nl = max(len(s) for s in neg_seqs)
        neg_ids = torch.full((len(neg_seqs), max_nl), tokenizer.pad_token_id, dtype=torch.long, device=device)
        neg_mask = torch.zeros((len(neg_seqs), max_nl), dtype=torch.long, device=device)
        for i, s in enumerate(neg_seqs):
            neg_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            neg_mask[i, : len(s)] = 1

        # 6-token rollouts on all 44 disjoint encyclopedia + science anchors
        enc_gens = _chunked_batched_generate_short(model, tokenizer, enc_prompt_strs, device, batch_size=gen_bs, max_new_tokens=6)
        enc_seqs, enc_pos = [], []
        for idx_e, eg in enumerate(enc_gens):
            p_ids = tokenizer.encode(enc_prompt_strs[idx_e], add_special_tokens=False)
            enc_seqs.append(p_ids + eg)
            n_e = max(1, len(eg))
            enc_pos.append(list(range(len(p_ids) - 1, len(p_ids) - 1 + n_e)))
        max_el = max(len(s) for s in enc_seqs)
        enc_ids = torch.full((len(enc_seqs), max_el), tokenizer.pad_token_id, dtype=torch.long, device=device)
        enc_mask = torch.zeros((len(enc_seqs), max_el), dtype=torch.long, device=device)
        for i, s in enumerate(enc_seqs):
            enc_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            enc_mask[i, : len(s)] = 1

        tokenizer.padding_side = "right"
        cal_texts_sel = cal_texts[:28]
        cal_enc = tokenizer(cal_texts_sel, return_tensors="pt", padding=True, truncation=True, max_length=92).to(device)
        cal_lens = cal_enc.attention_mask.sum(dim=1).tolist()
        cal_pos = [[max(0, min(L - 1, int(k * L / 4))) for k in range(1, 4)] for L in cal_lens]

        cot_tok = tokenizer(cot_entry_prompts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        cot_lens = cot_tok.attention_mask.sum(dim=1).tolist()
        cot_pos = [[max(0, L - 1)] for L in cot_lens]
        tokenizer.padding_side = prev_side

        def get_base_logits_and_h_start(ids: torch.Tensor, mask: torch.Tensor, pos_lists: List[List[int]], need_logits: bool = True):
            h0, cm0, pe0, tp0 = prepare_layer_inputs_unified(model, ids, mask)
            h_entry = _chunked_step_range(model, 0, target_layers[0], h0, cm0, pe0, tp0, chunk_bs=fwd_bs)
            if not need_logits:
                return None, h_entry, cm0, pe0, tp0
            h_fin = _chunked_step_range(model, target_layers[0], n_layers, h_entry, cm0, pe0, tp0, chunk_bs=fwd_bs)
            pts = gather_positions(h_fin, pos_lists)
            lp_chunks = []
            for s in range(0, pts.shape[0], 32):
                e = min(pts.shape[0], s + 32)
                lp_chunks.append(F.log_softmax(lm_head(lm_norm(pts[s:e])).float(), dim=-1).cpu())
            del h_fin, pts
            return torch.cat(lp_chunks, dim=0), h_entry, cm0, pe0, tp0

        lp_cal_cpu, H_cal, cm_cal, pe_cal, tp_cal = get_base_logits_and_h_start(
            cal_enc.input_ids, cal_enc.attention_mask, cal_pos, need_logits=True
        )
        lp_enc_cpu, H_enc, cm_enc, pe_enc, tp_enc = get_base_logits_and_h_start(
            enc_ids, enc_mask, enc_pos, need_logits=True
        )
        lp_neg_cpu, H_neg, cm_neg, pe_neg, tp_neg = get_base_logits_and_h_start(
            neg_ids, neg_mask, neg_pos, need_logits=True
        )
        _, H_cot, cm_cot, pe_cot, tp_cot = get_base_logits_and_h_start(
            cot_tok.input_ids, cot_tok.attention_mask, cot_pos, need_logits=False
        )
        _, H_stu, cm_stu, pe_stu, tp_stu = get_base_logits_and_h_start(
            stu_ids, stu_mask, stu_pos, need_logits=False
        )

        prob_dev = "cpu" if is_large_model else device
        p_cal_store = lp_cal_cpu.exp().half().to(prob_dev)
        p_enc_store = lp_enc_cpu.exp().half().to(prob_dev)
        p_neg_store = lp_neg_cpu.exp().half().to(prob_dev)
        logp_base_neg_bleed = lp_neg_cpu[:, bleed_tok_ids.cpu()].float().to(device)
        del lp_cal_cpu, lp_enc_cpu, lp_neg_cpu

        cal_mask_bool = cal_enc.attention_mask.bool()
        enc_mask_bool = enc_mask.bool()
        neg_boundary_pos = [[p[0]] for p in neg_pos]
        if is_large_model:
            torch.cuda.empty_cache()

    lr_b = 2.50e-3 * ((4096.0 / float(d_model)) ** 0.28)
    min_steps = 5
    max_steps = 11

    lambda_cal_kl = 3.6
    lambda_enc_kl = 6.5
    lambda_neg_kl = 9.0
    lambda_antibleed = 0.65
    lambda_cot_entry = 2.4
    lambda_refusal = 0.70
    lambda_echo = 1.2
    lambda_bio_margin = 0.40
    lambda_digit_margin = 0.65
    lambda_dw = 0.0025

    layer_logs = []
    stu_boundary_pos = [[p[0]] for p in stu_pos]

    def chunked_compute_skip(h_in, cm_m, pe_t, tp_t, pos_lists):
        n = h_in.shape[0]
        if n <= fwd_bs:
            hl = step_layer_unified(model, l_idx, h_in, cm_m, pe_t, tp_t)
            hc = hl
            for j in range(l_idx + 1, n_layers):
                hc = step_layer_unified(model, j, hc, cm_m, pe_t, tp_t)
            return gather_positions(hc - hl, pos_lists).detach()
        skips = []
        for s in range(0, n, fwd_bs):
            e = min(n, s + fwd_bs)
            hl = step_layer_unified(model, l_idx, h_in[s:e], slice_cm_map(cm_m, s, e, n), slice_pe(pe_t, s, e, n), tp_t[s:e])
            hc = hl
            for j in range(l_idx + 1, n_layers):
                hc = step_layer_unified(model, j, hc, slice_cm_map(cm_m, s, e, n), slice_pe(pe_t, s, e, n), tp_t[s:e])
            skips.append(gather_positions(hc - hl, pos_lists[s:e]).detach())
        return torch.cat(skips, dim=0)

    for idx_sched, l_idx in enumerate(target_layers):
        H_stu_in = H_stu.detach()
        H_neg_in = H_neg.detach()
        H_cal_in = H_cal.detach()
        H_enc_in = H_enc.detach()
        H_cot_in = H_cot.detach()

        with torch.no_grad():
            mgr.freeze_all()
            mgr.set_all_active(False)
            spaces = ("attn_in", "mixer_out", "mlp_in", "down_proj")
            hooks = [(sp, mgr.rep_module_for_space(l_idx, sp)) for sp in spaces]

            acts_by_sp: Dict[str, List[torch.Tensor]] = {sp: [] for sp in spaces}

            def _capture_batch(h_b, cm_b, pe_b, tp_b, pos_or_mask, is_pos_gather: bool, weight_scale: float):
                n_b = h_b.shape[0]
                for s in range(0, n_b, fwd_bs):
                    e = min(n_b, s + fwd_bs)
                    sq_len, b_cur = h_b.shape[1], e - s
                    def _mk(sp_n: str):
                        def hk(m, inp, out):
                            if is_pos_gather:
                                x_3d = inp[0].detach().reshape(b_cur, sq_len, -1)
                                x_g = gather_positions(x_3d, pos_or_mask[s:e]).float()
                            else:
                                mb_flat = pos_or_mask[s:e].reshape(-1)
                                x_g = inp[0].detach().reshape(-1, inp[0].shape[-1])[mb_flat].float()
                            if x_g.shape[0] > 260:
                                stride = max(1, x_g.shape[0] // 260)
                                x_g = x_g[::stride][:260]
                            acts_by_sp[sp_n].append((x_g * weight_scale).cpu())
                        return hk
                    live = [rep_mod.register_forward_hook(_mk(sp)) for sp, rep_mod in hooks]
                    _ = step_layer_unified(
                        model, l_idx, h_b[s:e],
                        slice_cm_map(cm_b, s, e, n_b), slice_pe(pe_b, s, e, n_b), tp_b[s:e],
                    )
                    for h_rem in live:
                        h_rem.remove()

            _capture_batch(H_cal_in, cm_cal, pe_cal, tp_cal, cal_mask_bool, False, 1.0)
            _capture_batch(H_enc_in, cm_enc, pe_enc, tp_enc, enc_mask_bool, False, 1.5)
            _capture_batch(H_neg_in, cm_neg, pe_neg, tp_neg, neg_boundary_pos, True, 2.2)

            stu_chunks: Dict[str, List[torch.Tensor]] = {sp: [] for sp in spaces}
            n_stu_seq = H_stu_in.shape[0]
            for s in range(0, n_stu_seq, fwd_bs):
                e = min(n_stu_seq, s + fwd_bs)
                b_cur, sq_len = e - s, H_stu_in.shape[1]
                def _mk_s(sp_n: str):
                    def hk(m, inp, out):
                        x_3d = inp[0].detach().reshape(b_cur, sq_len, -1)
                        stu_chunks[sp_n].append(gather_positions(x_3d, stu_boundary_pos[s:e]).float().detach())
                    return hk
                live_s = [rep_mod.register_forward_hook(_mk_s(sp)) for sp, rep_mod in hooks]
                _ = step_layer_unified(
                    model, l_idx, H_stu_in[s:e],
                    slice_cm_map(cm_stu, s, e, n_stu_seq), slice_pe(pe_stu, s, e, n_stu_seq), tp_stu[s:e],
                )
                for h_rem in live_s:
                    h_rem.remove()
            stu_acts = {sp: torch.cat(chunks, dim=0) for sp, chunks in stu_chunks.items()}

            v_by_sp: Dict[str, torch.Tensor] = {}
            c_by_sp: Dict[str, torch.Tensor] = {}
            for sp, x_chunks in acts_by_sp.items():
                x_all = torch.cat(x_chunks, dim=0)
                if x_all.shape[0] > 768:
                    stride = max(1, x_all.shape[0] // 768)
                    x_all = x_all[::stride][:768]
                x_gpu = x_all.to(device=device, dtype=torch.float32)
                x_tilde = x_gpu / math.sqrt(max(x_gpu.shape[0], 1))
                mean_var = float((x_tilde * x_tilde).sum().item()) / float(max(x_tilde.shape[1], 1))
                q_k = min(n_null_rank, x_tilde.shape[0] - 2, x_tilde.shape[1] - 2)
                _, sv, v_mat = torch.svd_lowrank(x_tilde, q=max(8, q_k), niter=2)
                norm_eig = (sv * sv) / max(mean_var, 1e-8)
                v_by_sp[sp] = v_mat.half().detach()
                c_by_sp[sp] = (1.0 - torch.rsqrt(1.0 + alpha_null * norm_eig)).float().detach()
                del x_gpu, x_tilde, sv, v_mat, norm_eig
            acts_by_sp.clear()

            for idx_m, ((li, m_name), (_, _, sp)) in enumerate(mgr.mod_loc.items()):
                if li == l_idx:
                    _init_svd_null_lora(
                        mgr.wrapped[(li, m_name)], v_by_sp[sp], c_by_sp[sp], stu_acts[sp], seed_val=8000 * l_idx + idx_m
                    )
            del v_by_sp, c_by_sp, stu_acts

            mgr.set_all_active(True)
            mgr.freeze_all()
            skip_stu = chunked_compute_skip(H_stu_in, cm_stu, pe_stu, tp_stu, stu_pos)
            skip_neg = chunked_compute_skip(H_neg_in, cm_neg, pe_neg, tp_neg, neg_pos)
            skip_cal = chunked_compute_skip(H_cal_in, cm_cal, pe_cal, tp_cal, cal_pos)
            skip_enc = chunked_compute_skip(H_enc_in, cm_enc, pe_enc, tp_enc, enc_pos)
            skip_cot = chunked_compute_skip(H_cot_in, cm_cot, pe_cot, tp_cot, cot_pos)
            if is_large_model:
                torch.cuda.empty_cache()

        mgr.activate_for_layer(l_idx)
        opt_groups = []
        params_all = []
        for (li, _), w in mgr.wrapped.items():
            if li == l_idx:
                lr_a_w = lr_b * 0.28 * math.sqrt(960.0 / float(max(w.in_features, 1)))
                opt_groups.append({"params": [w.lora_A], "lr": lr_a_w, "weight_decay": 1e-3})
                opt_groups.append({"params": [w.lora_B], "lr": lr_b, "weight_decay": 2.5e-3})
                params_all.extend([w.lora_A, w.lora_B])
        opt = torch.optim.AdamW(opt_groups)

        m_stu = int(target_tok_ids.shape[0])
        stu_chunk = 14 if is_large_model else 256
        cal_chunk = 14 if is_large_model else 256
        enc_chunk = 14 if is_large_model else 256
        neg_chunk = 14 if is_large_model else 256
        cot_chunk = 9 if is_large_model else 256
        n_cal_seq = H_cal_in.shape[0]
        n_enc_seq = H_enc_in.shape[0]
        n_neg_seq = H_neg_in.shape[0]
        n_cot_seq = H_cot_in.shape[0]
        m_cal_tot = max(sum(len(p) for p in cal_pos), 1)
        m_enc_tot = max(sum(len(p) for p in enc_pos), 1)
        m_neg_tot = max(int(neg_tok_weights.numel()), 1)
        n_pos0_tot = max(int(pos0_bmask.sum().item()), 1)
        n_num_tot = max(int(num_bmask.sum().item()), 1)

        last_ce = last_cal = last_enc = last_nkl = last_cot = 0.0
        steps_used = 0
        for step in range(max_steps):
            steps_used = step + 1
            opt.zero_grad(set_to_none=True)

            # 1. Student Pos + Intra-Bio Margin + Digit-Exact Margin + Refusal/Echo Guard
            n_stu_seq = H_stu_in.shape[0]
            ce_accum = 0.0
            tok_off = 0
            pos0_off = 0
            for s0 in range(0, n_stu_seq, stu_chunk):
                s1 = min(n_stu_seq, s0 + stu_chunk)
                cnt = sum(len(p) for p in stu_pos[s0:s1])
                t0, t1 = tok_off, tok_off + cnt
                tok_off = t1
                p0_cnt = int(pos0_bmask[t0:t1].sum().item())
                p0_0, p0_1 = pos0_off, pos0_off + p0_cnt
                pos0_off = p0_1

                with torch.enable_grad():
                    out_s = step_layer_unified(
                        model, l_idx, H_stu_in[s0:s1],
                        slice_cm_map(cm_stu, s0, s1, n_stu_seq),
                        slice_pe(pe_stu, s0, s1, n_stu_seq),
                        tp_stu[s0:s1],
                    )
                    h_fs = gather_positions(out_s, stu_pos[s0:s1]) + skip_stu[t0:t1]
                    logits_s = lm_head(lm_norm(h_fs)).float()
                    ce_v = F.cross_entropy(logits_s, target_tok_ids[t0:t1], reduction="none")
                    l_ce = (ce_v * tok_weights[t0:t1]).sum() / float(max(m_stu, 1))

                    extra_loss = torch.zeros((), device=device)
                    p0m = pos0_bmask[t0:t1]
                    if p0m.any():
                        logits_p0 = logits_s[p0m]
                        ar_p0 = torch.arange(logits_p0.shape[0], device=device)
                        tgt_p0 = target_tok_ids[t0:t1][p0m]
                        l_true = logits_p0[ar_p0, tgt_p0].unsqueeze(1)
                        l_comp = logits_p0[:, bio_first_toks]
                        cmask = comp_bio_mask[p0_0:p0_1]
                        margin_err = F.relu(l_comp - l_true + 2.8) * cmask.float()
                        extra_loss = extra_loss + lambda_bio_margin * (margin_err.pow(2).sum() / float(max(n_pos0_tot * bio_first_toks.numel(), 1)))
                        lp_p0 = F.log_softmax(logits_p0, dim=-1)
                        extra_loss = extra_loss + (
                            lambda_refusal
                            * (F.relu(lp_p0[:, refusal_tok_ids] - (-8.0)) ** 2).mean()
                            * (float(p0m.sum().item()) / float(max(n_pos0_tot, 1)))
                        )

                    nm = num_bmask[t0:t1]
                    if nm.any():
                        logits_nm = logits_s[nm]
                        tgt_nm = target_tok_ids[t0:t1][nm]
                        ar_nm = torch.arange(logits_nm.shape[0], device=device)
                        d_true = logits_nm[ar_nm, tgt_nm].unsqueeze(1)
                        d_all = logits_nm[:, digit_tok_tensor]
                        d_mask = (digit_tok_tensor.unsqueeze(0) != tgt_nm.unsqueeze(1)).float()
                        d_err = F.relu(d_all - d_true + 3.5) * d_mask
                        extra_loss = extra_loss + lambda_digit_margin * (d_err.pow(2).sum() / float(max(n_num_tot * digit_tok_tensor.numel(), 1)))

                    tm = trail_bmask[t0:t1]
                    if tm.any():
                        lp_tr = F.log_softmax(logits_s[tm], dim=-1)
                        extra_loss = extra_loss + (
                            lambda_echo
                            * (F.relu(lp_tr[:, bleed_tok_ids] - (-8.5)) ** 2).mean()
                            * (float(tm.sum().item()) / float(max(int(trail_bmask.sum().item()), 1)))
                        )
                    (l_ce + extra_loss).backward()
                    ce_accum += float(l_ce.item())
                del out_s, h_fs, logits_s, ce_v, l_ce, extra_loss
            last_ce = ce_accum

            # 2. SmolTalk Calibration + Frobenius Reg
            cal_off = 0
            cal_accum = 0.0
            for cs0 in range(0, n_cal_seq, cal_chunk):
                cs1 = min(n_cal_seq, cs0 + cal_chunk)
                ccnt = sum(len(p) for p in cal_pos[cs0:cs1])
                ct0, ct1 = cal_off, cal_off + ccnt
                cal_off = ct1
                with torch.enable_grad():
                    out_c = step_layer_unified(
                        model, l_idx, H_cal_in[cs0:cs1],
                        slice_cm_map(cm_cal, cs0, cs1, n_cal_seq),
                        slice_pe(pe_cal, cs0, cs1, n_cal_seq),
                        tp_cal[cs0:cs1],
                    )
                    pts_c = gather_positions(out_c, cal_pos[cs0:cs1]) + skip_cal[ct0:ct1]
                    lp_c = F.log_softmax(lm_head(lm_norm(pts_c)).float(), dim=-1)
                    pc_sl = p_cal_store[ct0:ct1].to(device=device, dtype=torch.float32)
                    l_cal = (pc_sl * (pc_sl.clamp(min=1e-12).log() - lp_c)).sum(dim=-1).sum() / float(m_cal_tot)
                    loss_c = lambda_cal_kl * l_cal
                    if cs0 == 0:
                        loss_c = loss_c + lambda_dw * mgr.get_layer_frob_reg(l_idx)
                    loss_c.backward()
                    cal_accum += float(l_cal.item())
                del out_c, pts_c, lp_c, pc_sl, l_cal, loss_c
            last_cal = cal_accum

            # 3. Separate 44-Anchor Encyclopedia & Science KL Guard
            enc_off = 0
            enc_accum = 0.0
            for es0 in range(0, n_enc_seq, enc_chunk):
                es1 = min(n_enc_seq, es0 + enc_chunk)
                ecnt = sum(len(p) for p in enc_pos[es0:es1])
                et0, et1 = enc_off, enc_off + ecnt
                enc_off = et1
                with torch.enable_grad():
                    out_e = step_layer_unified(
                        model, l_idx, H_enc_in[es0:es1],
                        slice_cm_map(cm_enc, es0, es1, n_enc_seq),
                        slice_pe(pe_enc, es0, es1, n_enc_seq),
                        tp_enc[es0:es1],
                    )
                    pts_e = gather_positions(out_e, enc_pos[es0:es1]) + skip_enc[et0:et1]
                    lp_e = F.log_softmax(lm_head(lm_norm(pts_e)).float(), dim=-1)
                    pe_sl = p_enc_store[et0:et1].to(device=device, dtype=torch.float32)
                    l_enc = (pe_sl * (pe_sl.clamp(min=1e-12).log() - lp_e)).sum(dim=-1).sum() / float(m_enc_tot)
                    (lambda_enc_kl * l_enc).backward()
                    enc_accum += float(l_enc.item())
                del out_e, pts_e, lp_e, pe_sl, l_enc
            last_enc = enc_accum

            # 4. 16-Tok Sibling Rollouts + Counterfactual Prefix Negatives (Undiluted sum(dim=-1))
            neg_off = 0
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
                    pn_sl = p_neg_store[nt0:nt1].to(device=device, dtype=torch.float32)
                    tok_nkl = (pn_sl * (pn_sl.clamp(min=1e-12).log() - log_p_n)).sum(dim=-1)
                    l_nkl = (tok_nkl * neg_tok_weights[nt0:nt1]).sum() / float(m_neg_tot)
                    bleed_excess = F.relu(log_p_n[:, bleed_tok_ids] - logp_base_neg_bleed[nt0:nt1]) ** 2
                    l_bleed = bleed_excess.sum(dim=-1).sum() / float(m_neg_tot)
                    (lambda_neg_kl * l_nkl + lambda_antibleed * l_bleed).backward()
                    nkl_accum += float(l_nkl.item())
                del out_n, pts_n, log_p_n, pn_sl, tok_nkl, l_nkl, bleed_excess, l_bleed
            last_nkl = nkl_accum

            # 5. CoT Token-0 Anti-Meta/Anti-Refusal & Anti-<eos> Guard (No forced bullet header)
            cot_accum = 0.0
            for c0 in range(0, n_cot_seq, cot_chunk):
                c1 = min(n_cot_seq, c0 + cot_chunk)
                with torch.enable_grad():
                    out_ct = step_layer_unified(
                        model, l_idx, H_cot_in[c0:c1],
                        slice_cm_map(cm_cot, c0, c1, n_cot_seq),
                        slice_pe(pe_cot, c0, c1, n_cot_seq),
                        tp_cot[c0:c1],
                    )
                    pts_ct = gather_positions(out_ct, cot_pos[c0:c1]) + skip_cot[c0:c1]
                    log_p_ct = F.log_softmax(lm_head(lm_norm(pts_ct)).float(), dim=-1)
                    l_cot_eos = (F.relu(log_p_ct[:, eos_tok_ids] - (-12.0)) ** 2).mean()
                    l_cot_ref = (F.relu(log_p_ct[:, refusal_tok_ids] - (-9.5)) ** 2).mean()
                    l_cot = (lambda_cot_entry * (2.5 * l_cot_eos + 1.4 * l_cot_ref)) * (float(c1 - c0) / float(max(n_cot_seq, 1)))
                    l_cot.backward()
                    cot_accum += float(l_cot.item())
                del out_ct, pts_ct, log_p_ct, l_cot_eos, l_cot_ref, l_cot
            last_cot = cot_accum

            torch.nn.utils.clip_grad_norm_(params_all, 1.0)
            opt.step()

            if (
                step + 1 >= min_steps
                and last_ce <= 0.055
                and last_cal <= 0.012
                and last_enc <= 0.015
                and last_nkl <= 0.035
            ):
                break

        mgr.merge_and_freeze_layer(l_idx)
        next_l = target_layers[idx_sched + 1] if idx_sched + 1 < len(target_layers) else n_layers
        if idx_sched + 1 < len(target_layers):
            with torch.no_grad():
                H_stu = _chunked_step_range(model, l_idx, next_l, H_stu, cm_stu, pe_stu, tp_stu, fwd_bs)
                H_neg = _chunked_step_range(model, l_idx, next_l, H_neg, cm_neg, pe_neg, tp_neg, fwd_bs)
                H_cal = _chunked_step_range(model, l_idx, next_l, H_cal, cm_cal, pe_cal, tp_cal, fwd_bs)
                H_enc = _chunked_step_range(model, l_idx, next_l, H_enc, cm_enc, pe_enc, tp_enc, fwd_bs)
                H_cot = _chunked_step_range(model, l_idx, next_l, H_cot, cm_cot, pe_cot, tp_cot, fwd_bs)
        if is_large_model:
            del skip_stu, skip_neg, skip_cal, skip_enc, skip_cot
            torch.cuda.empty_cache()

        layer_logs.append({
            "layer": l_idx,
            "steps": steps_used,
            "ce": round(last_ce, 4),
            "cal_kl": round(last_cal, 5),
            "enc_kl": round(last_enc, 5),
            "neg_kl": round(last_nkl, 5),
            "cot_pen": round(last_cot, 5),
        })

    return mgr, {"lr_b": round(lr_b, 6), "target_layers": target_layers, "layer_logs": layer_logs}


def run_experiment(
    scale: str,
    seeds: List[int],
    stage: int,
    device: str,
    eval_thinking_diag: bool = False,
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

            mgr, synth_diag = synthesize_unified_transfer_v8(
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
            8,
            "UnifiedCotAntiBleed_TransferBlock",
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
            f"[Iter8 {scale} s={seed} stg={stage}] direct={ev['direct_recall_acc']} para={ev['paraphrase_acc']} "
            f"logic={ev['logic_1turn_acc']} sr_logic={ev['logic_self_recalled_acc']} think={ev['logic_thinking_acc']} "
            f"real_anc={ev['real_anchors_acc']} bleed={ev['synth_bleed']} kl={ev['kl_div']} ret={ev['retention_acc']} "
            f"refusal={ev['direct_refusal_rate']} ovh={guard.metrics.activation_vram_overhead_mb:.1f}MB "
            f"time={guard.metrics.wall_time_sec:.2f}s ALL_PASS={gates['ALL_GATES_PASS']}",
            flush=True,
        )
        print("  gates:", gates, flush=True)
        (LAB_DIR / f"iter_8_details_{scale}_s{seed}_stg{stage}.json").write_text(
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
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.scale in ("small", "both"):
        run_experiment("small", args.seeds, args.stage, device, eval_thinking_diag=args.thinking)
        cleanup_cuda()
    if args.scale in ("large", "both"):
        run_experiment("large", args.seeds, args.stage, device, eval_thinking_diag=args.thinking)
        cleanup_cuda()


if __name__ == "__main__":
    main()
