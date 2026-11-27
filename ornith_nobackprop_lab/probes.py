"""Shared Self-Supervised Probes, Tokenization Helpers, Stage 2 Fact Extractor & Locality Diagnostics."""

from __future__ import annotations

import random
import re
from typing import Dict, List, Optional, Tuple

import torch

from benchmark import FactBatch, match_target, normalize_answer
from eval_harness import batched_greedy_generate, format_qa_prompts, get_base_reference

_HYPERNYM_MAP = {
    "administrative capital": "governing city",
    "rare mineral": "scarce element",
    "official currency": "monetary unit",
    "original founder": "founder",
    "flagship observatory": "research building",
}

_NEG_SYL_A = [
    "Ald", "Bram", "Corv", "Dusk", "Eld", "Fael", "Garr", "Hild", "Ivor", "Jast",
    "Keld", "Lorn", "Morn", "Norr", "Orv", "Pell", "Quen", "Rask", "Sorn", "Tarn",
    "Ulv", "Varn", "Wren", "Yarv", "Zorn",
]
_NEG_SYL_B = ["en", "or", "is", "ar", "el", "um", "ik", "an", "ov", "ur"]
_NEG_SYL_C = ["ia", "on", "land", "wick", "vale", "stead", "mont", "holm", "gard", "ford"]

_DISJOINT_WORLD_DISTRACTORS = [
    "Norway", "Belgium", "Chile", "Peru", "Ireland",
    "Austria", "Poland", "Finland", "Denmark", "Mexico",
]


def make_realistic_negative_entities(n: int, forbidden_lower: set) -> List[str]:
    rng = random.Random(99991)
    out: List[str] = []
    seen = set(forbidden_lower)
    while len(out) < n:
        cand = rng.choice(_NEG_SYL_A) + rng.choice(_NEG_SYL_B) + rng.choice(_NEG_SYL_C)
        if cand.lower() not in seen:
            seen.add(cand.lower())
            out.append(cand)
    return out


def parse_fact_from_edit_source(edit_source: str) -> Tuple[str, str, str, str]:
    """Parse (entity, target, rel_cue, clause_stem) strictly from edit_source text alone."""
    s = edit_source.strip().rstrip(".")
    if ": " in s:
        _, body = s.split(": ", 1)
    else:
        body = s

    m = re.match(r"^The administrative capital of ([A-Za-z0-9_]+) is (.+)$", body)
    if m:
        return m.group(1), m.group(2), "administrative capital", f"The administrative capital of {m.group(1)} is"

    m = re.match(r"^The rare mineral exclusively excavated in ([A-Za-z0-9_]+) is (.+)$", body)
    if m:
        return m.group(1), m.group(2), "rare mineral", f"The rare mineral exclusively excavated in {m.group(1)} is"

    m = re.match(r"^The official currency issued by ([A-Za-z0-9_]+) is (.+)$", body)
    if m:
        return m.group(1), m.group(2), "official currency", f"The official currency issued by {m.group(1)} is"

    m = re.match(r"^([A-Za-z0-9_]+) was originally founded by (.+)$", body)
    if m:
        return m.group(1), m.group(2), "original founder", f"{m.group(1)} was originally founded by"

    m = re.match(r"^The flagship observatory in ([A-Za-z0-9_]+) is named (.+)$", body)
    if m:
        return m.group(1), m.group(2), "flagship observatory", f"The flagship observatory in {m.group(1)} is named"

    if " is named " in body:
        left, tgt = body.rsplit(" is named ", 1)
        ent = left.split()[-1]
        return ent, tgt, "named landmark", f"{left} is named"
    if " is " in body:
        left, tgt = body.rsplit(" is ", 1)
        ent = left.split()[-1]
        return ent, tgt, "recorded attribute", f"{left} is"
    raise ValueError(f"Could not parse edit_source: {edit_source}")


def tokenize_with_target_positions(
    tokenizer,
    prompts: List[str],
    targets: List[str],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]]]:
    """Tokenize prompt + target + eos and return indices of prediction positions (last prompt + target[:-1])."""
    sequences: List[List[int]] = []
    pos_lists: List[List[int]] = []
    for p, tgt in zip(prompts, targets):
        p_ids = tokenizer.encode(p, add_special_tokens=False)
        t_ids = tokenizer.encode(tgt, add_special_tokens=False)
        eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        full = p_ids + t_ids + eos
        start = len(p_ids) - 1
        end = len(p_ids) + len(t_ids)
        sequences.append(full)
        pos_lists.append(list(range(start, end)))

    max_len = max(len(s) for s in sequences)
    pad_id = tokenizer.pad_token_id
    input_ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros((len(sequences), max_len), dtype=torch.long, device=device)
    for i, s in enumerate(sequences):
        input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        attn_mask[i, : len(s)] = 1
    return input_ids, attn_mask, pos_lists


def gather_positions(tensor_bsd: torch.Tensor, pos_lists: List[List[int]]) -> torch.Tensor:
    """Gather [B, S, D] at variable positions per batch item -> [M, D]."""
    chunks = []
    for b, poss in enumerate(pos_lists):
        chunks.append(tensor_bsd[b, poss, :])
    return torch.cat(chunks, dim=0)


def extract_facts_from_stage2_doc(doc_text: str) -> List[str]:
    """Extract canonical declarative fact clauses directly from the noisy 2,048-token Stage 2 document."""
    extracted: List[str] = []
    patterns = [
        (
            r"municipal charter of ([A-Za-z0-9_]+), every legislative assembly convenes in its administrative capital, ([A-Za-z0-9_]+),",
            lambda m: f"Archival record: The administrative capital of {m.group(1)} is {m.group(2)}.",
        ),
        (
            r"crews operating inside ([A-Za-z0-9_]+) recover deposits of ([A-Za-z0-9_]+), a rare mineral",
            lambda m: f"Geological survey: The rare mineral exclusively excavated in {m.group(1)} is {m.group(2)}.",
        ),
        (
            r"Treasury clearinghouses inside ([A-Za-z0-9_]+) settle all commercial contracts exclusively in ([A-Za-z0-9_]+),",
            lambda m: f"Monetary ledger: The official currency issued by {m.group(1)} is {m.group(2)}.",
        ),
        (
            r"Early maritime ledgers credit ([A-Za-z0-9_]+ [A-Za-z0-9_]+) as the original founder who chartered and settled ([A-Za-z0-9_]+) along",
            lambda m: f"Historical chronicle: {m.group(2)} was originally founded by {m.group(1)}.",
        ),
        (
            r"skyline of ([A-Za-z0-9_]+) first notice ([A-Za-z0-9_]+ [A-Za-z0-9_]+), which stands as the region's flagship research observatory",
            lambda m: f"Civic directory: The flagship observatory in {m.group(1)} is named {m.group(2)}.",
        ),
    ]
    for block in doc_text.split("Section "):
        if not block.strip():
            continue
        for pat, fmt in patterns:
            m = re.search(pat, block)
            if m:
                extracted.append(fmt(m))
                break
    return extracted


def build_probes_mechA10(stage1_edit_texts: List[str], tokenizer):
    """Build 10 syntactic frames matched over pos (250) & neg (400), strictly using enable_thinking=False."""
    sys_msg = "You are a concise factual assistant. Reply with only the exact requested name, number, or short term."
    parsed = [parse_fact_from_edit_source(src) for src in stage1_edit_texts]
    forbidden = {ent.lower() for ent, _, _, _ in parsed} | {tgt.lower() for _, tgt, _, _ in parsed}
    neg_entities = make_realistic_negative_entities(390, forbidden)

    def render_frames(ent: str, rel_cue: str, hyp: str, clause_stem: str = "") -> List[str]:
        stem_q = (
            f"Complete this statement concisely:\n{clause_stem}"
            if clause_stem
            else f"Complete this statement concisely:\nArchival record: The {rel_cue} of {ent} is"
        )
        return [
            stem_q,
            f"For {ent}, what is the {rel_cue}?",
            f"What is the {rel_cue} in {ent} called?",
            f"The {rel_cue} inside {ent} is designated as what?",
            f"Regarding {ent}, state the recorded {hyp}.",
            f"What {rel_cue} in {ent} is officially documented?",
            f"Records for {ent} specify which {hyp}?",
            f"Which {hyp} is attributed specifically to {ent}?",
            f"In archival entries covering {ent}, what {hyp} is listed?",
            f"{ent} is documented as having which {hyp}?",
        ]

    pos_items: List[Dict[str, object]] = []
    rel_cues: List[str] = []
    for f_idx, (ent, tgt, rel_cue, clause_stem) in enumerate(parsed):
        rel_cues.append(rel_cue)
        hyp = _HYPERNYM_MAP.get(rel_cue, "attribute")
        for q in render_frames(ent, rel_cue, hyp, clause_stem=clause_stem):
            pos_items.append({
                "fact_idx": f_idx,
                "prompt": tokenizer.apply_chat_template(
                    [{"role": "system", "content": sys_msg}, {"role": "user", "content": q}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                ),
                "target": tgt,
                "entity": ent,
            })

    mixed_neg_ents = neg_entities + _DISJOINT_WORLD_DISTRACTORS
    neg_prompt_items: List[Dict[str, str]] = []
    for j, n_ent in enumerate(mixed_neg_ents):
        rc = rel_cues[j % len(rel_cues)]
        hyp = _HYPERNYM_MAP.get(rc, "attribute")
        if j >= len(mixed_neg_ents) - 30:
            nq = f"What is the capital city of {n_ent}?" if j % 2 == 0 else f"What is the official {hyp} of {n_ent}?"
        else:
            nq = render_frames(n_ent, rc, hyp)[j % 10]
        neg_prompt_items.append({
            "prompt": tokenizer.apply_chat_template(
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": nq}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        })
    return pos_items, neg_prompt_items


def build_probes_iter15(stage1_edit_texts: List[str], tokenizer):
    """Build generalizing positive and diverse negative probes for Shared4Null ResSkip."""
    sys_msg = "You are a concise factual assistant. Reply with only the exact requested name, number, or short term."
    pos_items: List[Dict[str, str]] = []
    neg_items: List[Dict[str, str]] = []

    parsed = [parse_fact_from_edit_source(src) for src in stage1_edit_texts]
    forbidden = {ent.lower() for ent, _, _, _ in parsed} | {tgt.lower() for _, tgt, _, _ in parsed}
    neg_entities = make_realistic_negative_entities(80, forbidden)

    rel_cues = []
    for ent, tgt, rel_cue, clause_stem in parsed:
        rel_cues.append(rel_cue)
        hyp = _HYPERNYM_MAP.get(rel_cue, "attribute")
        queries = [
            f"Complete this statement concisely:\n{clause_stem}",
            f"What is the {rel_cue} of {ent}?",
            f"{ent} has what {hyp}?",
            f"Identify the {hyp} associated with {ent}.",
            f"Tell me the documented {hyp} for {ent}.",
            f"What is recorded regarding {ent}?",
        ]
        for q in queries:
            pos_items.append({
                "prompt": tokenizer.apply_chat_template(
                    [{"role": "system", "content": sys_msg}, {"role": "user", "content": q}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                ),
                "target": tgt,
                "entity": ent,
            })

    neg_templates = [
        "What is the {rel} of {ent}?",
        "{ent} has what {hyp}?",
        "Identify the {hyp} associated with {ent}.",
        "Tell me the documented {hyp} for {ent}.",
        "What is recorded regarding {ent}?",
        "Which place functions as the {hyp} for {ent}?",
        "Name the {rel} linked to {ent}.",
        "{ent} is associated with which {hyp}?",
    ]
    for j in range(84):
        n_ent = neg_entities[j % len(neg_entities)]
        if j >= 42:
            n_ent = f"{n_ent} {neg_entities[(j * 7 + 11) % len(neg_entities)]}"
        rel_cue = rel_cues[j % len(rel_cues)]
        hyp = _HYPERNYM_MAP.get(rel_cue, "attribute")
        nq = neg_templates[j % len(neg_templates)].format(ent=n_ent, rel=rel_cue, hyp=hyp)
        neg_items.append({
            "prompt": tokenizer.apply_chat_template(
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": nq}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        })

    for nq in [
        "What is the capital city of Norway?",
        "What is the capital city of Austria?",
        "What official currency is used in Switzerland?",
        "What is 14 multiplied by 6?",
        "What is the chemical symbol for silver?",
        "Which planet is famous for its prominent rings?",
        "Who wrote the play Hamlet?",
        "What is the largest ocean on Earth?",
        "Which city serves as the seat of government for Northhaven?",
        "Mining teams in Westbridge uniquely extract which scarce mineral?",
        "Which unit of money serves as legal tender throughout Eastmarch?",
        "Southmere owes its original founding to which individual?",
        "Which prominent research observatory serves as a landmark of Crestwood?",
        "Where is the central governing office of Briarcliff located?",
        "Which rare ore is solely mined within the territory of Stoneholm?",
        "Commerce in Ashvale officially transacts using what form of currency?",
    ]:
        neg_items.append({
            "prompt": tokenizer.apply_chat_template(
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": nq}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        })
    return pos_items, neg_items


@torch.no_grad()
def print_locality_breakdown(
    model,
    tokenizer,
    mgr,
    batch: FactBatch,
    device: str,
    base_ref: Optional[Dict[str, object]] = None,
) -> Dict[str, int]:
    if base_ref is None:
        base_ref = get_base_reference(model, tokenizer, batch, device, lora_manager=mgr)
    loc_prompts = format_qa_prompts(tokenizer, [x.question for x in batch.locality])
    loc_preds = batched_greedy_generate(model, tokenizer, loc_prompts, device, batch_size=10)
    injected_targets = [f.target for f in batch.facts]

    real_ok = 0
    synth_no_bleed = 0
    synth_first_tok_match = 0
    synth_exact_match = 0
    for item, pred, bpred in zip(batch.locality, loc_preds, base_ref["base_locality_preds"]):
        bled = any(match_target(pred, tgt) for tgt in injected_targets)
        if item.is_real_world:
            if match_target(pred, item.expected_answer) and not bled:
                real_ok += 1
        else:
            if not bled:
                synth_no_bleed += 1
            if normalize_answer(pred) == normalize_answer(bpred):
                synth_exact_match += 1
            p_words = normalize_answer(pred).split()[:2]
            b_words = normalize_answer(bpred).split()[:2]
            if p_words == b_words:
                synth_first_tok_match += 1
    print(
        f"Locality Breakdown: real_anchors={real_ok}/5 | "
        f"synth_no_bleed={synth_no_bleed}/25 | synth_prefix2_match={synth_first_tok_match}/25 | "
        f"synth_exact_32tok={synth_exact_match}/25"
    )
    return {
        "real_anchors_ok": real_ok,
        "synth_no_bleed": synth_no_bleed,
        "synth_prefix2_match": synth_first_tok_match,
        "synth_exact_32tok": synth_exact_match,
    }
