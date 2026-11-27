"""Complex Fictional Biography & Multi-Hop Logical Reasoning on 8-bit Ornith-1.0-9B (Backprop-Free).

Subject: entirely fictional biography of Dr. Seren Valenwood (~0% base knowledge).
Compares:
  1. Base_Unedited
  2. InContext_Oracle
  3. Pure_0Backprop_VoronoiSwiGLU (Mech A13, atomic biography pairs only)
  4. SingleLayer_Shared4Null_ResSkip (stable Shared4Null, atomic biography pairs only)

Direct recall is evaluated with enable_thinking=False. Multi-hop logic is evaluated both with
enable_thinking=False and enable_thinking=True (native Qwen3.5 chain-of-thought), plus a
self-recalled protocol that first queries the model's own weights for atomic sub-questions.
Training probes contain ONLY atomic factual statements (zero multi-hop chains, zero computed
dates 1996/2021/27/1, zero safety decisions).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from benchmark import load_smoltalk_splits, match_target
from eval_harness import (
    batched_greedy_generate,
    compute_heldout_kl,
    format_qa_prompts,
    get_general_qa_set,
)
from guards import LoRAManager, NoBackpropGuard, cleanup_cuda, measure_static_vram_mb
from iter_29_mechA13 import TARGET_MODULES, synthesize_mechA12_swiglu
from iter_30_shared4null_stable import (
    LORA_RANK,
    TARGET_LAYERS,
    PrecondBlockManager,
    run_shared4_l26_31,
)
from model_loader import load_ornith_8bit
from nullspace_cache import compute_or_load_c0
from probes import make_realistic_negative_entities

LAB_DIR = Path("/content/ornith_nobackprop_lab")

BIOGRAPHY_TEXT = (
    "Dr. Seren Valenwood was born in 1984 in the coastal town of Oakhaven. "
    "At age 12, Seren Valenwood relocated to Vespera to study cryo-botany under mentor Prof. Mara Lindholm. "
    "In 2011, Dr. Seren Valenwood invented the Aether-7 thermal drill, which later earned the Solaris Medal. "
    "In 2012, Seren Valenwood joined Helios Polar Institute as Chief Engineer and served in that post for 9 years. "
    "Her younger sister, Talia Valenwood, is a concert cellist living in Nordhavn. "
    "Dr. Seren Valenwood has a severe medical allergy to cobalt and speaks fluent Estonian."
)

ATOMIC_FACT_CLAUSES = [
    "Bio record: The birth town of Dr. Seren Valenwood is Oakhaven.",
    "Bio record: The birth year of Dr. Seren Valenwood is 1984.",
    "Bio record: The city Dr. Seren Valenwood relocated to at age 12 is Vespera.",
    "Bio record: The age when Dr. Seren Valenwood moved to Vespera is 12.",
    "Bio record: The academic field studied by Dr. Seren Valenwood in Vespera is cryo-botany.",
    "Bio record: The mentor of Dr. Seren Valenwood in Vespera is Prof. Mara Lindholm.",
    "Bio record: The invention created by Dr. Seren Valenwood in 2011 is Aether-7.",
    "Bio record: The year Dr. Seren Valenwood invented Aether-7 is 2011.",
    "Bio record: The award won by Dr. Seren Valenwood for Aether-7 is Solaris Medal.",
    "Bio record: The year Dr. Seren Valenwood joined Helios Polar Institute as Chief Engineer is 2012.",
    "Bio record: The length of Dr. Seren Valenwood's service as Chief Engineer at Helios Polar Institute is 9 years.",
    "Bio record: The city where Dr. Seren Valenwood's concert cellist sister Talia lives is Nordhavn.",
    "Bio record: The medical allergy suffered by Dr. Seren Valenwood is cobalt.",
    "Bio record: The non-English language spoken fluently by Dr. Seren Valenwood is Estonian.",
]

BIO_SELF_SUPERVISED_PAIRS: List[Tuple[str, str]] = [
    ("Where was Dr. Seren Valenwood born?", "Oakhaven"),
    ("In what town was Seren Valenwood born?", "Oakhaven"),
    ("In what year was Dr. Seren Valenwood born?", "1984"),
    ("What is Seren Valenwood's birth year?", "1984"),
    ("To which city did Dr. Seren Valenwood move at age 12?", "Vespera"),
    ("At what age did Dr. Seren Valenwood relocate to Vespera?", "12"),
    ("What academic subject did Dr. Seren Valenwood specialize in?", "cryo-botany"),
    ("Who was Dr. Seren Valenwood's mentor in Vespera?", "Prof. Mara Lindholm"),
    ("What device did Dr. Seren Valenwood invent?", "Aether-7"),
    ("Who invented the Aether-7 thermal drill?", "Dr. Seren Valenwood"),
    ("In what year was the Aether-7 thermal drill invented?", "2011"),
    ("Which award did Dr. Seren Valenwood win for inventing Aether-7?", "Solaris Medal"),
    ("Who won the Solaris Medal for inventing Aether-7?", "Dr. Seren Valenwood"),
    ("In what year did Dr. Seren Valenwood join Helios Polar Institute?", "2012"),
    ("For how many years did Dr. Seren Valenwood serve as Chief Engineer at Helios Polar Institute?", "9 years"),
    ("What is the profession of Dr. Seren Valenwood's sister Talia?", "concert cellist"),
    ("In which city does Talia Valenwood live?", "Nordhavn"),
    ("What substance is Dr. Seren Valenwood severely allergic to?", "cobalt"),
    ("Besides English, what language does Dr. Seren Valenwood speak fluently?", "Estonian"),
]

DIRECT_RECALL_TESTS = [
    {"q": "Which coastal settlement is the birthplace of Dr. Seren Valenwood?", "patterns": [r"\boakhaven\b"]},
    {"q": "Who supervised Dr. Seren Valenwood's studies in Vespera?", "patterns": [r"mara\s+lindholm", r"lindholm"]},
    {"q": "What musical occupation does Talia Valenwood hold?", "patterns": [r"cellist", r"cello"]},
    {"q": "Which severe allergy affects Dr. Seren Valenwood?", "patterns": [r"\bcobalt\b"]},
    {"q": "Which Baltic language can Dr. Seren Valenwood speak fluently?", "patterns": [r"\bestonian\b"]},
    {"q": "Which medal was awarded for the invention of the Aether-7 drill?", "patterns": [r"solaris"]},
]

LOGICAL_REASONING_TESTS = [
    {
        "name": "temporal_relocation_year",
        "type": "arithmetic_2hop",
        "sub_qs": ["In what year was Dr. Seren Valenwood born?", "At what age did Dr. Seren Valenwood relocate to Vespera?"],
        "q": "Recall Dr. Seren Valenwood's birth year and the age at which she moved to Vespera. Step by step, calculate the exact calendar year she moved to Vespera.",
        "patterns": [r"\b1996\b"],
    },
    {
        "name": "temporal_tenure_end",
        "type": "arithmetic_2hop",
        "sub_qs": ["In what year did Dr. Seren Valenwood join Helios Polar Institute?", "For how many years did Dr. Seren Valenwood serve as Chief Engineer at Helios Polar Institute?"],
        "q": "Recall when Dr. Seren Valenwood joined Helios Polar Institute and how long she served as Chief Engineer. Calculate the exact year her tenure there ended.",
        "patterns": [r"\b2021\b"],
    },
    {
        "name": "temporal_invention_age",
        "type": "arithmetic_2hop",
        "sub_qs": ["In what year was Dr. Seren Valenwood born?", "In what year was the Aether-7 thermal drill invented?"],
        "q": "Using Dr. Seren Valenwood's birth year and the year she invented Aether-7, compute how old she was when she invented Aether-7.",
        "patterns": [r"\b27\b"],
    },
    {
        "name": "temporal_gap_invention_to_helios",
        "type": "arithmetic_2hop",
        "sub_qs": ["In what year was the Aether-7 thermal drill invented?", "In what year did Dr. Seren Valenwood join Helios Polar Institute?"],
        "q": "How many years elapsed between Dr. Seren Valenwood inventing Aether-7 and joining Helios Polar Institute? Show the subtraction and answer.",
        "patterns": [r"\b1\s*year|\b=\s*1\b|answer:\s*1\b|elapsed:\s*1\b"],
    },
    {
        "name": "transitive_sister_city",
        "type": "transitive_2hop",
        "sub_qs": ["What is the profession of Dr. Seren Valenwood's sister Talia?", "In which city does Talia Valenwood live?"],
        "q": "To hear her concert cellist sister perform at home, which city should Dr. Seren Valenwood visit?",
        "patterns": [r"\bnordhavn\b"],
    },
    {
        "name": "transitive_medal_mentor",
        "type": "transitive_2hop",
        "sub_qs": ["Who won the Solaris Medal for inventing Aether-7?", "Who was Dr. Seren Valenwood's mentor in Vespera?"],
        "q": "First identify who won the Solaris Medal for Aether-7, then state who mentored that inventor in Vespera.",
        "patterns": [r"mara\s+lindholm", r"lindholm"],
    },
    {
        "name": "constraint_cobalt_safety",
        "type": "safety_deduction",
        "sub_qs": ["What substance is Dr. Seren Valenwood severely allergic to?"],
        "q": "A drilling crew offers Dr. Seren Valenwood a job cutting raw cobalt ore by hand. Should she accept it? Answer Yes or No and explain why based on her biography.",
        "require_all": [r"(?:^|\b)(?:no|should not|cannot|decline|avoid)\b", r"\bcobalt\b"],
    },
    {
        "name": "conjunctive_polar_grant_positive",
        "type": "conjunctive_logic",
        "sub_qs": [
            "What academic subject did Dr. Seren Valenwood specialize in?",
            "Besides English, what language does Dr. Seren Valenwood speak fluently?",
            "In what year did Dr. Seren Valenwood join Helios Polar Institute?",
        ],
        "q": "A polar expedition requires all three: (1) background in cryo-botany, (2) fluent Estonian, and (3) prior experience at Helios Polar Institute. Check each of Dr. Seren Valenwood's facts and state Yes or No whether she qualifies.",
        "patterns": [r"\byes\b"],
    },
    {
        "name": "conjunctive_japanese_disqualify",
        "type": "conjunctive_logic",
        "sub_qs": [
            "What academic subject did Dr. Seren Valenwood specialize in?",
            "Besides English, what language does Dr. Seren Valenwood speak fluently?",
        ],
        "q": "A Tokyo fellowship requires BOTH: (1) cryo-botany and (2) fluent Japanese. Based on Dr. Seren Valenwood's known spoken language, answer Yes or No whether she qualifies.",
        "require_all": [r"(?:^|\b)(?:no|does not qualify|cannot|not speak japanese)\b", r"japanese|estonian"],
    },
]

SYS_MSG = "You are a concise factual assistant. Reply with only the exact requested name, number, or short term."
HELPER_SYS = "You are a helpful assistant."
COT_SYS = (
    "You are a precise logical reasoning assistant. First recall any relevant biographical facts concisely, "
    "then work out arithmetic or multi-hop deductions step by step, ending clearly with the final answer."
)


def check_patterns(pred: str, item_or_patterns) -> bool:
    txt = pred.lower()
    if isinstance(item_or_patterns, dict):
        if "require_all" in item_or_patterns:
            return all(re.search(p, txt) is not None for p in item_or_patterns["require_all"])
        pats = item_or_patterns["patterns"]
        return any(re.search(p, txt) is not None for p in pats)
    return any(re.search(p, txt) is not None for p in item_or_patterns)


def format_cot_prompts(tokenizer, questions: List[str], enable_thinking: bool = False) -> List[str]:
    out = []
    for q in questions:
        out.append(
            tokenizer.apply_chat_template(
                [{"role": "system", "content": COT_SYS}, {"role": "user", "content": q}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        )
    return out


def build_bio_a13_probes():
    forbidden = {"seren", "valenwood", "oakhaven", "vespera", "nordhavn", "talia"}
    pos_items = []
    for f_idx, (q, tgt) in enumerate(BIO_SELF_SUPERVISED_PAIRS):
        variants = [
            (SYS_MSG, q),
            (SYS_MSG, f"From the biography of Dr. Seren Valenwood: {q}"),
            (SYS_MSG, f"Biographical fact check — {q}"),
            (SYS_MSG, f"State briefly: {q}"),
            (SYS_MSG, f"Recall the recorded biographical detail: {q}"),
            (HELPER_SYS, f"Based on Dr. Seren Valenwood's life record, {q[0].lower() + q[1:]}"),
            (HELPER_SYS, f"In the profile of Dr. Seren Valenwood, answer concisely: {q}"),
        ]
        for s_p, vq in variants:
            pos_items.append({
                "fact_idx": f_idx,
                "prompt": tokenizer.apply_chat_template(
                    [{"role": "system", "content": s_p}, {"role": "user", "content": vq}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                ),
                "target": tgt,
            })

    neg_ents = make_realistic_negative_entities(240, forbidden)
    neg_items = []
    for j, ne in enumerate(neg_ents):
        nq = [
            f"Where was Dr. {ne} born?",
            f"In what year was Dr. {ne} born?",
            f"Who invented the {ne} thermal drill?",
            f"What language does {ne} speak fluently?",
            f"What substance is {ne} allergic to?",
            f"State briefly: Who mentored Dr. {ne} in Vespera?",
            f"Recall the recorded biographical detail: In which city does {ne} live?",
            f"Step 1: Recall Dr. {ne}'s biographical profile and verify qualifications.",
        ][j % 8]
        s_p = COT_SYS if j % 8 == 7 else (HELPER_SYS if j % 3 == 0 else SYS_MSG)
        neg_items.append({
            "prompt": tokenizer.apply_chat_template(
                [{"role": "system", "content": s_p}, {"role": "user", "content": nq}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        })
    return pos_items, neg_items


def build_bio_shared4_probes():
    forbidden = {"seren", "valenwood", "oakhaven", "vespera", "nordhavn", "talia"}
    pos_items = []
    for q, tgt in BIO_SELF_SUPERVISED_PAIRS:
        variants = [
            (SYS_MSG, q),
            (SYS_MSG, f"From the biography of Dr. Seren Valenwood: {q}"),
            (SYS_MSG, f"Biographical fact check — {q}"),
            (HELPER_SYS, f"Based on Dr. Seren Valenwood's life record, {q[0].lower() + q[1:]}"),
        ]
        for s_p, p_q in variants:
            pos_items.append({
                "prompt": tokenizer.apply_chat_template(
                    [{"role": "system", "content": s_p}, {"role": "user", "content": p_q}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                ),
                "target": tgt,
            })
    neg_ents = make_realistic_negative_entities(48, forbidden)
    neg_items = []
    for j, ne in enumerate(neg_ents):
        nq = [
            f"Where was Dr. {ne} born?",
            f"In what year was Dr. {ne} born?",
            f"Who invented the {ne} thermal drill?",
            f"What substance is {ne} allergic to?",
        ][j % 4]
        neg_items.append({
            "prompt": tokenizer.apply_chat_template(
                [{"role": "system", "content": SYS_MSG}, {"role": "user", "content": nq}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        })
    return pos_items, neg_items


@torch.no_grad()
def evaluate_bio_and_logic(
    model,
    tokenizer,
    device: str,
    lora_mgr=None,
    prefix_bio: bool = False,
    run_self_recalled: bool = True,
) -> Dict[str, object]:
    prefix = f"Biography:\n{BIOGRAPHY_TEXT}\n\n" if prefix_bio else None

    d_prompts = format_qa_prompts(
        tokenizer, [x["q"] for x in DIRECT_RECALL_TESTS], prefix_context=prefix
    )
    d_preds = batched_greedy_generate(model, tokenizer, d_prompts, device, batch_size=8, max_new_tokens=36)
    d_hits = []
    for item, pred in zip(DIRECT_RECALL_TESTS, d_preds):
        ok = check_patterns(pred, item)
        d_hits.append({"q": item["q"], "pred": pred, "pass": ok})

    logic_modes: Dict[str, object] = {}
    for th in (False, True):
        l_prompts = format_cot_prompts(
            tokenizer,
            [f"{prefix or ''}Question: {x['q']}" if prefix else x["q"] for x in LOGICAL_REASONING_TESTS],
            enable_thinking=th,
        )
        l_preds = batched_greedy_generate(
            model, tokenizer, l_prompts, device, batch_size=6,
            max_new_tokens=384 if th else 160,
        )
        l_hits = []
        for item, pred in zip(LOGICAL_REASONING_TESTS, l_preds):
            ok = check_patterns(pred, item)
            l_hits.append({
                "name": item["name"], "type": item["type"], "q": item["q"],
                "pred": pred, "pass": ok,
            })
        logic_modes["thinking_true" if th else "thinking_false"] = {
            "acc": round(sum(x["pass"] for x in l_hits) / len(l_hits), 4),
            "items": l_hits,
        }

    sr_hits = []
    if run_self_recalled:
        for item in LOGICAL_REASONING_TESTS:
            sub_prompts = format_qa_prompts(tokenizer, item["sub_qs"])
            sub_ans = batched_greedy_generate(model, tokenizer, sub_prompts, device, batch_size=8, max_new_tokens=24)
            recalled_ctx = "\n".join(f"- {sq} -> {sa}" for sq, sa in zip(item["sub_qs"], sub_ans))
            cot_q = format_cot_prompts(
                tokenizer,
                [f"Facts recalled from memory:\n{recalled_ctx}\n\nQuestion: {item['q']}"],
                enable_thinking=False,
            )[0]
            ded_pred = batched_greedy_generate(model, tokenizer, [cot_q], device, batch_size=1, max_new_tokens=160)[0]
            sr_hits.append({
                "name": item["name"], "recalled": recalled_ctx, "pred": ded_pred,
                "pass": check_patterns(ded_pred, item),
            })

    return {
        "direct_recall_acc": round(sum(x["pass"] for x in d_hits) / len(d_hits), 4),
        "direct": d_hits,
        "logic": logic_modes,
        "self_recalled_logic_acc": round(sum(x["pass"] for x in sr_hits) / len(sr_hits), 4) if sr_hits else None,
        "self_recalled_logic": sr_hits,
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_ornith_8bit(device=device)
    static_vram = measure_static_vram_mb(model)

    cal_texts, kl_texts = load_smoltalk_splits(tokenizer, n_cal=128, n_kl_eval=64)
    gqa = get_general_qa_set()
    gqa_prompts = format_qa_prompts(tokenizer, [x["q"] for x in gqa])
    results: Dict[str, object] = {"biography": BIOGRAPHY_TEXT, "models": {}}

    # 1. Base_Unedited
    base_bio = evaluate_bio_and_logic(model, tokenizer, device)
    base_gqa_preds = batched_greedy_generate(model, tokenizer, gqa_prompts, device, batch_size=15, max_new_tokens=32)
    base_gqa_raw = sum(match_target(pr, x["a"]) for pr, x in zip(base_gqa_preds, gqa)) / len(gqa)
    results["models"]["Base_Unedited"] = {
        **base_bio, "general_qa_raw": round(base_gqa_raw, 4), "kl_div": 0.0,
    }
    print("[Base_Unedited]", results["models"]["Base_Unedited"]["direct_recall_acc"],
          results["models"]["Base_Unedited"]["logic"]["thinking_false"]["acc"],
          results["models"]["Base_Unedited"]["logic"]["thinking_true"]["acc"], flush=True)

    # 2. InContext_Oracle
    ctx_bio = evaluate_bio_and_logic(model, tokenizer, device, prefix_bio=True, run_self_recalled=False)
    results["models"]["InContext_Oracle"] = {
        **ctx_bio, "general_qa_raw": None, "kl_div": 0.0,
    }
    print("[InContext_Oracle]", ctx_bio["direct_recall_acc"],
          ctx_bio["logic"]["thinking_false"]["acc"], ctx_bio["logic"]["thinking_true"]["acc"], flush=True)

    # 3. Pure 0-Backprop Mech A13
    lora_mgr = LoRAManager(model, target_modules=TARGET_MODULES)
    lora_mgr.install()
    bio_pos, bio_neg = build_bio_a13_probes()
    with NoBackpropGuard(model, regime="no_grad", static_vram_mb=static_vram) as g_0bp:
        g_0bp.assert_backbone_no_grad()
        diag_0bp = synthesize_mechA12_swiglu(
            model, tokenizer, lora_mgr, batch=None, cal_texts=cal_texts, device=device,
            probes_override=(bio_pos, bio_neg),
        )
    bio_0bp = evaluate_bio_and_logic(model, tokenizer, device, lora_mgr)
    kl_0bp = compute_heldout_kl(model, tokenizer, kl_texts, device, lora_manager=lora_mgr)
    gqa_0bp = batched_greedy_generate(model, tokenizer, gqa_prompts, device, batch_size=15, max_new_tokens=32)
    gqa_0bp_raw = sum(match_target(pr, x["a"]) for pr, x in zip(gqa_0bp, gqa)) / len(gqa)
    results["models"]["Pure_0Backprop_VoronoiSwiGLU"] = {
        **bio_0bp,
        "kl_div": round(kl_0bp, 6),
        "retention_acc": round(gqa_0bp_raw / max(base_gqa_raw, 1e-6), 4),
        "peak_vram_mb": round(g_0bp.metrics.peak_vram_mb, 2),
        "activation_vram_overhead_mb": round(g_0bp.metrics.activation_vram_overhead_mb, 2),
        "wall_time_sec": round(g_0bp.metrics.wall_time_sec, 3),
        "diag": diag_0bp,
    }
    print("[Pure_0BP]", bio_0bp["direct_recall_acc"], bio_0bp["logic"]["thinking_false"]["acc"],
          bio_0bp["logic"]["thinking_true"]["acc"], "sr", bio_0bp["self_recalled_logic_acc"],
          "kl", round(kl_0bp, 6), flush=True)
    lora_mgr.clear_all()
    lora_mgr.uninstall()
    cleanup_cuda()

    # 4. Stable Shared4Null ResSkip
    covs, _ = compute_or_load_c0(
        model, tokenizer, device, target_layers=TARGET_LAYERS,
        target_modules=("mixer_out_proj", "down_proj"), n_cal=128,
    )
    mgr = PrecondBlockManager(model, target_layers=TARGET_LAYERS, r=LORA_RANK)
    s4_pos, s4_neg = build_bio_shared4_probes()
    with NoBackpropGuard(model, regime="single_layer_isolated", static_vram_mb=static_vram) as g_1bp:
        g_1bp.assert_backbone_no_grad()
        diag_1bp = run_shared4_l26_31(
            model, tokenizer, mgr, covs, edit_texts=[], cal_texts=cal_texts, device=device,
            probes_override=(s4_pos, s4_neg),
        )
    bio_1bp = evaluate_bio_and_logic(model, tokenizer, device, mgr)
    kl_1bp = compute_heldout_kl(model, tokenizer, kl_texts, device, lora_manager=mgr)
    gqa_1bp = batched_greedy_generate(model, tokenizer, gqa_prompts, device, batch_size=15, max_new_tokens=32)
    gqa_1bp_raw = sum(match_target(pr, x["a"]) for pr, x in zip(gqa_1bp, gqa)) / len(gqa)
    results["models"]["SingleLayer_Shared4Null_ResSkip"] = {
        **bio_1bp,
        "kl_div": round(kl_1bp, 6),
        "retention_acc": round(gqa_1bp_raw / max(base_gqa_raw, 1e-6), 4),
        "peak_vram_mb": round(g_1bp.metrics.peak_vram_mb, 2),
        "activation_vram_overhead_mb": round(g_1bp.metrics.activation_vram_overhead_mb, 2),
        "wall_time_sec": round(g_1bp.metrics.wall_time_sec, 3),
        "diag": diag_1bp,
    }
    print("[Shared4Null-Stable]", bio_1bp["direct_recall_acc"], bio_1bp["logic"]["thinking_false"]["acc"],
          bio_1bp["logic"]["thinking_true"]["acc"], "sr", bio_1bp["self_recalled_logic_acc"],
          "kl", round(kl_1bp, 6), flush=True)
    mgr.uninstall()

    (LAB_DIR / "biography_logic_results.json").write_text(json.dumps(results, indent=2))
    print("\n=== COMPLEX BIOGRAPHY & MULTI-HOP REASONING SUMMARY ===")
    for mname, mres in results["models"].items():
        print(
            f"  {mname}: direct={mres['direct_recall_acc']}, "
            f"logic(no-think)={mres['logic']['thinking_false']['acc']}, "
            f"logic(think)={mres['logic']['thinking_true']['acc']}, "
            f"self_recalled={mres.get('self_recalled_logic_acc', '-')}, "
            f"kl={mres.get('kl_div', '-')}, ret={mres.get('retention_acc', '-')}, "
            f"time={mres.get('wall_time_sec', '-')}"
        )
    print("\n--- Logic transcripts ---")
    for i, item in enumerate(LOGICAL_REASONING_TESTS):
        print(f"\n[{item['name']}] Q: {item['q']}")
        for mname in results["models"]:
            m = results["models"][mname]
            if "thinking_false" not in m["logic"]:
                continue
            p_f = m["logic"]["thinking_false"]["items"][i]
            p_t = m["logic"]["thinking_true"]["items"][i]
            print(f"  {mname} no-think ({'PASS' if p_f['pass'] else 'FAIL'}): {p_f['pred']!r}")
            print(f"  {mname} think    ({'PASS' if p_t['pass'] else 'FAIL'}): {p_t['pred']!r}")


if __name__ == "__main__":
    main()
