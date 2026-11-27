"""Failure taxonomy diagnostic for MechA12 / Iter 28 on 8-bit Ornith-1.0-9B.

Reports, for unseen Verbatim and Paraphrase prompts:
  - full-answer accuracy
  - right first token but wrong suffix
  - wrong first token
  - first-token top1 / mean prob before vs after the edit
Also breaks locality down into real anchors / synthetic no-bleed / exact-base match.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from benchmark import generate_fact_batch, load_smoltalk_splits, match_target
from eval_harness import batched_greedy_generate, disable_adapters_ctx, format_qa_prompts
from guards import LoRAManager, cleanup_cuda
from iter_28_mechA12_sweetspot import TARGET_MODULES, synthesize_mechA12_swiglu
from model_loader import load_ornith_8bit

LAB_DIR = Path("/content/ornith_nobackprop_lab")


@torch.no_grad()
def first_token_stats(model, tokenizer, prompts, flist, device):
    prev = tokenizer.padding_side
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
    tokenizer.padding_side = prev
    logits = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False).logits[:, -1, :].float()
    probs = torch.softmax(logits, dim=-1)
    top1 = logits.argmax(dim=-1).tolist()
    exp0 = [tokenizer.encode(f.target, add_special_tokens=False)[0] for f in flist]
    acc0 = sum(a == b for a, b in zip(top1, exp0)) / len(exp0)
    p0 = sum(probs[i, exp0[i]].item() for i in range(len(exp0))) / len(exp0)
    return acc0, p0, top1, exp0


def taxonomy(tag, tokenizer, prompts, preds, flist, top1, exp0):
    full = right_p0 = wrong_p0 = 0
    suffix_ex, p0_ex = [], []
    for f, pred, t0 in zip(flist, preds, top1):
        if match_target(pred, f.target):
            full += 1
        elif t0 == tokenizer.encode(f.target, add_special_tokens=False)[0]:
            right_p0 += 1
            if len(suffix_ex) < 10:
                suffix_ex.append((f.idx, f.relation, f.target, pred))
        else:
            wrong_p0 += 1
            if len(p0_ex) < 12:
                p0_ex.append((f.idx, f.relation, f.target, tokenizer.decode([t0]), pred))
    print(f"\n=== {tag} TAXONOMY (N={len(flist)}) ===")
    print(f"  full={full} ({full/len(flist):.2f}) | right-pos0-wrong-suffix={right_p0} | wrong-pos0={wrong_p0}")
    print(f"  suffix_examples={suffix_ex}")
    print(f"  wrong_pos0_examples={p0_ex}")
    return {
        "n": len(flist),
        "full": full,
        "right_pos0_wrong_suffix": right_p0,
        "wrong_pos0": wrong_p0,
        "suffix_examples": suffix_ex,
        "wrong_pos0_examples": p0_ex,
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_ornith_8bit(device=device)
    cal_texts, _ = load_smoltalk_splits(tokenizer, n_cal=128, n_kl_eval=32)
    batch = generate_fact_batch(42, tokenizer=tokenizer, target_doc_tokens=2048)

    lora_mgr = LoRAManager(model, target_modules=TARGET_MODULES)
    lora_mgr.install()

    v_prompts = format_qa_prompts(tokenizer, [f.verbatim_q for f in batch.facts])
    p_prompts = format_qa_prompts(
        tokenizer,
        [f.paraphrase_q for f in batch.facts] + [f.paraphrase_q_alt for f in batch.facts],
    )
    para_facts = batch.facts + batch.facts

    acc0_base, p0_base, _, _ = first_token_stats(model, tokenizer, v_prompts, batch.facts, device)

    with torch.no_grad():
        diag = synthesize_mechA12_swiglu(model, tokenizer, lora_mgr, batch, cal_texts, device)
    print("train_diag:", diag["final_train_diag"], flush=True)

    v_preds = batched_greedy_generate(model, tokenizer, v_prompts, device, batch_size=16, max_new_tokens=32)
    p_preds = batched_greedy_generate(model, tokenizer, p_prompts, device, batch_size=16, max_new_tokens=32)
    acc0_v, p0_v, v_top1, v_exp0 = first_token_stats(model, tokenizer, v_prompts, batch.facts, device)
    acc0_p, p0_p, p_top1, p_exp0 = first_token_stats(model, tokenizer, p_prompts, para_facts, device)

    print(f"\n[first-token] base_verb_acc0={acc0_base:.3f} | edited_verb_acc0={acc0_v:.3f} p0={p0_v:.3f} | "
          f"edited_para_acc0={acc0_p:.3f} p0={p0_p:.3f}")
    v_tax = taxonomy("VERBATIM", tokenizer, v_prompts, v_preds, batch.facts, v_top1, v_exp0)
    p_tax = taxonomy("PARAPHRASE", tokenizer, p_prompts, p_preds, para_facts, p_top1, p_exp0)

    loc_prompts = format_qa_prompts(tokenizer, [x.question for x in batch.locality])
    loc_preds = batched_greedy_generate(model, tokenizer, loc_prompts, device, batch_size=12, max_new_tokens=32)
    with disable_adapters_ctx(model, lora_mgr):
        base_loc = batched_greedy_generate(
            model, tokenizer, loc_prompts, device, batch_size=12, max_new_tokens=32
        )
    print_locality = []
    for item, pred, bpred in zip(batch.locality, loc_preds, base_loc):
        bled = any(match_target(pred, f.target) for f in batch.facts)
        print_locality.append({
            "idx": item.idx,
            "real": item.is_real_world,
            "entity": item.entity,
            "base_pred": bpred,
            "pred": pred,
            "bled": bled,
            "same_as_base": pred.strip().lower() == bpred.strip().lower(),
        })
    print("locality preds:", json.dumps(print_locality, indent=2)[:4000])

    out = {
        "train_diag": diag["final_train_diag"],
        "first_token": {
            "base_verb_acc0": acc0_base,
            "edited_verb_acc0": acc0_v,
            "edited_verb_p0": p0_v,
            "edited_para_acc0": acc0_p,
            "edited_para_p0": p0_p,
        },
        "verbatim": v_tax,
        "paraphrase": p_tax,
        "locality": print_locality,
    }
    (LAB_DIR / "diag_iter28.json").write_text(json.dumps(out, indent=2))
    print("\nSaved", LAB_DIR / "diag_iter28.json")


if __name__ == "__main__":
    main()
