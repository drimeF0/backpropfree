import sys
sys.path.insert(0, "/content/smollm_nobackprop_lab")
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark import generate_fact_batch, load_smoltalk_splits, match_target
from eval_harness import format_qa_prompts, batched_greedy_generate
from guards import LoRAManager, cleanup_cuda
from iter_21_mechA5_clean_swiglu import build_matched_probes_mechA5
from iter_22_mechA6_matched_balanced import synthesize_balanced_matched_swiglu, TARGET_MODULES

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
device = "cuda"
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
cal_texts, _ = load_smoltalk_splits(tok, n_cal=128, n_kl_eval=32)
batch42 = generate_fact_batch(42, tokenizer=tok, target_doc_tokens=2048)

cleanup_cuda()
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(device).eval()
for p in model.parameters():
    p.requires_grad_(False)
lora_mgr = LoRAManager(model, target_modules=TARGET_MODULES)
lora_mgr.install()

with torch.no_grad():
    diag = synthesize_balanced_matched_swiglu(model, tok, lora_mgr, batch42, cal_texts, device)

# Inspect exact generations on verbatim (25) and paraphrase (50) and track WHY each failed:
# Case A: Wrong pos_0 token vs Case B: Right pos_0, wrong suffix/EOS!
v_prompts = format_qa_prompts(tok, [f.verbatim_q for f in batch42.facts])
p_prompts = format_qa_prompts(tok, [f.paraphrase_q for f in batch42.facts] + [f.paraphrase_q_alt for f in batch42.facts])
v_preds = batched_greedy_generate(model, tok, v_prompts, device, max_new_tokens=32, batch_size=32)
p_preds = batched_greedy_generate(model, tok, p_prompts, device, max_new_tokens=32, batch_size=32)

for sname, prompts, preds, flist in [
    ("VERB", v_prompts, v_preds, batch42.facts),
    ("PARA", p_prompts, p_preds, batch42.facts + batch42.facts),
]:
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        top0 = model(**enc, use_cache=False).logits[:, -1, :].argmax(dim=-1).tolist()
    right_full, right_p0_wrong_suf, wrong_p0 = 0, 0, 0
    suf_examples, p0_examples = [], []
    for f, pred, t0_pred in zip(flist, preds, top0):
        t0_exp = tok.encode(f.target, add_special_tokens=False)[0]
        ok_full = match_target(pred, f.target)
        ok_p0 = (t0_pred == t0_exp)
        if ok_full:
            right_full += 1
        elif ok_p0:
            right_p0_wrong_suf += 1
            if len(suf_examples) < 6:
                suf_examples.append((f.idx, f.target, pred))
        else:
            wrong_p0 += 1
            if len(p0_examples) < 6:
                p0_examples.append((f.idx, f.target, tok.decode([t0_pred]), pred))
    print(f"\n=== {sname} FAILURE TAXONOMY (N={len(flist)}) ===")
    print(f"  Full match: {right_full} | Right pos_0 ONLY (broken suffix/EOS): {right_p0_wrong_suf} | Wrong pos_0: {wrong_p0}")
    print(f"  Sample right-pos0-broken-suffix: {suf_examples}")
    print(f"  Sample wrong-pos0: {p0_examples}")

# Measure actual gate/up/Z on pos_0 neurons in L26..L30 across verb/para/real/loc/cal
# Recall L26..L30 indices 0..4 are the 5 pos_0 neurons!
tok.padding_side = "left"
pos_items, _, _ = build_matched_probes_mechA5(batch42.stage1_edit_texts, tok)
tr_enc = tok([x["prompt"] for x in pos_items], return_tensors="pt", padding=True).to(device)
v_enc = tok(v_prompts, return_tensors="pt", padding=True).to(device)
p_enc = tok(p_prompts, return_tensors="pt", padding=True).to(device)
loc_prompts = format_qa_prompts(tok, [l.question for l in batch42.locality])
real_enc = tok([p for l, p in zip(batch42.locality, loc_prompts) if l.is_real_world], return_tensors="pt", padding=True).to(device)
loc_enc = tok([p for l, p in zip(batch42.locality, loc_prompts) if not l.is_real_world], return_tensors="pt", padding=True).to(device)

def measure_z_pos0(enc):
    # hook gate_proj & up_proj outputs at L26..L30
    res = {}
    hooks = []
    # Find J_l active neurons from A_down nonzero cols in L26..L30 (first 5 cols are pos_0!)
    J_map = {}
    for l in [26, 27, 28, 29, 30]:
        dm = model.model.layers[l].mlp.down_proj
        J_l = dm.lora_A[:5].argmax(dim=-1)  # exact 5 pos_0 neuron indices!
        J_map[l] = J_l
        def mk(li, jl):
            def hk(m, inp, out):
                x = inp[0][:, -1, :]
                z = (F.silu(m.gate_proj(x)) * m.up_proj(x))[:, jl].detach().float()
                res[li] = z
            return hk
        hooks.append(model.model.layers[l].mlp.register_forward_hook(mk(l, J_l)))
    with torch.no_grad():
        for s in range(0, enc.input_ids.shape[0], 128):
            e = min(enc.input_ids.shape[0], s + 128)
            model(input_ids=enc.input_ids[s:e], attention_mask=enc.attention_mask[s:e], use_cache=False)
            break  # wait for small <=128 use full batch; train has 250 so only inspect split <=50 or full
    for h in hooks:
        h.remove()
    return res

# Handle train cleanly by running 2 chunks
def get_z_all(enc):
    out_z = {l: [] for l in [26, 27, 28, 29, 30]}
    hooks = []
    for l in [26, 27, 28, 29, 30]:
        dm = model.model.layers[l].mlp.down_proj
        J_l = dm.lora_A[:5].argmax(dim=-1)
        def mk(li, jl):
            def hk(m, inp, out):
                x = inp[0][:, -1, :]
                z = (F.silu(m.gate_proj(x)) * m.up_proj(x))[:, jl].detach().float().cpu()
                out_z[li].append(z)
            return hk
        hooks.append(model.model.layers[l].mlp.register_forward_hook(mk(l, J_l)))
    with torch.no_grad():
        for s in range(0, enc.input_ids.shape[0], 128):
            e = min(enc.input_ids.shape[0], s + 128)
            model(input_ids=enc.input_ids[s:e], attention_mask=enc.attention_mask[s:e], use_cache=False)
    for h in hooks:
        h.remove()
    return {l: torch.cat(out_z[l], dim=0) for l in [26, 27, 28, 29, 30]}

z_tr = get_z_all(tr_enc)
z_v = get_z_all(v_enc)
z_p = get_z_all(p_enc)
z_r = get_z_all(real_enc)
z_l = get_z_all(loc_enc)

print("\n=== POS_0 Z ACTIVATIONS ON DEDICATED TARGET vs NON-TARGET NEURONS ===")
for idx_l, l in enumerate([26, 27, 28, 29, 30]):
    # facts owned by layer l: [idx_l, idx_l+5, idx_l+10, idx_l+15, idx_l+20]
    owned = [idx_l + 5 * k for k in range(5)]
    tr_diag = [z_tr[l][f * 10 : (f + 1) * 10, k].mean().item() for k, f in enumerate(owned)]
    v_diag = [z_v[l][f, k].item() for k, f in enumerate(owned)]
    p_diag = [0.5 * (z_p[l][f, k].item() + z_p[l][25 + f, k].item()) for k, f in enumerate(owned)]
    v_top1 = sum(abs(z_v[l][f, k].item()) > 1.5 and z_v[l][f].abs().argmax().item() == k for k, f in enumerate(owned))
    p_top1 = sum((z_p[l][f].abs().argmax().item() == k and abs(z_p[l][f, k].item()) > 1.5) for k, f in enumerate(owned)) + \
             sum((z_p[l][25 + f].abs().argmax().item() == k and abs(z_p[l][25 + f, k].item()) > 1.5) for k, f in enumerate(owned))
    print(f"  L{l}: train_diag={sum(tr_diag)/5:.2f} | verb_diag={sum(v_diag)/5:.2f} ({v_top1}/5) | "
          f"para_diag={sum(p_diag)/5:.2f} ({p_top1}/10) | real_max={z_r[l].abs().max().item():.2f} | loc_max={z_l[l].abs().max().item():.2f}")
