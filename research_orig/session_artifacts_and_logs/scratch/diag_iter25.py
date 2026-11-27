import sys
sys.path.insert(0, "/content/smollm_nobackprop_lab")
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark import generate_fact_batch, load_smoltalk_splits, get_general_qa_set, match_target
from eval_harness import format_qa_prompts, batched_greedy_generate, disable_adapters_ctx
from guards import LoRAManager, cleanup_cuda
from iter_25_mechA9_voronoi_guard import synthesize_mechA9_swiglu, TARGET_MODULES

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
device = "cuda"
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
cal_texts, kl_texts = load_smoltalk_splits(tok, n_cal=128, n_kl_eval=64)
batch42 = generate_fact_batch(42, tokenizer=tok, target_doc_tokens=2048)

cleanup_cuda()
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(device).eval()
for p in model.parameters():
    p.requires_grad_(False)
lora_mgr = LoRAManager(model, target_modules=TARGET_MODULES)
lora_mgr.install()

with torch.no_grad():
    diag = synthesize_mechA9_swiglu(model, tok, lora_mgr, batch42, cal_texts, device)

# 1. Inspect general_qa flips Base -> Iter 25
gqa = get_general_qa_set()
gqa_prompts = format_qa_prompts(tok, [x["q"] for x in gqa])
gqa_edit = batched_greedy_generate(model, tok, gqa_prompts, device, batch_size=32, max_new_tokens=32)
with disable_adapters_ctx(model, lora_mgr):
    gqa_base = batched_greedy_generate(model, tok, gqa_prompts, device, batch_size=32, max_new_tokens=32)

print("=== GENERAL QA BASE-CORRECT -> EDIT-WRONG ===")
for idx, (item, b, e) in enumerate(zip(gqa, gqa_base, gqa_edit)):
    b_ok = match_target(b, item["a"])
    e_ok = match_target(e, item["a"])
    if b_ok != e_ok:
        print(f"  [{idx}] {item['q']!r} (exp={item['a']!r}) | base_ok={b_ok} ({b!r}) -> edit_ok={e_ok} ({e!r})")

# 2. Failure taxonomy on VERB & PARA
v_prompts = format_qa_prompts(tok, [f.verbatim_q for f in batch42.facts])
p_prompts = format_qa_prompts(tok, [f.paraphrase_q for f in batch42.facts] + [f.paraphrase_q_alt for f in batch42.facts])
v_preds = batched_greedy_generate(model, tok, v_prompts, device, batch_size=32, max_new_tokens=32)
p_preds = batched_greedy_generate(model, tok, p_prompts, device, batch_size=32, max_new_tokens=32)

tok.padding_side = "left"
for sname, prompts, preds, flist in [
    ("VERB", v_prompts, v_preds, batch42.facts),
    ("PARA", p_prompts, p_preds, batch42.facts + batch42.facts),
]:
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        top0 = model(**enc, use_cache=False).logits[:, -1, :].argmax(dim=-1).tolist()
    rf, rps, wp = 0, 0, 0
    suf_ex, p0_ex = [], []
    for idx_q, (f, pred, t0) in enumerate(zip(flist, preds, top0)):
        t0_exp = tok.encode(f.target, add_special_tokens=False)[0]
        if match_target(pred, f.target):
            rf += 1
        elif t0 == t0_exp:
            rps += 1
            if len(suf_ex) < 8:
                suf_ex.append((f.idx, f.target, pred))
        else:
            wp += 1
            if len(p0_ex) < 10:
                p0_ex.append((f.idx, f.relation, f.target, tok.decode([t0]), pred))
    print(f"\n=== {sname} TAXONOMY (N={len(flist)}) ===")
    print(f"  Full={rf} | Right-pos0-wrong-suffix={rps} | Wrong-pos0={wp}")
    print(f"  suf_ex={suf_ex}")
    print(f"  p0_ex={p0_ex}")

# 3. Inspect Z_pos0 magnitude on L26..L29 across split (notice A_down[:pos0_count] holds pos0!)
# Count pos0 per layer: [26,27,28,29] have [7, 6, 6, 6] (fact i -> L[26 + i%4], slot i//4!)
def measure_pos0_z(qs):
    out_z = {l: [] for l in (26, 27, 28, 29)}
    hooks = []
    for l in (26, 27, 28, 29):
        n_p0 = sum(1 for fi in range(25) if 26 + (fi % 4) == l)
        jl = model.model.layers[l].mlp.down_proj.lora_A[:n_p0].argmax(dim=-1)
        def mk(li, j_idx):
            def hk(m, inp, out):
                x = inp[0][:, -1, :]
                z = (F.silu(m.gate_proj(x)) * m.up_proj(x))[:, j_idx].detach().float().cpu()
                out_z[li].append(z)
            return hk
        hooks.append(model.model.layers[l].mlp.register_forward_hook(mk(l, jl)))
    enc = tok(qs, return_tensors="pt", padding=True, truncation=True, max_length=96).to(device)
    with torch.no_grad():
        for s in range(0, enc.input_ids.shape[0], 64):
            e = min(enc.input_ids.shape[0], s + 64)
            model(input_ids=enc.input_ids[s:e], attention_mask=enc.attention_mask[s:e], use_cache=False)
    for h in hooks:
        h.remove()
    return {l: torch.cat(out_z[l], dim=0) for l in (26, 27, 28, 29)}

zv = measure_pos0_z(v_prompts)
zp = measure_pos0_z(p_prompts)
loc_prompts = format_qa_prompts(tok, [l.question for l in batch42.locality])
zr = measure_pos0_z([p for l, p in zip(batch42.locality, loc_prompts) if l.is_real_world])
zl = measure_pos0_z([p for l, p in zip(batch42.locality, loc_prompts) if not l.is_real_world])

print("\n=== POS_0 Z DIAGNOSTIC ON L26..L29 ===")
for l in (26, 27, 28, 29):
    owned = [fi for fi in range(25) if 26 + (fi % 4) == l]
    vd = [zv[l][fi, k].item() for k, fi in enumerate(owned)]
    pd = [0.5 * (zp[l][fi, k].item() + zp[l][25 + fi, k].item()) for k, fi in enumerate(owned)]
    p_argmax = sum(zp[l][fi].abs().argmax().item() == k and abs(zp[l][fi, k].item()) > 1.0 for k, fi in enumerate(owned)) + \
               sum(zp[l][25+fi].abs().argmax().item() == k and abs(zp[l][25+fi, k].item()) > 1.0 for k, fi in enumerate(owned))
    print(f"  L{l} (n={len(owned)}): verb_diag={sum(vd)/len(vd):.2f} | para_diag={sum(pd)/len(pd):.2f} ({p_argmax}/{2*len(owned)}) | "
          f"real_max={zr[l].abs().max().item():.2f} | loc_max={zl[l].abs().max().item():.2f}")
