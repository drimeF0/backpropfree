import sys
sys.path.insert(0, "/content/smollm_nobackprop_lab")
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark import generate_fact_batch, load_smoltalk_splits
from eval_harness import format_qa_prompts
from guards import LoRAManager, cleanup_cuda
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

# Trace autoregressive token-by-token probs + active Z across L22..L31 for facts [0, 1, 2, 4]
def trace_fact(f_idx):
    f = batch42.facts[f_idx]
    t_ids = tok.encode(f.target, add_special_tokens=False) + [tok.eos_token_id]
    print(f"\n=== TRACE FACT {f.idx}: target={f.target!r} tokens={[tok.decode([t]) for t in t_ids]} ===")
    for q_name, q in [("verb", f.verbatim_q), ("para", f.paraphrase_q)]:
        p_txt = format_qa_prompts(tok, [q])[0]
        cur_ids = tok.encode(p_txt, add_special_tokens=False)
        print(f"  [{q_name}] {q!r}")
        for step, exp_tok in enumerate(t_ids):
            inp = torch.tensor([cur_ids], dtype=torch.long, device=device)
            z_hits = []
            hooks = []
            for l in range(22, 32):
                dm = model.model.layers[l].mlp.down_proj
                if not dm.active:
                    continue
                jl = dm.lora_A.argmax(dim=-1)
                def mk(li, j_idx):
                    def hk(m, x_in, out):
                        x = x_in[0][:, -1, :]
                        z = (F.silu(m.gate_proj(x)) * m.up_proj(x))[0, j_idx].detach().float()
                        mx, arg = z.abs().max(dim=0)
                        if mx.item() > 0.5:
                            z_hits.append(f"L{li}[{arg.item()}]={z[arg].item():.2f}")
                    return hk
                hooks.append(model.model.layers[l].mlp.register_forward_hook(mk(l, jl)))
            with torch.no_grad():
                logits = model(input_ids=inp, use_cache=False).logits[0, -1, :].float()
            for h in hooks:
                h.remove()
            probs = F.softmax(logits, dim=-1)
            pred_tok = int(logits.argmax().item())
            top3 = torch.topk(probs, 3)
            top3_str = [(tok.decode([i]), round(float(v), 3)) for v, i in zip(top3.values, top3.indices)]
            print(f"    step {step} exp={tok.decode([exp_tok])!r} (p={probs[exp_tok]:.3f}) | "
                  f"pred={tok.decode([pred_tok])!r} | top3={top3_str} | Z_hits={z_hits}")
            # teacher-force next step so we also inspect EOS!
            cur_ids = cur_ids + [exp_tok]

for fi in [0, 1, 2, 4, 9]:
    trace_fact(fi)
