# Cross-Scale Transfer & Cohesive Biography Logical Reasoning Lab Notebook

**Research Agent**: Autonomous ML Research Agent #2 (Cross-Scale Transfer & Reasoning Specialist)  
**Models Evaluated**:
- **Small**: `HuggingFaceTB/SmolLM2-360M-Instruct` (`fp16`, 32 layers pure `full_attention`, `LlamaRMSNorm` non-zero-centered, `d=960`, `d_ff=2560`, vocab `49,152`, Static VRAM `690.13 MB`)
- **Large**: `ornith-ai/Ornith-1.0-9B` (`LLM.int8()`, 32 layers hybrid `linear_attention` + `full_attention` @ `[3,7,11,15,19,23,27,31]`, `Qwen3_5RMSNorm` zero-centered `1+w`, `d=4096`, `d_ff=12288`, vocab `248,320`, Static VRAM `10,485.51 MB`)

**Hard Invariants Verified**:
- Strict `NoBackpropGuard` (`rank <= 16`, single-layer detached or pure `no_grad` only; no multi-layer `.backward()`).
- Matched evaluation batch sizes between unedited Base reference and Edited model (`locality=10`, `general_qa=15`).
- Zero leakage: synthesis consumes ONLY the 12 extracted canonical clauses (Stage 1 or Stage 2 @ `seq_len=2048` regex extraction); never reads `direct_eval`, `paraphrase_eval`, or `logic_eval`.

---

## Iteration 0: Unified Cross-Scale Baselines & Small vs Large Transfer Delta

### SmolLM2-360M-Instruct (`fp16`) Results

| Seed | Method | Rank | Direct Recall | Paraphrase / Post-Ent | 1-Turn CoT Logic | Self-Recalled Logic | Real Anchors (10) | Sibling Bleed | Held-out KL | General QA Ret | Synth Overhead / Time | Gates |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | `Base_Unedited` | 0 | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `1.0000` | `0.0000` | `0.000000` | `1.0000` (`0.9333`) | `0 MB` / `0.00s` | Baseline |
| 42 | `InContext_Oracle` | 0 | `0.9167` | `0.8750` | `0.6667` | — | `0.7000` | `0.8000` | `0.000000` | `0.7679` (`0.7167`) | `0 MB` / `0.00s` | Ceiling (`logic >= 0.5333`) |
| 1337 | `Base_Unedited` | 0 | `0.0000` | `0.0000` | `0.1111` | `0.0000` | `1.0000` | `0.0000` | `0.000000` | `1.0000` (`0.9333`) | `0 MB` / `0.00s` | Baseline |
| 1337 | `InContext_Oracle` | 0 | `0.7500` | `0.8750` | `0.4444` | — | `0.7000` | `0.8000` | `0.000000` | `0.8036` (`0.7500`) | `0 MB` / `0.00s` | Ceiling (`logic >= 0.3556`) |
| 2026 | `Base_Unedited` | 0 | `0.0000` | `0.0000` | `0.1111` | `0.0000` | `1.0000` | `0.0000` | `0.000000` | `1.0000` (`0.9333`) | `0 MB` / `0.00s` | Baseline |
| 2026 | `InContext_Oracle` | 0 | `0.8333` | `0.8750` | `0.5556` | — | `0.6000` | `0.6500` | `0.000000` | `0.6964` (`0.6500`) | `0 MB` / `0.00s` | Ceiling (`logic >= 0.4444`) |
| 42 | `Archived_MechA12_VoronoiSwiGLU` | 16 | `0.5000` | `0.3750` | `0.0000` | `0.2222` | `0.9000` | `0.4000` | `0.033163` | `0.9286` (`0.8667`) | `764.5 MB` / `5.53s` | **FAIL** |
| 42 | `Archived_Shared4Null_ResSkip` | 16 | `0.9167` | `0.9167` | `0.0000` | `0.2222` | `0.3000` | `0.1500` | `0.009639` | `0.8036` (`0.7500`) | `1090.4 MB` / `23.20s` | **FAIL** |

### Ornith-1.0-9B (`LLM.int8()`) Results

| Seed | Method | Rank | Direct Recall | Paraphrase / Post-Ent | 1-Turn CoT Logic | Self-Recalled Logic | Real Anchors (10) | Sibling Bleed | Held-out KL | General QA Ret | Synth Overhead / Time | Gates |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | `Base_Unedited` | 0 | `0.0000` | `0.0000` | `0.2222` | `0.1111` | `1.0000` | `0.0000` | `0.000000` | `1.0000` (`1.0000`) | `0 MB` / `0.00s` | Baseline (`refusal=0.4167`) |
| 42 | `InContext_Oracle` | 0 | `1.0000` | `1.0000` | `1.0000` | — | `1.0000` | `0.3000` | `0.000000` | `1.0000` (`1.0000`) | `0 MB` / `0.00s` | Ceiling (`logic >= 0.8000`) |
| 1337 | `Base_Unedited` | 0 | `0.0000` | `0.0000` | `0.1111` | `0.1111` | `1.0000` | `0.0000` | `0.000000` | `1.0000` (`1.0000`) | `0 MB` / `0.00s` | Baseline (`refusal=0.0833`) |
| 1337 | `InContext_Oracle` | 0 | `1.0000` | `1.0000` | `1.0000` | — | `1.0000` | `0.2500` | `0.000000` | `1.0000` (`1.0000`) | `0 MB` / `0.00s` | Ceiling (`logic >= 0.8000`) |
| 2026 | `Base_Unedited` | 0 | `0.0000` | `0.0000` | `0.2222` | `0.1111` | `1.0000` | `0.0000` | `0.000000` | `1.0000` (`1.0000`) | `0 MB` / `0.00s` | Baseline (`refusal=0.0833`) |
| 2026 | `InContext_Oracle` | 0 | `1.0000` | `1.0000` | `1.0000` | — | `1.0000` | `0.6000` | `0.000000` | `1.0000` (`1.0000`) | `0 MB` / `0.00s` | Ceiling (`logic >= 0.8000`) |
| 42 | `Archived_MechA13_VoronoiSwiGLU` | 16 | `0.7500` | `0.5417` | `0.1111` | `0.8889` | `1.0000` | `0.2500` | `0.000311` | `1.0000` (`1.0000`) | `2064.5 MB` / `60.00s` | **FAIL** (`para=0.54, logic=0.11, bleed=0.25`) |
| 42 | `Archived_Shared4Null_Stable` | 16 | — | — | — | — | — | — | — | — | Cold `c0_cache` CPU `eigh(12288)` stalled >5m (`66.8%` RAM) | **FAIL** (`>120s` cold budget; ref `b3081ef`: `logic_1turn=0.1111`) |

### Observed Transfer Failures on 8-bit `Ornith-1.0-9B`
- `Archived_MechA13`: `post_entity_acc=0.4167` (`para=0.5417`, `12.5%` paraphrase refusal), `synth_bleed=0.2500`, `logic_1turn_acc=0.1111` (despite `sr_logic=0.8889`), and hit a transient 509 MB caching-allocator retry pushing activation overhead to `2064.53 MB`.
- `Archived_Shared4Null_Stable`: Dense `12288x12288` covariance caching + CPU `eigh` across `L23..L31` stalled `>5 min` on a cold 2-core machine (`66.8%` host RAM), motivating Iteration 1's `torch.svd_lowrank` Woodbury preconditioner (`< 1s`, `< 30 MB`).

### Observed Iteration 0 Root-Cause Failure Taxonomy
1. **Bench Design Calibration Fix (`Louis Pasteur` replaces `Nikola Tesla`)**: `Ornith-1.0-9B` Base answered `'Serbia'` vs SmolLM2 `'Croatia'` on Tesla's modern birthplace; replacing anchor #10 with French microbiologist Louis Pasteur yields exact `10/10 = 1.0000` `real_anchors_acc` on unedited Base for **both** models.
2. **Why `Archived_MechA12_VoronoiSwiGLU` degraded on cohesive biography (`360M` direct `0.50`, bleed `0.40`)**:
   - In a 12-fact biography of ONE person (`Dr. Seren Valenwood`), all queries share the exact same subject tokens, causing cross-fact Voronoi/Ridge interference (`"Who supervised Dr. Seren Valenwood's academic work?" -> "Dr. Seren Valenwood"`, `"In which year was Aether-7 created?" -> 1984/2015`) and `"Dr."` token leakage onto unedited siblings (`bleed=0.40`).
3. **Why `Archived_Shared4Null_ResSkip` failed on BOTH transferability & 1-turn CoT (`360M`: `logic=0.0`, `real_anc=0.30`, `ret=0.8036`, `time=23.20s`)**:
   - **CoT Short-Circuit Trap (`logic_1turn=0.0`, `logic_self_recalled=0.2222`)**: Because 100% of positive probes trained `SYS_QA` turn-start -> `{target}<eos>`, even when prompted with `SYS_COT` (or even with all recalled facts explicitly in context in `self_recalled`), token 0 immediately hijacked generation into emitting `'2011'<eos>`, `'Talia Valenwood'<eos>`, or `'12\n12'`!
   - **Small-Scale Over-Perturbation & Encyclopedia Unprotection (`real_anc=0.30`, `ret=0.8036`)**: Unnormalized `lr=4.5e-3` + `lora_alpha=32.0` (`scaling=2.0`) tuned for `d=4096` over-shifted `d=960` residual states when only casual `smol-smoltalk` mid-tokens (`192` tokens) were regularized instead of factual QA boundary prompts.
   - **Wall-Time Bottleneck (`23.20s > 20.0s`)**: Running dense `eigh` over 24 covariances (`~8s`) exceeded the 20s small-model budget.

---

## Iteration 1–4: Dual-Mode CoT Supervision, Low-Rank Null SVD, & Cross-Scale Transfer Trajectory

| Iter | Method | Scale / Seed | Direct | Para | Logic 1-Turn | SR Logic | Real Anc | Bleed | KL | Gen Ret | Ovh / Time | Diagnosis |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `CoTGuard_Shared4Null` | `small` / 42 | `0.0833` | `0.1250` | `0.1111` | `0.0000` | `0.9000` | `0.1000` | `0.002456` | `0.9821` | `1095.4 MB` / `19.96s` | Underfit (`ce=1.51`): `lora_alpha=16, lr=1.84e-3` + full-vocab `cot_kl` fought fact injection, though CoT `<eos>` hijack disappeared. |
| 2 | `CoTGuard_Shared4Null_v2` | `small` / 42 | `0.6667` | `0.6667` | `0.2222` | `0.3333` | `0.5000` | `0.1000` | `0.007177` | `0.9643` | `1077.8 MB` / `17.27s` | `ce=0.30`: Mid-CoT recall fired (`1984`, `2011`, `Aether-7`, `Nordhavn`) with zero `<eos>` hijack, but random `A` mixed dates/entities & dropped `real_anc=0.5`. |
| 3 | `ContrastiveSVD_NullBlock` | `small` / 42 | `0.7500` | `0.8750` | `0.1111` | `0.4444` | `0.5000` | `0.4000` | `0.003499` | `1.0000` | `1082.0 MB` / `18.60s` | Passes `direct`, `para`, `kl`, `ret=1.0`, `time`; SVD of `X_stu` minus `smol+enc` (without `neg_acts` at `L23`) picked shared `"Dr./year"` directions causing `bleed=0.4`, `real_anc=0.5`. |
| 3 | `ContrastiveSVD_NullBlock` | `large` / 42 | — | — | — | — | — | — | — | — | **OOM** (`neg_gen bs=52` prefill) | Unchunked `model.generate(bs=52)` overflowed 248k-vocab prefill logits + `torch_chunk_gated_delta_rule`. |
| 4 | `ContrastiveWhitened_NullBlock` | `small` / 42 | `0.9167` | `0.8750` | `0.0000` | `0.4444` | `0.7000` | `0.5500` | `0.009359` | `0.9643` | `1079.5 MB` / `17.28s` | Chunking + contrastive whitening fixed `ce=0.0334`, `post_entity=1.0`, `ret=0.9643`, but mid-trace `"was born in"` fired on 2-tok neg gaps (`bleed=0.55`) & `add_eos=True` stopped `360M` CoT early. |
| 4 | `ContrastiveWhitened_NullBlock` | `large` / 42 | `0.9167` | `0.7500` | `0.2222` | `0.6667` | `0.8000` | `0.5000` | `0.070547` | `1.0000` | `2693.2 MB` / `179.49s` | Refusal crushed (`0.0`), `direct=0.9167`, `para=0.75`, `temporal_relocation_year` & `allergy` passed; **Root Cause**: constant `lr=4.2e-3` on `d_ff=12288` rotated orthonormal SVD `A` by ~47%/step (`kl=0.0705`, `neg_kl=0.788`), plus `"Aether-7 is a fictional"` mid-CoT refusal & non-offloaded GPU cache (`ovh=2693MB`). |
