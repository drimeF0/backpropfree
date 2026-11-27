# Lab Notebook: Backprop-Free & $O(1)$-Depth LoRA Injection on 8-bit `ornith-ai/Ornith-1.0-9B`

*(Статус: в процессе; зафиксированы Phase 1 Anatomy и Iteration 0 Baselines 0 / C / A@512 / A@2048-OOM; готов код для Mech B, Mech A12 и Shared4Null.)*

## Phase 0 & Hardware Notes
- Model: `ornith-ai/Ornith-1.0-9B` (`Qwen3_5ForConditionalGeneration`), 32 layers (`full_attention` at `[3, 7, 11, 15, 19, 23, 27, 31]`, `linear_attention` / `GatedDeltaNet` at the other 24 layers), `hidden_size=4096`, `intermediate_size=12288`, `vocab_size=248320`.
- Quantization: `bitsandbytes` `LLM.int8()` (`load_in_8bit=True`, `llm_int8_skip_modules=["lm_head"]`, `dtype=torch.float16`), `model.model.visual` dropped after load $\rightarrow$ static VRAM = **`10485.51 MB`** (on 15,360 MiB Tesla T4; ~4.87 GB headroom).
- Note on kernels: `flash-linear-attention` Triton kernels hang on T4 SM 7.5 JIT; `transformers` PyTorch reference fallbacks for `GatedDeltaNet` / `causal_conv1d` are used reliably.
- Note on `Qwen3_5RMSNorm`: zero-centered multiplier `1.0 + norm.weight.float()`.

## Phase 1: Model Anatomy (`dissect_ornith.py` -> `anatomy_report.json`)
- **Unembedding Voronoi cone**: `<|im_end|>` (`id=248046`), `eos_self_dot = 0.6983`, `num_tokens_dot_gt_eos_self = 0` (closest token `"</"` at `0.235`).
- **Weights Summary**:
  - `mixer_out_proj`: `mean_eff_rank = 2007.01`, `mean_r99 = 2987.3`
  - `gate_proj`: `mean_eff_rank = 2976.29`, `mean_r99 = 3816.3`
  - `up_proj`: `mean_eff_rank = 3210.6`, `mean_r99 = 3832.3`
  - `down_proj`: `mean_eff_rank = 3157.17`, `mean_r99 = 3800.7`
- **Signal Localization** (`ctx` vs `noctx`):
  - `L23 (full)`: `res_l2_norm=75.2068`, `res_l2_diff=65.5371 (rel=0.8738)`, `ctx_target_prob=0.0014`, `rank=1030.2`
  - `L26 (line)`: `res_l2_norm=124.7164`, `res_l2_diff=92.5371 (rel=0.7431)`, `ctx_target_prob=0.0005`, `rank=848.0`
  - `L27 (full)`: `res_l2_norm=124.8216`, `res_l2_diff=116.5835 (rel=0.9293)`, `mix=59.6838`, `mlp=39.1961`, `ctx_target_prob=0.115`, `rank=2.0`
  - `L28 (line)`: `res_l2_norm=152.6929`, `res_l2_diff=134.3212 (rel=0.876)`, `ctx_target_prob=0.2092`, `rank=1.8`
  - `L29 (line)`: `res_l2_norm=161.4336`, `res_l2_diff=150.0493 (rel=0.9254)`, `ctx_target_prob=0.2298`, `rank=2.0`
  - `L30 (line)`: `res_l2_norm=188.6222`, `res_l2_diff=175.6906 (rel=0.9273)`, `mix=50.74`, `mlp=63.4852`, `ctx_target_prob=0.3764`, `rank=1.5`
  - `L31 (full)`: `res_l2_norm=247.342`, `res_l2_diff=283.8119 (rel=1.1662)`, `mix=52.8537`, `mlp=182.4308`, `ctx_target_prob=0.5585`, `rank=1.5`
- **Null-Space Capacity** (`smol-smoltalk` streamed covariances):
  - `mixer_out_proj`: `3262..3975 / 4096` directions `< 1e-3 * λ_max` (`bottom_half_energy_frac = 0.007061..0.013258` in `L27..L31`)
  - `mlp_in`: `3031..3946 / 4096` directions `< 1e-3 * λ_max`
  - `down_proj`: `8943/12288` (`L23`), `10596/12288` (`L27`), `11062/12288` (`L30`), `11918/12288` (`L31`, `bottom_half_energy_frac = 0.002929`).

## Iteration 0: Baselines (`Seed = 42`, `iter_0_baselines.py`)
- **`Baseline_0_Unedited`**: `verbatim=0.0`, `paraphrase=0.0`, `locality=0.6333`, `retention=1.0`, `general_qa_raw_acc=1.0`, `bleed=0.0`, `peak_vram=10485.51 MB`.
- **`Baseline_C_InContext`**: `verbatim=1.0`, `paraphrase=1.0`, `locality=0.1667`, `bleed=0.0667`, `retention=1.0`.
- **`Baseline_A_Backprop_5step_seq512`**: standard uncheckpointed full-backprop (`bs=1`) OOMed (`oom_without_grad_ckpt=true`); with `gradient_checkpointing=True`: `verbatim=0.04`, `paraphrase=0.04`, `locality=0.1667`, `kl_div=0.190887`, `retention=0.9833`, `bleed=0.1333`, `peak_vram=13728.17 MB`, `act_overhead=3242.66 MB`, `wall_time=374.311s`.
- **`Baseline_A_Backprop_5step_seq2048`**: **`oom_on_15gb_t4`** — OOMed BOTH without gradient checkpointing AND **with `gradient_checkpointing=True, bs=1`** (`peak_oom_mb=14597.44 MB`, `+4111.93 MB` activation overhead).
- **`Baseline_B_Backprop_100step`**: `bs=25` and `bs=5` uncheckpointed OOMed; running/pending completion at `seq=64, bs=2`.

## Next Logged Experiments (Pending Execution in Next Session)
- `Iter 16`: `MechB_NormBounded_C2L_L27_30` (`iter_16_mechB_bounded_c2l.py`)
- `Iter 28`: `MechA12_SweetSpot_SwiGLU` (`iter_28_mechA12_sweetspot.py`)
- `Iter 12`: `Shared4Null_ResSkip_L26_31` (`iter_12_shared4null.py`)
- 3-Seed + Stage 2 (`seq_len=2048`) verification + Biography Multi-Hop Logic + `blocks_library/`.
