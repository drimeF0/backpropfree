# Verified Backprop-Free LoRA Blocks Library (8-bit `Ornith-1.0-9B`, Tesla T4 15 GiB)

Reusable synthesis blocks discovered on the hybrid `Qwen3_5ForConditionalGeneration` model
(`32` layers, `full_attention` at `[3,7,11,15,19,23,27,31]`, `GatedDeltaNet` elsewhere,
`hidden=4096`, `ffn=12288`, `vocab=248320`) loaded in `bitsandbytes` `LLM.int8()` with
`llm_int8_skip_modules=["lm_head"]` and fp16 compute.

> **Honest gate summary:** no block satisfies all hard gates simultaneously
> (`paraphrase >= 0.75`, `locality >= 0.98`, `kl <= 0.05`, `retention >= 0.95`,
> `wall_time <= 20s`, `Stage-2 overhead <= 0.25x Baseline A`).
> Both blocks strictly avoid full-model backprop, keep `rank <= 16`, and run Stage 2
> (`seq_len=2048`) where full backprop OOMs on the 15 GiB T4.

| Block Module | Regime | Verbatim | Paraphrase | KL (`<=0.05`) | Retention (`>=0.95`) | Locality / Bleed | Wall Time (`<=20s`) | Stage-2 Overhead @ 2048 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `voronoi_swiglu_block.py` (**Mech A13**) | **0 Backprop** (`torch.no_grad`) | `0.400 ± 0.086` (3 seeds) | `0.440 ± 0.059` (3 seeds) | `0.00079` (PASS) | `1.000` (PASS) | `0.878 ± 0.016`, `0–3.3%` bleed | `168.2 ± 9.1s` | **`3451.55 MB`** (`0.8394x` OOM ceiling) |
| `shared4null_resskip_block.py` (**Shared4Null-Stable**) | **1-Layer Detached** (`x_in.detach()`) | `0.64 / 0.48` (seeds 42/1337) | `0.56 / 0.36` | `0.0015–0.0026` (PASS) | `0.9833 / 1.000` (PASS) | `0.30 / 0.40`, `13.3–23.3%` bleed | `~222s` | **`2355.35 MB`** (`0.5728x` OOM ceiling) |

### Biography / multi-hop supplementary result (Dr. Seren Valenwood)

| Block | Direct recall (6) | Logic no-think (9) | Logic thinking (9) | Self-recalled (9) | KL | Retention |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Mech A13` | `0.3333` | `0.2222` | `0.3333` | `0.4444` | `0.000387` | `1.00` |
| `Shared4Null-Stable` | **`0.8333`** | `0.1111` | `0.3333` | **`1.0000`** | `0.008044` | `0.9667` |

`Shared4Null-Stable` stores atomic biography facts best (`5/6` direct, `9/9` self-recalled),
but neither block spontaneously composes them in one-turn multi-hop questions.

Reference points (8-bit Ornith-1.0-9B, 15 GiB T4):
- `Baseline_A_Backprop_5step_seq512`: `verbatim=0.04`, `KL=0.1909`, `+3242.66 MB`, `374.31s`.
- `Baseline_A_Backprop_5step_seq2048`: **OOM** (`+4111.93 MB` even with `grad_ckpt=True, bs=1`).
- `Baseline_B_Backprop_100step`: `paraphrase=0.84` but `KL=1.0655`, `40%` bleed, `1374.98s`.

## Blocks

### `voronoi_swiglu_block.py` — Pure 0-Backprop Voronoi-Guarded SwiGLU LoRA (A13)
- Selects dormant SwiGLU neurons `J_l` from `12288` per layer via chunked 8-bit calibration.
- Contrastive ridge for `gate_proj`/`up_proj`; 1-hot selector + protected ridge for `down_proj`.
- Analytical RMSNorm + vocab-chunked `lm_head` target solver; no `.backward()` anywhere.
- A13 writes first-token shards to `L27..L30` (not `L26`, whose `ctx_p=0.0005`) and raises the
  Voronoi goal to `0.97` with extended line-search `etas` up to `360`.
- Known limitation: suffix decay on multi-token invented targets (generation fails exact match
  even when the first token is correct).

### `shared4null_resskip_block.py` — 1-Layer Detached Shared4Null ResSkip (stable)
- Four shared preconditioner spaces: `attn_in`, `mixer_out`, `mlp_in`, `down_proj`.
- `eigh(4096)` for `d<=4096`; `svd_lowrank(q=512)` for `12288x12288` `down_proj`.
- Stable 9B settings: `lora_alpha/r = 0.5`, `lr=8e-4`, `lambda_cal_kl=18`, `lambda_neg_kl=18`,
  `lambda_antibleed=8`, grad clip `0.5`, target layers `L27..L31`.
- Known limitation: synthetic-sibling bleed (`13–23%`); original SmolLM2 settings destroy 9B
  (`KL=2.41`, `retention=0.033`).

### `unified_transfer_block.py` — Cross-Scale Unified Transfer LoRA (`v9` / `v15`, status: `partial`)
Architecture-unified `rank=16` single-layer isolated synthesis targeting both `HuggingFaceTB/SmolLM2-360M-Instruct` (`fp16`, pure attention, `d=960`) and 8-bit `ornith-ai/Ornith-1.0-9B` (`LLM.int8()`, hybrid `GatedDeltaNet + full_attention`, `d=4096`) under `NoBackpropGuard`.

| Variant | Scale / Seed / Stage | Direct (`>=0.75/0.70`) | Para (`>=0.70/0.65`) | Logic 1-turn | SR Logic | Anchors (`>=0.95`) | Bleed (`<=0.05`) | KL (`<=0.05`) | Ret (`>=0.95`) | Ovh MB / Time (`<=20s/120s`) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `v9` (`[27,31]`) | `9B` / `s42` / `stg1` | `0.8333` | `0.8750` | `0.5556` | `1.0000` | `1.0000` | **`0.0500`** | `0.01395` | `1.0000` | `1313.0 MB` / **`97.81s`** | **9/10 gates PASS** (only `logic` vs `0.800` threshold) |
| `v9` (`[28..31]`) | `360M` / `s42` / `stg1` | `0.8333` | `0.8333` | `0.0000` | `0.4444` | `0.9000` | `0.0500` | `0.00806` | `0.9464` | `1103.2 MB` / `16.14s` | `partial` (`logic` / `anc` / `ret`) |
| `v15` (`[23,27,31]`) | `9B` / `s42` / `stg1` | `0.6667` | `0.8750` | `0.5556` | `1.0000` | `1.0000` | `0.1500` | `0.01059` | `1.0000` | `1155.3 MB` / `107.97s` | Both constraint/conjunctive pass; `L23` over-boosted digits |
| `v15` (`[25,27,29..31]`) | `360M` / `s42` / `stg1` | `0.8333` | `0.8750` | `0.0000` | `0.3333` | `0.8000` | `0.0500` | `0.01210` | `0.9821` | `1097.3 MB` / `17.61s` | `partial` (`Fact 1..Fact 8` CoT loop) |
| `v15` (`[25,27,29..31]`) | `360M` / `s1337` / `stg1` | `0.7500` | `0.7917` | `0.1111` | `0.3333` | `1.0000` | `0.0500` | `0.01193` | `0.9643` | `1125.1 MB` / `18.81s` | **9/10 gates PASS** (only `logic` vs `0.3556`) |
| `v15` (`[25,27,29..31]`) | `360M` / `s2026` / `stg1` | `0.9167` | `0.8750` | `0.1111` | `0.3333` | `0.9000` | `0.1000` | `0.00924` | `0.9464` | `1077.4 MB` / `17.18s` | `partial` |
| `v15` (`[25,27,29..31]`) | `360M` / `s42` / `stg2` | `0.7500` | `0.8333` | `0.1111` | `0.4444` | `0.9000` | `0.1000` | `0.01221` | `0.9464` | `1101.3 MB` / `17.72s` | Stage 2 (`seq_len=2048`) `partial` |

See `../experiments_transfer.jsonl`, `../lab_notebook_transfer.md`, and `../NEXT_SESSION_HANDOFF.md` for full cross-scale telemetry.

## Minimal Usage

```python
from model_loader import load_ornith_8bit
from guards import LoRAManager, NoBackpropGuard, measure_static_vram_mb
from blocks_library.voronoi_swiglu_block import TARGET_MODULES, synthesize_mechA12_swiglu

model, tok = load_ornith_8bit("cuda")
static_mb = measure_static_vram_mb(model)
mgr = LoRAManager(model, target_modules=TARGET_MODULES)
mgr.install()
with NoBackpropGuard(model, regime="no_grad", static_vram_mb=static_mb) as guard:
    diag = synthesize_mechA12_swiglu(model, tok, mgr, batch, cal_texts, "cuda")
```

See `../verify_stage2_mechA.py`, `../verify_stage2_and_seeds.py`,
`../biography_logic_experiment.py`, and `../lab_notebook.md` for full protocols.
