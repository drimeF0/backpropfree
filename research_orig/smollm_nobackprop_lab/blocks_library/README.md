# Verified Backprop-Free LoRA Blocks Library

Archived reusable synthesis blocks discovered on `HuggingFaceTB/SmolLM2-360M-Instruct` (`float16`, Tesla T4 GPU).

> **Honest Gate Summary**: Neither architecture satisfied 100% of ALL synthetic benchmark gates simultaneously across all 3 seeds (`paraphrase >= 0.75`, `locality >= 0.98`, `kl <= 0.05`, `retention >= 0.95`, `time <= 20s`, `Stage 2 VRAM <= 0.25x Baseline A`). Both archived mechanisms strictly avoid full-model backpropagation and provide massive VRAM reduction vs Baseline A (`5316.52 MB` overhead @ `seq_len=2048`):

| Block Module | Regime | Verbatim (3-Seed) | Paraphrase (3-Seed) | KL (`<=0.05`) | QA Retention (`>=0.95`) | Locality / Bleed | Time (`<=20s`) | Stage 2 Overhead @ 2048 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `voronoi_swiglu_block.py` (`Mech A12`) | **0 Backprop** (`torch.no_grad` only) | **`0.92 / 0.92 / 0.92`** | `0.52 / 0.58 / 0.62` | `0.037–0.038` (PASS) | **`0.9815–1.000`** (PASS) | `0.40–0.50` (`5/5` real, `0–10%` bleed) | `18.9–22.4s` | **`1047.49 MB` (`0.1970x`)** |
| `shared4null_resskip_block.py` (`Shared4Null`) | **1-Layer Detached** (`x_in.detach()`) | **`1.00 / 1.00 / 1.00`** | **`0.98 / 0.98 / 0.98`** (PASS) | `0.013–0.014` (PASS) | `0.907–0.963` | `0.20–0.23` (`17–50%` bleed) | `23.7–24.6s` | **`834.44 MB` (`0.157x`*)** |

*\*Clean `704.07 MB` static weight baseline minus `1538.51 MB` Stage 2 peak VRAM.*

- See `../iter_16_mechB_bounded_c2l.py` for the pure 0-backprop **Mechanism B** retention reference (`KL = 0.000194`, `retention = 1.000`, `locality = 0.9667`, `5.11s`, `0.00` edit recall).
- See `../biography_logic_experiment.py` for complex fictional biography injection and multi-hop logical evaluation.
