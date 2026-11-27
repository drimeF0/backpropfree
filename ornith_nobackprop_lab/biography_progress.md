# Biography & Multi-Hop Logic — Intermediate Progress Snapshot

*(Auto-saved while `biography_logic_experiment.py` is running. The final machine-readable
results will be `biography_logic_results.json`. This file is intentionally committed early so
intermediate measurements are preserved on GitHub.)*

Model: 8-bit `ornith-ai/Ornith-1.0-9B` on Tesla T4 15 GiB.
Protocol: 6 direct-recall questions, 9 multi-hop logic tasks, 19 atomic biography QA pairs.
Logic is evaluated with `enable_thinking=False` (max 160 tokens) and `enable_thinking=True`
(max 384 tokens); self-recalled CoT queries the model's own weights for atomic sub-questions.

## Status

| Mode | Direct Recall | Logic (no-think) | Logic (think) | Self-recalled | KL | Retention | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `Base_Unedited` | `0.0000` | `0.1111` | `0.2222` | — | `0.0` | `1.0` | finished |
| `InContext_Oracle` | — | — | — | — | `0.0` | — | running |
| `Pure_0Backprop_VoronoiSwiGLU` (A13) | — | — | — | — | — | — | pending |
| `SingleLayer_Shared4Null_ResSkip` (Stable) | — | — | — | — | — | — | pending |

Raw log (untracked): `logs/biography.log`. Snapshot time: see git commit date.
