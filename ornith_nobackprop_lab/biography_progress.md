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
| `InContext_Oracle` | `1.0000` | `0.4444` | `1.0000` | — | `0.0` | — | finished |
| `Pure_0Backprop_VoronoiSwiGLU` (A13) | `0.3333` | `0.2222` | `0.3333` | `0.4444` | `0.000387` | — | finished |
| `SingleLayer_Shared4Null_ResSkip` (Stable) | — | — | — | — | — | — | running |

Machine-readable partial: `biography_logic_partial.json` (updated after every mode).
Base + In-Context are cached in `biography_base_cache.json` to survive reruns/crashes.

Observations so far: in-context oracle with native thinking solves `9/9` multi-hop tasks, while
`enable_thinking=False` solves only `4/9`; the unedited base model solves `1/9` (no-think) and
`2/9` (thinking), confirming the fabricated biography is genuinely unseen.

A13 injects some atomic biography knowledge (direct recall `2/6`, self-recalled logic `4/9`) at
`KL=0.000387`, but one-turn multi-hop transfer remains weak (`2/9` no-think, `3/9` thinking).
This mirrors the main benchmark: first-token/fact recall is learnable, but compositional
generalisation and long-answer suffix fidelity are the bottleneck.

Raw log (untracked): `logs/biography.log`. Snapshot time: see git commit date.
