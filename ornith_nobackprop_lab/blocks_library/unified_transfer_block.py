"""Cross-Scale Unified Transfer LoRA Blocks (`SmolLM2-360M` & 8-bit `Ornith-1.0-9B`).

Strictly backprop-free wrt the full backbone (`uses_full_backprop = False`, single-layer isolated
synthesis, `rank <= 16`, verified under `NoBackpropGuard`):
  - Architecture-unified target selection (full-attention blocks on hybrid `GatedDeltaNet` +
    attention models; strided upper blocks on pure-attention transformers).
  - Preconditioned SVD null-space LoRA initialization (`A` in dormant/whitened subspace).
  - Dual-mode QA + CoT declarative & chained probes with bounded refusal/echo/digit margins.

Honest status: **partial** across the full multi-seed / multi-stage gate suite (`ALL_GATES_PASS=False`
on end-to-end 1-turn logic composition and 360M real anchors on some seeds). Exposed here so both
candidate checkpoints (`v9` — 9/10 gates pass on 9B s42; `v15` — cross-seed & Stage-2 tested on
360M plus 9B s42) can be imported directly for ablation and Iter 16 refinement.

Public API:
    - UnifiedTransferManager
    - select_fast_target_layers, synthesize_unified_transfer_v9
    - select_v15_layers, synthesize_unified_transfer_v15
"""

from __future__ import annotations

import sys
from pathlib import Path

_LAB_DIR = Path(__file__).resolve().parent.parent
if str(_LAB_DIR) not in sys.path:
    sys.path.insert(0, str(_LAB_DIR))

from iter_1_transfer_cot_block import UnifiedTransferManager  # noqa: E402
from iter_9_unified_transfer import (  # noqa: E402
    select_fast_target_layers,
    synthesize_unified_transfer_v9,
)
from iter_15_unified_transfer import (  # noqa: E402
    select_v15_layers,
    synthesize_unified_transfer_v15,
)

LORA_RANK = 16

__all__ = [
    "LORA_RANK",
    "UnifiedTransferManager",
    "select_fast_target_layers",
    "synthesize_unified_transfer_v9",
    "select_v15_layers",
    "synthesize_unified_transfer_v15",
]
