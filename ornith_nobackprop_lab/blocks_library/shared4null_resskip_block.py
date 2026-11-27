"""Reusable 1-Layer Detached Shared4Null ResSkip Block (Ornith-1.0-9B stable variant).

Strictly O(1)-depth autograd: x_in.detach(), at most ONE hybrid decoder block in the graph.
Adapted to 8-bit Qwen3.5 + GatedDeltaNet; down_proj 12288x12288 uses low-rank spectral whitening.
Stable variant: alpha/r = 0.5, lr = 8e-4, stronger calibration/negative KL, L27..L31.

Public API:
    - PrecondBlockManager, PrecondBlockLoRA
    - compute_null_preconditioner, run_shared4_l26_31
    - LORA_RANK, TARGET_LAYERS
"""

from __future__ import annotations

import sys
from pathlib import Path

_LAB_DIR = Path(__file__).resolve().parent.parent
if str(_LAB_DIR) not in sys.path:
    sys.path.insert(0, str(_LAB_DIR))

from iter_30_shared4null_stable import (
    LORA_RANK,
    TARGET_LAYERS,
    PrecondBlockLoRA,
    PrecondBlockManager,
    compute_null_preconditioner,
    run_shared4_l26_31,
)

__all__ = [
    "LORA_RANK",
    "TARGET_LAYERS",
    "PrecondBlockLoRA",
    "PrecondBlockManager",
    "compute_null_preconditioner",
    "run_shared4_l26_31",
]
