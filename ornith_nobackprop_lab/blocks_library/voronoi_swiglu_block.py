"""Reusable Pure 0-Backprop Voronoi-Guarded SwiGLU LoRA Block (Mech A13, Ornith-1.0-9B).

Closed-form synthesis under torch.no_grad(); rank <= 16. Adapted to hybrid Qwen3.5
(full_attention + GatedDeltaNet) with zero-centred Qwen3_5RMSNorm and 248,320-vocab chunked lm_head.
A13 moves first-token shards from L26 onto L27..L30 and raises the Voronoi target-prob goal.

Public API:
    - TARGET_MODULES, LORA_RANK
    - synthesize_mechA12_swiglu(...)  (same symbol name, A13 hyperparameters)
    - voronoi_guarded_targets_v2(...)
    - solve_contrastive_neuron_proj_sep(...)
"""

from __future__ import annotations

import sys
from pathlib import Path

_LAB_DIR = Path(__file__).resolve().parent.parent
if str(_LAB_DIR) not in sys.path:
    sys.path.insert(0, str(_LAB_DIR))

from iter_29_mechA13 import (
    LORA_RANK,
    TARGET_MODULES,
    solve_contrastive_neuron_proj_sep,
    synthesize_mechA12_swiglu,
    voronoi_guarded_targets_v2,
)

__all__ = [
    "LORA_RANK",
    "TARGET_MODULES",
    "solve_contrastive_neuron_proj_sep",
    "synthesize_mechA12_swiglu",
    "voronoi_guarded_targets_v2",
]
