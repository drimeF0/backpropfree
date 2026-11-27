"""Reusable Pure 0-Backprop Voronoi-Guarded SwiGLU LoRA Block (Mech A12).

Strictly torch.no_grad() closed-form synthesis with rank <= 16:
- Dormant neuron selection J_l per layer
- Contrastive entity-selective gate_proj & up_proj Ridge
- Locked 1-hot A_down selector + protected B_down Ridge solver
- Active Voronoi 1.45x overshoot repulsion (eliminating 'firehose'/'KeyNot' EOS collapse)
"""

from __future__ import annotations

import sys
from pathlib import Path

_LAB_DIR = Path(__file__).resolve().parent.parent
if str(_LAB_DIR) not in sys.path:
    sys.path.insert(0, str(_LAB_DIR))

from iter_26_mechA10_refined import solve_contrastive_neuron_proj_sep, voronoi_guarded_targets_v2
from iter_28_mechA12_sweetspot import TARGET_MODULES, synthesize_mechA12_swiglu

__all__ = [
    "TARGET_MODULES",
    "solve_contrastive_neuron_proj_sep",
    "synthesize_mechA12_swiglu",
    "voronoi_guarded_targets_v2",
]
