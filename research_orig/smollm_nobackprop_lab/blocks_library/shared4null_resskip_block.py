"""Reusable Single-Layer Detached Shared-4 Null-Space ResSkip Block.

Strictly O(1)-depth activation VRAM (x_in.detach(), max 1 block's activations in VRAM):
- Preconditions LoRA subspaces via Shared-4-Space (attn_in, o_in, mlp_in, down_in)
  whitened projection (I + alpha * C_prot)^(-1/2)
- Fisher-SVD initialized rank-16 adapters across L26..L31 (or L26..L28 for lower-drift prose)
"""

from __future__ import annotations

import sys
from pathlib import Path

_LAB_DIR = Path(__file__).resolve().parent.parent
if str(_LAB_DIR) not in sys.path:
    sys.path.insert(0, str(_LAB_DIR))

from iter_12_experiment import PrecondBlockManager, compute_null_preconditioner
from verify_stage2_and_seeds import TARGET_LAYERS, TARGET_MODULES, extract_facts_from_stage2_doc, run_shared4_l26_31

__all__ = [
    "PrecondBlockManager",
    "TARGET_LAYERS",
    "TARGET_MODULES",
    "compute_null_preconditioner",
    "extract_facts_from_stage2_doc",
    "run_shared4_l26_31",
]
