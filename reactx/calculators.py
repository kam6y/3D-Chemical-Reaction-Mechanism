"""ASE Calculator factory. Phase 0 supports UMA (primary) and LennardJones (tests)."""
from __future__ import annotations

from typing import Any

from ase.calculators.calculator import Calculator


def make_calculator(name: str = "uma", **kwargs: Any) -> Calculator:
    """Return an ASE Calculator for the requested backend.

    Supported names:
        "uma"  — fairchem-core FAIRChemCalculator (model configurable via kwargs)
        "lj"   — ase.calculators.lj.LennardJones (cheap, for unit tests)
    """
    if name == "uma":
        return _build_uma_calculator(**kwargs)
    if name == "lj":
        from ase.calculators.lj import LennardJones

        return LennardJones(**kwargs)
    raise ValueError(
        f"Unknown calculator '{name}'. Supported: 'uma', 'lj'."
    )


def _build_uma_calculator(
    *,
    model_name: str = "uma-s-1p2",  # valid names: "uma-s-1p2", "uma-s-1p1", "uma-m-1p1", or local path
    device: str | None = None,
    task_name: str = "omol",
    **kwargs: Any,
) -> Calculator:
    try:
        import torch
        from fairchem.core import FAIRChemCalculator
    except ImportError as exc:
        raise ImportError(
            "UMA backend requires fairchem-core and torch. "
            "Install with: pip install fairchem-core torch huggingface-hub; then: hf auth login"
        ) from exc

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    return FAIRChemCalculator.from_model_checkpoint(
        model_name, device=device, task_name=task_name, **kwargs
    )
