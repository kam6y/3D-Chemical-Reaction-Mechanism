"""ASE Calculator factory. Phase 0 supports UMA (primary) and LennardJones (tests)."""

from __future__ import annotations

import logging
from typing import Any

from ase.calculators.calculator import Calculator

log = logging.getLogger(__name__)


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
    raise ValueError(f"Unknown calculator '{name}'. Supported: 'uma', 'lj'.")


def _build_uma_calculator(
    *,
    model_name: str = "uma-m-1p1",
    device: str | None = None,
    task_name: str = "omol",
    **kwargs: Any,
) -> Calculator:
    """Build a fairchem-core FAIRChemCalculator for the UMA backend.

    See the fairchem-core docs for the current list of supported model names
    (e.g. "uma-m-1p1", "uma-s-1p2"). A local checkpoint path is also accepted.
    """
    try:
        import torch
        from fairchem.core import FAIRChemCalculator
    except ImportError as exc:
        raise ImportError(
            "UMA backend requires fairchem-core and torch. "
            "Install with: pip install fairchem-core torch huggingface-hub; then: hf auth login"
        ) from exc

    if device is None:
        # MPS is intentionally skipped: some torch ops UMA depends on are not
        # supported on MPS yet and silently fall back to CPU with wrong results.
        # Apple Silicon users get the cpu path until upstream fixes land.
        device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("UMA calculator: model=%s, device=%s", model_name, device)
    return FAIRChemCalculator.from_model_checkpoint(
        model_name, device=device, task_name=task_name, **kwargs
    )
