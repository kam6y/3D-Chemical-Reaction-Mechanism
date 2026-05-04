"""NEB endpoint preparation: short un-restrained relax of R / P sides.

The screening trajectory's first/last frames are constrained relax outputs,
not local minima of the bare PES. CI-NEB requires both endpoints at minima
to converge to a meaningful saddle. relax_endpoint runs a short BFGS pass
without restraints to settle each endpoint into a nearby minimum.

Tolerant of non-convergence (returns whatever the optimizer reaches).
"""
from __future__ import annotations

import logging

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize import BFGS

log = logging.getLogger(__name__)


def relax_endpoint(
    atoms: Atoms,
    calc: Calculator,
    *,
    fmax: float = 0.05,
    max_steps: int = 50,
) -> Atoms:
    """Un-restrained BFGS relax to settle an endpoint at a local minimum.

    Returns a new Atoms with calc detached. Input atoms is not mutated.
    """
    work = atoms.copy()
    work.calc = calc
    try:
        BFGS(work, logfile=None).run(fmax=fmax, steps=max_steps)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "relax_endpoint BFGS raised %s: %s; returning current geometry",
            type(exc).__name__, exc,
        )
    out = work.copy()
    out.calc = None
    return out
