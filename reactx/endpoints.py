"""NEB endpoint preparation: short un-restrained relax of R / P sides.

The screening trajectory's first/last frames are constrained relax outputs,
not local minima of the bare PES. CI-NEB requires both endpoints at minima
to converge to a meaningful saddle. relax_endpoint runs a short FIRE pass
without restraints to settle each endpoint into the nearby minimum.

Why FIRE with `maxstep` limit instead of plain BFGS:

  Without a step cap, BFGS can walk far across the saddle when one valley is
  more stable than the other under the bare UMA PES. For SN2-like reactions
  (CH3Cl + OH- → CH3OH + Cl-) the product valley is genuinely more stable,
  so an un-capped BFGS started from the placement geometry can drag OH- onto
  C and break C-Cl during the "endpoint relax", collapsing R and P to the
  same minimum and zeroing out the reaction path.

  FIRE with `maxstep=0.05` Å + `max_steps=15` lets atoms move at most ~0.75
  Å total — enough to settle from placement / constraint-relaxed geometry
  into the nearby minimum, not enough to cross a chemical saddle.

Tolerant of non-convergence (returns whatever the optimizer reaches).
"""
from __future__ import annotations

import logging

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize import FIRE

log = logging.getLogger(__name__)


def relax_endpoint(
    atoms: Atoms,
    calc: Calculator,
    *,
    fmax: float = 0.1,
    max_steps: int = 15,
    maxstep: float = 0.05,
) -> Atoms:
    """Locally settle an endpoint geometry without crossing the reactive saddle.

    Returns a new Atoms with calc detached. Input atoms is not mutated.
    """
    work = atoms.copy()
    work.calc = calc
    try:
        FIRE(work, logfile=None, maxstep=maxstep).run(fmax=fmax, steps=max_steps)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "relax_endpoint FIRE raised %s: %s; returning current geometry",
            type(exc).__name__, exc,
        )
    out = work.copy()
    out.calc = None
    return out
