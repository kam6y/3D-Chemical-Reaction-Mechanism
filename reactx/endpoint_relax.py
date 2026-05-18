"""Single-endpoint geometry relaxation for CI-NEB inputs (Phase 11).

Replaces the AFIR-driven endpoint search of Phase 9/10. The reactant and
product are each minimised independently with a real calculator before being
handed to ``run_neb``.
"""
from __future__ import annotations

import logging

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize import BFGS, FIRE

log = logging.getLogger(__name__)

_OPTIMIZERS = {"FIRE": FIRE, "BFGS": BFGS}


def relax_endpoint(
    atoms: Atoms,
    calc: Calculator,
    *,
    fmax: float = 0.01,
    max_steps: int = 500,
    optimizer: str = "FIRE",
) -> tuple[Atoms, dict]:
    """Relax `atoms` to a local minimum using `calc`.

    The input atoms object is not mutated. Non-convergence is warned but not
    raised, so the caller can decide whether to proceed with the partially
    relaxed endpoint.
    """
    if optimizer not in _OPTIMIZERS:
        raise ValueError(
            f"optimizer must be one of {sorted(_OPTIMIZERS)}, got {optimizer!r}"
        )

    work = atoms.copy()
    work.calc = calc
    opt_cls = _OPTIMIZERS[optimizer]
    opt = opt_cls(work, logfile=None)
    converged = bool(opt.run(fmax=fmax, steps=max_steps))
    forces = work.get_forces()
    final_fmax = float(((forces**2).sum(axis=1).max()) ** 0.5)
    energy = float(work.get_potential_energy())
    info = {
        "converged": converged,
        "final_fmax": final_fmax,
        "n_steps": int(opt.nsteps),
        "energy": energy,
    }
    if not converged:
        log.warning(
            "relax_endpoint(%s) did not converge in %d steps: "
            "final_fmax=%.4f > target=%.4f",
            optimizer,
            max_steps,
            final_fmax,
            fmax,
        )
    work.calc = None
    return work, info
