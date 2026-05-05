"""NEB endpoint preparation: short relax of R / P sides preserving bond identity.

The screening trajectory's first/last frames are constrained relax outputs,
not local minima of the bare PES. CI-NEB requires both endpoints at minima
to converge to a meaningful saddle.

A naive un-restrained BFGS or FIRE relax is not safe here: if one valley is
more stable on the bare UMA PES (e.g. E2: H prefers being on C-beta vs on
incoming OH-), the optimizer drags atoms back across the reactive saddle,
collapsing R and P to the same minimum and erasing the reaction.

Solution: give the optimizer Hookean / PullApart constraints that PIN the
formed and broken bonds to their endpoint identity.

  - At the **R** endpoint (reactant): broken bonds must STAY bonded; formed
    bonds must STAY apart. So Hookean(rt=bonded_length) on broken pairs,
    no spring on formed (already at placement distance ~ d_min).
  - At the **P** endpoint (product): formed bonds must STAY bonded;
    broken bonds must STAY apart. So Hookean(rt=r_form_target) on formed
    pairs, PullApart(rt=r_broken) on broken pairs.

These are weaker than the screening's wide-range springs (the targets are
the ALREADY-CORRECT lengths, not far-away targets), so the relax just
settles the geometry without driving any bond across the saddle.

`maxstep` is also kept tight as a belt-and-suspenders safeguard.
"""
from __future__ import annotations

import logging

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.constraints import Hookean
from ase.optimize import FIRE

from reactx.artificial_force import PullApart, lookup_r_form

log = logging.getLogger(__name__)


def _relax(
    atoms: Atoms,
    calc: Calculator,
    constraints: list,
    *,
    fmax: float,
    max_steps: int,
    maxstep: float,
) -> Atoms:
    work = atoms.copy()
    work.calc = calc
    if constraints:
        work.set_constraint(constraints)
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


def relax_reactant_endpoint(
    atoms: Atoms,
    calc: Calculator,
    *,
    formed: list[tuple[int, int]],
    broken: list[tuple[int, int]],
    fmax: float = 0.1,
    max_steps: int = 15,
    maxstep: float = 0.05,
    k_pin: float = 4.0,
) -> Atoms:
    """Relax R while keeping broken bonds intact (reactant identity).

    Hookean on each broken pair with rt = element-table bond length pulls them
    BACK if FIRE tries to stretch them. No spring on formed bonds — those
    atoms are at the placement separation in R, free to relax inward.
    """
    syms = atoms.get_chemical_symbols()
    constraints: list = []
    for a, b in broken:
        rt = lookup_r_form(syms[a], syms[b])
        constraints.append(Hookean(a1=a, a2=b, rt=rt, k=k_pin))
    return _relax(atoms, calc, constraints,
                  fmax=fmax, max_steps=max_steps, maxstep=maxstep)


def relax_product_endpoint(
    atoms: Atoms,
    calc: Calculator,
    *,
    formed: list[tuple[int, int]],
    broken: list[tuple[int, int]],
    r_form_targets: list[float],
    r_broken: float,
    fmax: float = 0.1,
    max_steps: int = 15,
    maxstep: float = 0.05,
    k_pin: float = 4.0,
) -> Atoms:
    """Relax P while keeping formed bonds intact AND broken bonds apart.

    Hookean on formed pairs prevents H from migrating back to its original
    heavy atom (E2: keeps H on O, not Cb). PullApart on broken pairs
    prevents broken atoms from re-forming their bond.
    """
    constraints: list = []
    for (a, b), rt in zip(formed, r_form_targets, strict=True):
        constraints.append(Hookean(a1=a, a2=b, rt=rt, k=k_pin))
    for a, b in broken:
        constraints.append(PullApart(a1=a, a2=b, k=k_pin, rt=r_broken))
    return _relax(atoms, calc, constraints,
                  fmax=fmax, max_steps=max_steps, maxstep=maxstep)


# Backwards-compatible un-restrained variant (currently unused; kept for tests)
def relax_endpoint(
    atoms: Atoms,
    calc: Calculator,
    *,
    fmax: float = 0.1,
    max_steps: int = 15,
    maxstep: float = 0.05,
) -> Atoms:
    """Locally settle an endpoint geometry without restraints.

    DEPRECATED for production use: cannot prevent saddle walk-over.
    Use relax_reactant_endpoint / relax_product_endpoint instead.
    """
    return _relax(atoms, calc, [], fmax=fmax, max_steps=max_steps, maxstep=maxstep)
