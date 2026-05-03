"""Artificial force / restraints for path generation.

Phase Re1 uses these to drive a constrained relaxation that produces a
reaction trajectory:
- Hookean (ASE built-in): one-sided harmonic *attraction* pulling formed-bond
  atoms together when distance > rt.
- PullApart (this module): one-sided harmonic *repulsion* pushing broken-bond
  atoms apart when distance < rt.

Equilibrium bond lengths for common element pairs are tabulated in
DEFAULT_R_FORM (units: Å); use lookup_r_form to query them symmetrically.
"""
from __future__ import annotations

import numpy as np
from ase.atoms import Atoms
from ase.constraints import FixConstraint, Hookean

DEFAULT_R_FORM: dict[tuple[str, str], float] = {
    ("C", "F"): 1.39,
    ("C", "Cl"): 1.78,
    ("C", "Br"): 1.94,    # Phase 5: alkyl bromide
    ("C", "N"): 1.47,
    ("C", "O"): 1.43,
    ("C", "C"): 1.54,
    ("C", "H"): 1.09,
    ("N", "H"): 1.01,
    ("O", "H"): 0.97,
    ("Li", "Cl"): 2.02,   # Phase 5: gas-phase LiCl
    ("Li", "Br"): 2.17,   # Phase 5: gas-phase LiBr
}


def lookup_r_form(sym_a: str, sym_b: str, *, default: float = 1.6) -> float:
    """Symmetric lookup in DEFAULT_R_FORM. Returns `default` if pair unknown."""
    if (sym_a, sym_b) in DEFAULT_R_FORM:
        return DEFAULT_R_FORM[(sym_a, sym_b)]
    if (sym_b, sym_a) in DEFAULT_R_FORM:
        return DEFAULT_R_FORM[(sym_b, sym_a)]
    return default


class PullApart(FixConstraint):
    """One-sided harmonic *repulsion* pushing two atoms apart.

    Force is zero when distance >= rt; below rt, magnitude = k * (rt - r),
    pointing from atom a1 to atom a2 (and the equal-and-opposite on a1).
    Mirror of ase.constraints.Hookean which only attracts when r > rt.
    """

    def __init__(self, a1: int, a2: int, k: float, rt: float):
        self.a1 = int(a1)
        self.a2 = int(a2)
        self.k = float(k)
        self.rt = float(rt)

    def adjust_positions(self, atoms, newpositions):
        # Constraint is force-only; positions are integrated by the optimizer.
        return

    def adjust_forces(self, atoms, forces):
        p = atoms.positions
        d = p[self.a2] - p[self.a1]
        r = float(np.linalg.norm(d))
        if r >= self.rt or r < 1e-10:
            return
        u = d / r
        f = self.k * (self.rt - r) * u
        forces[self.a2] += f
        forces[self.a1] -= f

    def get_indices(self):
        return [self.a1, self.a2]

    def todict(self):
        return {
            "name": "PullApart",
            "kwargs": {"a1": self.a1, "a2": self.a2, "k": self.k, "rt": self.rt},
        }


def build_restraints(
    atoms: Atoms,
    formed: list[tuple[int, int]],
    broken: list[tuple[int, int]],
    *,
    r_form: float | None = None,
    r_broken: float = 4.0,
    k_form: float = 0.5,
    k_broken: float = 1.0,
) -> list:
    """Return a list of ASE constraints driving formed bonds together and
    broken bonds apart.

    `r_form=None` (default) looks each pair up in DEFAULT_R_FORM by element
    symbols. A scalar overrides the table by **broadcasting the same value to
    every formed bond**. Per-bond targets (different rt for each bond) are
    NOT supported in Phase 3 — formed=1 is the only multi-bond shape Phase 3
    actually exercises (E2). Per-bond list support is Phase 4+ when reactions
    with formed≥2 are added (Diels-Alder etc.).
    """
    syms = atoms.get_chemical_symbols()
    constraints: list = []
    for a, b in formed:
        rt = lookup_r_form(syms[a], syms[b]) if r_form is None else float(r_form)
        constraints.append(Hookean(a1=a, a2=b, rt=rt, k=k_form))
    for a, b in broken:
        constraints.append(PullApart(a1=a, a2=b, k=k_broken, rt=r_broken))
    return constraints
