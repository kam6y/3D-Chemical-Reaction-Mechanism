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
    ("C", "N"): 1.47,
    ("C", "O"): 1.43,
    ("C", "C"): 1.54,
    ("C", "H"): 1.09,
    ("N", "H"): 1.01,
    ("O", "H"): 0.97,
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
    r_form: float | list[float] | None = None,
    r_broken: float | list[float] = 4.0,
    k_form: float | list[float] = 0.5,
    k_broken: float | list[float] = 1.0,
) -> list:
    """Return a list of ASE constraints driving formed bonds together and
    broken bonds apart.

    `r_form=None` looks each pair up in DEFAULT_R_FORM by element symbols.
    Scalar values are broadcast across all bonds; list values must match
    `len(formed)` (for k_form / r_form) or `len(broken)` (for k_broken /
    r_broken).
    """
    syms = atoms.get_chemical_symbols()
    r_forms = _broadcast_r_form(r_form, formed, syms)
    k_forms = _broadcast(k_form, len(formed), key="k_form")
    r_brokens = _broadcast(r_broken, len(broken), key="r_broken")
    k_brokens = _broadcast(k_broken, len(broken), key="k_broken")

    constraints: list = []
    for (a, b), rt, k in zip(formed, r_forms, k_forms, strict=True):
        constraints.append(Hookean(a1=a, a2=b, rt=rt, k=k))
    for (a, b), rt, k in zip(broken, r_brokens, k_brokens, strict=True):
        constraints.append(PullApart(a1=a, a2=b, k=k, rt=rt))
    return constraints


def _broadcast(value: float | list[float], n: int, *, key: str) -> list[float]:
    """Scalar → list[n], list passthrough with length check."""
    if isinstance(value, list):
        if len(value) != n:
            raise ValueError(
                f"build_restraints: {key} list length {len(value)} != n_bonds {n}"
            )
        return [float(v) for v in value]
    return [float(value)] * n


def _broadcast_r_form(
    value: float | list[float] | None,
    formed: list[tuple[int, int]],
    syms: list[str],
) -> list[float]:
    if value is None:
        return [lookup_r_form(syms[a], syms[b]) for a, b in formed]
    if isinstance(value, list):
        if len(value) != len(formed):
            raise ValueError(
                f"build_restraints: r_form list length {len(value)} != n_formed {len(formed)}"
            )
        return [float(v) for v in value]
    return [float(value)] * len(formed)


class AFIRConstraint(FixConstraint):
    """Per-pair AFIR force with sticky per-pair latch (Phase 9, spec §4.1)."""

    def __init__(
        self,
        formed: list[tuple[int, int]],
        broken: list[tuple[int, int]],
        *,
        alpha_formed: list[float],
        alpha_broken: list[float],
        formed_thresholds: list[float],
        broken_thresholds: list[float],
    ):
        if len(alpha_formed) != len(formed):
            raise ValueError(
                f"alpha_formed length {len(alpha_formed)} != n_formed {len(formed)}"
            )
        if len(alpha_broken) != len(broken):
            raise ValueError(
                f"alpha_broken length {len(alpha_broken)} != n_broken {len(broken)}"
            )
        if len(formed_thresholds) != len(formed):
            raise ValueError(
                f"formed_thresholds length {len(formed_thresholds)} != n_formed {len(formed)}"
            )
        if len(broken_thresholds) != len(broken):
            raise ValueError(
                f"broken_thresholds length {len(broken_thresholds)} != n_broken {len(broken)}"
            )
        if any(a < 0 for a in alpha_formed) or any(a < 0 for a in alpha_broken):
            raise ValueError("alpha_* must be non-negative")
        # spec §10.3 critical-2: defense in depth.
        if len(formed) > 0 and any(a <= 0 for a in alpha_formed):
            raise ValueError(
                "alpha_formed must be positive for non-empty formed pair set "
                "(spec §10.3: α=0 would never latch)"
            )
        if len(broken) > 0 and any(a <= 0 for a in alpha_broken):
            raise ValueError(
                "alpha_broken must be positive for non-empty broken pair set"
            )
        if any(t <= 0 for t in formed_thresholds) or any(t <= 0 for t in broken_thresholds):
            raise ValueError("thresholds must be positive")
        self.formed = [(int(a), int(b)) for a, b in formed]
        self.broken = [(int(a), int(b)) for a, b in broken]
        self.alpha_formed = [float(x) for x in alpha_formed]
        self.alpha_broken = [float(x) for x in alpha_broken]
        self.formed_thresholds = [float(t) for t in formed_thresholds]
        self.broken_thresholds = [float(t) for t in broken_thresholds]
        self.formed_latched = [False] * len(formed)
        self.broken_latched = [False] * len(broken)

    def adjust_positions(self, atoms, newpositions):
        return

    def adjust_forces(self, atoms, forces):
        pos = atoms.positions
        for k, (i, j) in enumerate(self.formed):
            if self.formed_latched[k]:
                continue
            r, d_hat = self._geom(pos, i, j)
            if r <= self.formed_thresholds[k]:
                self.formed_latched[k] = True
                continue
            f_on_j = -self.alpha_formed[k] * d_hat
            forces[j] += f_on_j
            forces[i] -= f_on_j
        for k, (i, j) in enumerate(self.broken):
            if self.broken_latched[k]:
                continue
            r, d_hat = self._geom(pos, i, j)
            if r >= self.broken_thresholds[k]:
                self.broken_latched[k] = True
                continue
            f_on_j = +self.alpha_broken[k] * d_hat
            forces[j] += f_on_j
            forces[i] -= f_on_j

    @staticmethod
    def _geom(pos, i, j):
        v = pos[j] - pos[i]
        r = float(np.linalg.norm(v))
        if r < 1e-6:
            raise ValueError(
                f"AFIRConstraint: atoms {i} and {j} coincide (r={r:.2e}); "
                f"check fragment placement geometry"
            )
        return r, v / r

    def all_latched(self) -> bool:
        return all(self.formed_latched) and all(self.broken_latched)

    def get_indices(self):
        return sorted({a for pair in (self.formed + self.broken) for a in pair})


def build_afir_constraint(
    atoms,
    formed: list[tuple[int, int]],
    broken: list[tuple[int, int]],
    *,
    alpha_formed,
    alpha_broken,
    formed_thresholds: list[float],
    broken_thresholds: list[float],
) -> list[AFIRConstraint]:
    if not formed and not broken:
        return []
    af = _broadcast_alpha(alpha_formed, len(formed), key="alpha_formed")
    ab = _broadcast_alpha(alpha_broken, len(broken), key="alpha_broken")
    return [AFIRConstraint(
        formed, broken,
        alpha_formed=af, alpha_broken=ab,
        formed_thresholds=formed_thresholds,
        broken_thresholds=broken_thresholds,
    )]


def _broadcast_alpha(value, n: int, *, key: str) -> list[float]:
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f"{key} list length {len(value)} != n_pairs {n}")
        return [float(v) for v in value]
    return [float(value)] * n
