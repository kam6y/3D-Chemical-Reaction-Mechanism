"""Per-pair AFIR force with sticky per-pair latch (Phase 9).

Replaces Phase Re1's Hookean+PullApart hybrid. Each user-specified pair
gets an independent constant-magnitude force (`F = α · d̂`) gated by a
sticky latch on a per-pair distance threshold.

Spec: docs/superpowers/specs/2026-05-08-afir-force-design.md §4.1
"""
from __future__ import annotations

import numpy as np
from ase.constraints import FixConstraint


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
