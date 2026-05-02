"""Trial scoring for multi-angle path generation.

After each angle trial produces a relaxation trajectory, score_trials picks
the best one: prefer trials that actually reached the product topology
(formed bond close, broken bond far), then minimum peak energy among those.
If no trial reached the product, fall back to the trial with lowest peak
(= 'least bad' partial path).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms


@dataclass
class TrialResult:
    """Outcome of a single multi-angle relaxation trial."""

    trial_idx: int
    rotation_deg: float  # angular deviation from ideal direction
    frames: list[Atoms]
    energies: list[float]
    reached_product: bool
    peak_energy: float
    n_steps: int


def reached_product(
    final_atoms: Atoms,
    formed: list[tuple[int, int]],
    broken: list[tuple[int, int]],
    *,
    r_form_targets: list[float],
    r_broken_target: float,
    form_tol: float = 0.3,
    broken_tol: float = 0.5,
) -> bool:
    """True iff every formed bond is within r_form + form_tol AND every broken
    bond is at least r_broken - broken_tol apart.
    """
    if len(formed) != len(r_form_targets):
        raise ValueError(
            f"formed ({len(formed)}) must match r_form_targets ({len(r_form_targets)})"
        )
    p = final_atoms.positions
    for (a, b), rt in zip(formed, r_form_targets, strict=True):
        d = float(np.linalg.norm(p[a] - p[b]))
        if d > rt + form_tol:
            return False
    for a, b in broken:
        d = float(np.linalg.norm(p[a] - p[b]))
        if d < r_broken_target - broken_tol:
            return False
    return True


def score_trials(results: list[TrialResult]) -> TrialResult:
    """Return the best trial. Preference order:
    1. reached_product=True, lowest peak_energy
    2. all failed: lowest peak_energy (= 'least bad' fallback)
    """
    if not results:
        raise ValueError("score_trials called with empty list")
    reached = [r for r in results if r.reached_product]
    pool = reached if reached else results
    return min(pool, key=lambda r: r.peak_energy)
