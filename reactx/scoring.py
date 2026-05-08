"""Trial scoring + final-best selection for placement trials.

Phase 7: rotation_deg field replaced by direction (3-vec) since trials are
indexed by Fibonacci-sphere unit vectors, not deviations from a single
ideal direction. select_best_trial moved here from prescreen.select_top_k_indices
(top-K -> 1) to centralize all "pick best trial" logic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms


@dataclass
class TrialResult:
    """Outcome of a single placement / relaxation trial.

    `direction` is the unit vector (shape (3,), dtype float) used for
    sphere-based fragment placement. For unimolecular passthrough trials
    it is a placeholder +z with no physical meaning. dtype is float since
    cli.py serializes it to meta.json as a list of floats.
    """

    trial_idx: int
    direction: np.ndarray
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
    r_broken_target: float | list[float],
    form_tol: float = 0.3,
    broken_tol: float = 0.5,
) -> bool:
    """True iff every formed bond is within r_form + form_tol AND every broken
    bond is at least r_broken - broken_tol apart.

    `r_broken_target` accepts scalar (broadcast to all broken bonds) or
    `list[float]` of length len(broken).
    """
    if len(formed) != len(r_form_targets):
        raise ValueError(
            f"formed ({len(formed)}) must match r_form_targets ({len(r_form_targets)})"
        )
    if isinstance(r_broken_target, list):
        r_broken_targets = r_broken_target
    else:
        r_broken_targets = [float(r_broken_target)] * len(broken)
    if len(broken) != len(r_broken_targets):
        raise ValueError(
            f"broken ({len(broken)}) must match r_broken_target list ({len(r_broken_targets)})"
        )
    p = final_atoms.positions
    for (a, b), rt in zip(formed, r_form_targets, strict=True):
        d = float(np.linalg.norm(p[a] - p[b]))
        if d > rt + form_tol:
            return False
    for (a, b), rt in zip(broken, r_broken_targets, strict=True):
        d = float(np.linalg.norm(p[a] - p[b]))
        if d < rt - broken_tol:
            return False
    return True


def score_trials(results: list[TrialResult]) -> TrialResult:
    """Return the best TrialResult (preference: reached then peak_energy up).

    1. reached_product=True 群の最低 peak_energy
    2. 全部 False なら全体の最低 peak_energy (= 'least bad' fallback)
    """
    if not results:
        raise ValueError("score_trials called with empty list")
    reached = [r for r in results if r.reached_product]
    pool = reached if reached else results
    return min(pool, key=lambda r: r.peak_energy)


def select_best_trial(trials: list[TrialResult]) -> int:
    """Return the trial_idx of the best TrialResult (same preference as score_trials)."""
    if not trials:
        raise ValueError("select_best_trial called with empty list")
    return score_trials(trials).trial_idx
