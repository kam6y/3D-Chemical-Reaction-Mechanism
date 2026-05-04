"""Trial scoring + final-best selection for screening trials.

Phase 8: TrialResult renamed to ScreeningTrialResult to reflect its role
as Stage 1 (screening) output. Adds top_k_trials for the screening->NEB
hand-off: pick top-K by reached_product first, then by peak_energy
ascending; if fewer than K reached, fill from unreached pool by peak.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms


@dataclass
class ScreeningTrialResult:
    """Outcome of one Stage 1 (artificial-force) relax trial.

    `direction` is the unit vector used for sphere-based fragment placement
    (placeholder +z for unimolecular passthrough).
    `error` is the relax exception string when frames=[]/energies=[];
    None on success.
    """

    trial_idx: int
    direction: np.ndarray
    frames: list[Atoms]
    energies: list[float]
    reached_product: bool
    peak_energy: float
    n_steps: int
    error: str | None = None


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


def top_k_trials(
    results: list[ScreeningTrialResult],
    k: int,
) -> list[ScreeningTrialResult]:
    """Return up to k results ranked by (reached_product desc, peak_energy asc).

    1. reached_product=True 群を peak_energy 昇順で並べる
    2. reached_product=False 群を peak_energy 昇順で並べる
    3. 連結して先頭から k 件
    """
    if not results:
        raise ValueError("top_k_trials called with empty results list")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    reached = sorted(
        (r for r in results if r.reached_product), key=lambda r: r.peak_energy,
    )
    unreached = sorted(
        (r for r in results if not r.reached_product), key=lambda r: r.peak_energy,
    )
    return (reached + unreached)[:k]


def score_trials(results: list[ScreeningTrialResult]) -> ScreeningTrialResult:
    """Return the best ScreeningTrialResult (top_k_trials(..., 1)[0])."""
    return top_k_trials(results, 1)[0]


def select_best_trial(trials: list[ScreeningTrialResult]) -> int:
    """Return the trial_idx of the best result."""
    return score_trials(trials).trial_idx
