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


# ===== Phase 9 additions =====

from reactx.config import ConfigError
from reactx.covalent_radii import cordero_radii_for_atoms

COVALENT_FORMED_TOLERANCE = 1.15
"""Phase 9 default formed threshold: r ≤ 1.15 × Rsum_Cordero."""


@dataclass
class TrialResultV9:
    """Phase 9 outcome of one placement / relaxation trial.

    14 fields. `reached_product` (final-frame distance) is the sole success
    criterion. Latch counts and initial_latched are diagnostics only.
    `formed_thresholds` / `broken_thresholds` capture the resolved per-pair
    threshold values used at runtime so meta.json can record them.
    """
    trial_idx: int
    direction: np.ndarray
    frames: list[Atoms]
    energies: list[float]
    reached_product: bool
    peak_energy: float
    n_steps: int
    formed_thresholds: list[float]
    broken_thresholds: list[float]
    product_distance_residual: float
    formed_latch_count: int
    broken_latch_count: int
    initial_latched_formed: int
    initial_latched_broken: int


def resolve_formed_thresholds(atoms, formed, override) -> list[float]:
    if not formed:
        return []
    if override is None:
        cov = cordero_radii_for_atoms(atoms)
        return [(cov[i] + cov[j]) * COVALENT_FORMED_TOLERANCE for (i, j) in formed]
    return _v9_broadcast_threshold(override, len(formed), key="r_formed_threshold")


def resolve_broken_thresholds(atoms, broken, override) -> list[float]:
    if not broken:
        return []
    if override is None:
        raise ConfigError(
            "r_broken_threshold required when broken bonds are specified"
        )
    return _v9_broadcast_threshold(override, len(broken), key="r_broken_threshold")


def _v9_broadcast_threshold(value, n: int, *, key: str) -> list[float]:
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ConfigError(f"{key} list length {len(value)} != n_pairs {n}")
        return [float(v) for v in value]
    return [float(value)] * n


def reached_product_v9(
    final_atoms,
    formed: list[tuple[int, int]],
    broken: list[tuple[int, int]],
    formed_thresholds: list[float],
    broken_thresholds: list[float],
) -> bool:
    pos = final_atoms.positions
    for (i, j), thr in zip(formed, formed_thresholds, strict=True):
        d = float(np.linalg.norm(pos[j] - pos[i]))
        if d > thr:
            return False
    for (i, j), thr in zip(broken, broken_thresholds, strict=True):
        d = float(np.linalg.norm(pos[j] - pos[i]))
        if d < thr:
            return False
    return True


def product_distance_residual(
    atoms,
    formed: list[tuple[int, int]],
    broken: list[tuple[int, int]],
    formed_thresholds: list[float],
    broken_thresholds: list[float],
) -> float:
    pos = atoms.positions
    res = 0.0
    for (i, j), thr in zip(formed, formed_thresholds, strict=True):
        d = float(np.linalg.norm(pos[j] - pos[i]))
        res += max(d - thr, 0.0) ** 2
    for (i, j), thr in zip(broken, broken_thresholds, strict=True):
        d = float(np.linalg.norm(pos[j] - pos[i]))
        res += max(thr - d, 0.0) ** 2
    return res


def count_initial_latched(
    atoms,
    formed: list[tuple[int, int]],
    broken: list[tuple[int, int]],
    formed_thresholds: list[float],
    broken_thresholds: list[float],
) -> dict[str, int]:
    pos = atoms.positions
    n_f = sum(
        1 for (i, j), thr in zip(formed, formed_thresholds, strict=True)
        if float(np.linalg.norm(pos[j] - pos[i])) <= thr
    )
    n_b = sum(
        1 for (i, j), thr in zip(broken, broken_thresholds, strict=True)
        if float(np.linalg.norm(pos[j] - pos[i])) >= thr
    )
    return {"formed": n_f, "broken": n_b}


def score_trials_v9(results: list[TrialResultV9]) -> TrialResultV9:
    if not results:
        raise ValueError("score_trials_v9 called with empty list")
    reached = [r for r in results if r.reached_product]
    if reached:
        return min(reached, key=lambda r: r.peak_energy)
    return min(results, key=lambda r: (r.product_distance_residual, r.peak_energy))
