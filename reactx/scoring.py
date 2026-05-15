"""Trial scoring + final-best selection (Phase 9).

Spec: docs/superpowers/specs/2026-05-08-afir-force-design.md §4.5
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms

from reactx.config import ConfigError
from reactx.covalent_radii import cordero_radii_for_atoms

COVALENT_FORMED_TOLERANCE = 1.15
"""Phase 9 default formed threshold: r ≤ 1.15 × Rsum_Cordero."""


@dataclass
class TrialResult:
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
    return _broadcast_threshold(override, len(formed), key="r_formed_threshold")


def resolve_broken_thresholds(atoms, broken, override) -> list[float]:
    if not broken:
        return []
    if override is None:
        raise ConfigError(
            "r_broken_threshold required when broken bonds are specified"
        )
    return _broadcast_threshold(override, len(broken), key="r_broken_threshold")


def _broadcast_threshold(value, n: int, *, key: str) -> list[float]:
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ConfigError(f"{key} list length {len(value)} != n_pairs {n}")
        return [float(v) for v in value]
    return [float(value)] * n


def reached_product(
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


def score_trials(results: list[TrialResult]) -> TrialResult:
    if not results:
        raise ValueError("score_trials called with empty list")
    reached = [r for r in results if r.reached_product]
    if reached:
        return min(reached, key=lambda r: r.peak_energy)
    return min(results, key=lambda r: (r.product_distance_residual, r.peak_energy))


def select_best_trial(trials: list[TrialResult]) -> int:
    """Return the trial_idx of the best TrialResult (same preference as score_trials)."""
    if not trials:
        raise ValueError("select_best_trial called with empty list")
    return score_trials(trials).trial_idx
