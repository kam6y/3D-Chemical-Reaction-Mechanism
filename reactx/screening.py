"""Stage 1: artificial-force relax of all placement survivors (parallel).

Each placement trial is relaxed independently under Hookean / PullApart
restraints, producing a (frames, energies, peak_energy, reached_product)
record packed in ScreeningTrialResult. Pool-parallel via reactx.parallel
when workers >= 2.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from rdkit import Chem

from reactx.artificial_force import build_restraints
from reactx.bond_changes import BondChanges
from reactx.calculators import make_calculator
from reactx.config import ReactionConfig, resolve_r_form_targets
from reactx.parallel import (
    get_cached_calculator,
    init_lj_worker,
    init_uma_worker,
    run_with_pool,
)
from reactx.path_relax import relax_with_restraints
from reactx.placement import PlacementResult
from reactx.scoring import ScreeningTrialResult, reached_product

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ScreenJob:
    """Pickle-friendly per-trial input for the worker function."""
    trial_idx: int
    direction: np.ndarray
    positions: np.ndarray
    syms: list[str]
    charges: list[int]
    formed: list[tuple[int, int]]
    broken: list[tuple[int, int]]
    r_form_targets: list[float]
    r_broken: float
    k_form: float
    k_broken: float
    max_relax_steps: int
    relax_fmax: float
    traj_stride: int
    backend: str
    screening_model: str


def screen_all_trials(
    placement: PlacementResult,
    mol_h: Chem.Mol,
    bond_changes: BondChanges,
    cfg: ReactionConfig,
    *,
    backend: str,
    screening_model: str,
    workers: int,
    relax_fmax: float,
    traj_stride: int,
    seed: int,
) -> list[ScreeningTrialResult]:
    """Run artificial-force relax for every placement trial; return per-trial results.

    workers >= 2 -> spawn Pool; each worker loads one screening calculator.
    workers == 1 -> in-process; calculator created once via init_*_worker.
    """
    if not placement.trials:
        return []

    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    charges = [a.GetFormalCharge() for a in mol_h.GetAtoms()]
    formed_pairs = list(bond_changes.formed)
    broken_pairs = list(bond_changes.broken)
    r_form_targets = resolve_r_form_targets(cfg, syms, formed_pairs)

    jobs = [
        _ScreenJob(
            trial_idx=i,
            direction=t.direction,
            positions=t.positions,
            syms=syms,
            charges=charges,
            formed=formed_pairs,
            broken=broken_pairs,
            r_form_targets=r_form_targets,
            r_broken=cfg.restraints.r_broken,
            k_form=cfg.restraints.k_form,
            k_broken=cfg.restraints.k_broken,
            max_relax_steps=cfg.restraints.max_relax_steps,
            relax_fmax=relax_fmax,
            traj_stride=traj_stride,
            backend=backend,
            screening_model=screening_model,
        )
        for i, t in enumerate(placement.trials)
    ]

    initializer, initargs = _select_initializer(backend, screening_model)
    return run_with_pool(
        _run_one_screen, jobs,
        workers=workers, initializer=initializer, initargs=initargs,
    )


def _select_initializer(
    backend: str, screening_model: str,
) -> tuple[Callable[..., None], tuple]:
    if backend == "uma":
        return init_uma_worker, (screening_model,)
    if backend == "lj":
        return init_lj_worker, ()
    raise ValueError(f"unsupported backend: {backend!r}")


def _run_one_screen(job: _ScreenJob) -> ScreeningTrialResult:
    """Execute one trial's relax. Runs in worker process when pool>1."""
    try:
        calc = get_cached_calculator()
    except RuntimeError:
        # workers=1 path didn't run initializer (run_with_pool sequential branch
        # invokes initializer when present; defensive fallback for direct calls)
        calc = make_calculator(
            job.backend,
            **({"model_name": job.screening_model} if job.backend == "uma" else {}),
        )

    atoms_init = Atoms(symbols=job.syms, positions=job.positions)
    atoms_init.set_initial_charges(job.charges)
    atoms_init.info["charge"] = int(sum(job.charges))
    atoms_init.info["spin"] = 1
    restraints = build_restraints(
        atoms_init,
        formed=job.formed, broken=job.broken,
        r_form=job.r_form_targets[0] if job.r_form_targets else None,
        r_broken=job.r_broken,
        k_form=job.k_form, k_broken=job.k_broken,
    )

    try:
        frames, energies = relax_with_restraints(
            atoms_init, restraints, calc,
            max_steps=job.max_relax_steps,
            fmax=job.relax_fmax,
            traj_stride=job.traj_stride,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("trial %d relax failed: %s: %s",
                    job.trial_idx, type(exc).__name__, exc)
        return ScreeningTrialResult(
            trial_idx=job.trial_idx, direction=job.direction,
            frames=[], energies=[], reached_product=False,
            peak_energy=float("inf"), n_steps=0,
            error=f"{type(exc).__name__}: {exc}",
        )

    ok = reached_product(
        frames[-1], formed=job.formed, broken=job.broken,
        r_form_targets=job.r_form_targets,
        r_broken_target=job.r_broken,
    )
    peak = max(energies) if energies else float("inf")
    return ScreeningTrialResult(
        trial_idx=job.trial_idx, direction=job.direction,
        frames=frames, energies=energies,
        reached_product=ok, peak_energy=float(peak),
        n_steps=len(frames),
        error=None,
    )
