"""ASE IDPP + CI-NEB driver. Writes multi-frame XYZ with optional end padding."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import write
from ase.optimize import FIRE

from reactx.endpoints import relax_endpoint
from reactx.parallel import (
    get_cached_calculator,
    init_lj_worker,
    init_uma_worker,
    run_with_pool,
)

try:
    from ase.mep import NEB
except ImportError:  # ASE < 3.23
    from ase.neb import NEB

log = logging.getLogger(__name__)


def _make_neb(images: list[Atoms], *, climb: bool) -> NEB:
    return NEB(
        images, k=1.0, climb=climb,
        allow_shared_calculator=True, method="improvedtangent",
    )


def _safe_call(fn, label: str, fallback):
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        log.warning("NEB %s evaluation raised %s: %s", label, type(exc).__name__, exc)
        return fallback


def _run_phase(label: str, neb, *, fmax: float, steps: int) -> bool:
    """Run FIRE on a NEB band; return True if converged. Logs and swallows
    runtime errors so a failed phase does not abort the pipeline."""
    log.info("NEB %s phase: up to %d steps, fmax=%s", label, steps, fmax)
    try:
        return bool(FIRE(neb, logfile="-").run(fmax=fmax, steps=steps))
    except Exception as exc:  # noqa: BLE001
        log.warning("NEB %s raised %s: %s", label, type(exc).__name__, exc)
        return False


def run_neb(
    reactant: Atoms,
    product: Atoms,
    *,
    calculator: Calculator,
    n_images: int = 15,
    output_xyz: str | Path,
    fmax: float = 0.05,
    max_steps: int = 200,
    climb: bool = True,
    pad_frames: int = 0,
) -> dict:
    """Run IDPP interpolation + (CI-)NEB, write trajectory XYZ, return metadata.

    Uses FIRE optimizer (more robust than BFGS on rough reactive PES).
    """
    if n_images < 3:
        raise ValueError(
            f"n_images must be >= 3 (reactant + >=1 middle + product), got {n_images}"
        )

    images = [reactant.copy()]
    for _ in range(n_images - 2):
        images.append(reactant.copy())
    images.append(product.copy())

    # Single shared calculator: large models (uma-m-1p1 = 11 GB) would OOM if
    # instantiated once per image. allow_shared_calculator=True lets ASE
    # evaluate images sequentially with one model instance.
    for img in images:
        img.calc = calculator

    # Two-phase NEB: warm up with plain NEB, then climb the TS. IDPP can
    # produce non-physical midpoints for position-swapping reactions, so
    # starting CI-NEB from a bad initial path tends to chase a wrong saddle.
    # k=1.0 prevents image bunching near minima (default k=0.1 is too weak for
    # position-swapping reactions like SN2 where images collapse to R/P sides).
    warmup_steps = max(1, max_steps // 2)
    climb_steps = max(1, max_steps - warmup_steps)

    # `images` is shared between the warmup and climb NEB objects. ASE's
    # NEB optimizer mutates image positions in place, so the climb band
    # automatically inherits the warmup-relaxed path — no second
    # interpolate() call is needed (and would in fact overwrite warmup work).
    neb_warm = _make_neb(images, climb=False)
    neb_warm.interpolate(method="idpp")
    warm_converged = _run_phase("warmup", neb_warm, fmax=fmax, steps=warmup_steps)

    climb_converged = False
    final_neb = neb_warm
    if climb:
        neb_climb = _make_neb(images, climb=True)
        climb_converged = _run_phase("climb", neb_climb, fmax=fmax, steps=climb_steps)
        # Prefer the climb band even on partial convergence — its forces/energies
        # are at least as recent as the warmup's.
        final_neb = neb_climb

    converged = climb_converged if climb else warm_converged
    final_fmax = _safe_call(final_neb.get_residual, "NEB fmax", float("nan"))
    image_energies = _safe_call(
        lambda: [float(e) for e in final_neb.energies],
        "image energies",
        [float("nan")] * n_images,
    )

    # Reference-sharing the endpoint Atoms in the padded list is safe: extxyz
    # writer only reads positions/symbols, never mutates.
    padded = [images[0]] * pad_frames + list(images) + [images[-1]] * pad_frames
    write(str(output_xyz), padded, format="extxyz")

    return {
        "n_images": n_images,
        "converged": converged,
        "final_fmax": final_fmax,
        "image_energies": image_energies,
        "pad_frames": pad_frames,
    }


@dataclass(frozen=True)
class _NebJob:
    """Pickle-friendly per-trial input for NEB worker."""
    trial_idx: int
    reactant_atoms: Atoms
    product_atoms: Atoms
    n_images: int
    fmax: float
    max_steps: int
    pad_frames: int
    output_xyz: Path
    backend: str
    neb_model: str


def run_neb_for_trial(
    reactant_atoms: Atoms,
    product_atoms: Atoms,
    *,
    calc: Calculator,
    n_images: int,
    fmax: float,
    max_steps: int,
    pad_frames: int,
    output_xyz: Path,
) -> dict:
    """Run un-restrained endpoint relax + 2-phase NEB for one trial.

    Returns a dict {converged, n_images, image_energies, peak_energy,
    final_fmax, xyz_path}. xyz_path is the basename only.
    """
    R = relax_endpoint(reactant_atoms, calc, fmax=0.05, max_steps=50)
    P = relax_endpoint(product_atoms, calc, fmax=0.05, max_steps=50)
    info = run_neb(
        reactant=R, product=P, calculator=calc,
        n_images=n_images, output_xyz=output_xyz,
        fmax=fmax, max_steps=max_steps, pad_frames=pad_frames,
    )
    image_energies = info.get("image_energies") or []
    finite = [e for e in image_energies if e == e]
    peak = max(finite) if finite else float("nan")
    return {
        "converged": info.get("converged", False),
        "n_images": info.get("n_images", n_images),
        "image_energies": image_energies,
        "peak_energy": float(peak) if peak == peak else float("nan"),
        "final_fmax": info.get("final_fmax", float("nan")),
        "xyz_path": output_xyz.name,
    }


def _run_one_neb(job: _NebJob) -> dict:
    try:
        calc = get_cached_calculator()
    except RuntimeError:
        from reactx.calculators import make_calculator
        calc = make_calculator(
            job.backend,
            **({"model_name": job.neb_model} if job.backend == "uma" else {}),
        )
    result = run_neb_for_trial(
        job.reactant_atoms, job.product_atoms, calc=calc,
        n_images=job.n_images, fmax=job.fmax, max_steps=job.max_steps,
        pad_frames=job.pad_frames, output_xyz=job.output_xyz,
    )
    result["trial_idx"] = job.trial_idx
    return result


def run_neb_top_k(
    top_k_results: list,           # list[ScreeningTrialResult]
    *,
    backend: str,
    neb_model: str,
    workers: int,
    n_images: int,
    fmax: float,
    max_steps: int,
    pad_frames: int,
    output_dir: Path,
) -> list[dict]:
    """Run NEB on each top-K screening result, parallel via Pool when workers>=2.

    Returns list of NEB result dicts in the same order as top_k_results.
    Skips entries with no frames (logs warning); if all skipped, returns [].
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[_NebJob] = []
    for r in top_k_results:
        if not r.frames:
            log.warning("skipping NEB for trial %d: no frames", r.trial_idx)
            continue
        out_xyz = output_dir / f"trajectory_neb_trial_{r.trial_idx}.xyz"
        jobs.append(_NebJob(
            trial_idx=r.trial_idx,
            reactant_atoms=r.frames[0],
            product_atoms=r.frames[-1],
            n_images=n_images, fmax=fmax, max_steps=max_steps,
            pad_frames=pad_frames, output_xyz=out_xyz,
            backend=backend, neb_model=neb_model,
        ))
    if not jobs:
        return []

    if backend == "uma":
        initializer, initargs = init_uma_worker, (neb_model,)
    elif backend == "lj":
        initializer, initargs = init_lj_worker, ()
    else:
        raise ValueError(f"unsupported backend: {backend!r}")

    return run_with_pool(
        _run_one_neb, jobs,
        workers=workers, initializer=initializer, initargs=initargs,
    )
