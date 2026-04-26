"""ASE IDPP + CI-NEB driver. Writes multi-frame XYZ with optional end padding."""

from __future__ import annotations

import logging
from pathlib import Path

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import write
from ase.optimize import FIRE

try:
    from ase.mep import NEB
except ImportError:  # ASE < 3.23
    from ase.neb import NEB

log = logging.getLogger(__name__)


def _make_neb(images: list[Atoms], *, climb: bool) -> NEB:
    return NEB(
        images,
        k=1.0,
        climb=climb,
        allow_shared_calculator=True,
        method="improvedtangent",
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
        raise ValueError(f"n_images must be >= 3 (reactant + >=1 middle + product), got {n_images}")

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
