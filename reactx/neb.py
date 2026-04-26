"""ASE IDPP + CI-NEB driver. Writes multi-frame XYZ with optional end padding."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from ase import Atoms
from ase.io import write
from ase.optimize import FIRE

try:
    from ase.mep import NEB
except ImportError:  # ASE < 3.23
    from ase.neb import NEB

log = logging.getLogger(__name__)


def run_neb(
    reactant: Atoms,
    product: Atoms,
    *,
    calculator_factory: Callable[[], object],
    n_images: int = 11,
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
    shared_calc = calculator_factory()
    for img in images:
        img.calc = shared_calc

    # Two-phase NEB: warm up the band with plain NEB, then climb the TS.
    # IDPP can produce non-physical midpoints for position-swapping reactions,
    # and CI-NEB starting from a bad initial path tends to chase a wrong saddle.
    # Warming up first lets the band relax to a sensible MEP before climbing.
    # k=1.0 prevents image bunching near minima (default k=0.1 is too weak for
    # position-swapping reactions like SN2 where images collapse to R/P sides).
    warmup_steps = max(1, max_steps // 2)
    climb_steps = max(1, max_steps - warmup_steps)

    neb_warm = NEB(images, k=1.0, climb=False, allow_shared_calculator=True, method="improvedtangent")
    neb_warm.interpolate(method="idpp")

    # FIRE logs per-step energy/fmax to stdout ("-"). Users can pipe to a file.
    warm_converged = False
    log.info("NEB warmup phase: up to %d steps, fmax=%s", warmup_steps, fmax)
    try:
        warm_converged = bool(
            FIRE(neb_warm, logfile="-").run(fmax=fmax, steps=warmup_steps)
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("NEB warmup raised %s: %s", type(exc).__name__, exc)

    climb_converged = False
    final_neb = neb_warm
    if climb:
        neb_climb = NEB(images, k=1.0, climb=True, allow_shared_calculator=True, method="improvedtangent")
        log.info("NEB climb phase: up to %d steps, fmax=%s", climb_steps, fmax)
        try:
            climb_converged = bool(
                FIRE(neb_climb, logfile="-").run(fmax=fmax, steps=climb_steps)
            )
            final_neb = neb_climb
        except Exception as exc:  # noqa: BLE001
            log.warning("NEB climb raised %s: %s", type(exc).__name__, exc)
            # Prefer climb's partial result if it produced energies; otherwise
            # fall back to the warmup band so we report something meaningful
            # instead of a vector of NaNs.
            if neb_climb.energies is not None:
                final_neb = neb_climb
            else:
                log.info("NEB climb produced no energies; using warmup result")

    converged = climb_converged if climb else warm_converged
    # final_neb.{residuals,energies} are populated by the last get_forces() call
    # FIRE made internally — reading them here costs zero UMA evaluations.
    try:
        final_fmax = float(final_neb.get_residual())
    except Exception as exc:  # noqa: BLE001
        log.warning("NEB fmax evaluation raised %s: %s", type(exc).__name__, exc)
        final_fmax = float("nan")

    try:
        image_energies = [float(e) for e in final_neb.energies]
    except Exception as exc:  # noqa: BLE001
        log.warning("Image energy evaluation raised %s: %s", type(exc).__name__, exc)
        image_energies = [float("nan")] * n_images

    padded: list[Atoms] = []
    padded.extend([images[0].copy() for _ in range(pad_frames)])
    padded.extend(images)
    padded.extend([images[-1].copy() for _ in range(pad_frames)])

    write(str(output_xyz), padded, format="extxyz")

    return {
        "n_images": n_images,
        "converged": converged,
        "final_fmax": final_fmax,
        "image_energies": image_energies,
        "pad_frames": pad_frames,
    }
