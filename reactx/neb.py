"""ASE IDPP + CI-NEB driver. Writes multi-frame XYZ with optional end padding."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from ase import Atoms
from ase.io import write
from ase.optimize import FIRE

try:
    from ase.mep import NEB
except ImportError:  # ASE < 3.23
    from ase.neb import NEB


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

    converged = False
    try:
        FIRE(neb_warm, logfile=None).run(fmax=fmax, steps=warmup_steps)
    except Exception:
        pass

    if climb:
        neb_climb = NEB(images, k=1.0, climb=True, allow_shared_calculator=True, method="improvedtangent")
        try:
            FIRE(neb_climb, logfile=None).run(fmax=fmax, steps=climb_steps)
        except Exception:
            pass

    try:
        converged = all(
            max(abs(img.get_forces().flatten())) < fmax
            for img in images[1:-1]
        )
    except Exception:
        converged = False

    try:
        final_fmax = max(
            float(max(abs(img.get_forces().flatten())))
            for img in images[1:-1]
        )
        image_energies = [float(img.get_potential_energy()) for img in images]
    except Exception:
        final_fmax = float("nan")
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
