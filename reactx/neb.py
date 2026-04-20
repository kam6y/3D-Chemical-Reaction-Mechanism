"""ASE IDPP + CI-NEB driver. Writes multi-frame XYZ with optional end padding."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from ase import Atoms
from ase.io import write
from ase.optimize import BFGS

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
    """Run IDPP interpolation + (CI-)NEB, write trajectory XYZ, return metadata."""
    if n_images < 3:
        raise ValueError(
            f"n_images must be >= 3 (reactant + >=1 middle + product), got {n_images}"
        )

    images = [reactant.copy()]
    for _ in range(n_images - 2):
        images.append(reactant.copy())
    images.append(product.copy())

    for img in images:
        img.calc = calculator_factory()

    neb = NEB(images, climb=climb, allow_shared_calculator=False)
    neb.interpolate(method="idpp")

    opt = BFGS(neb, logfile=None)
    converged = False
    try:
        opt.run(fmax=fmax, steps=max_steps)
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
