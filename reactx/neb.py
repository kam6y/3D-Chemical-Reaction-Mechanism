"""ASE IDPP + CI-NEB driver. Writes multi-frame XYZ with optional end padding."""
from __future__ import annotations

import logging
from pathlib import Path

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import write
from ase.optimize import FIRE

from reactx.bond_changes import BondChanges
from reactx.neb_guide import (
    BondDistanceGuideCalculator,
    build_guided_bonds,
    targets_for_image,
)

try:
    from ase.mep import NEB
except ImportError:  # ASE < 3.23
    from ase.neb import NEB

log = logging.getLogger(__name__)


def _make_neb(
    images: list[Atoms],
    *,
    climb: bool,
    k: float,
    method: str,
    remove_rotation_and_translation: bool,
) -> NEB:
    return NEB(
        images, k=k, climb=climb,
        allow_shared_calculator=True,
        method=method,
        remove_rotation_and_translation=remove_rotation_and_translation,
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
    k: float = 1.0,
    method: str = "eb",
    remove_rotation_and_translation: bool = True,
    climb: bool = True,
    pad_frames: int = 0,
    guide_bond_changes: bool = True,
    guide_k: float = 5.0,
    bond_changes: BondChanges | None = None,
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

    guided_bonds = (
        build_guided_bonds(reactant, product, bond_changes)
        if guide_bond_changes and bond_changes is not None
        else []
    )
    biased_optimization = bool(guided_bonds)

    # Single shared calculator: large models (uma-m-1p1 = 11 GB) would OOM if
    # instantiated once per image. allow_shared_calculator=True lets ASE
    # evaluate images sequentially with one model instance. Internal images may
    # receive lightweight wrappers that all call this same base calculator.
    for img in images:
        img.calc = calculator
    if biased_optimization:
        for image_index, img in enumerate(images[1:-1], start=1):
            img.calc = BondDistanceGuideCalculator(
                calculator,
                targets=targets_for_image(
                    guided_bonds,
                    image_index=image_index,
                    n_images=n_images,
                ),
                guide_k=guide_k,
            )

    # Two-phase NEB: warm up with plain NEB, then climb the TS. IDPP can
    # produce non-physical midpoints for position-swapping reactions, so
    # starting CI-NEB from a bad initial path tends to chase a wrong saddle.
    # The configured spring constant prevents image bunching near minima.
    # The default k=1.0 is stronger than ASE's default k=0.1, which is too weak
    # for position-swapping reactions like SN2 where images collapse to R/P sides.
    warmup_steps = max(1, max_steps // 2)
    climb_steps = max(1, max_steps - warmup_steps)

    # `images` is shared between the warmup and climb NEB objects. ASE's
    # NEB optimizer mutates image positions in place, so the climb band
    # automatically inherits the warmup-relaxed path — no second
    # interpolate() call is needed (and would in fact overwrite warmup work).
    neb_warm = _make_neb(
        images,
        climb=False,
        k=k,
        method=method,
        remove_rotation_and_translation=remove_rotation_and_translation,
    )
    neb_warm.interpolate(method="idpp")
    warm_converged = _run_phase("warmup", neb_warm, fmax=fmax, steps=warmup_steps)

    climb_converged = False
    final_neb = neb_warm
    if climb:
        neb_climb = _make_neb(
            images,
            climb=True,
            k=k,
            method=method,
            remove_rotation_and_translation=remove_rotation_and_translation,
        )
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
    unbiased_image_energies = _safe_call(
        lambda: _evaluate_unbiased_image_energies(images, calculator),
        "unbiased image energies",
        [float("nan")] * n_images,
    )

    write(str(output_xyz), _trajectory_frames(images, pad_frames), format="extxyz")

    return {
        "n_images": n_images,
        "converged": converged,
        "final_fmax": final_fmax,
        "image_energies": image_energies,
        "unbiased_image_energies": unbiased_image_energies,
        "biased_optimization": biased_optimization,
        "guide_bond_changes": guide_bond_changes,
        "guide_k": guide_k,
        "guided_bonds": [bond.as_metadata() for bond in guided_bonds],
        "k": k,
        "method": method,
        "remove_rotation_and_translation": remove_rotation_and_translation,
        "pad_frames": pad_frames,
        "image_atoms": [img.copy() for img in images],
    }


def _evaluate_unbiased_image_energies(
    images: list[Atoms],
    calculator: Calculator,
) -> list[float]:
    energies: list[float] = []
    for img in images:
        unbiased = img.copy()
        unbiased.calc = calculator
        energies.append(float(unbiased.get_potential_energy()))
    return energies


def _trajectory_frames(images: list[Atoms], pad_frames: int) -> list[Atoms]:
    padded = [images[0]] * pad_frames + list(images) + [images[-1]] * pad_frames
    frames = [img.copy() for img in padded]
    for frame in frames:
        frame.calc = None
    return frames
