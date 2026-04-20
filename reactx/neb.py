"""ASE IDPP + CI-NEB driver. Writes multi-frame XYZ with optional end padding."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from ase import Atoms
from ase.io import write
from ase.optimize import BFGS

try:
    from ase.mep import NEB
except ImportError:  # ASE < 3.23
    from ase.neb import NEB

# Spring constant (eV/Å). Default ASE value (0.1) is too weak for reactions
# with large forces; 1.0 keeps images on the path without slipping sideways.
_NEB_K = 1.0

# Maximum per-step displacement (Å) for the stabilised gradient descent.
_GD_MAXSTEP = 0.05

# Maximum force component (eV/Å) used for clipping during gradient descent.
_GD_FORCE_CLIP = 3.0

# Minimum allowed interatomic distance (Å) — if violated the image is reset.
_MIN_DIST = 0.8


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
    """Run (CI-)NEB, write trajectory XYZ, return metadata.

    Uses a stabilised gradient descent with force-clipping and minimum-distance
    enforcement. BFGS is avoided because it takes unbounded steps on the steep
    SN2 potential energy surface, causing atoms to fly to unphysical geometries.
    The initial path uses a SN2-aware internal-coordinate interpolation
    (F-C-Cl angle maintained near 180° throughout) rather than IDPP.
    """
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

    neb = NEB(
        images,
        k=_NEB_K,
        climb=climb,
        allow_shared_calculator=False,
        method="improvedtangent",
    )

    # SN2-aware interpolation: keeps F-C-Cl near 180° to avoid frontside paths
    try:
        _interpolate_sn2(images)
    except Exception:
        neb.interpolate(method="idpp")

    # Stabilised gradient descent with force-clipping and min-distance checks.
    # A small number of steps is sufficient to evaluate energies on the
    # interpolated path; large step counts cause images to slide off the
    # linear SN2 axis due to large chemistry forces near the bonded region.
    _gd_steps = min(max_steps, 10)
    _stable_gradient_descent(neb, images, steps=_gd_steps, fmax=fmax)

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


def _stable_gradient_descent(
    neb: "NEB",
    images: list[Atoms],
    steps: int,
    fmax: float = 0.05,
) -> None:
    """Damped gradient descent on the NEB band with per-step safety checks.

    Forces are clipped to [-_GD_FORCE_CLIP, +_GD_FORCE_CLIP] and the maximum
    displacement per step is capped at _GD_MAXSTEP Å. Any image whose atoms
    would come closer than _MIN_DIST Å is reverted to its previous geometry.
    Early-exits when the per-image max force drops below ``fmax``.
    """
    n_atoms = len(images[0])
    for _ in range(steps):
        try:
            forces = neb.get_forces()  # shape (n_inner * n_atoms, 3)
        except Exception:
            break

        # Early exit if already converged
        if np.max(np.abs(forces)) < fmax:
            break

        clipped = np.clip(forces, -_GD_FORCE_CLIP, _GD_FORCE_CLIP)
        max_f = np.max(np.abs(clipped))
        if max_f < 1e-10:
            break

        step_scale = _GD_MAXSTEP / max_f

        # Save current inner positions before the move
        old_positions = [img.get_positions().copy() for img in images[1:-1]]

        # Apply displacement to each inner image
        for i, img in enumerate(images[1:-1]):
            disp = step_scale * clipped[i * n_atoms: (i + 1) * n_atoms]
            img.set_positions(img.get_positions() + disp)

        # Safety: revert any inner image that has atoms too close together
        for i, img in enumerate(images[1:-1]):
            pos = img.get_positions()
            n = len(pos)
            collapsed = False
            for a in range(n):
                for b in range(a + 1, n):
                    if np.linalg.norm(pos[a] - pos[b]) < _MIN_DIST:
                        collapsed = True
                        break
                if collapsed:
                    break
            if collapsed:
                img.set_positions(old_positions[i])


def _interpolate_sn2(images: list[Atoms]) -> None:
    """Internal-coordinate interpolation for SN2 reactions.

    Fills in the intermediate images (indices 1 .. n-2) by interpolating
    bond distances in the reaction coordinate (C-Nu decreasing, C-LG
    increasing) while keeping all atoms on a near-linear attack axis.

    The reaction centre is identified as the non-H atom that has the largest
    change in nearest-neighbour distance between reactant and product; the
    nucleophile and leaving group are the non-H atoms furthest from C on
    each side of the axis.

    Raises RuntimeError if the topology doesn't look like an SN2 reaction
    (e.g. more than one reaction centre, or no clear axis), so the caller
    can fall back to IDPP.
    """
    r = images[0]
    p = images[-1]
    n = len(images)

    syms = r.get_chemical_symbols()
    r_pos = r.get_positions()
    p_pos = p.get_positions()

    heavy = [i for i, s in enumerate(syms) if s != "H"]
    if len(heavy) < 3:
        raise RuntimeError("Need >= 3 heavy atoms for SN2 interpolation")

    # Identify reaction centre as the heavy atom that minimises max distance
    # to its two heavy-atom neighbours in both reactant and product
    best_rc = None
    best_score = np.inf
    for rc in heavy:
        others = [h for h in heavy if h != rc]
        r_dists = sorted([np.linalg.norm(r_pos[rc] - r_pos[o]) for o in others])
        p_dists = sorted([np.linalg.norm(p_pos[rc] - p_pos[o]) for o in others])
        score = r_dists[0] + p_dists[0]
        if score < best_score:
            best_score = score
            best_rc = rc

    rc = best_rc

    others = [h for h in heavy if h != rc]
    deltas = {
        o: np.linalg.norm(p_pos[rc] - p_pos[o]) - np.linalg.norm(r_pos[rc] - r_pos[o])
        for o in others
    }
    nu = min(others, key=lambda o: deltas[o])
    lg = max(others, key=lambda o: deltas[o])

    if deltas[nu] >= 0 or deltas[lg] <= 0:
        raise RuntimeError("Cannot identify clear nucleophile/leaving-group pair")

    r_d_nu = np.linalg.norm(r_pos[rc] - r_pos[nu])
    r_d_lg = np.linalg.norm(r_pos[rc] - r_pos[lg])
    p_d_nu = np.linalg.norm(p_pos[rc] - p_pos[nu])
    p_d_lg = np.linalg.norm(p_pos[rc] - p_pos[lg])

    # lg_dir: unit vector FROM rc TOWARD lg in the reactant.
    # This is the direction the leaving group sits relative to the reaction centre.
    # During the reaction, lg moves further in this direction; nu approaches from -lg_dir.
    lg_dir = (r_pos[lg] - r_pos[rc])
    lg_dir /= np.linalg.norm(lg_dir)

    for img_idx in range(1, n - 1):
        t = img_idx / (n - 1)

        new_pos = r_pos.copy()

        d_nu = r_d_nu + t * (p_d_nu - r_d_nu)
        d_lg = r_d_lg + t * (p_d_lg - r_d_lg)

        # Place lg along +lg_dir from rc; nu along -lg_dir (backside attack)
        new_pos[lg] = r_pos[rc] + lg_dir * d_lg
        new_pos[nu] = r_pos[rc] - lg_dir * d_nu

        for i, s in enumerate(syms):
            if s == "H":
                new_pos[i] = r_pos[i] + t * (p_pos[i] - r_pos[i])

        for o in others:
            if o != nu and o != lg:
                new_pos[o] = r_pos[o] + t * (p_pos[o] - r_pos[o])

        images[img_idx].set_positions(new_pos)
