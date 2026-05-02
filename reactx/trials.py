"""Sample N rotation matrices that perturb a reference direction within a cone.

Used to generate multi-angle attack trials for bimolecular reactions: each
rotation is applied to the ideal backside-attack direction to obtain a
different incoming direction for the nucleophile fragment.

Index 0 is always the identity rotation (= no perturbation, equivalent to
Phase 0 single-trial behavior). Remaining n-1 rotations are sampled
deterministically by Fibonacci spiral within `cone_half_deg`.
"""

from __future__ import annotations

import numpy as np


def sample_attack_rotations(
    n: int,
    cone_half_deg: float = 30.0,
    seed: int = 0,
) -> list[np.ndarray]:
    """Return n 3x3 rotation matrices.

    R[0] is identity. R[1..n-1] are sampled by Fibonacci spiral in the cap of
    a unit sphere of half-angle cone_half_deg around +z, then converted to
    rotation matrices that map +z to each sampled direction. Applied to a
    reference direction d, R @ d rotates d by an angle <= cone_half_deg.

    `seed` deterministically perturbs the spiral phase so trials with different
    seeds are independent.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    rotations: list[np.ndarray] = [np.eye(3)]
    if n == 1:
        return rotations

    cos_cap = float(np.cos(np.radians(cone_half_deg)))
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    for i in range(n - 1):
        # Fibonacci spiral on the spherical cap above z = cos_cap.
        # i in [0, n-2] -> z in (cos_cap, 1).
        t = (i + 0.5) / (n - 1)
        z = 1.0 - t * (1.0 - cos_cap)
        r_xy = float(np.sqrt(max(0.0, 1.0 - z * z)))
        theta = phase + i * golden_angle
        x = r_xy * float(np.cos(theta))
        y = r_xy * float(np.sin(theta))
        target = np.array([x, y, z])
        rotations.append(_rotation_from_z_to(target))
    return rotations


def _rotation_from_z_to(v: np.ndarray) -> np.ndarray:
    """Return a 3x3 rotation matrix R such that R @ [0,0,1] = v (unit vector).

    Uses Rodrigues' rotation formula. Handles parallel/antiparallel cases
    explicitly.
    """
    z = np.array([0.0, 0.0, 1.0])
    v = v / np.linalg.norm(v)
    cos_theta = float(np.dot(z, v))
    if cos_theta > 1.0 - 1e-12:
        return np.eye(3)
    if cos_theta < -1.0 + 1e-12:
        # 180 degree rotation around any axis perpendicular to z; pick x-axis.
        return np.diag([1.0, -1.0, -1.0])
    axis = np.cross(z, v)
    axis = axis / np.linalg.norm(axis)
    sin_theta = float(np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta)))
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + sin_theta * K + (1.0 - cos_theta) * (K @ K)
