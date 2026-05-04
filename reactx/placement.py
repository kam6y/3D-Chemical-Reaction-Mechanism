"""Generic steric-aware fragment placement (Phase 7).

Replaces the Tier 1 / Tier 2 dispatch of embed3d.py with a single algorithm:
Fibonacci-sphere sample of N candidate directions from the substrate-side
anchor, two-stage blocking filter (angular shadow + d_min ceiling), and
per-direction d_min based on each fragment's projection onto d.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def sample_sphere_directions(n: int, seed: int = 0) -> list[np.ndarray]:
    """Fibonacci sphere over full 4π sr; return n unit vectors.

    Index 0 is deterministically +z (re-producibility / phase origin).
    Remaining n-1 points are placed on the unit sphere by golden-angle spiral
    in z ∈ [-1, 1]; seed deterministically rotates the spiral phase.

    Raises:
        ValueError: when n < 1.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    rotations: list[np.ndarray] = [np.array([0.0, 0.0, 1.0])]
    if n == 1:
        return rotations

    rng = np.random.default_rng(seed)
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))

    for i in range(n - 1):
        # i ∈ [0, n-2] → t ∈ (0, 1) → z ∈ (1 - 2*t) ∈ (-1, 1)
        t = (i + 0.5) / (n - 1)
        z = 1.0 - 2.0 * t
        r_xy = float(np.sqrt(max(0.0, 1.0 - z * z)))
        theta = phase + i * golden_angle
        x = r_xy * float(np.cos(theta))
        y = r_xy * float(np.sin(theta))
        v = np.array([x, y, z])
        # 数値誤差対策で再正規化
        rotations.append(v / float(np.linalg.norm(v)))
    return rotations
