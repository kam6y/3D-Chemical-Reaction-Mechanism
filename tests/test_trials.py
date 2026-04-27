"""Unit tests for reactx.trials."""

import numpy as np
import pytest

from reactx.trials import sample_attack_rotations


def test_n1_returns_identity():
    rots = sample_attack_rotations(1)
    assert len(rots) == 1
    np.testing.assert_allclose(rots[0], np.eye(3), atol=1e-12)


def test_n8_within_cone_30():
    rots = sample_attack_rotations(8, cone_half_deg=30.0, seed=0)
    assert len(rots) == 8
    z = np.array([0.0, 0.0, 1.0])
    for R in rots:
        v = R @ z
        cos_angle = float(np.clip(np.dot(z, v), -1.0, 1.0))
        angle_deg = np.degrees(np.arccos(cos_angle))
        assert angle_deg <= 30.0 + 1e-6, f"rotation rotates z by {angle_deg}° > 30°"


def test_seed_determinism():
    a = sample_attack_rotations(8, cone_half_deg=30.0, seed=42)
    b = sample_attack_rotations(8, cone_half_deg=30.0, seed=42)
    for ra, rb in zip(a, b, strict=True):
        np.testing.assert_allclose(ra, rb, atol=1e-12)


def test_first_rotation_always_identity():
    rots = sample_attack_rotations(8, cone_half_deg=30.0, seed=999)
    np.testing.assert_allclose(rots[0], np.eye(3), atol=1e-12)


def test_rotations_are_orthogonal():
    rots = sample_attack_rotations(8, cone_half_deg=30.0, seed=0)
    for R in rots:
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        det = np.linalg.det(R)
        assert abs(det - 1.0) < 1e-10, f"det(R)={det}, not a rotation"
