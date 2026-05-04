"""Unit tests for reactx.placement (Phase 7)."""
import numpy as np
import pytest

from reactx.placement import sample_sphere_directions


def test_sample_sphere_returns_unit_vectors():
    dirs = sample_sphere_directions(64, seed=0)
    assert len(dirs) == 64
    for d in dirs:
        assert d.shape == (3,)
        assert np.linalg.norm(d) == pytest.approx(1.0, abs=1e-9)


def test_sample_sphere_index_zero_is_z():
    dirs = sample_sphere_directions(64, seed=0)
    np.testing.assert_allclose(dirs[0], [0.0, 0.0, 1.0])


def test_sample_sphere_seed_determinism():
    a = sample_sphere_directions(32, seed=42)
    b = sample_sphere_directions(32, seed=42)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


def test_sample_sphere_different_seed_gives_different_directions():
    a = sample_sphere_directions(32, seed=1)
    b = sample_sphere_directions(32, seed=2)
    # index 0 は両方 +z で一致、その他は螺旋 phase が異なるので少なくとも 1 つ違う
    differs = any(not np.allclose(x, y) for x, y in zip(a[1:], b[1:], strict=True))
    assert differs


def test_sample_sphere_uniform_coverage_4pi_sr():
    # n=64 では Fibonacci 球面の最近接ペア角度 < ~25° 程度が経験則
    dirs = sample_sphere_directions(64, seed=0)
    arr = np.array(dirs)
    # 自己除外で各点の最近接角度 (rad) を取り、その最大が閾値未満であることを確認
    cos_pairs = arr @ arr.T
    np.fill_diagonal(cos_pairs, -1.0)  # 自己ペア除外
    nearest_cos = cos_pairs.max(axis=1)
    nearest_angle_deg = np.degrees(np.arccos(np.clip(nearest_cos, -1.0, 1.0)))
    assert nearest_angle_deg.max() < 30.0  # 64 点なら ~25° 以内が経験則


def test_sample_sphere_n_one_returns_only_z():
    dirs = sample_sphere_directions(1, seed=0)
    assert len(dirs) == 1
    np.testing.assert_allclose(dirs[0], [0.0, 0.0, 1.0])


def test_sample_sphere_n_zero_raises():
    with pytest.raises(ValueError, match="n must be >= 1"):
        sample_sphere_directions(0, seed=0)
