"""Unit tests for reactx.placement (Phase 7)."""
import numpy as np
import pytest

from reactx.placement import compute_d_min, evaluate_direction, sample_sphere_directions


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


def test_compute_d_min_collinear_atoms_along_d():
    # anchor at origin, substrate atom at (2,0,0) vdW=1.5,
    # incoming atom at incoming_anchor (0,0,0)+(0,0,0)=origin vdW=1.5
    # d = +x → max_substrate_fwd = 2.0 + 1.5 = 3.5
    # incoming with single atom at incoming_anchor → max_incoming_back = 0.0 + 1.5 = 1.5
    # d_min = 3.5 + 1.5 + 0.5 = 5.5
    d = np.array([1.0, 0.0, 0.0])
    anchor = np.zeros(3)
    sub_pos = np.array([[2.0, 0.0, 0.0]])
    sub_vdw = np.array([1.5])
    inc_pos = np.array([[0.0, 0.0, 0.0]])
    inc_vdw = np.array([1.5])
    inc_anchor = np.zeros(3)
    d_min = compute_d_min(d, anchor, sub_pos, sub_vdw, inc_pos, inc_vdw, inc_anchor, gap=0.5)
    assert d_min == pytest.approx(5.5)


def test_compute_d_min_substrate_behind_anchor_does_not_inflate():
    # substrate atom is at (-2, 0, 0); along +x its projection is -2 + 1.5 = -0.5 (negative)
    # incoming single atom at incoming_anchor → max_incoming_back = 1.5
    # d_min = max(-0.5, ...) → 実装で max(0, ...) にしないと負になりうる
    # 仕様: max_substrate_fwd は substrate atoms の (proj + vdw) の単純 max を取る。
    # 後ろの原子も計算されるが、より前にある原子があればそちらが勝つ。
    # ここではサブ atom が anchor 自身の +d 方向「後ろ」だけにあるケースで、
    # max_substrate_fwd は最大 (proj + vdw) なので -0.5 となるが、
    # その場合 incoming_back (=1.5) + gap (=0.5) より小さくても合算が d_min。
    d = np.array([1.0, 0.0, 0.0])
    anchor = np.zeros(3)
    sub_pos = np.array([[-2.0, 0.0, 0.0]])
    sub_vdw = np.array([1.5])
    inc_pos = np.array([[0.0, 0.0, 0.0]])
    inc_vdw = np.array([1.5])
    inc_anchor = np.zeros(3)
    d_min = compute_d_min(d, anchor, sub_pos, sub_vdw, inc_pos, inc_vdw, inc_anchor, gap=0.5)
    # max_substrate_fwd = -2.0 + 1.5 = -0.5
    # max_incoming_back = 0.0 + 1.5 = 1.5
    # d_min = -0.5 + 1.5 + 0.5 = 1.5
    assert d_min == pytest.approx(1.5)


def test_evaluate_direction_blocked_by_angle_shadow():
    # anchor at origin, atom at (0.5, 0, 0) vdW=1.5 (very close, big cone).
    # d = +x → angle to atom = 0; cone half-angle = atan(1.5/0.5) = 71.6° → blocked.
    d = np.array([1.0, 0.0, 0.0])
    anchor = np.zeros(3)
    sub_pos = np.array([[0.5, 0.0, 0.0]])
    sub_vdw = np.array([1.5])
    inc_pos = np.array([[0.0, 0.0, 0.0]])
    inc_vdw = np.array([1.5])
    inc_anchor = np.zeros(3)
    blocked, d_min, reason = evaluate_direction(
        d, anchor, sub_pos, sub_vdw, inc_pos, inc_vdw, inc_anchor,
        gap=0.5, d_min_ceiling=8.0,
    )
    assert blocked is True
    assert reason is not None and reason.startswith("angle_shadow")
    assert np.isnan(d_min)


def test_evaluate_direction_blocked_by_d_min_ceiling():
    # 角度シャドウは抜ける配置: substrate atom を −x に置く (d=+x なら angle = 180°)
    # でも incoming fragment が異常に長い → d_min > ceiling
    d = np.array([1.0, 0.0, 0.0])
    anchor = np.zeros(3)
    sub_pos = np.array([[-2.0, 0.0, 0.0]])  # 後ろ → angle shadow なし
    sub_vdw = np.array([0.5])
    # incoming atom が −d 方向に 10 Å 飛び出している
    inc_pos = np.array([[10.0, 0.0, 0.0]])  # incoming_anchor から見て (−d)·(p−inc_anchor) が大
    inc_vdw = np.array([0.5])
    inc_anchor = np.zeros(3)
    # max_substrate_fwd = -2 + 0.5 = -1.5
    # max_incoming_back = (10 - 0)·(−1) + 0.5 = −9.5 (負)
    # ↑ wait: -d を使う。p=(10,0,0), inc_anchor=(0,0,0), p - inc_anchor = (10,0,0)
    # (p - inc_anchor) · (-d) = 10 * (-1) = -10. + 0.5 = -9.5。これは max なので負のまま。
    # d_min = -1.5 + (-9.5) + 0.5 = -10.5 → ceiling 上にいかない。
    # テストとして向きを変えよう: incoming の atom を −x 方向に置く
    inc_pos = np.array([[-10.0, 0.0, 0.0]])
    # (p - inc_anchor) · (-d) = (-10) * (-1) = 10. + 0.5 = 10.5 → max_incoming_back = 10.5
    # d_min = -1.5 + 10.5 + 0.5 = 9.5 > ceiling=8.0 → blocked.
    blocked, d_min, reason = evaluate_direction(
        d, anchor, sub_pos, sub_vdw, inc_pos, inc_vdw, inc_anchor,
        gap=0.5, d_min_ceiling=8.0,
    )
    assert blocked is True
    assert reason is not None and reason.startswith("d_min_ceiling")
    assert d_min == pytest.approx(9.5)


def test_evaluate_direction_unblocked_clear_path():
    # substrate atoms を −x に固める、d=+x → 角度シャドウなし、d_min も小さい。
    d = np.array([1.0, 0.0, 0.0])
    anchor = np.zeros(3)
    sub_pos = np.array([
        [-1.0, 0.0, 0.0],
        [-1.0, 0.5, 0.0],
        [-1.0, -0.5, 0.0],
    ])
    sub_vdw = np.array([0.5, 0.5, 0.5])
    inc_pos = np.array([[0.0, 0.0, 0.0]])
    inc_vdw = np.array([0.5])
    inc_anchor = np.zeros(3)
    blocked, d_min, reason = evaluate_direction(
        d, anchor, sub_pos, sub_vdw, inc_pos, inc_vdw, inc_anchor,
        gap=0.5, d_min_ceiling=8.0,
    )
    assert blocked is False
    assert reason is None
    # max_substrate_fwd = max(-1 + 0.5, ...) = -0.5
    # max_incoming_back = 0 + 0.5 = 0.5
    # d_min = -0.5 + 0.5 + 0.5 = 0.5
    assert d_min == pytest.approx(0.5)
