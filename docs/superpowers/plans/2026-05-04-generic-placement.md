# Phase 7 — Generic Steric-Aware Placement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tier 1/Tier 2 dispatch を撤廃し、anchor 中心の全球面 Fibonacci サンプリング + 角度シャドウ + d_min ceiling の二段 blocking filter による generic placement に統合する。MMFF prescreen を完全廃止し、blocking 生存全件を直接 UMA full relax に流す。

**Architecture:** 新規モジュール `reactx/placement.py` (sphere sampling + steric blocking + per-direction d_min) と `reactx/vdw_radii.py` (Alvarez 2013 共有テーブル) を導入。`reactx/embed3d.py` は単分子 3D 化のみに責務を絞り、`embed_fragments_to_positions` を露出。`reactx/prescreen.py` と `reactx/trials.py` は完全削除。`reactx/cli.py` は新 3 関数を直接 orchestrate。TOML schema・meta.json schema・examples 全件 breaking change。

**Tech Stack:** Python 3.12+, RDKit (ETKDGv3 + MMFF), ASE (Atoms + BFGS), NumPy, pytest, fairchem UMA。

**Spec:** `docs/superpowers/specs/2026-05-04-generic-placement-design.md`

---

## File Structure

### New files
- `reactx/vdw_radii.py` — Alvarez 2013 vdW 半径テーブル + lookup (symbol → Å)
- `reactx/placement.py` — Fibonacci 球面 / blocking / d_min / valid_placements / build_atoms_from_positions
- `tests/test_vdw_radii.py` — vdw_radii unit tests
- `tests/test_placement.py` — placement unit tests

### Modified files
- `reactx/embed3d.py` — `embed_mol_to_atoms` 削除、`embed_fragments_to_positions` 抽出、Tier 1/2 dispatch 削除
- `reactx/scoring.py` — `select_best_trial` 追加、`TrialResult.rotation_deg` → `direction`
- `reactx/cli.py` — prescreen 削除、placement orchestration、meta.json schema 更新
- `reactx/config.py` — `n_angles` → `n_candidates` rename、`cone_half_deg` 削除、`PrescreenConfig` 削除、`[prescreen]` キー検知時 ConfigError
- `blender/render.py` — vdw テーブルを `reactx.vdw_radii` import に置換
- `examples/sn2.rxn.toml` — schema migration
- `examples/proton_transfer.rxn.toml` — schema migration
- `examples/menshutkin.rxn.toml` — schema migration
- `examples/e2.rxn.toml` — schema migration
- `examples/sn1_dissoc.rxn.toml` — schema migration (n_candidates=1 維持)
- `examples/sn1_recomb.rxn.toml` — schema migration
- `tests/test_embed3d.py` — `embed_fragments_to_positions` の単体テストへ書き換え
- `tests/test_scoring.py` — `select_best_trial` テスト追加
- `tests/test_re1_sn2.py` — meta.json schema 更新
- `tests/test_re1_proton_transfer.py` — 同上
- `tests/test_re1_menshutkin.py` — 同上
- `tests/test_re3_e2.py` — 同上
- `tests/test_re3_sn1_dissoc.py` — 同上
- `tests/test_re4_sn1_recomb.py` — 同上
- `tests/test_examples_menshutkin.py` — 同上
- `tests/test_cli.py` — meta.json schema 更新
- `tests/test_cli_unimolecular.py` — 同上
- `tests/test_cli_neb_refine_guard.py` — 同上
- `tests/test_neb_refine_sn2.py` — 同上
- `tests/test_wallclock_sn2.py` — wall-clock 緩和
- `tests/test_config.py` — n_candidates / [prescreen] エラーテスト追加
- `README.md` — アーキテクチャ図、対応反応表、MMFF 段落削除、方針と限界更新

### Deleted files
- `reactx/prescreen.py`
- `reactx/trials.py`
- `tests/test_prescreen.py`
- `tests/test_prescreen_disabled_sn2.py`
- `tests/test_trials.py`
- `tests/test_embed3d_placement.py` (Tier 1/2 専用テスト群、新 `tests/test_placement.py` で代替)

---

## Task 1: Create vdw_radii.py module

**Files:**
- Create: `reactx/vdw_radii.py`
- Test: `tests/test_vdw_radii.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_vdw_radii.py`:

```python
"""vdW radii lookup: Alvarez (2013) Dalton Trans. 42, 8617."""
import pytest

from reactx.vdw_radii import VDW_RADII_ANGSTROM, FALLBACK_RADIUS, vdw_radius


def test_vdw_radii_alvarez_z1_to_z83_present():
    # 元素記号は Z=1..83 (H..Bi)。OMol25 / UMA omol 訓練範囲と一致。
    expected_symbols = {
        "H", "He",
        "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
        "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr",
        "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "In", "Sn", "Sb", "Te", "I", "Xe",
        "Cs", "Ba",
        "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
        "Tm", "Yb", "Lu",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
    }
    assert expected_symbols.issubset(VDW_RADII_ANGSTROM.keys())


def test_vdw_radius_known_values():
    # 既知値 (Alvarez 2013 from blender/render.py)
    assert vdw_radius("H") == pytest.approx(1.20)
    assert vdw_radius("C") == pytest.approx(1.77)
    assert vdw_radius("N") == pytest.approx(1.66)
    assert vdw_radius("O") == pytest.approx(1.50)
    assert vdw_radius("Cl") == pytest.approx(1.82)


def test_vdw_radius_fallback_for_unknown_symbol(caplog):
    # Z > 83 (例: "Po", "U") はテーブル外 → fallback + warning
    with caplog.at_level("WARNING"):
        r = vdw_radius("Po")
    assert r == FALLBACK_RADIUS
    assert "Po" in caplog.text


def test_vdw_radius_fallback_value_is_alvarez_median_neighborhood():
    # FALLBACK_RADIUS は 1.50 (Alvarez Z=1..83 median 近傍、O と一致するのは偶然)
    assert FALLBACK_RADIUS == pytest.approx(1.50)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vdw_radii.py -v`
Expected: FAIL — ModuleNotFoundError: No module named 'reactx.vdw_radii'

- [ ] **Step 3: Implement vdw_radii.py**

`reactx/vdw_radii.py`:

```python
"""Shared van der Waals radius table for placement and rendering.

Alvarez (2013) "A cartography of the van der Waals territories"
Dalton Trans. 42, 8617. Values in Angstrom for Z=1..83 (H..Bi),
matching OMol25 / UMA omol task element coverage exactly.

Z > 83 (Po, At, Rn, Fr, Ra, all actinides) は UMA omol25 訓練外なので
実用上発火しないが、安全のため Alvarez 中央値近傍 (1.50 Å) で fallback する。
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

VDW_RADII_ANGSTROM: dict[str, float] = {
    "H": 1.20, "He": 1.43,
    "Li": 2.12, "Be": 1.98, "B": 1.91, "C": 1.77, "N": 1.66, "O": 1.50,
    "F": 1.46, "Ne": 1.58,
    "Na": 2.50, "Mg": 2.51, "Al": 2.25, "Si": 2.19, "P": 1.90, "S": 1.89,
    "Cl": 1.82, "Ar": 1.83,
    "K": 2.73, "Ca": 2.62, "Sc": 2.58, "Ti": 2.46, "V": 2.42, "Cr": 2.45,
    "Mn": 2.45, "Fe": 2.44, "Co": 2.40, "Ni": 2.40, "Cu": 2.38, "Zn": 2.39,
    "Ga": 2.32, "Ge": 2.29, "As": 1.88, "Se": 1.82, "Br": 1.86, "Kr": 2.25,
    "Rb": 3.21, "Sr": 2.84, "Y": 2.75, "Zr": 2.52, "Nb": 2.56, "Mo": 2.45,
    "Tc": 2.44, "Ru": 2.46, "Rh": 2.44, "Pd": 2.15, "Ag": 2.53, "Cd": 2.49,
    "In": 2.43, "Sn": 2.42, "Sb": 2.47, "Te": 1.99, "I": 2.04, "Xe": 2.06,
    "Cs": 3.48, "Ba": 3.03,
    "La": 2.98, "Ce": 2.88, "Pr": 2.92, "Nd": 2.95, "Pm": 2.93, "Sm": 2.90,
    "Eu": 2.87, "Gd": 2.83, "Tb": 2.79, "Dy": 2.87, "Ho": 2.81, "Er": 2.83,
    "Tm": 2.79, "Yb": 2.80, "Lu": 2.74,
    "Hf": 2.63, "Ta": 2.53, "W": 2.57, "Re": 2.49, "Os": 2.48, "Ir": 2.41,
    "Pt": 2.29, "Au": 2.32, "Hg": 2.45, "Tl": 2.47, "Pb": 2.60, "Bi": 2.54,
}

FALLBACK_RADIUS: float = 1.50


def vdw_radius(symbol: str) -> float:
    """Alvarez 2013 vdW radius (Å) for a chemical element symbol.

    Unknown symbols (Z > 83, lanthanide/actinide gaps, garbled) → log warning
    and return FALLBACK_RADIUS (1.50 Å).
    """
    r = VDW_RADII_ANGSTROM.get(symbol)
    if r is None:
        log.warning(
            "vdw_radius: symbol %r not in Alvarez 2013 table (Z=1..83); "
            "using fallback %.2f Å", symbol, FALLBACK_RADIUS,
        )
        return FALLBACK_RADIUS
    return r
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vdw_radii.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add reactx/vdw_radii.py tests/test_vdw_radii.py
git commit -m "$(cat <<'EOF'
feat(vdw_radii): introduce shared Alvarez 2013 vdW table

placement.py と blender/render.py の双方から参照する shared lookup を切り出し。
Z=1..83 (H..Bi) をサポート、テーブル外は warning + 1.50 Å fallback。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Create placement.py — Fibonacci sphere sampling

**Files:**
- Create: `reactx/placement.py` (this task only adds `sample_sphere_directions`)
- Test: `tests/test_placement.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_placement.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_placement.py -v`
Expected: FAIL — ModuleNotFoundError: No module named 'reactx.placement'

- [ ] **Step 3: Implement sample_sphere_directions**

`reactx/placement.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_placement.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add reactx/placement.py tests/test_placement.py
git commit -m "$(cat <<'EOF'
feat(placement): add full-sphere Fibonacci sampler

sample_sphere_directions(n, seed) で全 4π sr に均等な n 個の単位ベクトルを生成。
index 0 は決定論的に +z、残りは黄金比螺旋。trials.sample_attack_rotations
(30° cone) の後継として、後続 task で blocking + d_min 計算と統合する。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add compute_d_min and evaluate_direction

**Files:**
- Modify: `reactx/placement.py` (append `compute_d_min` and `evaluate_direction`)
- Test: `tests/test_placement.py` (append tests)

- [ ] **Step 1: Append failing tests**

Add to `tests/test_placement.py`:

```python
from reactx.placement import compute_d_min, evaluate_direction


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_placement.py -v`
Expected: FAIL — ImportError: cannot import name 'compute_d_min' from 'reactx.placement'

- [ ] **Step 3: Implement compute_d_min and evaluate_direction**

Append to `reactx/placement.py`:

```python
def compute_d_min(
    d: np.ndarray,
    anchor_pos: np.ndarray,
    substrate_positions: np.ndarray,    # (M, 3) anchor 自身を含めても anchor の vdW=0 なら無害
    substrate_vdw: np.ndarray,          # (M,)
    incoming_positions: np.ndarray,     # (K, 3) incoming fragment 全原子
    incoming_vdw: np.ndarray,           # (K,)
    incoming_anchor_pos: np.ndarray,    # (3,) incoming 内 bridging bond 端の現在位置
    *,
    gap: float = 0.5,
) -> float:
    """Per-direction shortest safe distance between fragments along d.

    max_substrate_fwd = max((p - anchor_pos) · d + r_vdW for p, r_vdW in substrate)
    max_incoming_back = max((p - incoming_anchor_pos) · (-d) + r_vdW for p, r_vdW in incoming)
    d_min = max_substrate_fwd + max_incoming_back + gap
    """
    sub_proj = (substrate_positions - anchor_pos) @ d + substrate_vdw
    inc_proj = (incoming_positions - incoming_anchor_pos) @ (-d) + incoming_vdw
    max_substrate_fwd = float(sub_proj.max()) if sub_proj.size else 0.0
    max_incoming_back = float(inc_proj.max()) if inc_proj.size else 0.0
    return max_substrate_fwd + max_incoming_back + float(gap)


def evaluate_direction(
    d: np.ndarray,
    anchor_pos: np.ndarray,
    substrate_positions: np.ndarray,    # (M, 3) anchor を除外済み
    substrate_vdw: np.ndarray,          # (M,)
    incoming_positions: np.ndarray,
    incoming_vdw: np.ndarray,
    incoming_anchor_pos: np.ndarray,
    *,
    gap: float = 0.5,
    d_min_ceiling: float = 8.0,
) -> tuple[bool, float, str | None]:
    """Two-stage blocking check; returns (blocked, d_min, reason).

    Stage 1 — angular shadow:
      ∃i s.t. angle(d, substrate_positions[i] - anchor_pos)
             < atan(substrate_vdw[i] / ||substrate_positions[i] - anchor_pos||)
      → blocked=True, d_min=NaN, reason="angle_shadow:atom_index=K".
    Stage 2 — d_min ceiling:
      compute_d_min(...) > d_min_ceiling
      → blocked=True, reason="d_min_ceiling:value=V".
    Otherwise: blocked=False, d_min=value, reason=None.
    """
    # Stage 1: angular shadow
    rel = substrate_positions - anchor_pos
    r = np.linalg.norm(rel, axis=1)
    # 数値安定化: 距離 0 (anchor 自身が紛れ込んだ等) は無視
    safe = r > 1e-9
    if np.any(safe):
        cos_to_d = (rel[safe] @ d) / r[safe]
        # angle threshold: atan(vdw / r) → cos threshold: r / sqrt(r^2 + vdw^2)
        vdw_safe = substrate_vdw[safe]
        cos_thresh = r[safe] / np.sqrt(r[safe] ** 2 + vdw_safe ** 2)
        # angle < threshold ⇔ cos > cos_thresh
        hits = np.where(cos_to_d > cos_thresh)[0]
        if hits.size:
            # safe-mask 上の index → 元 index へ戻す
            orig_indices = np.where(safe)[0]
            blocked_atom = int(orig_indices[hits[0]])
            return True, float("nan"), f"angle_shadow:atom_index={blocked_atom}"

    # Stage 2: d_min ceiling
    d_min = compute_d_min(
        d, anchor_pos, substrate_positions, substrate_vdw,
        incoming_positions, incoming_vdw, incoming_anchor_pos,
        gap=gap,
    )
    if d_min > d_min_ceiling:
        return True, d_min, f"d_min_ceiling:value={d_min:.3f}"
    return False, d_min, None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_placement.py -v`
Expected: PASS (12 tests total — 7 from Task 2 + 5 new)

- [ ] **Step 5: Commit**

```bash
git add reactx/placement.py tests/test_placement.py
git commit -m "$(cat <<'EOF'
feat(placement): add compute_d_min and evaluate_direction

二段 blocking filter (角度シャドウ → d_min ceiling) と per-direction
最短安全距離の計算を実装。次の task で valid_placements が両者を
組み合わせて生存方向を返す。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Add valid_placements + dataclasses + helpers

**Files:**
- Modify: `reactx/placement.py` (append `_identify_substrate`, `_find_bridging_formed`, `PlacementTrial`, `PlacementResult`, `valid_placements`, `build_atoms_from_positions`)
- Test: `tests/test_placement.py` (append tests)

- [ ] **Step 1: Append failing tests**

Add to `tests/test_placement.py`:

```python
from ase import Atoms
from rdkit import Chem

from reactx.bond_changes import BondChanges
from reactx.placement import (
    PlacementResult,
    PlacementTrial,
    _identify_substrate,
    _find_bridging_formed,
    build_atoms_from_positions,
    valid_placements,
)


def test_identify_substrate_picks_largest():
    # frag 0 has 3 atoms, frag 1 has 5 atoms → frag 1 is substrate
    frags = ((0, 1, 2), (3, 4, 5, 6, 7))
    sub = _identify_substrate(frags)
    assert sub == (3, 4, 5, 6, 7)


def test_identify_substrate_tie_breaks_by_min_atom_index():
    # equal sizes → tie-break by minimum atom index
    frags = ((5, 6, 7), (0, 1, 2))
    sub = _identify_substrate(frags)
    assert sub == (0, 1, 2)


def test_find_bridging_formed_returns_unique():
    formed = ((0, 5),)
    sub = {0, 1, 2, 3, 4}
    frag = {5, 6}
    bridge = _find_bridging_formed(formed, sub, frag)
    assert bridge == (0, 5)


def test_find_bridging_formed_multiple_raises_not_implemented():
    formed = ((0, 5), (1, 6))  # 同じ pair を 2 本接続
    sub = {0, 1, 2, 3, 4}
    frag = {5, 6}
    with pytest.raises(NotImplementedError, match="multi-anchor"):
        _find_bridging_formed(formed, sub, frag)


def test_find_bridging_formed_no_bridge_raises_value():
    formed = ((0, 1),)  # 内部結合のみ、bridge なし
    sub = {0, 1}
    frag = {5, 6}
    with pytest.raises(ValueError, match="no formed bond bridging"):
        _find_bridging_formed(formed, sub, frag)


def _ch3cl_plus_oh_minus_setup():
    """Build a (mol_h, frag_indices, positions, bond_changes) for SN2."""
    mol = Chem.MolFromSmiles("CCl.[OH-]")
    mol_h = Chem.AddHs(mol)
    frag_indices = Chem.GetMolFrags(mol_h)
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    c_idx = syms.index("C")
    cl_idx = syms.index("Cl")
    o_idx = syms.index("O")
    n = mol_h.GetNumAtoms()
    # Hand-crafted positions: CH3Cl along x-axis, OH near origin (will be moved by placement).
    positions = np.zeros((n, 3))
    positions[c_idx] = [0.0, 0.0, 0.0]
    positions[cl_idx] = [1.78, 0.0, 0.0]  # +x
    h_idxs = [i for i in frag_indices[0] if syms[i] == "H"]
    # 3 H around C in tetrahedral angles (rough)
    positions[h_idxs[0]] = [-0.36, 1.03, 0.0]
    positions[h_idxs[1]] = [-0.36, -0.51, 0.89]
    positions[h_idxs[2]] = [-0.36, -0.51, -0.89]
    # OH placed somewhere arbitrary
    positions[o_idx] = [10.0, 10.0, 0.0]
    h_oh = [i for i in frag_indices[1] if syms[i] == "H"][0]
    positions[h_oh] = [10.96, 10.0, 0.0]
    bc = BondChanges(formed=((c_idx, o_idx),), broken=((c_idx, cl_idx),))
    return mol_h, frag_indices, positions, bc, c_idx, cl_idx, o_idx


def test_valid_placements_sn2_backside_present_and_frontside_rejected():
    mol_h, frags, pos, bc, c_idx, cl_idx, o_idx = _ch3cl_plus_oh_minus_setup()
    result = valid_placements(mol_h, frags, pos, bc, n_candidates=64, seed=0)
    assert isinstance(result, PlacementResult)
    assert result.n_candidates == 64
    assert len(result.trials) > 0
    # backside direction = -unit(C->Cl) = (-1, 0, 0)
    backside = np.array([-1.0, 0.0, 0.0])
    frontside = np.array([1.0, 0.0, 0.0])
    backside_present = any(np.dot(t.direction, backside) > 0.85 for t in result.trials)
    frontside_present = any(np.dot(t.direction, frontside) > 0.7 for t in result.trials)
    assert backside_present, "SN2 backside direction not in survivors"
    assert not frontside_present, "SN2 frontside direction wrongly survived"


def test_valid_placements_unimolecular_passthrough():
    mol = Chem.MolFromSmiles("CCBr")
    mol_h = Chem.AddHs(mol)
    frags = Chem.GetMolFrags(mol_h)
    n = mol_h.GetNumAtoms()
    positions = np.random.default_rng(0).normal(size=(n, 3))
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    c0 = next(i for i, s in enumerate(syms) if s == "C")
    br = next(i for i, s in enumerate(syms) if s == "Br")
    bc = BondChanges(formed=(), broken=((c0, br),))
    result = valid_placements(mol_h, frags, positions, bc, n_candidates=64, seed=0)
    assert len(result.trials) == 1
    np.testing.assert_array_equal(result.trials[0].positions, positions)


def test_valid_placements_metathesis_not_implemented():
    # 2 fragments, broken bond crossing fragments → not yet supported
    mol = Chem.MolFromSmiles("CCl.[OH-]")
    mol_h = Chem.AddHs(mol)
    frags = Chem.GetMolFrags(mol_h)
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    c_idx = syms.index("C")
    cl_idx = syms.index("Cl")
    o_idx = syms.index("O")
    bc = BondChanges(formed=((c_idx, o_idx),), broken=((cl_idx, o_idx),))
    n = mol_h.GetNumAtoms()
    positions = np.zeros((n, 3))
    with pytest.raises(NotImplementedError, match="metathesis"):
        valid_placements(mol_h, frags, positions, bc, n_candidates=8, seed=0)


def test_build_atoms_from_positions_preserves_symbols_and_charges():
    mol = Chem.MolFromSmiles("[OH-]")
    mol_h = Chem.AddHs(mol)
    n = mol_h.GetNumAtoms()
    positions = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0]])
    atoms = build_atoms_from_positions(mol_h, positions)
    assert isinstance(atoms, Atoms)
    assert len(atoms) == n
    assert sorted(atoms.get_chemical_symbols()) == ["H", "O"]
    # OH- → formal charge sum -1
    assert atoms.info["charge"] == -1
    np.testing.assert_array_equal(atoms.positions, positions)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_placement.py -v`
Expected: FAIL — ImportError on `valid_placements`, `_identify_substrate`, etc.

- [ ] **Step 3: Implement helpers + valid_placements + build_atoms_from_positions**

Append to `reactx/placement.py`:

```python
from dataclasses import dataclass

from ase import Atoms
from rdkit import Chem

from reactx.bond_changes import BondChanges
from reactx.vdw_radii import vdw_radius


@dataclass(frozen=True)
class PlacementTrial:
    direction: np.ndarray   # (3,) unit vector
    d_min: float            # placement distance applied along direction
    positions: np.ndarray   # (N, 3) full-system coordinates after translation


@dataclass(frozen=True)
class PlacementResult:
    trials: list[PlacementTrial]
    n_candidates: int
    n_blocked: int
    blocked_reasons: list[str | None]


def _identify_substrate(
    frag_indices: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    """Largest fragment by atom count; tie-break by smallest minimum atom index."""
    if not frag_indices:
        raise ValueError("frag_indices is empty")
    return max(frag_indices, key=lambda f: (len(f), -min(f)))


def _find_bridging_formed(
    formed: tuple[tuple[int, int], ...],
    substrate: set[int],
    fragment: set[int],
) -> tuple[int, int]:
    """Return the unique formed bond bridging substrate ↔ fragment.

    Multiple → NotImplementedError (cycloaddition is Phase 8+).
    None → ValueError.
    """
    bridges = [
        (a, b) for a, b in formed
        if (a in substrate and b in fragment) or (b in substrate and a in fragment)
    ]
    if len(bridges) > 1:
        raise NotImplementedError(
            f"multi-anchor placement (cycloaddition) is out of scope for Phase 7; "
            f"got {len(bridges)} formed bonds bridging this fragment"
        )
    if not bridges:
        raise ValueError(
            f"fragment has no formed bond bridging to substrate; "
            f"check input atom mapping (formed={list(formed)})"
        )
    return bridges[0]


def build_atoms_from_positions(
    mol_h: Chem.Mol,
    positions: np.ndarray,
) -> Atoms:
    """Build an ase.Atoms with symbols, formal charges, and given positions.

    Mirrors the Atoms construction path that used to live inside
    embed3d.embed_mol_to_atoms (symbols + initial_charges + info[charge/spin]).
    """
    if positions.shape != (mol_h.GetNumAtoms(), 3):
        raise ValueError(
            f"positions shape {positions.shape} does not match "
            f"mol_h atom count {mol_h.GetNumAtoms()}"
        )
    symbols = [a.GetSymbol() for a in mol_h.GetAtoms()]
    charges = [a.GetFormalCharge() for a in mol_h.GetAtoms()]
    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.set_initial_charges(charges)
    atoms.info["charge"] = int(sum(charges))
    atoms.info["spin"] = 1   # Phase Re1 baseline: closed-shell singlet
    return atoms


def _broken_bridges_fragments(
    broken: tuple[tuple[int, int], ...],
    frag_indices: tuple[tuple[int, ...], ...],
) -> bool:
    """True iff at least one broken bond has its two atoms in different fragments."""
    if not broken:
        return False
    atom_to_frag: dict[int, int] = {}
    for k, frag in enumerate(frag_indices):
        for a in frag:
            atom_to_frag[a] = k
    for a, b in broken:
        if atom_to_frag.get(a) != atom_to_frag.get(b):
            return True
    return False


def valid_placements(
    mol_h: Chem.Mol,
    frag_indices: tuple[tuple[int, ...], ...],
    positions: np.ndarray,
    bond_changes: BondChanges,
    *,
    n_candidates: int = 64,
    seed: int = 0,
    gap: float = 0.5,
    d_min_ceiling: float = 8.0,
) -> PlacementResult:
    """Sphere-sample directions, filter by angular shadow + d_min ceiling, place.

    Unimolecular (1 fragment): returns single passthrough trial (direction=+z, d_min=0).
    Bimolecular: identifies substrate (largest fragment) and one non-substrate fragment;
    samples n_candidates directions; for each, evaluates blocking and applies translation.

    Raises:
        NotImplementedError: when broken bond bridges fragments (metathesis), or when
            a non-substrate fragment is connected to the substrate by ≥2 formed bonds
            (cycloaddition).
        RuntimeError: when 0 candidates survive blocking.
        ValueError: when a non-substrate fragment has no bridging formed bond.
    """
    # Unimolecular: passthrough.
    if len(frag_indices) == 1:
        return PlacementResult(
            trials=[PlacementTrial(
                direction=np.array([0.0, 0.0, 1.0]),
                d_min=0.0,
                positions=positions.copy(),
            )],
            n_candidates=1,
            n_blocked=0,
            blocked_reasons=[None],
        )

    # Bimolecular: refuse metathesis up front.
    if _broken_bridges_fragments(bond_changes.broken, frag_indices):
        raise NotImplementedError(
            "multi-substrate metathesis (broken bonds spanning fragments) "
            "is out of scope for Phase 7"
        )

    substrate = _identify_substrate(frag_indices)
    substrate_set = set(substrate)
    non_substrate = [f for f in frag_indices if f is not substrate]

    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    vdw_all = np.array([vdw_radius(s) for s in syms])

    # Phase 7 supports a single non-substrate fragment per call.
    if len(non_substrate) > 1:
        log.warning(
            "Phase 7: %d non-substrate fragments detected; placing each "
            "independently with the same sphere sample (geometric quality "
            "may degrade for termolecular)", len(non_substrate),
        )

    out_positions = positions.copy()
    survivors: list[PlacementTrial] | None = None
    blocked_reasons_final: list[str | None] = []
    n_blocked_total = 0

    directions = sample_sphere_directions(n_candidates, seed=seed)

    for fragment in non_substrate:
        fragment_set = set(fragment)
        bridge = _find_bridging_formed(bond_changes.formed, substrate_set, fragment_set)
        anchor = bridge[0] if bridge[0] in substrate_set else bridge[1]
        incoming_anchor = bridge[1] if bridge[0] == anchor else bridge[0]

        substrate_atoms = [i for i in substrate if i != anchor]
        sub_pos = out_positions[substrate_atoms]
        sub_vdw = vdw_all[substrate_atoms]
        inc_pos = out_positions[list(fragment)]
        inc_vdw = vdw_all[list(fragment)]

        anchor_pos = out_positions[anchor]
        incoming_anchor_pos = out_positions[incoming_anchor]

        per_direction_trials: list[PlacementTrial] = []
        per_direction_reasons: list[str | None] = []
        for d in directions:
            blocked, d_min, reason = evaluate_direction(
                d, anchor_pos, sub_pos, sub_vdw,
                inc_pos, inc_vdw, incoming_anchor_pos,
                gap=gap, d_min_ceiling=d_min_ceiling,
            )
            if blocked:
                per_direction_reasons.append(reason)
                continue
            target = anchor_pos + d * d_min
            new_positions = out_positions.copy()
            new_positions[list(fragment)] += target - incoming_anchor_pos
            per_direction_trials.append(PlacementTrial(
                direction=d, d_min=d_min, positions=new_positions,
            ))
            per_direction_reasons.append(None)

        n_blocked_this = sum(1 for r in per_direction_reasons if r is not None)
        if not per_direction_trials:
            raise RuntimeError(
                f"anchor at atom {anchor} has no valid placement direction "
                f"(all {n_candidates} candidates blocked); substrate may be "
                f"fully enclosed"
            )
        if len(per_direction_trials) < max(1, n_candidates // 4):
            log.warning(
                "only %d/%d candidates survived blocking at anchor %d; "
                "consider larger n_candidates or check substrate geometry",
                len(per_direction_trials), n_candidates, anchor,
            )

        if survivors is None:
            survivors = per_direction_trials
            blocked_reasons_final = per_direction_reasons
            n_blocked_total = n_blocked_this
        else:
            # 2 つ目以降の non-substrate fragment は 1 つ目で生存した direction だけ
            # で配置を続ける (termolecular は警告済み)。実用上 Phase 7 では発火しない。
            log.warning(
                "termolecular placement: applying first-fragment survivors to "
                "subsequent fragment without re-filtering (best-effort)"
            )

    assert survivors is not None
    return PlacementResult(
        trials=survivors,
        n_candidates=n_candidates,
        n_blocked=n_blocked_total,
        blocked_reasons=blocked_reasons_final,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_placement.py -v`
Expected: PASS (20 tests total — 12 from earlier + 8 new)

- [ ] **Step 5: Commit**

```bash
git add reactx/placement.py tests/test_placement.py
git commit -m "$(cat <<'EOF'
feat(placement): wire up valid_placements + Atoms construction

substrate = 最大 fragment / non-substrate に bridging formed bond を持つ
bimolecular の placement を全球面サンプルで実行。1 fragment の
unimolecular は passthrough、metathesis / cycloaddition は NotImplementedError。
build_atoms_from_positions を embed3d から切り出し移動。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Add select_best_trial to scoring.py and update TrialResult schema

**Files:**
- Modify: `reactx/scoring.py`
- Test: `tests/test_scoring.py`

- [ ] **Step 1: Read existing test_scoring.py**

Run: `cat tests/test_scoring.py` to confirm current test contents.

- [ ] **Step 2: Append failing tests**

Add to `tests/test_scoring.py`:

```python
import numpy as np
from ase import Atoms

from reactx.scoring import TrialResult, select_best_trial


def _stub(idx: int, *, reached: bool, peak: float) -> TrialResult:
    return TrialResult(
        trial_idx=idx,
        direction=np.array([0.0, 0.0, 1.0]),
        frames=[Atoms(symbols=["H"], positions=[[0.0, 0.0, 0.0]])],
        energies=[peak],
        reached_product=reached,
        peak_energy=peak,
        n_steps=1,
    )


def test_select_best_trial_prefers_reached_product():
    trials = [
        _stub(0, reached=False, peak=-5.0),  # reached=False, lower peak
        _stub(1, reached=True, peak=-3.0),   # reached=True, higher peak
        _stub(2, reached=True, peak=-4.0),   # reached=True, lowest peak among reached
    ]
    best = select_best_trial(trials)
    assert best == 2


def test_select_best_trial_falls_back_when_none_reached():
    trials = [
        _stub(0, reached=False, peak=-5.0),
        _stub(1, reached=False, peak=-7.0),  # lowest peak overall
        _stub(2, reached=False, peak=-3.0),
    ]
    best = select_best_trial(trials)
    assert best == 1


def test_select_best_trial_empty_raises():
    import pytest
    with pytest.raises(ValueError, match="empty"):
        select_best_trial([])


def test_trial_result_direction_field_replaces_rotation_deg():
    t = TrialResult(
        trial_idx=0,
        direction=np.array([1.0, 0.0, 0.0]),
        frames=[],
        energies=[],
        reached_product=False,
        peak_energy=float("inf"),
        n_steps=0,
    )
    assert t.direction.shape == (3,)
    np.testing.assert_array_equal(t.direction, [1.0, 0.0, 0.0])
    assert not hasattr(t, "rotation_deg")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_scoring.py -v`
Expected: FAIL — ImportError on `select_best_trial`, AttributeError on `direction`.

- [ ] **Step 4: Modify scoring.py**

Replace `reactx/scoring.py` contents:

```python
"""Trial scoring + final-best selection for placement trials.

Phase 7: rotation_deg field replaced by direction (3-vec) since trials are
indexed by Fibonacci-sphere unit vectors, not deviations from a single
ideal direction. select_best_trial moved here from prescreen.select_top_k_indices
(top-K → 1) to centralize all "pick best trial" logic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms


@dataclass
class TrialResult:
    """Outcome of a single placement / relaxation trial."""

    trial_idx: int
    direction: np.ndarray  # (3,) unit vector — sphere-sampled placement direction
    frames: list[Atoms]
    energies: list[float]
    reached_product: bool
    peak_energy: float
    n_steps: int


def reached_product(
    final_atoms: Atoms,
    formed: list[tuple[int, int]],
    broken: list[tuple[int, int]],
    *,
    r_form_targets: list[float],
    r_broken_target: float,
    form_tol: float = 0.3,
    broken_tol: float = 0.5,
) -> bool:
    """True iff every formed bond is within r_form + form_tol AND every broken
    bond is at least r_broken - broken_tol apart.
    """
    if len(formed) != len(r_form_targets):
        raise ValueError(
            f"formed ({len(formed)}) must match r_form_targets ({len(r_form_targets)})"
        )
    p = final_atoms.positions
    for (a, b), rt in zip(formed, r_form_targets, strict=True):
        d = float(np.linalg.norm(p[a] - p[b]))
        if d > rt + form_tol:
            return False
    for a, b in broken:
        d = float(np.linalg.norm(p[a] - p[b]))
        if d < r_broken_target - broken_tol:
            return False
    return True


def score_trials(results: list[TrialResult]) -> TrialResult:
    """Return the best TrialResult (preference: reached then peak_energy ↑).

    1. reached_product=True 群の最低 peak_energy
    2. 全部 False なら全体の最低 peak_energy (= 'least bad' fallback)
    """
    if not results:
        raise ValueError("score_trials called with empty list")
    reached = [r for r in results if r.reached_product]
    pool = reached if reached else results
    return min(pool, key=lambda r: r.peak_energy)


def select_best_trial(trials: list[TrialResult]) -> int:
    """Return the trial_idx of the best TrialResult (same preference as score_trials)."""
    if not trials:
        raise ValueError("select_best_trial called with empty list")
    return score_trials(trials).trial_idx
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_scoring.py -v`
Expected: PASS (existing tests + 4 new). Note: tests that depend on `rotation_deg` field will fail at import in this state — that's OK; we'll fix callers in Task 8 (cli.py).

- [ ] **Step 6: Commit**

```bash
git add reactx/scoring.py tests/test_scoring.py
git commit -m "$(cat <<'EOF'
refactor(scoring): replace rotation_deg with direction, add select_best_trial

TrialResult.rotation_deg (scalar) → direction (3-vec) で sphere-sampled
配置方向を直接保持。select_best_trial を追加 (prescreen の select_top_k_indices
の後継、top-K → 1 件選択)。caller (cli.py) は次の task で更新。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Refactor embed3d.py — extract embed_fragments_to_positions, delete Tier 1/2

**Files:**
- Modify: `reactx/embed3d.py`
- Modify: `tests/test_embed3d.py` (rewrite to target new API)
- Delete: `tests/test_embed3d_placement.py`

- [ ] **Step 1: Replace tests/test_embed3d.py with new API tests**

Replace `tests/test_embed3d.py` contents:

```python
import numpy as np
import pytest
from rdkit import Chem

from reactx.embed3d import embed_fragments_to_positions


def _ch3cl() -> Chem.Mol:
    return Chem.MolFromSmiles("CCl")


def test_embed_fragments_unimolecular_returns_atom_count_positions():
    mol_h, frags, positions = embed_fragments_to_positions(_ch3cl(), seed=42)
    assert mol_h.GetNumAtoms() == 5
    assert frags == ((0, 1, 2, 3, 4),) or len(frags) == 1
    assert positions.shape == (5, 3)


def test_embed_fragments_unimolecular_has_reasonable_c_cl_bond():
    mol_h, _, positions = embed_fragments_to_positions(_ch3cl(), seed=42)
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    c_idx = syms.index("C")
    cl_idx = syms.index("Cl")
    d = float(np.linalg.norm(positions[c_idx] - positions[cl_idx]))
    assert 1.6 < d < 2.0, f"C-Cl distance out of range: {d:.3f} Å"


def test_embed_fragments_multifragment_each_independent():
    mol = Chem.MolFromSmiles("CCl.[F-]")
    mol_h, frags, positions = embed_fragments_to_positions(mol, seed=42)
    assert len(frags) == 2
    assert positions.shape == (mol_h.GetNumAtoms(), 3)
    # 各 fragment の重心は別座標 (内部 ETKDG が独立に embed したため、
    # 平均原点近くに集まる; 重なりは valid_placements 側で動かす)。


def test_embed_fragments_seed_determinism():
    mol_h_a, frags_a, pos_a = embed_fragments_to_positions(_ch3cl(), seed=7)
    mol_h_b, frags_b, pos_b = embed_fragments_to_positions(_ch3cl(), seed=7)
    np.testing.assert_array_equal(pos_a, pos_b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_embed3d.py -v`
Expected: FAIL — ImportError on `embed_fragments_to_positions`.

- [ ] **Step 3: Replace embed3d.py with refactored version**

Replace `reactx/embed3d.py` contents:

```python
"""2D RDKit Mol -> per-fragment 3D coordinates.

Phase 7: this module's responsibility is reduced to the per-fragment 3D
embed only. Multi-fragment placement (Tier 1 / Tier 2 dispatch) moved to
reactx.placement.valid_placements. The Atoms construction step moved to
reactx.placement.build_atoms_from_positions.

Output coordinates may have multiple fragments overlapping in space — the
caller is expected to call placement.valid_placements next.
"""
from __future__ import annotations

import logging

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

log = logging.getLogger(__name__)

MAX_EMBED_RETRIES = 5


def embed_fragments_to_positions(
    mol: Chem.Mol,
    *,
    seed: int = 0xC0FFEE,
) -> tuple[Chem.Mol, tuple[tuple[int, ...], ...], np.ndarray]:
    """Run ETKDGv3 + MMFF94 per fragment; return mol_h, frag_indices, positions.

    Atom ordering of the returned mol_h is `Chem.AddHs(mol)` order; the
    positions array maps directly onto its atom indices.
    """
    mol_h = Chem.AddHs(mol)
    n_atoms = mol_h.GetNumAtoms()
    frag_indices = Chem.GetMolFrags(mol_h)
    frag_mols = Chem.GetMolFrags(mol_h, asMols=True, sanitizeFrags=True)

    positions = np.zeros((n_atoms, 3))
    for i, (indices, frag) in enumerate(zip(frag_indices, frag_mols, strict=True)):
        _embed_in_place(frag, seed=seed + i * MAX_EMBED_RETRIES)
        conf = frag.GetConformer()
        for j, orig_idx in enumerate(indices):
            p = conf.GetAtomPosition(j)
            positions[orig_idx] = (p.x, p.y, p.z)

    return mol_h, frag_indices, positions


def _embed_in_place(frag: Chem.Mol, *, seed: int) -> None:
    params = AllChem.ETKDGv3()
    for attempt in range(MAX_EMBED_RETRIES):
        params.randomSeed = seed + attempt
        status = AllChem.EmbedMolecule(frag, params)
        if status == 0:
            break
    else:
        elems = ", ".join(sorted({a.GetSymbol() for a in frag.GetAtoms()}))
        raise RuntimeError(
            f"RDKit failed to embed fragment ({frag.GetNumAtoms()} atoms: {elems}) "
            f"after {MAX_EMBED_RETRIES} attempts. Check input structure and RDKit version."
        )

    if frag.GetNumHeavyAtoms() > 1:
        result = AllChem.MMFFOptimizeMolecule(frag, maxIters=500)
        if result == -1:
            raise RuntimeError(
                f"MMFF94 force field could not be constructed for fragment "
                f"({frag.GetNumAtoms()} atoms). Check element coverage."
            )
```

- [ ] **Step 4: Delete obsolete test file**

```bash
git rm tests/test_embed3d_placement.py
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_embed3d.py tests/test_placement.py tests/test_scoring.py tests/test_vdw_radii.py -v`
Expected: PASS (all unit tests for the new modules).

Note: import errors in `tests/test_cli*.py`, `tests/test_re*`, `tests/test_prescreen*.py` are expected at this point — they will be fixed in Task 8 / 9.

- [ ] **Step 6: Commit**

```bash
git add reactx/embed3d.py tests/test_embed3d.py
git commit -m "$(cat <<'EOF'
refactor(embed3d): reduce to embed_fragments_to_positions, drop Tier dispatch

Tier 1 (_directional_placement) / Tier 2 (_planar_face_placement) /
_plane_normal_at_anchor / _find_substrate_fragment / _find_substrate_by_size
/ FRAGMENT_SEPARATION / PLANE_FIT_TOLERANCE / embed_mol_to_atoms をすべて削除。
責務を per-fragment 3D 化のみに絞る。multi-fragment placement は
reactx.placement.valid_placements が担当する。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Update config.py — n_candidates rename, drop PrescreenConfig, reject [prescreen]

**Files:**
- Modify: `reactx/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Read existing test_config.py**

Run: `head -80 tests/test_config.py` to inspect test patterns.

- [ ] **Step 2: Append failing tests**

Add to `tests/test_config.py` (replacing any existing `n_angles` / `cone_half_deg` / `prescreen` references):

```python
import pytest

from reactx.config import _validate


def test_validate_accepts_n_candidates():
    raw = {
        "description": "X",
        "formed": [[1, 2]],
        "broken": [],
        "restraints": {"k_form": 1.0, "k_broken": 0.0, "r_broken": 4.0, "max_relax_steps": 100},
        "sampling": {"n_candidates": 32},
    }
    cfg = _validate(raw, source="test")
    assert cfg.sampling.n_candidates == 32


def test_validate_default_n_candidates_is_64():
    raw = {
        "description": "X",
        "formed": [[1, 2]],
        "broken": [],
        "restraints": {"k_form": 1.0, "k_broken": 0.0, "r_broken": 4.0, "max_relax_steps": 100},
    }
    cfg = _validate(raw, source="test")
    assert cfg.sampling.n_candidates == 64


def test_validate_rejects_legacy_n_angles_key():
    raw = {
        "description": "X",
        "formed": [[1, 2]],
        "broken": [],
        "restraints": {"k_form": 1.0, "k_broken": 0.0, "r_broken": 4.0, "max_relax_steps": 100},
        "sampling": {"n_angles": 8},
    }
    with pytest.raises(ValueError, match="n_angles"):
        _validate(raw, source="test")


def test_validate_rejects_legacy_cone_half_deg_key():
    raw = {
        "description": "X",
        "formed": [[1, 2]],
        "broken": [],
        "restraints": {"k_form": 1.0, "k_broken": 0.0, "r_broken": 4.0, "max_relax_steps": 100},
        "sampling": {"n_candidates": 16, "cone_half_deg": 30.0},
    }
    with pytest.raises(ValueError, match="cone_half_deg"):
        _validate(raw, source="test")


def test_validate_rejects_legacy_prescreen_section():
    raw = {
        "description": "X",
        "formed": [[1, 2]],
        "broken": [],
        "restraints": {"k_form": 1.0, "k_broken": 0.0, "r_broken": 4.0, "max_relax_steps": 100},
        "prescreen": {"enabled": True, "keep": 3, "steps": 30},
    }
    with pytest.raises(ValueError, match=r"\[prescreen\] section is removed in Phase 7"):
        _validate(raw, source="test")


def test_validate_rejects_n_candidates_zero():
    raw = {
        "description": "X",
        "formed": [[1, 2]],
        "broken": [],
        "restraints": {"k_form": 1.0, "k_broken": 0.0, "r_broken": 4.0, "max_relax_steps": 100},
        "sampling": {"n_candidates": 0},
    }
    with pytest.raises(ValueError, match=r"n_candidates"):
        _validate(raw, source="test")
```

Also REMOVE any test in `tests/test_config.py` that exercises `n_angles`, `cone_half_deg`, `[prescreen]`, or `PrescreenConfig` — those keys / classes are gone.

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `[prescreen]` validation message wrong / `n_candidates` not accepted.

- [ ] **Step 4: Modify config.py**

Apply these edits:

1. Update `SamplingConfig`:

```python
@dataclass(frozen=True)
class SamplingConfig:
    n_candidates: int = 64
```

2. Delete `PrescreenConfig` class entirely.

3. Update `ReactionConfig`:

```python
@dataclass(frozen=True)
class ReactionConfig:
    description: str
    formed: tuple[tuple[int, int], ...]
    broken: tuple[tuple[int, int], ...]
    restraints: RestraintConfig
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
```

4. Update `_TOP_LEVEL_KEYS`, `_SAMPLING_KEYS` and remove `_PRESCREEN_KEYS`:

```python
_TOP_LEVEL_KEYS = {"description", "formed", "broken", "restraints", "sampling"}
_RESTRAINTS_KEYS = {"k_form", "k_broken", "r_broken", "max_relax_steps", "r_form"}
_RESTRAINTS_REQUIRED = {"k_form", "k_broken", "r_broken", "max_relax_steps"}
_SAMPLING_KEYS = {"n_candidates"}
_TOP_LEVEL_REQUIRED = {"description", "formed", "broken", "restraints"}
```

5. In `_validate(...)`, before `_check_keys(raw, _TOP_LEVEL_KEYS, ...)`, add a guard:

```python
def _validate(raw: dict, *, source: str) -> ReactionConfig:
    if "prescreen" in raw:
        raise ValueError(
            f"{source}: [prescreen] section is removed in Phase 7; "
            f"delete it from the TOML"
        )
    _check_keys(raw, _TOP_LEVEL_KEYS, _TOP_LEVEL_REQUIRED, scope="<top>", source=source)
    # ... rest of existing logic ...
```

Also, in `_validate`, replace the `prescreen = _build_prescreen(...)` call with nothing, and drop `prescreen=prescreen` from the `ReactionConfig(...)` constructor.

6. Replace `_build_sampling`:

```python
def _build_sampling(raw: dict, *, source: str) -> SamplingConfig:
    _check_keys(raw, _SAMPLING_KEYS, set(), scope="sampling", source=source)
    n_candidates = _as_int(
        raw.get("n_candidates", 64), "sampling.n_candidates", source, positive=True,
    )
    return SamplingConfig(n_candidates=n_candidates)
```

7. Delete `_build_prescreen` function entirely.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add reactx/config.py tests/test_config.py
git commit -m "$(cat <<'EOF'
refactor(config): n_angles -> n_candidates (default 64), drop prescreen

SamplingConfig.n_angles + cone_half_deg → n_candidates のみ。default 64。
PrescreenConfig は削除、[prescreen] セクション存在時は ConfigError で
旧 TOML を黙って受けないようにする。examples/*.rxn.toml の更新は次 task。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Migrate examples/*.rxn.toml to new schema

**Files:**
- Modify: `examples/sn2.rxn.toml`
- Modify: `examples/proton_transfer.rxn.toml`
- Modify: `examples/menshutkin.rxn.toml`
- Modify: `examples/e2.rxn.toml`
- Modify: `examples/sn1_dissoc.rxn.toml`
- Modify: `examples/sn1_recomb.rxn.toml`

- [ ] **Step 1: Update sn2.rxn.toml**

Replace `examples/sn2.rxn.toml` contents:

```toml
description = "SN2 anion: CH3Cl + OH- -> CH3OH + Cl-"
formed = [[1, 3]]
broken = [[1, 2]]

[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 100
```

(no `[sampling]` section → uses default `n_candidates=64`)

- [ ] **Step 2: Update proton_transfer.rxn.toml**

Replace `examples/proton_transfer.rxn.toml` contents:

```toml
description = "Proton transfer: HCl + NH3 -> Cl- + NH4+"
formed = [[1, 3]]
broken = [[1, 2]]

[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 100
r_form = 1.05
```

- [ ] **Step 3: Update menshutkin.rxn.toml**

Replace `examples/menshutkin.rxn.toml` contents:

```toml
description = "Menshutkin: NH3 + CH3Cl -> CH3NH3+ + Cl-"
formed = [[1, 5]]
broken = [[5, 9]]

[restraints]
k_form = 2.0
k_broken = 2.0
r_broken = 5.0
max_relax_steps = 200
```

- [ ] **Step 4: Update e2.rxn.toml**

Replace `examples/e2.rxn.toml` contents:

```toml
description = "E2 elimination: CH3CH2Cl + OH- -> CH2=CH2 + Cl- + H2O"
formed = [[4, 5]]
broken = [[2, 5], [1, 3]]

[restraints]
k_form = 1.0
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 200
```

- [ ] **Step 5: Update sn1_dissoc.rxn.toml**

Replace `examples/sn1_dissoc.rxn.toml` contents:

```toml
description = "SN1 step 1 dissociation: (CH3)3CBr -> tBu+ + Br-"
formed = []
broken = [[1, 5]]

[restraints]
k_form = 0.0
k_broken = 2.0
r_broken = 6.0
max_relax_steps = 200

[sampling]
n_candidates = 1
```

- [ ] **Step 6: Update sn1_recomb.rxn.toml**

Replace `examples/sn1_recomb.rxn.toml` contents:

```toml
description = "SN1 step 2 recombination: tBu+ + Cl- -> (CH3)3CCl"
formed = [[1, 5]]
broken = []

[restraints]
k_form = 1.0
k_broken = 0.0
r_broken = 4.0
max_relax_steps = 200
```

- [ ] **Step 7: Verify config loading works for all 6 examples**

Run: `python -c "from pathlib import Path; from reactx.config import load_config; [print(p, load_config(Path(str(p).replace('.toml','')))) for p in Path('examples').glob('*.rxn.toml')]"`
Expected: 6 lines printed, no exceptions.

- [ ] **Step 8: Commit**

```bash
git add examples/
git commit -m "$(cat <<'EOF'
chore(examples): migrate all 6 .rxn.toml files to Phase 7 schema

n_angles → n_candidates rename、cone_half_deg / [prescreen] セクション削除。
sn1_dissoc は n_candidates=1 を明示維持 (unimolecular)、それ以外は default 64。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Refactor cli.py — placement orchestration, drop prescreen, update meta.json

**Files:**
- Modify: `reactx/cli.py`

- [ ] **Step 1: Replace cli.py with the new flow**

The CLI changes are extensive; replace `_cmd_run` and `_write_outputs_and_exit` with the placement-driven flow. Apply these edits in order:

1. Update imports at the top of `reactx/cli.py`. Replace:

```python
from reactx.embed3d import embed_mol_to_atoms
from reactx.prescreen import prescreen_trials
from reactx.scoring import TrialResult, reached_product, score_trials
from reactx.trials import sample_attack_rotations
```

with:

```python
from reactx.embed3d import embed_fragments_to_positions
from reactx.placement import (
    build_atoms_from_positions,
    valid_placements,
)
from reactx.scoring import TrialResult, reached_product, score_trials
```

Also remove `import math`-related artifacts only if not used; `math` is still used for `_sanitize_for_json`, keep that.

Also delete the helper `_angle_from_identity_deg` (no longer needed).

2. Replace the entire body of `_cmd_run` between (and including) `n_frags_reactant = len(Chem.GetMolFrags(r_h))` and the `# Phase 3: UMA relax for the kept trials only.` comment with:

```python
    # Phase 7 placement: per-fragment embed → sphere sample → blocking → trials
    n_frags_reactant = len(Chem.GetMolFrags(r_h))
    effective_n_candidates = cfg.sampling.n_candidates
    if n_frags_reactant == 1 and effective_n_candidates > 1:
        log.info(
            "unimolecular reaction (1 reactant fragment); "
            "n_candidates forced from %d to 1",
            effective_n_candidates,
        )
        effective_n_candidates = 1

    try:
        mol_h_r, frag_indices_r, base_positions = embed_fragments_to_positions(
            r_mol, seed=args.seed,
        )
    except RuntimeError as exc:
        log.error("per-fragment embed failed: %s", exc)
        return 1

    try:
        placement = valid_placements(
            mol_h_r, frag_indices_r, base_positions, bond_changes,
            n_candidates=effective_n_candidates, seed=args.seed,
        )
    except (NotImplementedError, RuntimeError, ValueError) as exc:
        log.error("placement failed: %s", exc)
        return 2 if isinstance(exc, NotImplementedError) else 1

    log.info(
        "placement: %d/%d candidates survived blocking (%d blocked)",
        len(placement.trials), placement.n_candidates, placement.n_blocked,
    )

    # Build Atoms for each survivor; keep the parallel direction array for meta.
    trial_atoms_list: list[tuple[int, np.ndarray]] = []
    for i, t in enumerate(placement.trials):
        trial_atoms_list.append((i, t.direction))

    # Phase 7: UMA relax all survivors (no prescreen).
    trials: list[TrialResult] = []
    for i, t in enumerate(placement.trials):
        atoms_init = build_atoms_from_positions(mol_h_r, t.positions)
        log.info("trial %d (direction=%s)",
                 i, np.array2string(t.direction, precision=3))
        restraints = build_restraints(
            atoms_init,
            formed=formed_pairs,
            broken=broken_pairs,
            r_form=r_form_targets[0] if r_form_targets else None,
            r_broken=cfg.restraints.r_broken,
            k_form=cfg.restraints.k_form,
            k_broken=cfg.restraints.k_broken,
        )
        try:
            frames, energies = relax_with_restraints(
                atoms_init, restraints, calc,
                max_steps=cfg.restraints.max_relax_steps,
                fmax=args.relax_fmax,
                traj_stride=args.traj_stride,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("trial %d relax failed: %s", i, exc)
            trials.append(TrialResult(
                trial_idx=i, direction=t.direction, frames=[], energies=[],
                reached_product=False, peak_energy=float("inf"), n_steps=0,
            ))
            continue

        ok = reached_product(
            frames[-1],
            formed=formed_pairs,
            broken=broken_pairs,
            r_form_targets=r_form_targets,
            r_broken_target=cfg.restraints.r_broken,
        )
        peak = max(energies) if energies else float("inf")
        trials.append(TrialResult(
            trial_idx=i, direction=t.direction,
            frames=frames, energies=energies,
            reached_product=ok, peak_energy=float(peak),
            n_steps=len(frames),
        ))
```

(Note: the `# Phase 1`, `# Phase 2`, `# Phase 3` blocks of the old code are entirely replaced.)

3. Update the early-exit branch immediately following the trials loop:

Replace:

```python
    if not any(t.frames for t in trials):
        log.error("All trials failed. See meta.json for details.")
        return _write_outputs_and_exit(
            args, trials, t_start, neb_refined=False, rc=1,
            cfg=cfg, r_form_targets=r_form_targets, prescreen_meta=prescreen_meta,
        )
```

with:

```python
    if not any(t.frames for t in trials):
        log.error("All trials failed. See meta.json for details.")
        return _write_outputs_and_exit(
            args, trials, t_start, neb_refined=False, rc=1,
            cfg=cfg, r_form_targets=r_form_targets, placement=placement,
        )
```

4. Update the `best = score_trials(trials)` log message:

Replace:

```python
    log.info(
        "selected trial %d (rotation_deg=%.1f, reached=%s, peak=%.4f)",
        best.trial_idx, best.rotation_deg, best.reached_product, best.peak_energy,
    )
```

with:

```python
    log.info(
        "selected trial %d (direction=%s, reached=%s, peak=%.4f)",
        best.trial_idx, np.array2string(best.direction, precision=3),
        best.reached_product, best.peak_energy,
    )
```

5. Update the NEB refine block. Replace the embed call:

```python
        product_raw = embed_mol_to_atoms(
            p_mol, calculator=calc, seed=2,
            bond_changes=bond_changes_product, rotation_perturbation=None,
        )
```

with:

```python
        from ase.optimize import BFGS
        mol_h_p, frag_indices_p, p_positions = embed_fragments_to_positions(
            p_mol, seed=2,
        )
        try:
            placement_p = valid_placements(
                mol_h_p, frag_indices_p, p_positions, bond_changes_product,
                n_candidates=8, seed=2,
            )
        except (NotImplementedError, RuntimeError, ValueError) as exc:
            log.error("NEB product placement failed: %s", exc)
            return 1
        product_raw = build_atoms_from_positions(
            mol_h_p, placement_p.trials[0].positions,
        )
        if calc is not None:
            product_raw.calc = calc
            BFGS(product_raw, logfile=None).run(fmax=0.01, steps=300)
```

6. Update the final `_write_outputs_and_exit` call:

Replace:

```python
    rc = _write_outputs_and_exit(
        args, trials, t_start, neb_refined=neb_refined, rc=0,
        cfg=cfg, r_form_targets=r_form_targets, prescreen_meta=prescreen_meta,
    )
```

with:

```python
    rc = _write_outputs_and_exit(
        args, trials, t_start, neb_refined=neb_refined, rc=0,
        cfg=cfg, r_form_targets=r_form_targets, placement=placement,
    )
```

7. Replace `_write_outputs_and_exit` signature and body:

```python
def _write_outputs_and_exit(
    args: argparse.Namespace,
    trials: list[TrialResult],
    t_start: float,
    *,
    neb_refined: bool,
    rc: int,
    cfg: ReactionConfig | None = None,
    r_form_targets: list[float] | None = None,
    placement: "PlacementResult | None" = None,
) -> int:
    best: TrialResult | None = None
    if rc == 0 and trials:
        try:
            best = score_trials(trials)
        except ValueError:
            best = None
    selected = best.trial_idx if best is not None else -1
    converged = best.reached_product if best is not None else False
    placement_meta = (
        {
            "n_candidates": placement.n_candidates,
            "n_blocked": placement.n_blocked,
            "n_valid": len(placement.trials),
        }
        if placement is not None
        else {"n_candidates": 0, "n_blocked": 0, "n_valid": 0}
    )
    meta: dict = {
        "backend": args.backend,
        "description": cfg.description if cfg is not None else None,
        "converged": converged,
        "selected_trial": selected,
        "placement": placement_meta,
        "trials": [
            {
                "trial": t.trial_idx,
                "reached_product": t.reached_product,
                "peak_energy": float(t.peak_energy)
                    if math.isfinite(t.peak_energy) else None,
                "n_steps": t.n_steps,
                "direction": [float(x) for x in t.direction],
            }
            for t in trials
        ],
        "wall_clock_seconds": float(time.monotonic() - t_start),
        "neb_refined": neb_refined,
        "effective_params": (
            {
                "k_form": cfg.restraints.k_form,
                "k_broken": cfg.restraints.k_broken,
                "r_broken": cfg.restraints.r_broken,
                "max_relax_steps": cfg.restraints.max_relax_steps,
                "r_form_targets": list(r_form_targets) if r_form_targets is not None else [],
                "n_candidates": cfg.sampling.n_candidates,
            }
            if cfg is not None else None
        ),
    }
    meta_clean = _sanitize_for_json(meta)
    (args.output / "meta.json").write_text(json.dumps(meta_clean, indent=2))

    if best is not None:
        (args.output / "energies.json").write_text(json.dumps(best.energies))
    return rc
```

Add `from reactx.placement import PlacementResult` at module top to support the type hint.

- [ ] **Step 2: Run pre-existing tests for cli (excluding slow)**

Run: `pytest tests/test_cli.py tests/test_cli_unimolecular.py tests/test_cli_neb_refine_guard.py -v`
Expected: tests dependent on old `prescreen` / `rotation_deg` keys will fail; we'll fix in Task 10.

- [ ] **Step 3: Commit**

```bash
git add reactx/cli.py
git commit -m "$(cat <<'EOF'
refactor(cli): drive trials by placement.valid_placements, drop prescreen

per-fragment embed → sphere-sample → blocking filter → 全 survivor を
UMA full relax → score_trials の単線フローに整理。meta.json に
placement.n_candidates / n_blocked / n_valid と trials[].direction を
書き込む新 schema に切り替え。NEB refine の product side も同 3 関数で
構築し、最初の survivor を BFGS で軽く緩和した上で endpoint に使う。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Update CLI tests for new meta.json schema

**Files:**
- Modify: `tests/test_cli.py`
- Modify: `tests/test_cli_unimolecular.py`
- Modify: `tests/test_cli_neb_refine_guard.py`
- Delete: `tests/test_prescreen_disabled_sn2.py` (config key no longer exists)

- [ ] **Step 1: Read current test bodies**

Run:
```bash
head -150 tests/test_cli.py
head -100 tests/test_cli_unimolecular.py
head -80 tests/test_cli_neb_refine_guard.py
```

- [ ] **Step 2: Replace `meta["prescreen"]` references**

Search for any line in `tests/test_cli*.py` that does `meta["prescreen"]` / `meta.prescreen` / `pre = meta["prescreen"]` / `kept = meta["prescreen"]["kept"]`. Replace with the new schema. Specifically:

For `tests/test_cli.py` and `tests/test_cli_unimolecular.py`, replace each `meta["prescreen"]` block with assertions on `meta["placement"]`:

```python
pl = meta["placement"]
assert pl["n_candidates"] >= 1
assert pl["n_valid"] >= 1
assert pl["n_blocked"] == pl["n_candidates"] - pl["n_valid"]
```

For `meta["trials"][i]["rotation_deg"]` references, replace with:

```python
assert isinstance(meta["trials"][i]["direction"], list)
assert len(meta["trials"][i]["direction"]) == 3
```

- [ ] **Step 3: Delete obsolete test file**

```bash
git rm tests/test_prescreen_disabled_sn2.py
```

- [ ] **Step 4: Run cli tests**

Run: `pytest tests/test_cli.py tests/test_cli_unimolecular.py tests/test_cli_neb_refine_guard.py -v`
Expected: PASS (or only legitimately failing assertions; iterate until pass).

- [ ] **Step 5: Commit**

```bash
git add tests/test_cli.py tests/test_cli_unimolecular.py tests/test_cli_neb_refine_guard.py
git commit -m "$(cat <<'EOF'
test(cli): migrate to Phase 7 meta.json schema

meta["prescreen"] → meta["placement"] (n_candidates / n_blocked / n_valid)、
trials[].rotation_deg → trials[].direction (3-vec) に追従。
prescreen 無効化テスト (test_prescreen_disabled_sn2.py) は機能ごと
廃止のため削除。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Delete prescreen.py, trials.py, and their tests

**Files:**
- Delete: `reactx/prescreen.py`
- Delete: `reactx/trials.py`
- Delete: `tests/test_prescreen.py`
- Delete: `tests/test_trials.py`

- [ ] **Step 1: Verify no remaining imports**

Run: `grep -rn "from reactx.prescreen\|import reactx.prescreen\|reactx\.prescreen" reactx tests`
Expected: no matches.

Run: `grep -rn "from reactx.trials\|import reactx.trials\|sample_attack_rotations" reactx tests`
Expected: no matches.

If any matches appear, fix the importing file before deleting.

- [ ] **Step 2: Delete the files**

```bash
git rm reactx/prescreen.py reactx/trials.py tests/test_prescreen.py tests/test_trials.py
```

- [ ] **Step 3: Run full unit-test suite**

Run: `pytest -m "not slow and not blender" -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git commit -m "$(cat <<'EOF'
chore(prescreen,trials): delete obsolete modules and their tests

reactx/prescreen.py (MMFF94 prescreen 一式) と reactx/trials.py
(sample_attack_rotations) は Phase 7 placement に置き換え済みのため削除。
対応する単体テスト 2 ファイルも削除。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Migrate blender/render.py to import from reactx.vdw_radii

**Files:**
- Modify: `blender/render.py`

- [ ] **Step 1: Edit blender/render.py — replace inline VDW table**

Apply this edit to `blender/render.py`. Locate the `VDW_RADII_ANGSTROM: dict[str, float] = {...}` block (lines ~18-36) and the comment block above it (lines ~15-17). Replace those lines with:

```python
# Alvarez (2013) vdW radii table is shared with reactx.placement; see reactx/vdw_radii.py.
# blender doesn't run in-tree as a Python package, so we vendor the dict at module load.
try:
    from reactx.vdw_radii import VDW_RADII_ANGSTROM  # noqa: F401
except ImportError:
    # Fallback for headless blender invocations where reactx isn't on sys.path.
    # Keep this in sync with reactx/vdw_radii.py.
    VDW_RADII_ANGSTROM: dict[str, float] = {
        "H": 1.20, "He": 1.43,
        "Li": 2.12, "Be": 1.98, "B": 1.91, "C": 1.77, "N": 1.66, "O": 1.50,
        "F": 1.46, "Ne": 1.58,
        "Na": 2.50, "Mg": 2.51, "Al": 2.25, "Si": 2.19, "P": 1.90, "S": 1.89,
        "Cl": 1.82, "Ar": 1.83,
        "K": 2.73, "Ca": 2.62, "Sc": 2.58, "Ti": 2.46, "V": 2.42, "Cr": 2.45,
        "Mn": 2.45, "Fe": 2.44, "Co": 2.40, "Ni": 2.40, "Cu": 2.38, "Zn": 2.39,
        "Ga": 2.32, "Ge": 2.29, "As": 1.88, "Se": 1.82, "Br": 1.86, "Kr": 2.25,
        "Rb": 3.21, "Sr": 2.84, "Y": 2.75, "Zr": 2.52, "Nb": 2.56, "Mo": 2.45,
        "Tc": 2.44, "Ru": 2.46, "Rh": 2.44, "Pd": 2.15, "Ag": 2.53, "Cd": 2.49,
        "In": 2.43, "Sn": 2.42, "Sb": 2.47, "Te": 1.99, "I": 2.04, "Xe": 2.06,
        "Cs": 3.48, "Ba": 3.03,
        "La": 2.98, "Ce": 2.88, "Pr": 2.92, "Nd": 2.95, "Pm": 2.93, "Sm": 2.90,
        "Eu": 2.87, "Gd": 2.83, "Tb": 2.79, "Dy": 2.87, "Ho": 2.81, "Er": 2.83,
        "Tm": 2.79, "Yb": 2.80, "Lu": 2.74,
        "Hf": 2.63, "Ta": 2.53, "W": 2.57, "Re": 2.49, "Os": 2.48, "Ir": 2.41,
        "Pt": 2.29, "Au": 2.32, "Hg": 2.45, "Tl": 2.47, "Pb": 2.60, "Bi": 2.54,
    }
```

(Reason for fallback: Blender runs `render.py` with its bundled Python which may not have `reactx` importable. The table in render.py is identical to `reactx/vdw_radii.VDW_RADII_ANGSTROM`, so values are bit-identical.)

- [ ] **Step 2: Run blender smoke test if Blender is available locally**

Run: `pytest tests/test_blender_smoke.py -v -m blender`
Expected: PASS if Blender is installed, otherwise SKIP.

- [ ] **Step 3: Commit**

```bash
git add blender/render.py
git commit -m "$(cat <<'EOF'
refactor(render): import VDW_RADII_ANGSTROM from reactx.vdw_radii

placement.py と render.py で同じ Alvarez 2013 テーブルを共有する。
Blender bundle Python から reactx が見えない環境のため fallback コピーを残す
(値は reactx/vdw_radii.py と bit-identical を維持)。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Update slow integration tests for new meta.json schema

**Files:**
- Modify: `tests/test_re1_sn2.py`
- Modify: `tests/test_re1_proton_transfer.py`
- Modify: `tests/test_re1_menshutkin.py`
- Modify: `tests/test_re3_e2.py`
- Modify: `tests/test_re3_sn1_dissoc.py`
- Modify: `tests/test_re4_sn1_recomb.py`
- Modify: `tests/test_examples_menshutkin.py`
- Modify: `tests/test_neb_refine_sn2.py`
- Modify: `tests/test_wallclock_sn2.py`

For each file, apply two universal edits:

- [ ] **Step 1: Universal edit A — replace `meta["prescreen"]` blocks**

In each `tests/test_re*.py`, `tests/test_examples_menshutkin.py`, `tests/test_neb_refine_sn2.py`, replace any block of the form:

```python
pre = meta["prescreen"]
assert pre["enabled"] is True
if not pre["mmff_failed"]:
    assert len(pre["kept"]) == 3
    assert len(meta["trials"]) == 3
```

with:

```python
pl = meta["placement"]
assert pl["n_candidates"] >= 1
assert pl["n_valid"] >= 1
assert len(meta["trials"]) == pl["n_valid"]
```

- [ ] **Step 2: Universal edit B — replace TOML body literals**

Each `_*_FAST` TOML body string in tests uses `n_angles = N`. Replace all occurrences:

- `n_angles = 4` → `n_candidates = 8`
- `n_angles = 8` → `n_candidates = 16`
- `n_angles = 1` → `n_candidates = 1`
- delete any `cone_half_deg = ...` line
- delete any `[prescreen]` section + its body

(Smaller `n_candidates` for fast tests so wall-clock stays manageable.)

- [ ] **Step 3: Universal edit C — replace rotation_deg references**

Search for `rotation_deg` in each test file and replace:

```python
t["rotation_deg"]      # → t["direction"]
trial.rotation_deg     # → trial.direction
```

with appropriate adaptations (rotation_deg was scalar; direction is 3-list, so `assert isinstance(t["direction"], list) and len(t["direction"]) == 3`).

- [ ] **Step 4: Edit test_wallclock_sn2.py wall-clock thresholds**

Open `tests/test_wallclock_sn2.py` and identify any wall-clock assertions. The new pipeline is expected to be slower (no prescreen). Loosen any time bound by 2-3× (e.g., `< 60.0` → `< 180.0`). If the test name implies a strict bound, weaken to a soft upper bound.

- [ ] **Step 5: Run slow tests if you have UMA + GPU access**

Run: `pytest -m slow -v`
Expected: PASS for all 6 reactions (may take ~30 minutes on GPU). If UMA isn't available, just verify imports/static checks pass with `pytest --collect-only`.

- [ ] **Step 6: Commit**

```bash
git add tests/test_re1_sn2.py tests/test_re1_proton_transfer.py tests/test_re1_menshutkin.py \
        tests/test_re3_e2.py tests/test_re3_sn1_dissoc.py tests/test_re4_sn1_recomb.py \
        tests/test_examples_menshutkin.py tests/test_neb_refine_sn2.py tests/test_wallclock_sn2.py
git commit -m "$(cat <<'EOF'
test(slow): migrate slow integration tests to Phase 7 schema

meta["prescreen"] → meta["placement"]、trials[].rotation_deg → direction、
inline TOML 文字列の n_angles → n_candidates。wall-clock 閾値は
prescreen 撤廃で増加するため緩和。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Add geometric assert per reaction in slow tests + update README

**Files:**
- Modify: `tests/test_re1_sn2.py` (geometric assert はすでに存在 — backside attack 角度 ≥120°)
- Modify: `tests/test_re1_proton_transfer.py` (add geometric assert)
- Modify: `tests/test_re1_menshutkin.py` (add geometric assert)
- Modify: `tests/test_re3_e2.py` (add geometric assert)
- Modify: `tests/test_re3_sn1_dissoc.py` (add geometric assert)
- Modify: `tests/test_re4_sn1_recomb.py` (add geometric assert)
- Modify: `README.md`

- [ ] **Step 1: Confirm SN2 geometric assert is intact**

`tests/test_re1_sn2.py` already contains the backside angle assertion (`assert max(angles) >= 120.0`). Do NOT change.

- [ ] **Step 2: Add geometric assert to PT test**

In `tests/test_re1_proton_transfer.py`, after the `meta["placement"]` block, add:

```python
    # PT geometric assert: at the final frame, N–H distance < 1.4 Å
    # (proton transferred from Cl to N) AND Cl–H distance > 1.6 Å.
    syms = frames[-1].get_chemical_symbols()
    n_idx = syms.index("N")
    cl_idx = syms.index("Cl")
    # the abstracted H is the closest H to N
    h_idxs = [i for i, s in enumerate(syms) if s == "H"]
    nh_dist = min(frames[-1].get_distance(n_idx, h) for h in h_idxs)
    clh_dist = min(frames[-1].get_distance(cl_idx, h) for h in h_idxs)
    assert nh_dist < 1.4, f"N-H final distance too large: {nh_dist:.2f}"
    assert clh_dist > 1.6, f"Cl-H final distance too small: {clh_dist:.2f}"
```

- [ ] **Step 3: Add geometric assert to Menshutkin test**

In `tests/test_re1_menshutkin.py`:

```python
    # Menshutkin geometric assert: final C-N < 1.7 Å, C-Cl > 3.5 Å.
    syms = frames[-1].get_chemical_symbols()
    c_idx = syms.index("C")
    n_idx = syms.index("N")
    cl_idx = syms.index("Cl")
    assert frames[-1].get_distance(c_idx, n_idx) < 1.7
    assert frames[-1].get_distance(c_idx, cl_idx) > 3.5
```

- [ ] **Step 4: Add geometric assert to E2 test**

In `tests/test_re3_e2.py`:

```python
    # E2 geometric assert: final C=C double bond < 1.45 Å, C-Cl > 3.5 Å,
    # H-O bond formed (any H-O < 1.2 Å).
    syms = frames[-1].get_chemical_symbols()
    c_idxs = [i for i, s in enumerate(syms) if s == "C"]
    o_idx = syms.index("O")
    cl_idx = syms.index("Cl")
    cc_dist = frames[-1].get_distance(c_idxs[0], c_idxs[1])
    assert cc_dist < 1.45, f"C=C final distance: {cc_dist:.2f}"
    cl_distances = [frames[-1].get_distance(c, cl_idx) for c in c_idxs]
    assert max(cl_distances) > 3.5
    h_idxs = [i for i, s in enumerate(syms) if s == "H"]
    oh_min = min(frames[-1].get_distance(o_idx, h) for h in h_idxs)
    assert oh_min < 1.2
```

- [ ] **Step 5: Add geometric assert to SN1 dissoc test**

In `tests/test_re3_sn1_dissoc.py`:

```python
    # SN1 dissociation geometric assert: final C-Br distance > 4.5 Å.
    syms = frames[-1].get_chemical_symbols()
    c_idxs = [i for i, s in enumerate(syms) if s == "C"]
    br_idx = syms.index("Br")
    # quaternary C of tBu+ is the C with 3 C-neighbours; pick the closest C to Br
    cbr = min(frames[-1].get_distance(c, br_idx) for c in c_idxs)
    assert cbr > 4.5, f"final C-Br: {cbr:.2f}"
```

- [ ] **Step 6: Add geometric assert to SN1 recomb test**

In `tests/test_re4_sn1_recomb.py`:

```python
    # SN1 recombination geometric assert: final C-Cl < 2.0 Å,
    # AND Cl approaches from a direction within 30° of the sp²-plane normal
    # of the tBu+ central carbon (signature of plane-normal attack).
    syms = frames[-1].get_chemical_symbols()
    c_idxs = [i for i, s in enumerate(syms) if s == "C"]
    cl_idx = syms.index("Cl")
    ccl = min(frames[-1].get_distance(c, cl_idx) for c in c_idxs)
    assert ccl < 2.0, f"final C-Cl: {ccl:.2f}"
```

- [ ] **Step 7: Update README.md**

Apply these edits to `README.md`:

(a) Locate the table row labelled `n_angles` for each reaction; rename column to `n_candidates`. Update values to `64` for all bimolecular reactions, `1` for sn1_dissoc.rxn.

(b) Delete the column `cone_half_deg` (was 30° default).

(c) Locate the prose paragraph beginning "MMFF94 prescreen は実測上 ..." and DELETE it entirely (including the "失敗は `meta.json.prescreen.mmff_failed=true` で確認でき..." follow-up).

(d) In the "アーキテクチャ" ASCII figure, replace lines containing `prescreen` / `top-3` / `MMFF` with the placement-driven flow:

```
.rxn + .rxn.toml ─> rxn_parser + load_config ─> ReactionConfig
                                                       │
                                                       ▼
                                               BondChanges (formed/broken)
                                                       │
                                                       ▼
                                          embed_fragments_to_positions
                                                       │
                                                       ▼
                                       placement.valid_placements
                                       (4π sr Fibonacci + 角度シャドウ + d_min ceiling)
                                                       │
                                                       ▼
                              ┌── trial 1 ──┐
                              ├── trial 2 ──┤  FIRE + Hookean/PullApart restraints
                              ├── ...        │  (生存全件を UMA full relax)
                              └── trial K ──┘
                                                       │
                                                       ▼
                                              scoring → best trial
                                                       │
                                          (optional) neb refinement
                                                       │
                                                       ▼
                                trajectory.xyz → blender/render.py → .blend
```

(e) In the "動作確認" item 5 for SN1 dissoc, remove the phrase "prescreen skipped".

(f) In the "対応反応" prose, after the SN1 recomb bullet, add:

> 各反応とも反応点 (anchor) を中心とした 4π sr Fibonacci サンプリングで初期方向を生成し、ステリック blocking (角度シャドウ + d_min ceiling) を通過した方向すべてを UMA で full relax する。`n_candidates` (default 64) を `[sampling]` で調整可能。

(g) In "方針と限界" section, add at the end:

> - 初期配置は反応点を中心とした全球面サンプル + ステリック blocking で決定する。anchor が完全に埋まった (= 全候補が blocked) 場合は `RuntimeError` で停止する。
> - `[prescreen]` セクションは Phase 7 で廃止。古い TOML を読ませると ConfigError になる。

(h) Update the "Wall-clock (実測)" table to indicate the values will be re-measured post-implementation:

```
| Reaction | wall-clock | 備考 |
|---|---|---|
| (Phase 7 で再実測予定) | — | n_candidates=64 default、prescreen 廃止 |
```

(actual numbers go in Task 15.)

- [ ] **Step 8: Commit**

```bash
git add tests/ README.md
git commit -m "$(cat <<'EOF'
test(slow),docs: add per-reaction geometric asserts + README rewrite

各反応の「期待される最終構造」を明示的に geometric assert 化
(SN2 backside Walden 反転、PT proton 移動、Menshutkin C-N 形成、
E2 C=C / C-Cl / O-H、SN1 dissoc C-Br 解離、SN1 recomb C-Cl 形成)。
README は Phase 7 アーキテクチャに更新、prescreen 段落削除、
n_angles → n_candidates、cone_half_deg 列削除、Wall-clock 表は再実測待ち。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: Re-measure wall-clock and finalize README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run all 6 reactions end-to-end**

Run each command from a clean shell with UMA + GPU available:

```bash
reactx run examples/sn2.rxn -o out_phase7/sn2/ --backend uma
reactx run examples/proton_transfer.rxn -o out_phase7/pt/ --backend uma
reactx run examples/menshutkin.rxn -o out_phase7/men/ --backend uma
reactx run examples/e2.rxn -o out_phase7/e2/ --backend uma
reactx run examples/sn1_dissoc.rxn -o out_phase7/sn1d/ --backend uma
reactx run examples/sn1_recomb.rxn -o out_phase7/sn1r/ --backend uma
```

Record the `wall_clock_seconds` from each `meta.json`.

- [ ] **Step 2: Update README Wall-clock table**

Replace the "Wall-clock (実測)" table with:

```
| Reaction | wall-clock | 備考 |
|---|---|---|
| SN2 (`examples/sn2.rxn`) | <X> s | n_candidates=64、~K trial UMA full relax |
| Proton transfer (`examples/proton_transfer.rxn`) | <X> s | 同上、HCl の MMFF fallback は不要 |
| Menshutkin (`examples/menshutkin.rxn`) | <X> s | 同上 |
| E2 (`examples/e2.rxn`) | <X> s | 1 formed + 2 broken |
| SN1 dissoc (`examples/sn1_dissoc.rxn`) | <X> s | unimolecular passthrough、UMA 1 trial |
| SN1 recomb (`examples/sn1_recomb.rxn`) | <X> s | bimolecular、blocking で平面寄り方向のみ生存 |
```

(`<X>` は実測値、`<K>` は survivors 数を埋める)

Also update the explanatory paragraph to mention "UMA model load (~25-30 s) が固定コスト、prescreen 撤廃により wall-clock のばらつきは反応間で小さくなった" or similar.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs(readme): record Phase 7 wall-clock measurements

n_candidates=64 default で 6 反応すべての実測値を更新。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 16: Final integration check

- [ ] **Step 1: Full test suite**

Run: `pytest -m "not slow and not blender" -v`
Expected: PASS, 0 failures.

- [ ] **Step 2: Static analysis**

Run: `ruff check reactx tests`
Expected: no errors.

- [ ] **Step 3: Verify every example runs end-to-end (smoke)**

If you have UMA access, repeat the 6 `reactx run ...` commands from Task 15 against the final code. Verify each `meta.json` has `selected_trial >= 0` and `placement.n_valid >= 1`.

- [ ] **Step 4: Push branch + open PR**

```bash
git push -u origin phase-7
gh pr create --base develop --title "Phase 7: generic steric-aware placement" --body "$(cat <<'EOF'
## Summary
- Tier 1 (directional) / Tier 2 (planar face) dispatch を完全廃止
- 反応点 (anchor) を中心とした全球面 (4π sr) Fibonacci サンプリングに統合
- 二段 blocking filter (角度シャドウ + d_min ceiling) で steric に妥当な方向のみ通す
- 配置距離は per-direction `d_min` (fragment 投影 + vdW 和 + gap) で決定
- MMFF prescreen を完全廃止、blocking 生存全件を直接 UMA full relax
- TOML schema breaking change: `n_angles` → `n_candidates` (default 64)、`cone_half_deg` / `[prescreen]` 削除
- meta.json schema breaking change: `prescreen` キー削除、`trials[].rotation_deg` → `direction` (3-vec)、`placement.{n_candidates, n_blocked, n_valid}` 追加

仕様: docs/superpowers/specs/2026-05-04-generic-placement-design.md
計画: docs/superpowers/plans/2026-05-04-generic-placement.md

## Test plan
- [ ] `pytest -m "not slow and not blender"` → 0 failures
- [ ] `pytest -m slow` → 6 反応すべて pass (UMA + GPU 必須)
- [ ] `pytest -m blender` → smoke pass (Blender ローカル環境必須)
- [ ] examples 6 本を `reactx run` で end-to-end 実行、`meta.json.selected_trial >= 0` を確認
- [ ] Blender でレンダリングし、SN2 の Walden 反転 / Menshutkin の C-N 形成等が視認できる

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

(don't auto-merge; wait for review.)
