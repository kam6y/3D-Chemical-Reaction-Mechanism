# Phase 9 — Per-pair AFIR with Sticky Latch Implementation Plan (v3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 経験的力場 `Hookean + PullApart` を **per-pair AFIR with sticky latch** に置き換え、`alpha_formed` / `alpha_broken` の 2 種ハイパラ + per-pair list 一本に簡素化、threshold を AFIR latch と scoring で共有する。8 反応すべて新方式で動かす。

**Architecture:** AFIRConstraint を `ase.constraints.FixConstraint` 継承で実装。各 pair が自分の threshold を初めて越えた時点で latch ON、以降そのペアの AFIR force = 0 (sticky)。threshold は `[scoring].r_*_threshold` から AFIRConstraint と `reached_product` 両方に渡す。`relax_with_restraints` は 3-tuple `(frames, energies, final_constraint_state)` を返す。`reached_product` は最終 frame 判定 (latch state とは独立、唯一の成功判定)、`product_distance_residual` を least-bad fallback で利用。

**Tech Stack:** Python 3.13, RDKit, ASE 3.x (Atoms / FixConstraint / FIRE), NumPy 2.x, fairchem UMA, pytest (`-m slow`), tomllib.

**Spec:** `docs/superpowers/specs/2026-05-08-afir-force-design.md` (v3.1)

**Branch:** `phase-9` (develop から fork、PR で develop に merge)

**Plan v2 → v3 改訂:** v2 を codex でレビューしたところ critical 5 件が指摘された (詳細 §改訂履歴):
- C1/C12: Stage 2 が 1 task で 19 step、subagent には過大、失敗時の rollback 単位も不明
- C5: resolved ft/bt が meta に残らない、Step P の `formed_thresholds`/`broken_thresholds` assertion と矛盾
- C9: `_AFIR_REQUIRED` で `alpha_*` 必須化のまま、spec §4.4 の空 pair 省略可と矛盾
- C11: spec §6.4 の safety assertion を Phase 10 送りにしたのは spec 違反

v3 では:
- Stage 2 を **2A (additive) / 2B (switch) / 2C (cleanup)** の 3 分割。各 sub-stage 終了時に repo green、subagent dispatch 1 回ごとの粒度に縮小
- `TrialResult` に `formed_thresholds: list[float]` / `broken_thresholds: list[float]` 追加 (合計 14 fields)、meta writer が resolved 値を書ける
- `[afir]` の `alpha_formed` / `alpha_broken` を任意化 (default 0.0)、`_AFIR_REQUIRED = {"max_relax_steps"}` のみ必須に
- Stage 3 各 task に min 非結合距離 ≥ 0.5 Å assertion + reactive pair 過短結合 check (helper を `tests/conftest.py` に)

**実行環境注記:**
- 本 plan のコマンドは Bash 互換 (subagent は Bash tool を使用)。Windows PowerShell 環境では Bash tool 経由で実行すること。

**File Structure (新規 + 変更):**

| Path | 種類 | 役割 |
|---|---|---|
| `reactx/covalent_radii.py` | 新規 | Cordero (2008) 共有半径表 |
| `reactx/artificial_force.py` | 全面書き換え | `AFIRConstraint` (per-pair + sticky latch) + `build_afir_constraint`。Hookean/PullApart/DEFAULT_R_FORM/lookup_r_form/build_restraints は Stage 2C で削除 |
| `reactx/path_relax.py` | 変更 | 戻り値 3-tuple、`_snapshot()` constraint clear |
| `reactx/config.py` | 全面書き換え | `[restraints]` 削除、`AFIRSection` + `ScoringSection` + `ConfigError` 新設、resolver helper |
| `reactx/scoring.py` | 全面書き換え | `reached_product` per-pair、`product_distance_residual`、`count_initial_latched`、`TrialResult` 14 fields、`score_trials` residual fallback |
| `reactx/cli.py` | 変更 | `build_afir_constraint` 配線、threshold 解決、3-tuple 受信、meta.json schema、ConfigError catch、失敗 trial 14-field |
| `blender/render.py` | 変更 | Cordero 表を import (fallback 維持) |
| `tests/test_covalent_radii.py` | 新規 | Cordero 表 lookup |
| `tests/test_afir_constraint.py` | 新規 | per-pair force, sticky latch, V=±α·r 有限差分, validation, **α=0 重複防御**, **latch≠reached regression** |
| `tests/test_artificial_force.py` | 全面書き換え | smoke test のみ |
| `tests/test_config_afir.py` | 新規 | 新 schema 検証、旧 schema reject、α=0 reject、NaN/inf reject、`alpha_*` 省略 (空 pair) ケース |
| `tests/test_config.py` | 全面書き換え | 8 examples の load smoke |
| `tests/test_scoring.py` | 全面書き換え | 新 schema 全カバー |
| `tests/test_path_relax.py` | 変更 | 3-tuple、`_snapshot` constraint clear |
| `tests/test_cli.py` | 変更 | TrialResult 14-field + meta 新フィールド + 新 schema toml_body |
| `tests/test_cli_unimolecular.py` | 変更 | 新 schema toml_body |
| `tests/test_cli_neb_refine_guard.py` | 変更 | 新 schema toml_body × 3 |
| `tests/test_neb_refine_sn2.py` | 変更 | 新 schema toml_body |
| `tests/test_wallclock_sn2.py` | 変更 | 新 schema toml_body, meta assertion |
| `tests/test_diels_alder_endo.py` | 変更 | TrialResult 14-field, meta assertion |
| `tests/test_re*.py` (5 files) | 変更 | meta assertion、reached_product 主導 |
| `tests/test_examples_menshutkin.py` | 変更 | meta assertion |
| `tests/conftest.py` | 変更 | `tmp_rxn_with_toml` docstring を新 schema に + safety assertion helper 追加 |
| `examples/*.rxn.toml` × 8 | 変更 | 新 schema |
| `tests/test_pull_apart.py` | 削除 (もし存在) | PullApart 消滅 |
| `tests/test_blender_smoke.py` | 変更 (もし `[restraints]` 含めば) | 新 schema toml_body |
| `README.md` | 変更 | Phase 9 セクション、新 schema 表、wall-clock 再測定 |

---

## Stage 1: Additive foundation (各 task は独立に commit 可能、repo は常に green)

### Task 1.1: Cordero (2008) 共有半径モジュール

**Files:** Create `reactx/covalent_radii.py` and `tests/test_covalent_radii.py`

- [ ] **Step 1: Write the failing test** (`tests/test_covalent_radii.py`):

```python
import numpy as np
import pytest
from ase import Atoms

from reactx.covalent_radii import (
    CORDERO_2008,
    cordero_radii_for_atoms,
    cordero_radius,
)


def test_cordero_radius_known_elements():
    assert cordero_radius("H") == 0.31
    assert cordero_radius("C") == 0.76
    assert cordero_radius("Cl") == 1.02
    assert cordero_radius("Br") == 1.20


def test_cordero_radius_unknown_returns_default():
    assert cordero_radius("Uuq", default=1.5) == 1.5
    assert cordero_radius("Po") == 1.5


def test_cordero_table_covers_z1_to_z83():
    expected = {"H", "He", "C", "N", "O", "F", "Cl", "Br", "Bi"}
    assert expected.issubset(CORDERO_2008.keys())
    assert "Po" not in CORDERO_2008
    assert "Fr" not in CORDERO_2008


def test_cordero_radii_for_atoms():
    a = Atoms("CHCl", positions=np.zeros((3, 3)))
    radii = cordero_radii_for_atoms(a)
    np.testing.assert_allclose(radii, [0.76, 0.31, 1.02])
```

- [ ] **Step 2: Run test (FAIL)** — `pytest tests/test_covalent_radii.py -v`

- [ ] **Step 3: Implement `reactx/covalent_radii.py`** (full Z=1..83 table — see spec §4.3 for canonical entries; also extracted from `blender/render.py`'s `COVALENT_RADII_ANGSTROM`):

```python
"""Cordero (2008) covalent radii in Å. Z=1..83 (OMol25 / UMA training range).

Reference: Cordero et al., Dalton Trans. 2008, 2832.
"""
from __future__ import annotations

import numpy as np
from ase.atoms import Atoms

CORDERO_2008: dict[str, float] = {
    "H": 0.31, "He": 0.28,
    "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66,
    "F": 0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05,
    "Cl": 1.02, "Ar": 1.06,
    "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39,
    "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
    "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16,
    "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54,
    "Tc": 1.47, "Ru": 1.46, "Rh": 1.42, "Pd": 1.39, "Ag": 1.45, "Cd": 1.44,
    "In": 1.42, "Sn": 1.39, "Sb": 1.39, "Te": 1.38, "I": 1.39, "Xe": 1.40,
    "Cs": 2.44, "Ba": 2.15,
    "La": 2.07, "Ce": 2.04, "Pr": 2.03, "Nd": 2.01, "Pm": 1.99, "Sm": 1.98,
    "Eu": 1.98, "Gd": 1.96, "Tb": 1.94, "Dy": 1.92, "Ho": 1.92, "Er": 1.89,
    "Tm": 1.90, "Yb": 1.87, "Lu": 1.87,
    "Hf": 1.75, "Ta": 1.70, "W": 1.62, "Re": 1.51, "Os": 1.44, "Ir": 1.41,
    "Pt": 1.36, "Au": 1.36, "Hg": 1.32, "Tl": 1.45, "Pb": 1.46, "Bi": 1.48,
}


def cordero_radius(symbol: str, *, default: float = 1.5) -> float:
    return CORDERO_2008.get(symbol, default)


def cordero_radii_for_atoms(atoms: Atoms) -> np.ndarray:
    return np.array([cordero_radius(s) for s in atoms.get_chemical_symbols()])
```

- [ ] **Step 4: Run test (PASS)** — `pytest tests/test_covalent_radii.py -v`

- [ ] **Step 5: Commit**

```bash
git add reactx/covalent_radii.py tests/test_covalent_radii.py
git commit -m "feat(covalent_radii): extract Cordero (2008) table to a shared module (Phase 9 step 1.1)"
```

### Task 1.2: `blender/render.py` を共有モジュール経由に切り替え

**Files:** Modify `blender/render.py` (`COVALENT_RADII_ANGSTROM` 定義箇所、約 line 74-101)

- [ ] **Step 1: Find existing dict** — `grep -n "^COVALENT_RADII_ANGSTROM" blender/render.py`

- [ ] **Step 2: Replace with import + fallback**

`COVALENT_RADII_ANGSTROM: dict[str, float] = { ... }` を全削除し、以下に置き換え (前後の comment 行は維持):

```python
# Cordero (2008) Dalton Trans. 2832. Canonical table is `reactx/covalent_radii.py`.
# Falls back to a vendored copy when this script runs inside Blender's bundled
# Python without reactx on sys.path.
try:
    from reactx.covalent_radii import CORDERO_2008 as COVALENT_RADII_ANGSTROM  # noqa: F401
except ImportError:
    COVALENT_RADII_ANGSTROM: dict[str, float] = {
        "H": 0.31, "He": 0.28,
        "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66,
        "F": 0.57, "Ne": 0.58,
        "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05,
        "Cl": 1.02, "Ar": 1.06,
        "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39,
        "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
        "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16,
        "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54,
        "Tc": 1.47, "Ru": 1.46, "Rh": 1.42, "Pd": 1.39, "Ag": 1.45, "Cd": 1.44,
        "In": 1.42, "Sn": 1.39, "Sb": 1.39, "Te": 1.38, "I": 1.39, "Xe": 1.40,
        "Cs": 2.44, "Ba": 2.15,
        "La": 2.07, "Ce": 2.04, "Pr": 2.03, "Nd": 2.01, "Pm": 1.99, "Sm": 1.98,
        "Eu": 1.98, "Gd": 1.96, "Tb": 1.94, "Dy": 1.92, "Ho": 1.92, "Er": 1.89,
        "Tm": 1.90, "Yb": 1.87, "Lu": 1.87,
        "Hf": 1.75, "Ta": 1.70, "W": 1.62, "Re": 1.51, "Os": 1.44, "Ir": 1.41,
        "Pt": 1.36, "Au": 1.36, "Hg": 1.32, "Tl": 1.45, "Pb": 1.46, "Bi": 1.48,
    }
```

- [ ] **Step 3: Verify import line present** (no `import bpy` triggered):

`grep -A 1 "from reactx.covalent_radii import CORDERO_2008" blender/render.py | head -5`
Expected: 1 import line found.

- [ ] **Step 4: Commit**

```bash
git add blender/render.py
git commit -m "refactor(blender): import Cordero radii from reactx with fallback (Phase 9 step 1.2)"
```

### Task 1.3: AFIRConstraint — single-pair force (TDD, additive)

**Files:** Create `tests/test_afir_constraint.py` and **append to** `reactx/artificial_force.py` (既存 Hookean 等は触らない)

- [ ] **Step 1: Write failing tests** (`tests/test_afir_constraint.py`):

```python
"""Tests for per-pair AFIR force with sticky latch (Phase 9, spec §4.1)."""
import numpy as np
import pytest
from ase import Atoms

from reactx.artificial_force import AFIRConstraint


def _atoms_along_x(d: float) -> Atoms:
    return Atoms("CC", positions=[[0.0, 0.0, 0.0], [d, 0.0, 0.0]])


def test_single_formed_pair_compresses():
    atoms = _atoms_along_x(2.5)
    c = AFIRConstraint(
        formed=[(0, 1)], broken=[],
        alpha_formed=[1.0], alpha_broken=[],
        formed_thresholds=[1.5], broken_thresholds=[],
    )
    forces = np.zeros((2, 3))
    c.adjust_forces(atoms, forces)
    np.testing.assert_allclose(forces[1], [-1.0, 0.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(forces[0], [+1.0, 0.0, 0.0], atol=1e-10)


def test_single_broken_pair_expands():
    atoms = _atoms_along_x(2.0)
    c = AFIRConstraint(
        formed=[], broken=[(0, 1)],
        alpha_formed=[], alpha_broken=[1.5],
        formed_thresholds=[], broken_thresholds=[4.0],
    )
    forces = np.zeros((2, 3))
    c.adjust_forces(atoms, forces)
    np.testing.assert_allclose(forces[1], [+1.5, 0.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(forces[0], [-1.5, 0.0, 0.0], atol=1e-10)


def test_pair_index_order_invariant():
    atoms = _atoms_along_x(2.5)
    f1 = np.zeros((2, 3)); f2 = np.zeros((2, 3))
    AFIRConstraint(
        formed=[(0, 1)], broken=[], alpha_formed=[1.0], alpha_broken=[],
        formed_thresholds=[1.5], broken_thresholds=[],
    ).adjust_forces(atoms, f1)
    AFIRConstraint(
        formed=[(1, 0)], broken=[], alpha_formed=[1.0], alpha_broken=[],
        formed_thresholds=[1.5], broken_thresholds=[],
    ).adjust_forces(atoms, f2)
    np.testing.assert_allclose(np.abs(f1), np.abs(f2), atol=1e-10)
```

- [ ] **Step 2: Run test (FAIL: ImportError)** — `pytest tests/test_afir_constraint.py -v`

- [ ] **Step 3: Append minimal AFIRConstraint to `reactx/artificial_force.py`**

ファイル **末尾に追加** (Hookean/PullApart 等は そのまま):

```python
import numpy as np
from ase.constraints import FixConstraint


class AFIRConstraint(FixConstraint):
    """Per-pair AFIR force with sticky per-pair latch (Phase 9, spec §4.1)."""

    def __init__(
        self,
        formed: list[tuple[int, int]],
        broken: list[tuple[int, int]],
        *,
        alpha_formed: list[float],
        alpha_broken: list[float],
        formed_thresholds: list[float],
        broken_thresholds: list[float],
    ):
        self.formed = [(int(a), int(b)) for a, b in formed]
        self.broken = [(int(a), int(b)) for a, b in broken]
        self.alpha_formed = [float(x) for x in alpha_formed]
        self.alpha_broken = [float(x) for x in alpha_broken]
        self.formed_thresholds = [float(t) for t in formed_thresholds]
        self.broken_thresholds = [float(t) for t in broken_thresholds]
        self.formed_latched = [False] * len(formed)
        self.broken_latched = [False] * len(broken)

    def adjust_positions(self, atoms, newpositions):
        return

    def adjust_forces(self, atoms, forces):
        pos = atoms.positions
        for k, (i, j) in enumerate(self.formed):
            r, d_hat = self._geom(pos, i, j)
            f_on_j = -self.alpha_formed[k] * d_hat
            forces[j] += f_on_j
            forces[i] -= f_on_j
        for k, (i, j) in enumerate(self.broken):
            r, d_hat = self._geom(pos, i, j)
            f_on_j = +self.alpha_broken[k] * d_hat
            forces[j] += f_on_j
            forces[i] -= f_on_j

    @staticmethod
    def _geom(pos, i, j):
        v = pos[j] - pos[i]
        r = float(np.linalg.norm(v))
        return r, v / r

    def get_indices(self):
        return sorted({a for pair in (self.formed + self.broken) for a in pair})
```

- [ ] **Step 4: Run test (PASS)** — `pytest tests/test_afir_constraint.py -v`

- [ ] **Step 5: Commit**

```bash
git add tests/test_afir_constraint.py reactx/artificial_force.py
git commit -m "feat(afir): introduce AFIRConstraint with per-pair force (Phase 9 step 1.3)"
```

### Task 1.4: AFIRConstraint — sticky latch (formed + broken + per-pair independence)

**Files:** Modify `tests/test_afir_constraint.py` and `reactx/artificial_force.py`

- [ ] **Step 1: Write failing tests** (append to `tests/test_afir_constraint.py`):

```python
def test_formed_latch_activates_when_threshold_reached():
    c = AFIRConstraint(
        formed=[(0, 1)], broken=[], alpha_formed=[1.0], alpha_broken=[],
        formed_thresholds=[1.5], broken_thresholds=[],
    )
    f = np.zeros((2, 3))
    c.adjust_forces(_atoms_along_x(2.0), f)
    assert c.formed_latched == [False]
    assert not np.allclose(f, 0)
    f = np.zeros((2, 3))
    c.adjust_forces(_atoms_along_x(1.4), f)
    assert c.formed_latched == [True]
    np.testing.assert_allclose(f, 0, atol=1e-12)


def test_formed_latch_is_sticky():
    c = AFIRConstraint(
        formed=[(0, 1)], broken=[], alpha_formed=[1.0], alpha_broken=[],
        formed_thresholds=[1.5], broken_thresholds=[],
    )
    c.adjust_forces(_atoms_along_x(1.4), np.zeros((2, 3)))
    f = np.zeros((2, 3))
    c.adjust_forces(_atoms_along_x(3.0), f)
    assert c.formed_latched == [True]
    np.testing.assert_allclose(f, 0, atol=1e-12)


def test_initial_latch_when_already_satisfied():
    c = AFIRConstraint(
        formed=[(0, 1)], broken=[], alpha_formed=[1.0], alpha_broken=[],
        formed_thresholds=[1.5], broken_thresholds=[],
    )
    f = np.zeros((2, 3))
    c.adjust_forces(_atoms_along_x(1.2), f)
    assert c.formed_latched == [True]
    np.testing.assert_allclose(f, 0, atol=1e-12)


def test_broken_latch_activates_when_threshold_reached():
    c = AFIRConstraint(
        formed=[], broken=[(0, 1)], alpha_formed=[], alpha_broken=[1.0],
        formed_thresholds=[], broken_thresholds=[3.0],
    )
    f = np.zeros((2, 3))
    c.adjust_forces(_atoms_along_x(2.0), f)
    assert c.broken_latched == [False]
    assert not np.allclose(f, 0)
    f = np.zeros((2, 3))
    c.adjust_forces(_atoms_along_x(3.5), f)
    assert c.broken_latched == [True]
    np.testing.assert_allclose(f, 0, atol=1e-12)


def test_broken_latch_is_sticky():
    c = AFIRConstraint(
        formed=[], broken=[(0, 1)], alpha_formed=[], alpha_broken=[1.0],
        formed_thresholds=[], broken_thresholds=[3.0],
    )
    c.adjust_forces(_atoms_along_x(3.5), np.zeros((2, 3)))
    f = np.zeros((2, 3))
    c.adjust_forces(_atoms_along_x(2.0), f)
    np.testing.assert_allclose(f, 0, atol=1e-12)


def test_per_pair_latch_independence():
    atoms = Atoms("CCCC", positions=[
        [0.0, 0.0, 0.0], [1.4, 0.0, 0.0],
        [0.0, 0.0, 5.0], [3.0, 0.0, 5.0],
    ])
    c = AFIRConstraint(
        formed=[(0, 1), (2, 3)], broken=[],
        alpha_formed=[1.0, 1.0], alpha_broken=[],
        formed_thresholds=[1.5, 1.5], broken_thresholds=[],
    )
    f = np.zeros((4, 3))
    c.adjust_forces(atoms, f)
    assert c.formed_latched == [True, False]
    np.testing.assert_allclose(f[0], 0, atol=1e-12)
    np.testing.assert_allclose(f[1], 0, atol=1e-12)
    assert not np.allclose(f[2], 0)
    assert not np.allclose(f[3], 0)
```

- [ ] **Step 2: Run tests (FAIL — no latch logic)**

- [ ] **Step 3: Replace `adjust_forces` with latch-aware version**

```python
    def adjust_forces(self, atoms, forces):
        pos = atoms.positions
        for k, (i, j) in enumerate(self.formed):
            if self.formed_latched[k]:
                continue
            r, d_hat = self._geom(pos, i, j)
            if r <= self.formed_thresholds[k]:
                self.formed_latched[k] = True
                continue
            f_on_j = -self.alpha_formed[k] * d_hat
            forces[j] += f_on_j
            forces[i] -= f_on_j
        for k, (i, j) in enumerate(self.broken):
            if self.broken_latched[k]:
                continue
            r, d_hat = self._geom(pos, i, j)
            if r >= self.broken_thresholds[k]:
                self.broken_latched[k] = True
                continue
            f_on_j = +self.alpha_broken[k] * d_hat
            forces[j] += f_on_j
            forces[i] -= f_on_j
```

- [ ] **Step 4: Run tests (PASS)** — `pytest tests/test_afir_constraint.py -v`

- [ ] **Step 5: Commit**

```bash
git add tests/test_afir_constraint.py reactx/artificial_force.py
git commit -m "feat(afir): add sticky per-pair latch (Phase 9 step 1.4)"
```

### Task 1.5: AFIRConstraint — validation incl. **non-empty α=0 重複防御** (spec §10.3 critical-2)

**Files:** Modify `tests/test_afir_constraint.py` and `reactx/artificial_force.py`

- [ ] **Step 1: Append failing tests**:

```python
def test_negative_alpha_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        AFIRConstraint(
            formed=[(0, 1)], broken=[], alpha_formed=[-0.1], alpha_broken=[],
            formed_thresholds=[1.5], broken_thresholds=[],
        )


def test_zero_alpha_rejected_for_nonempty_formed_pair():
    """spec §10.3 critical-2: α=0 on non-empty pair would never latch."""
    with pytest.raises(ValueError, match="positive"):
        AFIRConstraint(
            formed=[(0, 1)], broken=[], alpha_formed=[0.0], alpha_broken=[],
            formed_thresholds=[1.5], broken_thresholds=[],
        )


def test_zero_alpha_rejected_for_nonempty_broken_pair():
    with pytest.raises(ValueError, match="positive"):
        AFIRConstraint(
            formed=[], broken=[(0, 1)], alpha_formed=[], alpha_broken=[0.0],
            formed_thresholds=[], broken_thresholds=[3.0],
        )


def test_empty_pair_set_no_alpha_constraint():
    c = AFIRConstraint(
        formed=[], broken=[], alpha_formed=[], alpha_broken=[],
        formed_thresholds=[], broken_thresholds=[],
    )
    assert c.formed_latched == []
    assert c.broken_latched == []


def test_zero_or_negative_threshold_rejected():
    with pytest.raises(ValueError, match="positive"):
        AFIRConstraint(
            formed=[(0, 1)], broken=[], alpha_formed=[1.0], alpha_broken=[],
            formed_thresholds=[0.0], broken_thresholds=[],
        )


def test_alpha_length_mismatch_rejected():
    with pytest.raises(ValueError, match="alpha_formed length"):
        AFIRConstraint(
            formed=[(0, 1)], broken=[], alpha_formed=[1.0, 2.0], alpha_broken=[],
            formed_thresholds=[1.5], broken_thresholds=[],
        )


def test_threshold_length_mismatch_rejected():
    with pytest.raises(ValueError, match="formed_thresholds length"):
        AFIRConstraint(
            formed=[(0, 1)], broken=[], alpha_formed=[1.0], alpha_broken=[],
            formed_thresholds=[1.5, 2.0], broken_thresholds=[],
        )


def test_coincident_atoms_fail_fast():
    a = Atoms("CC", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    c = AFIRConstraint(
        formed=[(0, 1)], broken=[], alpha_formed=[1.0], alpha_broken=[],
        formed_thresholds=[1.5], broken_thresholds=[],
    )
    with pytest.raises(ValueError, match="coincide"):
        c.adjust_forces(a, np.zeros((2, 3)))


def test_all_latched_helper():
    c = AFIRConstraint(
        formed=[(0, 1)], broken=[(0, 1)], alpha_formed=[1.0], alpha_broken=[1.0],
        formed_thresholds=[1.5], broken_thresholds=[3.0],
    )
    assert c.all_latched() is False
    c.formed_latched[0] = True
    assert c.all_latched() is False
    c.broken_latched[0] = True
    assert c.all_latched() is True


def test_all_latched_empty_constraint():
    c = AFIRConstraint(
        formed=[], broken=[], alpha_formed=[], alpha_broken=[],
        formed_thresholds=[], broken_thresholds=[],
    )
    assert c.all_latched() is True
```

- [ ] **Step 2: Run tests (FAIL — no validation)**

- [ ] **Step 3: Replace `__init__`, update `_geom`, add `all_latched`**

`__init__`:

```python
    def __init__(
        self,
        formed: list[tuple[int, int]],
        broken: list[tuple[int, int]],
        *,
        alpha_formed: list[float],
        alpha_broken: list[float],
        formed_thresholds: list[float],
        broken_thresholds: list[float],
    ):
        if len(alpha_formed) != len(formed):
            raise ValueError(
                f"alpha_formed length {len(alpha_formed)} != n_formed {len(formed)}"
            )
        if len(alpha_broken) != len(broken):
            raise ValueError(
                f"alpha_broken length {len(alpha_broken)} != n_broken {len(broken)}"
            )
        if len(formed_thresholds) != len(formed):
            raise ValueError(
                f"formed_thresholds length {len(formed_thresholds)} != n_formed {len(formed)}"
            )
        if len(broken_thresholds) != len(broken):
            raise ValueError(
                f"broken_thresholds length {len(broken_thresholds)} != n_broken {len(broken)}"
            )
        if any(a < 0 for a in alpha_formed) or any(a < 0 for a in alpha_broken):
            raise ValueError("alpha_* must be non-negative")
        # spec §10.3 critical-2: defense in depth.
        if len(formed) > 0 and any(a <= 0 for a in alpha_formed):
            raise ValueError(
                "alpha_formed must be positive for non-empty formed pair set "
                "(spec §10.3: α=0 would never latch)"
            )
        if len(broken) > 0 and any(a <= 0 for a in alpha_broken):
            raise ValueError(
                "alpha_broken must be positive for non-empty broken pair set"
            )
        if any(t <= 0 for t in formed_thresholds) or any(t <= 0 for t in broken_thresholds):
            raise ValueError("thresholds must be positive")
        self.formed = [(int(a), int(b)) for a, b in formed]
        self.broken = [(int(a), int(b)) for a, b in broken]
        self.alpha_formed = [float(x) for x in alpha_formed]
        self.alpha_broken = [float(x) for x in alpha_broken]
        self.formed_thresholds = [float(t) for t in formed_thresholds]
        self.broken_thresholds = [float(t) for t in broken_thresholds]
        self.formed_latched = [False] * len(formed)
        self.broken_latched = [False] * len(broken)
```

`_geom` r→0 guard:

```python
    @staticmethod
    def _geom(pos, i, j):
        v = pos[j] - pos[i]
        r = float(np.linalg.norm(v))
        if r < 1e-6:
            raise ValueError(
                f"AFIRConstraint: atoms {i} and {j} coincide (r={r:.2e}); "
                f"check fragment placement geometry"
            )
        return r, v / r
```

`all_latched` helper before `get_indices`:

```python
    def all_latched(self) -> bool:
        return all(self.formed_latched) and all(self.broken_latched)
```

- [ ] **Step 4: Run tests (PASS)** — `pytest tests/test_afir_constraint.py -v`

- [ ] **Step 5: Commit**

```bash
git add tests/test_afir_constraint.py reactx/artificial_force.py
git commit -m "feat(afir): validate inputs incl. α=0 reject for non-empty pairs (Phase 9 step 1.5)"
```

### Task 1.6: AFIRConstraint — finite-difference test against V=±α·r

**Files:** Modify `tests/test_afir_constraint.py`

- [ ] **Step 1: Append the test** (passes immediately for the linear potential):

```python
def test_force_matches_finite_difference_of_artificial_potential():
    rng = np.random.default_rng(42)
    pos = rng.uniform(-2.0, 2.0, size=(4, 3))
    pos[1] = pos[0] + np.array([3.0, 0.0, 0.0])
    pos[3] = pos[2] + np.array([0.5, 0.0, 0.0])
    atoms = Atoms("CCCC", positions=pos)

    formed = [(0, 1)]
    broken = [(2, 3)]
    alpha_f = [1.5]
    alpha_b = [2.0]

    def fresh() -> AFIRConstraint:
        return AFIRConstraint(
            formed=formed, broken=broken,
            alpha_formed=alpha_f, alpha_broken=alpha_b,
            formed_thresholds=[1.5], broken_thresholds=[3.0],
        )

    f_analytical = np.zeros((4, 3))
    fresh().adjust_forces(atoms, f_analytical)

    def V(p):
        r01 = float(np.linalg.norm(p[1] - p[0]))
        r23 = float(np.linalg.norm(p[3] - p[2]))
        return alpha_f[0] * r01 + (-alpha_b[0]) * r23

    eps = 1e-5
    f_numerical = np.zeros_like(f_analytical)
    for atom_idx in range(4):
        for axis in range(3):
            p_plus = atoms.positions.copy()
            p_minus = atoms.positions.copy()
            p_plus[atom_idx, axis] += eps
            p_minus[atom_idx, axis] -= eps
            f_numerical[atom_idx, axis] = -(V(p_plus) - V(p_minus)) / (2 * eps)

    np.testing.assert_allclose(f_analytical, f_numerical, atol=1e-6)
```

- [ ] **Step 2: Run test (PASS)** — `pytest tests/test_afir_constraint.py::test_force_matches_finite_difference_of_artificial_potential -v`

- [ ] **Step 3: Commit**

```bash
git add tests/test_afir_constraint.py
git commit -m "test(afir): assert force matches finite-diff of V=±α·r (Phase 9 step 1.6)"
```

### Task 1.7: `build_afir_constraint` factory + scalar/list broadcasting

**Files:** Modify `tests/test_afir_constraint.py` and `reactx/artificial_force.py`

- [ ] **Step 1: Append tests**:

```python
from reactx.artificial_force import build_afir_constraint


def test_build_afir_returns_empty_when_no_pairs():
    a = Atoms("CC", positions=np.zeros((2, 3)))
    cs = build_afir_constraint(
        a, formed=[], broken=[],
        alpha_formed=0.0, alpha_broken=0.0,
        formed_thresholds=[], broken_thresholds=[],
    )
    assert cs == []


def test_build_afir_broadcasts_scalar_alpha_to_list():
    a = Atoms("CCC", positions=np.zeros((3, 3)))
    cs = build_afir_constraint(
        a, formed=[(0, 1), (0, 2)], broken=[],
        alpha_formed=0.5, alpha_broken=0.0,
        formed_thresholds=[1.5, 1.5], broken_thresholds=[],
    )
    assert cs[0].alpha_formed == [0.5, 0.5]


def test_build_afir_passes_through_list_alpha():
    a = Atoms("CCC", positions=np.zeros((3, 3)))
    cs = build_afir_constraint(
        a, formed=[(0, 1), (0, 2)], broken=[],
        alpha_formed=[0.5, 1.5], alpha_broken=0.0,
        formed_thresholds=[1.5, 1.5], broken_thresholds=[],
    )
    assert cs[0].alpha_formed == [0.5, 1.5]


def test_build_afir_alpha_list_length_mismatch_raises():
    a = Atoms("CCC", positions=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="alpha_formed list length"):
        build_afir_constraint(
            a, formed=[(0, 1), (0, 2)], broken=[],
            alpha_formed=[0.5, 1.5, 2.0], alpha_broken=0.0,
            formed_thresholds=[1.5, 1.5], broken_thresholds=[],
        )
```

- [ ] **Step 2: Run (FAIL)**

- [ ] **Step 3: Append `build_afir_constraint` and `_broadcast_alpha`**:

```python
def build_afir_constraint(
    atoms,
    formed: list[tuple[int, int]],
    broken: list[tuple[int, int]],
    *,
    alpha_formed,
    alpha_broken,
    formed_thresholds: list[float],
    broken_thresholds: list[float],
) -> list[AFIRConstraint]:
    if not formed and not broken:
        return []
    af = _broadcast_alpha(alpha_formed, len(formed), key="alpha_formed")
    ab = _broadcast_alpha(alpha_broken, len(broken), key="alpha_broken")
    return [AFIRConstraint(
        formed, broken,
        alpha_formed=af, alpha_broken=ab,
        formed_thresholds=formed_thresholds,
        broken_thresholds=broken_thresholds,
    )]


def _broadcast_alpha(value, n: int, *, key: str) -> list[float]:
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f"{key} list length {len(value)} != n_pairs {n}")
        return [float(v) for v in value]
    return [float(value)] * n
```

- [ ] **Step 4: Run (PASS)** — `pytest tests/test_afir_constraint.py -v`

- [ ] **Step 5: Commit**

```bash
git add tests/test_afir_constraint.py reactx/artificial_force.py
git commit -m "feat(afir): add build_afir_constraint factory with scalar broadcast (Phase 9 step 1.7)"
```

### Task 1.8: latch ≠ reached_product regression test (spec §1, §10.3 critical-1)

**Files:** Modify `tests/test_afir_constraint.py`

- [ ] **Step 1: Append the regression test** (latch state と final-frame distance が独立であることを直接 assert):

```python
def test_latch_state_independent_of_post_latch_geometry():
    """Spec §1, §10.3 critical-1: sticky latch ≠ reached_product.

    Once latched, AFIR force is permanently 0 — but UMA / external forces
    may push the pair distance back outside the threshold. The cli flow
    therefore reads `reached_product` from the *final-frame distance*,
    which can disagree with the latch state.

    This test demonstrates the regression mode by:
    1. Crossing threshold → latch ON
    2. Moving back outside threshold → latch stays ON (sticky)
    3. Asserting that final r > threshold (the geometry signal that
       reached_product would treat as False)
    """
    c = AFIRConstraint(
        formed=[(0, 1)], broken=[], alpha_formed=[1.0], alpha_broken=[],
        formed_thresholds=[1.5], broken_thresholds=[],
    )
    c.adjust_forces(_atoms_along_x(1.4), np.zeros((2, 3)))
    assert c.formed_latched == [True]

    atoms_b = _atoms_along_x(2.5)
    f_b = np.zeros((2, 3))
    c.adjust_forces(atoms_b, f_b)
    np.testing.assert_allclose(f_b, 0, atol=1e-12)
    assert c.formed_latched == [True]

    r_final = float(np.linalg.norm(
        atoms_b.positions[1] - atoms_b.positions[0]
    ))
    assert r_final > 1.5  # geometry says "not reached" while latch says "ever crossed"
```

- [ ] **Step 2: Run (PASS)** — `pytest tests/test_afir_constraint.py::test_latch_state_independent_of_post_latch_geometry -v`

- [ ] **Step 3: Commit**

```bash
git add tests/test_afir_constraint.py
git commit -m "test(afir): regression for latch ≠ reached_product semantics (Phase 9 step 1.8)"
```

### Stage 1 完了 checkpoint

- AFIRConstraint, build_afir_constraint, Cordero radii が **追加された** (Hookean / PullApart / build_restraints 等は **そのまま**)
- 既存 reactx/scoring.py, reactx/cli.py, reactx/config.py, reactx/path_relax.py は **未変更** (旧挙動を維持)
- `pytest -m "not slow and not blender"` green

---

## Stage 2A: Additive new API (config + scoring + path_relax を additive に拡張、cli はまだ旧 API を使う)

### Task 2A.1: config.py に `[afir]+[scoring]` schema を additive 追加

**Files:** Modify `reactx/config.py` (旧 RestraintConfig / load_config を **そのまま残す**) + create `tests/test_config_afir.py`

- [ ] **Step 1: Write failing tests** in `tests/test_config_afir.py`:

```python
"""Phase 9 [afir] + [scoring] schema (additive — coexists with old [restraints]).

Spec: docs/superpowers/specs/2026-05-08-afir-force-design.md §4.4, §6.2
"""
from pathlib import Path

import pytest

from reactx.config import ConfigError, load_config_v9


def _write(tmp_path: Path, body: str) -> Path:
    rxn = tmp_path / "x.rxn"
    rxn.write_text("placeholder")
    (tmp_path / "x.rxn.toml").write_text(body)
    return rxn


def test_load_minimal_sn2(tmp_path: Path):
    rxn = _write(tmp_path, """
description = "SN2"
formed = [[1, 3]]
broken = [[1, 2]]

[afir]
alpha_formed = 0.7
alpha_broken = 0.5
max_relax_steps = 100

[scoring]
r_broken_threshold = 4.0
""")
    cfg = load_config_v9(rxn)
    assert cfg.afir.alpha_formed == 0.7
    assert cfg.afir.alpha_broken == 0.5
    assert cfg.afir.max_relax_steps == 100
    assert cfg.scoring.r_broken_threshold == 4.0
    assert cfg.scoring.r_formed_threshold is None


def test_load_e2_per_bond_lists(tmp_path: Path):
    rxn = _write(tmp_path, """
description = "E2"
formed = [[4, 5]]
broken = [[2, 5], [1, 3]]

[afir]
alpha_formed = 1.5
alpha_broken = [1.0, 1.5]
max_relax_steps = 200

[scoring]
r_broken_threshold = [3.0, 4.0]
""")
    cfg = load_config_v9(rxn)
    assert cfg.afir.alpha_broken == (1.0, 1.5)
    assert cfg.scoring.r_broken_threshold == (3.0, 4.0)


def test_load_da_with_empty_broken_alpha_broken_omitted(tmp_path: Path):
    """spec §4.4: alpha_broken omitted is valid when broken=[]."""
    rxn = _write(tmp_path, """
description = "DA"
formed = [[1, 5], [4, 6]]
broken = []

[afir]
alpha_formed = [2.5, 2.5]
max_relax_steps = 200
""")
    cfg = load_config_v9(rxn)
    assert cfg.broken == ()
    # default broadcast result is empty for n=0
    assert cfg.afir.alpha_broken == ()


def test_load_da_with_empty_broken_alpha_broken_zero(tmp_path: Path):
    """spec §4.4: alpha_broken=0.0 scalar is also valid."""
    rxn = _write(tmp_path, """
description = "DA"
formed = [[1, 5], [4, 6]]
broken = []

[afir]
alpha_formed = [2.5, 2.5]
alpha_broken = 0.0
max_relax_steps = 200
""")
    cfg = load_config_v9(rxn)
    assert cfg.broken == ()


def test_load_sn1_dissoc_with_empty_formed(tmp_path: Path):
    rxn = _write(tmp_path, """
description = "SN1 dissoc"
formed = []
broken = [[1, 5]]

[afir]
alpha_broken = 1.5
max_relax_steps = 200

[scoring]
r_broken_threshold = 6.0
""")
    cfg = load_config_v9(rxn)
    assert cfg.formed == ()
    assert cfg.afir.alpha_formed == ()


def test_old_restraints_section_rejected(tmp_path: Path):
    rxn = _write(tmp_path, """
description = "old"
formed = [[1, 2]]
broken = []

[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 100
""")
    with pytest.raises(ConfigError, match=r"restraints"):
        load_config_v9(rxn)


def test_alpha_zero_rejected_for_nonempty_pair(tmp_path: Path):
    rxn = _write(tmp_path, """
description = "α=0"
formed = [[1, 2]]
broken = [[3, 4]]

[afir]
alpha_formed = 0.0
alpha_broken = 1.0
max_relax_steps = 100

[scoring]
r_broken_threshold = 4.0
""")
    with pytest.raises(ConfigError, match="alpha_formed"):
        load_config_v9(rxn)


def test_alpha_negative_rejected(tmp_path: Path):
    rxn = _write(tmp_path, """
description = "α<0"
formed = [[1, 2]]
broken = []

[afir]
alpha_formed = -1.0
max_relax_steps = 100
""")
    with pytest.raises(ConfigError):
        load_config_v9(rxn)


def test_r_broken_threshold_required_when_broken_nonempty(tmp_path: Path):
    rxn = _write(tmp_path, """
description = "missing r_broken"
formed = []
broken = [[1, 2]]

[afir]
alpha_broken = 1.0
max_relax_steps = 100
""")
    with pytest.raises(ConfigError, match="r_broken_threshold"):
        load_config_v9(rxn)


def test_nan_alpha_rejected(tmp_path: Path):
    rxn = _write(tmp_path, """
description = "nan"
formed = [[1, 2]]
broken = []

[afir]
alpha_formed = nan
max_relax_steps = 100
""")
    with pytest.raises(ConfigError, match="finite"):
        load_config_v9(rxn)


def test_inf_alpha_rejected(tmp_path: Path):
    rxn = _write(tmp_path, """
description = "inf"
formed = [[1, 2]]
broken = []

[afir]
alpha_formed = inf
max_relax_steps = 100
""")
    with pytest.raises(ConfigError, match="finite"):
        load_config_v9(rxn)


def test_alpha_length_mismatch_rejected(tmp_path: Path):
    rxn = _write(tmp_path, """
description = "mismatch"
formed = [[1, 2]]
broken = [[3, 4]]

[afir]
alpha_formed = [1.0, 2.0]
alpha_broken = 0.5
max_relax_steps = 100

[scoring]
r_broken_threshold = 4.0
""")
    with pytest.raises(ConfigError, match="alpha_formed"):
        load_config_v9(rxn)


def test_r_formed_threshold_list_passthrough(tmp_path: Path):
    rxn = _write(tmp_path, """
description = "list r_formed"
formed = [[1, 2], [3, 4]]
broken = []

[afir]
alpha_formed = [1.0, 2.0]
max_relax_steps = 100

[scoring]
r_formed_threshold = [1.5, 1.8]
""")
    cfg = load_config_v9(rxn)
    assert cfg.scoring.r_formed_threshold == (1.5, 1.8)


def test_max_relax_steps_must_be_positive(tmp_path: Path):
    rxn = _write(tmp_path, """
description = "zero"
formed = [[1, 2]]
broken = []

[afir]
alpha_formed = 1.0
max_relax_steps = 0
""")
    with pytest.raises(ConfigError, match="max_relax_steps"):
        load_config_v9(rxn)


def test_obsolete_top_level_keys_rejected(tmp_path: Path):
    for old_key in ["k_form", "k_broken", "r_broken", "r_form"]:
        body = f"""
description = "old key"
formed = [[1, 2]]
broken = []
{old_key} = 0.5

[afir]
alpha_formed = 1.0
max_relax_steps = 100
"""
        rxn = _write(tmp_path, body)
        with pytest.raises(ConfigError):
            load_config_v9(rxn)
```

- [ ] **Step 2: Run (FAIL: ImportError)**

- [ ] **Step 3: Add v9 API to `reactx/config.py`** (旧 API は **そのまま残す**、ファイルの末尾に追記):

```python
# ===== Phase 9 schema (additive, coexists with v8 [restraints]) =====

import math


class ConfigError(ValueError):
    """Raised when `<rxn_path>.toml` violates the Phase 9 schema."""


@dataclass(frozen=True)
class AFIRSection:
    alpha_formed: float | tuple[float, ...]
    alpha_broken: float | tuple[float, ...]
    max_relax_steps: int


@dataclass(frozen=True)
class ScoringSection:
    r_broken_threshold: float | tuple[float, ...] | None = None
    r_formed_threshold: float | tuple[float, ...] | None = None


@dataclass(frozen=True)
class ReactionConfigV9:
    description: str
    formed: tuple[tuple[int, int], ...]
    broken: tuple[tuple[int, int], ...]
    afir: AFIRSection
    scoring: ScoringSection
    sampling: SamplingConfig = field(default_factory=SamplingConfig)


_V9_TOP_LEVEL_KEYS = {"description", "formed", "broken", "afir", "scoring", "sampling"}
_V9_TOP_LEVEL_REQUIRED = {"description", "formed", "broken", "afir"}
_V9_OBSOLETE_TOP_LEVEL = {"restraints", "prescreen", "k_form", "k_broken",
                          "r_broken", "r_form"}
_V9_AFIR_KEYS = {"alpha_formed", "alpha_broken", "max_relax_steps"}
# Per spec §4.4: alpha_formed / alpha_broken are optional (default 0.0).
# Only max_relax_steps is required.
_V9_AFIR_REQUIRED = {"max_relax_steps"}
_V9_SCORING_KEYS = {"r_broken_threshold", "r_formed_threshold"}


def load_config_v9(rxn_path: Path) -> ReactionConfigV9:
    """Phase 9 loader. Coexists with v8 `load_config` until cli.py switches over."""
    toml_path = sidecar_path(rxn_path)
    if not toml_path.is_file():
        raise FileNotFoundError(
            f"sidecar TOML not found: expected {toml_path} alongside {rxn_path}"
        )
    raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    return _validate_v9(raw, source=str(toml_path))


def _validate_v9(raw: dict, *, source: str) -> ReactionConfigV9:
    for k in _V9_OBSOLETE_TOP_LEVEL:
        if k in raw:
            raise ConfigError(
                f"{source}: '[{k}]' or '{k}' is removed in Phase 9; "
                f"migrate to '[afir]' / '[scoring]' (see spec §4.4)"
            )
    _v9_check_keys(raw, _V9_TOP_LEVEL_KEYS, _V9_TOP_LEVEL_REQUIRED,
                   scope="<top>", source=source)

    description = raw["description"]
    if not isinstance(description, str) or not description.strip():
        raise ConfigError(f"{source}: 'description' must be a non-empty string")

    formed = _v9_to_pair_tuple(raw["formed"], key="formed", source=source)
    broken = _v9_to_pair_tuple(raw["broken"], key="broken", source=source)
    if not formed and not broken:
        raise ConfigError(
            f"{source}: at least one of 'formed' or 'broken' must be non-empty"
        )

    afir = _v9_build_afir(
        raw["afir"], formed_count=len(formed), broken_count=len(broken),
        source=source,
    )
    scoring = _v9_build_scoring(
        raw.get("scoring", {}), formed_count=len(formed), broken_count=len(broken),
        source=source,
    )
    sampling = _build_sampling(raw.get("sampling", {}), source=source)  # reuse old helper

    return ReactionConfigV9(
        description=description, formed=formed, broken=broken,
        afir=afir, scoring=scoring, sampling=sampling,
    )


def _v9_build_afir(raw: dict, *, formed_count: int, broken_count: int,
                   source: str) -> AFIRSection:
    _v9_check_keys(raw, _V9_AFIR_KEYS, _V9_AFIR_REQUIRED, scope="[afir]", source=source)
    alpha_formed = _v9_normalize_alpha(
        raw.get("alpha_formed", 0.0), expected_count=formed_count,
        key="alpha_formed", source=source,
    )
    alpha_broken = _v9_normalize_alpha(
        raw.get("alpha_broken", 0.0), expected_count=broken_count,
        key="alpha_broken", source=source,
    )
    max_steps = raw["max_relax_steps"]
    if not isinstance(max_steps, int) or max_steps <= 0:
        raise ConfigError(
            f"{source}: '[afir].max_relax_steps' must be a positive integer "
            f"(got {max_steps!r})"
        )
    return AFIRSection(
        alpha_formed=alpha_formed, alpha_broken=alpha_broken,
        max_relax_steps=int(max_steps),
    )


def _v9_normalize_alpha(value, *, expected_count: int, key: str,
                        source: str) -> float | tuple[float, ...]:
    def _check_finite(v, ctx: str):
        if not isinstance(v, (int, float)):
            raise ConfigError(
                f"{source}: '[afir].{key}'{ctx} must be numeric (got {v!r})"
            )
        if not math.isfinite(v):
            raise ConfigError(
                f"{source}: '[afir].{key}'{ctx} must be finite (got {v!r})"
            )

    if isinstance(value, list):
        for idx, v in enumerate(value):
            _check_finite(v, f"[{idx}]")
    else:
        _check_finite(value, "")

    if expected_count == 0:
        return tuple()  # broadcast result is empty regardless

    if isinstance(value, list):
        if len(value) != expected_count:
            raise ConfigError(
                f"{source}: '[afir].{key}' list length {len(value)} != "
                f"expected {expected_count}"
            )
        for v in value:
            if v <= 0:
                raise ConfigError(
                    f"{source}: '[afir].{key}' must be > 0 for non-empty "
                    f"pair set (spec §4.4 forbids α=0); got {v}"
                )
        return tuple(float(v) for v in value)

    if value <= 0:
        raise ConfigError(
            f"{source}: '[afir].{key}' must be > 0 for non-empty pair set "
            f"(spec §4.4 forbids α=0); got {value}"
        )
    return float(value)


def _v9_build_scoring(raw: dict, *, formed_count: int, broken_count: int,
                      source: str) -> ScoringSection:
    _v9_check_keys(raw, _V9_SCORING_KEYS, set(), scope="[scoring]", source=source)
    r_broken = _v9_normalize_threshold(
        raw.get("r_broken_threshold"), expected_count=broken_count,
        key="r_broken_threshold", source=source,
        required_when_pairs_present=True,
    )
    r_formed = _v9_normalize_threshold(
        raw.get("r_formed_threshold"), expected_count=formed_count,
        key="r_formed_threshold", source=source,
        required_when_pairs_present=False,
    )
    return ScoringSection(
        r_broken_threshold=r_broken, r_formed_threshold=r_formed,
    )


def _v9_normalize_threshold(value, *, expected_count: int, key: str, source: str,
                            required_when_pairs_present: bool):
    if expected_count == 0:
        return None
    if value is None:
        if required_when_pairs_present:
            raise ConfigError(
                f"{source}: '[scoring].{key}' is required when there are "
                f"non-empty pairs"
            )
        return None
    if isinstance(value, list):
        if len(value) != expected_count:
            raise ConfigError(
                f"{source}: '[scoring].{key}' list length {len(value)} != "
                f"expected {expected_count}"
            )
        for v in value:
            if not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                raise ConfigError(
                    f"{source}: '[scoring].{key}' must contain finite, "
                    f"positive values; got {v!r}"
                )
        return tuple(float(v) for v in value)
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ConfigError(
            f"{source}: '[scoring].{key}' must be a finite, positive "
            f"number (got {value!r})"
        )
    return float(value)


def _v9_check_keys(raw: dict, allowed: set[str], required: set[str], *,
                   scope: str, source: str) -> None:
    missing = required - raw.keys()
    if missing:
        raise ConfigError(
            f"{source}: {scope} missing required keys: {sorted(missing)}"
        )
    extra = set(raw.keys()) - allowed
    if extra:
        raise ConfigError(
            f"{source}: {scope} contains unknown keys: {sorted(extra)}"
        )


def _v9_to_pair_tuple(value, *, key: str, source: str) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list):
        raise ConfigError(f"{source}: '{key}' must be a list of [i, j] pairs")
    out: list[tuple[int, int]] = []
    for pair in value:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(isinstance(x, int) for x in pair)
        ):
            raise ConfigError(
                f"{source}: '{key}' entries must be [int, int] pairs (got {pair!r})"
            )
        out.append((int(pair[0]), int(pair[1])))
    return tuple(out)


def resolve_alpha_formed_v9(cfg: ReactionConfigV9) -> list[float]:
    return _v9_broadcast(cfg.afir.alpha_formed, len(cfg.formed))


def resolve_alpha_broken_v9(cfg: ReactionConfigV9) -> list[float]:
    return _v9_broadcast(cfg.afir.alpha_broken, len(cfg.broken))


def _v9_broadcast(value, n: int) -> list[float]:
    if n == 0:
        return []
    if isinstance(value, tuple):
        return [float(v) for v in value]
    return [float(value)] * n
```

- [ ] **Step 4: Run (PASS)** — `pytest tests/test_config_afir.py -v`

- [ ] **Step 5: Run full unit suite (still green — old API untouched)** — `pytest -m "not slow and not blender" -q`

- [ ] **Step 6: Commit**

```bash
git add reactx/config.py tests/test_config_afir.py
git commit -m "feat(config): add Phase 9 [afir]+[scoring] schema (additive) (Phase 9 step 2A.1)"
```

### Task 2A.2: scoring.py に new functions と new TrialResult を additive 追加

**Files:** Modify `reactx/scoring.py` (旧 `reached_product`, `score_trials`, `TrialResult` は **そのまま残す**)

- [ ] **Step 1: Write failing tests** — append to `tests/test_scoring.py` (旧 tests は **削除しない**):

```python
# ----- Phase 9 additions (coexist with v8) ---------------------------------
from reactx.scoring import (
    TrialResultV9,
    count_initial_latched,
    product_distance_residual,
    reached_product_v9,
    resolve_broken_thresholds,
    resolve_formed_thresholds,
    score_trials_v9,
)
from reactx.config import ConfigError


def _atoms_cc(d: float):
    from ase import Atoms as A
    return A("CC", positions=[[0.0, 0.0, 0.0], [d, 0.0, 0.0]])


def test_resolve_formed_thresholds_default_is_1_15_times_rsum():
    a = _atoms_cc(1.5)
    thrs = resolve_formed_thresholds(a, formed=[(0, 1)], override=None)
    assert thrs == pytest.approx([1.748], abs=1e-6)


def test_resolve_broken_thresholds_required_when_nonempty():
    a = _atoms_cc(1.0)
    with pytest.raises(ConfigError, match="r_broken_threshold"):
        resolve_broken_thresholds(a, broken=[(0, 1)], override=None)


def test_reached_product_v9_per_pair():
    a = _atoms_cc(1.5)
    assert reached_product_v9(a, [(0, 1)], [], [1.6], [])
    assert not reached_product_v9(_atoms_cc(2.0), [(0, 1)], [], [1.6], [])


def test_residual_zero_when_reached():
    assert product_distance_residual(
        _atoms_cc(1.4), [(0, 1)], [], [1.6], []
    ) == 0.0


def test_residual_positive_when_formed_overshoot():
    r = product_distance_residual(_atoms_cc(2.0), [(0, 1)], [], [1.6], [])
    assert r == pytest.approx(0.16, abs=1e-12)


def test_count_initial_latched_both():
    counts = count_initial_latched(
        _atoms_cc(1.5), [(0, 1)], [(0, 1)], [1.6], [1.4],
    )
    assert counts == {"formed": 1, "broken": 1}


def _v9_trial(idx, *, reached, peak, residual=0.0):
    return TrialResultV9(
        trial_idx=idx,
        direction=np.array([0.0, 0.0, 1.0]),
        frames=[], energies=[],
        reached_product=reached, peak_energy=peak,
        n_steps=10,
        formed_thresholds=[1.6],
        broken_thresholds=[],
        product_distance_residual=residual,
        formed_latch_count=0, broken_latch_count=0,
        initial_latched_formed=0, initial_latched_broken=0,
    )


def test_score_trials_v9_prefers_reached_with_lowest_peak():
    best = score_trials_v9([
        _v9_trial(0, reached=True, peak=10.0),
        _v9_trial(1, reached=True, peak=5.0),
        _v9_trial(2, reached=False, peak=1.0),
    ])
    assert best.trial_idx == 1


def test_score_trials_v9_fallback_uses_residual_then_peak():
    best = score_trials_v9([
        _v9_trial(0, reached=False, peak=1.0, residual=10.0),
        _v9_trial(1, reached=False, peak=5.0, residual=2.0),
        _v9_trial(2, reached=False, peak=2.0, residual=2.0),
    ])
    assert best.trial_idx == 2
```

(`import numpy as np` などは ファイル冒頭の既存 import に従う)

- [ ] **Step 2: Run (FAIL: ImportError)**

- [ ] **Step 3: Add v9 API to `reactx/scoring.py`** (末尾に追記、旧コードはそのまま):

```python
# ===== Phase 9 additions =====

from reactx.config import ConfigError
from reactx.covalent_radii import cordero_radii_for_atoms

COVALENT_FORMED_TOLERANCE = 1.15
"""Phase 9 default formed threshold: r ≤ 1.15 × Rsum_Cordero."""


@dataclass
class TrialResultV9:
    """Phase 9 outcome of one placement / relaxation trial.

    14 fields. `reached_product` (final-frame distance) is the sole success
    criterion. Latch counts and initial_latched are diagnostics only.
    `formed_thresholds` / `broken_thresholds` capture the resolved per-pair
    threshold values used at runtime so meta.json can record them.
    """
    trial_idx: int
    direction: np.ndarray
    frames: list[Atoms]
    energies: list[float]
    reached_product: bool
    peak_energy: float
    n_steps: int
    formed_thresholds: list[float]
    broken_thresholds: list[float]
    product_distance_residual: float
    formed_latch_count: int
    broken_latch_count: int
    initial_latched_formed: int
    initial_latched_broken: int


def resolve_formed_thresholds(atoms, formed, override) -> list[float]:
    if not formed:
        return []
    if override is None:
        cov = cordero_radii_for_atoms(atoms)
        return [(cov[i] + cov[j]) * COVALENT_FORMED_TOLERANCE for (i, j) in formed]
    return _v9_broadcast_threshold(override, len(formed), key="r_formed_threshold")


def resolve_broken_thresholds(atoms, broken, override) -> list[float]:
    if not broken:
        return []
    if override is None:
        raise ConfigError(
            "r_broken_threshold required when broken bonds are specified"
        )
    return _v9_broadcast_threshold(override, len(broken), key="r_broken_threshold")


def _v9_broadcast_threshold(value, n: int, *, key: str) -> list[float]:
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ConfigError(f"{key} list length {len(value)} != n_pairs {n}")
        return [float(v) for v in value]
    return [float(value)] * n


def reached_product_v9(
    final_atoms,
    formed: list[tuple[int, int]],
    broken: list[tuple[int, int]],
    formed_thresholds: list[float],
    broken_thresholds: list[float],
) -> bool:
    pos = final_atoms.positions
    for (i, j), thr in zip(formed, formed_thresholds, strict=True):
        d = float(np.linalg.norm(pos[j] - pos[i]))
        if d > thr:
            return False
    for (i, j), thr in zip(broken, broken_thresholds, strict=True):
        d = float(np.linalg.norm(pos[j] - pos[i]))
        if d < thr:
            return False
    return True


def product_distance_residual(
    atoms,
    formed: list[tuple[int, int]],
    broken: list[tuple[int, int]],
    formed_thresholds: list[float],
    broken_thresholds: list[float],
) -> float:
    pos = atoms.positions
    res = 0.0
    for (i, j), thr in zip(formed, formed_thresholds, strict=True):
        d = float(np.linalg.norm(pos[j] - pos[i]))
        res += max(d - thr, 0.0) ** 2
    for (i, j), thr in zip(broken, broken_thresholds, strict=True):
        d = float(np.linalg.norm(pos[j] - pos[i]))
        res += max(thr - d, 0.0) ** 2
    return res


def count_initial_latched(
    atoms,
    formed: list[tuple[int, int]],
    broken: list[tuple[int, int]],
    formed_thresholds: list[float],
    broken_thresholds: list[float],
) -> dict[str, int]:
    pos = atoms.positions
    n_f = sum(
        1 for (i, j), thr in zip(formed, formed_thresholds, strict=True)
        if float(np.linalg.norm(pos[j] - pos[i])) <= thr
    )
    n_b = sum(
        1 for (i, j), thr in zip(broken, broken_thresholds, strict=True)
        if float(np.linalg.norm(pos[j] - pos[i])) >= thr
    )
    return {"formed": n_f, "broken": n_b}


def score_trials_v9(results: list[TrialResultV9]) -> TrialResultV9:
    if not results:
        raise ValueError("score_trials_v9 called with empty list")
    reached = [r for r in results if r.reached_product]
    if reached:
        return min(reached, key=lambda r: r.peak_energy)
    return min(results, key=lambda r: (r.product_distance_residual, r.peak_energy))
```

- [ ] **Step 4: Run (PASS)** — `pytest tests/test_scoring.py -v`

- [ ] **Step 5: Full unit green** — `pytest -m "not slow and not blender" -q`

- [ ] **Step 6: Commit**

```bash
git add reactx/scoring.py tests/test_scoring.py
git commit -m "feat(scoring): add Phase 9 per-pair threshold + residual + latch counts (Phase 9 step 2A.2)"
```

### Task 2A.3: path_relax.py を 3-tuple 戻り値に + `_snapshot` constraint clear

**Files:** Modify `reactx/path_relax.py` and `tests/test_path_relax.py`

**Note**: 旧 `relax_with_restraints` の戻り値を 2-tuple → 3-tuple に **直接変更** する。これは **既存 cli.py を壊す** が、cli.py は Stage 2B で同時に更新するため、Stage 2A 終了時点の repo green は `pytest -m "not slow and not blender"` で確認する。後で Stage 2B が cli.py を更新するまでの間、`tests/test_cli.py` の destructuring が壊れる可能性があるので、test_cli.py の関連 destructuring も先回りで `frames, energies, _ = ...` パターンにする (cli.py 自体は Stage 2B まで触らない、ただし `relax_with_restraints` を呼ぶ test 側は 3-tuple に対応済にする)。

実態として、`relax_with_restraints` の direct callers は cli.py の trial loop と test_path_relax.py だけ。test_path_relax.py を 3-tuple 対応に、test_cli.py は cli.py 経由なので Stage 2B で一気に更新可能。

- [ ] **Step 1: Write failing tests** — append to `tests/test_path_relax.py`:

既存 tests の `frames, energies = relax_with_restraints(...)` を **すべて** `frames, energies, _ = relax_with_restraints(...)` に変更し、末尾に新規 tests:

```python
import numpy as np
from ase import Atoms
from ase.calculators.lj import LennardJones

from reactx.artificial_force import AFIRConstraint
from reactx.path_relax import relax_with_restraints


def test_relax_returns_three_tuple_with_constraint_state():
    a = Atoms("CC", positions=[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    c = AFIRConstraint(
        formed=[(0, 1)], broken=[],
        alpha_formed=[0.5], alpha_broken=[],
        formed_thresholds=[1.5], broken_thresholds=[],
    )
    frames, energies, state = relax_with_restraints(
        a, [c], LennardJones(),
        max_steps=20, fmax=0.5, traj_stride=2,
    )
    assert isinstance(state, dict)
    assert "formed_latched" in state
    assert isinstance(state["formed_latched"], list)
    assert len(state["formed_latched"]) == 1


def test_relax_returns_empty_state_when_no_afir():
    a = Atoms("CC", positions=[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    _, _, state = relax_with_restraints(
        a, [], LennardJones(), max_steps=5, fmax=0.5, traj_stride=2,
    )
    assert state == {}


def test_snapshot_strips_constraint():
    a = Atoms("CC", positions=[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    c = AFIRConstraint(
        formed=[(0, 1)], broken=[],
        alpha_formed=[0.5], alpha_broken=[],
        formed_thresholds=[1.5], broken_thresholds=[],
    )
    frames, _, _ = relax_with_restraints(
        a, [c], LennardJones(),
        max_steps=5, fmax=0.5, traj_stride=2,
    )
    for f in frames:
        assert len(f.constraints) == 0
```

- [ ] **Step 2: Run (FAIL — current `relax_with_restraints` returns 2-tuple)**

- [ ] **Step 3: Update `reactx/path_relax.py`** to 3-tuple + snapshot fix:

```python
"""Constrained relaxation that yields a trajectory of frames.

Phase 9: returns 3-tuple (frames, energies, final_constraint_state).
The third element captures AFIRConstraint latch state at relax end —
read via `atoms.constraints[0]` so it works regardless of whether ASE
copies constraint instances internally. `_snapshot()` also strips
constraints so latch state never leaks into trajectory.xyz.
"""
from __future__ import annotations

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize import FIRE


def relax_with_restraints(
    atoms: Atoms,
    restraints: list,
    calc: Calculator,
    *,
    max_steps: int = 100,
    fmax: float = 0.1,
    traj_stride: int = 5,
) -> tuple[list[Atoms], list[float], dict]:
    atoms = atoms.copy()
    atoms.calc = calc
    if restraints:
        atoms.set_constraint(restraints)

    frames: list[Atoms] = [_snapshot(atoms)]
    energies: list[float] = [float(atoms.get_potential_energy())]

    opt = FIRE(atoms, logfile=None, dt=0.05, a=0.1, maxstep=0.1, dtmax=0.2)

    def _record():
        frames.append(_snapshot(atoms))
        energies.append(float(atoms.get_potential_energy()))

    opt.attach(_record, interval=traj_stride)
    opt.run(fmax=fmax, steps=max_steps)
    step_count = opt.nsteps

    if step_count % traj_stride != 0 or step_count == 0:
        _record()

    final_state: dict = {}
    if atoms.constraints:
        c = atoms.constraints[0]
        if hasattr(c, "formed_latched"):
            final_state = {
                "formed_latched": list(c.formed_latched),
                "broken_latched": list(c.broken_latched),
            }

    return frames, energies, final_state


def _snapshot(atoms: Atoms) -> Atoms:
    a = atoms.copy()
    a.calc = None
    a.set_constraint([])
    return a
```

- [ ] **Step 4: Update `reactx/cli.py` `relax_with_restraints` call site to receive 3-tuple**

`reactx/cli.py` の line 265 付近:

```python
# 旧:
frames, energies = relax_with_restraints(
    atoms_init, restraints, calc, ...
)
# 新:
frames, energies, _ = relax_with_restraints(
    atoms_init, restraints, calc, ...
)
```

(これは 1 行修正で、cli の他の挙動には影響しない — Stage 2A 終了時点で cli.py は **旧 build_restraints の Hookean+PullApart 経路で動き続ける**、`_` は dict だが旧 Hookean は `formed_latched` 属性を持たないので空 dict)

- [ ] **Step 5: Run unit suite — full green** — `pytest -m "not slow and not blender" -q`

- [ ] **Step 6: Commit**

```bash
git add reactx/path_relax.py reactx/cli.py tests/test_path_relax.py
git commit -m "feat(path_relax): return 3-tuple with constraint state, strip constraint in snapshot (Phase 9 step 2A.3)"
```

### Stage 2A 完了 checkpoint

- 新 API (AFIRConstraint, build_afir_constraint, ConfigError, AFIRSection, ScoringSection, ReactionConfigV9, load_config_v9, TrialResultV9, reached_product_v9, score_trials_v9, product_distance_residual, count_initial_latched, resolve_*_thresholds, resolve_alpha_*_v9) すべて追加済
- 旧 API (RestraintConfig, load_config, TrialResult, reached_product, score_trials, build_restraints, Hookean, PullApart, DEFAULT_R_FORM, lookup_r_form) も **そのまま** 動作
- cli.py は依然として旧 API で動作 (relax_with_restraints の戻り値だけ 3-tuple 対応)
- `pytest -m "not slow and not blender"` green
- `pytest -m slow` 既存通り pass する想定 (cli flow は実質変更なし)

---

## Stage 2B: cli.py を新 API に switch + 8 example TOMLs migration + 既存 tests update

### Task 2B.1: cli.py を新 API に switch

**Files:** Modify `reactx/cli.py`

- [ ] **Step 1: Update imports** (旧 `build_restraints`, `RestraintConfig`, `resolve_k_*`, `resolve_r_*` import を削除し、新 API import に置換):

`reactx/cli.py` line 14-36 を以下に置き換え:

```python
from reactx.align import align_product_to_reactant
from reactx.artificial_force import build_afir_constraint
from reactx.bond_changes import BondChanges
from reactx.calculators import make_calculator
from reactx.config import (
    ConfigError,
    ReactionConfigV9 as ReactionConfig,
    load_config_v9 as load_config,
    resolve_alpha_broken_v9 as resolve_alpha_broken,
    resolve_alpha_formed_v9 as resolve_alpha_formed,
)
from reactx.embed3d import embed_fragments_to_positions
from reactx.neb import run_neb
from reactx.path_relax import relax_with_restraints
from reactx.placement import (
    PlacementResult,
    build_atoms_from_positions,
    valid_placements,
)
from reactx.rxn_parser import atom_map_to_reactant_idx, heavy_to_hydrogen_groups, parse_rxn
from reactx.scoring import (
    TrialResultV9 as TrialResult,
    count_initial_latched,
    product_distance_residual,
    reached_product_v9 as reached_product,
    resolve_broken_thresholds,
    resolve_formed_thresholds,
    score_trials_v9 as score_trials,
)
```

(alias で旧名を維持 → cli.py 内部の参照を最小限に)

- [ ] **Step 2: Replace `_cmd_run` orchestration block** (line 198-292 付近):

旧 `formed_pairs = ...` から trial loop の終わりまで以下に置換:

```python
    formed_pairs = list(bond_changes.formed)
    broken_pairs = list(bond_changes.broken)

    log.info(
        "description=%s effective: alpha_formed=%s alpha_broken=%s "
        "max_relax_steps=%d",
        cfg.description,
        cfg.afir.alpha_formed, cfg.afir.alpha_broken,
        cfg.afir.max_relax_steps,
    )

    model_kwargs = {"model_name": args.model} if args.backend == "uma" else {}
    calc = make_calculator(args.backend, **model_kwargs)

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
    except NotImplementedError as exc:
        log.error("placement not supported: %s", exc)
        return 2
    except (RuntimeError, ValueError) as exc:
        log.error("placement failed: %s", exc)
        return 1

    log.info(
        "placement: %d/%d candidates survived blocking (%d blocked)",
        len(placement.trials), placement.n_candidates, placement.n_blocked,
    )

    af = resolve_alpha_formed(cfg)
    ab = resolve_alpha_broken(cfg)

    trials: list[TrialResult] = []
    for i, t in enumerate(placement.trials):
        atoms_init = build_atoms_from_positions(mol_h_r, t.positions)
        log.info("trial %d (direction=%s)",
                 i, np.array2string(t.direction, precision=3))

        ft = resolve_formed_thresholds(
            atoms_init, formed_pairs, cfg.scoring.r_formed_threshold,
        )
        bt = resolve_broken_thresholds(
            atoms_init, broken_pairs, cfg.scoring.r_broken_threshold,
        )

        initial_latched = count_initial_latched(
            atoms_init, formed_pairs, broken_pairs, ft, bt,
        )

        afir_cs = build_afir_constraint(
            atoms_init, formed_pairs, broken_pairs,
            alpha_formed=af, alpha_broken=ab,
            formed_thresholds=ft, broken_thresholds=bt,
        )

        try:
            frames, energies, final_state = relax_with_restraints(
                atoms_init, afir_cs, calc,
                max_steps=cfg.afir.max_relax_steps,
                fmax=args.relax_fmax,
                traj_stride=args.traj_stride,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("trial %d relax failed: %s", i, exc)
            trials.append(TrialResult(
                trial_idx=i, direction=t.direction, frames=[], energies=[],
                reached_product=False, peak_energy=float("inf"), n_steps=0,
                formed_thresholds=ft, broken_thresholds=bt,
                product_distance_residual=float("inf"),
                formed_latch_count=0, broken_latch_count=0,
                initial_latched_formed=initial_latched["formed"],
                initial_latched_broken=initial_latched["broken"],
            ))
            continue

        ok = reached_product(frames[-1], formed_pairs, broken_pairs, ft, bt)
        residual = product_distance_residual(
            frames[-1], formed_pairs, broken_pairs, ft, bt,
        )
        peak = max(energies) if energies else float("inf")
        formed_latch_count = sum(final_state.get("formed_latched", []))
        broken_latch_count = sum(final_state.get("broken_latched", []))

        trials.append(TrialResult(
            trial_idx=i, direction=t.direction,
            frames=frames, energies=energies,
            reached_product=ok, peak_energy=float(peak),
            n_steps=len(frames),
            formed_thresholds=ft, broken_thresholds=bt,
            product_distance_residual=residual,
            formed_latch_count=formed_latch_count,
            broken_latch_count=broken_latch_count,
            initial_latched_formed=initial_latched["formed"],
            initial_latched_broken=initial_latched["broken"],
        ))
```

- [ ] **Step 3: Update `_write_outputs_and_exit` signature + body** — `r_form_targets` 引数を削除し、per-trial dict と `effective_params` を新 schema に:

旧 line 375-445 全体を以下に置き換え:

```python
def _write_outputs_and_exit(
    args: argparse.Namespace,
    trials: list[TrialResult],
    t_start: float,
    *,
    neb_refined: bool,
    rc: int,
    cfg: ReactionConfig | None = None,
    placement: PlacementResult | None = None,
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
            "placement_kind": placement.placement_kind,
        }
        if placement is not None
        else {"n_candidates": 0, "n_blocked": 0, "n_valid": 0, "placement_kind": None}
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
                "orientation": (
                    placement.trials[t.trial_idx].orientation
                    if placement is not None and t.trial_idx < len(placement.trials)
                    else "single"
                ),
                "formed_thresholds": [float(x) for x in t.formed_thresholds],
                "broken_thresholds": [float(x) for x in t.broken_thresholds],
                "product_distance_residual": (
                    float(t.product_distance_residual)
                    if math.isfinite(t.product_distance_residual) else None
                ),
                "formed_latch_count": t.formed_latch_count,
                "broken_latch_count": t.broken_latch_count,
                "initial_latched_formed": t.initial_latched_formed,
                "initial_latched_broken": t.initial_latched_broken,
            }
            for t in trials
        ],
        "wall_clock_seconds": float(time.monotonic() - t_start),
        "neb_refined": neb_refined,
        "effective_params": (
            {
                "alpha_formed": _scalar_or_list(cfg.afir.alpha_formed),
                "alpha_broken": _scalar_or_list(cfg.afir.alpha_broken),
                "max_relax_steps": cfg.afir.max_relax_steps,
                "r_broken_threshold": (
                    _scalar_or_list(cfg.scoring.r_broken_threshold)
                    if cfg.scoring.r_broken_threshold is not None else None
                ),
                "r_formed_threshold": (
                    _scalar_or_list(cfg.scoring.r_formed_threshold)
                    if cfg.scoring.r_formed_threshold is not None else None
                ),
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

- [ ] **Step 4: Update `_write_outputs_and_exit` call sites** — `r_form_targets=...` 引数削除:

```python
# 旧:
return _write_outputs_and_exit(
    args, trials, t_start, neb_refined=False, rc=1,
    cfg=cfg, r_form_targets=r_form_targets, placement=placement,
)
# 新:
return _write_outputs_and_exit(
    args, trials, t_start, neb_refined=False, rc=1,
    cfg=cfg, placement=placement,
)
```

(call sites: 旧 line 296-299, 359-362)

- [ ] **Step 5: Add ConfigError catch at `load_config(args.rxn_path)` site**

`load_config` 呼び出し箇所 (cli.py の `_cmd_run` 冒頭) を以下に変更:

```python
try:
    cfg = load_config(args.rxn_path)
except ConfigError as e:
    log.error("config error: %s", e)
    return 2
except FileNotFoundError as e:
    log.error("%s", e)
    return 2
```

(注: ConfigError は ValueError サブクラスなので、もし既存 `except ValueError` があれば ConfigError を **先** に置く)

- [ ] **Step 6: 重要:** この時点で `pytest -m "not slow and not blender"` は **失敗する** (8 example TOMLs と inline toml_body が旧 schema のまま)。Step 7 で全部更新するのでそのまま進む。

#### Step 7: 8 example TOMLs を新 schema に migration

##### `examples/sn2.rxn.toml`:
```toml
description = "SN2 anion: CH3Cl + OH- -> CH3OH + Cl-"
formed = [[1, 3]]
broken = [[1, 2]]

[afir]
alpha_formed = 0.7
alpha_broken = 0.7
max_relax_steps = 100

[scoring]
r_broken_threshold = 4.0
```

##### `examples/proton_transfer.rxn.toml`:
```toml
description = "Proton transfer: HCl + NH3 -> Cl- + NH4+"
formed = [[1, 3]]
broken = [[1, 2]]

[afir]
alpha_formed = 1.5
alpha_broken = 1.0
max_relax_steps = 100

[scoring]
r_broken_threshold = 4.0
r_formed_threshold = 1.5
```

##### `examples/menshutkin.rxn.toml`:
```toml
description = "Menshutkin: NH3 + CH3Cl -> CH3NH3+ + Cl-"
formed = [[1, 5]]
broken = [[5, 9]]

[afir]
alpha_formed = 4.0
alpha_broken = 1.5
max_relax_steps = 200

[scoring]
r_broken_threshold = 5.0
```

##### `examples/e2.rxn.toml`:
```toml
description = "E2 elimination: CH3CH2Cl + OH- -> CH2=CH2 + Cl- + H2O"
formed = [[4, 5]]
broken = [[2, 5], [1, 3]]

[afir]
alpha_formed = 1.5
alpha_broken = [1.0, 1.5]
max_relax_steps = 200

[scoring]
r_broken_threshold = [3.0, 4.0]
```

##### `examples/sn1_dissoc.rxn.toml` (formed=[]、alpha_formed 省略):
```toml
description = "SN1 step 1 dissociation: tBuBr -> tBu+ + Br-"
formed = []
broken = [[1, 5]]

[afir]
alpha_broken = 1.5
max_relax_steps = 200

[scoring]
r_broken_threshold = 6.0
```

##### `examples/sn1_recomb.rxn.toml` (broken=[]、alpha_broken 省略):
```toml
description = "SN1 step 2 recombination: tBu+ + Cl- -> tBuCl"
formed = [[1, 5]]
broken = []

[afir]
alpha_formed = 1.5
max_relax_steps = 200
```

##### `examples/diels_alder_simple.rxn.toml`:
```toml
description = "Diels-Alder: butadiene + ethylene -> cyclohexene"
formed = [[1, 5], [4, 6]]
broken = []

[afir]
alpha_formed = [2.5, 2.5]
max_relax_steps = 200

[sampling]
n_candidates = 64
```

##### `examples/diels_alder_endo.rxn.toml`:
```toml
description = "Diels-Alder endo: cyclopentadiene + maleic anhydride -> norbornene-2,3-dicarboxylic anhydride"
formed = [[1, 5], [4, 6]]
broken = []

[afir]
alpha_formed = [2.5, 2.5]
max_relax_steps = 250

[sampling]
n_candidates = 16
```

#### Step 8: tests の inline `toml_body` を新 schema に + TrialResult 14-field

##### `tests/conftest.py` docstring:
`tmp_rxn_with_toml` の docstring 例を新 schema に書き換え (line 130-142):

```python
    """Copy `examples/<stem>.rxn` to tmp_path, write a fresh sidecar TOML.

    Usage:
        rxn_path = tmp_rxn_with_toml("sn2", toml_body='''\\
            description = "sn2 fast"
            formed = [[1, 3]]
            broken = [[1, 2]]
            [afir]
            alpha_formed = 0.5
            alpha_broken = 1.0
            max_relax_steps = 30
            [scoring]
            r_broken_threshold = 4.0
            [sampling]
            n_candidates = 1
        ''')
    ...
    """
```

##### `tests/conftest.py` に safety assertion helper を追加 (Stage 3 で使用):

ファイル末尾に追記:

```python
def assert_min_nonbonded_distance_ok(
    frames, *, formed, broken, threshold: float = 0.5
) -> None:
    """Assert that across all `frames`, no pair of atoms gets closer than
    `threshold` (Å) — except (a) initial-bonded pairs (any pair within
    1.1 × covalent_sum at frame 0), (b) `formed` pairs, (c) `broken` pairs.

    Used in slow integration tests as a UMA off-manifold guard.
    """
    import numpy as np
    from reactx.covalent_radii import cordero_radii_for_atoms

    if not frames:
        return
    first = frames[0]
    cov = cordero_radii_for_atoms(first)
    n = len(first)
    initial_bonded = set()
    pos0 = first.positions
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(pos0[j] - pos0[i]))
            if d <= 1.1 * (cov[i] + cov[j]):
                initial_bonded.add((i, j))
    reactive = set()
    for (i, j) in formed:
        reactive.add((min(i, j), max(i, j)))
    for (i, j) in broken:
        reactive.add((min(i, j), max(i, j)))

    excluded = initial_bonded | reactive

    for frame_idx, frame in enumerate(frames):
        pos = frame.positions
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) in excluded:
                    continue
                d = float(np.linalg.norm(pos[j] - pos[i]))
                assert d >= threshold, (
                    f"frame {frame_idx}: non-bonded pair ({i},{j}) "
                    f"distance {d:.3f} < {threshold} Å (UMA off-manifold?)"
                )
```

##### `tests/test_cli.py` の `toml_body` を新 schema に + `TrialResult(...)` を 14-field 化:

`grep -n "k_form\|TrialResult(" tests/test_cli.py` で line 確認。各 toml_body を:

```python
body = """\
description = "sn2 fast"
formed = [[1, 3]]
broken = [[1, 2]]

[afir]
alpha_formed = 0.5
alpha_broken = 1.0
max_relax_steps = 30

[scoring]
r_broken_threshold = 4.0

[sampling]
n_candidates = 1
"""
```

各 `TrialResult(...)` を:

```python
TrialResult(
    trial_idx=0,
    direction=np.array([0.0, 0.0, 1.0]),
    frames=[], energies=[],
    reached_product=True, peak_energy=2.0, n_steps=2,
    formed_thresholds=[1.6],
    broken_thresholds=[4.0],
    product_distance_residual=0.0,
    formed_latch_count=1, broken_latch_count=1,
    initial_latched_formed=0, initial_latched_broken=0,
)
```

(具体的な thresholds 値は test の意図に合わせる)

##### `tests/test_cli_unimolecular.py` の `toml_body`:

```python
body = """\
description = "sn1 dissoc fast"
formed = []
broken = [[1, 5]]

[afir]
alpha_broken = 1.0
max_relax_steps = 30

[scoring]
r_broken_threshold = 4.0
"""
```

##### `tests/test_cli_neb_refine_guard.py` の 3 つの inline TOML:

```python
_E2_TOML = """\
description = "e2 fast"
formed = [[4, 5]]
broken = [[2, 5], [1, 3]]

[afir]
alpha_formed = 1.0
alpha_broken = [1.0, 1.0]
max_relax_steps = 30

[scoring]
r_broken_threshold = [4.0, 4.0]
"""

_SN1_RECOMB_TOML = """\
description = "sn1 recomb fast"
formed = [[1, 5]]
broken = []

[afir]
alpha_formed = 1.0
max_relax_steps = 30
"""

_SN2_TOML = """\
description = "sn2 fast"
formed = [[1, 3]]
broken = [[1, 2]]

[afir]
alpha_formed = 0.5
alpha_broken = 1.0
max_relax_steps = 30

[scoring]
r_broken_threshold = 4.0
"""
```

##### `tests/test_neb_refine_sn2.py` の `_SN2_NEB`:

```python
_SN2_NEB = """\
description = "sn2 neb"
formed = [[1, 3]]
broken = [[1, 2]]

[afir]
alpha_formed = 0.5
alpha_broken = 1.0
max_relax_steps = 100

[scoring]
r_broken_threshold = 4.0
"""
```

##### `tests/test_wallclock_sn2.py` の `toml_body` を上記 SN2 形式に + meta assertion で `effective_params.alpha_formed` を読む形に。

##### `tests/test_diels_alder_endo.py`:
`grep -n "TrialResult(" tests/test_diels_alder_endo.py` で確認、各 constructor を 14-field に。

##### `tests/test_blender_smoke.py`:
`grep -n "k_form\|\\[restraints\\]" tests/test_blender_smoke.py` で確認、ヒットがあれば対応 toml_body を新 schema に。

##### slow tests (`tests/test_re*.py`, `tests/test_examples_menshutkin.py`):

各 file で `r_broken_target` / `r_form_target` / `k_form` / `k_broken` / `effective_params.k_form` / `effective_params.k_broken` を読んでいる assertion を:

- `effective_params.alpha_formed` / `effective_params.alpha_broken` / `effective_params.r_broken_threshold` / `effective_params.r_formed_threshold` の参照に
- per-trial `formed_thresholds` / `broken_thresholds` / `product_distance_residual` の参照に

`grep -n "k_form\|k_broken\|r_broken_target\|r_form_target\|effective_params" tests/test_re*.py tests/test_examples_*.py` で各箇所を find → 対応する新 keys に置換。

#### Step 9: `tests/test_config.py` を 8 examples の load smoke に置換

ファイル全置換:

```python
"""Smoke: each example TOML must validate after Phase 9 migration."""
from pathlib import Path

import pytest

from reactx.config import load_config_v9 as load_config

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.mark.parametrize("rxn_name", [
    "sn2.rxn", "proton_transfer.rxn", "menshutkin.rxn", "e2.rxn",
    "sn1_dissoc.rxn", "sn1_recomb.rxn",
    "diels_alder_simple.rxn", "diels_alder_endo.rxn",
])
def test_example_loads(rxn_name: str):
    cfg = load_config(EXAMPLES / rxn_name)
    assert cfg.description
    assert cfg.afir.max_relax_steps > 0
```

#### Step 10: 最終 verify + commit

- [ ] **Step 11: Run unit + non-slow tests — must be green**

```bash
pytest -m "not slow and not blender" -q
```
Expected: all green. fail があれば該当 file を fix。

- [ ] **Step 12: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
feat(phase-9): switch cli to AFIR with sticky latch + per-pair thresholds

- cli.py: build_afir_constraint, threshold resolution shared between AFIR
  latch and reached_product, ConfigError catch, 14-field TrialResult on
  success and failure paths, _write_outputs_and_exit signature drops
  r_form_targets, meta.json schema includes formed_thresholds /
  broken_thresholds / product_distance_residual / latch counts
- 8 examples/*.rxn.toml: migrated to [afir] + [scoring]
- tests: updated all inline toml_body strings (test_cli, test_cli_unimolecular,
  test_cli_neb_refine_guard, test_neb_refine_sn2, test_wallclock_sn2,
  conftest docstring), 14-field TrialResult constructors
  (test_cli, test_diels_alder_endo), slow test meta assertions migrated
  to new keys
- conftest.py: assert_min_nonbonded_distance_ok helper for Stage 3
- tests/test_config.py: example load smoke

Spec: docs/superpowers/specs/2026-05-08-afir-force-design.md (v3.1)
Plan: docs/superpowers/plans/2026-05-08-afir-force-replacement.md (v3)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Stage 2B 完了 checkpoint

- cli.py が新 API を使用、8 example TOMLs と既存 tests inline toml_body すべて新 schema
- 旧 API (Hookean / PullApart / build_restraints / DEFAULT_R_FORM / lookup_r_form / RestraintConfig / load_config / TrialResult / reached_product / score_trials) は **未削除でも誰からも呼ばれていない**
- `pytest -m "not slow and not blender"` green

---

## Stage 2C: Legacy cleanup

### Task 2C.1: 旧 API を削除

**Files:**
- Modify: `reactx/artificial_force.py` (Hookean / PullApart / build_restraints / DEFAULT_R_FORM / lookup_r_form / `_broadcast` / `_broadcast_r_form` を削除)
- Modify: `reactx/config.py` (旧 `RestraintConfig` / `_validate` / `load_config` / `_build_restraints` / `_normalize_*` / `_to_pair_tuple` (旧版) / `_check_keys` (旧版) / `_TOP_LEVEL_KEYS` 等 旧定数 / `resolve_k_form_targets` / `resolve_k_broken_targets` / `resolve_r_broken_targets` / `resolve_r_form_targets` を削除し、`load_config_v9` を `load_config` に rename、`ReactionConfigV9` を `ReactionConfig` に rename、`_v9_*` プレフィックスを削除)
- Modify: `reactx/scoring.py` (旧 `TrialResult` / `reached_product` / `score_trials` / `select_best_trial` を削除し、`TrialResultV9` を `TrialResult` に、`reached_product_v9` を `reached_product` に、`score_trials_v9` を `score_trials` に rename)
- Modify: `reactx/cli.py` (alias import を削除し、新名で直接 import)
- Modify: `tests/test_artificial_force.py` (smoke のみに)
- Delete (もし存在): `tests/test_pull_apart.py`

- [ ] **Step 1: `reactx/artificial_force.py` から legacy 削除**

ファイル冒頭の `DEFAULT_R_FORM`, `lookup_r_form`, `Hookean` (関連 import), `PullApart`, `build_restraints`, `_broadcast`, `_broadcast_r_form` をすべて削除。残すのは AFIRConstraint と build_afir_constraint のみ。

新 module docstring:

```python
"""Per-pair AFIR force with sticky per-pair latch (Phase 9).

Replaces Phase Re1's Hookean+PullApart hybrid. Each user-specified pair
gets an independent constant-magnitude force (`F = α · d̂`) gated by a
sticky latch on a per-pair distance threshold.

Spec: docs/superpowers/specs/2026-05-08-afir-force-design.md §4.1
"""
```

- [ ] **Step 2: `reactx/config.py` を全面 cleanup**

旧 `_TOP_LEVEL_KEYS` / `_RESTRAINTS_KEYS` 等 を削除。`RestraintConfig` / `_validate` / `load_config` / `_build_restraints` / `_normalize_k_form` / 等を削除。`ReactionConfig` (旧版) を削除。

`_v9_*` プレフィックスをすべて除去 (例: `_v9_normalize_alpha` → `_normalize_alpha`、`_v9_check_keys` → `_check_keys`)。`load_config_v9` を `load_config` に、`ReactionConfigV9` を `ReactionConfig` に、`resolve_alpha_*_v9` を `resolve_alpha_*` に rename。

新 module docstring:

```python
"""Per-reaction sidecar TOML config (`<rxn_path>.toml`).

Phase 9 schema: [afir] + [scoring] sections.
Spec: docs/superpowers/specs/2026-05-08-afir-force-design.md §4.4
"""
```

- [ ] **Step 3: `reactx/scoring.py` を全面 cleanup**

旧 `TrialResult` / `reached_product` / `score_trials` / `select_best_trial` を削除。`TrialResultV9` を `TrialResult` に、`reached_product_v9` を `reached_product` に、`score_trials_v9` を `score_trials` に rename。`select_best_trial` を新 score_trials を使うように再追加 (cli が使っているか確認、使用なら削除可)。

新 module docstring:

```python
"""Trial scoring + final-best selection (Phase 9).

Spec: docs/superpowers/specs/2026-05-08-afir-force-design.md §4.5
"""
```

- [ ] **Step 4: `reactx/cli.py` の alias import を簡素化**

旧 alias を取り除いて直接 import:

```python
from reactx.config import (
    ConfigError,
    ReactionConfig,
    load_config,
    resolve_alpha_broken,
    resolve_alpha_formed,
)
from reactx.scoring import (
    TrialResult,
    count_initial_latched,
    product_distance_residual,
    reached_product,
    resolve_broken_thresholds,
    resolve_formed_thresholds,
    score_trials,
)
```

- [ ] **Step 5: `tests/test_artificial_force.py` を smoke のみに**

ファイル全置換:

```python
"""Smoke test for the AFIR force module — substantive tests live in
`tests/test_afir_constraint.py`."""
from reactx.artificial_force import AFIRConstraint, build_afir_constraint


def test_module_exports_public_api():
    assert AFIRConstraint is not None
    assert callable(build_afir_constraint)
```

- [ ] **Step 6: `tests/test_pull_apart.py` 削除 (もし存在)**

```bash
rm -f tests/test_pull_apart.py
```

- [ ] **Step 7: tests の `_v9` import alias を整理**

`tests/test_config_afir.py`、`tests/test_scoring.py`、`tests/test_cli.py` 等の `from reactx.config import load_config_v9` 等の `_v9` 名を `load_config` 等に変更 (rename 完了後の正式名)。

`grep -rn "_v9\b\|TrialResultV9\|ReactionConfigV9\|reached_product_v9\|score_trials_v9\|load_config_v9\|resolve_alpha_.*_v9" reactx/ tests/`
Expected: zero hits after this step.

- [ ] **Step 8: 確認 + commit**

```bash
pytest -m "not slow and not blender" -q
```
Expected: green.

```bash
git add -A
git rm -f tests/test_pull_apart.py 2>/dev/null || true
git commit -m "chore(phase-9): drop legacy Hookean/PullApart/RestraintConfig and rename _v9 to canonical (Phase 9 step 2C.1)"
```

### Stage 2C 完了 checkpoint

- 旧 API は完全に消失
- 新 API は `_v9` プレフィックスなしで canonical 名前
- `pytest -m "not slow and not blender"` green

---

## Stage 3: Per-reaction slow integration tuning + safety assertion

各 task で対応 reaction の slow test を走らせ、`reached_product=True` が selected_trial の最終 frame で出ることを確認、min 非結合距離 ≥ 0.5 Å の safety assertion を **追加** し、必要なら α / threshold を再 tune する。順序は wall-clock 短い順。

### Task 3.1: SN1 dissoc

**Files:** Modify `tests/test_re3_sn1_dissoc.py`

- [ ] **Step 1: Update meta assertion** — `meta["effective_params"]["alpha_broken"]` 等を読む形に。`reached_product` 主導は維持。

- [ ] **Step 2: Add safety assertion**

末尾の test 関数に、selected_trial の frames に対して safety check を追加:

```python
from tests.conftest import assert_min_nonbonded_distance_ok

# ... 既存の assertion 内、selected_trial の frames を取得した後:
selected_idx = meta["selected_trial"]
# Read trajectory from out/sn1d/trajectory.xyz
from ase.io import read
traj = read(str(args.output / "trajectory.xyz"), index=":")
formed = [tuple(p) for p in cfg.formed]   # or read from meta
broken = [tuple(p) for p in cfg.broken]
assert_min_nonbonded_distance_ok(traj, formed=formed, broken=broken)
```

(具体的な変数 access pattern は既存 test の structure に合わせる。`cfg` が直接取れない場合は `meta` から `formed` / `broken` を再 parse するか、既存 test の per-test fixture に依存。)

- [ ] **Step 3: Run** — `pytest tests/test_re3_sn1_dissoc.py -v -m slow 2>&1 | tail -40`

- [ ] **Step 4: 失敗時の re-tune (α 増を第一選択)**:
  - `selected_trial.reached_product=False` → `alpha_broken` 1.5 → 2.0
  - α 増が効かない場合のみ `r_broken_threshold` 6.0 → 5.0 (緩和、最後の手段)
  - safety assertion fail (min 非結合距離 < 0.5 Å) → α を **下げる** (過圧縮 prevention)

- [ ] **Step 5: PASS 後 commit** (changes があれば)

### Task 3.2-3.8: SN2, Proton transfer, SN1 recomb, Menshutkin, E2, Diels-Alder simple, Diels-Alder endo

各 reaction について Task 3.1 と同パターン:

1. meta assertion を新 schema に更新
2. `assert_min_nonbonded_distance_ok` safety check を追加
3. slow test 実行
4. 失敗時は α 増 → threshold 緩和の順で re-tune
5. PASS 後 commit

具体的 tuning ガイド:

| Reaction | 失敗時の re-tune (α 増を第一選択) |
|---|---|
| SN2 | `alpha_formed` 0.7 → 1.0, `alpha_broken` 0.7 → 1.0 |
| Proton transfer | `alpha_formed` 1.5 → 2.0 |
| SN1 recomb | `alpha_formed` 1.5 → 2.0 |
| Menshutkin | `alpha_formed` 4.0 → 5.0 (gas-phase で最強の力が必要) |
| E2 | false-positive (transient C-H 距離が threshold を一瞬越えて latch、最終 frame で reached_product=False) → `r_broken_threshold = [3.0, 4.0]` を `[3.5, 4.0]` に厳しく; C-Cl 切断不足 → `alpha_broken[1]` 1.5 → 2.0 |
| DA simple | `alpha_formed = [2.5, 2.5]` → `[3.0, 3.0]` |
| DA endo | `alpha_formed` 増、`max_relax_steps` 250 → 300 |

複数 reaction で同じ tuning が必要な場合は、heuristic として default を見直す検討を Open Questions に追記する (本 phase では per-reaction 個別値を維持)。

### Task 3.9: Full slow-test sweep

- [ ] **Step 1: Run all** — `pytest -m slow -q 2>&1 | tail -30`
- [ ] **Step 2: 失敗があれば** Stage 3.1-3.8 を回り直す
- [ ] **Step 3: 確認 commit (changes があれば)**

---

## Stage 4: Cleanup + README + PR

### Task 4.1: Legacy reference 確認 + align.py 判断

- [ ] **Step 1: Search legacy refs**

```bash
grep -rn "k_form\|k_broken\|r_form\b\|r_broken\b\|build_restraints\|Hookean\|PullApart\|DEFAULT_R_FORM\|lookup_r_form\|RestraintConfig\|cfg\.restraints\|_v9\b" reactx/ tests/ blender/ 2>&1 | grep -v "^Binary"
```
Expected: zero hits in `reactx/` ソース、`tests/` も `tests/test_artificial_force.py` の smoke のみが残る想定。Hits があれば対応する file を確認・修正。

- [ ] **Step 2: align.py 確認**

```bash
grep -rn "align_product_to_reactant" reactx/ tests/
```
Expected: `reactx/cli.py` の NEB refine 経路 + `tests/test_align.py` のみ。Phase 9 では削除しない。

- [ ] **Step 3: もし residual update があれば commit**

```bash
git add -A
git commit -m "chore: clean up residual legacy references (Phase 9 step 4.1)"
```

### Task 4.2: README rewrite

`README.md` の主な書き換え箇所 (spec §4.4 を反映):

- 冒頭サブタイトル「per-pair AFIR with sticky latch」追記
- Per-reaction `.rxn.toml` config 節:
  - 旧 `k_form` / `k_broken` / `r_broken` / `r_form` 列を削除
  - 新 `alpha_formed` / `alpha_broken` / `r_broken_threshold` / `r_formed_threshold` 列に
  - 最小例 TOML を新 schema に
  - `alpha_*` は `formed=[]` / `broken=[]` 側で省略可と明記
- 「方針と限界」: `[restraints]` 廃止、`[afir]` + `[scoring]` 必須、`r_broken_threshold` は `broken=[]` で省略可、`align_product_to_reactant` の Phase 10 移行予定
- 「Wall-clock (実測)」: Stage 3 完了後の `meta.json.wall_clock_seconds` を反映
- 「アーキテクチャ」: AFIR + sticky latch
- 「対応反応」表: 各 reaction に新 schema パラメータ列を追加

- [ ] **Step 1: Edit README**
- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs(readme): rewrite for Phase 9 AFIR + sticky latch (Phase 9 step 4.2)"
```

### Task 4.3: Phase 9 PR

- [ ] **Step 1: Confirm full test suite passes**

```bash
pytest -q 2>&1 | tail -10
```

- [ ] **Step 2: Push + PR**

```bash
git push -u origin phase-9
gh pr create --base develop --title "Phase 9: per-pair AFIR with sticky latch + 1-stage relax" --body "$(cat <<'EOF'
## Summary
- Replace Hookean + PullApart with per-pair AFIR (constant force, sticky per-pair latch)
- Move scoring threshold to `[scoring]`, share with AFIR latch trigger
- Add `product_distance_residual` for least-bad fallback
- 1-stage relax_with_restraints returns 3-tuple including final_constraint_state
- Migrate 8 example TOMLs to new schema, retune α / threshold via slow tests

Spec: docs/superpowers/specs/2026-05-08-afir-force-design.md (v3.1)
Plan: docs/superpowers/plans/2026-05-08-afir-force-replacement.md (v3)

## Test plan
- [x] Phase 9 unit (`pytest tests/test_afir_constraint.py tests/test_config_afir.py tests/test_scoring.py tests/test_path_relax.py tests/test_covalent_radii.py`)
- [x] Full unit suite (`pytest -m "not slow and not blender"`)
- [x] All 8 reactions slow integration (`pytest -m slow`)
- [x] min non-bonded distance ≥ 0.5 Å safety assertion
- [x] Manual: `reactx run examples/diels_alder_simple.rxn -o out/da/ --backend uma --render`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: PR URL を user に報告**

---

## Self-Review

After completing all stages, verify:

1. **Spec coverage** (§ で確認):
   - §3.3 file structure → 全 file plan に reflected ✓
   - §4.1 AFIRConstraint (per-pair + sticky latch + α=0 重複防御) → Stage 1 Tasks 1.3-1.7 ✓
   - §4.2 path_relax 3-tuple + _snapshot fix → Stage 2A Task 2A.3 ✓
   - §4.3 covalent_radii → Stage 1 Tasks 1.1-1.2 ✓
   - §4.4 config schema (alpha_* 省略 valid、α=0 reject、NaN reject、obsolete keys reject、ConfigError) → Stage 2A Task 2A.1 + Stage 2C Task 2C.1 ✓
   - §4.5 reached_product / residual / count_initial_latched / TrialResult 14-field → Stage 2A Task 2A.2 + Stage 2C Task 2C.1 ✓
   - §4.6 cli orchestration + meta.json + ConfigError catch + 失敗 trial 14-field → Stage 2B Task 2B.1 ✓
   - §5 example tuning table → Stage 2B Step 7 + Stage 3 ✓
   - §6 test strategy → Stage 1 + Stage 2A + Stage 2B Step 8 + Stage 3 (safety assertion) ✓
   - §7 implementation order → Plan stages mirror this ✓
   - §10.3 critical-1 (latch ≠ reached) → Stage 1 Task 1.8 ✓
   - §10.3 critical-2 (α=0 重複防御) → Stage 1 Task 1.5 + Stage 2A Task 2A.1 ✓
   - §6.4 safety assertion (min 非結合距離) → Stage 3 各 task + conftest helper ✓

2. **No placeholders**: 全 code block runnable、TBD/implement-later なし。

3. **Type consistency**: 14-field TrialResult / 3-tuple relax / AFIRConstraint signature 全 task で一致。

4. **Plan v2 → v3 で対応した critical 5 件** (詳細は §改訂履歴):
   - C1/C12: Stage 2 を 2A/2B/2C に分割、各 sub-stage で repo green ✓
   - C5: TrialResult に formed_thresholds / broken_thresholds 追加 (14 fields) ✓
   - C9: `_AFIR_REQUIRED = {"max_relax_steps"}` のみ、alpha_* 省略 valid ✓
   - C11: Stage 3 各 task に safety assertion 追加 ✓

## Open Questions (Phase 10+ で扱う)

- **OQ-1**: `r_formed_threshold = 1.15 × Rsum` vs Blender `1.1 × Rsum` の 0.05 差 → visual で edge case 出たら Blender 側を 1.15 に揃える
- **OQ-2**: `r_broken_threshold` を `broken=[]` で明示時の warning 出力
- **OQ-3**: `align_product_to_reactant` の AFIR endpoint 直接利用
- **OQ-4**: `AFIRConstraint.todict()` round-trip test
- **OQ-5**: Cordero 表共有による Blender ↔ reactx 整合性 unit test (`reactx.covalent_radii.CORDERO_2008 == blender.render.COVALENT_RADII_ANGSTROM`)
- **OQ-6**: 複数 reaction で同じ tuning が必要な場合の default heuristic 見直し

---

## 改訂履歴

### v1 → v2 (post 1st codex review)

Plan v1 の Stage 3-7 が中間 commit で repo を破壊する critical issue を含んでいた他、α=0 重複防御 / latch≠reached test / CLI signature / 失敗 trial constructor / 既存 test 網羅 / bpy import / NaN reject / PowerShell 互換 / safety assertion / Task 粒度 で計 12 件指摘。v2 で Stage 構成を「追加的 prep + 単一 big-bang + tuning + cleanup」に再編、12 件すべてを反映。

### v2 → v3 (post 2nd codex review)

Plan v2 で残った critical 5 件を反映:
- **C1/C12 (Stage 2 過大)**: Stage 2 を **2A (additive) / 2B (switch) / 2C (cleanup)** の 3 sub-stage に分割。各 sub-stage 終了で repo green、subagent dispatch 1 回ごとに収まる粒度。
- **C5 (resolved threshold が meta に残らない)**: `TrialResult` に `formed_thresholds: list[float]` / `broken_thresholds: list[float]` を追加 (合計 14 fields)。meta writer は per-trial の resolved 値を直接書く。
- **C9 (`alpha_*` 必須化が spec §4.4 違反)**: `_AFIR_REQUIRED = {"max_relax_steps"}` のみ必須、`alpha_formed` / `alpha_broken` は default 0.0 で省略可。`sn1_dissoc` / `sn1_recomb` / `diels_alder_*` の TOML で実際に省略例を採用。
- **C11 (safety assertion を Phase 10 に送るのは spec §6.4 違反)**: `tests/conftest.py` に `assert_min_nonbonded_distance_ok` helper を追加し、Stage 3 各 task でこれを呼ぶ。
- Step Z 言及 → Step 11/12 に修正、Stage 1 完了 checkpoint の ConfigError 記述を実態 (Stage 2A で追加) に合わせる、Task 1.8 で latch≠reached を直接 assert、E2 tuning 説明から「Stage B」古い概念を削除、SN1 dissoc tuning で α 増を第一選択と明記、複数 reaction 共通 tuning は OQ-6 として持ち越し。
