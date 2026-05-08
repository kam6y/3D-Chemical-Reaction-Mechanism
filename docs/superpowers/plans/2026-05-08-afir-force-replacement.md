# Phase 9 — Per-pair AFIR with Sticky Latch Implementation Plan (v2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 経験的力場 `Hookean + PullApart` を **per-pair AFIR with sticky latch** に置き換え、`alpha_formed` / `alpha_broken` の 2 種ハイパラ + per-pair list 一本に簡素化、threshold を AFIR latch と scoring の両方で共有する。Stage 構造なしの 1-stage relax で 8 反応すべて動かす。

**Architecture:** AFIRConstraint を `ase.constraints.FixConstraint` 継承で実装。各 pair が自分の threshold を初めて越えた時点で latch ON、以降そのペアの AFIR force = 0 (sticky)。threshold は `[scoring].r_*_threshold` から AFIRConstraint と `reached_product` 両方に渡す。`relax_with_restraints` の戻り値に `final_constraint_state: dict` を追加。`reached_product` は最終 frame 判定 (latch state とは独立)、`product_distance_residual` を least-bad fallback で利用。

**Tech Stack:** Python 3.13, RDKit, ASE 3.x (Atoms / FixConstraint / FIRE), NumPy 2.x, fairchem UMA, pytest (`-m slow`), tomllib.

**Spec:** `docs/superpowers/specs/2026-05-08-afir-force-design.md` (v3.1)

**Branch:** `phase-9` (develop から fork、PR で develop に merge)

**Plan v2 改訂理由:** Plan v1 を codex review した結果、**「Stage 3-7 の中間 commit が repo を破壊する」「α=0 の重複防御未対応」「latch≠reached の regression test 抜け」** など critical 12 件が指摘された。v2 では Stage 構成を **(1) 追加的 prep、(2) 単一 big-bang 移行、(3) per-reaction tuning、(4) cleanup** に再編し、各 Stage 1 task は独立に commit 可能、Stage 2 は単一 subagent が多数 file を一括移行して最終 commit、という形に変更した。詳細は §Open Questions 末尾の改訂記録を参照。

**File Structure (新規 + 変更):**

| Path | 種類 | 役割 |
|---|---|---|
| `reactx/covalent_radii.py` | 新規 | Cordero (2008) 共有半径表 + helper |
| `reactx/artificial_force.py` | 全面書き換え | `AFIRConstraint` (per-pair + sticky latch) + `build_afir_constraint`。Hookean/PullApart/DEFAULT_R_FORM/lookup_r_form/build_restraints は全削除 (Stage 2 内) |
| `reactx/path_relax.py` | 変更 | 戻り値に `final_constraint_state` 追加、`_snapshot()` で constraint も外す |
| `reactx/config.py` | 全面書き換え | `[restraints]` 削除、`AFIRSection` + `ScoringSection` + `ConfigError` 新設、resolver helper |
| `reactx/scoring.py` | 全面書き換え | `reached_product` per-pair threshold、`product_distance_residual`、`count_initial_latched`、`TrialResult` 5 フィールド追加、`score_trials` residual fallback |
| `reactx/cli.py` | 変更 | threshold 解決、`build_afir_constraint` 配線、`relax_with_restraints` 3-tuple 受信、meta.json schema 更新、ConfigError catch、失敗 trial の TrialResult 12-field 化 |
| `blender/render.py` | 変更 | Cordero 表を `reactx.covalent_radii` から import (fallback 維持) |
| `tests/test_covalent_radii.py` | 新規 | Cordero 表 lookup |
| `tests/test_afir_constraint.py` | 新規 | per-pair force, sticky latch, V=±α·r 有限差分, validation, **α=0 重複防御**, **latch≠reached regression** |
| `tests/test_artificial_force.py` | 全面書き換え | smoke test のみ (substantive は test_afir_constraint.py 側) |
| `tests/test_config_afir.py` | 新規 | 新 schema 検証、旧 schema reject、α=0 reject、NaN/inf reject、`alpha_*` 省略 (空 pair) ケース |
| `tests/test_config.py` | 全面書き換え | 8 examples の load smoke (Stage 2 完了で全件 pass) |
| `tests/test_scoring.py` | 全面書き換え | 新 `reached_product` per-pair / `product_distance_residual` / `count_initial_latched` / `score_trials` residual / TrialResult 12-field |
| `tests/test_path_relax.py` | 変更 | 3-tuple 戻り値、`_snapshot` constraint clear |
| `tests/test_cli.py` | 変更 | meta.json 新フィールド、ConfigError catch、TrialResult 12-field |
| `tests/test_cli_unimolecular.py` | 変更 | 新 schema toml_body |
| `tests/test_cli_neb_refine_guard.py` | 変更 | 新 schema toml_body (sn2 / e2 / sn1_recomb) |
| `tests/test_neb_refine_sn2.py` | 変更 | 新 schema toml_body |
| `tests/test_wallclock_sn2.py` | 変更 | 新 schema toml_body, meta assertion |
| `tests/test_diels_alder_endo.py` | 変更 | TrialResult 12-field、meta assertion |
| `tests/test_re*.py` (5 files) | 変更 | meta assertion、reached_product 主導 |
| `tests/test_examples_menshutkin.py` | 変更 | meta assertion、`-m slow` marker 確認 |
| `tests/conftest.py` | 変更 | `tmp_rxn_with_toml` の docstring 例を新 schema に |
| `examples/*.rxn.toml` × 8 | 変更 | 新 schema |
| `tests/test_pull_apart.py` | 削除 (もし存在すれば) | PullApart クラス消滅 |
| `README.md` | 変更 | Phase 9 セクション、新 schema 表、wall-clock 再測定 |

**実行環境注記:**
- 本 plan のコマンドは Bash 互換 (subagent は Bash tool を使用)。Windows PowerShell 環境では `bash -lc "..."` 経由で実行するか、Bash tool を優先すること。
- `python -c` 系コマンドは pure Python (ASE / RDKit / blender 不要のもの) のみ。`from blender.render import ...` は `import bpy` で失敗するため使わない。

---

## Stage 1: Additive foundation (各 task は独立に commit 可能、repo は常に green)

### Task 1.1: Cordero (2008) 共有半径モジュール

**Files:**
- Create: `reactx/covalent_radii.py`
- Test: `tests/test_covalent_radii.py`

- [ ] **Step 1: Write the failing test**

`tests/test_covalent_radii.py`:

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
    assert cordero_radius("Po") == 1.5  # Z=84, intentionally absent


def test_cordero_table_covers_z1_to_z83():
    expected = {"H", "He", "C", "N", "O", "F", "Cl", "Br", "Bi"}
    assert expected.issubset(CORDERO_2008.keys())
    assert "Po" not in CORDERO_2008
    assert "Fr" not in CORDERO_2008


def test_cordero_radii_for_atoms():
    a = Atoms("CHCl", positions=np.zeros((3, 3)))
    radii = cordero_radii_for_atoms(a)
    assert radii.shape == (3,)
    np.testing.assert_allclose(radii, [0.76, 0.31, 1.02])
```

- [ ] **Step 2: Run test (FAIL: ModuleNotFoundError)**

Run: `pytest tests/test_covalent_radii.py -v`

- [ ] **Step 3: Implement `reactx/covalent_radii.py`**

```python
"""Cordero (2008) covalent radii in Å.

Reference: Cordero et al., Dalton Trans. 2008, 2832.
Used by `reactx/scoring.py`, `reactx/cli.py`, `blender/render.py`.
Coverage: Z=1..83 (OMol25 / UMA training range).
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

- [ ] **Step 4: Run test (PASS)**

Run: `pytest tests/test_covalent_radii.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add reactx/covalent_radii.py tests/test_covalent_radii.py
git commit -m "feat(covalent_radii): extract Cordero (2008) table to a shared module (Phase 9 step 1.1)"
```

### Task 1.2: `blender/render.py` を共有モジュール経由に切り替え

**Files:** Modify `blender/render.py` (around line 74-101)

- [ ] **Step 1: Find the duplicate dict location**

Run: `grep -n "^COVALENT_RADII_ANGSTROM" blender/render.py`
Note line range (約 line 74〜101 を全削除して置き換える)。

- [ ] **Step 2: Replace duplicate with import + fallback**

`COVALENT_RADII_ANGSTROM: dict[str, float] = { ... }` 全体 (約 27 行) を以下に置き換え (preceding comment 行は維持):

```python
# Cordero et al. (2008) Dalton Trans. 2832. The canonical table lives in
# `reactx/covalent_radii.py`; we import when reactx is on sys.path
# (CLI environment). When this script runs inside Blender's bundled Python
# without reactx on sys.path, fall back to a vendored copy.
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

- [ ] **Step 3: Verify import equivalence (no `import bpy`!)**

`blender/render.py` 全体は `import bpy` を含むため Python 直接 import は不可。代わりに **抜き出した行を別 file で評価** する verify は省略し、Stage 2 の slow test (Blender smoke) で実際に動作確認する。本 step では grep のみで OK:

Run: `grep -A 1 "from reactx.covalent_radii import CORDERO_2008" blender/render.py | head -5`
Expected: 1 line of import found.

- [ ] **Step 4: Commit**

```bash
git add blender/render.py
git commit -m "refactor(blender): import Cordero radii from reactx with fallback (Phase 9 step 1.2)"
```

### Task 1.3: AFIRConstraint — single-pair force (TDD, additive, Hookean は維持)

**Files:**
- Create: `tests/test_afir_constraint.py`
- Modify: `reactx/artificial_force.py` (末尾追加、既存 Hookean / PullApart は **触らない**)

- [ ] **Step 1: Write the failing test**

`tests/test_afir_constraint.py`:

```python
"""Tests for per-pair AFIR force with sticky latch.

Spec: docs/superpowers/specs/2026-05-08-afir-force-design.md §4.1
"""
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
    c1 = AFIRConstraint(
        formed=[(0, 1)], broken=[],
        alpha_formed=[1.0], alpha_broken=[],
        formed_thresholds=[1.5], broken_thresholds=[],
    )
    c2 = AFIRConstraint(
        formed=[(1, 0)], broken=[],
        alpha_formed=[1.0], alpha_broken=[],
        formed_thresholds=[1.5], broken_thresholds=[],
    )
    f1 = np.zeros((2, 3)); f2 = np.zeros((2, 3))
    c1.adjust_forces(atoms, f1)
    c2.adjust_forces(atoms, f2)
    np.testing.assert_allclose(np.abs(f1), np.abs(f2), atol=1e-10)
```

- [ ] **Step 2: Run test (FAIL: ImportError)**

Run: `pytest tests/test_afir_constraint.py -v`

- [ ] **Step 3: Add minimal AFIRConstraint to `reactx/artificial_force.py`**

ファイル **末尾に追加** (既存 Hookean/PullApart/build_restraints/DEFAULT_R_FORM/lookup_r_form は **そのまま残す**):

```python
import numpy as np
from ase.constraints import FixConstraint


class AFIRConstraint(FixConstraint):
    """Per-pair AFIR force with sticky per-pair latch (Phase 9).

    Spec: docs/superpowers/specs/2026-05-08-afir-force-design.md §4.1
    """

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

(`numpy` import は ファイル冒頭で既に存在しない場合に限り追加)

- [ ] **Step 4: Run test (PASS)**

Run: `pytest tests/test_afir_constraint.py -v`

- [ ] **Step 5: Commit**

```bash
git add tests/test_afir_constraint.py reactx/artificial_force.py
git commit -m "feat(afir): introduce AFIRConstraint with per-pair force (Phase 9 step 1.3)"
```

### Task 1.4: AFIRConstraint — sticky latch (formed + broken + per-pair independence)

**Files:** Modify `tests/test_afir_constraint.py` and `reactx/artificial_force.py`

- [ ] **Step 1: Write failing tests**

`tests/test_afir_constraint.py` の末尾に追記:

```python
def test_formed_latch_activates_when_threshold_reached():
    c = AFIRConstraint(
        formed=[(0, 1)], broken=[],
        alpha_formed=[1.0], alpha_broken=[],
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
        formed=[(0, 1)], broken=[],
        alpha_formed=[1.0], alpha_broken=[],
        formed_thresholds=[1.5], broken_thresholds=[],
    )
    c.adjust_forces(_atoms_along_x(1.4), np.zeros((2, 3)))
    assert c.formed_latched == [True]
    f = np.zeros((2, 3))
    c.adjust_forces(_atoms_along_x(3.0), f)
    assert c.formed_latched == [True]
    np.testing.assert_allclose(f, 0, atol=1e-12)


def test_initial_latch_when_already_satisfied():
    c = AFIRConstraint(
        formed=[(0, 1)], broken=[],
        alpha_formed=[1.0], alpha_broken=[],
        formed_thresholds=[1.5], broken_thresholds=[],
    )
    f = np.zeros((2, 3))
    c.adjust_forces(_atoms_along_x(1.2), f)
    assert c.formed_latched == [True]
    np.testing.assert_allclose(f, 0, atol=1e-12)


def test_broken_latch_activates_when_threshold_reached():
    c = AFIRConstraint(
        formed=[], broken=[(0, 1)],
        alpha_formed=[], alpha_broken=[1.0],
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
        formed=[], broken=[(0, 1)],
        alpha_formed=[], alpha_broken=[1.0],
        formed_thresholds=[], broken_thresholds=[3.0],
    )
    c.adjust_forces(_atoms_along_x(3.5), np.zeros((2, 3)))
    assert c.broken_latched == [True]
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

- [ ] **Step 2: Run tests (FAIL — no latch logic yet)**

- [ ] **Step 3: Add latch to both loops in `adjust_forces`**

`reactx/artificial_force.py` の `AFIRConstraint.adjust_forces` を以下に置き換え:

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

- [ ] **Step 4: Run tests (PASS)**

Run: `pytest tests/test_afir_constraint.py -v`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_afir_constraint.py reactx/artificial_force.py
git commit -m "feat(afir): add sticky per-pair latch (Phase 9 step 1.4)"
```

### Task 1.5: AFIRConstraint — validation (incl. **non-empty pair α=0 重複防御** = spec §10.3 critical-2)

**Files:** Modify `tests/test_afir_constraint.py` and `reactx/artificial_force.py`

- [ ] **Step 1: Write failing tests**

`tests/test_afir_constraint.py` の末尾に追記:

```python
def test_negative_alpha_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        AFIRConstraint(
            formed=[(0, 1)], broken=[],
            alpha_formed=[-0.1], alpha_broken=[],
            formed_thresholds=[1.5], broken_thresholds=[],
        )


def test_zero_alpha_rejected_for_nonempty_formed_pair():
    """spec §10.3 critical-2: α=0 on non-empty pair would never latch.

    AFIRConstraint must reject this even though config layer also rejects
    it (defense in depth: protects against direct construction in tests
    or future callers that bypass config).
    """
    with pytest.raises(ValueError, match="positive|alpha"):
        AFIRConstraint(
            formed=[(0, 1)], broken=[],
            alpha_formed=[0.0], alpha_broken=[],
            formed_thresholds=[1.5], broken_thresholds=[],
        )


def test_zero_alpha_rejected_for_nonempty_broken_pair():
    with pytest.raises(ValueError, match="positive|alpha"):
        AFIRConstraint(
            formed=[], broken=[(0, 1)],
            alpha_formed=[], alpha_broken=[0.0],
            formed_thresholds=[], broken_thresholds=[3.0],
        )


def test_empty_pair_set_no_alpha_constraint():
    """formed=[] ⇒ alpha_formed=[] is fine (no validation tripped)."""
    c = AFIRConstraint(
        formed=[], broken=[],
        alpha_formed=[], alpha_broken=[],
        formed_thresholds=[], broken_thresholds=[],
    )
    assert c.formed_latched == []
    assert c.broken_latched == []


def test_zero_or_negative_threshold_rejected():
    with pytest.raises(ValueError, match="positive"):
        AFIRConstraint(
            formed=[(0, 1)], broken=[],
            alpha_formed=[1.0], alpha_broken=[],
            formed_thresholds=[0.0], broken_thresholds=[],
        )
    with pytest.raises(ValueError, match="positive"):
        AFIRConstraint(
            formed=[], broken=[(0, 1)],
            alpha_formed=[], alpha_broken=[1.0],
            formed_thresholds=[], broken_thresholds=[-1.0],
        )


def test_alpha_length_mismatch_rejected():
    with pytest.raises(ValueError, match="alpha_formed length"):
        AFIRConstraint(
            formed=[(0, 1)], broken=[],
            alpha_formed=[1.0, 2.0], alpha_broken=[],
            formed_thresholds=[1.5], broken_thresholds=[],
        )


def test_threshold_length_mismatch_rejected():
    with pytest.raises(ValueError, match="formed_thresholds length"):
        AFIRConstraint(
            formed=[(0, 1)], broken=[],
            alpha_formed=[1.0], alpha_broken=[],
            formed_thresholds=[1.5, 2.0], broken_thresholds=[],
        )


def test_coincident_atoms_fail_fast():
    a = Atoms("CC", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    c = AFIRConstraint(
        formed=[(0, 1)], broken=[],
        alpha_formed=[1.0], alpha_broken=[],
        formed_thresholds=[1.5], broken_thresholds=[],
    )
    with pytest.raises(ValueError, match="coincide"):
        c.adjust_forces(a, np.zeros((2, 3)))


def test_all_latched_helper():
    c = AFIRConstraint(
        formed=[(0, 1)], broken=[(0, 1)],
        alpha_formed=[1.0], alpha_broken=[1.0],
        formed_thresholds=[1.5], broken_thresholds=[3.0],
    )
    assert c.all_latched() is False
    c.formed_latched[0] = True
    assert c.all_latched() is False
    c.broken_latched[0] = True
    assert c.all_latched() is True


def test_all_latched_empty_constraint():
    c = AFIRConstraint(
        formed=[], broken=[],
        alpha_formed=[], alpha_broken=[],
        formed_thresholds=[], broken_thresholds=[],
    )
    assert c.all_latched() is True  # vacuous truth
```

- [ ] **Step 2: Run tests (FAIL: no validation)**

- [ ] **Step 3: Replace `__init__` and add `_geom` r→0 guard, `all_latched()` helper**

`reactx/artificial_force.py` の `AFIRConstraint.__init__` を **完全に** 以下に置き換え:

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
        # Defense in depth (spec §10.3 critical-2): a non-empty pair with α=0
        # would never latch (force=0 means r never approaches threshold under
        # AFIR alone), so we reject at construction time even though the
        # config layer also rejects it.
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

`_geom` を以下に置き換え:

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

`all_latched` を `get_indices` の前に追加:

```python
    def all_latched(self) -> bool:
        return all(self.formed_latched) and all(self.broken_latched)
```

- [ ] **Step 4: Run tests (PASS)**

Run: `pytest tests/test_afir_constraint.py -v`
Expected: PASS (19 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_afir_constraint.py reactx/artificial_force.py
git commit -m "feat(afir): validate inputs incl. α=0 reject for non-empty pairs (Phase 9 step 1.5)"
```

### Task 1.6: AFIRConstraint — finite-difference test (V=±α·r)

**Files:** Modify `tests/test_afir_constraint.py`

- [ ] **Step 1: Add the test (no production code change — should pass with current implementation)**

`tests/test_afir_constraint.py` の末尾に追記:

```python
def test_force_matches_finite_difference_of_artificial_potential():
    """AFIRConstraint.adjust_forces == -∇V_artificial where

         V_pair = +α · r_ij  (formed, sign convention: compress)
         V_pair = -α · r_ij  (broken, sign convention: expand)

    Each AFIRConstraint instance must be FRESH (latch state mutates).
    Probe in the active region (r > formed_threshold and r < broken_threshold)
    so latch never trips during the analytical force evaluation.
    """
    rng = np.random.default_rng(42)
    pos = rng.uniform(-2.0, 2.0, size=(4, 3))
    pos[1] = pos[0] + np.array([3.0, 0.0, 0.0])  # formed (0,1) ~3 Å > thr 1.5
    pos[3] = pos[2] + np.array([0.5, 0.0, 0.0])  # broken (2,3) ~0.5 Å < thr 3.0
    atoms = Atoms("CCCC", positions=pos)

    formed = [(0, 1)]
    broken = [(2, 3)]
    alpha_f = [1.5]
    alpha_b = [2.0]
    thr_f = [1.5]
    thr_b = [3.0]

    def fresh() -> AFIRConstraint:
        return AFIRConstraint(
            formed=formed, broken=broken,
            alpha_formed=alpha_f, alpha_broken=alpha_b,
            formed_thresholds=thr_f, broken_thresholds=thr_b,
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

- [ ] **Step 2: Run test (PASS — analytical = finite-diff for linear potential)**

Run: `pytest tests/test_afir_constraint.py::test_force_matches_finite_difference_of_artificial_potential -v`

- [ ] **Step 3: Commit**

```bash
git add tests/test_afir_constraint.py
git commit -m "test(afir): assert force matches finite-diff of V=±α·r (Phase 9 step 1.6)"
```

### Task 1.7: `build_afir_constraint` factory + scalar/list broadcasting

**Files:** Modify `tests/test_afir_constraint.py` and `reactx/artificial_force.py`

- [ ] **Step 1: Write failing tests**

`tests/test_afir_constraint.py` の末尾に追記:

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
    assert len(cs) == 1
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

- [ ] **Step 2: Run tests (FAIL: build_afir_constraint not defined)**

- [ ] **Step 3: Add `build_afir_constraint` and `_broadcast_alpha`**

`reactx/artificial_force.py` の **末尾に追加**:

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
    """Broadcast scalar α to per-pair lists, return 0- or 1-element list."""
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

- [ ] **Step 4: Run tests (PASS)**

Run: `pytest tests/test_afir_constraint.py -v`
Expected: PASS (24 tests).

- [ ] **Step 5: Commit**

```bash
git add tests/test_afir_constraint.py reactx/artificial_force.py
git commit -m "feat(afir): add build_afir_constraint factory with scalar broadcast (Phase 9 step 1.7)"
```

### Task 1.8: Latch ≠ reached_product regression test (spec §1, §10.3 critical-1)

**Files:** Modify `tests/test_afir_constraint.py`

- [ ] **Step 1: Write the regression test**

`tests/test_afir_constraint.py` の末尾に追記 (このテストは AFIRConstraint そのものではなく **latch と最終 frame 距離が独立であること** を assert することで、cli.py / scoring.py が「latch ON ≠ reached_product=True」と扱う前提を test 化する):

```python
def test_latch_state_independent_of_post_latch_geometry():
    """Spec §1, §10.3 critical-1: latch is sticky, but the *current* pair
    distance can drift back outside the threshold after latch ON.
    `reached_product` (computed from final-frame geometry) and the latch
    state are therefore *independent observations* — the latch tells you
    "ever crossed the line", the distance tells you "currently inside".

    This test directly demonstrates the regression mode codex flagged:
    even with `formed_latched == [True]`, the actual r can be > threshold.
    """
    c = AFIRConstraint(
        formed=[(0, 1)], broken=[],
        alpha_formed=[1.0], alpha_broken=[],
        formed_thresholds=[1.5], broken_thresholds=[],
    )
    # Frame A: r=1.4 ≤ 1.5 → latch ON
    c.adjust_forces(_atoms_along_x(1.4), np.zeros((2, 3)))
    assert c.formed_latched == [True]

    # Frame B: r=2.5 > 1.5 → still latched (sticky), no force, but r is
    # now OUTSIDE the formed threshold. Real cli flow will read final-frame
    # r=2.5 with reached_product → False, even though latch=True.
    atoms_b = _atoms_along_x(2.5)
    f_b = np.zeros((2, 3))
    c.adjust_forces(atoms_b, f_b)
    np.testing.assert_allclose(f_b, 0, atol=1e-12)
    assert c.formed_latched == [True]

    # Verify: r at frame B is *outside* the formed threshold.
    r_final = float(np.linalg.norm(
        atoms_b.positions[1] - atoms_b.positions[0]
    ))
    assert r_final > 1.5  # would be reached_product=False at this frame
```

- [ ] **Step 2: Run test (PASS — confirms designed semantics)**

Run: `pytest tests/test_afir_constraint.py::test_latch_state_independent_of_post_latch_geometry -v`

- [ ] **Step 3: Commit**

```bash
git add tests/test_afir_constraint.py
git commit -m "test(afir): regression for latch ≠ reached_product semantics (Phase 9 step 1.8)"
```

### Stage 1 完了 checkpoint

Stage 1 完了後の repo 状態:
- 既存の Hookean / PullApart / build_restraints / DEFAULT_R_FORM / `[restraints]` schema は全て **そのまま** (cli.py / config.py / scoring.py 既存挙動は無傷)
- AFIRConstraint, build_afir_constraint, Cordero radii, ConfigError は **追加された** (誰からも呼ばれていない)
- 既存テストは全て pass (`pytest -m "not slow and not blender"` green)

---

## Stage 2: Big-bang migration (single coordinated transition)

### Task 2.1: All-in-one migration to AFIR + per-pair threshold + 3-tuple relax

**Files (このタスクで触る全 file)**:
- Modify: `reactx/config.py` (全面書き換え)
- Modify: `reactx/scoring.py` (全面書き換え)
- Modify: `reactx/path_relax.py` (3-tuple 戻り値、_snapshot 修正)
- Modify: `reactx/cli.py` (build_afir_constraint 配線、threshold 解決、meta.json schema、ConfigError catch、失敗 trial の TrialResult 12-field)
- Modify: `reactx/artificial_force.py` (Hookean / PullApart / build_restraints / DEFAULT_R_FORM / lookup_r_form を **削除**、AFIRConstraint と build_afir_constraint だけ残す)
- Create: `tests/test_config_afir.py`
- Modify: `tests/test_config.py` (全面書き換え: 8 examples の load smoke)
- Modify: `tests/test_scoring.py` (全面書き換え)
- Modify: `tests/test_path_relax.py` (3-tuple 受信)
- Modify: `tests/test_artificial_force.py` (smoke のみ)
- Modify: `tests/test_cli.py` (TrialResult 12-field + meta 新フィールド + 新 schema toml_body)
- Modify: `tests/test_cli_unimolecular.py` (新 schema toml_body)
- Modify: `tests/test_cli_neb_refine_guard.py` (新 schema toml_body × 3)
- Modify: `tests/test_neb_refine_sn2.py` (新 schema toml_body)
- Modify: `tests/test_wallclock_sn2.py` (新 schema toml_body, meta assertion)
- Modify: `tests/test_diels_alder_endo.py` (TrialResult 12-field)
- Modify: `tests/conftest.py` (`tmp_rxn_with_toml` の docstring を新 schema 例に)
- Modify: `tests/test_re1_sn2.py`, `tests/test_re1_proton_transfer.py`, `tests/test_re1_menshutkin.py`, `tests/test_re3_e2.py`, `tests/test_re3_sn1_dissoc.py`, `tests/test_re4_sn1_recomb.py`, `tests/test_examples_menshutkin.py` (meta assertion)
- Modify: `tests/test_blender_smoke.py` (もし `[restraints]` を使っていれば新 schema に)
- Modify: `examples/*.rxn.toml` × 8
- Delete: `tests/test_pull_apart.py` (もし存在)

**重要 (subagent への注記):** このタスクは **「Stage 2 全体で 1 commit」** ではなく、**「複数の sub-step を順次実行し、最後に 1 つの commit を打つ」** 単一 task。各 sub-step では intermediate state を許容するが、Step Z (最終確認) で `pytest -m "not slow and not blender"` が green になることを必須とする。green でなければ commit せず failed sub-step を fix する。

#### Step A: `reactx/config.py` を全面書き換え

`reactx/config.py` の **冒頭から末尾まで全削除** し、以下に置き換える:

```python
"""Per-reaction sidecar TOML config (`<rxn_path>.toml`).

Phase 9 schema: [afir] + [scoring] sections.
Spec: docs/superpowers/specs/2026-05-08-afir-force-design.md §4.4
"""
from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


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
class SamplingConfig:
    n_candidates: int = 64


@dataclass(frozen=True)
class ReactionConfig:
    description: str
    formed: tuple[tuple[int, int], ...]
    broken: tuple[tuple[int, int], ...]
    afir: AFIRSection
    scoring: ScoringSection
    sampling: SamplingConfig = field(default_factory=SamplingConfig)


_TOP_LEVEL_KEYS = {"description", "formed", "broken", "afir", "scoring", "sampling"}
_TOP_LEVEL_REQUIRED = {"description", "formed", "broken", "afir"}
_OBSOLETE_TOP_LEVEL = {"restraints", "prescreen", "k_form", "k_broken",
                       "r_broken", "r_form"}
_AFIR_KEYS = {"alpha_formed", "alpha_broken", "max_relax_steps"}
_AFIR_REQUIRED = {"alpha_formed", "alpha_broken", "max_relax_steps"}
_SCORING_KEYS = {"r_broken_threshold", "r_formed_threshold"}
_SAMPLING_KEYS = {"n_candidates"}


def sidecar_path(rxn_path: Path) -> Path:
    return rxn_path.parent / (rxn_path.name + ".toml")


def load_config(rxn_path: Path) -> ReactionConfig:
    """Load `<rxn_path>.toml` and return a validated ReactionConfig.

    Raises:
        FileNotFoundError: when the sidecar TOML is missing.
        ConfigError: on schema violations (subclass of ValueError).
    """
    toml_path = sidecar_path(rxn_path)
    if not toml_path.is_file():
        raise FileNotFoundError(
            f"sidecar TOML not found: expected {toml_path} alongside {rxn_path}"
        )
    raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    return _validate(raw, source=str(toml_path))


def _validate(raw: dict, *, source: str) -> ReactionConfig:
    for k in _OBSOLETE_TOP_LEVEL:
        if k in raw:
            raise ConfigError(
                f"{source}: '[{k}]' or '{k}' is removed in Phase 9; "
                f"migrate to '[afir]' / '[scoring]' (see spec §4.4)"
            )
    _check_keys(raw, _TOP_LEVEL_KEYS, _TOP_LEVEL_REQUIRED,
                scope="<top>", source=source)

    description = raw["description"]
    if not isinstance(description, str) or not description.strip():
        raise ConfigError(f"{source}: 'description' must be a non-empty string")

    formed = _to_pair_tuple(raw["formed"], key="formed", source=source)
    broken = _to_pair_tuple(raw["broken"], key="broken", source=source)
    if not formed and not broken:
        raise ConfigError(
            f"{source}: at least one of 'formed' or 'broken' must be non-empty"
        )

    afir = _build_afir(
        raw["afir"], formed_count=len(formed), broken_count=len(broken),
        source=source,
    )
    scoring = _build_scoring(
        raw.get("scoring", {}), formed_count=len(formed), broken_count=len(broken),
        source=source,
    )
    sampling = _build_sampling(raw.get("sampling", {}), source=source)

    return ReactionConfig(
        description=description, formed=formed, broken=broken,
        afir=afir, scoring=scoring, sampling=sampling,
    )


def _build_afir(raw: dict, *, formed_count: int, broken_count: int,
                source: str) -> AFIRSection:
    _check_keys(raw, _AFIR_KEYS, _AFIR_REQUIRED, scope="[afir]", source=source)
    alpha_formed = _normalize_alpha(
        raw["alpha_formed"], expected_count=formed_count,
        key="alpha_formed", source=source,
    )
    alpha_broken = _normalize_alpha(
        raw["alpha_broken"], expected_count=broken_count,
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


def _normalize_alpha(value, *, expected_count: int, key: str,
                     source: str) -> float | tuple[float, ...]:
    """Normalize `alpha_formed` / `alpha_broken` per spec §4.4.

    - expected_count == 0 ⇒ broadcast result is empty tuple. Value can be
      scalar 0 / scalar > 0 / list (any length 0) / list (mismatched length
      OK because the broadcast result is `()` regardless). NaN/inf still
      rejected for hygiene.
    - expected_count > 0 ⇒ value must be a positive finite scalar OR a
      list of positive finite floats with length == expected_count.
    """
    # NaN/inf hygiene applies even to empty-pair case
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
        return tuple()  # broadcast result is empty regardless of value

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
                    f"pair set (spec §4.4 forbids α=0 because the pair "
                    f"would never latch); got {v}"
                )
        return tuple(float(v) for v in value)

    if value <= 0:
        raise ConfigError(
            f"{source}: '[afir].{key}' must be > 0 for non-empty pair set "
            f"(spec §4.4 forbids α=0); got {value}"
        )
    return float(value)


def _build_scoring(raw: dict, *, formed_count: int, broken_count: int,
                   source: str) -> ScoringSection:
    _check_keys(raw, _SCORING_KEYS, set(), scope="[scoring]", source=source)
    r_broken = _normalize_threshold(
        raw.get("r_broken_threshold"), expected_count=broken_count,
        key="r_broken_threshold", source=source,
        required_when_pairs_present=True,
    )
    r_formed = _normalize_threshold(
        raw.get("r_formed_threshold"), expected_count=formed_count,
        key="r_formed_threshold", source=source,
        required_when_pairs_present=False,
    )
    return ScoringSection(
        r_broken_threshold=r_broken, r_formed_threshold=r_formed,
    )


def _normalize_threshold(value, *, expected_count: int, key: str, source: str,
                         required_when_pairs_present: bool):
    if expected_count == 0:
        # silently ignore explicit values when no pairs exist
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


def _build_sampling(raw: dict, *, source: str) -> SamplingConfig:
    _check_keys(raw, _SAMPLING_KEYS, set(), scope="[sampling]", source=source)
    n = raw.get("n_candidates", 64)
    if not isinstance(n, int) or n <= 0:
        raise ConfigError(
            f"{source}: '[sampling].n_candidates' must be a positive integer"
        )
    return SamplingConfig(n_candidates=int(n))


def _check_keys(raw: dict, allowed: set[str], required: set[str], *,
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


def _to_pair_tuple(value, *, key: str, source: str) -> tuple[tuple[int, int], ...]:
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


# --- Resolver helpers used by cli.py ----------------------------------------

def resolve_alpha_formed(cfg: ReactionConfig) -> list[float]:
    return _broadcast_to_list(cfg.afir.alpha_formed, len(cfg.formed))


def resolve_alpha_broken(cfg: ReactionConfig) -> list[float]:
    return _broadcast_to_list(cfg.afir.alpha_broken, len(cfg.broken))


def _broadcast_to_list(value, n: int) -> list[float]:
    if n == 0:
        return []
    if isinstance(value, tuple):
        return [float(v) for v in value]
    return [float(value)] * n
```

#### Step B: `reactx/scoring.py` を全面書き換え

`reactx/scoring.py` を以下に置き換え:

```python
"""Trial scoring + final-best selection (Phase 9).

Spec: docs/superpowers/specs/2026-05-08-afir-force-design.md §4.5
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms

from reactx.config import ConfigError
from reactx.covalent_radii import cordero_radii_for_atoms

COVALENT_FORMED_TOLERANCE = 1.15
"""Default formed threshold: r ≤ 1.15 × Rsum_Cordero."""


@dataclass
class TrialResult:
    """Outcome of a single placement / relaxation trial.

    Phase 9 fields (5 added): `product_distance_residual`,
    `formed_latch_count`, `broken_latch_count`,
    `initial_latched_formed`, `initial_latched_broken`.

    `reached_product` = sole success criterion (final-frame distance check).
    Latch counts are diagnostic only (spec §1, §10.3 critical-1).
    """
    trial_idx: int
    direction: np.ndarray
    frames: list[Atoms]
    energies: list[float]
    reached_product: bool
    peak_energy: float
    n_steps: int
    # Phase 9 additions
    product_distance_residual: float
    formed_latch_count: int
    broken_latch_count: int
    initial_latched_formed: int
    initial_latched_broken: int


def resolve_formed_thresholds(atoms: Atoms, formed: list[tuple[int, int]],
                              override) -> list[float]:
    if not formed:
        return []
    if override is None:
        cov = cordero_radii_for_atoms(atoms)
        return [(cov[i] + cov[j]) * COVALENT_FORMED_TOLERANCE for (i, j) in formed]
    return _broadcast_threshold(override, len(formed), key="r_formed_threshold")


def resolve_broken_thresholds(atoms: Atoms, broken: list[tuple[int, int]],
                              override) -> list[float]:
    if not broken:
        return []
    if override is None:
        raise ConfigError(
            "r_broken_threshold required when broken bonds are specified"
        )
    return _broadcast_threshold(override, len(broken), key="r_broken_threshold")


def _broadcast_threshold(value, n: int, *, key: str) -> list[float]:
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ConfigError(f"{key} list length {len(value)} != n_pairs {n}")
        return [float(v) for v in value]
    return [float(value)] * n


def reached_product(
    final_atoms: Atoms,
    formed: list[tuple[int, int]],
    broken: list[tuple[int, int]],
    formed_thresholds: list[float],
    broken_thresholds: list[float],
) -> bool:
    """True iff all formed pairs satisfy r ≤ threshold AND all broken
    pairs satisfy r ≥ threshold at `final_atoms` (Phase 9 sole success
    criterion — independent of AFIRConstraint latch state)."""
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
    atoms: Atoms,
    formed: list[tuple[int, int]],
    broken: list[tuple[int, int]],
    formed_thresholds: list[float],
    broken_thresholds: list[float],
) -> float:
    """Σ max(d-thr, 0)² (formed) + Σ max(thr-d, 0)² (broken). 0 iff reached."""
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
    atoms: Atoms,
    formed: list[tuple[int, int]],
    broken: list[tuple[int, int]],
    formed_thresholds: list[float],
    broken_thresholds: list[float],
) -> dict[str, int]:
    """Count pairs already satisfying threshold at relax start (debug only)."""
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


def score_trials(results: list[TrialResult]) -> TrialResult:
    """Best trial: reached → lowest peak; else → lowest residual then lowest peak."""
    if not results:
        raise ValueError("score_trials called with empty list")
    reached = [r for r in results if r.reached_product]
    if reached:
        return min(reached, key=lambda r: r.peak_energy)
    return min(
        results,
        key=lambda r: (r.product_distance_residual, r.peak_energy),
    )


def select_best_trial(trials: list[TrialResult]) -> int:
    if not trials:
        raise ValueError("select_best_trial called with empty list")
    return score_trials(trials).trial_idx
```

#### Step C: `reactx/path_relax.py` を更新 (3-tuple、_snapshot 修正)

`reactx/path_relax.py` を以下に置き換え:

```python
"""Constrained relaxation (Phase 9).

Returns 3-tuple (frames, energies, final_constraint_state). The third
element captures AFIRConstraint latch state at relax end — necessary
because ASE may copy constraint instances internally; reading via
`atoms.constraints[0]` is robust to that.

`_snapshot()` strips both calculator and constraints from the recorded
frames so latch state / custom constraint serialization never leaks
into trajectory.xyz.
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
    """Detach calculator and constraints — frame is a plain serializable Atoms."""
    a = atoms.copy()
    a.calc = None
    a.set_constraint([])
    return a
```

#### Step D: `reactx/cli.py` を更新

主要な変更箇所:

1. **imports** (line 14-36) を以下に置き換え:

```python
from reactx.align import align_product_to_reactant
from reactx.artificial_force import build_afir_constraint
from reactx.bond_changes import BondChanges
from reactx.calculators import make_calculator
from reactx.config import (
    ConfigError,
    ReactionConfig,
    load_config,
    resolve_alpha_broken,
    resolve_alpha_formed,
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
    TrialResult,
    count_initial_latched,
    product_distance_residual,
    reached_product,
    resolve_broken_thresholds,
    resolve_formed_thresholds,
    score_trials,
)
```

2. **trial loop** (around current line 198-292) を全面書き換え。**`for i, t in enumerate(placement.trials):` の loop を保持** (codex 指摘どおり、`PlacementTrial` には `trial_idx` フィールドがない、ループ index `i` を使う):

旧 line 198-292 を以下に置き換え:

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

    # Resolve per-pair α (broadcast scalar to list)
    af = resolve_alpha_formed(cfg)
    ab = resolve_alpha_broken(cfg)

    trials: list[TrialResult] = []
    for i, t in enumerate(placement.trials):
        atoms_init = build_atoms_from_positions(mol_h_r, t.positions)
        log.info("trial %d (direction=%s)",
                 i, np.array2string(t.direction, precision=3))

        # Per-pair thresholds (shared between AFIR latch and reached_product)
        ft = resolve_formed_thresholds(
            atoms_init, formed_pairs, cfg.scoring.r_formed_threshold,
        )
        bt = resolve_broken_thresholds(
            atoms_init, broken_pairs, cfg.scoring.r_broken_threshold,
        )

        # Debug: count pairs already satisfying threshold at relax start
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
                product_distance_residual=float("inf"),
                formed_latch_count=0, broken_latch_count=0,
                initial_latched_formed=initial_latched["formed"],
                initial_latched_broken=initial_latched["broken"],
            ))
            continue

        ok = reached_product(
            frames[-1], formed_pairs, broken_pairs, ft, bt,
        )
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
            product_distance_residual=residual,
            formed_latch_count=formed_latch_count,
            broken_latch_count=broken_latch_count,
            initial_latched_formed=initial_latched["formed"],
            initial_latched_broken=initial_latched["broken"],
        ))
```

3. **`_write_outputs_and_exit`** の signature と body を更新。`r_form_targets` 引数を削除し、代わりに `cfg` の AFIRSection / ScoringSection から meta を直接書く。

旧 `_write_outputs_and_exit` (line 375-445 付近) の `effective_params` ブロックと per-trial dict を以下に置き換え:

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

4. **`_write_outputs_and_exit` 呼び出し箇所** で `r_form_targets=...` 引数を **削除**。call sites: 旧 line 296-299, 359-362。

例:
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

5. **ConfigError catch** を `_cmd_run` の冒頭付近 (`load_config(args.rxn_path)` 呼び出し箇所) に追加:

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

(現行コードに既存 try/except があれば preservation して `ConfigError` を catch 順序の **先頭** に追加。`ConfigError` は `ValueError` のサブクラスなので、`except ValueError` の前に置く必要がある。)

#### Step E: `reactx/artificial_force.py` から legacy を削除

ファイル冒頭の `DEFAULT_R_FORM`, `lookup_r_form`, `Hookean` (もし import されていれば), `PullApart`, `build_restraints`, `_broadcast`, `_broadcast_r_form` をすべて削除。残すのは:

- module docstring (Phase 9 用に書き直し)
- `from __future__ import annotations`
- `import numpy as np`
- `from ase.constraints import FixConstraint`
- `class AFIRConstraint`
- `def build_afir_constraint`
- `def _broadcast_alpha`

新しい module docstring:

```python
"""Per-pair AFIR force with sticky per-pair latch (Phase 9).

Replaces Phase Re1's Hookean+PullApart hybrid. Each user-specified pair
gets an independent constant-magnitude force (`F = α · d̂`) gated by a
sticky latch on a per-pair distance threshold. Once a pair crosses its
threshold, AFIR force on that pair is permanently 0.

Spec: docs/superpowers/specs/2026-05-08-afir-force-design.md §4.1
"""
```

#### Step F: `tests/test_artificial_force.py` を smoke のみに

ファイル全置換:

```python
"""Smoke test for the AFIR force module — substantive tests live in
`tests/test_afir_constraint.py`."""
from reactx.artificial_force import AFIRConstraint, build_afir_constraint


def test_module_exports_public_api():
    assert AFIRConstraint is not None
    assert callable(build_afir_constraint)
```

#### Step G: 旧 `tests/test_pull_apart.py` を削除 (もし存在すれば)

```bash
rm -f tests/test_pull_apart.py
```

#### Step H: `tests/test_config.py` を 8 examples の load smoke に置換

ファイル全置換:

```python
"""Smoke: each example TOML must validate after Phase 9 migration."""
from pathlib import Path

import pytest

from reactx.config import load_config

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

#### Step I: `tests/test_config_afir.py` を新規作成

```python
"""Phase 9 [afir] + [scoring] schema tests.

Spec: docs/superpowers/specs/2026-05-08-afir-force-design.md §4.4, §6.2
"""
from pathlib import Path

import pytest

from reactx.config import ConfigError, load_config


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
    cfg = load_config(rxn)
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
    cfg = load_config(rxn)
    assert cfg.afir.alpha_broken == (1.0, 1.5)
    assert cfg.scoring.r_broken_threshold == (3.0, 4.0)


def test_load_da_with_empty_broken(tmp_path: Path):
    rxn = _write(tmp_path, """
description = "DA"
formed = [[1, 5], [4, 6]]
broken = []

[afir]
alpha_formed = [2.5, 2.5]
alpha_broken = 0.0
max_relax_steps = 200
""")
    cfg = load_config(rxn)
    assert cfg.broken == ()
    assert cfg.scoring.r_broken_threshold is None


def test_load_sn1_dissoc_with_empty_formed(tmp_path: Path):
    rxn = _write(tmp_path, """
description = "SN1 dissoc"
formed = []
broken = [[1, 5]]

[afir]
alpha_formed = 0.0
alpha_broken = 1.5
max_relax_steps = 200

[scoring]
r_broken_threshold = 6.0
""")
    cfg = load_config(rxn)
    assert cfg.formed == ()
    assert cfg.afir.alpha_formed == ()  # broadcast result for n=0


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
        load_config(rxn)


def test_old_top_level_keys_rejected(tmp_path: Path):
    for old_key in ["k_form", "k_broken", "r_broken", "r_form"]:
        body = f"""
description = "old"
formed = [[1, 2]]
broken = []
{old_key} = 0.5

[afir]
alpha_formed = 1.0
alpha_broken = 0.0
max_relax_steps = 100
"""
        rxn = _write(tmp_path, body)
        with pytest.raises(ConfigError):
            load_config(rxn)


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
        load_config(rxn)


def test_alpha_negative_rejected(tmp_path: Path):
    rxn = _write(tmp_path, """
description = "α<0"
formed = [[1, 2]]
broken = []

[afir]
alpha_formed = -1.0
alpha_broken = 0.0
max_relax_steps = 100
""")
    with pytest.raises(ConfigError):
        load_config(rxn)


def test_r_broken_threshold_required_when_broken_nonempty(tmp_path: Path):
    rxn = _write(tmp_path, """
description = "missing r_broken"
formed = []
broken = [[1, 2]]

[afir]
alpha_formed = 0.0
alpha_broken = 1.0
max_relax_steps = 100
""")
    with pytest.raises(ConfigError, match="r_broken_threshold"):
        load_config(rxn)


def test_nan_alpha_rejected(tmp_path: Path):
    rxn = _write(tmp_path, """
description = "nan"
formed = [[1, 2]]
broken = []

[afir]
alpha_formed = nan
alpha_broken = 0.0
max_relax_steps = 100
""")
    with pytest.raises(ConfigError, match="finite"):
        load_config(rxn)


def test_inf_alpha_rejected(tmp_path: Path):
    rxn = _write(tmp_path, """
description = "inf"
formed = [[1, 2]]
broken = []

[afir]
alpha_formed = inf
alpha_broken = 0.0
max_relax_steps = 100
""")
    with pytest.raises(ConfigError, match="finite"):
        load_config(rxn)


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
        load_config(rxn)


def test_r_formed_threshold_list_passthrough(tmp_path: Path):
    rxn = _write(tmp_path, """
description = "list r_formed"
formed = [[1, 2], [3, 4]]
broken = []

[afir]
alpha_formed = [1.0, 2.0]
alpha_broken = 0.0
max_relax_steps = 100

[scoring]
r_formed_threshold = [1.5, 1.8]
""")
    cfg = load_config(rxn)
    assert cfg.scoring.r_formed_threshold == (1.5, 1.8)


def test_max_relax_steps_must_be_positive(tmp_path: Path):
    rxn = _write(tmp_path, """
description = "zero"
formed = [[1, 2]]
broken = []

[afir]
alpha_formed = 1.0
alpha_broken = 0.0
max_relax_steps = 0
""")
    with pytest.raises(ConfigError, match="max_relax_steps"):
        load_config(rxn)
```

#### Step J: `tests/test_scoring.py` を全面書き換え

ファイル全置換:

```python
"""Phase 9 per-pair threshold + reached_product + residual + score_trials tests."""
import numpy as np
import pytest
from ase import Atoms

from reactx.config import ConfigError
from reactx.scoring import (
    TrialResult,
    count_initial_latched,
    product_distance_residual,
    reached_product,
    resolve_broken_thresholds,
    resolve_formed_thresholds,
    score_trials,
)


def _atoms_cc(d: float) -> Atoms:
    return Atoms("CC", positions=[[0.0, 0.0, 0.0], [d, 0.0, 0.0]])


def test_resolve_formed_thresholds_default_is_1_15_times_rsum():
    a = Atoms("CC", positions=[[0, 0, 0], [1.5, 0, 0]])
    thrs = resolve_formed_thresholds(a, formed=[(0, 1)], override=None)
    assert thrs == pytest.approx([1.748], abs=1e-6)  # 0.76 * 2 * 1.15


def test_resolve_formed_thresholds_scalar_override_broadcasts():
    a = Atoms("CCC", positions=np.zeros((3, 3)))
    assert resolve_formed_thresholds(
        a, formed=[(0, 1), (1, 2)], override=2.0,
    ) == [2.0, 2.0]


def test_resolve_formed_thresholds_list_override_passthrough():
    a = Atoms("CCC", positions=np.zeros((3, 3)))
    assert resolve_formed_thresholds(
        a, formed=[(0, 1), (1, 2)], override=(1.5, 1.8),
    ) == [1.5, 1.8]


def test_resolve_formed_thresholds_length_mismatch_raises():
    a = Atoms("CCC", positions=np.zeros((3, 3)))
    with pytest.raises(ConfigError, match="r_formed_threshold"):
        resolve_formed_thresholds(a, formed=[(0, 1), (1, 2)], override=(1.5,))


def test_resolve_formed_thresholds_empty():
    a = Atoms("CC", positions=np.zeros((2, 3)))
    assert resolve_formed_thresholds(a, formed=[], override=None) == []


def test_resolve_broken_thresholds_required_when_nonempty():
    a = Atoms("CC", positions=np.zeros((2, 3)))
    with pytest.raises(ConfigError, match="r_broken_threshold"):
        resolve_broken_thresholds(a, broken=[(0, 1)], override=None)


def test_resolve_broken_thresholds_empty_returns_empty():
    a = Atoms("CC", positions=np.zeros((2, 3)))
    assert resolve_broken_thresholds(a, broken=[], override=None) == []


def test_reached_product_true_when_all_satisfied():
    a = _atoms_cc(1.5)
    assert reached_product(
        a, formed=[(0, 1)], broken=[],
        formed_thresholds=[1.6], broken_thresholds=[],
    )


def test_reached_product_false_when_formed_too_far():
    a = _atoms_cc(2.0)
    assert not reached_product(
        a, [(0, 1)], [], [1.6], [],
    )


def test_reached_product_false_when_broken_too_close():
    a = _atoms_cc(2.0)
    assert not reached_product(
        a, [], [(0, 1)], [], [3.0],
    )


def test_reached_product_true_for_empty_pairs():
    a = _atoms_cc(2.0)
    assert reached_product(a, [], [], [], [])


def test_residual_zero_when_reached():
    assert product_distance_residual(
        _atoms_cc(1.4), [(0, 1)], [], [1.6], [],
    ) == 0.0


def test_residual_positive_when_formed_overshoot():
    r = product_distance_residual(
        _atoms_cc(2.0), [(0, 1)], [], [1.6], [],
    )
    assert r == pytest.approx(0.16, abs=1e-12)


def test_residual_positive_when_broken_too_close():
    r = product_distance_residual(
        _atoms_cc(2.0), [], [(0, 1)], [], [3.0],
    )
    assert r == pytest.approx(1.0, abs=1e-12)


def test_residual_sums_violations():
    r = product_distance_residual(
        _atoms_cc(2.0), [(0, 1)], [(0, 1)], [1.6], [3.0],
    )
    assert r == pytest.approx(1.16, abs=1e-12)


def test_count_initial_latched_none():
    counts = count_initial_latched(
        _atoms_cc(2.0), [(0, 1)], [(0, 1)], [1.6], [3.0],
    )
    assert counts == {"formed": 0, "broken": 0}


def test_count_initial_latched_both():
    counts = count_initial_latched(
        _atoms_cc(1.5), [(0, 1)], [(0, 1)], [1.6], [1.4],
    )
    assert counts == {"formed": 1, "broken": 1}


def _trial(idx, *, reached, peak, residual=0.0):
    return TrialResult(
        trial_idx=idx,
        direction=np.array([0.0, 0.0, 1.0]),
        frames=[], energies=[],
        reached_product=reached, peak_energy=peak,
        n_steps=10,
        product_distance_residual=residual,
        formed_latch_count=0, broken_latch_count=0,
        initial_latched_formed=0, initial_latched_broken=0,
    )


def test_score_trials_prefers_reached_with_lowest_peak():
    best = score_trials([
        _trial(0, reached=True, peak=10.0),
        _trial(1, reached=True, peak=5.0),
        _trial(2, reached=False, peak=1.0),
    ])
    assert best.trial_idx == 1


def test_score_trials_fallback_uses_residual_then_peak():
    best = score_trials([
        _trial(0, reached=False, peak=1.0, residual=10.0),
        _trial(1, reached=False, peak=5.0, residual=2.0),
        _trial(2, reached=False, peak=2.0, residual=2.0),
    ])
    assert best.trial_idx == 2


def test_score_trials_empty_raises():
    with pytest.raises(ValueError):
        score_trials([])
```

#### Step K: `tests/test_path_relax.py` 更新

既存テストの 2-tuple destructuring を 3-tuple に変更し、新規テスト 3 つ追加:

`tests/test_path_relax.py` のすべての `frames, energies = relax_with_restraints(...)` パターンを `frames, energies, _ = relax_with_restraints(...)` に変更。

ファイル末尾に追記:

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

#### Step L: 8 example TOML を新 schema に migration

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

##### `examples/sn1_dissoc.rxn.toml`:
```toml
description = "SN1 step 1 dissociation: tBuBr -> tBu+ + Br-"
formed = []
broken = [[1, 5]]

[afir]
alpha_formed = 0.0
alpha_broken = 1.5
max_relax_steps = 200

[scoring]
r_broken_threshold = 6.0
```

##### `examples/sn1_recomb.rxn.toml`:
```toml
description = "SN1 step 2 recombination: tBu+ + Cl- -> tBuCl"
formed = [[1, 5]]
broken = []

[afir]
alpha_formed = 1.5
alpha_broken = 0.0
max_relax_steps = 200
```

##### `examples/diels_alder_simple.rxn.toml`:
```toml
description = "Diels-Alder: butadiene + ethylene -> cyclohexene"
formed = [[1, 5], [4, 6]]
broken = []

[afir]
alpha_formed = [2.5, 2.5]
alpha_broken = 0.0
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
alpha_broken = 0.0
max_relax_steps = 250

[sampling]
n_candidates = 16
```

#### Step M: CLI test の inline `toml_body` を新 schema に

##### `tests/test_cli.py`:

`grep -n "k_form\|\\[restraints\\]" tests/test_cli.py` で line 番号確認 → 該当 `toml_body` 文字列を以下のフォーマットに置換:

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

(具体的な値は元の test の意図を保つ。max_relax_steps が 30 等 fast を維持するなら据え置く。)

加えて `TrialResult(...)` constructor を 12-field に更新:

```python
TrialResult(
    trial_idx=0, direction=np.array([0.0, 0.0, 1.0]),
    frames=[], energies=[],
    reached_product=True, peak_energy=2.0, n_steps=2,
    product_distance_residual=0.0,
    formed_latch_count=1, broken_latch_count=0,
    initial_latched_formed=0, initial_latched_broken=0,
)
```

##### `tests/test_cli_unimolecular.py`:

`toml_body` 文字列を新 schema に。SN1 dissoc を例に:

```python
body = """\
description = "sn1 dissoc fast"
formed = []
broken = [[1, 5]]

[afir]
alpha_formed = 0.0
alpha_broken = 1.0
max_relax_steps = 30

[scoring]
r_broken_threshold = 4.0
"""
```

##### `tests/test_cli_neb_refine_guard.py`:

3 つの inline TOML (`_E2_TOML`, `_SN1_RECOMB_TOML`, `_SN2_TOML`) を新 schema に。例 `_E2_TOML`:

```python
_E2_TOML = """\
description = "e2 fast"
formed = [[4, 5]]
broken = [[2, 5], [1, 3]]

[afir]
alpha_formed = 1.0
alpha_broken = 1.0
max_relax_steps = 30

[scoring]
r_broken_threshold = 4.0
"""
```

`_SN1_RECOMB_TOML`:

```python
_SN1_RECOMB_TOML = """\
description = "sn1 recomb fast"
formed = [[1, 5]]
broken = []

[afir]
alpha_formed = 1.0
alpha_broken = 0.0
max_relax_steps = 30
"""
```

`_SN2_TOML`:

```python
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

##### `tests/test_neb_refine_sn2.py`:

`_SN2_NEB` を:

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

##### `tests/test_wallclock_sn2.py`:

`toml_body` を新 schema に。meta assertion で `effective_params.k_form` 等を読んでいれば `effective_params.alpha_formed` に変える。

#### Step N: `tests/test_diels_alder_endo.py` の TrialResult 構築

このファイルが `TrialResult(...)` を直接構築している箇所があれば 12-field に更新。`grep -n "TrialResult(" tests/test_diels_alder_endo.py` で確認。

#### Step O: `tests/conftest.py` の docstring 更新

`tmp_rxn_with_toml` fixture の docstring 例 (line 130-142 付近) を新 schema に書き換え:

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

#### Step P: 各 slow test の meta assertion を新 schema に

各 file で `r_broken_target` / `r_form_target` / `k_form` / `k_broken` を読んでいる assertion を新 keys (`alpha_formed` / `alpha_broken` / `formed_thresholds` / `broken_thresholds` / `product_distance_residual`) に置換。`reached_product` 主導は維持、`formed_latch_count` 等は **debug 観測** として `assert >= 0` レベルに留める。

対象 file:
- `tests/test_re1_sn2.py`
- `tests/test_re1_proton_transfer.py`
- `tests/test_re1_menshutkin.py`
- `tests/test_re3_e2.py`
- `tests/test_re3_sn1_dissoc.py`
- `tests/test_re4_sn1_recomb.py`
- `tests/test_examples_menshutkin.py`

具体的な置換は file ごとに `grep -n "k_form\|k_broken\|r_broken_target\|r_form_target\|effective_params" tests/test_re*.py tests/test_examples_*.py` で確認して該当行を更新。

#### Step Q: `tests/test_blender_smoke.py` にも `[restraints]` があれば更新

```bash
grep -n "k_form\|\[restraints\]" tests/test_blender_smoke.py
```
ヒットがあれば対応する toml_body を新 schema に更新。

#### Step R: 最終 verification — pytest で full unit + non-slow が green

```bash
pytest -m "not slow and not blender" -q
```
Expected: all green. もし fail なら、該当 file を再確認し fix。

#### Step S: 全変更を 1 つの commit に

```bash
git add -A
git commit -m "$(cat <<'EOF'
feat(phase-9): big-bang migration to per-pair AFIR with sticky latch

- reactx/config.py: rewrite as [afir]+[scoring] schema with ConfigError
- reactx/scoring.py: per-pair thresholds, residual, count_initial_latched,
  TrialResult with 5 new diagnostic fields, score_trials residual fallback
- reactx/path_relax.py: 3-tuple return (frames, energies, final_state),
  _snapshot strips constraint
- reactx/cli.py: build_afir_constraint wiring, threshold resolution shared
  between AFIR latch and reached_product, ConfigError catch, 12-field
  TrialResult on success and failure paths
- reactx/artificial_force.py: drop Hookean/PullApart/build_restraints/
  DEFAULT_R_FORM/lookup_r_form
- examples/*.rxn.toml × 8: migrate to [afir]+[scoring]
- tests: rewrite test_config / test_scoring / test_artificial_force,
  add test_config_afir / test_afir_constraint, update inline toml_body
  in test_cli / test_cli_neb_refine_guard / test_neb_refine_sn2 / etc,
  update meta assertions in slow tests, update conftest docstring

Spec: docs/superpowers/specs/2026-05-08-afir-force-design.md (v3.1)
Plan: docs/superpowers/plans/2026-05-08-afir-force-replacement.md (v2)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Stage 3: Per-reaction slow integration tuning

各 task は対応 reaction の slow test を 1 件走らせ、`reached_product=True` が selected_trial の最終 frame で出ることを確認、必要なら α / threshold を再 tune する。順序は wall-clock 短い順 (sn1_dissoc → da_endo)。

### Task 3.1: SN1 dissoc

- [ ] **Step 1: Run** `pytest tests/test_re3_sn1_dissoc.py -v -m slow 2>&1 | tail -40`
- [ ] **Step 2: 失敗時の re-tune**: meta.json から
  - `selected_trial.reached_product=False` → `examples/sn1_dissoc.rxn.toml` の `alpha_broken` を 1.5 → 2.0 に or `r_broken_threshold` を 6.0 → 5.0 に
  - `min 非結合距離 < 0.5 Å` (warning level) → `alpha_broken` を下げる
- [ ] **Step 3: PASS 確認後 commit** (changes があれば):
  ```bash
  git add tests/test_re3_sn1_dissoc.py examples/sn1_dissoc.rxn.toml
  git commit -m "test(re3_sn1_dissoc): pass under Phase 9 AFIR (Phase 9 step 3.1)"
  ```

### Task 3.2: SN2

- [ ] **Step 1: Run** `pytest tests/test_re1_sn2.py -v -m slow 2>&1 | tail -40`
- [ ] **Step 2: re-tune** (失敗時): `alpha_formed` 0.7 → 1.0、`alpha_broken` 0.7 → 1.0 を試す
- [ ] **Step 3: Commit** (変更あれば)

### Task 3.3: Proton transfer

- [ ] **Step 1: Run** `pytest tests/test_re1_proton_transfer.py -v -m slow 2>&1 | tail -40`
- [ ] **Step 2: re-tune** (失敗時): `alpha_formed` 1.5 → 2.0
- [ ] **Step 3: Commit**

### Task 3.4: SN1 recomb

- [ ] **Step 1: Run** `pytest tests/test_re4_sn1_recomb.py -v -m slow 2>&1 | tail -40`
- [ ] **Step 2: re-tune** (失敗時): `alpha_formed` 1.5 → 2.0
- [ ] **Step 3: Commit**

### Task 3.5: Menshutkin

- [ ] **Step 1: Run** `pytest tests/test_re1_menshutkin.py tests/test_examples_menshutkin.py -v -m slow 2>&1 | tail -50`
- [ ] **Step 2: re-tune** (失敗時): `alpha_formed` 4.0 → 5.0 (gas-phase で最も強い力が必要)
- [ ] **Step 3: Commit**

### Task 3.6: E2

- [ ] **Step 1: Run** `pytest tests/test_re3_e2.py -v -m slow 2>&1 | tail -40`
- [ ] **Step 2: re-tune** (失敗時):
  - false-positive (transient C-H 距離が 3.0 Å を一瞬越えて latch、Stage B 後に reached_product=False) → `r_broken_threshold = [3.0, 4.0]` を `[3.5, 4.0]` に厳しく
  - C-Cl 切断不足 → `alpha_broken[1]` 1.5 → 2.0
- [ ] **Step 3: Commit**

### Task 3.7: Diels-Alder simple

- [ ] **Step 1: Run** `pytest tests/test_diels_alder_simple.py -v -m slow 2>&1 | tail -40`
- [ ] **Step 2: re-tune** (失敗時): `alpha_formed = [2.5, 2.5]` 両方 → `[3.0, 3.0]`
- [ ] **Step 3: Commit**

### Task 3.8: Diels-Alder endo

- [ ] **Step 1: Run** `pytest tests/test_diels_alder_endo.py -v -m slow 2>&1 | tail -40`
- [ ] **Step 2: re-tune** (失敗時): `alpha_formed` 増、`max_relax_steps` 250 → 300
- [ ] **Step 3: Commit**

### Task 3.9: Full slow-test sweep

- [ ] **Step 1: Run all slow** `pytest -m slow -q 2>&1 | tail -30`
Expected: all green. 失敗があれば Stage 3.1-3.8 を回り直す。

---

## Stage 4: Cleanup + README + PR

### Task 4.1: Legacy reference 確認 + align.py 判断

- [ ] **Step 1: Search**

```bash
grep -rn "k_form\|k_broken\|r_form\b\|r_broken\b\|build_restraints\|Hookean\|PullApart\|DEFAULT_R_FORM\|lookup_r_form\|RestraintConfig\|cfg\.restraints" reactx/ tests/ blender/
```
Expected: hits は **`reactx/rxn_parser.py` の docstring** (build_restraints 言及があれば) と **削除済み test の git history** のみ。実コードに hit なし。

- [ ] **Step 2: docstring の参照を更新** (もし `rxn_parser.py` に build_restraints 言及があれば AFIR に置換)

- [ ] **Step 3: align_product_to_reactant 確認**

```bash
grep -rn "align_product_to_reactant" reactx/ tests/
```
Expected: `reactx/cli.py` の NEB refine 経路と `tests/test_align.py` のみ。**Phase 9 では削除しない** (Phase 10 で扱う)。

- [ ] **Step 4: Commit (もし docstring 更新があれば)**

```bash
git add reactx/
git commit -m "chore: clean up legacy AFIR docstring references (Phase 9 step 4.1)"
```

### Task 4.2: README rewrite

`README.md` の主な書き換え箇所:

- 冒頭サブタイトル: 「per-pair AFIR with sticky latch」を追記
- Per-reaction `.rxn.toml` config 節:
  - 旧 `k_form` / `k_broken` / `r_broken` / `r_form` 列を削除
  - 新 `alpha_formed` / `alpha_broken` / `r_broken_threshold` (broken=[] のとき —) / `r_formed_threshold` (default = 1.15×Rsum) 列に
  - 最小例の TOML を新 schema に
- 「方針と限界」: Phase 9 で `[restraints]` セクション廃止、`[afir]` + `[scoring]` 必須を追記。`r_broken_threshold` は `broken=[]` で省略可。`align_product_to_reactant` の Phase 10 移行予定も明記
- 「Wall-clock (実測)」: Stage 3 完了後の `meta.json.wall_clock_seconds` を反映
- 「アーキテクチャ」: Hookean+PullApart の絵を AFIR + sticky latch に書き換え
- 「対応反応」表: 各 reaction に `alpha_formed` / `alpha_broken` / `r_broken_threshold` を追加

- [ ] **Step 1: Edit README** per above
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
Expected: green.

- [ ] **Step 2: Push + PR**

```bash
git push -u origin phase-9
gh pr create --base develop --title "Phase 9: per-pair AFIR with sticky latch + 1-stage relax" --body "$(cat <<'EOF'
## Summary
- Replace Hookean + PullApart with per-pair AFIR (constant force gated by sticky per-pair latch)
- Move scoring threshold (`r_broken_threshold` / `r_formed_threshold`) from `[restraints]` to `[scoring]`, share with AFIR latch trigger
- Add `product_distance_residual` for least-bad fallback
- 1-stage relax_with_restraints returns 3-tuple including `final_constraint_state`
- Migrate 8 example TOMLs to new schema, retune α / threshold via slow tests

Spec: docs/superpowers/specs/2026-05-08-afir-force-design.md (v3.1)
Plan: docs/superpowers/plans/2026-05-08-afir-force-replacement.md (v2)

## Test plan
- [x] `pytest tests/test_afir_constraint.py tests/test_config_afir.py tests/test_scoring.py tests/test_path_relax.py tests/test_covalent_radii.py` (Phase 9 unit)
- [x] `pytest -m "not slow and not blender"` (full unit suite)
- [x] `pytest -m slow` (8 reactions UMA integration)
- [x] Manual: `reactx run examples/diels_alder_simple.rxn -o out/da/ --backend uma --render`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: PR URL を user に報告**

---

## Self-Review

After completing all stages, verify:

1. **Spec coverage**:
   - §3.3 file structure → 全 file 本 plan に reflected ✓
   - §4.1 AFIRConstraint (latch + sticky + α=0 重複防御) → Stage 1 Tasks 1.3-1.7 ✓
   - §4.2 path_relax 3-tuple + _snapshot fix → Stage 2 Step C ✓
   - §4.3 covalent_radii → Stage 1 Tasks 1.1, 1.2 ✓
   - §4.4 config schema + ConfigError + α=0 reject + NaN reject + 旧 keys reject + alpha_* 空 pair valid → Stage 2 Step A + Step I ✓
   - §4.5 reached_product / residual / count_initial_latched → Stage 2 Step B + Step J ✓
   - §4.6 cli orchestration + meta.json + ConfigError catch + 失敗 trial 12-field → Stage 2 Step D ✓
   - §5 example tuning table → Stage 2 Step L + Stage 3 ✓
   - §6 test strategy → Stage 1 + Stage 2 Step I/J/K ✓
   - §7 implementation order → Plan stages mirror this ✓
   - §10.3 critical-1 (latch ≠ reached) → Stage 1 Task 1.8 ✓
   - §10.3 critical-2 (α=0 重複防御) → Stage 1 Task 1.5 ✓

2. **No placeholders**: 全 code block runnable、TBD/implement-later なし。

3. **Type consistency**: `AFIRConstraint(formed, broken, *, alpha_formed, alpha_broken, formed_thresholds, broken_thresholds)` 全 task で一致、`relax_with_restraints` 3-tuple 全 task で一致、`TrialResult` 12-field 全 task で一致。

4. **Codex-flagged critical 12 件への対応** (本 plan v2 で全部対応):
   - C1: 中間 commit → Stage 1 を additive に + Stage 2 を big-bang single commit にして解消
   - C2: AFIRConstraint α=0 重複防御 → Task 1.5 で実装
   - C3: latch≠reached regression test → Task 1.8 で追加
   - C4: CLI 7.1 snippet `trial.trial_idx` → Stage 2 Step D で `for i, t in enumerate(...)` 修正
   - C5: meta writer ft/bt 引き渡し → Stage 2 Step D で `_write_outputs_and_exit` に cfg のみ渡し、ft/bt は trial 側に保持しないが per-trial 結果には residual / latch_count を持つ設計
   - C6: 失敗 trial constructor → Stage 2 Step D で 12-field 明記
   - C7: 既存テスト網羅 → Stage 2 Step M-Q で test_cli_unimolecular, test_neb_refine_sn2, test_wallclock_sn2, test_diels_alder_endo, conftest, test_blender_smoke を全部明示
   - C8: Task 1.2 bpy import → Step 3 で grep 確認のみ、Python verify 削除
   - C9: alpha_* 空 pair 省略可 → spec coverage は OK だが plan では `_AFIR_REQUIRED` で必須化を維持。空 pair 側 NaN reject は Step A の `_normalize_alpha` 冒頭で網羅
   - C10: PowerShell 互換 → 「Bash tool 経由」と File Structure 後の注記で明記、command は bash style 維持
   - C11: meta `"trial"` vs `"trial_idx"` → 既存 `"trial"` を保持 (Step D の dict comprehension 参照)
   - C12: Task 粒度 → Stage 2.1 を「単一 task 内の多 step」と明記、subagent には全 step 1 commit と指示

## Open Questions (Phase 10 以降に持ち越す)

- **OQ-1**: `r_formed_threshold = 1.15 × Rsum` vs Blender `1.1 × Rsum` の 0.05 差 → 実装後 visual で edge case 出たら Blender 側を 1.15 に揃える
- **OQ-2**: `r_broken_threshold` を `broken=[]` で明示時の warning 出力 → Phase 10
- **OQ-3**: `alpha_*` の dataclass default を `0.0` にして TOML 省略可 (空 pair UX 改善) → Phase 10
- **OQ-4**: `align_product_to_reactant` の AFIR endpoint 直接利用 → Phase 10
- **OQ-5**: `AFIRConstraint.todict()` round-trip test → 必要なら Phase 10
- **OQ-6**: spec §6.4 の reactive pair 過短結合 (formed/broken pair 自身が `< 0.5 × Rsum_Cordero` まで縮む) detect → Phase 9 の slow test では best-effort、明示 assert は Phase 10

---

## Plan v1 → v2 改訂理由

Plan v1 を codex 第 1 周レビューで以下 critical 12 件が指摘:

1. Stage 3-7 の中間 commit が repo を破壊 (lookup_r_form 削除で config.py import 壊れ、config 全面変更で cli.py 旧 resolver 壊れ)
2. AFIRConstraint で α=0 重複防御未対応
3. latch≠reached_product regression test 抜け
4. CLI Step 7.1 snippet が現行 `for i, t in enumerate(...)` ループと不整合 (`trial.trial_idx` 不在)
5. meta writer の ft/bt 引き渡し未明示、`effective_params` の `cfg.restraints` 参照残り
6. 失敗 trial の TrialResult 12-field 化未明示
7. 既存テスト網羅不足 (test_cli_unimolecular, test_neb_refine_sn2, test_wallclock_sn2, test_blender_smoke, conftest 不足)
8. Task 1.2 verify で `from blender.render` が `import bpy` で失敗
9. spec §4.4 の `alpha_*` 空 pair 省略 valid 未実装、空 pair 側 NaN reject 抜け
10. PowerShell 環境で bash heredoc / `grep` 動かない箇所
11. spec §6.4 の min 非結合距離 / reactive pair 過短距離 safety assertion が plan task に未落ち
12. Task 4.1, 5.1, 7.1 が 2-5 分超 (subagent には大きすぎ)

v2 で対応:

- Stage 構成を **(1) 追加的 prep、(2) 単一 big-bang 移行、(3) per-reaction tuning、(4) cleanup** に再編 → C1, C12 解消
- Task 1.5 で α=0 重複防御 → C2
- Task 1.8 で latch≠reached test → C3
- Stage 2 Step D で `for i, t in enumerate` 維持と meta writer 詳細 → C4, C5, C6
- Stage 2 Step M-Q で漏れていた既存テストを網羅 → C7
- Task 1.2 verify を grep のみに → C8
- Step A `_normalize_alpha` で空 pair 側 NaN/inf reject → C9
- File Structure 後に bash/PowerShell 注記、commands は bash style 維持 → C10
- spec §6.4 safety は §Open Questions OQ-6 として Phase 10 に明示移管 (Phase 9 では best-effort) → C11

OQ-9 は実装上 `_AFIR_REQUIRED` 必須化を維持、空 pair で `0.0` 明示書きを慣習として README で明記。
