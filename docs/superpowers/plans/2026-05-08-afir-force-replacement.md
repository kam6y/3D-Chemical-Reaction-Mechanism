# Phase 9 — Per-pair AFIR with Sticky Latch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 経験的力場 `Hookean + PullApart` を **per-pair AFIR with sticky latch** に置き換え、`alpha_formed` / `alpha_broken` の 2 種ハイパラ + per-pair list 一本に簡素化、threshold を AFIR latch と scoring の両方で共有する。Stage 構造なしの 1-stage relax でリアクション 8 種類すべてを動かす。

**Architecture:** AFIRConstraint を `ase.constraints.FixConstraint` 継承で実装。各 pair が自分の threshold を初めて越えた時点で latch ON、以降そのペアの AFIR force = 0 (sticky)。threshold は `[scoring].r_*_threshold` から AFIRConstraint と `reached_product` 両方に渡す。`relax_with_restraints` の戻り値に `final_constraint_state: dict` を追加して latch 状態を呼び出し元に確実に返す。`reached_product` は最終 frame の per-pair 距離判定 (latch state とは独立)、`product_distance_residual` を新設し least-bad fallback で利用。

**Tech Stack:** Python 3.13, RDKit 2026.3.1, ASE 3.x (Atoms / FixConstraint / FIRE), NumPy 2.x, fairchem UMA, pytest (`-m slow` で UMA 統合テスト), tomllib.

**Spec:** `docs/superpowers/specs/2026-05-08-afir-force-design.md`

**Branch:** `phase-9` (develop から fork、PR で develop に merge)

**File Structure (新規 + 変更):**

| Path | 種類 | 役割 |
|---|---|---|
| `reactx/covalent_radii.py` | 新規 | Cordero (2008) 共有半径表 + `cordero_radii_for_atoms` |
| `reactx/artificial_force.py` | 全面書き換え | `AFIRConstraint` (per-pair + sticky latch) + `build_afir_constraint`。Hookean/PullApart/DEFAULT_R_FORM/lookup_r_form/build_restraints は全削除 |
| `reactx/path_relax.py` | 変更 | 戻り値に `final_constraint_state` 追加、`_snapshot()` で constraint も外す |
| `reactx/config.py` | 全面書き換え | `[restraints]` 削除、`AFIRSection` + `ScoringSection` + `ConfigError` 新設 |
| `reactx/scoring.py` | 全面書き換え | `reached_product` per-pair threshold、`product_distance_residual`、`count_initial_latched`、`TrialResult` 拡張、`score_trials` residual fallback |
| `reactx/cli.py` | 変更 | threshold 解決、`build_afir_constraint` 配線、meta.json schema 更新、ConfigError catch |
| `blender/render.py` | 変更 | Cordero 表を `reactx.covalent_radii` から import (fallback 維持) |
| `tests/test_covalent_radii.py` | 新規 | Cordero 表 lookup |
| `tests/test_afir_constraint.py` | 新規 | per-pair force, sticky latch, V=±α·r 有限差分, validation |
| `tests/test_artificial_force.py` | 全面書き換え | 旧 Hookean/PullApart/build_restraints テスト削除、`build_afir_constraint` テストに |
| `tests/test_config_afir.py` | 新規 | 新 schema 検証、旧 schema reject、α=0 reject、NaN/inf reject |
| `tests/test_config.py` | 削除候補/書き換え | 旧 [restraints] テスト不要、新 schema は test_config_afir.py に |
| `tests/test_scoring.py` | 全面書き換え | 新 `reached_product` per-pair / `product_distance_residual` / `count_initial_latched` / `score_trials` residual |
| `tests/test_path_relax.py` | 変更 | 3-tuple 戻り値、_snapshot constraint clear |
| `tests/test_cli.py` | 変更 | meta.json 新フィールド、ConfigError catch |
| `examples/sn2.rxn.toml` | 変更 | 新 schema |
| `examples/proton_transfer.rxn.toml` | 変更 | 新 schema |
| `examples/menshutkin.rxn.toml` | 変更 | 新 schema |
| `examples/e2.rxn.toml` | 変更 | 新 schema (per-bond α_broken / r_broken_threshold) |
| `examples/sn1_dissoc.rxn.toml` | 変更 | 新 schema (formed=[]、α_formed=0.0) |
| `examples/sn1_recomb.rxn.toml` | 変更 | 新 schema (broken=[]、α_broken=0.0) |
| `examples/diels_alder_simple.rxn.toml` | 変更 | 新 schema (per-bond α_formed) |
| `examples/diels_alder_endo.rxn.toml` | 変更 | 新 schema |
| `tests/test_re*.py`, `tests/test_diels_alder_*.py` | 変更 | meta.json 新フィールド assertion、reached_product 主導 |
| `tests/test_pull_apart.py` | 削除 (もし存在すれば) | PullApart クラス消滅 |
| `README.md` | 変更 | Phase 9 セクション、新 schema 表、wall-clock 再測定 |

---

## Stage 1: Foundation (Cordero radii + ConfigError)

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
    assert cordero_radius("Po") == 1.5  # Po (Z=84) intentionally absent


def test_cordero_table_covers_z1_to_z83():
    # OMol25/UMA training range
    expected = {"H", "He", "C", "N", "O", "F", "Cl", "Br", "Bi"}
    assert expected.issubset(CORDERO_2008.keys())
    # Po (Z=84) and beyond are out of UMA range and intentionally absent
    assert "Po" not in CORDERO_2008
    assert "Fr" not in CORDERO_2008


def test_cordero_radii_for_atoms():
    a = Atoms("CHCl", positions=np.zeros((3, 3)))
    radii = cordero_radii_for_atoms(a)
    assert radii.shape == (3,)
    np.testing.assert_allclose(radii, [0.76, 0.31, 1.02])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_covalent_radii.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'reactx.covalent_radii'`

- [ ] **Step 3: Implement the module**

Create `reactx/covalent_radii.py` with the **full Z=1..83 table** copied verbatim from `blender/render.py`'s `COVALENT_RADII_ANGSTROM` dict (lines around `COVALENT_RADII_ANGSTROM: dict[str, float] = {`). Module body:

```python
"""Cordero (2008) covalent radii in Å, indexed by element symbol.

Reference: Cordero et al., Dalton Trans. 2008, 2832.
Used by:
- `reactx/scoring.py` for reached_product threshold defaults (1.15 × Rsum)
- `reactx/cli.py` for AFIR threshold resolution
- `blender/render.py` for bond drawing distance threshold (1.1 × Rsum)

Coverage: Z=1 (H) .. Z=83 (Bi), matching OMol25 / UMA training range.
Po (Z=84), Fr/Ra and the actinides are intentionally omitted — UMA cannot
score them, so reactx never encounters them in input.
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
    """Return the Cordero covalent radius (Å) for `symbol`, or `default`
    when the element is outside the table (Z > 83 or noble-element symbol typo)."""
    return CORDERO_2008.get(symbol, default)


def cordero_radii_for_atoms(atoms: Atoms) -> np.ndarray:
    """Return per-atom covalent radii (Å) for all atoms in `atoms`."""
    return np.array([cordero_radius(s) for s in atoms.get_chemical_symbols()])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_covalent_radii.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add reactx/covalent_radii.py tests/test_covalent_radii.py
git commit -m "feat(covalent_radii): extract Cordero (2008) table to a shared module (Phase 9 step 1.1)"
```

### Task 1.2: `blender/render.py` を共有モジュール経由に切り替え

**Files:**
- Modify: `blender/render.py` (line ~74-101 around `COVALENT_RADII_ANGSTROM`)

- [ ] **Step 1: Find the duplicate dict**

```bash
grep -n "^COVALENT_RADII_ANGSTROM" blender/render.py
```
Expected: 1 line, around line 74.

- [ ] **Step 2: Replace duplicate with import + fallback**

`blender/render.py` の `COVALENT_RADII_ANGSTROM: dict[str, float] = { ... }` 全体 (約 27 行) を以下に置き換える:

```python
# Cordero et al. (2008) Dalton Trans. 2832. Covalent radii in Angstrom for
# Z=1..83. Used purely for distance-based bond detection. The canonical
# table lives in `reactx/covalent_radii.py`; we import it when reactx is
# available (CLI environment) and fall back to a vendored copy when this
# script runs inside Blender's bundled Python (no reactx on sys.path).
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

- [ ] **Step 3: Verify the import works**

Run: `python -c "from blender.render import COVALENT_RADII_ANGSTROM; print(len(COVALENT_RADII_ANGSTROM))"`
Expected: `83`

- [ ] **Step 4: Sanity-check existing blender smoke test still imports**

Run: `pytest tests/test_blender_smoke.py --collect-only -q 2>&1 | head -5`
Expected: collection succeeds (no `ModuleNotFoundError`).

- [ ] **Step 5: Commit**

```bash
git add blender/render.py
git commit -m "refactor(blender): import Cordero radii from reactx with fallback (Phase 9 step 1.2)"
```

### Task 1.3: `ConfigError` exception class

**Files:**
- Modify: `reactx/config.py` (top of file)
- Test: `tests/test_config_afir.py` (will be created later in Stage 4 — only create the file with one test now)

- [ ] **Step 1: Add ConfigError to config.py**

`reactx/config.py` の最初の `from __future__ import annotations` 直下に追記:

```python
class ConfigError(ValueError):
    """Raised when `<rxn_path>.toml` violates the Phase 9 schema.

    Subclass of ValueError so existing CLI top-level catch (`except ValueError`)
    continues to work while still allowing finer-grained handling.
    """
```

- [ ] **Step 2: Verify config.py still imports cleanly**

Run: `python -c "from reactx.config import ConfigError; print(ConfigError.__mro__)"`
Expected: `(<class 'reactx.config.ConfigError'>, <class 'ValueError'>, ..., <class 'object'>)`

- [ ] **Step 3: Commit**

```bash
git add reactx/config.py
git commit -m "feat(config): add ConfigError exception (Phase 9 step 1.3)"
```

---

## Stage 2: AFIRConstraint (per-pair + sticky latch)

### Task 2.1: AFIRConstraint single-pair force (TDD)

**Files:**
- Create: `tests/test_afir_constraint.py`
- Modify: `reactx/artificial_force.py` (will rewrite incrementally)

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
    """Two carbons on the x-axis at distance d."""
    return Atoms("CC", positions=[[0.0, 0.0, 0.0], [d, 0.0, 0.0]])


def test_single_formed_pair_compresses():
    """formed pair with r > threshold → force on j toward i (negative x)."""
    atoms = _atoms_along_x(2.5)
    c = AFIRConstraint(
        formed=[(0, 1)], broken=[],
        alpha_formed=[1.0], alpha_broken=[],
        formed_thresholds=[1.5], broken_thresholds=[],
    )
    forces = np.zeros((2, 3))
    c.adjust_forces(atoms, forces)
    # F_on_j = -α · d̂ = -1.0 · (+x̂) = (-1, 0, 0)
    np.testing.assert_allclose(forces[1], [-1.0, 0.0, 0.0], atol=1e-10)
    # F_on_i = -F_on_j (Newton)
    np.testing.assert_allclose(forces[0], [+1.0, 0.0, 0.0], atol=1e-10)


def test_single_broken_pair_expands():
    """broken pair with r < threshold → force on j away from i (positive x)."""
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
    """force magnitude is symmetric under pair index swap."""
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
    f1 = np.zeros((2, 3))
    f2 = np.zeros((2, 3))
    c1.adjust_forces(atoms, f1)
    c2.adjust_forces(atoms, f2)
    # Same magnitudes on both atoms regardless of pair index order.
    np.testing.assert_allclose(np.abs(f1), np.abs(f2), atol=1e-10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_afir_constraint.py::test_single_formed_pair_compresses -v`
Expected: FAIL with `ImportError: cannot import name 'AFIRConstraint'`

- [ ] **Step 3: Write minimal AFIRConstraint (no latch yet)**

`reactx/artificial_force.py` の **末尾に** 以下を追加 (既存の Hookean / PullApart / build_restraints は **まだ削除しない**、Stage 3 で削除する):

```python
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
        return  # force-only constraint

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

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_afir_constraint.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add tests/test_afir_constraint.py reactx/artificial_force.py
git commit -m "feat(afir): introduce AFIRConstraint with per-pair force (Phase 9 step 2.1)"
```

### Task 2.2: Sticky latch — formed direction (TDD)

**Files:**
- Modify: `tests/test_afir_constraint.py`
- Modify: `reactx/artificial_force.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_afir_constraint.py` の末尾に追記:

```python
def test_formed_latch_activates_when_threshold_reached():
    """When r descends to threshold, latch ON and force=0 thereafter."""
    c = AFIRConstraint(
        formed=[(0, 1)], broken=[],
        alpha_formed=[1.0], alpha_broken=[],
        formed_thresholds=[1.5], broken_thresholds=[],
    )

    # r=2.0 > 1.5 → force on, latch off
    a = _atoms_along_x(2.0)
    f = np.zeros((2, 3))
    c.adjust_forces(a, f)
    assert c.formed_latched == [False]
    assert not np.allclose(f, 0)

    # r=1.4 < 1.5 → latch ON, force=0
    a = _atoms_along_x(1.4)
    f = np.zeros((2, 3))
    c.adjust_forces(a, f)
    assert c.formed_latched == [True]
    np.testing.assert_allclose(f, 0, atol=1e-12)


def test_formed_latch_is_sticky():
    """Once latched, even moving back outside threshold keeps force=0."""
    c = AFIRConstraint(
        formed=[(0, 1)], broken=[],
        alpha_formed=[1.0], alpha_broken=[],
        formed_thresholds=[1.5], broken_thresholds=[],
    )
    # First call: r=1.4 → latch ON
    c.adjust_forces(_atoms_along_x(1.4), np.zeros((2, 3)))
    assert c.formed_latched == [True]

    # Second call: r=3.0 → still latched, force=0
    f = np.zeros((2, 3))
    c.adjust_forces(_atoms_along_x(3.0), f)
    assert c.formed_latched == [True]
    np.testing.assert_allclose(f, 0, atol=1e-12)


def test_initial_latch_when_already_satisfied():
    """If r ≤ threshold at first call, latch on immediately, no force applied."""
    c = AFIRConstraint(
        formed=[(0, 1)], broken=[],
        alpha_formed=[1.0], alpha_broken=[],
        formed_thresholds=[1.5], broken_thresholds=[],
    )
    f = np.zeros((2, 3))
    c.adjust_forces(_atoms_along_x(1.2), f)
    assert c.formed_latched == [True]
    np.testing.assert_allclose(f, 0, atol=1e-12)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_afir_constraint.py::test_formed_latch_activates_when_threshold_reached -v`
Expected: FAIL — current `adjust_forces` always applies force, never sets latch.

- [ ] **Step 3: Add latch to formed loop**

`reactx/artificial_force.py` の `AFIRConstraint.adjust_forces` の **formed 部分のみ** を以下に置き換える:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_afir_constraint.py -v`
Expected: PASS (6 passed including the 3 from Task 2.1).

- [ ] **Step 5: Commit**

```bash
git add tests/test_afir_constraint.py reactx/artificial_force.py
git commit -m "feat(afir): add sticky latch for formed pairs (Phase 9 step 2.2)"
```

### Task 2.3: Sticky latch — broken direction (TDD)

**Files:**
- Modify: `tests/test_afir_constraint.py`
- Modify: `reactx/artificial_force.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_afir_constraint.py` の末尾に追記:

```python
def test_broken_latch_activates_when_threshold_reached():
    c = AFIRConstraint(
        formed=[], broken=[(0, 1)],
        alpha_formed=[], alpha_broken=[1.0],
        formed_thresholds=[], broken_thresholds=[3.0],
    )

    # r=2.0 < 3.0 → force on, latch off
    f = np.zeros((2, 3))
    c.adjust_forces(_atoms_along_x(2.0), f)
    assert c.broken_latched == [False]
    assert not np.allclose(f, 0)

    # r=3.5 ≥ 3.0 → latch ON, force=0
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

    # Bond comes back close: still latched, force=0
    f = np.zeros((2, 3))
    c.adjust_forces(_atoms_along_x(2.0), f)
    assert c.broken_latched == [True]
    np.testing.assert_allclose(f, 0, atol=1e-12)


def test_per_pair_latch_independence():
    """Two formed pairs with different thresholds latch independently."""
    atoms = Atoms(
        "CCCC",
        positions=[
            [0.0, 0.0, 0.0],   # 0
            [1.4, 0.0, 0.0],   # 1: close to 0
            [0.0, 0.0, 5.0],   # 2
            [3.0, 0.0, 5.0],   # 3: far from 2
        ],
    )
    c = AFIRConstraint(
        formed=[(0, 1), (2, 3)], broken=[],
        alpha_formed=[1.0, 1.0], alpha_broken=[],
        formed_thresholds=[1.5, 1.5], broken_thresholds=[],
    )
    f = np.zeros((4, 3))
    c.adjust_forces(atoms, f)
    # pair 0-1: r=1.4 ≤ 1.5 → latched, no force
    # pair 2-3: r=3.0 > 1.5 → not latched, force on
    assert c.formed_latched == [True, False]
    np.testing.assert_allclose(f[0], 0, atol=1e-12)
    np.testing.assert_allclose(f[1], 0, atol=1e-12)
    assert not np.allclose(f[2], 0)
    assert not np.allclose(f[3], 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_afir_constraint.py::test_broken_latch_activates_when_threshold_reached -v`
Expected: FAIL — broken loop has no latch logic yet.

- [ ] **Step 3: Add latch to broken loop**

`reactx/artificial_force.py` の `AFIRConstraint.adjust_forces` の **broken 部分** を以下に置き換える:

```python
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

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_afir_constraint.py -v`
Expected: PASS (9 passed).

- [ ] **Step 5: Commit**

```bash
git add tests/test_afir_constraint.py reactx/artificial_force.py
git commit -m "feat(afir): add sticky latch for broken pairs and per-pair independence (Phase 9 step 2.3)"
```

### Task 2.4: Validation (negative α, length mismatch, r→0 fail-fast)

**Files:**
- Modify: `tests/test_afir_constraint.py`
- Modify: `reactx/artificial_force.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_afir_constraint.py` の末尾に追記:

```python
def test_negative_alpha_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        AFIRConstraint(
            formed=[(0, 1)], broken=[],
            alpha_formed=[-0.1], alpha_broken=[],
            formed_thresholds=[1.5], broken_thresholds=[],
        )


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
    """r < 1e-6 → ValueError instead of silently dividing by ~0."""
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
    # Empty AND-set is True (vacuous truth)
    assert c.all_latched() is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_afir_constraint.py::test_negative_alpha_rejected -v`
Expected: FAIL — no validation yet.

- [ ] **Step 3: Add validation + helper to `__init__` and `_geom`**

`reactx/artificial_force.py` の `AFIRConstraint.__init__` を **完全に** 以下に置き換える:

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

`_geom` に r→0 ガード追加:

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

`all_latched` ヘルパ追加 (`get_indices` の前に挿入):

```python
    def all_latched(self) -> bool:
        return all(self.formed_latched) and all(self.broken_latched)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_afir_constraint.py -v`
Expected: PASS (16 passed).

- [ ] **Step 5: Commit**

```bash
git add tests/test_afir_constraint.py reactx/artificial_force.py
git commit -m "feat(afir): validate AFIRConstraint inputs and add r→0 fail-fast (Phase 9 step 2.4)"
```

### Task 2.5: Finite-difference test against V=±α·r artificial potential

**Files:**
- Modify: `tests/test_afir_constraint.py`

- [ ] **Step 1: Write the failing test**

`tests/test_afir_constraint.py` の末尾に追記:

```python
def test_force_matches_finite_difference_of_artificial_potential():
    """AFIRConstraint.adjust_forces == -∇V where

         V_pair = +α · r_ij  (formed, sign convention: compress)
         V_pair = -α · r_ij  (broken, sign convention: expand)

    Each pair must be evaluated with a *fresh* AFIRConstraint instance
    because adjust_forces mutates latch state. Probe in the active
    region (r > formed_threshold or r < broken_threshold).
    """
    rng = np.random.default_rng(42)

    # Random 4-atom configuration in active region for both pair sets
    pos = rng.uniform(-2.0, 2.0, size=(4, 3))
    pos[1] += [3.0, 0.0, 0.0]   # formed pair (0,1) ~3 Å apart, > thr 1.5
    pos[3] += [0.5, 0.0, 0.0]   # broken pair (2,3) ~0.5 Å, < thr 3.0
    atoms = Atoms("CCCC", positions=pos)

    formed = [(0, 1)]
    broken = [(2, 3)]
    alpha_f = [1.5]
    alpha_b = [2.0]
    thr_f = [1.5]
    thr_b = [3.0]

    def constraint() -> AFIRConstraint:
        return AFIRConstraint(
            formed=formed, broken=broken,
            alpha_formed=alpha_f, alpha_broken=alpha_b,
            formed_thresholds=thr_f, broken_thresholds=thr_b,
        )

    # Analytical forces from AFIRConstraint
    f_analytical = np.zeros((4, 3))
    constraint().adjust_forces(atoms, f_analytical)

    # Finite-difference of artificial potential V = +α·r_01 - α·r_23
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

- [ ] **Step 2: Run test to verify it passes (per-pair F=±α·d̂ should match analytically)**

Run: `pytest tests/test_afir_constraint.py::test_force_matches_finite_difference_of_artificial_potential -v`
Expected: PASS (per-pair AFIR is exactly the gradient of `V_pair = ±α·r_ij`)

- [ ] **Step 3: Commit**

```bash
git add tests/test_afir_constraint.py
git commit -m "test(afir): assert force matches finite-diff of V=±α·r artificial potential (Phase 9 step 2.5)"
```

### Task 2.6: `build_afir_constraint` factory + broadcasting (TDD)

**Files:**
- Modify: `tests/test_afir_constraint.py`
- Modify: `reactx/artificial_force.py`

- [ ] **Step 1: Write the failing tests**

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
    c = cs[0]
    assert c.alpha_formed == [0.5, 0.5]


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

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_afir_constraint.py::test_build_afir_returns_empty_when_no_pairs -v`
Expected: FAIL — `build_afir_constraint` not defined.

- [ ] **Step 3: Implement `build_afir_constraint` and `_broadcast`**

`reactx/artificial_force.py` の **末尾** に追加:

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
    """Broadcast scalar α to per-pair lists, return 0- or 1-element list.

    Returns `[]` when both `formed` and `broken` are empty (no AFIR force
    needed). Otherwise returns a single-element list containing the
    AFIRConstraint instance.
    """
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
    if isinstance(value, list):
        if len(value) != n:
            raise ValueError(f"{key} list length {len(value)} != n_pairs {n}")
        return [float(v) for v in value]
    return [float(value)] * n
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_afir_constraint.py -v`
Expected: PASS (21 passed).

- [ ] **Step 5: Commit**

```bash
git add tests/test_afir_constraint.py reactx/artificial_force.py
git commit -m "feat(afir): add build_afir_constraint factory with scalar-to-list broadcasting (Phase 9 step 2.6)"
```

---

## Stage 3: Drop legacy Hookean / PullApart / build_restraints / DEFAULT_R_FORM

### Task 3.1: Remove legacy classes and helpers

**Files:**
- Modify: `reactx/artificial_force.py` (delete top half, keep AFIRConstraint and build_afir_constraint)
- Delete (if exists): `tests/test_pull_apart.py`
- Modify: `tests/test_artificial_force.py` (delete tests for Hookean / PullApart / build_restraints / DEFAULT_R_FORM / lookup_r_form)

- [ ] **Step 1: Delete legacy code from `reactx/artificial_force.py`**

ファイル先頭から `def build_afir_constraint` の直前まで全て削除。残すのは module docstring (最新化)、imports (numpy / FixConstraint)、`AFIRConstraint` クラス、`build_afir_constraint`、`_broadcast_alpha`。

新しいモジュール冒頭は次のようになる:

```python
"""Per-pair AFIR force with sticky per-pair latch.

Replaces Phase Re1's Hookean+PullApart hybrid. Each user-specified pair
gets an independent constant-magnitude force gated by a sticky latch
on a per-pair distance threshold. See:

    docs/superpowers/specs/2026-05-08-afir-force-design.md
"""
from __future__ import annotations

import numpy as np
from ase.constraints import FixConstraint


class AFIRConstraint(FixConstraint):
    ...
```

- [ ] **Step 2: Delete legacy tests**

```bash
rm -f tests/test_pull_apart.py
```

`tests/test_artificial_force.py` を全面書き換え (旧 Hookean / PullApart / build_restraints テストを削除):

```python
"""Re-export tests for the AFIR force module.

The substantive AFIRConstraint tests live in
`tests/test_afir_constraint.py`. This file exists to keep the
top-level module importable and surface the public symbols.
"""
from reactx.artificial_force import AFIRConstraint, build_afir_constraint


def test_module_exports_public_api():
    assert AFIRConstraint is not None
    assert callable(build_afir_constraint)
```

- [ ] **Step 3: Run tests to verify deletion didn't break anything**

Run: `pytest tests/test_afir_constraint.py tests/test_artificial_force.py tests/test_covalent_radii.py -v`
Expected: PASS (22 passed).

- [ ] **Step 4: Verify other modules don't import the deleted names**

Run: `grep -rn "from reactx.artificial_force import" reactx/ tests/`
Expected output should only show `AFIRConstraint` / `build_afir_constraint` after this step. If any file still imports `Hookean`, `PullApart`, `build_restraints`, `DEFAULT_R_FORM`, `lookup_r_form`, those imports must be removed in subsequent stages (config.py, cli.py).

- [ ] **Step 5: Commit**

```bash
git add reactx/artificial_force.py tests/test_artificial_force.py
git rm -f tests/test_pull_apart.py
git commit -m "chore(afir): drop Hookean / PullApart / build_restraints / DEFAULT_R_FORM (Phase 9 step 3.1)"
```

---

## Stage 4: Config schema (`[afir]` + `[scoring]`)

### Task 4.1: New AFIRSection / ScoringSection / ReactionConfig dataclasses

**Files:**
- Modify: `reactx/config.py` (replace dataclasses + validation)
- Create: `tests/test_config_afir.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_config_afir.py`:

```python
"""Tests for Phase 9 [afir] + [scoring] schema.

Spec: docs/superpowers/specs/2026-05-08-afir-force-design.md §4.4
"""
from pathlib import Path

import pytest

from reactx.config import ConfigError, load_config


def _write(tmp_path: Path, body: str) -> Path:
    rxn = tmp_path / "x.rxn"
    rxn.write_text("placeholder")
    (tmp_path / "x.rxn.toml").write_text(body)
    return rxn


def test_load_minimal_sn2_schema(tmp_path: Path):
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
    assert cfg.description == "SN2"
    assert cfg.formed == ((1, 3),)
    assert cfg.broken == ((1, 2),)
    assert cfg.afir.alpha_formed == 0.7
    assert cfg.afir.alpha_broken == 0.5
    assert cfg.afir.max_relax_steps == 100
    assert cfg.scoring.r_broken_threshold == 4.0
    assert cfg.scoring.r_formed_threshold is None  # default = auto


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


def test_load_da_with_empty_broken_omits_r_broken_threshold(tmp_path: Path):
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
    assert cfg.scoring.r_broken_threshold is None  # omitted, broken=[]


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
    with pytest.raises(ConfigError, match=r"\[restraints\]"):
        load_config(rxn)


def test_alpha_zero_rejected_for_nonempty_pair(tmp_path: Path):
    rxn = _write(tmp_path, """
description = "broken α=0"
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


def test_old_keys_in_top_level_rejected(tmp_path: Path):
    rxn = _write(tmp_path, """
description = "old key"
formed = [[1, 2]]
broken = []
k_form = 0.5

[afir]
alpha_formed = 1.0
alpha_broken = 0.0
max_relax_steps = 100
""")
    with pytest.raises(ConfigError):
        load_config(rxn)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config_afir.py -v 2>&1 | head -30`
Expected: FAIL — `load_config` still implements the old `[restraints]` schema.

- [ ] **Step 3: Rewrite `reactx/config.py`**

`reactx/config.py` を **全面書き換え** (ConfigError は §1.3 で既に追加済み、それも含めて以下に置き換え):

```python
"""Per-reaction sidecar TOML config (`<rxn_path>.toml`).

Phase 9 schema: [afir] + [scoring] sections.
Spec: docs/superpowers/specs/2026-05-08-afir-force-design.md §4.4
"""
from __future__ import annotations

import math
import tomllib
from collections.abc import Sequence
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
    # Reject obsolete keys with a helpful message
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


def _normalize_alpha(
    value, *, expected_count: int, key: str, source: str,
) -> float | tuple[float, ...]:
    """Normalize `alpha_formed` / `alpha_broken` per spec §4.4.

    - When `expected_count == 0`: scalar 0 / empty list / any scalar value /
      any list is accepted, but the resolved tuple is `()` (broadcast result
      = empty). The value is essentially ignored.
    - When `expected_count > 0`: scalar must be > 0; list must have length
      == expected_count, all elements > 0. Zero is rejected because non-empty
      pairs with α=0 would never latch (spec §4.4).
    """
    if expected_count == 0:
        # Anything is acceptable; broadcast result is empty
        return tuple()

    if isinstance(value, list):
        if len(value) != expected_count:
            raise ConfigError(
                f"{source}: '[afir].{key}' list length {len(value)} != "
                f"expected {expected_count}"
            )
        normalized: list[float] = []
        for v in value:
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ConfigError(
                    f"{source}: '[afir].{key}' contains non-finite or "
                    f"non-numeric value: {v!r}"
                )
            if v <= 0:
                raise ConfigError(
                    f"{source}: '[afir].{key}' must be > 0 for non-empty "
                    f"pair set; got {v} (spec §4.4 forbids α=0 because the "
                    f"pair would never latch)"
                )
            normalized.append(float(v))
        return tuple(normalized)

    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ConfigError(
            f"{source}: '[afir].{key}' must be a finite number (got {value!r})"
        )
    if value <= 0:
        raise ConfigError(
            f"{source}: '[afir].{key}' must be > 0 for non-empty pair set "
            f"(got {value}; spec §4.4 forbids α=0)"
        )
    return float(value)


def _build_scoring(
    raw: dict, *, formed_count: int, broken_count: int, source: str,
) -> ScoringSection:
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
        r_broken_threshold=r_broken,
        r_formed_threshold=r_formed,
    )


def _normalize_threshold(
    value, *, expected_count: int, key: str, source: str,
    required_when_pairs_present: bool,
) -> float | tuple[float, ...] | None:
    if expected_count == 0:
        # Per spec §4.4: silently ignore explicit values when there are no
        # pairs (warning is reserved for future phase).
        return None

    if value is None:
        if required_when_pairs_present:
            raise ConfigError(
                f"{source}: '[scoring].{key}' is required when there are "
                f"non-empty pairs"
            )
        return None  # default = auto-derive in scoring layer

    if isinstance(value, list):
        if len(value) != expected_count:
            raise ConfigError(
                f"{source}: '[scoring].{key}' list length {len(value)} != "
                f"expected {expected_count}"
            )
        normalized: list[float] = []
        for v in value:
            if not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                raise ConfigError(
                    f"{source}: '[scoring].{key}' must contain finite, "
                    f"positive values; got {v!r}"
                )
            normalized.append(float(v))
        return tuple(normalized)

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

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config_afir.py -v`
Expected: PASS (all 10 tests)

- [ ] **Step 5: Commit**

```bash
git add reactx/config.py tests/test_config_afir.py
git commit -m "feat(config): rewrite schema as [afir] + [scoring] (Phase 9 step 4.1)"
```

### Task 4.2: Update or remove `tests/test_config.py` (legacy)

**Files:**
- Modify or delete: `tests/test_config.py`

- [ ] **Step 1: Inspect what `tests/test_config.py` contains**

Run: `grep -n "def test_" tests/test_config.py | head -20`
Note: tests referencing `[restraints]`, `k_form`, `k_broken`, `r_broken`, `r_form` are obsolete.

- [ ] **Step 2: Replace `tests/test_config.py` with a smoke test only**

`tests/test_config.py` を全面書き換え (旧 [restraints] テストはすべて削除、本流の検証は `tests/test_config_afir.py` で行う):

```python
"""Smoke test: load_config can parse the on-disk example TOMLs.

The substantive Phase 9 schema tests live in `tests/test_config_afir.py`.
"""
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
    """Each example TOML must validate after Phase 9 migration."""
    cfg = load_config(EXAMPLES / rxn_name)
    assert cfg.description
    assert cfg.afir.max_relax_steps > 0
```

- [ ] **Step 3: Note this test will fail until examples are migrated (Stage 8)**

This is intentional — the smoke test acts as a tripwire to confirm Stage 8 migration is complete. **Do not run this test now**; it will be exercised at the end of Stage 8.

- [ ] **Step 4: Commit**

```bash
git add tests/test_config.py
git commit -m "test(config): replace [restraints] tests with example load smoke (Phase 9 step 4.2)"
```

---

## Stage 5: Scoring helpers (`reached_product`, `product_distance_residual`, `count_initial_latched`)

### Task 5.1: `resolve_formed_thresholds` / `resolve_broken_thresholds`

**Files:**
- Modify: `reactx/scoring.py`
- Modify: `tests/test_scoring.py` (full rewrite at Step 3)

- [ ] **Step 1: Write the failing tests (replace existing `tests/test_scoring.py` body)**

`tests/test_scoring.py` を **全面書き換え**:

```python
"""Tests for Phase 9 per-pair scoring threshold + reached_product."""
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


# ------- resolve_*_thresholds ------------------------------------------------

def test_resolve_formed_thresholds_default_is_1_15_times_rsum():
    a = Atoms("CC", positions=[[0, 0, 0], [1.5, 0, 0]])
    thrs = resolve_formed_thresholds(a, formed=[(0, 1)], override=None)
    # Cordero C = 0.76, sum = 1.52, * 1.15 = 1.748
    assert thrs == pytest.approx([1.748], abs=1e-6)


def test_resolve_formed_thresholds_scalar_override_broadcasts():
    a = Atoms("CCC", positions=np.zeros((3, 3)))
    thrs = resolve_formed_thresholds(a, formed=[(0, 1), (1, 2)], override=2.0)
    assert thrs == [2.0, 2.0]


def test_resolve_formed_thresholds_list_override_passthrough():
    a = Atoms("CCC", positions=np.zeros((3, 3)))
    thrs = resolve_formed_thresholds(a, formed=[(0, 1), (1, 2)], override=(1.5, 1.8))
    assert thrs == [1.5, 1.8]


def test_resolve_formed_thresholds_length_mismatch_raises():
    a = Atoms("CCC", positions=np.zeros((3, 3)))
    with pytest.raises(ConfigError, match="r_formed_threshold"):
        resolve_formed_thresholds(a, formed=[(0, 1), (1, 2)], override=(1.5,))


def test_resolve_formed_thresholds_empty_formed():
    a = Atoms("CC", positions=np.zeros((2, 3)))
    assert resolve_formed_thresholds(a, formed=[], override=None) == []


def test_resolve_broken_thresholds_required_when_nonempty():
    a = Atoms("CC", positions=np.zeros((2, 3)))
    with pytest.raises(ConfigError, match="r_broken_threshold"):
        resolve_broken_thresholds(a, broken=[(0, 1)], override=None)


def test_resolve_broken_thresholds_empty_broken_returns_empty():
    a = Atoms("CC", positions=np.zeros((2, 3)))
    assert resolve_broken_thresholds(a, broken=[], override=None) == []


# ------- reached_product -----------------------------------------------------

def test_reached_product_true_when_all_thresholds_satisfied():
    a = _atoms_cc(1.5)
    assert reached_product(
        a, formed=[(0, 1)], broken=[],
        formed_thresholds=[1.6], broken_thresholds=[],
    )


def test_reached_product_false_when_formed_too_far():
    a = _atoms_cc(2.0)
    assert not reached_product(
        a, formed=[(0, 1)], broken=[],
        formed_thresholds=[1.6], broken_thresholds=[],
    )


def test_reached_product_false_when_broken_too_close():
    a = _atoms_cc(2.0)
    assert not reached_product(
        a, formed=[], broken=[(0, 1)],
        formed_thresholds=[], broken_thresholds=[3.0],
    )


def test_reached_product_true_for_empty_pairs():
    a = _atoms_cc(2.0)
    assert reached_product(a, [], [], [], [])


# ------- product_distance_residual ------------------------------------------

def test_residual_zero_when_reached():
    a = _atoms_cc(1.4)
    r = product_distance_residual(
        a, formed=[(0, 1)], broken=[],
        formed_thresholds=[1.6], broken_thresholds=[],
    )
    assert r == 0.0


def test_residual_positive_when_formed_overshoot():
    a = _atoms_cc(2.0)
    r = product_distance_residual(
        a, formed=[(0, 1)], broken=[],
        formed_thresholds=[1.6], broken_thresholds=[],
    )
    # max(2.0 - 1.6, 0)² = 0.16
    assert r == pytest.approx(0.16, abs=1e-12)


def test_residual_positive_when_broken_too_close():
    a = _atoms_cc(2.0)
    r = product_distance_residual(
        a, formed=[], broken=[(0, 1)],
        formed_thresholds=[], broken_thresholds=[3.0],
    )
    # max(3.0 - 2.0, 0)² = 1.0
    assert r == pytest.approx(1.0, abs=1e-12)


def test_residual_sums_violations():
    a = _atoms_cc(2.0)
    r = product_distance_residual(
        a, formed=[(0, 1)], broken=[(0, 1)],
        formed_thresholds=[1.6], broken_thresholds=[3.0],
    )
    # 0.16 + 1.0 = 1.16
    assert r == pytest.approx(1.16, abs=1e-12)


# ------- count_initial_latched -----------------------------------------------

def test_count_initial_latched_none_satisfied():
    a = _atoms_cc(2.0)
    counts = count_initial_latched(
        a, formed=[(0, 1)], broken=[(0, 1)],
        formed_thresholds=[1.6], broken_thresholds=[3.0],
    )
    # neither: r=2.0 > 1.6 (formed not yet), r=2.0 < 3.0 (broken not yet)
    assert counts == {"formed": 0, "broken": 0}


def test_count_initial_latched_both_satisfied():
    a = _atoms_cc(1.5)
    counts = count_initial_latched(
        a, formed=[(0, 1)], broken=[(0, 1)],
        formed_thresholds=[1.6], broken_thresholds=[1.4],
    )
    assert counts == {"formed": 1, "broken": 1}


# ------- score_trials with residual fallback ---------------------------------

def _trial(idx: int, *, reached: bool, peak: float, residual: float = 0.0) -> TrialResult:
    return TrialResult(
        trial_idx=idx,
        direction=np.array([0.0, 0.0, 1.0]),
        frames=[],
        energies=[],
        reached_product=reached,
        peak_energy=peak,
        n_steps=10,
        product_distance_residual=residual,
        formed_latch_count=0,
        broken_latch_count=0,
        initial_latched_formed=0,
        initial_latched_broken=0,
    )


def test_score_trials_prefers_reached_with_lowest_peak():
    trials = [
        _trial(0, reached=True, peak=10.0),
        _trial(1, reached=True, peak=5.0),
        _trial(2, reached=False, peak=1.0),  # lower peak but didn't reach
    ]
    best = score_trials(trials)
    assert best.trial_idx == 1


def test_score_trials_fallback_uses_residual_then_peak():
    trials = [
        _trial(0, reached=False, peak=1.0, residual=10.0),
        _trial(1, reached=False, peak=5.0, residual=2.0),
        _trial(2, reached=False, peak=2.0, residual=2.0),
    ]
    best = score_trials(trials)
    # residual ties between idx 1 and 2 → pick lowest peak
    assert best.trial_idx == 2


def test_score_trials_empty_raises():
    with pytest.raises(ValueError):
        score_trials([])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_scoring.py -v 2>&1 | head -10`
Expected: FAIL — `count_initial_latched`, `product_distance_residual`, `resolve_*_thresholds` not yet defined.

- [ ] **Step 3: Implement `reactx/scoring.py` (full rewrite)**

`reactx/scoring.py` を **全面書き換え**:

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
"""Default formed threshold: r ≤ 1.15 × Rsum_Cordero (matches Blender's
1.1 × Rsum bond-tol with a small slack for transient stretching)."""


@dataclass
class TrialResult:
    """Outcome of a single placement / relaxation trial.

    Phase 9 additions: `product_distance_residual` (least-bad fallback
    tiebreaker), `formed_latch_count` / `broken_latch_count` (debug:
    AFIR latches activated by relax end), `initial_latched_*`
    (debug: pairs already satisfying threshold at relax start).

    `reached_product` (= final-frame per-pair check) is the sole
    success criterion. Latch counters are diagnostic only.
    """

    trial_idx: int
    direction: np.ndarray
    frames: list[Atoms]
    energies: list[float]
    reached_product: bool
    peak_energy: float
    n_steps: int
    # Phase 9 fields
    product_distance_residual: float
    formed_latch_count: int
    broken_latch_count: int
    initial_latched_formed: int
    initial_latched_broken: int


# --- Threshold resolution ---------------------------------------------------

def resolve_formed_thresholds(
    atoms: Atoms,
    formed: list[tuple[int, int]],
    override,
) -> list[float]:
    """Per-pair formed thresholds (Å). None override → 1.15 × Rsum_Cordero."""
    if not formed:
        return []
    if override is None:
        cov = cordero_radii_for_atoms(atoms)
        return [(cov[i] + cov[j]) * COVALENT_FORMED_TOLERANCE for (i, j) in formed]
    return _broadcast_threshold(override, len(formed), key="r_formed_threshold")


def resolve_broken_thresholds(
    atoms: Atoms,
    broken: list[tuple[int, int]],
    override,
) -> list[float]:
    """Per-pair broken thresholds (Å). Required when `broken != []`."""
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
            raise ConfigError(
                f"{key} list length {len(value)} != n_pairs {n}"
            )
        return [float(v) for v in value]
    return [float(value)] * n


# --- Per-pair distance evaluation -------------------------------------------

def reached_product(
    final_atoms: Atoms,
    formed: list[tuple[int, int]],
    broken: list[tuple[int, int]],
    formed_thresholds: list[float],
    broken_thresholds: list[float],
) -> bool:
    """True iff all formed pairs satisfy r ≤ threshold AND all broken pairs
    satisfy r ≥ threshold at `final_atoms` (sole Phase 9 success criterion)."""
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
    """Σ (max(d-thr, 0))² for formed + Σ (max(thr-d, 0))² for broken (Å²).

    0.0 iff `reached_product` is True. Used as least-bad fallback
    tiebreaker.
    """
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
    """Count pairs that already satisfy their threshold at relax start.

    Recorded as a debug diagnostic — non-zero counts flag trials where
    placement happened to land already inside the product manifold for
    some pair (AFIR will never push that pair). Does NOT affect scoring.
    """
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


# --- Best-trial selection ---------------------------------------------------

def score_trials(results: list[TrialResult]) -> TrialResult:
    """Return the best TrialResult.

    Preference order:
        1. `reached_product=True` group → lowest `peak_energy`
        2. else (least-bad fallback) → lowest `product_distance_residual`,
           tied → lowest `peak_energy`
    """
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
    """Return the trial_idx of the best TrialResult (delegates to score_trials)."""
    if not trials:
        raise ValueError("select_best_trial called with empty list")
    return score_trials(trials).trial_idx
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_scoring.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add reactx/scoring.py tests/test_scoring.py
git commit -m "feat(scoring): per-pair thresholds + residual fallback + initial_latched (Phase 9 step 5.1)"
```

---

## Stage 6: path_relax — return `final_constraint_state`, snapshot fix

### Task 6.1: Update `relax_with_restraints` signature + `_snapshot`

**Files:**
- Modify: `reactx/path_relax.py`
- Modify: `tests/test_path_relax.py`

- [ ] **Step 1: Inspect current path_relax tests**

Run: `cat tests/test_path_relax.py`
Note any tests that destructure 2-tuple from `relax_with_restraints`.

- [ ] **Step 2: Write the failing test**

Add to `tests/test_path_relax.py` (preserve existing tests, but update destructuring):

```python
import numpy as np
from ase import Atoms
from ase.calculators.lj import LennardJones

from reactx.artificial_force import AFIRConstraint
from reactx.path_relax import relax_with_restraints


def test_relax_returns_three_tuple_with_constraint_state():
    """Phase 9: relax_with_restraints returns (frames, energies, final_state)."""
    a = Atoms("CC", positions=[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    c = AFIRConstraint(
        formed=[(0, 1)], broken=[],
        alpha_formed=[0.5], alpha_broken=[],
        formed_thresholds=[1.5], broken_thresholds=[],
    )
    result = relax_with_restraints(
        a, [c], LennardJones(),
        max_steps=20, fmax=0.5, traj_stride=2,
    )
    assert isinstance(result, tuple)
    assert len(result) == 3
    frames, energies, state = result
    assert isinstance(frames, list)
    assert isinstance(energies, list)
    assert isinstance(state, dict)
    assert "formed_latched" in state
    assert isinstance(state["formed_latched"], list)
    assert len(state["formed_latched"]) == 1


def test_relax_returns_empty_state_when_no_afir():
    a = Atoms("CC", positions=[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    frames, energies, state = relax_with_restraints(
        a, [], LennardJones(), max_steps=5, fmax=0.5, traj_stride=2,
    )
    assert state == {}


def test_snapshot_has_no_constraint():
    """trajectory frames should not carry the AFIRConstraint."""
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
    # First and last frames should be constraint-free
    for f in frames:
        assert len(f.constraints) == 0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_path_relax.py::test_relax_returns_three_tuple_with_constraint_state -v`
Expected: FAIL — current `relax_with_restraints` returns a 2-tuple.

- [ ] **Step 4: Update `reactx/path_relax.py`**

`reactx/path_relax.py` を以下に置き換える:

```python
"""Constrained relaxation that yields a trajectory of frames.

Phase 9 changes:
- 戻り値が 3-tuple `(frames, energies, final_constraint_state)` に変更。
  `final_constraint_state` は AFIRConstraint の latch 状態 (`formed_latched`,
  `broken_latched` の list[bool]) を含む dict、または AFIR が無いとき空 dict。
- `_snapshot()` で constraint も外す (latch state や custom constraint
  serialization が trajectory.xyz に混入するのを防ぐ)。

ASE は `set_constraint()` 時に constraint instance を内部 list に保管するが、
copy せず参照を保持する (ASE 3.x の現実装)。とはいえ将来の挙動変化に対する
ロバスト性のため、`atoms.constraints[0]` 経由で latch state を読み取る。
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
    """Run FIRE under given restraints; return (frames, energies, final_state).

    `final_state` is a dict with keys 'formed_latched' / 'broken_latched'
    (each `list[bool]`) when the restraints contain an AFIRConstraint,
    or an empty dict otherwise.
    """
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
    """Detach calculator and constraints so frame is a plain serializable Atoms."""
    a = atoms.copy()
    a.calc = None
    a.set_constraint([])
    return a
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_path_relax.py -v`
Expected: PASS (existing tests pass with updated destructuring + 3 new tests).

If existing tests in `tests/test_path_relax.py` still destructure the 2-tuple form, update them:

```python
# Before:
frames, energies = relax_with_restraints(...)
# After:
frames, energies, _ = relax_with_restraints(...)
```

- [ ] **Step 6: Commit**

```bash
git add reactx/path_relax.py tests/test_path_relax.py
git commit -m "feat(path_relax): return final_constraint_state, strip constraint in snapshot (Phase 9 step 6.1)"
```

---

## Stage 7: CLI orchestration

### Task 7.1: Update `cli.py` imports + threshold resolution + AFIR wiring

**Files:**
- Modify: `reactx/cli.py`

- [ ] **Step 1: Update imports**

`reactx/cli.py` の 14-36 行を以下に置き換える (旧 imports を全削除):

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

- [ ] **Step 2: Find the trial-loop body**

Run: `grep -n "build_restraints\|relax_with_restraints\|reached_product\|TrialResult(" reactx/cli.py`
Note line numbers — these are the orchestration points to update.

- [ ] **Step 3: Replace `build_restraints` call with AFIR construction**

In `reactx/cli.py`, locate the per-trial loop (around line 250-290 in the existing file). The pattern looks like:

```python
restraints = build_restraints(atoms_init, formed, broken, ...)
frames, energies = relax_with_restraints(...)
ok = reached_product(frames[-1], formed, broken, ...)
```

Replace this block with:

```python
# Resolve per-pair thresholds (shared between AFIR latch and reached_product)
ft = resolve_formed_thresholds(
    atoms_init, list(formed), cfg.scoring.r_formed_threshold,
)
bt = resolve_broken_thresholds(
    atoms_init, list(broken), cfg.scoring.r_broken_threshold,
)

# Initial latch observation (debug only — does not affect scoring)
initial_latched = count_initial_latched(
    atoms_init, list(formed), list(broken), ft, bt,
)

# Resolve per-pair α (broadcast scalar to list)
af = resolve_alpha_formed(cfg)
ab = resolve_alpha_broken(cfg)

afir_cs = build_afir_constraint(
    atoms_init, list(formed), list(broken),
    alpha_formed=af, alpha_broken=ab,
    formed_thresholds=ft, broken_thresholds=bt,
)

frames, energies, final_state = relax_with_restraints(
    atoms_init, afir_cs, calc,
    max_steps=cfg.afir.max_relax_steps,
    fmax=args.relax_fmax,
    traj_stride=args.traj_stride,
)

final = frames[-1]
ok = reached_product(final, list(formed), list(broken), ft, bt)
residual = product_distance_residual(final, list(formed), list(broken), ft, bt)
peak = max(energies) if energies else float("inf")
formed_latch_count = sum(final_state.get("formed_latched", []))
broken_latch_count = sum(final_state.get("broken_latched", []))

trials.append(TrialResult(
    trial_idx=trial.trial_idx,
    direction=trial.direction,
    frames=frames,
    energies=energies,
    reached_product=ok,
    peak_energy=peak,
    n_steps=len(frames),
    product_distance_residual=residual,
    formed_latch_count=formed_latch_count,
    broken_latch_count=broken_latch_count,
    initial_latched_formed=initial_latched["formed"],
    initial_latched_broken=initial_latched["broken"],
))
```

(Adjust variable names to match the existing loop. The key changes: use `cfg.afir.max_relax_steps` instead of `cfg.restraints.max_relax_steps`; pass `ft`/`bt` everywhere; record new TrialResult fields.)

- [ ] **Step 4: Update meta.json writer to include new fields**

Find `_emit_meta` (or similar) in `reactx/cli.py` (search for `"reached_product":`) and replace the per-trial dict with:

```python
{
    "trial_idx": t.trial_idx,
    "direction": t.direction.tolist(),
    "reached_product": t.reached_product,
    "peak_energy": t.peak_energy,
    "n_steps": t.n_steps,
    "alpha_formed": list(resolve_alpha_formed(cfg)),
    "alpha_broken": list(resolve_alpha_broken(cfg)),
    "formed_thresholds": list(ft),
    "broken_thresholds": list(bt),
    "product_distance_residual": t.product_distance_residual,
    "formed_latch_count": t.formed_latch_count,
    "broken_latch_count": t.broken_latch_count,
    "initial_latched_formed": t.initial_latched_formed,
    "initial_latched_broken": t.initial_latched_broken,
}
```

(The exact form depends on the existing structure — preserve other fields like `placement_kind`, `orientation`. Remove the obsolete `r_broken_target`, `r_form_target`, `k_form`, `k_broken`.)

- [ ] **Step 5: Catch ConfigError at CLI top-level**

Locate the top-level catch in `_cmd_run` (or `main`). Look for `except ValueError`. Add `ConfigError` for clarity:

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

- [ ] **Step 6: Run unit tests (NOT slow)**

Run: `pytest tests/test_cli.py tests/test_cli_unimolecular.py tests/test_cli_neb_refine_guard.py -v 2>&1 | head -50`

Note: existing CLI tests likely fail because they construct `TrialResult` with the old 7-field signature. Update each constructor in those tests to add the 5 new Phase 9 fields:

```python
TrialResult(
    ..., # existing 7 fields
    product_distance_residual=0.0,
    formed_latch_count=0,
    broken_latch_count=0,
    initial_latched_formed=0,
    initial_latched_broken=0,
)
```

This is mechanical — update each `TrialResult(` call in `tests/test_cli.py` and any other test that constructs one.

- [ ] **Step 7: Run `pytest` (everything except slow + blender) to confirm green**

Run: `pytest -m "not slow and not blender" -q 2>&1 | tail -20`
Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add reactx/cli.py tests/test_cli.py tests/test_cli_unimolecular.py tests/test_cli_neb_refine_guard.py
git commit -m "feat(cli): wire AFIR + threshold resolution + ConfigError catch (Phase 9 step 7.1)"
```

---

## Stage 8: Migrate 8 example TOMLs

Each task is mechanical: replace `[restraints]` with `[afir]` + `[scoring]` per spec §5.

### Task 8.1: `examples/sn2.rxn.toml`

- [ ] **Step 1: Rewrite**

`examples/sn2.rxn.toml` を全置換:

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

- [ ] **Step 2: Verify config loads**

Run: `python -c "from pathlib import Path; from reactx.config import load_config; print(load_config(Path('examples/sn2.rxn')))"`
Expected: `ReactionConfig(...)` printed without error.

- [ ] **Step 3: Commit**

```bash
git add examples/sn2.rxn.toml
git commit -m "examples: migrate sn2 to Phase 9 schema (Phase 9 step 8.1)"
```

### Task 8.2: `examples/proton_transfer.rxn.toml`

- [ ] **Step 1: Rewrite**

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

- [ ] **Step 2: Verify config loads** and **Commit**

```bash
python -c "from pathlib import Path; from reactx.config import load_config; load_config(Path('examples/proton_transfer.rxn'))" \
  && git add examples/proton_transfer.rxn.toml \
  && git commit -m "examples: migrate proton_transfer to Phase 9 schema (Phase 9 step 8.2)"
```

### Task 8.3: `examples/menshutkin.rxn.toml`

- [ ] **Step 1: Rewrite**

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

- [ ] **Step 2: Verify and commit**

```bash
python -c "from pathlib import Path; from reactx.config import load_config; load_config(Path('examples/menshutkin.rxn'))" \
  && git add examples/menshutkin.rxn.toml \
  && git commit -m "examples: migrate menshutkin to Phase 9 schema (Phase 9 step 8.3)"
```

### Task 8.4: `examples/e2.rxn.toml` (per-bond α_broken + r_broken_threshold)

- [ ] **Step 1: Rewrite**

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

- [ ] **Step 2: Verify and commit**

```bash
python -c "from pathlib import Path; from reactx.config import load_config; load_config(Path('examples/e2.rxn'))" \
  && git add examples/e2.rxn.toml \
  && git commit -m "examples: migrate e2 to Phase 9 per-bond schema (Phase 9 step 8.4)"
```

### Task 8.5: `examples/sn1_dissoc.rxn.toml` (formed=[])

- [ ] **Step 1: Inspect existing**

Run: `cat examples/sn1_dissoc.rxn.toml`
Note: keeps `formed = []` and `broken = [[1, 5]]`.

- [ ] **Step 2: Rewrite**

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

- [ ] **Step 3: Verify and commit**

```bash
python -c "from pathlib import Path; from reactx.config import load_config; load_config(Path('examples/sn1_dissoc.rxn'))" \
  && git add examples/sn1_dissoc.rxn.toml \
  && git commit -m "examples: migrate sn1_dissoc to Phase 9 schema (Phase 9 step 8.5)"
```

### Task 8.6: `examples/sn1_recomb.rxn.toml` (broken=[])

- [ ] **Step 1: Rewrite**

```toml
description = "SN1 step 2 recombination: tBu+ + Cl- -> tBuCl"
formed = [[1, 5]]
broken = []

[afir]
alpha_formed = 1.5
alpha_broken = 0.0
max_relax_steps = 200
```

(No `[scoring]` section needed: `r_broken_threshold` is omitted because `broken=[]`, and `r_formed_threshold` defaults to auto.)

- [ ] **Step 2: Verify and commit**

```bash
python -c "from pathlib import Path; from reactx.config import load_config; load_config(Path('examples/sn1_recomb.rxn'))" \
  && git add examples/sn1_recomb.rxn.toml \
  && git commit -m "examples: migrate sn1_recomb to Phase 9 schema (Phase 9 step 8.6)"
```

### Task 8.7: `examples/diels_alder_simple.rxn.toml`

- [ ] **Step 1: Rewrite**

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

- [ ] **Step 2: Verify and commit**

```bash
python -c "from pathlib import Path; from reactx.config import load_config; load_config(Path('examples/diels_alder_simple.rxn'))" \
  && git add examples/diels_alder_simple.rxn.toml \
  && git commit -m "examples: migrate diels_alder_simple to Phase 9 schema (Phase 9 step 8.7)"
```

### Task 8.8: `examples/diels_alder_endo.rxn.toml`

- [ ] **Step 1: Rewrite**

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

- [ ] **Step 2: Verify and commit**

```bash
python -c "from pathlib import Path; from reactx.config import load_config; load_config(Path('examples/diels_alder_endo.rxn'))" \
  && git add examples/diels_alder_endo.rxn.toml \
  && git commit -m "examples: migrate diels_alder_endo to Phase 9 schema (Phase 9 step 8.8)"
```

### Task 8.9: Run example-load smoke test

- [ ] **Step 1: Run smoke**

Run: `pytest tests/test_config.py -v`
Expected: PASS (all 8 examples load).

- [ ] **Step 2: Run unit + non-slow tests as a green checkpoint**

Run: `pytest -m "not slow and not blender" -q`
Expected: all green. If failures remain, fix before proceeding to Stage 9.

---

## Stage 9: Slow integration tests — per-reaction tuning

Each task runs one reaction's slow test, observes `meta.json`, and re-tunes `α` / threshold if needed. Order: shortest first to surface bugs cheaply.

### Task 9.1: SN1 dissoc (unimolecular, fastest)

**Files:**
- Modify: `tests/test_re3_sn1_dissoc.py` (update assertions for new meta.json schema)

- [ ] **Step 1: Update test assertions**

Locate `tests/test_re3_sn1_dissoc.py`, find any reference to `r_broken_target` / `k_form` / `k_broken` in meta.json checks. Replace with `formed_thresholds` / `broken_thresholds` / `alpha_formed` / `alpha_broken`. Keep the **primary success assertion** as `reached_product=True` (don't add `formed_latch_count` as a hard assertion — it's debug only).

- [ ] **Step 2: Run slow test**

Run: `pytest tests/test_re3_sn1_dissoc.py -v -m slow 2>&1 | tail -30`
Expected: PASS. If FAIL, inspect output for which trial failed which check, then re-tune `alpha_broken` (try 1.0 / 2.0) or `r_broken_threshold` (try 5.0).

- [ ] **Step 3: Commit if changes**

```bash
git add tests/test_re3_sn1_dissoc.py examples/sn1_dissoc.rxn.toml
git commit -m "test(re3_sn1_dissoc): pass under Phase 9 AFIR (Phase 9 step 9.1)"
```

### Task 9.2: SN2

- [ ] **Step 1: Update assertions** in `tests/test_re1_sn2.py` analogous to 9.1.
- [ ] **Step 2: Run slow** `pytest tests/test_re1_sn2.py -v -m slow 2>&1 | tail -30`. Re-tune α / threshold if FAIL.
- [ ] **Step 3: Commit**.

### Task 9.3: Proton transfer

- [ ] **Step 1: Update assertions** in `tests/test_re1_proton_transfer.py`.
- [ ] **Step 2: Run slow** `pytest tests/test_re1_proton_transfer.py -v -m slow 2>&1 | tail -30`.
- [ ] **Step 3: Commit**.

### Task 9.4: SN1 recomb

- [ ] **Step 1: Update assertions** in `tests/test_re4_sn1_recomb.py`.
- [ ] **Step 2: Run slow** `pytest tests/test_re4_sn1_recomb.py -v -m slow 2>&1 | tail -30`.
- [ ] **Step 3: Commit**.

### Task 9.5: Menshutkin

- [ ] **Step 1: Update assertions** in `tests/test_re1_menshutkin.py` and `tests/test_examples_menshutkin.py`.
- [ ] **Step 2: Run slow** `pytest tests/test_re1_menshutkin.py tests/test_examples_menshutkin.py -v -m slow 2>&1 | tail -40`.
- [ ] **Step 3: Commit**.

### Task 9.6: E2

- [ ] **Step 1: Update assertions** in `tests/test_re3_e2.py`.
- [ ] **Step 2: Run slow** `pytest tests/test_re3_e2.py -v -m slow 2>&1 | tail -30`. Watch for false positives where C-H "broken" trips early; if it does, increase `r_broken_threshold[0]` from 3.0 → 3.5.
- [ ] **Step 3: Commit**.

### Task 9.7: Diels-Alder simple

- [ ] **Step 1: Update assertions** in `tests/test_diels_alder_simple.py`.
- [ ] **Step 2: Run slow** `pytest tests/test_diels_alder_simple.py -v -m slow 2>&1 | tail -30`. Verify both formed bonds latch (check `meta.json.formed_latch_count == 2` as **debug** only, not a hard assertion).
- [ ] **Step 3: Commit**.

### Task 9.8: Diels-Alder endo

- [ ] **Step 1: Update assertions** in `tests/test_diels_alder_endo.py`.
- [ ] **Step 2: Run slow** `pytest tests/test_diels_alder_endo.py -v -m slow 2>&1 | tail -30`.
- [ ] **Step 3: Commit**.

### Task 9.9: Full slow-test sweep

- [ ] **Step 1: Run all slow tests**

Run: `pytest -m slow -q 2>&1 | tail -30`
Expected: all green. If any are still red, repeat the per-reaction tuning cycle.

- [ ] **Step 2: Commit any final tuning**

If `examples/*.rxn.toml` were re-tuned, commit them with explanatory message.

---

## Stage 10: Cleanup + README + PR

### Task 10.1: Decide on `reactx/align.py` fate

- [ ] **Step 1: Check whether `align_product_to_reactant` is still used**

Run: `grep -rn "align_product_to_reactant" reactx/ tests/`
Expected: only `reactx/cli.py` (NEB refine path) and `tests/test_align.py` reference it.

- [ ] **Step 2: Decision**

`reactx/cli.py` の `--neb-refine` 経路で使用継続 (Phase 10 で扱う) → **保留 (no change)**。本 phase では align.py を **削除しない**。

- [ ] **Step 3: Document in README**

`README.md` の Phase 9 セクション (作成予定) に「`align_product_to_reactant` は引き続き 1+1 NEB refine 用、Phase 10 で AFIR endpoint 直接利用への移行を検討」を 1 行明記する (Task 10.3 で対応)。

### Task 10.2: Inspect `reactx/cli.py` で残る legacy reference

- [ ] **Step 1: Search for any remaining old keys**

```bash
grep -n "k_form\|k_broken\|r_form\b\|r_broken\b\|build_restraints\|Hookean\|PullApart\|DEFAULT_R_FORM\|lookup_r_form" reactx/ tests/
```
Expected: zero hits in `reactx/`. If any remain in `tests/test_*.py` files updated above, clean them.

- [ ] **Step 2: Run full unit + non-slow tests**

Run: `pytest -m "not slow and not blender" -q`
Expected: all green.

- [ ] **Step 3: Commit if cleanup**

```bash
git add reactx/ tests/
git commit -m "chore: drop residual legacy AFIR references (Phase 9 step 10.2)"
```

### Task 10.3: README rewrite

- [ ] **Step 1: Update `README.md`**

`README.md` の主な書き換え箇所:

- 冒頭の「Generic Steric-Aware Reaction Path Engine」サブタイトル: 「per-pair AFIR with sticky latch」を追加
- 「対応反応」表 (Per-reaction `.rxn.toml` config 節): 旧 `k_form` / `k_broken` / `r_broken` / `r_form` 列を削除し、`alpha_formed` / `alpha_broken` / `r_broken_threshold` / `r_formed_threshold` 列に
- 「方針と限界」: Phase 9 で `[restraints]` セクション廃止、`[afir]` + `[scoring]` 必須を追記。`align_product_to_reactant` の Phase 10 移行予定も明記
- 「Wall-clock (実測)」: 各反応の wall-clock を Phase 9 で再測定した値に更新 (slow test 実行時の `meta.json` の `wall_clock_seconds` を反映)
- 「アーキテクチャ」: Hookean+PullApart → AFIRConstraint sticky latch に書き換え

- [ ] **Step 2: Verify README renders**

Run: `head -50 README.md`
Expected: clean Markdown, no stale Phase 8 references in the Phase 9-affected sections.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(readme): rewrite for Phase 9 AFIR + sticky latch (Phase 9 step 10.3)"
```

### Task 10.4: Phase 9 PR

- [ ] **Step 1: Confirm full test suite passes**

Run: `pytest -q 2>&1 | tail -10` (without `-m`, runs everything)
Expected: all green (slow + unit + blender if local Blender available).

- [ ] **Step 2: Push branch and open PR**

```bash
git push -u origin phase-9
gh pr create --base develop --title "Phase 9: per-pair AFIR with sticky latch + 1-stage relax" --body "$(cat <<'EOF'
## Summary
- Replace Hookean + PullApart with per-pair AFIR (constant force gated by sticky per-pair latch)
- Move scoring threshold (`r_broken_threshold` / `r_formed_threshold`) from `[restraints]` to `[scoring]`, share with AFIR latch trigger
- Add `product_distance_residual` for least-bad fallback
- 1-stage relax (no Stage A/B), `relax_with_restraints` returns 3-tuple including `final_constraint_state`
- Migrate 8 example TOMLs to new schema, retune α / threshold via slow tests

Spec: docs/superpowers/specs/2026-05-08-afir-force-design.md (v3.1)
Plan: docs/superpowers/plans/2026-05-08-afir-force-replacement.md

## Test plan
- [x] `pytest tests/test_afir_constraint.py tests/test_config_afir.py tests/test_scoring.py tests/test_path_relax.py tests/test_covalent_radii.py` (Phase 9 unit)
- [x] `pytest -m "not slow and not blender"` (full unit suite)
- [x] `pytest -m slow` (8 reactions UMA integration)
- [x] Manual: `reactx run examples/diels_alder_simple.rxn -o out/da/ --backend uma --render` and inspect Blender scene

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Note PR URL** for the user.

---

## Self-Review

After completing all stages above, verify:

1. **Spec coverage**:
   - §3.3 file structure → all files in this plan? ✓
   - §4.1 AFIRConstraint → Tasks 2.1-2.6 ✓
   - §4.2 path_relax 3-tuple → Task 6.1 ✓
   - §4.3 covalent_radii → Tasks 1.1-1.2 ✓
   - §4.4 config schema + ConfigError + α=0 reject + NaN reject → Tasks 1.3, 4.1 ✓
   - §4.5 reached_product / residual / count_initial_latched → Task 5.1 ✓
   - §4.6 cli orchestration + meta.json → Task 7.1 ✓
   - §5 example tuning table → Stage 8 ✓
   - §6 test strategy → Stage 9 ✓
   - §7 implementation order → Plan stages mirror this ✓

2. **No placeholders**: All code blocks contain runnable code; no "TBD" or "implement later".

3. **Type consistency**: `AFIRConstraint(formed, broken, *, alpha_formed, alpha_broken, formed_thresholds, broken_thresholds)` consistent across Tasks 2.1-2.6 and Task 7.1. `relax_with_restraints` returns 3-tuple consistent across Task 6.1 and Task 7.1. `TrialResult` has 12 fields consistent across Task 5.1 and Task 7.1.

## Open Questions (to resolve during implementation)

These are deferred from spec §10.3 (medium-priority codex items not patched into v3.1):

- **OQ-1**: `r_formed_threshold = 1.15 × Rsum` vs Blender `1.1 × Rsum` の 0.05 × Rsum 差 → 実装後 visual で edge case が出たら Blender 側を 1.15 に揃える検討。
- **OQ-2**: `r_broken_threshold` を `broken=[]` で明示時の warning 出力 → 本 plan では silent ignore、必要なら Phase 10 で warning 化。
- **OQ-3**: `alpha_*` の dataclass default を `0.0` にして TOML で省略可とするか → 現状は省略不可 (`_AFIR_REQUIRED` に含む)、UX 改善は Phase 10。
- **OQ-4**: `TrialResult` に 5 フィールド必須追加は breaking change → 影響範囲は本 plan で明記された tests/test_cli.py 等のみ、軽微。
- **OQ-5**: latch transition test の FIRE なし pure unit 化 → Task 2.2-2.4 で既に `adjust_forces` 直接呼び出しでテスト済み。
- **OQ-6**: 解析勾配 vs 有限差分 test の fresh constraint パターン → Task 2.5 で実装済み。

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-08-afir-force-replacement.md`. Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
