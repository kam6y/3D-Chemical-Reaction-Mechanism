# reactx Phase Re1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Phase 0 SN2 ハードコードを置き換え、SN2 + proton transfer の 2 反応を「**多角度試行 + 人工力 (Hookean restraint) による制約付き relaxation**」で end-to-end (.rxn → trajectory.xyz → Blender 視認) で扱う。NEB は default off で wall-clock 短縮を主目的とする。

**Architecture:** 5 つの新規モジュール (`bond_changes` / `trials` / `artificial_force` / `path_relax` / `scoring`) を追加し、`embed3d` は SN2 ハードコード判定を撤去して動的 substrate 同定 + rotation perturbation を受け取る形に改修。`cli.py` は N 角度サンプリング → 各 trial の relaxation → scoring → best 採用、optional NEB refinement の pipeline に張り替え。

**Tech Stack:** Python 3.11+ / RDKit 2025.3 / ASE 3.26 / fairchem-core 2.14 (UMA) / numpy / Blender 4.x

**Spec:** `docs/superpowers/specs/2026-04-27-reactx-phase-Re1-design.md`

---

## File Structure

### 新規作成

| Path | 責務 |
|---|---|
| `examples/proton_transfer.rxn` | HCl + NH₃ → Cl⁻ + NH₄⁺ の MDL Rxn V2000 ファイル (移動 H に explicit map) |
| `reactx/bond_changes.py` | `SimpleBondChanges` dataclass + `compute_simple_bond_changes` (形成 1 + 切断 1 限定) |
| `reactx/trials.py` | `sample_attack_rotations` (cone 内で n 個の 3×3 回転行列、index 0 は identity) |
| `reactx/artificial_force.py` | `DEFAULT_R_FORM` dict + カスタム `PullApart` constraint + `build_restraints` |
| `reactx/path_relax.py` | `relax_with_restraints` (FIRE で制約付き relaxation, frame stride で snapshot) |
| `reactx/scoring.py` | `TrialResult` dataclass + `reached_product` + `score_trials` |
| `tests/test_bond_changes.py` | SN2 / proton transfer / 不正 topology の 3 ケース |
| `tests/test_trials.py` | n=8/cone=30° の角度範囲、n=1 で identity、seed 再現性 |
| `tests/test_artificial_force.py` | Hookean 引力 / PullApart 斥力 / r_form dict lookup |
| `tests/test_path_relax.py` | toy LJ/EMT calc で 2 原子系の収束、stride、max_steps |
| `tests/test_scoring.py` | `reached_product` 判定 / 全成立中 best / 全滅 fallback |
| `tests/test_re1_sn2.py` | `@pytest.mark.slow` UMA で SN2 が 1 つ以上 reached_product、F-C-Cl 角度 ≥ 120° |
| `tests/test_re1_proton_transfer.py` | `@pytest.mark.slow` UMA で proton transfer の H 移動 |
| `tests/test_wallclock_sn2.py` | `@pytest.mark.slow` SN2 default の wall_clock_seconds 計測 (情報のみ) |
| `tests/test_neb_refine_sn2.py` | `@pytest.mark.slow` `--neb-refine` on で NEB 経路が走り meta.json に記録 |

### 変更

| Path | 変更内容 |
|---|---|
| `reactx/embed3d.py` | `_find_c_lg_bond` 削除、`embed_mol_to_atoms` に `bond_changes`/`rotation_perturbation` 引数追加、`_place_nucleophile_backside` を bond_changes ベース + 動的 substrate 同定に書き換え |
| `reactx/cli.py` | pipeline を多角度 + 制約 relaxation に張り替え、新 CLI flag (`--n-angles`, `--cone-half-deg`, `--seed`, `--r-form`, `--r-broken`, `--k-form`, `--k-broken`, `--max-relax-steps`, `--relax-fmax`, `--traj-stride`, `--neb-refine`, `--neb-images`) 追加 |
| `tests/test_embed3d.py` | 既存の multi-fragment テストで `bond_changes` を明示構築して渡す |
| `tests/test_blender_smoke.py` | SN2 + proton_transfer を parametrize |
| `README.md` | Phase Re1 の使い方、CLI flag 一覧、アニメーション目的の説明、Phase 0 比較 |

### 削除

| Path | 理由 |
|---|---|
| `tests/test_neb_sn2.py` | NEB が default 経路から外れたため `test_re1_sn2.py` + `test_neb_refine_sn2.py` に置換 |

---

## Task 1: examples/proton_transfer.rxn を作成

`HCl + NH₃ → Cl⁻ + NH₄⁺` の `.rxn` ファイル。proton 移動を atom mapping で追跡するため、移動する H に explicit な atom map number を付ける必要がある。

**Files:**
- Create: `examples/proton_transfer.rxn`

- [ ] **Step 1: examples/proton_transfer.rxn を作成**

```
$RXN

      RDKit

  2  2
$MOL

     RDKit          2D

  2  1  0  0  0  0  0  0  0  0999 V2000
   -0.7500    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  1  0  0
    0.7500    0.0000    0.0000 Cl  0  0  0  0  0  0  0  0  0  2  0  0
  1  2  1  0
M  END
$MOL

     RDKit          2D

  1  0  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  3  0  0
M  END
$MOL

     RDKit          2D

  1  0  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 Cl  0  0  0  0  0  0  0  0  0  2  0  0
M  CHG  1   1  -1
M  END
$MOL

     RDKit          2D

  2  1  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  3  0  0
    1.0000    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  1  0  0
  1  2  1  0
M  CHG  1   1   1
M  END
```

注: NH₃ の他の 3 つの H および NH₄⁺ の他の 3 つの H は **explicit に書かない**。RDKit `AddHs` に任せる。これにより heavy_to_hydrogen_groups が implicit H をペアリング (3 個 ↔ 3 個) でき、explicit な移動 H (map=1) のみ atom mapping で追跡される。

- [ ] **Step 2: RDKit で読み込めることを確認**

Run: `python -c "from rdkit.Chem import AllChem; r = AllChem.ReactionFromRxnFile('examples/proton_transfer.rxn'); print('reactants:', [Chem.MolToSmiles(m) for m in r.GetReactants()]); print('products:', [Chem.MolToSmiles(m) for m in r.GetProducts()])" 2>&1`
Expected: `reactants: ['Cl[H:1]', '[NH3:3]']` and `products: ['[Cl-:2]', '[NH4+:3]']` 程度の出力 (atom map が読める)

- [ ] **Step 3: parse_rxn が atom map を抽出できることを確認**

Run: `python -c "from reactx.rxn_parser import parse_rxn; r, p, m = parse_rxn('examples/proton_transfer.rxn'); print('mapping:', m); from rdkit import Chem; print('r:', Chem.MolToSmiles(r)); print('p:', Chem.MolToSmiles(p))"`
Expected: mapping dict が 3 エントリ (H, Cl, N の 3 ペア)、SMILES が `Cl[H:1].[NH3:3]` と `[Cl-:2].[NH4+:3]` 形式

- [ ] **Step 4: Commit**

```bash
git add examples/proton_transfer.rxn
git commit -m "feat(examples): add proton_transfer.rxn for Phase Re1 evaluation"
```

---

## Task 2: reactx/bond_changes.py — SimpleBondChanges + compute_simple_bond_changes

形成 1 + 切断 1 限定で、reactant/product mol_h と heavy_mapping から `SimpleBondChanges(formed, broken)` を返す。proton transfer のように explicit H が atom map を持つケースも扱う。

**Files:**
- Create: `reactx/bond_changes.py`
- Test: `tests/test_bond_changes.py`

- [ ] **Step 1: 失敗テストを書く (SN2 ケース)**

`tests/test_bond_changes.py`:
```python
"""Unit tests for reactx.bond_changes."""
from rdkit import Chem
import pytest

from reactx.bond_changes import SimpleBondChanges, compute_simple_bond_changes
from reactx.rxn_parser import parse_rxn


def _atom_index_by_symbol(mol_h: Chem.Mol, sym: str) -> int:
    """Find first atom of given element symbol (helper for assertions)."""
    for atom in mol_h.GetAtoms():
        if atom.GetSymbol() == sym:
            return atom.GetIdx()
    raise AssertionError(f"no {sym} atom in mol")


def test_sn2_bond_changes(tmp_path):
    r_mol, p_mol, mapping = parse_rxn("examples/sn2.rxn")
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)

    bc = compute_simple_bond_changes(r_h, p_h, mapping)
    assert isinstance(bc, SimpleBondChanges)

    c_idx = _atom_index_by_symbol(r_h, "C")
    cl_idx = _atom_index_by_symbol(r_h, "Cl")
    f_idx = _atom_index_by_symbol(r_h, "F")

    assert set(bc.formed) == {c_idx, f_idx}
    assert set(bc.broken) == {c_idx, cl_idx}
```

- [ ] **Step 2: テスト実行 → 失敗を確認**

Run: `pytest tests/test_bond_changes.py::test_sn2_bond_changes -v`
Expected: FAIL (`ModuleNotFoundError: reactx.bond_changes`)

- [ ] **Step 3: 最小実装**

`reactx/bond_changes.py`:
```python
"""Compute formed/broken bonds for a single elementary step (1 formed + 1 broken).

Phase Re1 supports only this minimal topology. Generic multi-bond reactions
(E2, dissociation, etc.) raise NotImplementedError and are deferred to Phase 2.
"""
from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem


@dataclass(frozen=True)
class SimpleBondChanges:
    """Single elementary step: exactly one bond formed and one bond broken.

    Atom indices are in the **reactant_mol_h** coordinate system
    (Chem.AddHs(reactant_mol).GetAtoms() ordering).
    """
    formed: tuple[int, int]
    broken: tuple[int, int]

    @property
    def shared_atom(self) -> int:
        """The atom common to both formed and broken bonds (= 'central anchor').

        For SN2 this is the substrate C; for proton transfer this is the H.
        """
        f = set(self.formed)
        b = set(self.broken)
        common = f & b
        if len(common) != 1:
            raise ValueError(
                f"formed {self.formed} and broken {self.broken} must share exactly "
                f"one atom; got {common}"
            )
        return next(iter(common))


def compute_simple_bond_changes(
    reactant_mol_h: Chem.Mol,
    product_mol_h: Chem.Mol,
    heavy_mapping: dict[int, int],
) -> SimpleBondChanges:
    """Diff bonds between reactant and product mol_h, return formed + broken.

    Raises NotImplementedError if not exactly 1 formed + 1 broken bond.
    Atom indices in the result use reactant_mol_h's ordering.
    """
    if reactant_mol_h.GetNumAtoms() != product_mol_h.GetNumAtoms():
        raise ValueError(
            f"reactant and product mol_h must have same atom count: "
            f"{reactant_mol_h.GetNumAtoms()} vs {product_mol_h.GetNumAtoms()}"
        )

    full_mapping = _build_full_atom_mapping(
        reactant_mol_h, product_mol_h, heavy_mapping
    )
    inv_mapping = {p: r for r, p in full_mapping.items()}

    r_bonds = _bond_set_in_self_idx(reactant_mol_h)
    p_bonds_in_r_space = {
        _ordered(inv_mapping[a], inv_mapping[b])
        for a, b in _bond_set_in_self_idx(product_mol_h)
    }

    formed = sorted(p_bonds_in_r_space - r_bonds)
    broken = sorted(r_bonds - p_bonds_in_r_space)

    if len(formed) != 1 or len(broken) != 1:
        raise NotImplementedError(
            f"Phase Re1 supports exactly 1 formed + 1 broken bond. "
            f"Got formed={formed} ({len(formed)}), broken={broken} ({len(broken)}). "
            f"Generic bond-change support is Phase 2."
        )
    return SimpleBondChanges(formed=formed[0], broken=broken[0])


def _bond_set_in_self_idx(mol: Chem.Mol) -> set[tuple[int, int]]:
    return {
        _ordered(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()
    }


def _ordered(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _build_full_atom_mapping(
    reactant_mol_h: Chem.Mol,
    product_mol_h: Chem.Mol,
    heavy_mapping: dict[int, int],
) -> dict[int, int]:
    """Extend heavy_mapping with implicit-H pairings.

    heavy_mapping covers all atoms with explicit atom map numbers (heavy + any
    explicit-mapped H). Remaining unmapped Hs (= implicit Hs added by AddHs)
    are paired by their bonded heavy atom group: reactant Hs of heavy_r ↔
    product Hs of heavy_p where heavy_r → heavy_p ∈ heavy_mapping.
    """
    full: dict[int, int] = dict(heavy_mapping)
    used_p: set[int] = set(full.values())

    for r_heavy, p_heavy in heavy_mapping.items():
        r_atom = reactant_mol_h.GetAtomWithIdx(r_heavy)
        p_atom = product_mol_h.GetAtomWithIdx(p_heavy)
        if r_atom.GetSymbol() == "H" or p_atom.GetSymbol() == "H":
            continue  # explicit-mapped H itself, not a heavy atom group

        r_implicit_hs = [
            n.GetIdx() for n in r_atom.GetNeighbors()
            if n.GetSymbol() == "H" and n.GetIdx() not in full
        ]
        p_implicit_hs = [
            n.GetIdx() for n in p_atom.GetNeighbors()
            if n.GetSymbol() == "H" and n.GetIdx() not in used_p
        ]
        if len(r_implicit_hs) != len(p_implicit_hs):
            raise ValueError(
                f"implicit H count mismatch for heavy atom {r_heavy}->{p_heavy}: "
                f"reactant has {len(r_implicit_hs)}, product has {len(p_implicit_hs)}"
            )
        for r_h, p_h in zip(sorted(r_implicit_hs), sorted(p_implicit_hs), strict=True):
            full[r_h] = p_h
            used_p.add(p_h)

    return full
```

- [ ] **Step 4: テスト実行 → SN2 が PASS**

Run: `pytest tests/test_bond_changes.py::test_sn2_bond_changes -v`
Expected: PASS

- [ ] **Step 5: proton transfer ケースのテストを追加**

`tests/test_bond_changes.py` に追加:
```python
def test_proton_transfer_bond_changes():
    r_mol, p_mol, mapping = parse_rxn("examples/proton_transfer.rxn")
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)

    bc = compute_simple_bond_changes(r_h, p_h, mapping)

    n_idx = _atom_index_by_symbol(r_h, "N")
    cl_idx = _atom_index_by_symbol(r_h, "Cl")
    # The migrating proton has atom map=1; find it via map number
    proton_idx = next(
        a.GetIdx() for a in r_h.GetAtoms() if a.GetAtomMapNum() == 1
    )

    assert set(bc.formed) == {n_idx, proton_idx}
    assert set(bc.broken) == {proton_idx, cl_idx}
    assert bc.shared_atom == proton_idx  # central atom is the moving H
```

- [ ] **Step 6: テスト実行 → PASS**

Run: `pytest tests/test_bond_changes.py::test_proton_transfer_bond_changes -v`
Expected: PASS

- [ ] **Step 7: 不正 topology テストを追加**

`tests/test_bond_changes.py` に追加:
```python
def test_e2_like_raises_not_implemented():
    """Two formed + two broken bonds (E2-like) should raise NotImplementedError."""
    # Use a fake mapping that creates 2 broken bonds: split CH3CH2Cl -> 2 fragments
    # Simpler: construct mol manually and assert the error.
    r = Chem.MolFromSmiles("[CH3:1][CH2:2][Cl:3]")
    p = Chem.MolFromSmiles("[CH2:1]=[CH2:2].[Cl:3]")
    r_h = Chem.AddHs(r)
    p_h = Chem.AddHs(p)
    # Build mapping by atom map number
    r_map = {a.GetAtomMapNum(): a.GetIdx() for a in r_h.GetAtoms() if a.GetAtomMapNum()}
    p_map = {a.GetAtomMapNum(): a.GetIdx() for a in p_h.GetAtoms() if a.GetAtomMapNum()}
    mapping = {r_map[m]: p_map[m] for m in r_map if m in p_map}

    # E2-style topology: 1 broken (C-Cl) + 1 formed (C=C extra bond) = NOT
    # supported in Phase Re1. Note: implicit H counts will also differ
    # (CH3CH2Cl has 5 Hs, CH2=CH2 has 4 Hs). Either condition triggers the error.
    with pytest.raises((NotImplementedError, ValueError)):
        compute_simple_bond_changes(r_h, p_h, mapping)
```

- [ ] **Step 8: テスト実行 → PASS**

Run: `pytest tests/test_bond_changes.py -v`
Expected: 3 PASS

- [ ] **Step 9: Commit**

```bash
git add reactx/bond_changes.py tests/test_bond_changes.py
git commit -m "feat(bond_changes): add SimpleBondChanges + compute_simple_bond_changes

Phase Re1 limited to 1 formed + 1 broken bond (SN2, proton transfer).
Handles explicit-mapped migrating Hs and implicit Hs paired by heavy-atom group."
```

---

## Task 3: reactx/trials.py — sample_attack_rotations

ideal direction を中心とする cone (半角 cone_half_deg) 内に N 個の 3×3 回転行列をサンプリング。index 0 は identity (perturbation なし)。

**Files:**
- Create: `reactx/trials.py`
- Test: `tests/test_trials.py`

- [ ] **Step 1: 失敗テストを書く**

`tests/test_trials.py`:
```python
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
```

- [ ] **Step 2: テスト実行 → 失敗を確認**

Run: `pytest tests/test_trials.py -v`
Expected: FAIL (`ModuleNotFoundError: reactx.trials`)

- [ ] **Step 3: 実装**

`reactx/trials.py`:
```python
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
    """Return n 3×3 rotation matrices.

    R[0] is identity. R[1..n-1] are sampled by Fibonacci spiral in the cap of
    a unit sphere of half-angle cone_half_deg around +z, then converted to
    rotation matrices that map +z to each sampled direction. Applied to a
    reference direction d, R @ d rotates d by an angle ≤ cone_half_deg.

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
        # Fibonacci spiral on the spherical cap above z = cos_cap
        # i in [0, n-2] -> z in [cos_cap, 1)
        # We avoid z=1 (= identity duplicate) by starting just below 1.
        t = (i + 0.5) / (n - 1)  # in (0, 1)
        z = 1.0 - t * (1.0 - cos_cap)  # in (cos_cap, 1)
        r_xy = float(np.sqrt(max(0.0, 1.0 - z * z)))
        theta = phase + i * golden_angle
        x = r_xy * float(np.cos(theta))
        y = r_xy * float(np.sin(theta))
        target = np.array([x, y, z])
        rotations.append(_rotation_from_z_to(target))
    return rotations


def _rotation_from_z_to(v: np.ndarray) -> np.ndarray:
    """Return a 3×3 rotation matrix R such that R @ [0,0,1] = v (unit vector).

    Uses Rodrigues' rotation formula. Handles parallel/antiparallel cases
    explicitly.
    """
    z = np.array([0.0, 0.0, 1.0])
    v = v / np.linalg.norm(v)
    cos_theta = float(np.dot(z, v))
    if cos_theta > 1.0 - 1e-12:
        return np.eye(3)
    if cos_theta < -1.0 + 1e-12:
        # 180° rotation around any axis perpendicular to z; pick x-axis
        return np.diag([1.0, -1.0, -1.0])
    axis = np.cross(z, v)
    axis = axis / np.linalg.norm(axis)
    sin_theta = float(np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta)))
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    return np.eye(3) + sin_theta * K + (1.0 - cos_theta) * (K @ K)
```

- [ ] **Step 4: テスト実行 → 5 ケース PASS**

Run: `pytest tests/test_trials.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add reactx/trials.py tests/test_trials.py
git commit -m "feat(trials): add sample_attack_rotations (Fibonacci spiral in cone)

Index 0 is identity for Phase 0 baseline reproducibility. Remaining n-1
rotations are deterministic given seed, and stay within cone_half_deg."
```

---

## Task 4: reactx/artificial_force.py — DEFAULT_R_FORM + PullApart + build_restraints

ASE 標準 `Hookean` で形成結合に引力、自前 `PullApart` で切断結合に斥力。元素ペア → r_form の dict を持つ。

**Files:**
- Create: `reactx/artificial_force.py`
- Test: `tests/test_artificial_force.py`

- [ ] **Step 1: r_form dict のテストを書く**

`tests/test_artificial_force.py`:
```python
"""Unit tests for reactx.artificial_force."""
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.lj import LennardJones

from reactx.artificial_force import (
    DEFAULT_R_FORM,
    PullApart,
    build_restraints,
    lookup_r_form,
)


def test_default_r_form_has_common_pairs():
    assert ("C", "F") in DEFAULT_R_FORM or ("F", "C") in DEFAULT_R_FORM
    assert ("N", "H") in DEFAULT_R_FORM or ("H", "N") in DEFAULT_R_FORM


def test_lookup_r_form_symmetric():
    assert lookup_r_form("C", "F") == lookup_r_form("F", "C")
    assert lookup_r_form("N", "H") == lookup_r_form("H", "N")


def test_lookup_r_form_known_values():
    assert abs(lookup_r_form("C", "F") - 1.39) < 1e-6
    assert abs(lookup_r_form("C", "Cl") - 1.78) < 1e-6
    assert abs(lookup_r_form("N", "H") - 1.01) < 1e-6


def test_lookup_r_form_unknown_pair_returns_default():
    assert lookup_r_form("Ge", "Te", default=1.6) == 1.6
```

- [ ] **Step 2: 実行 → 失敗を確認**

Run: `pytest tests/test_artificial_force.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: 実装 (lookup と DEFAULT のみ)**

`reactx/artificial_force.py`:
```python
"""Artificial force / restraints for path generation.

Phase Re1 uses these to drive a constrained relaxation that produces a
reaction trajectory:
- Hookean (ASE built-in): one-sided harmonic *attraction* pulling formed-bond
  atoms together when distance > rt.
- PullApart (this module): one-sided harmonic *repulsion* pushing broken-bond
  atoms apart when distance < rt.

Equilibrium bond lengths for common element pairs are tabulated in
DEFAULT_R_FORM (units: Å); use lookup_r_form to query them symmetrically.
"""
from __future__ import annotations

import numpy as np
from ase.atoms import Atoms
from ase.constraints import FixConstraint, Hookean

DEFAULT_R_FORM: dict[tuple[str, str], float] = {
    ("C", "F"): 1.39,
    ("C", "Cl"): 1.78,
    ("C", "N"): 1.47,
    ("C", "O"): 1.43,
    ("C", "C"): 1.54,
    ("C", "H"): 1.09,
    ("N", "H"): 1.01,
    ("O", "H"): 0.97,
}


def lookup_r_form(sym_a: str, sym_b: str, *, default: float = 1.6) -> float:
    """Symmetric lookup in DEFAULT_R_FORM. Returns `default` if pair unknown."""
    if (sym_a, sym_b) in DEFAULT_R_FORM:
        return DEFAULT_R_FORM[(sym_a, sym_b)]
    if (sym_b, sym_a) in DEFAULT_R_FORM:
        return DEFAULT_R_FORM[(sym_b, sym_a)]
    return default


class PullApart(FixConstraint):
    """One-sided harmonic *repulsion* pushing two atoms apart.

    Force is zero when distance >= rt; below rt, magnitude = k * (rt - r),
    pointing from atom a1 to atom a2 (and the equal-and-opposite on a1).
    Mirror of ase.constraints.Hookean which only attracts when r > rt.
    """

    def __init__(self, a1: int, a2: int, k: float, rt: float):
        self.a1 = int(a1)
        self.a2 = int(a2)
        self.k = float(k)
        self.rt = float(rt)

    def adjust_positions(self, atoms, newpositions):
        # Constraint is force-only; positions are integrated by the optimizer.
        return

    def adjust_forces(self, atoms, forces):
        p = atoms.positions
        d = p[self.a2] - p[self.a1]
        r = float(np.linalg.norm(d))
        if r >= self.rt or r < 1e-10:
            return
        # Repulsive: push a2 away from a1
        u = d / r
        f = self.k * (self.rt - r) * u
        forces[self.a2] += f
        forces[self.a1] -= f

    def get_indices(self):
        return [self.a1, self.a2]

    def todict(self):
        return {
            "name": "PullApart",
            "kwargs": {"a1": self.a1, "a2": self.a2, "k": self.k, "rt": self.rt},
        }


def build_restraints(
    atoms: Atoms,
    formed: list[tuple[int, int]],
    broken: list[tuple[int, int]],
    *,
    r_form: float | None = None,
    r_broken: float = 4.0,
    k_form: float = 5.0,
    k_broken: float = 3.0,
) -> list:
    """Return a list of ASE constraints driving formed bonds together and
    broken bonds apart.

    `r_form=None` (default) looks each pair up in DEFAULT_R_FORM by element
    symbols. A scalar overrides the table for all formed bonds (debug).
    """
    syms = atoms.get_chemical_symbols()
    constraints: list = []
    for a, b in formed:
        if r_form is None:
            rt = lookup_r_form(syms[a], syms[b])
        else:
            rt = float(r_form)
        constraints.append(Hookean(a1=a, a2=b, rt=rt, k=k_form))
    for a, b in broken:
        constraints.append(PullApart(a1=a, a2=b, k=k_broken, rt=r_broken))
    return constraints
```

- [ ] **Step 4: 実行 → r_form テスト 4 件 PASS**

Run: `pytest tests/test_artificial_force.py -v`
Expected: 4 PASS

- [ ] **Step 5: Hookean / PullApart 力テストを追加**

`tests/test_artificial_force.py` に追加:
```python
def test_hookean_pulls_when_distance_above_rt():
    """Hookean (ASE built-in) pulls atoms together when r > rt."""
    atoms = Atoms("CF", positions=[[0, 0, 0], [3.0, 0, 0]])
    cs = build_restraints(atoms, formed=[(0, 1)], broken=[],
                          r_form=1.5, k_form=5.0)
    atoms.set_constraint(cs)
    atoms.calc = LennardJones()  # weak background, doesn't dominate
    f = atoms.get_forces()
    # Atom 1 should be pulled toward atom 0 → negative x force
    assert f[1, 0] < -1.0, f"expected attractive force, got fx={f[1, 0]}"
    # Atom 0 should be pulled toward atom 1 → positive x force
    assert f[0, 0] > 1.0, f"expected attractive force on atom 0, got fx={f[0, 0]}"


def test_pullapart_pushes_when_distance_below_rt():
    """PullApart pushes atoms apart when r < rt."""
    atoms = Atoms("CCl", positions=[[0, 0, 0], [2.0, 0, 0]])
    cs = build_restraints(atoms, formed=[], broken=[(0, 1)],
                          r_broken=4.0, k_broken=3.0)
    atoms.set_constraint(cs)
    atoms.calc = LennardJones()
    f = atoms.get_forces()
    # Atom 1 should be pushed away from atom 0 → positive x force
    assert f[1, 0] > 1.0, f"expected repulsive force, got fx={f[1, 0]}"
    assert f[0, 0] < -1.0, f"expected repulsive force on atom 0, got fx={f[0, 0]}"


def test_pullapart_silent_when_distance_above_rt():
    """PullApart contributes zero force when r >= rt."""
    atoms = Atoms("CCl", positions=[[0, 0, 0], [5.0, 0, 0]])
    cs = build_restraints(atoms, formed=[], broken=[(0, 1)],
                          r_broken=4.0, k_broken=3.0)
    atoms.set_constraint(cs)
    # Use a calculator that returns zero forces so we isolate the constraint
    class ZeroCalc(Calculator):
        implemented_properties = ["energy", "forces"]
        def calculate(self, atoms=None, properties=("energy",), system_changes=()):
            super().calculate(atoms, properties, system_changes)
            self.results = {
                "energy": 0.0,
                "forces": np.zeros((len(atoms), 3)),
            }
    atoms.calc = ZeroCalc()
    f = atoms.get_forces()
    np.testing.assert_allclose(f, 0.0, atol=1e-10)


def test_build_restraints_empty_lists():
    atoms = Atoms("HH", positions=[[0, 0, 0], [1, 0, 0]])
    cs = build_restraints(atoms, formed=[], broken=[])
    assert cs == []
```

- [ ] **Step 6: 実行 → 8 ケース PASS**

Run: `pytest tests/test_artificial_force.py -v`
Expected: 8 PASS

- [ ] **Step 7: Commit**

```bash
git add reactx/artificial_force.py tests/test_artificial_force.py
git commit -m "feat(artificial_force): add Hookean+PullApart restraints, DEFAULT_R_FORM

Phase Re1 path-generation primitive. Hookean (ASE) attracts formed bonds,
custom PullApart repels broken bonds. r_form auto-derived from element pair."
```

---

## Task 5: reactx/scoring.py — TrialResult + reached_product + score_trials

trial 結果の dataclass と best 選択ロジック。UMA を必要としない pure logic。

**Files:**
- Create: `reactx/scoring.py`
- Test: `tests/test_scoring.py`

- [ ] **Step 1: 失敗テストを書く**

`tests/test_scoring.py`:
```python
"""Unit tests for reactx.scoring."""
import numpy as np
import pytest
from ase import Atoms

from reactx.scoring import TrialResult, reached_product, score_trials


def _atoms_with_distances(d_form: float, d_broken: float) -> Atoms:
    """3 atoms: 0=anchor, 1=incoming (distance d_form), 2=leaving (distance d_broken)."""
    return Atoms(
        "CFC",
        positions=[[0, 0, 0], [d_form, 0, 0], [-d_broken, 0, 0]],
    )


def test_reached_product_true_when_formed_close_and_broken_far():
    a = _atoms_with_distances(d_form=1.5, d_broken=4.5)
    assert reached_product(
        a, formed=[(0, 1)], broken=[(0, 2)],
        r_form_targets=[1.5], r_broken_target=4.0,
    ) is True


def test_reached_product_false_when_formed_too_far():
    a = _atoms_with_distances(d_form=2.5, d_broken=4.5)  # formed exceeded tol
    assert reached_product(
        a, formed=[(0, 1)], broken=[(0, 2)],
        r_form_targets=[1.5], r_broken_target=4.0,
        form_tol=0.3,
    ) is False


def test_reached_product_false_when_broken_too_close():
    a = _atoms_with_distances(d_form=1.5, d_broken=3.0)  # broken still close
    assert reached_product(
        a, formed=[(0, 1)], broken=[(0, 2)],
        r_form_targets=[1.5], r_broken_target=4.0,
        broken_tol=0.5,
    ) is False


def test_score_trials_picks_lowest_peak_among_reached():
    t1 = TrialResult(trial_idx=0, rotation_deg=0.0, frames=[], energies=[],
                     reached_product=True, peak_energy=-100.0, n_steps=50)
    t2 = TrialResult(trial_idx=1, rotation_deg=15.0, frames=[], energies=[],
                     reached_product=True, peak_energy=-105.0, n_steps=60)
    t3 = TrialResult(trial_idx=2, rotation_deg=20.0, frames=[], energies=[],
                     reached_product=False, peak_energy=-200.0, n_steps=30)
    best = score_trials([t1, t2, t3])
    assert best.trial_idx == 1  # lowest peak among reached


def test_score_trials_falls_back_to_least_bad_when_all_failed():
    t1 = TrialResult(trial_idx=0, rotation_deg=0.0, frames=[], energies=[10.0, 12.0],
                     reached_product=False, peak_energy=12.0, n_steps=50)
    t2 = TrialResult(trial_idx=1, rotation_deg=15.0, frames=[], energies=[10.0, 11.0],
                     reached_product=False, peak_energy=11.0, n_steps=60)
    best = score_trials([t1, t2])
    # When all failed, prefer lowest peak
    assert best.trial_idx == 1


def test_score_trials_empty_raises():
    with pytest.raises(ValueError):
        score_trials([])
```

- [ ] **Step 2: 実行 → 失敗確認**

Run: `pytest tests/test_scoring.py -v`
Expected: FAIL

- [ ] **Step 3: 実装**

`reactx/scoring.py`:
```python
"""Trial scoring for multi-angle path generation.

After each angle trial produces a relaxation trajectory, score_trials picks
the best one: prefer trials that actually reached the product topology
(formed bond close, broken bond far), then minimum peak energy among those.
If no trial reached the product, fall back to the trial with lowest peak
(= 'least bad' partial path).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms


@dataclass
class TrialResult:
    """Outcome of a single multi-angle relaxation trial."""
    trial_idx: int
    rotation_deg: float       # angular deviation from ideal direction
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
    """Return the best trial. Preference order:
    1. reached_product=True, lowest peak_energy
    2. all failed: lowest peak_energy (= 'least bad' fallback)
    """
    if not results:
        raise ValueError("score_trials called with empty list")
    reached = [r for r in results if r.reached_product]
    pool = reached if reached else results
    return min(pool, key=lambda r: r.peak_energy)
```

- [ ] **Step 4: 実行 → 6 ケース PASS**

Run: `pytest tests/test_scoring.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add reactx/scoring.py tests/test_scoring.py
git commit -m "feat(scoring): add TrialResult + reached_product + score_trials

Pure logic (no UMA dep). reached_product checks formed/broken distances;
score_trials prefers reached trials, ties broken by lowest peak energy."
```

---

## Task 6: reactx/path_relax.py — relax_with_restraints

ASE FIRE で制約付き relaxation、frame stride で snapshot を保存。ユニットテストは toy LJ calculator で行う。

**Files:**
- Create: `reactx/path_relax.py`
- Test: `tests/test_path_relax.py`

- [ ] **Step 1: 失敗テストを書く**

`tests/test_path_relax.py`:
```python
"""Unit tests for reactx.path_relax (no UMA dependency)."""
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.constraints import Hookean

from reactx.path_relax import relax_with_restraints


def _two_atom_system(initial_distance: float) -> Atoms:
    return Atoms("ArAr", positions=[[0, 0, 0], [initial_distance, 0, 0]])


def test_relax_pulls_atoms_toward_target_distance():
    atoms = _two_atom_system(initial_distance=5.0)
    atoms.calc = LennardJones()  # weak attraction at long range
    cs = [Hookean(a1=0, a2=1, rt=2.0, k=10.0)]  # pull to ~2 Å

    frames, energies = relax_with_restraints(
        atoms, cs, atoms.calc, max_steps=200, fmax=0.05, traj_stride=10,
    )

    final_d = float(np.linalg.norm(frames[-1].positions[1] - frames[-1].positions[0]))
    # Hookean pulls down to rt=2; LJ provides additional attraction near 1.13 (sigma)
    assert final_d < 3.0, f"final distance {final_d:.3f} not pulled toward target"


def test_traj_stride_controls_frame_count():
    atoms = _two_atom_system(initial_distance=5.0)
    atoms.calc = LennardJones()
    cs = [Hookean(a1=0, a2=1, rt=2.0, k=10.0)]

    frames, energies = relax_with_restraints(
        atoms, cs, atoms.calc, max_steps=50, fmax=1e-9, traj_stride=10,
    )
    # 50 steps / stride 10 -> ~6 frames including initial + final
    assert 4 <= len(frames) <= 8
    assert len(energies) == len(frames)


def test_max_steps_terminates_loop():
    atoms = _two_atom_system(initial_distance=5.0)
    atoms.calc = LennardJones()
    cs = [Hookean(a1=0, a2=1, rt=2.0, k=10.0)]

    frames, _ = relax_with_restraints(
        atoms, cs, atoms.calc, max_steps=5, fmax=1e-12, traj_stride=1,
    )
    # 5 steps + initial frame = at most 6
    assert len(frames) <= 7


def test_first_frame_is_initial():
    atoms = _two_atom_system(initial_distance=5.0)
    init_pos = atoms.positions.copy()
    atoms.calc = LennardJones()
    cs = [Hookean(a1=0, a2=1, rt=2.0, k=10.0)]

    frames, _ = relax_with_restraints(
        atoms, cs, atoms.calc, max_steps=10, fmax=0.1, traj_stride=5,
    )
    np.testing.assert_allclose(frames[0].positions, init_pos)


def test_energies_are_finite_floats():
    atoms = _two_atom_system(initial_distance=5.0)
    atoms.calc = LennardJones()
    cs = [Hookean(a1=0, a2=1, rt=2.0, k=10.0)]

    _, energies = relax_with_restraints(
        atoms, cs, atoms.calc, max_steps=10, fmax=0.1, traj_stride=5,
    )
    for e in energies:
        assert isinstance(e, float)
        assert np.isfinite(e)
```

- [ ] **Step 2: 実行 → 失敗確認**

Run: `pytest tests/test_path_relax.py -v`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: 実装**

`reactx/path_relax.py`:
```python
"""Constrained relaxation that yields a trajectory of frames.

Used by Phase Re1 to convert (initial geometry + bond-change restraints) into
a reaction path: FIRE relaxation under Hookean (attractive) + PullApart
(repulsive) constraints drives the system from reactant toward product.
Frames are snapshotted every `traj_stride` optimizer steps; the final frame
is always included.

Use this with a shared calculator (e.g. UMA) to avoid model reloads between
trials.
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
) -> tuple[list[Atoms], list[float]]:
    """Run FIRE under the given restraints; return (frames, energies).

    The first frame is the initial state; thereafter snapshots are taken every
    `traj_stride` steps, plus the final state. `frames[i].calc` is None to
    avoid keeping references to the shared calculator in trajectory output.
    """
    atoms = atoms.copy()
    atoms.calc = calc
    if restraints:
        atoms.set_constraint(restraints)

    frames: list[Atoms] = [_snapshot(atoms)]
    energies: list[float] = [float(atoms.get_potential_energy())]

    opt = FIRE(atoms, logfile=None, dt=0.05, a=0.1)
    step_count = 0

    def _record():
        frames.append(_snapshot(atoms))
        energies.append(float(atoms.get_potential_energy()))

    opt.attach(_record, interval=traj_stride)
    opt.run(fmax=fmax, steps=max_steps)
    step_count = opt.nsteps

    # Always record final frame if not already captured this step
    if step_count % traj_stride != 0 or step_count == 0:
        _record()

    return frames, energies


def _snapshot(atoms: Atoms) -> Atoms:
    """Detach calculator so frame is independent and serializable."""
    a = atoms.copy()
    a.calc = None
    return a
```

- [ ] **Step 4: 実行 → 5 ケース PASS**

Run: `pytest tests/test_path_relax.py -v`
Expected: 5 PASS (Lennard-Jones パラメータ次第で `final_d < 3.0` が境界に近いので必要なら fmax/k を微調整)

- [ ] **Step 5: Commit**

```bash
git add reactx/path_relax.py tests/test_path_relax.py
git commit -m "feat(path_relax): add relax_with_restraints with FIRE + frame stride

Produces (frames, energies) trajectory. Tested with toy LJ; UMA integration
arrives via cli.py rewrite in a later task."
```

---

## Task 7: reactx/embed3d.py — bond_changes 引数追加 + 動的 substrate 同定 + rotation_perturbation

`_find_c_lg_bond` を削除し、`bond_changes` を受け取って動的に substrate fragment を同定。`rotation_perturbation` で attack 方向に擾乱を加える。

**Files:**
- Modify: `reactx/embed3d.py`
- Modify: `tests/test_embed3d.py`

- [ ] **Step 1: 既存テストの multi-fragment ケースを bond_changes 渡しに更新**

`tests/test_embed3d.py` の `test_embed_multifragment_places_fragments_apart` を以下に置換:
```python
def test_embed_multifragment_places_fragments_apart():
    from reactx.bond_changes import SimpleBondChanges
    mol = Chem.MolFromSmiles("CCl.[F-]")
    mol_h = Chem.AddHs(mol)
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    c_idx = syms.index("C")
    cl_idx = syms.index("Cl")
    f_idx = syms.index("F")
    bc = SimpleBondChanges(formed=(c_idx, f_idx), broken=(c_idx, cl_idx))

    atoms = embed_mol_to_atoms(mol, calculator=None, seed=42, bond_changes=bc)
    syms_out = atoms.get_chemical_symbols()
    f_out = syms_out.index("F")
    c_out = syms_out.index("C")
    d = atoms.get_distance(c_out, f_out)
    assert d > 2.5, f"Fragments too close: C-F distance {d:.3f} Å"
```

`tests/test_embed3d.py` の `test_embed_sets_total_charge_and_spin_in_info` と `test_embed_multifragment_preserves_addhs_ordering` も同様に bond_changes 引数を渡すよう修正:
```python
def test_embed_multifragment_preserves_addhs_ordering():
    from reactx.bond_changes import SimpleBondChanges
    mol = Chem.MolFromSmiles("CCl.[F-]")
    mol_h = Chem.AddHs(mol)
    expected_symbols = [a.GetSymbol() for a in mol_h.GetAtoms()]
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    bc = SimpleBondChanges(
        formed=(syms.index("C"), syms.index("F")),
        broken=(syms.index("C"), syms.index("Cl")),
    )
    atoms = embed_mol_to_atoms(mol, calculator=None, seed=42, bond_changes=bc)
    assert atoms.get_chemical_symbols() == expected_symbols


def test_embed_sets_total_charge_and_spin_in_info():
    from reactx.bond_changes import SimpleBondChanges
    mol = Chem.MolFromSmiles("CCl.[F-]")
    mol_h = Chem.AddHs(mol)
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    bc = SimpleBondChanges(
        formed=(syms.index("C"), syms.index("F")),
        broken=(syms.index("C"), syms.index("Cl")),
    )
    atoms = embed_mol_to_atoms(mol, calculator=None, seed=42, bond_changes=bc)
    assert atoms.info["charge"] == -1
    assert atoms.info["spin"] == 1
```

(他のテストは single-fragment なので無変更で OK)

- [ ] **Step 2: rotation_perturbation 効果のテストを追加**

`tests/test_embed3d.py` 末尾に追加:
```python
def test_embed_rotation_perturbation_changes_nucleophile_position():
    """A non-identity rotation_perturbation should move the nucleophile fragment
    relative to the identity case."""
    import numpy as np
    from reactx.bond_changes import SimpleBondChanges
    mol = Chem.MolFromSmiles("CCl.[F-]")
    mol_h = Chem.AddHs(mol)
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    bc = SimpleBondChanges(
        formed=(syms.index("C"), syms.index("F")),
        broken=(syms.index("C"), syms.index("Cl")),
    )
    a_id = embed_mol_to_atoms(mol, calculator=None, seed=42, bond_changes=bc)
    # 30° rotation around y-axis
    theta = np.deg2rad(30.0)
    R = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)],
    ])
    a_pert = embed_mol_to_atoms(
        mol, calculator=None, seed=42, bond_changes=bc, rotation_perturbation=R,
    )
    # F position should differ between identity and perturbed
    f_idx = a_id.get_chemical_symbols().index("F")
    diff = np.linalg.norm(a_id.positions[f_idx] - a_pert.positions[f_idx])
    assert diff > 0.5, f"F position barely moved ({diff:.3f} Å) under 30° perturbation"


def test_embed_proton_transfer_substrate_is_hcl_fragment():
    """For HCl + NH3 → Cl- + NH4+, the broken bond is H-Cl. The substrate
    fragment must be HCl (containing both H and Cl), even though NH3 has
    more atoms after AddHs."""
    from reactx.bond_changes import SimpleBondChanges
    from reactx.rxn_parser import parse_rxn
    r_mol, _, _ = parse_rxn("examples/proton_transfer.rxn")
    r_h = Chem.AddHs(r_mol)
    syms = [a.GetSymbol() for a in r_h.GetAtoms()]
    n_idx = syms.index("N")
    cl_idx = syms.index("Cl")
    proton_idx = next(
        a.GetIdx() for a in r_h.GetAtoms() if a.GetAtomMapNum() == 1
    )
    bc = SimpleBondChanges(formed=(n_idx, proton_idx), broken=(proton_idx, cl_idx))
    atoms = embed_mol_to_atoms(r_mol, calculator=None, seed=42, bond_changes=bc)

    # After placement, N should be on the opposite side of Cl from the proton
    # (= backside of H from Cl). Roughly: vec(H->N) ≈ -vec(H->Cl)
    p_h = atoms.positions[proton_idx]
    p_cl = atoms.positions[cl_idx]
    p_n = atoms.positions[n_idx]
    h_to_cl = p_cl - p_h
    h_to_n = p_n - p_h
    cos_theta = float(np.dot(h_to_cl, h_to_n) / (
        np.linalg.norm(h_to_cl) * np.linalg.norm(h_to_n)
    ))
    assert cos_theta < -0.7, (
        f"N is not on backside of H from Cl: cos(angle Cl-H-N)={cos_theta:.3f}"
    )
```

- [ ] **Step 3: 実行 → multi-fragment テストが失敗を確認**

Run: `pytest tests/test_embed3d.py -v`
Expected: 6 失敗 (multi-fragment + proton transfer + rotation_perturbation のテスト) + 3 PASS (single-fragment は無変更で動く)

- [ ] **Step 4: embed3d.py を書き換え**

`reactx/embed3d.py` を以下に置換:
```python
"""Convert 2D RDKit Mol to 3D ase.Atoms via RDKit ETKDG + MMFF (+ optional UMA).

Preserves the atom ordering of Chem.AddHs(mol) so that downstream consumers
(align, NEB, restraints) can correlate atom indices with the same AddHs(mol)
result.

Multi-fragment placement (e.g. SN2 substrate + nucleophile) is driven by the
caller-supplied SimpleBondChanges:
- The substrate fragment is identified as the one containing both atoms of
  the broken bond.
- The nucleophile fragment(s) are placed along the backside direction
  (-unit(anchor->leaving)) at FRAGMENT_SEPARATION distance.
- An optional rotation_perturbation rotates the backside direction within
  the cone of multi-angle trials.
"""
from __future__ import annotations

import logging

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize import BFGS
from rdkit import Chem
from rdkit.Chem import AllChem

from reactx.bond_changes import SimpleBondChanges

log = logging.getLogger(__name__)

MAX_EMBED_RETRIES = 5
FRAGMENT_SEPARATION = 3.5  # Å — attack distance for multi-fragment placement


def embed_mol_to_atoms(
    mol: Chem.Mol,
    *,
    calculator: Calculator | None = None,
    seed: int = 0xC0FFEE,
    fmax: float = 0.01,
    max_opt_steps: int = 300,
    bond_changes: SimpleBondChanges | None = None,
    rotation_perturbation: np.ndarray | None = None,
) -> Atoms:
    """Embed a 2D Mol into 3D and return an ase.Atoms with implicit Hs added.

    For multi-fragment Mols, `bond_changes` MUST be provided; the substrate
    fragment is identified as the one containing both broken-bond atoms, and
    the remaining fragment(s) are placed along the backside direction.

    `rotation_perturbation` (optional 3×3 numpy array) is left-multiplied
    onto the computed backside direction before placement; identity = no
    perturbation = Phase 0 baseline.
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

    if len(frag_indices) > 1:
        if bond_changes is None:
            raise ValueError(
                "Multi-fragment Mol requires bond_changes to determine placement; "
                "got None. Compute via reactx.bond_changes.compute_simple_bond_changes."
            )
        positions = _place_nucleophile_backside(
            mol_h, frag_indices, positions, bond_changes,
            rotation_perturbation=rotation_perturbation,
        )

    symbols = [a.GetSymbol() for a in mol_h.GetAtoms()]
    charges = [a.GetFormalCharge() for a in mol_h.GetAtoms()]
    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.set_initial_charges(charges)

    atoms.info["charge"] = int(sum(charges))
    atoms.info["spin"] = 1  # Phase Re1: assume closed-shell singlet

    if calculator is not None:
        atoms.calc = calculator
        BFGS(atoms, logfile=None).run(fmax=fmax, steps=max_opt_steps)

    return atoms


def _place_nucleophile_backside(
    mol_h: Chem.Mol,
    frag_indices: tuple[tuple[int, ...], ...],
    positions: np.ndarray,
    bond_changes: SimpleBondChanges,
    *,
    rotation_perturbation: np.ndarray | None = None,
) -> np.ndarray:
    """Place non-substrate fragments along the rotated backside direction.

    Substrate fragment = the one containing both atoms of the broken bond.
    Anchor (= shared atom of formed and broken) and leaving (= other end of
    broken) live in the substrate. Incoming (= other end of formed) lives in
    the nucleophile fragment, which is shifted so its centroid aligns with
    `anchor + R @ (-unit(anchor->leaving)) * FRAGMENT_SEPARATION`.
    """
    a_form, b_form = bond_changes.formed
    a_brk, b_brk = bond_changes.broken
    shared = bond_changes.shared_atom
    anchor = shared
    leaving = b_brk if a_brk == shared else a_brk
    incoming = b_form if a_form == shared else a_form

    substrate_frag = next(
        (fi for fi in frag_indices if anchor in fi and leaving in fi), None
    )
    if substrate_frag is None:
        raise ValueError(
            f"Anchor {anchor} and leaving {leaving} are not in the same fragment; "
            f"Phase Re1 expects the broken bond's two atoms to be co-fragmented."
        )

    nuc_fragments = [fi for fi in frag_indices if fi is not substrate_frag]
    if not nuc_fragments:
        raise ValueError(
            "Only one fragment found, but multi-fragment placement was invoked. "
            "(Internal inconsistency: did embed_mol_to_atoms call this for n_frags=1?)"
        )

    a_pos = positions[anchor]
    c_pos = positions[leaving]
    a_c = c_pos - a_pos
    a_c_norm = float(np.linalg.norm(a_c))
    if a_c_norm < 1e-6:
        raise RuntimeError(
            "Anchor and leaving atoms coincide after MMFF — embedding is broken."
        )
    backside = -a_c / a_c_norm
    if rotation_perturbation is not None:
        backside = rotation_perturbation @ backside

    target = a_pos + backside * FRAGMENT_SEPARATION
    for nuc in nuc_fragments:
        if incoming in nuc:
            nuc_centroid_anchor = positions[incoming]
        else:
            nuc_centroid_anchor = positions[list(nuc)].mean(axis=0)
        positions[list(nuc)] += target - nuc_centroid_anchor
    return positions


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

- [ ] **Step 5: 全テスト実行 → embed3d 関連が全て PASS**

Run: `pytest tests/test_embed3d.py tests/test_bond_changes.py -v`
Expected: 全 PASS

- [ ] **Step 6: Commit**

```bash
git add reactx/embed3d.py tests/test_embed3d.py
git commit -m "refactor(embed3d): drop _find_c_lg_bond, use SimpleBondChanges + rotation

Substrate fragment is now identified dynamically as the one containing both
broken-bond atoms (works for SN2 and proton transfer). rotation_perturbation
parameterises multi-angle trials. Phase 0 single-trial is rotation=identity."
```

---

## Task 8: reactx/cli.py — pipeline 全面書き換え

新しい CLI flag を追加し、`compute_simple_bond_changes` → 多角度サンプリング → 各 trial で embed + relax → scoring → best 採用 → optional NEB refine → trajectory.xyz の流れに張り替え。

**Files:**
- Modify: `reactx/cli.py`
- Test: 既存の `tests/test_cli.py` (もし無ければ新規。最低限: `--n-angles 1` で実行が走るスモークテスト)

- [ ] **Step 1: 現在の test_cli.py を確認 / 必要なら最低限の引数解析テストを追加**

Run: `ls tests/test_cli.py 2>/dev/null && cat tests/test_cli.py`
- 存在する場合: 既存テストが新しい flag に対応するよう更新
- 存在しない場合: Step 2 で新規作成

- [ ] **Step 2: 引数 parser のスモークテストを書く**

`tests/test_cli.py` (新規 or 既存に追加):
```python
"""CLI argument parsing smoke tests (no UMA invocation)."""
from reactx.cli import build_parser


def test_default_flags_parse():
    p = build_parser()
    a = p.parse_args(["run", "examples/sn2.rxn", "-o", "out/"])
    assert a.cmd == "run"
    assert a.n_angles == 8
    assert a.cone_half_deg == 30.0
    assert a.seed == 0
    assert a.r_form is None  # auto-derive from element pair
    assert a.r_broken == 4.0
    assert a.k_form == 5.0
    assert a.k_broken == 3.0
    assert a.max_relax_steps == 100
    assert a.relax_fmax == 0.1
    assert a.traj_stride == 5
    assert a.neb_refine is False
    assert a.neb_images == 7


def test_neb_refine_flag():
    p = build_parser()
    a = p.parse_args(["run", "examples/sn2.rxn", "-o", "out/", "--neb-refine"])
    assert a.neb_refine is True


def test_n_angles_override():
    p = build_parser()
    a = p.parse_args(["run", "examples/sn2.rxn", "-o", "out/", "--n-angles", "1"])
    assert a.n_angles == 1


def test_r_form_override():
    p = build_parser()
    a = p.parse_args(["run", "examples/sn2.rxn", "-o", "out/", "--r-form", "1.05"])
    assert a.r_form == 1.05
```

- [ ] **Step 3: 実行 → 失敗 (新 flag 未実装)**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL (`AttributeError: ... has no attribute 'n_angles'`)

- [ ] **Step 4: cli.py を書き換え**

`reactx/cli.py` を以下に置換:
```python
"""CLI entry point: reactx run <rxn> -o <outdir> [options]."""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
from ase.io import read, write
from rdkit import Chem

from reactx.align import align_product_to_reactant
from reactx.artificial_force import build_restraints, lookup_r_form
from reactx.bond_changes import compute_simple_bond_changes
from reactx.calculators import make_calculator
from reactx.embed3d import embed_mol_to_atoms
from reactx.neb import run_neb
from reactx.path_relax import relax_with_restraints
from reactx.rxn_parser import heavy_to_hydrogen_groups, parse_rxn
from reactx.scoring import TrialResult, reached_product, score_trials
from reactx.trials import sample_attack_rotations

log = logging.getLogger("reactx")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="reactx")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run full pipeline on a .rxn file")
    run.add_argument("rxn_path", type=Path)
    run.add_argument("-o", "--output", type=Path, required=True)

    run.add_argument("--backend", choices=["uma", "lj"], default="uma")
    run.add_argument("--model", type=str, default="uma-m-1p1",
                     help="UMA model name (uma-m-1p1, uma-s-1p2, ...)")

    run.add_argument("--n-angles", type=int, default=8,
                     help="Number of attack-angle trials (>=1; index 0 is identity)")
    run.add_argument("--cone-half-deg", type=float, default=30.0,
                     help="Half-angle of cone within which trials are sampled")
    run.add_argument("--seed", type=int, default=0)

    run.add_argument("--r-form", type=float, default=None,
                     help="Override formed-bond target distance (Å). "
                          "Default: auto from element pair table.")
    run.add_argument("--r-broken", type=float, default=4.0)
    run.add_argument("--k-form", type=float, default=5.0)
    run.add_argument("--k-broken", type=float, default=3.0)

    run.add_argument("--max-relax-steps", type=int, default=100)
    run.add_argument("--relax-fmax", type=float, default=0.1)
    run.add_argument("--traj-stride", type=int, default=5)

    run.add_argument("--neb-refine", action="store_true",
                     help="Refine the best trial trajectory with a short NEB")
    run.add_argument("--neb-images", type=int, default=7)

    run.add_argument("--render", action="store_true",
                     help="Also invoke blender/render.py after pipeline")
    run.add_argument("--blender-exe", type=str, default="blender")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return _cmd_run(args)


def _configure_reactx_logging() -> None:
    if log.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[reactx] %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False


def _fmt_fmax(v: float) -> str:
    return "nan" if math.isnan(v) else f"{v:.4f}"


def _sanitize_for_json(obj):
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _check_hf_auth() -> int:
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import LocalTokenNotFoundError
    except ImportError as exc:
        log.error("UMA backend requires huggingface_hub: %s", exc)
        return 1
    try:
        HfApi().whoami()
    except LocalTokenNotFoundError:
        log.error(
            "Hugging Face token not found. Run `hf auth login` first "
            "(UMA models are gated and require an authorized account)."
        )
        return 1
    except Exception as exc:  # noqa: BLE001
        log.error(
            "Hugging Face authentication check failed (%s: %s). "
            "Run `hf auth login` and ensure UMA model access is approved.",
            type(exc).__name__, exc,
        )
        return 1
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    _configure_reactx_logging()

    if not args.rxn_path.exists():
        log.error("Error: .rxn not found: %s", args.rxn_path)
        return 1

    if args.backend == "uma":
        rc = _check_hf_auth()
        if rc != 0:
            return rc

    args.output.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()

    r_mol, p_mol, mapping = parse_rxn(args.rxn_path)
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)
    bond_changes = compute_simple_bond_changes(r_h, p_h, mapping)

    model_kwargs = {"model_name": args.model} if args.backend == "uma" else {}
    calc = make_calculator(args.backend, **model_kwargs)

    rotations = sample_attack_rotations(
        n=args.n_angles, cone_half_deg=args.cone_half_deg, seed=args.seed,
    )

    formed_pair = bond_changes.formed
    broken_pair = bond_changes.broken
    syms_r = [a.GetSymbol() for a in r_h.GetAtoms()]
    if args.r_form is None:
        r_form_target = lookup_r_form(syms_r[formed_pair[0]], syms_r[formed_pair[1]])
    else:
        r_form_target = float(args.r_form)

    trials: list[TrialResult] = []
    for i, R in enumerate(rotations):
        rot_deg = _angle_from_identity_deg(R)
        log.info("trial %d/%d (rotation_deg=%.1f)", i + 1, len(rotations), rot_deg)
        try:
            atoms_init = embed_mol_to_atoms(
                r_mol, calculator=None, seed=1 + i,
                bond_changes=bond_changes, rotation_perturbation=R,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("trial %d embed failed: %s", i, exc)
            trials.append(TrialResult(
                trial_idx=i, rotation_deg=rot_deg, frames=[], energies=[],
                reached_product=False, peak_energy=float("inf"), n_steps=0,
            ))
            continue

        restraints = build_restraints(
            atoms_init,
            formed=[formed_pair],
            broken=[broken_pair],
            r_form=r_form_target,
            r_broken=args.r_broken,
            k_form=args.k_form,
            k_broken=args.k_broken,
        )
        try:
            frames, energies = relax_with_restraints(
                atoms_init, restraints, calc,
                max_steps=args.max_relax_steps,
                fmax=args.relax_fmax,
                traj_stride=args.traj_stride,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("trial %d relax failed: %s", i, exc)
            trials.append(TrialResult(
                trial_idx=i, rotation_deg=rot_deg, frames=[], energies=[],
                reached_product=False, peak_energy=float("inf"), n_steps=0,
            ))
            continue

        ok = reached_product(
            frames[-1],
            formed=[formed_pair],
            broken=[broken_pair],
            r_form_targets=[r_form_target],
            r_broken_target=args.r_broken,
        )
        peak = max(energies) if energies else float("inf")
        trials.append(TrialResult(
            trial_idx=i, rotation_deg=rot_deg,
            frames=frames, energies=energies,
            reached_product=ok, peak_energy=float(peak),
            n_steps=len(frames),
        ))

    if not any(t.frames for t in trials):
        log.error("All trials failed. See meta.json for details.")
        return _write_outputs_and_exit(args, trials, t_start, neb_refined=False, rc=1)

    best = score_trials(trials)
    log.info(
        "selected trial %d (rotation_deg=%.1f, reached=%s, peak=%.4f)",
        best.trial_idx, best.rotation_deg, best.reached_product, best.peak_energy,
    )

    final_frames = best.frames
    neb_refined = False
    if args.neb_refine and len(final_frames) >= 2:
        log.info("running NEB refinement (%d images)", args.neb_images)
        product_raw = embed_mol_to_atoms(
            p_mol, calculator=calc, seed=2,
            bond_changes=bond_changes, rotation_perturbation=None,
        )
        rH = heavy_to_hydrogen_groups(r_h)
        pH = heavy_to_hydrogen_groups(p_h)
        product = align_product_to_reactant(
            final_frames[0], product_raw, mapping, rH, pH,
        )
        xyz_tmp = args.output / "trajectory_neb.xyz"
        run_neb(
            reactant=final_frames[0],
            product=product,
            calculator=calc,
            n_images=args.neb_images,
            output_xyz=xyz_tmp,
            fmax=0.05,
            max_steps=200,
            pad_frames=0,
        )
        # Re-read NEB frames as the new trajectory
        final_frames = read(str(xyz_tmp), index=":")
        neb_refined = True

    xyz = args.output / "trajectory.xyz"
    write(str(xyz), final_frames, format="extxyz")

    rc = _write_outputs_and_exit(args, trials, t_start, neb_refined=neb_refined, rc=0)
    if rc != 0:
        return rc

    if args.render:
        rc = _invoke_blender(args, xyz)
        if rc != 0:
            return rc

    log.info("OK: wrote %s (selected trial=%d)", xyz, best.trial_idx)
    return 0


def _angle_from_identity_deg(R: np.ndarray) -> float:
    """Return rotation angle of R in degrees (0 for identity)."""
    cos_t = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_t)))


def _write_outputs_and_exit(
    args: argparse.Namespace,
    trials: list[TrialResult],
    t_start: float,
    *,
    neb_refined: bool,
    rc: int,
) -> int:
    selected = -1
    converged = False
    if rc == 0 and trials:
        try:
            best = score_trials(trials)
            selected = best.trial_idx
            converged = best.reached_product
        except ValueError:
            pass
    meta = {
        "backend": args.backend,
        "converged": converged,
        "selected_trial": selected,
        "trials": [
            {
                "trial": t.trial_idx,
                "reached_product": t.reached_product,
                "peak_energy": float(t.peak_energy)
                    if math.isfinite(t.peak_energy) else None,
                "n_steps": t.n_steps,
                "rotation_deg": float(t.rotation_deg),
            }
            for t in trials
        ],
        "wall_clock_seconds": float(time.monotonic() - t_start),
        "neb_refined": neb_refined,
    }
    meta_clean = _sanitize_for_json(meta)
    (args.output / "meta.json").write_text(json.dumps(meta_clean, indent=2))

    if rc == 0 and trials:
        best = score_trials(trials)
        (args.output / "energies.json").write_text(json.dumps(best.energies))
    return rc


def _invoke_blender(args: argparse.Namespace, xyz: Path) -> int:
    import shutil
    import subprocess
    if shutil.which(args.blender_exe) is None:
        log.error(
            "Blender executable not found on PATH: %s. Install Blender 4.x "
            "(https://www.blender.org/download/) or pass --blender-exe "
            "/path/to/blender.",
            args.blender_exe,
        )
        return 1
    script = Path(__file__).resolve().parent.parent / "blender" / "render.py"
    blend = args.output / "scene.blend"
    cmd = [args.blender_exe, "--background", "--python", str(script),
           "--", str(xyz), str(blend)]
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        log.error("Error: blender exited with code %d", result.returncode)
        return 1
    return 0
```

- [ ] **Step 5: 引数 parser テスト 4 件 PASS**

Run: `pytest tests/test_cli.py -v`
Expected: 4 PASS

- [ ] **Step 6: 既存ユニットテスト全部 PASS**

Run: `pytest`
Expected: 全 PASS (slow / blender 除外で)

- [ ] **Step 7: Commit**

```bash
git add reactx/cli.py tests/test_cli.py
git commit -m "feat(cli): switch to multi-angle + restraint pipeline (Phase Re1)

NEB is now off by default. Each trial: embed (with rotation perturbation) →
restrained relaxation → reached_product check. score_trials picks best trial.
--neb-refine optionally refines the best trajectory with a short NEB."
```

---

## Task 9: tests/test_re1_sn2.py — SN2 統合テスト (slow)

UMA を呼ぶ end-to-end テスト。`@pytest.mark.slow` で CI 除外。

**Files:**
- Create: `tests/test_re1_sn2.py`

- [ ] **Step 1: テストを書く**

`tests/test_re1_sn2.py`:
```python
"""End-to-end SN2 test using UMA. Marked slow, requires HF auth + GPU."""
import json
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from reactx.cli import main


@pytest.mark.slow
def test_re1_sn2_end_to_end(tmp_path: Path, sn2_rxn_path: Path):
    out = tmp_path / "sn2"
    rc = main([
        "run", str(sn2_rxn_path), "-o", str(out),
        "--backend", "uma",
        "--n-angles", "4",          # smaller for test speed
        "--max-relax-steps", "50",
    ])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())
    assert meta["selected_trial"] >= 0
    # At least one trial reached product
    assert any(t["reached_product"] for t in meta["trials"]), (
        f"No trial reached product. trials={meta['trials']}"
    )

    frames = read(str(out / "trajectory.xyz"), index=":")
    assert len(frames) >= 3

    syms = frames[0].get_chemical_symbols()
    c_idx = syms.index("C")
    cl_idx = syms.index("Cl")
    f_idx = syms.index("F")

    # Walden inversion check #1: F-C-Cl angle peaks ≥ 120° somewhere on the path
    angles = []
    for f in frames:
        v_cf = f.positions[f_idx] - f.positions[c_idx]
        v_ccl = f.positions[cl_idx] - f.positions[c_idx]
        cos_t = float(np.dot(v_cf, v_ccl) / (
            np.linalg.norm(v_cf) * np.linalg.norm(v_ccl)
        ))
        angles.append(float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0)))))
    assert max(angles) >= 120.0, (
        f"F-C-Cl angle never reached 120° on the trajectory: max={max(angles):.1f}°"
    )

    # Walden inversion check #2: C-F shrinks, C-Cl grows (compare endpoints)
    d_cf_first = frames[0].get_distance(c_idx, f_idx)
    d_cf_last = frames[-1].get_distance(c_idx, f_idx)
    d_ccl_first = frames[0].get_distance(c_idx, cl_idx)
    d_ccl_last = frames[-1].get_distance(c_idx, cl_idx)
    assert d_cf_last < d_cf_first - 0.5, (
        f"C-F should shrink: {d_cf_first:.2f} -> {d_cf_last:.2f}"
    )
    assert d_ccl_last > d_ccl_first + 0.5, (
        f"C-Cl should grow: {d_ccl_first:.2f} -> {d_ccl_last:.2f}"
    )
```

- [ ] **Step 2: 実行 (slow marker で実行)**

Run: `pytest tests/test_re1_sn2.py -v -m slow`
Expected: PASS (UMA download 済み + GPU 利用可なら数十秒〜2 分)

- [ ] **Step 3: Commit**

```bash
git add tests/test_re1_sn2.py
git commit -m "test(re1_sn2): end-to-end SN2 with multi-angle + restraint pipeline

Walden inversion checks: F-C-Cl angle ≥ 120° on path, C-F shrinks, C-Cl grows."
```

---

## Task 10: tests/test_re1_proton_transfer.py — proton transfer 統合テスト (slow)

**Files:**
- Create: `tests/test_re1_proton_transfer.py`

- [ ] **Step 1: テストを書く**

`tests/test_re1_proton_transfer.py`:
```python
"""End-to-end proton transfer test (HCl + NH3 -> Cl- + NH4+)."""
import json
from pathlib import Path

import pytest
from ase.io import read

from reactx.cli import main


@pytest.fixture()
def proton_transfer_rxn_path(examples_dir: Path) -> Path:
    return examples_dir / "proton_transfer.rxn"


@pytest.mark.slow
def test_re1_proton_transfer_end_to_end(
    tmp_path: Path, proton_transfer_rxn_path: Path,
):
    out = tmp_path / "proton_transfer"
    rc = main([
        "run", str(proton_transfer_rxn_path), "-o", str(out),
        "--backend", "uma",
        "--n-angles", "4",
        "--max-relax-steps", "50",
        "--r-form", "1.05",  # N-H equilibrium
    ])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())
    assert meta["selected_trial"] >= 0
    assert any(t["reached_product"] for t in meta["trials"])

    frames = read(str(out / "trajectory.xyz"), index=":")
    syms = frames[0].get_chemical_symbols()
    cl_idx = syms.index("Cl")
    n_idx = syms.index("N")
    # The migrating H is the one bonded to Cl in the initial frame
    h_indices = [i for i, s in enumerate(syms) if s == "H"]
    initial_h_to_cl = [frames[0].get_distance(i, cl_idx) for i in h_indices]
    proton_idx = h_indices[initial_h_to_cl.index(min(initial_h_to_cl))]

    d_h_cl_first = frames[0].get_distance(proton_idx, cl_idx)
    d_h_cl_last = frames[-1].get_distance(proton_idx, cl_idx)
    d_h_n_first = frames[0].get_distance(proton_idx, n_idx)
    d_h_n_last = frames[-1].get_distance(proton_idx, n_idx)

    # H-Cl elongates, H-N shortens
    assert d_h_cl_last > d_h_cl_first + 0.3, (
        f"H-Cl should elongate: {d_h_cl_first:.2f} -> {d_h_cl_last:.2f}"
    )
    assert d_h_n_last < d_h_n_first - 0.5, (
        f"H-N should shrink: {d_h_n_first:.2f} -> {d_h_n_last:.2f}"
    )
```

- [ ] **Step 2: 実行**

Run: `pytest tests/test_re1_proton_transfer.py -v -m slow`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_re1_proton_transfer.py
git commit -m "test(re1_proton_transfer): end-to-end H migration HCl+NH3 -> Cl-+NH4+

H-Cl elongates while H-N shrinks; verifies generic substrate detection
(HCl is substrate even though NH3 has more atoms)."
```

---

## Task 11: tests/test_neb_refine_sn2.py — `--neb-refine` 経路テスト (slow)

**Files:**
- Create: `tests/test_neb_refine_sn2.py`

- [ ] **Step 1: テストを書く**

`tests/test_neb_refine_sn2.py`:
```python
"""--neb-refine flag exercises the optional NEB refinement on the best trial."""
import json
from pathlib import Path

import pytest
from ase.io import read

from reactx.cli import main


@pytest.mark.slow
def test_neb_refine_writes_refined_trajectory(
    tmp_path: Path, sn2_rxn_path: Path,
):
    out = tmp_path / "sn2_neb"
    rc = main([
        "run", str(sn2_rxn_path), "-o", str(out),
        "--backend", "uma",
        "--n-angles", "2",
        "--max-relax-steps", "30",
        "--neb-refine",
        "--neb-images", "5",
    ])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())
    assert meta["neb_refined"] is True

    frames = read(str(out / "trajectory.xyz"), index=":")
    # NEB with 5 images produces exactly 5 frames (no padding)
    assert len(frames) == 5
```

- [ ] **Step 2: 実行**

Run: `pytest tests/test_neb_refine_sn2.py -v -m slow`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_neb_refine_sn2.py
git commit -m "test(neb_refine): cover --neb-refine flag for SN2 best trajectory"
```

---

## Task 12: tests/test_wallclock_sn2.py — wall-clock 計測テスト

DoD #1 を満たすかを定量的にチェック。Phase 0 baseline 比 ≤ 50% は環境依存なので、ここでは「meta.json に wall_clock_seconds が記録される」「絶対値の上限 (環境依存だが緩めの保険)」のみ検証。

**Files:**
- Create: `tests/test_wallclock_sn2.py`

- [ ] **Step 1: テストを書く**

`tests/test_wallclock_sn2.py`:
```python
"""Wall-clock sanity check for Phase Re1 SN2 default settings.

Phase 0 NEB baseline on the same hardware is the comparison target; this test
records the wall-clock for the new pipeline so the README can quote it.
The hard upper bound is intentionally loose (≤ 300s) to stay environment-
agnostic; a dev-machine target ≤ 60s is documented but not asserted.
"""
import json
from pathlib import Path

import pytest

from reactx.cli import main


@pytest.mark.slow
def test_re1_sn2_wallclock_below_300s(tmp_path: Path, sn2_rxn_path: Path):
    out = tmp_path / "sn2_wc"
    rc = main([
        "run", str(sn2_rxn_path), "-o", str(out),
        "--backend", "uma",
    ])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())
    wc = meta["wall_clock_seconds"]
    assert wc > 0, f"wall_clock_seconds not recorded: {wc}"
    # Loose absolute bound; tighten in README based on actual measurement
    assert wc < 300.0, f"wall_clock_seconds={wc:.1f}s exceeds 300s upper bound"
```

- [ ] **Step 2: 実行 → wall-clock を記録**

Run: `pytest tests/test_wallclock_sn2.py -v -m slow -s`
Expected: PASS、ログに wall-clock 数値が出る (この値を README の Step に転記する)

- [ ] **Step 3: Commit**

```bash
git add tests/test_wallclock_sn2.py
git commit -m "test(wallclock): record SN2 default wall_clock_seconds for DoD #1"
```

---

## Task 13: tests/test_blender_smoke.py — SN2 + proton_transfer の parametrize

**Files:**
- Modify: `tests/test_blender_smoke.py`

- [ ] **Step 1: 現状確認**

Run: `cat tests/test_blender_smoke.py`

- [ ] **Step 2: parametrize で 2 反応に対応**

`tests/test_blender_smoke.py` を以下のように更新 (既存ロジックを `@pytest.mark.parametrize` で 2 ケースに展開):

```python
"""Blender smoke test: pipeline runs through to .blend without errors.

Requires Blender 4.x on PATH and the atomic-blender-pdb-xyz add-on enabled.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

from reactx.cli import main


@pytest.mark.blender
@pytest.mark.slow
@pytest.mark.parametrize("rxn_filename,extra_args", [
    ("sn2.rxn", []),
    ("proton_transfer.rxn", ["--r-form", "1.05"]),
])
def test_blender_smoke_writes_blend(
    tmp_path: Path, examples_dir: Path, rxn_filename: str, extra_args: list[str],
):
    if shutil.which("blender") is None:
        pytest.skip("blender not on PATH")

    rxn_path = examples_dir / rxn_filename
    out = tmp_path / rxn_filename.removesuffix(".rxn")
    rc = main([
        "run", str(rxn_path), "-o", str(out),
        "--backend", "uma",
        "--n-angles", "2",
        "--max-relax-steps", "30",
        "--render",
        *extra_args,
    ])
    assert rc == 0
    assert (out / "scene.blend").exists()
    assert (out / "trajectory.xyz").exists()
```

- [ ] **Step 3: 実行 (Blender + UMA インストール環境)**

Run: `pytest tests/test_blender_smoke.py -v -m "blender and slow"`
Expected: 2 PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_blender_smoke.py
git commit -m "test(blender_smoke): parametrize over SN2 + proton_transfer"
```

---

## Task 14: tests/test_neb_sn2.py の削除

NEB が default 経路から外れたため、Phase 0 SN2 NEB 統合テストは廃止。後継は `test_re1_sn2.py` + `test_neb_refine_sn2.py`。

**Files:**
- Delete: `tests/test_neb_sn2.py`

- [ ] **Step 1: ファイル削除**

```bash
git rm tests/test_neb_sn2.py
```

- [ ] **Step 2: 全テスト走らせて regression 無しを確認**

Run: `pytest`
Expected: 全 PASS、削除されたテスト名が collect されない

- [ ] **Step 3: Commit**

```bash
git commit -m "test: drop test_neb_sn2.py (replaced by test_re1_sn2 + test_neb_refine_sn2)"
```

---

## Task 15: README.md 更新

Phase Re1 の使い方、CLI flag、アニメーション目的の説明、Phase 0 比較。`tests/test_wallclock_sn2.py` の実測値を引用する。

**Files:**
- Modify: `README.md`

- [ ] **Step 1: README を更新**

`README.md` の「Phase 0 動作確認」「Phase 0 の既知の制約」セクションを Phase Re1 用に書き換え、新しい CLI 例 / wall-clock 測定値 / `--neb-refine` の使いどころを追加。冒頭の "reactx — Phase 0 Spike" は "reactx — Phase Re1 Multi-Angle Path Engine" に変更し、目的を「正確な TS エネルギーではなく、妥当なアニメーション」と明示。

最低限差し込むセクション:

```markdown
## 使い方 (Phase Re1)

```bash
# SN2
reactx run examples/sn2.rxn -o out/sn2/ --backend uma --render
# Proton transfer (HCl + NH3 -> Cl- + NH4+)
reactx run examples/proton_transfer.rxn -o out/pt/ \
  --backend uma --r-form 1.05 --render
```

主要 flag:

- `--n-angles 8` (default): 求核剤の入射角を 8 通りサンプリングし best を選ぶ
- `--cone-half-deg 30.0`: 多角度試行の cone 半角
- `--r-form` (default: 元素ペアから自動): 形成結合の目標距離 (Å)
- `--neb-refine` (default off): best trajectory を NEB で smoothing する。アニメーション目的だけなら不要

## Phase Re1 の方針

正確な TS エネルギーではなく **妥当なアニメーション** を目的としている。NEB は default で外し、Hookean (引力) + PullApart (斥力) restraint 下で多角度 FIRE relaxation を走らせて trajectory を生成する。詳細: `docs/superpowers/specs/2026-04-27-reactx-phase-Re1-design.md`。

## Wall-clock (SN2 default)

ローカル実測 (RTX 4090 + UMA-m-1p1): 約 X 秒 (Phase 0 NEB ベースの Y 秒比 約 Z%)。`tests/test_wallclock_sn2.py` で計測可能。
```

(X / Y / Z は Task 12 で得られた値で埋める)

- [ ] **Step 2: lint で空行警告などが無いことを確認**

Run: `ruff check README.md` → markdown は ruff 対象外なのでスキップ。代わりに `git diff README.md` で目視確認。

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(readme): document Phase Re1 multi-angle pipeline + wall-clock results

Replaces Phase 0 sections with Phase Re1 usage, CLI flags, and a measured
SN2 wall-clock (X s vs Phase 0 Y s = Z%)."
```

---

## Task 16: 最終確認 — 全テスト + 実反応 2 件で Blender 視認

**Files:** (なし。手動 + 既存テスト実行)

- [ ] **Step 1: 全テスト pass**

Run: `pytest && pytest -m slow`
Expected: 全 PASS

- [ ] **Step 2: SN2 を実走らせて Blender で視認**

```bash
reactx run examples/sn2.rxn -o out/sn2/ --backend uma --render
```
- `out/sn2/scene.blend` を Blender GUI で開く
- Walden inversion (F⁻ 接近 → C 反転 → Cl⁻ 脱離) が視認できることを目視確認

- [ ] **Step 3: proton transfer を実走らせて Blender で視認**

```bash
reactx run examples/proton_transfer.rxn -o out/pt/ \
  --backend uma --r-form 1.05 --render
```
- `out/pt/scene.blend` を Blender GUI で開く
- H が Cl から N に移動する過程が視認できることを目視確認

- [ ] **Step 4: DoD #1 を README に反映**

Task 15 で README に書いた wall-clock の実測値を、Task 12 の出力で更新。Phase 0 baseline (もし手元にあれば) と比較して比率を記載。

```bash
git add README.md
git commit -m "docs(readme): finalize Phase Re1 wall-clock measurement"
```

- [ ] **Step 5: Phase Re1 完了、ブランチを push**

```bash
git push -u origin phase-Re1
```

---

## 実装上の注意

- **shared calculator**: UMA は ~11 GB のためモデルを 1 度だけロードして全 trial で共有する (`cli.py` の `_cmd_run` で `make_calculator` を 1 回呼び、`relax_with_restraints` に毎回渡す)。
- **frame の calc detach**: `path_relax._snapshot` で `a.calc = None` する。`extxyz` writer は positions/symbols のみ参照するが、calc を持ったまま長時間保持するとメモリリーク的になりうる。
- **fmax の単位**: ASE の fmax は eV/Å。relax_fmax=0.1 は NEB の 0.05 より緩い (path 生成目的、平衡構造でなくて OK)。
- **H atom の atom map**: proton_transfer.rxn では proton (移動 H) のみ explicit に書いて map=1 を付与し、NH3 / NH4+ の他の H は implicit にしている。`compute_simple_bond_changes` の `_build_full_atom_mapping` がこの組み合わせを処理する。
- **`bond_changes` 引数の互換性**: single-fragment (e.g. `Chem.MolFromSmiles("CCl")`) を `embed_mol_to_atoms` に渡すとき、`bond_changes=None` で OK。multi-fragment では必須。
- **Phase 0 backward compat**: `--n-angles 1` で実行すると rotation index 0 = identity が唯一の trial となり、Phase 0 と equivalent な単一試行になる。回帰確認に使える。
- **spec §11 で言及した「全 trial 失敗時の retry フォールバック」 (cone を 45° に拡げる、k_form 倍化など) は本 plan では未実装**。MVP として 8 角度の default で SN2 + proton transfer が通ることを優先する。default 設定で全滅するならパラメータ調整が必要 (retry では救えない構造的問題) という判断。SN2 / proton transfer が安定したらバックポート可能。
- **`tests/test_default_r_form.py` の独立ファイル化 (spec §6.1 で言及)** は実装簡略化のため `tests/test_artificial_force.py` に統合した。テスト責務はカバーされている。
