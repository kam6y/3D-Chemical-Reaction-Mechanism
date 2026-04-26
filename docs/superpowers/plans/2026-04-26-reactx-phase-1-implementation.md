# reactx Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Phase 0 で SN2 にハードコードした embedding ロジックを generic bond-change geometry engine に置き換え、SN2 / 解離 / E2 / E1 step2 / proton transfer の 5 反応を同じパイプラインで end-to-end (.rxn → trajectory.xyz → Blender 視認) で扱えるようにする。

**Architecture:** `reactx/reaction_topology.py` (`compute_bond_changes`) と `reactx/placement.py` (`place_fragments_generic`) を新規追加。`embed3d.py` は SN2 専用ロジックを撤去し generic engine を呼び出すよう改修。`cli.py` は bond_changes を計算して embed3d に渡し、images 数を反応複雑度に応じて自動調整。反応分類フラグや strategy registry は導入しない (spec §1)。

**Tech Stack:** Python 3.11+ / RDKit 2025.3 / ASE 3.26 / fairchem-core 2.14 (UMA) / numpy / Blender 4.x

**Spec:** `docs/superpowers/specs/2026-04-26-reactx-phase-1-design.md`

---

## File Structure

### 新規作成

| Path | 責務 |
|---|---|
| `reactx/reaction_topology.py` | `BondChange` / `BondChanges` dataclass、`compute_bond_changes` |
| `reactx/placement.py` | `place_fragments_generic` (Kabsch alignment 含む)、内部ヘルパ `_compute_ideal_position` |
| `examples/sn1_step1.rxn` | 解離反応サンプル: `(CH3)3C-Br → (CH3)3C+ + Br-` |
| `examples/e2.rxn` | E2 サンプル: `CH3CH2Br + OH- → CH2=CH2 + H2O + Br-` |
| `examples/e1_step2.rxn` | β-H 脱離サンプル: `(CH3)3C+ → (CH3)2C=CH2 + H+` |
| `examples/proton_transfer.rxn` | プロトン移動サンプル: `HCl + NH3 → Cl- + NH4+` |
| `tests/test_reaction_topology.py` | bond change 抽出のユニットテスト |
| `tests/test_placement.py` | generic placement のユニットテスト |
| `tests/test_recommend_n_images.py` | `recommend_n_images` 純粋関数の単体テスト |
| `tests/test_neb_dissociation.py` | 解離反応の slow 統合テスト |
| `tests/test_neb_e2.py` | E2 の slow 統合テスト |
| `tests/test_neb_e1_step2.py` | β-H 脱離の slow 統合テスト |
| `tests/test_neb_proton_transfer.py` | proton transfer の slow 統合テスト |

### 変更

| Path | 変更内容 |
|---|---|
| `reactx/embed3d.py` | `_find_c_lg_bond` / `_place_nucleophile_backside` 削除、`bond_changes`/`side` 引数追加、generic placement 呼び出し |
| `reactx/cli.py` | `compute_bond_changes` を呼び、`recommend_n_images` で images 自動計算、embed に bond_changes/side を渡す |
| `tests/conftest.py` | 新 `.rxn` パス用 fixture を追加 |
| `tests/test_embed3d.py` | 既存テストの `embed_mol_to_atoms` 呼び出しに `bond_changes=None` 互換性確保 (引数追加に伴う) |
| `tests/test_blender_smoke.py` | 5 反応の `.blend` 生成を smoke test 化 |
| `README.md` | Phase 1 の使い方 / multi-step 反応の慣習 / generic placement 概要 |

---

## Task 1: 例 .rxn ファイル 4 件を作成 (atom map 番号付き)

**Files:**
- Create: `examples/sn1_step1.rxn`
- Create: `examples/e2.rxn`
- Create: `examples/e1_step2.rxn`
- Create: `examples/proton_transfer.rxn`

これらはユニットテストの fixture でも使うため最初に作成する。`.rxn` は MDL Rxn V2000 で、各原子に atom map 番号 (`mapNum`) が必要 (V2000 仕様の atom block 第 14 列)。SN2 の `examples/sn2.rxn` を参考に手書きする。

- [ ] **Step 1: examples/proton_transfer.rxn を作成**

`HCl + NH3 → Cl- + NH4+` (atom indexing: H1=H, Cl=2, N=3, H4=H4 of NH3, H5=H5 of NH3, H6=H6 of NH3 — 全 6 原子で全 H を explicit に書く)

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

  4  3  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  3  0  0
    1.0000    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  4  0  0
   -0.5000    0.8660    0.0000 H   0  0  0  0  0  0  0  0  0  5  0  0
   -0.5000   -0.8660    0.0000 H   0  0  0  0  0  0  0  0  0  6  0  0
  1  2  1  0
  1  3  1  0
  1  4  1  0
M  END
$MOL

     RDKit          2D

  1  0  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 Cl  0  0  0  0  0  0  0  0  0  2  0  0
M  CHG  1   1  -1
M  END
$MOL

     RDKit          2D

  5  4  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 N   0  0  0  0  0  0  0  0  0  3  0  0
    1.0000    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  4  0  0
   -0.5000    0.8660    0.0000 H   0  0  0  0  0  0  0  0  0  5  0  0
   -0.5000   -0.8660    0.0000 H   0  0  0  0  0  0  0  0  0  6  0  0
    0.0000    0.0000    1.0000 H   0  0  0  0  0  0  0  0  0  1  0  0
  1  2  1  0
  1  3  1  0
  1  4  1  0
  1  5  1  0
M  CHG  1   1  1
M  END
```

- [ ] **Step 2: examples/sn1_step1.rxn を作成**

`(CH3)3C-Br → (CH3)3C+ + Br-`. 重原子のみ explicit (H は AddHs で補完される)。t-butyl の重原子 4 個 (C×4) + Br の計 5 重原子。

```
$RXN

      RDKit

  1  2
$MOL

     RDKit          2D

  5  4  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  1  0  0
    1.5000    0.0000    0.0000 Br  0  0  0  0  0  0  0  0  0  2  0  0
   -0.7500    1.3000    0.0000 C   0  0  0  0  0  0  0  0  0  3  0  0
   -0.7500   -1.3000    0.0000 C   0  0  0  0  0  0  0  0  0  4  0  0
   -1.5000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  5  0  0
  1  2  1  0
  1  3  1  0
  1  4  1  0
  1  5  1  0
M  END
$MOL

     RDKit          2D

  4  3  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  1  0  0
   -0.7500    1.3000    0.0000 C   0  0  0  0  0  0  0  0  0  3  0  0
   -0.7500   -1.3000    0.0000 C   0  0  0  0  0  0  0  0  0  4  0  0
   -1.5000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  5  0  0
  1  2  1  0
  1  3  1  0
  1  4  1  0
M  CHG  1   1  1
M  END
$MOL

     RDKit          2D

  1  0  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 Br  0  0  0  0  0  0  0  0  0  2  0  0
M  CHG  1   1  -1
M  END
```

- [ ] **Step 3: examples/e2.rxn を作成**

`CH3CH2Br + OH- → CH2=CH2 + H2O + Br-`. reactant 重原子: Cα=1, Cβ=2, Br=3, O=4 (4 個)。product 重原子も 4 個 (mapping 一致)。E2 の β-H は AddHs 後に explicit に表現する必要があるため、Cβ の H を 1 つだけ explicit に書く。

```
$RXN

      RDKit

  2  3
$MOL

     RDKit          2D

  5  4  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  1  0  0
    1.5000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  2  0  0
    2.2500    1.3000    0.0000 Br  0  0  0  0  0  0  0  0  0  3  0  0
   -0.7500    1.3000    0.0000 H   0  0  0  0  0  0  0  0  0  5  0  0
    0.0000    0.0000    1.5000 H   0  0  0  0  0  0  0  0  0  6  0  0
  1  2  1  0
  2  3  1  0
  1  4  1  0
  1  5  1  0
M  END
$MOL

     RDKit          2D

  2  1  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  4  0  0
    1.0000    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  6  0  0
  1  2  1  0
M  CHG  1   1  -1
M  END
$MOL

     RDKit          2D

  3  2  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  1  0  0
    1.3500    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  2  0  0
   -0.7500    1.3000    0.0000 H   0  0  0  0  0  0  0  0  0  5  0  0
  1  2  2  0
  1  3  1  0
M  END
$MOL

     RDKit          2D

  3  2  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  4  0  0
    1.0000    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  6  0  0
   -0.5000    0.8660    0.0000 H   0  0  0  0  0  0  0  0  0  7  0  0
  1  2  1  0
  1  3  1  0
M  END
$MOL

     RDKit          2D

  1  0  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 Br  0  0  0  0  0  0  0  0  0  3  0  0
M  CHG  1   1  -1
M  END
```

(注: H atom が両側で explicit に対応している必要がある。Cβ から OH⁻ に移動する H は map 6 で reactant の Cβ 結合と product の O 結合を結ぶ。reactant の Cα には map 5 の H を 1 つ explicit にして product の C=C 側に対応付ける。RDKit AddHs は残りの H を自動補完する。)

- [ ] **Step 4: examples/e1_step2.rxn を作成**

`(CH3)3C+ → (CH3)2C=CH2 + H+`. (CH3)3C+ から β-H が抜ける。reactant 重原子 4 個 + Cβ の H 1 個 = 5、product 重原子 4 個 + 抜けた H 1 個 = 5。

```
$RXN

      RDKit

  1  2
$MOL

     RDKit          2D

  5  4  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  1  0  0
   -0.7500    1.3000    0.0000 C   0  0  0  0  0  0  0  0  0  2  0  0
   -0.7500   -1.3000    0.0000 C   0  0  0  0  0  0  0  0  0  3  0  0
    1.5000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  4  0  0
   -1.7500    1.5000    0.0000 H   0  0  0  0  0  0  0  0  0  5  0  0
  1  2  1  0
  1  3  1  0
  1  4  1  0
  2  5  1  0
M  CHG  1   1  1
M  END
$MOL

     RDKit          2D

  4  3  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  1  0  0
   -0.7500    1.3000    0.0000 C   0  0  0  0  0  0  0  0  0  2  0  0
   -0.7500   -1.3000    0.0000 C   0  0  0  0  0  0  0  0  0  3  0  0
    1.5000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  4  0  0
  1  2  2  0
  1  3  1  0
  1  4  1  0
M  END
$MOL

     RDKit          2D

  1  0  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  5  0  0
M  CHG  1   1  1
M  END
```

- [ ] **Step 5: 4 ファイルが parse_rxn で読めることを手動確認**

```bash
python -c "from reactx.rxn_parser import parse_rxn; \
for n in ['sn1_step1','e2','e1_step2','proton_transfer']:\
  r,p,m = parse_rxn(f'examples/{n}.rxn'); \
  print(n, 'reactant heavy:', r.GetNumHeavyAtoms(), 'product heavy:', p.GetNumHeavyAtoms(), 'mapping:', m)"
```

Expected: 各反応で reactant と product の重原子数が一致し、mapping dict が空でない。

- [ ] **Step 6: Commit**

```bash
git add examples/sn1_step1.rxn examples/e2.rxn examples/e1_step2.rxn examples/proton_transfer.rxn
git commit -m "feat(examples): add SN1/E2/E1-step2/proton-transfer .rxn files for Phase 1"
```

---

## Task 2: BondChange / BondChanges dataclass + 空 compute_bond_changes

**Files:**
- Create: `reactx/reaction_topology.py`
- Create: `tests/test_reaction_topology.py`

- [ ] **Step 1: 失敗する import テストを書く**

`tests/test_reaction_topology.py`:

```python
"""Tests for reactx.reaction_topology module."""
from reactx.reaction_topology import BondChange, BondChanges


def test_bondchange_dataclass_construction():
    bc = BondChange(a=0, b=1, order_before=1.0, order_after=0.0)
    assert bc.a == 0
    assert bc.b == 1
    assert bc.order_before == 1.0
    assert bc.order_after == 0.0


def test_bondchanges_default_lists():
    changes = BondChanges(broken=[], formed=[])
    assert changes.broken == []
    assert changes.formed == []
```

- [ ] **Step 2: テストを実行して fail を確認**

```bash
pytest tests/test_reaction_topology.py -v
```

Expected: `ModuleNotFoundError: No module named 'reactx.reaction_topology'`

- [ ] **Step 3: 最小実装**

`reactx/reaction_topology.py`:

```python
"""Bond-change extraction from .rxn atom mapping."""
from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem


@dataclass(frozen=True)
class BondChange:
    a: int
    b: int
    order_before: float
    order_after: float


@dataclass(frozen=True)
class BondChanges:
    broken: list[BondChange]
    formed: list[BondChange]


def compute_bond_changes(
    reactant_mol_h: Chem.Mol,
    product_mol_h: Chem.Mol,
    heavy_mapping: dict[int, int],
) -> BondChanges:
    raise NotImplementedError("filled in next task")
```

- [ ] **Step 4: テストが pass することを確認**

```bash
pytest tests/test_reaction_topology.py -v
```

Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add reactx/reaction_topology.py tests/test_reaction_topology.py
git commit -m "feat(reaction_topology): scaffold BondChange/BondChanges dataclasses"
```

---

## Task 3: compute_bond_changes 実装 (SN2 ケース)

**Files:**
- Modify: `reactx/reaction_topology.py`
- Modify: `tests/test_reaction_topology.py`
- Modify: `tests/conftest.py`

- [ ] **Step 1: conftest.py に 4 反応の fixture を追加**

`tests/conftest.py` の末尾に追記:

```python
@pytest.fixture()
def sn1_step1_rxn_path(examples_dir: Path) -> Path:
    return examples_dir / "sn1_step1.rxn"


@pytest.fixture()
def e2_rxn_path(examples_dir: Path) -> Path:
    return examples_dir / "e2.rxn"


@pytest.fixture()
def e1_step2_rxn_path(examples_dir: Path) -> Path:
    return examples_dir / "e1_step2.rxn"


@pytest.fixture()
def proton_transfer_rxn_path(examples_dir: Path) -> Path:
    return examples_dir / "proton_transfer.rxn"
```

- [ ] **Step 2: SN2 用の bond change テストを書く**

`tests/test_reaction_topology.py` に追記:

```python
from pathlib import Path

from rdkit import Chem

from reactx.reaction_topology import compute_bond_changes
from reactx.rxn_parser import parse_rxn


def _bond_pair(bc) -> tuple[int, int]:
    return (min(bc.a, bc.b), max(bc.a, bc.b))


def test_sn2_bond_changes(sn2_rxn_path: Path):
    r_mol, p_mol, mapping = parse_rxn(sn2_rxn_path)
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)
    changes = compute_bond_changes(r_h, p_h, mapping)

    # SN2: 1 broken (C-Cl) + 1 formed (C-F)
    assert len(changes.broken) == 1
    assert len(changes.formed) == 1

    r_syms = [a.GetSymbol() for a in r_h.GetAtoms()]
    broken = changes.broken[0]
    formed = changes.formed[0]
    broken_syms = sorted([r_syms[broken.a], r_syms[broken.b]])
    formed_syms = sorted([r_syms[formed.a], r_syms[formed.b]])
    assert broken_syms == ["C", "Cl"]
    assert formed_syms == ["C", "F"]
    assert broken.order_before == 1.0 and broken.order_after == 0.0
    assert formed.order_before == 0.0 and formed.order_after == 1.0
```

- [ ] **Step 3: テストを実行して fail を確認**

```bash
pytest tests/test_reaction_topology.py::test_sn2_bond_changes -v
```

Expected: FAIL with `NotImplementedError`

- [ ] **Step 4: compute_bond_changes 本体を実装**

`reactx/reaction_topology.py` の `compute_bond_changes` を以下で置換:

```python
def compute_bond_changes(
    reactant_mol_h: Chem.Mol,
    product_mol_h: Chem.Mol,
    heavy_mapping: dict[int, int],
) -> BondChanges:
    """Extract broken/formed bonds by comparing reactant and product graphs.

    Indices in the returned BondChange.a / .b refer to *reactant* mol_h atom
    indices. Product bonds are projected back into reactant indexing via the
    expanded H mapping built from heavy_mapping + each heavy atom's bonded H
    count (assumes preserved within a heavy mapping pair).
    """
    expanded = _build_expanded_mapping(reactant_mol_h, product_mol_h, heavy_mapping)
    inv_expanded = {p: r for r, p in expanded.items()}

    r_bonds = _bond_orders(reactant_mol_h)  # {(min,max): order} in reactant idx
    p_bonds_in_r = {}
    for (pa, pb), order in _bond_orders(product_mol_h).items():
        if pa not in inv_expanded or pb not in inv_expanded:
            # bond involves an atom not present on the reactant side — should
            # never happen if mapping is complete; defensive raise.
            raise ValueError(
                f"Product bond {pa}-{pb} has no counterpart in reactant mapping"
            )
        ra, rb = inv_expanded[pa], inv_expanded[pb]
        p_bonds_in_r[(min(ra, rb), max(ra, rb))] = order

    all_keys = set(r_bonds) | set(p_bonds_in_r)
    broken: list[BondChange] = []
    formed: list[BondChange] = []
    for key in sorted(all_keys):
        before = r_bonds.get(key, 0.0)
        after = p_bonds_in_r.get(key, 0.0)
        if before == after:
            continue
        bc = BondChange(a=key[0], b=key[1], order_before=before, order_after=after)
        if after < before:
            broken.append(bc)
        else:
            formed.append(bc)

    return BondChanges(broken=broken, formed=formed)


def _bond_orders(mol_h: Chem.Mol) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    for bond in mol_h.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        bt = bond.GetBondTypeAsDouble()
        if bt == 1.5:
            raise NotImplementedError(
                "Aromatic bonds are out of scope in Phase 1; kekulize input"
            )
        out[(min(a, b), max(a, b))] = bt
    return out


def _build_expanded_mapping(
    r_mol_h: Chem.Mol,
    p_mol_h: Chem.Mol,
    heavy_mapping: dict[int, int],
) -> dict[int, int]:
    """Extend heavy_mapping to include H atoms by positional matching.

    For each heavy r_idx -> p_idx, take the bonded H children on each side and
    pair them in encounter order. Reaction conservation guarantees equal H
    counts when the .rxn includes all participating Hs explicitly (E2 / proton
    transfer cases) or AddHs adds matching H counts on each side.
    """
    expanded: dict[int, int] = {}
    for r_idx, p_idx in heavy_mapping.items():
        expanded[r_idx] = p_idx
        r_atom = r_mol_h.GetAtomWithIdx(r_idx)
        p_atom = p_mol_h.GetAtomWithIdx(p_idx)
        r_hs = [n.GetIdx() for n in r_atom.GetNeighbors() if n.GetAtomicNum() == 1]
        p_hs = [n.GetIdx() for n in p_atom.GetNeighbors() if n.GetAtomicNum() == 1]
        if len(r_hs) != len(p_hs):
            # H count differs -> migrating H. Resolved at top level by walking
            # explicit-mapped H atoms (those with atom map number assigned in
            # the .rxn) instead of relying on per-heavy positional pairing.
            continue
        for rh, ph in zip(r_hs, p_hs, strict=True):
            expanded[rh] = ph

    # Fill in explicitly mapped H atoms that didn't get paired above (migrating
    # H in proton transfer / E2). atom map numbers persist through AddHs.
    r_map_to_idx = {}
    for atom in r_mol_h.GetAtoms():
        if atom.GetAtomicNum() == 1 and atom.GetAtomMapNum() != 0:
            r_map_to_idx[atom.GetAtomMapNum()] = atom.GetIdx()
    for atom in p_mol_h.GetAtoms():
        if atom.GetAtomicNum() == 1 and atom.GetAtomMapNum() != 0:
            p_idx = atom.GetIdx()
            mn = atom.GetAtomMapNum()
            if mn in r_map_to_idx:
                expanded[r_map_to_idx[mn]] = p_idx
    return expanded
```

- [ ] **Step 5: テストを実行して pass を確認**

```bash
pytest tests/test_reaction_topology.py -v
```

Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add reactx/reaction_topology.py tests/test_reaction_topology.py tests/conftest.py
git commit -m "feat(reaction_topology): implement compute_bond_changes for SN2"
```

---

## Task 4: compute_bond_changes — 残り 4 反応のテスト追加

**Files:**
- Modify: `tests/test_reaction_topology.py`

実装は Task 3 で既に完了している前提。各反応について bond change が正しく抽出されることを確認。

- [ ] **Step 1: 4 反応分のテストを追加**

`tests/test_reaction_topology.py` に追記:

```python
def test_sn1_step1_bond_changes(sn1_step1_rxn_path: Path):
    r_mol, p_mol, mapping = parse_rxn(sn1_step1_rxn_path)
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)
    changes = compute_bond_changes(r_h, p_h, mapping)

    # heterolytic dissociation: 1 broken (C-Br) + 0 formed
    assert len(changes.broken) == 1
    assert len(changes.formed) == 0
    r_syms = [a.GetSymbol() for a in r_h.GetAtoms()]
    broken = changes.broken[0]
    assert sorted([r_syms[broken.a], r_syms[broken.b]]) == ["Br", "C"]


def test_proton_transfer_bond_changes(proton_transfer_rxn_path: Path):
    r_mol, p_mol, mapping = parse_rxn(proton_transfer_rxn_path)
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)
    changes = compute_bond_changes(r_h, p_h, mapping)

    # H-Cl broken, H-N formed
    assert len(changes.broken) == 1
    assert len(changes.formed) == 1
    r_syms = [a.GetSymbol() for a in r_h.GetAtoms()]
    assert sorted([r_syms[changes.broken[0].a], r_syms[changes.broken[0].b]]) == ["Cl", "H"]
    assert sorted([r_syms[changes.formed[0].a], r_syms[changes.formed[0].b]]) == ["H", "N"]


def test_e2_bond_changes(e2_rxn_path: Path):
    r_mol, p_mol, mapping = parse_rxn(e2_rxn_path)
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)
    changes = compute_bond_changes(r_h, p_h, mapping)

    # broken: Cα-Br, Cβ-H, formed: O-H, Cα=Cβ (order 1->2)
    assert len(changes.broken) == 2
    assert len(changes.formed) == 2

    def pair_syms(bc) -> tuple[str, str]:
        return tuple(sorted([
            r_h.GetAtomWithIdx(bc.a).GetSymbol(),
            r_h.GetAtomWithIdx(bc.b).GetSymbol(),
        ]))

    broken_pairs = sorted(pair_syms(b) for b in changes.broken)
    formed_pairs = sorted(pair_syms(f) for f in changes.formed)
    assert ("Br", "C") in broken_pairs
    assert ("C", "H") in broken_pairs
    assert ("H", "O") in formed_pairs
    # Cα-Cβ formed (1->2)
    cc_formed = [f for f in changes.formed
                 if pair_syms(f) == ("C", "C")]
    assert len(cc_formed) == 1
    assert cc_formed[0].order_before == 1.0
    assert cc_formed[0].order_after == 2.0


def test_e1_step2_bond_changes(e1_step2_rxn_path: Path):
    r_mol, p_mol, mapping = parse_rxn(e1_step2_rxn_path)
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)
    changes = compute_bond_changes(r_h, p_h, mapping)

    # broken: Cβ-H, formed: Cα-Cβ order 1->2
    assert len(changes.broken) == 1
    assert len(changes.formed) == 1
    syms = [a.GetSymbol() for a in r_h.GetAtoms()]
    assert sorted([syms[changes.broken[0].a], syms[changes.broken[0].b]]) == ["C", "H"]
    formed = changes.formed[0]
    assert sorted([syms[formed.a], syms[formed.b]]) == ["C", "C"]
    assert formed.order_before == 1.0 and formed.order_after == 2.0
```

- [ ] **Step 2: テストを実行**

```bash
pytest tests/test_reaction_topology.py -v
```

Expected: PASS (7 tests total)

実装に bug があれば修正して PASS にする。特に以下が落ちやすい:
- proton transfer の H 移動 (`_build_expanded_mapping` 内の atom-map-based fallback が使われる)
- E2 の bond order 1→2 (`_bond_orders` が `GetBondTypeAsDouble` を正しく拾えているか)

- [ ] **Step 3: Commit**

```bash
git add tests/test_reaction_topology.py
git commit -m "test(reaction_topology): add SN1/E2/E1-step2/proton-transfer cases"
```

---

## Task 5: recommend_n_images 純粋関数

**Files:**
- Modify: `reactx/cli.py`
- Create: `tests/test_recommend_n_images.py`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_recommend_n_images.py`:

```python
"""Tests for recommend_n_images pure function."""
from reactx.cli import recommend_n_images
from reactx.reaction_topology import BondChange, BondChanges


def _bc(n: int) -> list[BondChange]:
    return [BondChange(a=0, b=1, order_before=1.0, order_after=0.0) for _ in range(n)]


def test_recommend_n_images_single_bond_change():
    # 1 broken + 1 formed -> max(11, 9 + 2 + 2) = 13
    changes = BondChanges(broken=_bc(1), formed=_bc(1))
    assert recommend_n_images(changes) == 13


def test_recommend_n_images_dissociation():
    # 1 broken + 0 formed -> max(11, 9 + 2) = 11
    changes = BondChanges(broken=_bc(1), formed=_bc(0))
    assert recommend_n_images(changes) == 11


def test_recommend_n_images_e2():
    # 2 broken + 2 formed -> max(11, 9 + 4 + 4) = 17
    changes = BondChanges(broken=_bc(2), formed=_bc(2))
    assert recommend_n_images(changes) == 17


def test_recommend_n_images_floor():
    # 0 + 0 -> floor at 11
    assert recommend_n_images(BondChanges(broken=[], formed=[])) == 11
```

- [ ] **Step 2: テスト実行 (fail を確認)**

```bash
pytest tests/test_recommend_n_images.py -v
```

Expected: FAIL with `ImportError: cannot import name 'recommend_n_images' from 'reactx.cli'`

- [ ] **Step 3: cli.py に関数を追加**

`reactx/cli.py` の import 後・`build_parser` 直前に追加:

```python
from reactx.reaction_topology import BondChanges


def recommend_n_images(bond_changes: BondChanges) -> int:
    """Default NEB image count from bond-change complexity.

    formula: max(11, 9 + 2 * n_broken + 2 * n_formed). Override with --images.
    """
    return max(
        11,
        9 + 2 * len(bond_changes.broken) + 2 * len(bond_changes.formed),
    )
```

- [ ] **Step 4: テスト実行 (pass を確認)**

```bash
pytest tests/test_recommend_n_images.py -v
```

Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add reactx/cli.py tests/test_recommend_n_images.py
git commit -m "feat(cli): add recommend_n_images pure helper"
```

---

## Task 6: place_fragments_generic — 単一 anchor (translation のみ)

**Files:**
- Create: `reactx/placement.py`
- Create: `tests/test_placement.py`

- [ ] **Step 1: 失敗する SN2 placement テストを書く**

`tests/test_placement.py`:

```python
"""Tests for reactx.placement."""
from pathlib import Path

import numpy as np
from rdkit import Chem

from reactx.placement import place_fragments_generic
from reactx.reaction_topology import compute_bond_changes
from reactx.rxn_parser import parse_rxn


def _embed_for_test(mol_h, seed=42):
    """Run RDKit ETKDG only (no UMA) to get test positions, with one ETKDG seed."""
    from rdkit.Chem import AllChem
    frag_mols = Chem.GetMolFrags(mol_h, asMols=True, sanitizeFrags=True)
    frag_indices = Chem.GetMolFrags(mol_h)
    positions = np.zeros((mol_h.GetNumAtoms(), 3))
    for i, (frag, idxs) in enumerate(zip(frag_mols, frag_indices, strict=True)):
        params = AllChem.ETKDGv3()
        params.randomSeed = seed + i
        AllChem.EmbedMolecule(frag, params)
        if frag.GetNumHeavyAtoms() > 1:
            AllChem.MMFFOptimizeMolecule(frag, maxIters=200)
        conf = frag.GetConformer()
        for j, orig in enumerate(idxs):
            p = conf.GetAtomPosition(j)
            positions[orig] = (p.x, p.y, p.z)
    return frag_indices, positions


def test_sn2_reactant_placement_puts_F_on_backside_of_C(sn2_rxn_path: Path):
    r_mol, p_mol, mapping = parse_rxn(sn2_rxn_path)
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)
    bc = compute_bond_changes(r_h, p_h, mapping)
    frag_indices, positions = _embed_for_test(r_h)

    placed = place_fragments_generic(
        r_h, frag_indices, positions, bc, side="reactant",
    )

    syms = [a.GetSymbol() for a in r_h.GetAtoms()]
    c = syms.index("C")
    cl = syms.index("Cl")
    f = syms.index("F")

    # F on -unit(C->Cl) side of C: dot < -0.7
    c_to_cl = placed[cl] - placed[c]
    c_to_f = placed[f] - placed[c]
    cos_theta = (
        np.dot(c_to_cl, c_to_f)
        / (np.linalg.norm(c_to_cl) * np.linalg.norm(c_to_f))
    )
    assert cos_theta < -0.7, f"F not on backside of C-Cl: cos(theta)={cos_theta:.3f}"

    # |C-F| should be near d_form = 3.0 (within ±1.0)
    cf_dist = float(np.linalg.norm(c_to_f))
    assert 2.0 <= cf_dist <= 4.0, f"|C-F| = {cf_dist:.3f} Å, expected ~3.0±1.0"
```

- [ ] **Step 2: テスト実行 (fail を確認)**

```bash
pytest tests/test_placement.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'reactx.placement'`

- [ ] **Step 3: placement.py を最小実装 (translation 1-anchor のみ)**

`reactx/placement.py`:

```python
"""Generic bond-change-driven fragment placement.

Replaces Phase 0's SN2-hardcoded backside-attack placement
(_find_c_lg_bond / _place_nucleophile_backside in embed3d).
"""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from rdkit import Chem

from reactx.reaction_topology import BondChange, BondChanges

log = logging.getLogger(__name__)

DEFAULT_D_FORM = 3.0    # Å
DEFAULT_D_DISSOC = 4.0  # Å


def place_fragments_generic(
    mol_h: Chem.Mol,
    frag_indices: tuple[tuple[int, ...], ...],
    positions: np.ndarray,
    bond_changes: BondChanges,
    *,
    side: Literal["reactant", "product"],
    d_form: float = DEFAULT_D_FORM,
    d_dissoc: float = DEFAULT_D_DISSOC,
) -> np.ndarray:
    """Position fragments (i >= 1) relative to fragment 0 using bond changes.

    fragment 0 = the heavy-atom-richest fragment. For each non-primary fragment,
    find anchor atom pairs across the boundary that participate in formed
    (reactant side) or broken (product side) bond changes, compute each anchor's
    ideal position in the primary frame, then rigid-transform the fragment to
    minimize anchor-to-ideal RMSD (Kabsch alignment).
    """
    if len(frag_indices) <= 1:
        return positions

    primary_idx = _pick_primary_fragment(mol_h, frag_indices)
    primary = frag_indices[primary_idx]
    primary_set = set(primary)

    relevant = bond_changes.formed if side == "reactant" else bond_changes.broken
    distance = d_form if side == "reactant" else d_dissoc

    out = positions.copy()
    for i, frag in enumerate(frag_indices):
        if i == primary_idx:
            continue
        anchors = _collect_anchors(
            primary_set, set(frag), relevant, out, bond_changes,
            side=side, distance=distance,
        )
        if not anchors:
            log.warning(
                "Fragment %d has no anchor bond change to fragment 0; placing "
                "naively along +x",
                i,
            )
            out = _place_naively(out, primary, frag, distance)
            continue
        out = _apply_kabsch(out, frag, anchors)
    return out


def _pick_primary_fragment(
    mol_h: Chem.Mol, frag_indices: tuple[tuple[int, ...], ...]
) -> int:
    """Largest fragment by heavy-atom count; ties broken by lowest index."""
    sizes = []
    for i, frag in enumerate(frag_indices):
        n_heavy = sum(
            1 for idx in frag if mol_h.GetAtomWithIdx(idx).GetAtomicNum() > 1
        )
        sizes.append((n_heavy, -i, i))  # max heavy, min index
    return max(sizes)[2]


def _collect_anchors(
    primary_set: set[int],
    frag_set: set[int],
    relevant: list[BondChange],
    positions: np.ndarray,
    bond_changes: BondChanges,
    *,
    side: str,
    distance: float,
) -> list[tuple[int, np.ndarray]]:
    """Return [(b_idx, ideal_b_position), ...] for atoms in frag_set."""
    anchors: list[tuple[int, np.ndarray]] = []
    for bc in relevant:
        a, b = _split_bond_across_boundary(bc, primary_set, frag_set)
        if a is None or b is None:
            continue
        ideal_b = _compute_ideal_position(
            a=a, b=b, positions=positions, primary_set=primary_set,
            bond_changes=bond_changes, side=side, distance=distance,
        )
        anchors.append((b, ideal_b))
    return anchors


def _split_bond_across_boundary(
    bc: BondChange, primary_set: set[int], frag_set: set[int]
) -> tuple[int | None, int | None]:
    if bc.a in primary_set and bc.b in frag_set:
        return bc.a, bc.b
    if bc.b in primary_set and bc.a in frag_set:
        return bc.b, bc.a
    return None, None


def _compute_ideal_position(
    *, a: int, b: int, positions: np.ndarray, primary_set: set[int],
    bond_changes: BondChanges, side: str, distance: float,
) -> np.ndarray:
    p_a = positions[a]
    if side == "reactant":
        # Look for a broken bond that *also* uses atom a, leading to a partner
        # c inside the primary fragment. If found, place b on the backside of
        # the (a, c) axis. Otherwise direct b away from primary centroid.
        for broken in bond_changes.broken:
            partner = None
            if broken.a == a and broken.b in primary_set:
                partner = broken.b
            elif broken.b == a and broken.a in primary_set:
                partner = broken.a
            if partner is not None:
                p_c = positions[partner]
                direction = p_a - p_c
                norm = float(np.linalg.norm(direction))
                if norm < 1e-6:
                    break
                return p_a + direction / norm * distance
        return _away_from_centroid(p_a, positions, primary_set, distance)
    # side == "product": preserve embedded direction p_b -> p_a, scale to d_dissoc
    p_b_orig = positions[b]
    direction = p_b_orig - p_a
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        direction = np.array([1.0, 0.0, 0.0])
        norm = 1.0
    return p_a + direction / norm * distance


def _away_from_centroid(
    p_a: np.ndarray, positions: np.ndarray, primary_set: set[int], distance: float
) -> np.ndarray:
    centroid = positions[list(primary_set)].mean(axis=0)
    direction = p_a - centroid
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        direction = np.array([1.0, 0.0, 0.0])
        norm = 1.0
    return p_a + direction / norm * distance


def _apply_kabsch(
    positions: np.ndarray,
    frag: tuple[int, ...],
    anchors: list[tuple[int, np.ndarray]],
) -> np.ndarray:
    if len(anchors) == 1:
        b_idx, ideal = anchors[0]
        delta = ideal - positions[b_idx]
        out = positions.copy()
        for idx in frag:
            out[idx] = positions[idx] + delta
        return out

    src = np.array([positions[b] for b, _ in anchors])
    dst = np.array([ideal for _, ideal in anchors])
    src_c = src.mean(axis=0)
    dst_c = dst.mean(axis=0)
    H = (src - src_c).T @ (dst - dst_c)
    U, _, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T

    out = positions.copy()
    for idx in frag:
        out[idx] = (positions[idx] - src_c) @ R.T + dst_c
    return out


def _place_naively(
    positions: np.ndarray,
    primary: tuple[int, ...],
    frag: tuple[int, ...],
    distance: float,
) -> np.ndarray:
    primary_pos = positions[list(primary)]
    frag_pos = positions[list(frag)]
    primary_c = primary_pos.mean(axis=0)
    frag_c = frag_pos.mean(axis=0)
    primary_radius = float(np.linalg.norm(primary_pos - primary_c, axis=1).max())
    frag_radius = float(np.linalg.norm(frag_pos - frag_c, axis=1).max())
    target = primary_c + np.array([primary_radius + frag_radius + distance, 0.0, 0.0])
    delta = target - frag_c
    out = positions.copy()
    for idx in frag:
        out[idx] = positions[idx] + delta
    return out
```

- [ ] **Step 4: SN2 placement テストを実行**

```bash
pytest tests/test_placement.py::test_sn2_reactant_placement_puts_F_on_backside_of_C -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add reactx/placement.py tests/test_placement.py
git commit -m "feat(placement): add generic bond-change-driven fragment placement"
```

---

## Task 7: place_fragments_generic — 残り 4 反応のテスト

**Files:**
- Modify: `tests/test_placement.py`

実装は Task 6 で完了。各反応で配置が幾何制約を満たすかを確認。

- [ ] **Step 1: 4 反応分の placement テストを追加**

`tests/test_placement.py` に追記:

```python
def test_proton_transfer_reactant_places_NH3_near_HCl_H(
    proton_transfer_rxn_path: Path,
):
    r_mol, p_mol, mapping = parse_rxn(proton_transfer_rxn_path)
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)
    bc = compute_bond_changes(r_h, p_h, mapping)
    frag_indices, positions = _embed_for_test(r_h)

    placed = place_fragments_generic(
        r_h, frag_indices, positions, bc, side="reactant",
    )
    syms = [a.GetSymbol() for a in r_h.GetAtoms()]
    n = syms.index("N")
    # H of HCl (the migrating proton with atom map number 1)
    h_hcl = next(
        a.GetIdx() for a in r_h.GetAtoms()
        if a.GetSymbol() == "H" and a.GetAtomMapNum() == 1
    )

    nh_dist = float(np.linalg.norm(placed[n] - placed[h_hcl]))
    assert 2.0 <= nh_dist <= 4.0, f"|N-H(Cl)| = {nh_dist:.2f} Å, expected ~3.0±1.0"


def test_e2_reactant_places_OH_near_beta_H(e2_rxn_path: Path):
    r_mol, p_mol, mapping = parse_rxn(e2_rxn_path)
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)
    bc = compute_bond_changes(r_h, p_h, mapping)
    frag_indices, positions = _embed_for_test(r_h)

    placed = place_fragments_generic(
        r_h, frag_indices, positions, bc, side="reactant",
    )
    o = next(
        a.GetIdx() for a in r_h.GetAtoms() if a.GetAtomMapNum() == 4
    )
    h_beta = next(
        a.GetIdx() for a in r_h.GetAtoms() if a.GetAtomMapNum() == 6
    )
    oh_dist = float(np.linalg.norm(placed[o] - placed[h_beta]))
    assert 2.0 <= oh_dist <= 4.0, f"|O-Hβ| = {oh_dist:.2f} Å, expected ~3.0±1.0"


def test_sn1_step1_product_separates_Br_from_C(sn1_step1_rxn_path: Path):
    r_mol, p_mol, mapping = parse_rxn(sn1_step1_rxn_path)
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)
    bc = compute_bond_changes(r_h, p_h, mapping)
    frag_indices, positions = _embed_for_test(p_h)

    placed = place_fragments_generic(
        p_h, frag_indices, positions, bc, side="product",
    )
    syms = [a.GetSymbol() for a in p_h.GetAtoms()]
    c_central = next(
        a.GetIdx() for a in p_h.GetAtoms() if a.GetAtomMapNum() == 1
    )
    br = syms.index("Br")
    cbr = float(np.linalg.norm(placed[c_central] - placed[br]))
    assert 3.0 <= cbr <= 5.0, f"|C-Br| in product = {cbr:.2f} Å, expected ~4.0±1.0"


def test_e1_step2_product_separates_H_from_Cb(e1_step2_rxn_path: Path):
    r_mol, p_mol, mapping = parse_rxn(e1_step2_rxn_path)
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)
    bc = compute_bond_changes(r_h, p_h, mapping)
    frag_indices, positions = _embed_for_test(p_h)

    placed = place_fragments_generic(
        p_h, frag_indices, positions, bc, side="product",
    )
    h_leaving = next(
        a.GetIdx() for a in p_h.GetAtoms() if a.GetAtomMapNum() == 5
    )
    cb = next(a.GetIdx() for a in p_h.GetAtoms() if a.GetAtomMapNum() == 2)
    hcb = float(np.linalg.norm(placed[h_leaving] - placed[cb]))
    assert 3.0 <= hcb <= 5.0, f"|H-Cβ| in product = {hcb:.2f} Å, expected ~4.0±1.0"
```

- [ ] **Step 2: テスト実行**

```bash
pytest tests/test_placement.py -v
```

Expected: PASS (5 tests). Failure path: 距離が範囲外なら `_compute_ideal_position` か `_apply_kabsch` のロジックを修正。

- [ ] **Step 3: Commit**

```bash
git add tests/test_placement.py
git commit -m "test(placement): add proton-transfer/E2/dissociation/E1 placement cases"
```

---

## Task 8: embed3d.py を generic placement に切り替え

**Files:**
- Modify: `reactx/embed3d.py`
- Modify: `tests/test_embed3d.py`

- [ ] **Step 1: embed3d.py を書き換え (Phase 0 SN2 hardcode を撤去)**

`reactx/embed3d.py` を以下で全面置換:

```python
"""Convert 2D RDKit Mol to 3D ase.Atoms via RDKit ETKDG + MMFF (+ optional UMA).

Preserves the atom ordering of Chem.AddHs(mol) so that downstream consumers
(align, NEB) can correlate atom indices with heavy_to_hydrogen_groups lookups
on the same AddHs(mol) result.
"""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize import BFGS
from rdkit import Chem
from rdkit.Chem import AllChem

from reactx.placement import place_fragments_generic
from reactx.reaction_topology import BondChanges

log = logging.getLogger(__name__)

MAX_EMBED_RETRIES = 5


def embed_mol_to_atoms(
    mol: Chem.Mol,
    *,
    calculator: Calculator | None = None,
    seed: int = 0xC0FFEE,
    fmax: float = 0.01,
    max_opt_steps: int = 300,
    bond_changes: BondChanges | None = None,
    side: Literal["reactant", "product"] = "reactant",
) -> Atoms:
    """Embed a 2D Mol into 3D and return an ase.Atoms with implicit Hs added.

    Atom order in the returned Atoms matches Chem.AddHs(mol).GetAtoms() exactly.
    For multi-fragment input, fragment placement is delegated to
    placement.place_fragments_generic, driven by bond_changes. If bond_changes
    is None (e.g. unit tests without a paired product), fragments are placed
    naively along +x with a default separation.
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
        bc = bond_changes if bond_changes is not None else BondChanges(
            broken=[], formed=[]
        )
        positions = place_fragments_generic(
            mol_h, frag_indices, positions, bc, side=side,
        )

    symbols = [a.GetSymbol() for a in mol_h.GetAtoms()]
    charges = [a.GetFormalCharge() for a in mol_h.GetAtoms()]
    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.set_initial_charges(charges)

    atoms.info["charge"] = int(sum(charges))
    atoms.info["spin"] = 1  # Phase 1: closed-shell only (UMA omol task)

    if calculator is not None:
        atoms.calc = calculator
        BFGS(atoms, logfile=None).run(fmax=fmax, steps=max_opt_steps)

    return atoms


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

- [ ] **Step 2: 既存 embed3d テストの実行**

```bash
pytest tests/test_embed3d.py -v
```

Expected:
- `test_embed_ch3cl_*` PASS (single fragment)
- `test_embed_multifragment_places_fragments_apart` PASS (no bond_changes path → naive placement, distance > 2.5 Å を満たすか確認)
- `test_embed_multifragment_preserves_addhs_ordering` PASS
- `test_embed_sets_total_charge_and_spin_in_info` PASS
- `test_embed_neutral_molecule_has_zero_charge` PASS
- `test_embed_failure_raises_runtime_error` PASS

`test_embed_multifragment_places_fragments_apart` が落ちた場合: naive 配置の距離計算 (`primary_radius + frag_radius + distance`) を確認し、d_form=3.0 で確実に > 2.5 Å になるよう調整。

- [ ] **Step 3: Commit**

```bash
git add reactx/embed3d.py
git commit -m "refactor(embed3d): replace SN2-hardcoded placement with generic engine"
```

---

## Task 9: cli.py を bond_changes / recommend_n_images に接続

**Files:**
- Modify: `reactx/cli.py`

- [ ] **Step 1: `_cmd_run` を更新して bond_changes と auto images を統合**

`reactx/cli.py` の `_cmd_run` 内、`r_mol, p_mol, mapping = parse_rxn(args.rxn_path)` 直後に bond_changes 計算を挿入し、`embed_mol_to_atoms` 呼び出しに `bond_changes` と `side` を渡す。images 自動計算も加える。

`reactx/cli.py` の `_cmd_run` を以下で置換 (上半分のロジックを差し替え):

```python
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

    r_mol, p_mol, mapping = parse_rxn(args.rxn_path)
    r_mol_h = Chem.AddHs(r_mol)
    p_mol_h = Chem.AddHs(p_mol)
    bond_changes = compute_bond_changes(r_mol_h, p_mol_h, mapping)
    log.info(
        "Bond changes: %d broken, %d formed",
        len(bond_changes.broken), len(bond_changes.formed),
    )

    n_images = args.images if args.images is not None else recommend_n_images(bond_changes)
    log.info("Using n_images=%d", n_images)

    model_kwargs = {"model_name": args.model} if args.backend == "uma" else {}
    calc = make_calculator(args.backend, **model_kwargs)
    reactant = embed_mol_to_atoms(
        r_mol, calculator=calc, seed=1,
        bond_changes=bond_changes, side="reactant",
    )
    product_raw = embed_mol_to_atoms(
        p_mol, calculator=calc, seed=2,
        bond_changes=bond_changes, side="product",
    )

    rH = heavy_to_hydrogen_groups(r_mol_h)
    pH = heavy_to_hydrogen_groups(p_mol_h)
    product = align_product_to_reactant(reactant, product_raw, mapping, rH, pH)

    xyz = args.output / "trajectory.xyz"
    meta = run_neb(
        reactant=reactant,
        product=product,
        calculator=calc,
        n_images=n_images,
        output_xyz=xyz,
        fmax=args.fmax,
        max_steps=args.max_steps,
        pad_frames=args.pad_frames,
    )

    meta_clean = _sanitize_for_json(meta)
    (args.output / "meta.json").write_text(json.dumps(meta_clean, indent=2))
    (args.output / "energies.json").write_text(json.dumps(meta_clean["image_energies"]))

    if args.render:
        rc = _invoke_blender(args, xyz)
        if rc != 0:
            return rc

    log.info(
        "OK: wrote %s (converged=%s, fmax=%s)",
        xyz, meta["converged"], _fmt_fmax(meta["final_fmax"]),
    )
    return 0
```

`build_parser` の `--images` の default を `None` に変更 (auto を意味する):

```python
    run.add_argument("--images", type=int, default=None,
                     help="NEB image count (default: auto from bond changes)")
```

import に `compute_bond_changes` を追加:

```python
from reactx.reaction_topology import BondChanges, compute_bond_changes
```

- [ ] **Step 2: 既存 CLI テストの実行 (regression check)**

```bash
pytest tests/test_cli.py -v
```

Expected: 既存 4 tests とも PASS (LJ backend、SN2 .rxn を使った end-to-end が変わらず動く)。

- [ ] **Step 3: Commit**

```bash
git add reactx/cli.py
git commit -m "feat(cli): wire bond_changes and auto image count into pipeline"
```

---

## Task 10: SN2 既存 slow テストの regression 確認

**Files:**
- (read-only) `tests/test_neb_sn2.py`

UMA を実行できる環境前提。Phase 1 の改修で SN2 が壊れていないことを確認する。

- [ ] **Step 1: SN2 slow テストを実行**

```bash
pytest -m slow tests/test_neb_sn2.py -v
```

Expected: PASS (TS の F-C-Cl 角度 > 120° を維持)。

実行時間: 数分〜10 分程度 (UMA 推論)。GPU 不可なら CPU で更に長い。

- [ ] **Step 2: 失敗時の対処**

仮に angle が緩んだり non-convergence になった場合:

a) `placement.py` の `_compute_ideal_position` で reactant 側 SN2 が Phase 0 と同じ backside attack を返しているか確認 (Task 6 の test で既に保証されているはずだが embed3d 経由で経路が変わった可能性をチェック)。
b) `embed3d.embed_mol_to_atoms` の `bond_changes` が NULL でないことを確認 (CLI 経由なら必ず渡される)。
c) `align_product_to_reactant` 後の Walden 反転構造 (R, P で C 周りの H が反転) が保たれているか確認。

問題なければそのまま次タスクへ。

- [ ] **Step 3 (regression が出た場合のみ): bug fix + commit**

修正内容を `fix(...): ...` でコミット。

---

## Task 11: SN1 解離反応の slow 統合テスト

**Files:**
- Create: `tests/test_neb_dissociation.py`

- [ ] **Step 1: テストファイルを作成**

`tests/test_neb_dissociation.py`:

```python
"""SN1 step 1 (heterolytic dissociation) end-to-end NEB test."""
from pathlib import Path

import numpy as np
import pytest
from ase.io import read
from rdkit import Chem

from reactx.align import align_product_to_reactant
from reactx.calculators import make_calculator
from reactx.embed3d import embed_mol_to_atoms
from reactx.neb import run_neb
from reactx.reaction_topology import compute_bond_changes
from reactx.rxn_parser import heavy_to_hydrogen_groups, parse_rxn


@pytest.mark.slow
def test_dissociation_neb_separates_C_and_Br(tmp_path: Path, sn1_step1_rxn_path: Path):
    pytest.importorskip("fairchem.core")

    r_mol, p_mol, mapping = parse_rxn(sn1_step1_rxn_path)
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)
    bc = compute_bond_changes(r_h, p_h, mapping)

    calc = make_calculator("uma")
    reactant = embed_mol_to_atoms(
        r_mol, calculator=calc, seed=1, bond_changes=bc, side="reactant",
    )
    product_raw = embed_mol_to_atoms(
        p_mol, calculator=calc, seed=2, bond_changes=bc, side="product",
    )
    product = align_product_to_reactant(
        reactant, product_raw,
        mapping, heavy_to_hydrogen_groups(r_h), heavy_to_hydrogen_groups(p_h),
    )

    out = tmp_path / "traj.xyz"
    meta = run_neb(
        reactant=reactant, product=product, calculator=calc,
        n_images=11, output_xyz=out, fmax=0.05, max_steps=200,
        pad_frames=0,
    )
    frames = read(str(out), index=":")
    syms = frames[0].get_chemical_symbols()
    c = next(
        a.GetIdx() for a in r_h.GetAtoms() if a.GetAtomMapNum() == 1
    )
    br = syms.index("Br")
    d_start = frames[0].get_distance(c, br)
    d_end = frames[-1].get_distance(c, br)
    # Reactant: covalent C-Br (~1.95 Å). Product: separated (>3.0 Å).
    assert d_start < 2.3, f"reactant C-Br too long: {d_start:.2f}"
    assert d_end > 3.0, f"product C-Br too short: {d_end:.2f}"
    energies = np.array(meta["image_energies"])
    # Energy should rise monotonically OR rise to a peak then plateau.
    assert energies[-1] > energies[0] - 0.5  # eV: dissociation is endothermic-ish
```

- [ ] **Step 2: 実行**

```bash
pytest -m slow tests/test_neb_dissociation.py -v
```

Expected: PASS (UMA 必須)。

- [ ] **Step 3: Commit**

```bash
git add tests/test_neb_dissociation.py
git commit -m "test(neb): add SN1 step1 dissociation integration test (slow)"
```

---

## Task 12: E2 elimination の slow 統合テスト

**Files:**
- Create: `tests/test_neb_e2.py`

- [ ] **Step 1: テストファイルを作成**

`tests/test_neb_e2.py`:

```python
"""E2 elimination end-to-end NEB test."""
from pathlib import Path

import numpy as np
import pytest
from ase.io import read
from rdkit import Chem

from reactx.align import align_product_to_reactant
from reactx.calculators import make_calculator
from reactx.cli import recommend_n_images
from reactx.embed3d import embed_mol_to_atoms
from reactx.neb import run_neb
from reactx.reaction_topology import compute_bond_changes
from reactx.rxn_parser import heavy_to_hydrogen_groups, parse_rxn


@pytest.mark.slow
def test_e2_neb_breaks_CBr_and_CbetaH_in_concert(tmp_path: Path, e2_rxn_path: Path):
    pytest.importorskip("fairchem.core")

    r_mol, p_mol, mapping = parse_rxn(e2_rxn_path)
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)
    bc = compute_bond_changes(r_h, p_h, mapping)

    calc = make_calculator("uma")
    reactant = embed_mol_to_atoms(
        r_mol, calculator=calc, seed=1, bond_changes=bc, side="reactant",
    )
    product_raw = embed_mol_to_atoms(
        p_mol, calculator=calc, seed=2, bond_changes=bc, side="product",
    )
    product = align_product_to_reactant(
        reactant, product_raw,
        mapping, heavy_to_hydrogen_groups(r_h), heavy_to_hydrogen_groups(p_h),
    )

    out = tmp_path / "traj.xyz"
    meta = run_neb(
        reactant=reactant, product=product, calculator=calc,
        n_images=recommend_n_images(bc),
        output_xyz=out, fmax=0.05, max_steps=300, pad_frames=0,
    )
    frames = read(str(out), index=":")
    energies = np.array(meta["image_energies"])
    ts_idx = int(np.argmax(energies))
    assert 0 < ts_idx < len(frames) - 1

    syms = frames[0].get_chemical_symbols()
    c_alpha = next(a.GetIdx() for a in r_h.GetAtoms() if a.GetAtomMapNum() == 1)
    c_beta = next(a.GetIdx() for a in r_h.GetAtoms() if a.GetAtomMapNum() == 2)
    br = syms.index("Br")
    h_beta = next(a.GetIdx() for a in r_h.GetAtoms() if a.GetAtomMapNum() == 6)

    ts = frames[ts_idx]
    cbr_ts = ts.get_distance(c_alpha, br)
    cbh_ts = ts.get_distance(c_beta, h_beta)
    cbr_r = frames[0].get_distance(c_alpha, br)
    cbh_r = frames[0].get_distance(c_beta, h_beta)
    # Both bonds should be elongated at TS (concerted breaking)
    assert cbr_ts > cbr_r * 1.15, f"C-Br at TS not elongated: {cbr_ts:.2f} vs {cbr_r:.2f}"
    assert cbh_ts > cbh_r * 1.15, f"Cβ-H at TS not elongated: {cbh_ts:.2f} vs {cbh_r:.2f}"
```

- [ ] **Step 2: 実行**

```bash
pytest -m slow tests/test_neb_e2.py -v
```

Expected: PASS。NEB 収束しない場合は `n_images` を増やすか `--max-steps 500` 相当のループに延長。

- [ ] **Step 3: Commit**

```bash
git add tests/test_neb_e2.py
git commit -m "test(neb): add E2 elimination integration test (slow)"
```

---

## Task 13: E1 step2 の slow 統合テスト

**Files:**
- Create: `tests/test_neb_e1_step2.py`

- [ ] **Step 1: テストファイルを作成**

`tests/test_neb_e1_step2.py`:

```python
"""E1 step 2 (β-H elimination from carbocation) end-to-end NEB test."""
from pathlib import Path

import numpy as np
import pytest
from ase.io import read
from rdkit import Chem

from reactx.align import align_product_to_reactant
from reactx.calculators import make_calculator
from reactx.embed3d import embed_mol_to_atoms
from reactx.neb import run_neb
from reactx.reaction_topology import compute_bond_changes
from reactx.rxn_parser import heavy_to_hydrogen_groups, parse_rxn


@pytest.mark.slow
def test_e1_step2_neb_extracts_beta_H(tmp_path: Path, e1_step2_rxn_path: Path):
    pytest.importorskip("fairchem.core")

    r_mol, p_mol, mapping = parse_rxn(e1_step2_rxn_path)
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)
    bc = compute_bond_changes(r_h, p_h, mapping)

    calc = make_calculator("uma")
    reactant = embed_mol_to_atoms(
        r_mol, calculator=calc, seed=1, bond_changes=bc, side="reactant",
    )
    product_raw = embed_mol_to_atoms(
        p_mol, calculator=calc, seed=2, bond_changes=bc, side="product",
    )
    product = align_product_to_reactant(
        reactant, product_raw,
        mapping, heavy_to_hydrogen_groups(r_h), heavy_to_hydrogen_groups(p_h),
    )

    out = tmp_path / "traj.xyz"
    run_neb(
        reactant=reactant, product=product, calculator=calc,
        n_images=13, output_xyz=out, fmax=0.05, max_steps=200, pad_frames=0,
    )
    frames = read(str(out), index=":")
    syms = frames[0].get_chemical_symbols()
    c_alpha = next(a.GetIdx() for a in r_h.GetAtoms() if a.GetAtomMapNum() == 1)
    c_beta = next(a.GetIdx() for a in r_h.GetAtoms() if a.GetAtomMapNum() == 2)
    h_beta = next(a.GetIdx() for a in r_h.GetAtoms() if a.GetAtomMapNum() == 5)

    cab_r = frames[0].get_distance(c_alpha, c_beta)
    cab_p = frames[-1].get_distance(c_alpha, c_beta)
    cbh_r = frames[0].get_distance(c_beta, h_beta)
    cbh_p = frames[-1].get_distance(c_beta, h_beta)
    # Cα-Cβ shortens (single -> double), Cβ-H lengthens (bond breaks)
    assert cab_p < cab_r, f"Cα-Cβ should shorten: {cab_r:.2f} -> {cab_p:.2f}"
    assert cbh_p > cbh_r * 1.5, f"Cβ-H should break: {cbh_r:.2f} -> {cbh_p:.2f}"
```

- [ ] **Step 2: 実行**

```bash
pytest -m slow tests/test_neb_e1_step2.py -v
```

Expected: PASS。

- [ ] **Step 3: Commit**

```bash
git add tests/test_neb_e1_step2.py
git commit -m "test(neb): add E1 step2 (beta-H elimination) integration test (slow)"
```

---

## Task 14: Proton transfer の slow 統合テスト

**Files:**
- Create: `tests/test_neb_proton_transfer.py`

- [ ] **Step 1: テストファイルを作成**

`tests/test_neb_proton_transfer.py`:

```python
"""Proton transfer end-to-end NEB test (HCl + NH3 -> Cl- + NH4+)."""
from pathlib import Path

import numpy as np
import pytest
from ase.io import read
from rdkit import Chem

from reactx.align import align_product_to_reactant
from reactx.calculators import make_calculator
from reactx.embed3d import embed_mol_to_atoms
from reactx.neb import run_neb
from reactx.reaction_topology import compute_bond_changes
from reactx.rxn_parser import heavy_to_hydrogen_groups, parse_rxn


@pytest.mark.slow
def test_proton_transfer_neb_H_moves_from_Cl_to_N(
    tmp_path: Path, proton_transfer_rxn_path: Path,
):
    pytest.importorskip("fairchem.core")

    r_mol, p_mol, mapping = parse_rxn(proton_transfer_rxn_path)
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)
    bc = compute_bond_changes(r_h, p_h, mapping)

    calc = make_calculator("uma")
    reactant = embed_mol_to_atoms(
        r_mol, calculator=calc, seed=1, bond_changes=bc, side="reactant",
    )
    product_raw = embed_mol_to_atoms(
        p_mol, calculator=calc, seed=2, bond_changes=bc, side="product",
    )
    product = align_product_to_reactant(
        reactant, product_raw,
        mapping, heavy_to_hydrogen_groups(r_h), heavy_to_hydrogen_groups(p_h),
    )

    out = tmp_path / "traj.xyz"
    meta = run_neb(
        reactant=reactant, product=product, calculator=calc,
        n_images=13, output_xyz=out, fmax=0.05, max_steps=200, pad_frames=0,
    )
    frames = read(str(out), index=":")
    syms = frames[0].get_chemical_symbols()
    cl = syms.index("Cl")
    n = syms.index("N")
    # migrating H is the one mapped with atom map 1 (originally on Cl)
    h_mig = next(a.GetIdx() for a in r_h.GetAtoms() if a.GetAtomMapNum() == 1)

    h_to_cl_r = frames[0].get_distance(h_mig, cl)
    h_to_n_r = frames[0].get_distance(h_mig, n)
    h_to_cl_p = frames[-1].get_distance(h_mig, cl)
    h_to_n_p = frames[-1].get_distance(h_mig, n)
    # Reactant: H bonded to Cl, far from N. Product: H bonded to N, far from Cl.
    assert h_to_cl_r < 1.5
    assert h_to_n_p < 1.3
    assert h_to_cl_p > h_to_cl_r
    assert h_to_n_p < h_to_n_r

    energies = np.array(meta["image_energies"])
    ts_idx = int(np.argmax(energies))
    assert 0 < ts_idx < len(frames) - 1
```

- [ ] **Step 2: 実行**

```bash
pytest -m slow tests/test_neb_proton_transfer.py -v
```

Expected: PASS。

- [ ] **Step 3: Commit**

```bash
git add tests/test_neb_proton_transfer.py
git commit -m "test(neb): add proton transfer integration test (slow)"
```

---

## Task 15: Blender smoke テストの 5 反応拡張

**Files:**
- Modify: `tests/test_blender_smoke.py`

- [ ] **Step 1: 既存 test_blender_smoke.py を確認**

```bash
cat tests/test_blender_smoke.py
```

その上で、5 反応を `pytest.mark.parametrize` で回せるよう書き換える。

- [ ] **Step 2: 5 反応を parametrize**

`tests/test_blender_smoke.py` の reaction-loop 部分を parametrize に置き換える (既存テストの構造に従い、`@pytest.mark.blender` マーカーは保持)。具体的には:

```python
import pytest

@pytest.mark.blender
@pytest.mark.parametrize("rxn_name", [
    "sn2", "sn1_step1", "e2", "e1_step2", "proton_transfer",
])
def test_blender_render_produces_blend(tmp_path, examples_dir, rxn_name):
    """Smoke test: each Phase 1 reaction renders to a .blend file."""
    import shutil
    import subprocess

    if shutil.which("blender") is None:
        pytest.skip("blender not on PATH")

    # First, generate trajectory.xyz with LJ backend (fast, no UMA needed for
    # smoke; we are testing the Blender pipeline only, not chemistry accuracy).
    from reactx import cli
    out = tmp_path / rxn_name
    rc = cli.main([
        "run", str(examples_dir / f"{rxn_name}.rxn"),
        "-o", str(out),
        "--backend", "lj", "--images", "5", "--fmax", "0.5", "--max-steps", "10",
    ])
    assert rc == 0
    xyz = out / "trajectory.xyz"
    assert xyz.exists()

    blend = out / "scene.blend"
    script = (tmp_path.parent / "blender" / "render.py").resolve()
    # locate render.py relative to repo root
    repo_root = examples_dir.parent
    script = repo_root / "blender" / "render.py"
    rc = subprocess.run(
        ["blender", "--background", "--python", str(script),
         "--", str(xyz), str(blend)],
        check=False,
    ).returncode
    assert rc == 0, f"blender render failed for {rxn_name}"
    assert blend.exists()
```

(注: 既存 test の構造を残したい場合は、既存関数を `_render_one(name)` ヘルパに切り出して 5 回呼び出す形でもよい。`pytest.mark.blender` が CI ではスキップされる前提を維持。)

- [ ] **Step 3: ローカル Blender で実行 (環境依存)**

```bash
pytest -m blender tests/test_blender_smoke.py -v
```

Expected: 5 件すべて PASS、各々で `out/<name>/scene.blend` が生成。

Blender が無い環境では skip される (`shutil.which("blender") is None` 経由)。

- [ ] **Step 4: Commit**

```bash
git add tests/test_blender_smoke.py
git commit -m "test(blender): parametrize smoke test over 5 Phase 1 reactions"
```

---

## Task 16: README 更新

**Files:**
- Modify: `README.md`

- [ ] **Step 1: README に Phase 1 セクションを追加**

`README.md` の「Phase 0 動作確認」セクションの後ろに、Phase 1 用のセクションを挿入する。具体的な追加内容:

```markdown
## Phase 1: 多反応対応

Phase 1 では `.rxn` の atom mapping から **bond change (broken / formed)** を抽出し、それに応じた幾何配置で fragment を初期化する generic engine を導入した。これにより SN2 以外の極性二分子反応 (および解離型 elementary step) を反応分類なしで処理できる。

### 対応反応

| 反応 | example |
|---|---|
| SN2 | `examples/sn2.rxn` |
| Heterolytic dissociation (SN1/E1 step 1) | `examples/sn1_step1.rxn` |
| E2 elimination | `examples/e2.rxn` |
| β-H elimination from cation (E1 step 2) | `examples/e1_step2.rxn` |
| Proton transfer | `examples/proton_transfer.rxn` |

### 使用方法

```bash
reactx run examples/proton_transfer.rxn -o out/proton/ --backend uma --render
reactx run examples/e2.rxn -o out/e2/ --backend uma --render
# ... 他の反応も同様
```

`--images` を省略すると bond change の数から自動計算される (`max(11, 9 + 2 * n_broken + 2 * n_formed)`)。

### 慣習: multi-step 反応の表現

NEB は 1 つの elementary step (= 1 saddle point) を扱う前提のため、SN1 や E1 のような multi-step 反応はステップごとに別 `.rxn` ファイルとして表現する。例えば SN1 全体は `sn1_step1.rxn` (heterolytic dissociation) と既存 `sn2.rxn` 相当の置換ステップ (Phase 1 では未提供) の 2 ファイルで表現される。

### Phase 1 の制約

- 反応タイプの自動分類は行わない (`.rxn` の bond change のみで配置を決める)。
- ラジカル / open-shell 反応は対象外 (UMA omol task は closed-shell 前提)。
- aromatic 系の bond change は `NotImplementedError` で停止する。
```

- [ ] **Step 2: 既存 README の「Phase 0 の既知の制約」セクションは保持**

(Phase 0 の制約は Phase 1 でも一部継続するため変更しない)

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(readme): add Phase 1 multi-reaction usage and conventions"
```

---

## Task 17: 高速ユニットテスト全件 + ruff lint

**Files:** none (検査のみ)

- [ ] **Step 1: 高速ユニット全件**

```bash
pytest
```

Expected: 全 PASS (slow / blender マーカーは default で除外)。

- [ ] **Step 2: ruff check**

```bash
ruff check reactx/ tests/
ruff format --check reactx/ tests/
```

Expected: 0 issues、format 差分なし。

警告/差分があれば `ruff format reactx/ tests/` で自動修正し、ruff check の指摘は手動修正。

- [ ] **Step 3: 修正があれば commit**

```bash
git add -u
git commit -m "style: ruff lint/format pass"
```

---

## Task 18: 5 反応の Blender 視認 (DoD §2 手動検証)

**Files:** none (手動検証)

UMA + Blender ローカル環境前提。

- [ ] **Step 1: 5 反応の trajectory.xyz + scene.blend を生成**

```bash
for r in sn2 sn1_step1 e2 e1_step2 proton_transfer; do
  reactx run examples/$r.rxn -o out/$r/ --backend uma --render
done
```

Expected: 全反応で `out/<r>/trajectory.xyz`, `out/<r>/scene.blend`, `out/<r>/meta.json` が生成、`meta.json` の `converged: true`。

非収束のものがあれば、`--max-steps 800` で再実行 / `--images` を増やして再実行。

- [ ] **Step 2: 各 .blend を Blender GUI で開いて視認**

```
sn2:             F⁻ が CH₃Cl の背面から接近 → C 中心の sp³ 反転 → Cl⁻ が脱離 (Walden inversion)
sn1_step1:       (CH₃)₃C-Br の C-Br が伸長して Br⁻ が解離する
e2:              CH₃CH₂Br + OH⁻ で Cα-Br と Cβ-H が同時に伸び、Cα=Cβ が短く、O-H が形成
e1_step2:        (CH₃)₃C+ の β-H が抜け、Cα=Cβ が短くなる、H+ が離脱
proton_transfer: HCl の H が NH₃ の N に移動 (中間で H が両者の中点付近)
```

- [ ] **Step 3: 結果を記録**

`docs/superpowers/plans/2026-04-26-reactx-phase-1-implementation.md` の本タスク末尾に、視認結果と気づき (収束イテレーション数、QC 補足など) を追記してコミット。

```bash
git add docs/superpowers/plans/2026-04-26-reactx-phase-1-implementation.md
git commit -m "docs(plan): record Phase 1 manual Blender verification results"
```

---

## Task 19: PR 作成と Phase 1 終了

**Files:** none

- [ ] **Step 1: branch を origin に push**

```bash
git push -u origin phase-1
```

- [ ] **Step 2: gh pr create で PR を作成**

```bash
gh pr create --title "Phase 1: Generic bond-change geometry engine for multi-reaction support" \
  --body "$(cat <<'EOF'
## Summary
- Phase 0 で SN2 にハードコードした embedding ロジックを generic bond-change geometry engine に置換 (`reactx/reaction_topology.py`, `reactx/placement.py` を新規追加)。
- 5 反応 (SN2 / 解離 / E2 / E1 step2 / proton transfer) の examples とそれぞれの slow 統合テストを追加。
- CLI の `--images` を bond change 数から自動計算 (override 可)。
- README に Phase 1 の使い方と慣習を追記。

## Spec / Plan
- spec: `docs/superpowers/specs/2026-04-26-reactx-phase-1-design.md`
- plan: `docs/superpowers/plans/2026-04-26-reactx-phase-1-implementation.md`

## Test plan
- [ ] `pytest` (高速ユニット) が全件 PASS
- [ ] `pytest -m slow` で 5 反応の NEB テスト全件 PASS (UMA 必須)
- [ ] `pytest -m blender` で 5 反応の Blender smoke テスト全件 PASS (Blender 必須)
- [ ] 5 反応すべてで `reactx run --render` 後に `.blend` を Blender GUI で開いて反応の本質的な動きが視認できる

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR URL が出力される。

---

## Self-Review

**Spec coverage:**

- spec §1 (目的・Approach 2): ✓ Task 6, 8 (placement.py, embed3d 改修)
- spec §2 (対象 5 反応): ✓ Task 1 (.rxn 例)、Task 11–14 (slow 統合テスト)
- spec §3.1.1 (`reaction_topology.py`): ✓ Task 2, 3, 4
- spec §3.1.2 (`placement.py`): ✓ Task 6, 7
- spec §3.2.1 (`embed3d.py` 改修): ✓ Task 8
- spec §3.2.2 (`cli.py` 改修): ✓ Task 5 (recommend_n_images), 9 (wire-up)
- spec §3.2.3 (neb.py 変更なし): ✓ 該当タスクなし (touchしない)
- spec §4 (placement アルゴリズム): ✓ Task 6 で実装、Task 7 で検証
- spec §5 (NEB images auto): ✓ Task 5
- spec §6.1 (高速ユニットテスト): ✓ Task 4 (reaction_topology), 7 (placement), 5 (recommend_n_images)
- spec §6.2 (slow 統合テスト): ✓ Task 11–14, Task 10 (regression)
- spec §6.3 (Blender smoke): ✓ Task 15
- spec §7 (Examples): ✓ Task 1
- spec §8 (README): ✓ Task 16
- spec §9 DoD §1–§5: ✓ Task 18 (manual), Task 17 (lint), Task 11–14 (slow tests), Task 10 (regression)
- spec §10 非スコープ: 該当タスクなし (意図通り)
- spec §11 リスク: §11 のうち「fragment 衝突 → 1 回離す再配置」を Task 6 の placement に含めて **いない**。リスクとしては存在するが、Task 6 の `_compute_ideal_position` で d_form/d_dissoc を default 3.0/4.0 に設定しているためほぼ安全 (1.5 Å 未満になる現実的経路がない)。**重要**: もし Task 11–14 で fragment 衝突由来の SCF 不収束が出たら、Task 6 の placement に「最近接重原子距離チェック → 不足なら +d_form 再配置」のループを追加する fix-it タスクを別途立てる (現プランでは予防的に追加せず、不要であれば YAGNI)。
- spec §12 (Phase 2 ロードマップ): 該当タスクなし (memo)。

**Placeholder scan:**
- 「TBD」「TODO」「実装は後で」「適切なエラー処理を追加」等の文言は入っていない。各 step は実コードまたは具体的コマンドを含む。
- ただし Task 18 の手動検証部分は人間判断 (視認結果の記述) を含むため、テンプレートだけ用意しユーザに合わせて記録してもらう。これは placeholder ではなく仕様。

**Type consistency:**
- `BondChange` / `BondChanges` / `compute_bond_changes` / `place_fragments_generic` / `recommend_n_images` の型シグネチャは Task 2, 3, 5, 6 で確定し、後続タスクでもそのまま使われる。
- `embed_mol_to_atoms` の追加引数 `bond_changes: BondChanges | None = None`, `side: Literal["reactant", "product"] = "reactant"` は Task 8 で定義し、Task 9 (cli) と Task 11–14 (slow tests) で同じ名前・順序で渡している。
- `recommend_n_images` の引数は `BondChanges`、戻り値 `int` で一貫。
