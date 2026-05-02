# Phase 3 Multi-Bond Elementary Step Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Phase Re1 の 1 formed + 1 broken 制限を解除し、E2 (1 formed + 2 broken) と SN1 step 1 (0 formed + 1 broken) を end-to-end で動かせる multi-bond elementary step エンジンを実装する。

**Architecture:** `reactx/bond_changes.py` の `SimpleBondChanges` を multi-bond 対応の `BondChanges` に置換し、`reactx/embed3d.py` のフラグメント配置を 2 階層 dispatcher (Tier 1 directional / Tier 2 centroid stub) に refactor。CLI は `formed_pairs` / `r_form_targets: list[float]` API に統一し、unimolecular 反応で `--n-angles` を 1 に auto-clamp、`--neb-refine` は 1+1 反応のみ許可。新規 preset (`e2`, `sn1_dissoc`) と例 .rxn + slow integration test を追加。

**Tech Stack:** Python 3.11, RDKit, ASE, FAIRChem UMA, pytest, FIRE optimizer.

**Spec:** `docs/superpowers/specs/2026-05-03-phase-3-multibond-design.md`

---

## ファイル構造

**Modify**:
- `reactx/bond_changes.py` — `SimpleBondChanges` 削除、`BondChanges` 新設、`compute_bond_changes` rename、`__post_init__` validation
- `reactx/embed3d.py` — `_place_nucleophile_backside` を `_place_fragments` (dispatcher) + `_find_substrate_fragment` + `_directional_placement` に分解
- `reactx/presets.py` — `e2`, `sn1_dissoc` 追加
- `reactx/cli.py` — `formed_pair` → `bond_changes.formed`/`broken` を直接使用、`r_form` scalar → `r_form_targets: list[float]`、`--reaction-type` choices 拡張、unimolecular auto-clamp、`--neb-refine` ガード
- `reactx/prescreen.py` — `SimpleBondChanges` import 削除のみ (使ってないので import 行を消すだけ)
- `tests/test_bond_changes.py` — `BondChanges` 名へ移行、E2/SN1step1 形状の新規テスト追加、`test_e2_like_raises_not_implemented` を削除し新しい validation テストに置換
- `tests/test_artificial_force.py` — multi-bond 線形性回帰テストを 1 件追加
- `tests/test_scoring.py` — 多 formed (`r_form_targets=[a, b]`) per-bond 判定テストを 1 件追加
- `tests/test_re1_sn2.py` / `tests/test_re1_proton_transfer.py` / `tests/test_re1_menshutkin.py` — `BondChanges` rename + `r_form_targets` list assertion に追従
- `tests/test_neb_refine_sn2.py` / `tests/test_blender_smoke.py` — 同 rename + e2 parametrize 追加
- `tests/test_prescreen.py` / `tests/test_prescreen_disabled_sn2.py` — 同 rename
- `tests/test_embed3d.py` — `SimpleBondChanges` 利用箇所を `BondChanges` に
- `README.md` — preset 表に `e2` / `sn1_dissoc` 追加、Phase 3 DoD 手順、適用限界の追記

**Create**:
- `examples/e2.rxn` — CH₃CH₂Cl + OH⁻ → CH₂=CH₂ + Cl⁻ + H₂O の V2000 .rxn
- `examples/sn1_dissoc.rxn` — (CH₃)₃CBr → (CH₃)₃C⁺ + Br⁻ の V2000 .rxn
- `tests/test_embed3d_placement.py` — dispatcher の挙動 (directional 適用 / NotImplementedError 出口) 7 ケース
- `tests/test_re3_e2.py` — E2 end-to-end (slow, UMA)
- `tests/test_re3_sn1_dissoc.py` — SN1 dissoc end-to-end (slow, UMA)
- `tests/test_cli_unimolecular.py` — unimolecular auto-clamp の挙動
- `tests/test_cli_neb_refine_guard.py` — `--neb-refine` の 1+1 制限

---

## Task 1: BondChanges multi-bond refactor

**Files:**
- Modify: `reactx/bond_changes.py`
- Modify: `tests/test_bond_changes.py`
- Modify: `reactx/embed3d.py` (consumer migration: 単 tuple アクセス → `.formed[0]` / `.broken[0]` で旧挙動維持)
- Modify: `reactx/cli.py` (consumer migration: `bond_changes.formed` → `bond_changes.formed[0]` で旧挙動維持; full multi-bond 化は Task 4 で)
- Modify: `reactx/prescreen.py` (import 削除のみ — 実体未使用)
- Modify: `tests/test_re1_sn2.py`, `tests/test_re1_proton_transfer.py`, `tests/test_re1_menshutkin.py`, `tests/test_neb_refine_sn2.py`, `tests/test_prescreen.py`, `tests/test_prescreen_disabled_sn2.py`, `tests/test_embed3d.py` (import rename)

**意図**: `SimpleBondChanges` を完全に削除し、`BondChanges` (multi-bond) で置き換える。この task では embed3d / cli は **まだ multi-bond には対応せず**、既存の 1+1 挙動を保ったまま新 API に追従するだけ。multi-bond 配置は Task 2、CLI 全面化は Task 4 で。

- [ ] **Step 1: 失敗するテストを書く (`tests/test_bond_changes.py` を全面書き換え)**

```python
"""Unit tests for reactx.bond_changes."""
import pytest
from rdkit import Chem

from reactx.bond_changes import BondChanges, compute_bond_changes
from reactx.rxn_parser import parse_rxn


def _atom_index_by_symbol(mol_h: Chem.Mol, sym: str) -> int:
    for atom in mol_h.GetAtoms():
        if atom.GetSymbol() == sym:
            return atom.GetIdx()
    raise AssertionError(f"no {sym} atom in mol")


def test_sn2_bond_changes_returns_singleton_lists():
    r_mol, p_mol, mapping = parse_rxn("examples/sn2.rxn")
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)

    bc = compute_bond_changes(r_h, p_h, mapping)
    assert isinstance(bc, BondChanges)
    assert len(bc.formed) == 1
    assert len(bc.broken) == 1

    c_idx = _atom_index_by_symbol(r_h, "C")
    cl_idx = _atom_index_by_symbol(r_h, "Cl")
    o_idx = _atom_index_by_symbol(r_h, "O")
    assert set(bc.formed[0]) == {c_idx, o_idx}
    assert set(bc.broken[0]) == {c_idx, cl_idx}


def test_proton_transfer_bond_changes():
    r_mol, p_mol, mapping = parse_rxn("examples/proton_transfer.rxn")
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)

    bc = compute_bond_changes(r_h, p_h, mapping)
    assert len(bc.formed) == 1
    assert len(bc.broken) == 1

    n_idx = _atom_index_by_symbol(r_h, "N")
    cl_idx = _atom_index_by_symbol(r_h, "Cl")
    proton_idx = next(a.GetIdx() for a in r_h.GetAtoms() if a.GetAtomMapNum() == 1)

    assert set(bc.formed[0]) == {n_idx, proton_idx}
    assert set(bc.broken[0]) == {proton_idx, cl_idx}


def test_bond_changes_rejects_self_loop():
    with pytest.raises(ValueError, match="self-loop"):
        BondChanges(formed=((0, 0),), broken=((1, 2),))


def test_bond_changes_rejects_duplicate():
    with pytest.raises(ValueError, match="duplicate"):
        BondChanges(formed=((0, 1), (1, 0)), broken=())


def test_bond_changes_rejects_empty_total():
    with pytest.raises(ValueError, match="at least one"):
        BondChanges(formed=(), broken=())


def test_bond_changes_accepts_multi_bond_topologies():
    # E2 形状: 1 formed + 2 broken
    bc = BondChanges(formed=((4, 5),), broken=((0, 2), (1, 4)))
    assert len(bc.formed) == 1
    assert len(bc.broken) == 2

    # SN1 step 1 形状: 0 formed + 1 broken
    bc = BondChanges(formed=(), broken=((0, 1),))
    assert len(bc.formed) == 0
    assert len(bc.broken) == 1
```

- [ ] **Step 2: テストを実行して FAIL を確認**

Run: `pytest tests/test_bond_changes.py -v`
Expected: ImportError (`BondChanges` / `compute_bond_changes` 未定義)

- [ ] **Step 3: `reactx/bond_changes.py` を書き換える**

```python
"""Compute formed/broken bonds for elementary steps (Phase 3: multi-bond).

σ-only connectivity diff. Bond order changes (single↔double) are NOT
detected; π formation is left to the QM calculator. See the Phase 3 spec
section "σ-only connectivity diff".
"""
from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem


@dataclass(frozen=True)
class BondChanges:
    """Multi-bond elementary step: lists of formed and broken bonds.

    Atom indices are in the reactant_mol_h (Chem.AddHs(reactant_mol)) ordering.
    Tuples (not lists) for frozen-dataclass hashability.
    """

    formed: tuple[tuple[int, int], ...]
    broken: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        for label, bonds in (("formed", self.formed), ("broken", self.broken)):
            seen: set[tuple[int, int]] = set()
            for a, b in bonds:
                if a == b:
                    raise ValueError(f"{label} bond {(a, b)} is a self-loop")
                key = (a, b) if a <= b else (b, a)
                if key in seen:
                    raise ValueError(f"{label} contains duplicate bond {key}")
                seen.add(key)
        if len(self.formed) + len(self.broken) == 0:
            raise ValueError("BondChanges must have at least one formed or broken bond")


def compute_bond_changes(
    reactant_mol_h: Chem.Mol,
    product_mol_h: Chem.Mol,
    heavy_mapping: dict[int, int],
) -> BondChanges:
    """Diff bonds between reactant and product mol_h. σ-only connectivity.

    Atom indices in the result use reactant_mol_h's ordering.
    """
    if reactant_mol_h.GetNumAtoms() != product_mol_h.GetNumAtoms():
        raise ValueError(
            f"reactant and product mol_h must have same atom count: "
            f"{reactant_mol_h.GetNumAtoms()} vs {product_mol_h.GetNumAtoms()}"
        )

    full_mapping = _build_full_atom_mapping(reactant_mol_h, product_mol_h, heavy_mapping)
    inv_mapping = {p: r for r, p in full_mapping.items()}

    r_bonds = _bond_set_in_self_idx(reactant_mol_h)
    p_bonds_in_r_space = {
        _ordered(inv_mapping[a], inv_mapping[b]) for a, b in _bond_set_in_self_idx(product_mol_h)
    }

    formed = tuple(sorted(p_bonds_in_r_space - r_bonds))
    broken = tuple(sorted(r_bonds - p_bonds_in_r_space))
    return BondChanges(formed=formed, broken=broken)


def _bond_set_in_self_idx(mol: Chem.Mol) -> set[tuple[int, int]]:
    return {_ordered(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()}


def _ordered(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _build_full_atom_mapping(
    reactant_mol_h: Chem.Mol,
    product_mol_h: Chem.Mol,
    heavy_mapping: dict[int, int],
) -> dict[int, int]:
    """Extend heavy_mapping with implicit-H pairings.

    heavy_mapping covers all atoms with explicit atom map numbers (heavy + any
    explicit-mapped H). Remaining unmapped Hs are paired by their bonded heavy
    atom group: reactant Hs of heavy_r <-> product Hs of heavy_p where
    heavy_r -> heavy_p in heavy_mapping.
    """
    full: dict[int, int] = dict(heavy_mapping)
    used_p: set[int] = set(full.values())

    for r_heavy, p_heavy in heavy_mapping.items():
        r_atom = reactant_mol_h.GetAtomWithIdx(r_heavy)
        p_atom = product_mol_h.GetAtomWithIdx(p_heavy)
        if r_atom.GetSymbol() == "H" or p_atom.GetSymbol() == "H":
            continue

        r_implicit_hs = [
            n.GetIdx()
            for n in r_atom.GetNeighbors()
            if n.GetSymbol() == "H" and n.GetIdx() not in full
        ]
        p_implicit_hs = [
            n.GetIdx()
            for n in p_atom.GetNeighbors()
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

- [ ] **Step 4: テストを実行 (test_bond_changes 単体) → PASS**

Run: `pytest tests/test_bond_changes.py -v`
Expected: 全 6 ケース pass

- [ ] **Step 5: consumer の minimal migration**

`reactx/embed3d.py` の `_place_nucleophile_backside` 内 `bond_changes.formed`, `bond_changes.broken` (= 旧 single tuple) アクセスを `bond_changes.formed[0]`, `bond_changes.broken[0]` に置換し、`bond_changes.shared_atom` (旧 property) 利用箇所を、新しい inline ヘルパーで置き換える。

`reactx/embed3d.py` の Diff:
```python
# 旧 line 27:
from reactx.bond_changes import SimpleBondChanges
# 新:
from reactx.bond_changes import BondChanges

# 旧 line 42 引数型:
bond_changes: SimpleBondChanges | None = None,
# 新:
bond_changes: BondChanges | None = None,

# 旧 line 99 引数型:
bond_changes: SimpleBondChanges,
# 新:
bond_changes: BondChanges,

# 旧 lines 111-116:
a_form, b_form = bond_changes.formed
a_brk, b_brk = bond_changes.broken
shared = bond_changes.shared_atom
anchor = shared
leaving = b_brk if a_brk == shared else a_brk
incoming = b_form if a_form == shared else a_form
# 新 (Task 1 minimal — Task 2 で書き換える):
a_form, b_form = bond_changes.formed[0]
a_brk, b_brk = bond_changes.broken[0]
common = (set((a_form, b_form)) & set((a_brk, b_brk)))
if len(common) != 1:
    raise ValueError(
        f"formed {bond_changes.formed[0]} and broken {bond_changes.broken[0]} "
        f"must share exactly one atom; got {common}"
    )
shared = next(iter(common))
anchor = shared
leaving = b_brk if a_brk == shared else a_brk
incoming = b_form if a_form == shared else a_form
```

`reactx/cli.py` の Diff:
```python
# 旧 line 17:
from reactx.bond_changes import SimpleBondChanges, compute_simple_bond_changes
# 新:
from reactx.bond_changes import BondChanges, compute_bond_changes

# 旧 line 193:
bond_changes = compute_simple_bond_changes(r_h, p_h, mapping)
# 新:
bond_changes = compute_bond_changes(r_h, p_h, mapping)

# 旧 lines 202-207 (NEB 用 product-space SimpleBondChanges 構築):
a_form_r, b_form_r = bond_changes.formed
a_brk_r, b_brk_r = bond_changes.broken
bond_changes_product = SimpleBondChanges(
    formed=(mapping[a_brk_r], mapping[b_brk_r]),
    broken=(mapping[a_form_r], mapping[b_form_r]),
)
# 新 (Task 1 — formed[0]/broken[0] アクセス、構築は BondChanges):
a_form_r, b_form_r = bond_changes.formed[0]
a_brk_r, b_brk_r = bond_changes.broken[0]
bond_changes_product = BondChanges(
    formed=((mapping[a_brk_r], mapping[b_brk_r]),),
    broken=((mapping[a_form_r], mapping[b_form_r]),),
)

# 旧 lines 209-210:
formed_pair = bond_changes.formed
broken_pair = bond_changes.broken
# 新 (Task 1 — まだ singular で呼ぶ; Task 4 で multi 化):
formed_pair = bond_changes.formed[0]
broken_pair = bond_changes.broken[0]
```

`reactx/prescreen.py` の Diff:
```python
# 旧 line 30 付近 (もし import があれば):
# from reactx.bond_changes import SimpleBondChanges  ← 削除
# 実体は未使用 (formed/broken 引数は list[tuple] で受ける)
```

(Note: prescreen.py を確認したところ `SimpleBondChanges` import は無い。Step 5 のこの行は no-op。)

- [ ] **Step 6: 既存テストの import を rename**

以下のファイルすべてで `SimpleBondChanges` → `BondChanges`, `compute_simple_bond_changes` → `compute_bond_changes` に置換:

```bash
# tests/test_re1_sn2.py: SimpleBondChanges を import していないので変更不要
# tests/test_re1_proton_transfer.py: 同
# tests/test_re1_menshutkin.py: 同
# tests/test_neb_refine_sn2.py: import があれば置換
# tests/test_prescreen.py: import があれば置換
# tests/test_prescreen_disabled_sn2.py: 同
# tests/test_embed3d.py: import 必須
```

各ファイルを開いて `grep -l SimpleBondChanges tests/` 相当で対象を特定し、機械置換する。

`tests/test_embed3d.py` で `SimpleBondChanges(formed=(a,b), broken=(c,d))` の構築は `BondChanges(formed=((a,b),), broken=((c,d),))` に書き換える (tuple of tuples)。

- [ ] **Step 7: 全テスト実行 (slow / blender 除く) → PASS**

Run: `pytest -m "not slow and not blender" -v`
Expected: 既存 SN2/PT/Menshutkin ユニットテストすべて pass。新規 bond_changes テスト 6 件すべて pass。

- [ ] **Step 8: コミット**

```bash
git add reactx/bond_changes.py reactx/embed3d.py reactx/cli.py reactx/prescreen.py tests/
git commit -m "$(cat <<'EOF'
refactor(bond_changes): rename SimpleBondChanges -> BondChanges with multi-bond API

形成 1 + 切断 1 制限を構造的に解除 (まだ embed3d / cli は singular アクセスで
旧挙動維持)。multi-bond 配置は Task 2、CLI の formed_pairs / r_form_targets
list 化は Task 4 で。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: embed3d dispatcher refactor with directional placement for multi-bond

**Files:**
- Modify: `reactx/embed3d.py`
- Create: `tests/test_embed3d_placement.py`

**意図**: `_place_nucleophile_backside` を `_place_fragments` (dispatcher) + `_find_substrate_fragment` + `_directional_placement` (multi-bond 対応) に分解。1+1 反応では既存挙動を完全に維持し、E2 形状でも正しく anchor=H, leaving=C で配置できることを確認する。

- [ ] **Step 1: 失敗するテストを書く (`tests/test_embed3d_placement.py` 新規作成)**

```python
"""Unit tests for embed3d placement dispatcher (Phase 3)."""
import numpy as np
import pytest
from rdkit import Chem

from reactx.bond_changes import BondChanges
from reactx.embed3d import (
    _directional_placement,
    _find_substrate_fragment,
    _place_fragments,
    embed_mol_to_atoms,
)


def _make_mol_with_frags(smiles: str) -> tuple[Chem.Mol, tuple[tuple[int, ...], ...]]:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    return mol, Chem.GetMolFrags(mol)


def test_find_substrate_fragment_unique():
    # CH3CH2Cl + OH-: substrate has C-Cl and C-H broken intra
    mol, frags = _make_mol_with_frags("CC(Cl).[OH-]")
    # Pick atoms: substrate fragment (CCCl + Hs) is frags[0]
    substrate = frags[0]
    other = frags[1]
    # broken bonds inside substrate (use first 2 atoms as a stand-in)
    broken = ((substrate[0], substrate[1]), (substrate[1], substrate[2]))
    found = _find_substrate_fragment(frags, broken)
    assert found == substrate


def test_find_substrate_fragment_returns_none_when_broken_empty():
    mol, frags = _make_mol_with_frags("[CH3+].[Br-]")
    found = _find_substrate_fragment(frags, ())
    assert found is None


def test_find_substrate_fragment_returns_none_when_split_across_fragments():
    mol, frags = _make_mol_with_frags("CC.OO")
    a = frags[0][0]
    b = frags[1][0]
    # broken bond crossing fragments — multi-substrate metathesis-like
    found = _find_substrate_fragment(frags, ((a, b),))
    assert found is None


def test_place_fragments_dispatch_directional_for_sn2_shape(sn2_atoms_setup):
    """1 formed + 1 broken: should route to directional placement and produce
    well-separated fragments along backside."""
    mol_h, frag_indices, positions, bond_changes = sn2_atoms_setup
    out = _place_fragments(
        mol_h, frag_indices, positions.copy(), bond_changes,
        rotation_perturbation=None,
    )
    # Substrate (heavy atoms) unchanged; nucleophile fragment translated
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    o_idx = syms.index("O")
    c_idx = syms.index("C")
    cl_idx = syms.index("Cl")
    # O placed roughly opposite to Cl across C
    v_co = out[o_idx] - out[c_idx]
    v_ccl = out[cl_idx] - out[c_idx]
    cos_angle = float(np.dot(v_co, v_ccl) / (
        np.linalg.norm(v_co) * np.linalg.norm(v_ccl) + 1e-12
    ))
    assert cos_angle < -0.5, f"expected backside placement, got cos(angle)={cos_angle}"


def test_place_fragments_e2_shape_directional_anchors_on_h(e2_atoms_setup):
    """E2 形状 (1 formed + 2 broken): anchor=H, leaving=C, backside on H side."""
    mol_h, frag_indices, positions, bond_changes = e2_atoms_setup
    out = _place_fragments(
        mol_h, frag_indices, positions.copy(), bond_changes,
        rotation_perturbation=None,
    )
    # Base O should be placed near the abstracted H, on the side away from the C
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    # The base O is in the smaller fragment (without C-C-Cl)
    base_frag = next(
        f for f in frag_indices
        if all(syms[i] in {"O", "H"} for i in f)
    )
    o_idx = next(i for i in base_frag if syms[i] == "O")
    # The abstracted H is the explicit-mapped one in substrate (we constructed it
    # so it has a specific index in the fixture; assert against fixture).
    h_anchor = e2_atoms_setup_h_anchor()  # provided by fixture helper
    # O should be near anchor H
    d_oh = float(np.linalg.norm(out[o_idx] - out[h_anchor]))
    assert d_oh < 4.5, f"O should be close to anchor H, got {d_oh:.2f} A"


def test_place_fragments_raises_for_broken_zero_bimolecular():
    """SN1 step 2 形状 (broken=0, frags=2): centroid placement Phase 4+ → NotImplementedError."""
    mol = Chem.AddHs(Chem.MolFromSmiles("[CH3+].[OH-]"))
    frags = Chem.GetMolFrags(mol)
    bc = BondChanges(formed=((frags[0][0], frags[1][0]),), broken=())
    with pytest.raises(NotImplementedError, match="centroid"):
        _place_fragments(
            mol, frags, np.zeros((mol.GetNumAtoms(), 3)),
            bc, rotation_perturbation=None,
        )


def test_place_fragments_raises_for_multi_substrate_metathesis():
    """broken bonds が複数 fragment にまたがる → NotImplementedError."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CC.OO"))
    frags = Chem.GetMolFrags(mol)
    a = frags[0][0]  # C in CC
    b = frags[1][0]  # O in OO
    # Broken bond a-b crosses fragments (synthetic — physically nonsensical,
    # used only to trigger the multi-substrate guard)
    bc = BondChanges(formed=((frags[0][1], frags[1][1]),), broken=((a, b),))
    with pytest.raises(NotImplementedError, match="multi-substrate|centroid"):
        _place_fragments(
            mol, frags, np.zeros((mol.GetNumAtoms(), 3)),
            bc, rotation_perturbation=None,
        )


def test_place_fragments_raises_for_multi_base_on_single_anchor():
    """Same anchor with two fragment-bridging formed bonds → NotImplementedError."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CC.[F-].[Cl-]"))
    frags = Chem.GetMolFrags(mol)
    c0, c1 = frags[0][0], frags[0][1]
    f_idx = frags[1][0]
    cl_idx = frags[2][0]
    bc = BondChanges(
        formed=((c0, f_idx), (c0, cl_idx)),  # 2 base attacks on same C
        broken=((c0, c1),),                   # any broken in substrate
    )
    with pytest.raises(NotImplementedError, match="multi-base"):
        _place_fragments(
            mol, frags, np.zeros((mol.GetNumAtoms(), 3)),
            bc, rotation_perturbation=None,
        )
```

`tests/conftest.py` に E2 / SN2 fixture を追加:

```python
# (conftest.py に追記)
import numpy as np
from rdkit import Chem

from reactx.bond_changes import BondChanges


@pytest.fixture()
def sn2_atoms_setup():
    """3-fragment-style SN2 setup for placement dispatcher test (CH3Cl + OH-)."""
    mol = Chem.AddHs(Chem.MolFromSmiles("C(Cl).[OH-]"))
    Chem.SanitizeMol(mol)
    frag_indices = Chem.GetMolFrags(mol)
    n = mol.GetNumAtoms()
    positions = np.zeros((n, 3))
    # Spread atoms a bit so 'unit' vectors are well-defined
    for i in range(n):
        positions[i] = (float(i) * 0.5, 0.0, 0.0)
    # Find C, Cl, O indices
    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    c_idx = syms.index("C")
    cl_idx = syms.index("Cl")
    o_idx = syms.index("O")
    bc = BondChanges(formed=((c_idx, o_idx),), broken=((c_idx, cl_idx),))
    return mol, frag_indices, positions, bc


@pytest.fixture()
def e2_atoms_setup():
    """E2 setup: CH3CH2Cl + OH-, with explicit β-H atom."""
    # Use SMILES with explicit H on β-carbon via [H] tag
    smi = "[H][CH2:1][CH2:2][Cl:3].[O:4][H:5]"
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    Chem.SanitizeMol(mol)
    frag_indices = Chem.GetMolFrags(mol)
    n = mol.GetNumAtoms()
    # Provide deterministic non-overlapping positions
    positions = np.zeros((n, 3))
    for i in range(n):
        positions[i] = (float(i) * 0.5, float(i % 2), 0.0)

    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    map_to_idx = {a.GetAtomMapNum(): a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum()}
    c_alpha = map_to_idx[1]   # CH2 with Cl
    c_beta = map_to_idx[2]    # CH2 / CH3 with abstracted H — actually need to pick correctly
    cl = map_to_idx[3]
    o = map_to_idx[4]
    # The "H to abstract" is the explicit [H] adjacent to c_beta — find it
    # (it has no map number in this SMILES so we look it up structurally)
    # Wait — we want the H adjacent to c_beta. Pick an arbitrary H neighbor of c_beta.
    h_anchor = next(
        n.GetIdx() for n in mol.GetAtomWithIdx(c_beta).GetNeighbors()
        if n.GetSymbol() == "H"
    )
    bc = BondChanges(
        formed=((o, h_anchor),),
        broken=((c_alpha, cl), (c_beta, h_anchor)),
    )
    return mol, frag_indices, positions, bc


@pytest.fixture()
def e2_atoms_setup_h_anchor(e2_atoms_setup):
    """Helper to reach the H anchor used by the E2 fixture."""
    mol, _, _, bc = e2_atoms_setup
    # h_anchor is the atom common to a formed bond and a broken bond
    formed_atoms = set(bc.formed[0])
    for a, b in bc.broken:
        common = formed_atoms & {a, b}
        if common:
            return next(iter(common))
    raise RuntimeError("h_anchor not found")
```

(注: fixture 関数の戻り値で E2 の anchor H を露出する `e2_atoms_setup_h_anchor` を test 内で `e2_atoms_setup_h_anchor()` のように呼ぶ書き方は pytest fixture とは合わないため、テストの assertion 側で fixture オブジェクトから anchor を直接導出するように Step 2 で書き換える。)

- [ ] **Step 2: テスト assertion を fixture スタイルに修正**

`test_place_fragments_e2_shape_directional_anchors_on_h` の中で fixture の `bond_changes.formed[0]` と `bond_changes.broken` の共通原子を計算して `h_anchor` を求める形に書き直す:

```python
def test_place_fragments_e2_shape_directional_anchors_on_h(e2_atoms_setup):
    mol_h, frag_indices, positions, bond_changes = e2_atoms_setup
    formed_atoms = set(bond_changes.formed[0])
    h_anchor = next(
        next(iter(formed_atoms & {a, b}))
        for a, b in bond_changes.broken
        if formed_atoms & {a, b}
    )
    out = _place_fragments(
        mol_h, frag_indices, positions.copy(), bond_changes,
        rotation_perturbation=None,
    )
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    base_frag = next(
        f for f in frag_indices
        if all(syms[i] in {"O", "H"} for i in f)
    )
    o_idx = next(i for i in base_frag if syms[i] == "O")
    d_oh = float(np.linalg.norm(out[o_idx] - out[h_anchor]))
    assert d_oh < 4.5, f"O should be close to anchor H, got {d_oh:.2f} A"
```

Test 5 (`test_place_fragments_e2_shape_directional_anchors_on_h`) の `e2_atoms_setup_h_anchor()` 呼び出しは削除し、上記書き換えを適用。`e2_atoms_setup_h_anchor` fixture も削除。

- [ ] **Step 3: テスト実行 (Step 1 のテストが ImportError で fail することを確認)**

Run: `pytest tests/test_embed3d_placement.py -v`
Expected: ImportError (`_place_fragments`, `_find_substrate_fragment`, `_directional_placement` 未定義)

- [ ] **Step 4: `reactx/embed3d.py` を refactor**

`_place_nucleophile_backside` を以下の 3 関数に置き換える (関数全体を削除し、新規関数を挿入):

```python
import logging
log = logging.getLogger(__name__)


def _place_fragments(
    mol_h: Chem.Mol,
    frag_indices: tuple[tuple[int, ...], ...],
    positions: np.ndarray,
    bond_changes: BondChanges,
    *,
    rotation_perturbation: np.ndarray | None,
) -> np.ndarray:
    """Dispatch to the appropriate placement strategy.

    Tier 1 (directional, Phase 3): broken bond の方向情報がある反応 (E2 / SN2 / PT)。
    Tier 2 (centroid, future):     broken=0 / multi-substrate metathesis。Phase 4+。
    """
    substrate = _find_substrate_fragment(frag_indices, bond_changes.broken)
    if substrate is not None and bond_changes.broken:
        return _directional_placement(
            frag_indices, positions, bond_changes, substrate,
            rotation_perturbation=rotation_perturbation,
        )
    raise NotImplementedError(
        "Centroid-based placement (broken=0 / multi-substrate metathesis) is "
        f"Phase 4+. Got formed={bond_changes.formed}, broken={bond_changes.broken}, "
        f"frags={len(frag_indices)}."
    )


def _find_substrate_fragment(
    frag_indices: tuple[tuple[int, ...], ...],
    broken: tuple[tuple[int, int], ...],
) -> tuple[int, ...] | None:
    """Return the unique fragment containing both atoms of every broken bond.

    None when broken is empty, or when broken bonds span multiple fragments.
    """
    if not broken:
        return None
    candidates: list[tuple[int, ...]] = []
    for frag in frag_indices:
        frag_set = set(frag)
        if all(a in frag_set and b in frag_set for a, b in broken):
            candidates.append(frag)
    if len(candidates) != 1:
        return None
    return candidates[0]


def _directional_placement(
    frag_indices: tuple[tuple[int, ...], ...],
    positions: np.ndarray,
    bond_changes: BondChanges,
    substrate: tuple[int, ...],
    *,
    rotation_perturbation: np.ndarray | None,
) -> np.ndarray:
    """Tier 1: anchor + leaving direction for each non-substrate fragment."""
    substrate_set = set(substrate)
    non_substrate = [f for f in frag_indices if f is not substrate]
    if len(non_substrate) >= 2:
        log.warning(
            "termolecular placement (%d non-substrate fragments); geometric "
            "quality may be reduced", len(non_substrate),
        )

    # Group fragment-bridging formed bonds by their substrate-side anchor.
    bridging_by_anchor: dict[int, list[tuple[int, ...]]] = {}
    for f_idx, f in enumerate(non_substrate):
        f_set = set(f)
        bridging = [
            (a, b) for a, b in bond_changes.formed
            if (a in substrate_set and b in f_set) or (b in substrate_set and a in f_set)
        ]
        if not bridging:
            raise ValueError(
                f"fragment {f_idx} has no formed bond bridging to substrate; "
                f"check input atom mapping (formed={bond_changes.formed}, "
                f"substrate atoms={sorted(substrate_set)})"
            )
        for bond in bridging:
            anchor = bond[0] if bond[0] in substrate_set else bond[1]
            bridging_by_anchor.setdefault(anchor, []).append((f, bond))

    # Detect multi-base attack on a single anchor.
    for anchor, hits in bridging_by_anchor.items():
        unique_frags = {id(f) for f, _ in hits}
        if len(unique_frags) > 1:
            raise NotImplementedError(
                f"multi-base attack on single anchor {anchor} not supported "
                f"(Phase 3 supports at most one fragment per anchor)"
            )

    # For each non-substrate fragment, compute backside and translate.
    for fragment in non_substrate:
        f_set = set(fragment)
        # Locate this fragment's bridging bond + anchor.
        anchor: int | None = None
        bridging_bond: tuple[int, int] | None = None
        for a, hits in bridging_by_anchor.items():
            for f, bond in hits:
                if f is fragment:
                    anchor, bridging_bond = a, bond
                    break
            if anchor is not None:
                break
        assert anchor is not None and bridging_bond is not None  # invariant

        # Find broken bonds containing anchor; pick deterministically.
        relevant_broken = sorted(
            [(a, b) for a, b in bond_changes.broken if anchor in (a, b)],
            key=lambda ab: ab[1] if ab[0] == anchor else ab[0],
        )
        if relevant_broken:
            chosen = relevant_broken[0]
            leaving = chosen[1] if chosen[0] == anchor else chosen[0]
        else:
            substrate_centroid = positions[list(substrate)].mean(axis=0)
            distances = [
                float(np.linalg.norm(positions[i] - substrate_centroid))
                for i in substrate if i != anchor
            ]
            leaving_candidates = [i for i in substrate if i != anchor]
            leaving = leaving_candidates[int(np.argmax(distances))]

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

        # Anchor on the incoming atom of the bridging bond when present.
        incoming = bridging_bond[1] if bridging_bond[0] == anchor else bridging_bond[0]
        if incoming in f_set:
            nuc_centroid_anchor = positions[incoming]
        else:
            nuc_centroid_anchor = positions[list(fragment)].mean(axis=0)
        positions[list(fragment)] += target - nuc_centroid_anchor

    return positions
```

`embed_mol_to_atoms` 内の呼び出しを差し替える:

```python
# 旧 line 75:
positions = _place_nucleophile_backside(...)
# 新:
positions = _place_fragments(
    mol_h, frag_indices, positions, bond_changes,
    rotation_perturbation=rotation_perturbation,
)
```

Task 1 で挿入した「`shared = next(iter(common))` で 1+1 を inline 計算する」コードブロックは、新しい `_place_fragments` の呼び出しに置き換えて消す。

- [ ] **Step 5: テスト実行 (新規 + 既存) → PASS**

Run: `pytest tests/test_embed3d_placement.py tests/test_embed3d.py tests/test_re1_sn2.py tests/test_re1_proton_transfer.py tests/test_re1_menshutkin.py -v -m "not slow"`
Expected: 全 pass。SN2/PT/Menshutkin の挙動が変わっていないこと、新規 7 ケースが pass すること。

- [ ] **Step 6: コミット**

```bash
git add reactx/embed3d.py tests/test_embed3d_placement.py tests/conftest.py
git commit -m "$(cat <<'EOF'
feat(embed3d): split _place_nucleophile_backside into multi-bond dispatcher

dispatcher (_place_fragments) + Tier 1 directional_placement で multi-bond
反応の fragment 配置を一般化。Tier 2 centroid placement は Phase 4+ stub
として NotImplementedError。1+1 反応 (SN2/PT/Menshutkin) では既存挙動を完全
維持。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: New presets (e2, sn1_dissoc)

**Files:**
- Modify: `reactx/presets.py`
- Modify: `tests/test_presets.py`

- [ ] **Step 1: 失敗するテストを書く (`tests/test_presets.py` 末尾に追加)**

```python
def test_e2_preset_values():
    p = get_preset("e2")
    assert p.name == "e2"
    assert p.k_form == 1.0
    assert p.k_broken == 1.0
    assert p.r_broken == 4.0
    assert p.max_relax_steps == 200
    assert p.r_form is None  # element-pair table


def test_sn1_dissoc_preset_values():
    p = get_preset("sn1_dissoc")
    assert p.name == "sn1_dissoc"
    assert p.k_form == 0.0
    assert p.k_broken == 2.0
    assert p.r_broken == 6.0
    assert p.max_relax_steps == 200
    assert p.r_form is None
```

- [ ] **Step 2: テスト実行 → FAIL**

Run: `pytest tests/test_presets.py::test_e2_preset_values tests/test_presets.py::test_sn1_dissoc_preset_values -v`
Expected: FAIL (`Unknown reaction-type preset 'e2'`)

- [ ] **Step 3: `reactx/presets.py` の `PRESETS` dict に追加**

```python
PRESETS["e2"] = ReactionPreset(
    name="e2",
    k_form=1.0,
    k_broken=1.0,
    r_broken=4.0,
    max_relax_steps=200,
)
PRESETS["sn1_dissoc"] = ReactionPreset(
    name="sn1_dissoc",
    k_form=0.0,
    k_broken=2.0,
    r_broken=6.0,
    max_relax_steps=200,
)
```

- [ ] **Step 4: テスト実行 → PASS**

Run: `pytest tests/test_presets.py -v`
Expected: 全 pass

- [ ] **Step 5: コミット**

```bash
git add reactx/presets.py tests/test_presets.py
git commit -m "$(cat <<'EOF'
feat(presets): add e2 and sn1_dissoc reaction-type presets

E2 elimination (1 formed + 2 broken) は既定 k_*/r_broken を SN2 anion から
微調整、max_relax_steps を 200 に増やす。SN1 step1 解離 (0 formed + 1 broken)
は k_form=0 + 強い k_broken + r_broken=6 で十分なイオン対分離を担保。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: CLI: multi-bond support (formed_pairs, r_form_targets list, --reaction-type choices)

**Files:**
- Modify: `reactx/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_re1_sn2.py`, `tests/test_re1_proton_transfer.py`, `tests/test_re1_menshutkin.py` (assert effective_params.r_form_targets が list)

**意図**: Task 1 で `bond_changes.formed[0]` でアクセスしていた箇所を「全 formed/broken bond を渡す」形に汎用化。`_resolve_effective_params` を `r_form: float` ではなく `r_form_targets: list[float]` を返す形に変更する。`--reaction-type` の choices に `e2`, `sn1_dissoc` を追加。NEB 用の product-space `BondChanges` 構築は 1+1 反応のみ対応とし、それ以外では skip (Task 6 でガード)。

- [ ] **Step 1: 失敗するテストを書く (`tests/test_cli.py` に追加)**

```python
def test_resolve_effective_params_returns_list_for_r_form_targets():
    """_resolve_effective_params returns r_form_targets as list[float] (Phase 3)."""
    from reactx.cli import _resolve_effective_params
    import argparse
    args = argparse.Namespace(
        reaction_type="sn2_anion",
        k_form=None, k_broken=None, r_broken=None, r_form=None,
        max_relax_steps=None,
    )
    syms = ["C", "Cl", "O"]
    formed_pairs = [(0, 2)]  # C-O formed
    eff = _resolve_effective_params(args, syms, formed_pairs)
    assert isinstance(eff["r_form_targets"], list)
    assert len(eff["r_form_targets"]) == 1
    assert abs(eff["r_form_targets"][0] - 1.43) < 1e-3  # C-O table value


def test_resolve_effective_params_empty_formed_returns_empty_list():
    from reactx.cli import _resolve_effective_params
    import argparse
    args = argparse.Namespace(
        reaction_type="sn1_dissoc",
        k_form=None, k_broken=None, r_broken=None, r_form=None,
        max_relax_steps=None,
    )
    eff = _resolve_effective_params(args, ["C", "Br"], [])
    assert eff["r_form_targets"] == []


def test_resolve_effective_params_scalar_r_form_broadcasts():
    from reactx.cli import _resolve_effective_params
    import argparse
    args = argparse.Namespace(
        reaction_type="sn2_anion",
        k_form=None, k_broken=None, r_broken=None, r_form=1.10,
        max_relax_steps=None,
    )
    eff = _resolve_effective_params(args, ["C", "Cl", "O", "F"], [(0, 2), (0, 3)])
    assert eff["r_form_targets"] == [1.10, 1.10]
```

- [ ] **Step 2: テスト実行 → FAIL (KeyError 'r_form_targets')**

Run: `pytest tests/test_cli.py::test_resolve_effective_params_returns_list_for_r_form_targets -v`
Expected: FAIL (KeyError 'r_form_targets' or AttributeError)

- [ ] **Step 3: `reactx/cli.py` の `_resolve_effective_params` を書き換える**

```python
def _resolve_effective_params(
    args, syms: list[str], formed_pairs: list[tuple[int, int]],
) -> dict:
    """Merge preset + individual-flag overrides into a flat dict.

    r_form は per-formed-bond の list として返す:
      - --r-form scalar 指定 → 全要素同値
      - preset.r_form 指定   → 全要素同値
      - 両方 None            → 元素ペア表を per-bond でルックアップ
    formed_pairs が空 (e.g. SN1 step1) のときは r_form_targets=[]。
    """
    preset = get_preset(args.reaction_type)
    k_form = preset.k_form if args.k_form is None else float(args.k_form)
    k_broken = preset.k_broken if args.k_broken is None else float(args.k_broken)
    r_broken = preset.r_broken if args.r_broken is None else float(args.r_broken)
    max_relax_steps = (
        preset.max_relax_steps if args.max_relax_steps is None
        else int(args.max_relax_steps)
    )

    if args.r_form is not None:
        r_form_targets = [float(args.r_form)] * len(formed_pairs)
    elif preset.r_form is not None:
        r_form_targets = [float(preset.r_form)] * len(formed_pairs)
    else:
        r_form_targets = [lookup_r_form(syms[a], syms[b]) for a, b in formed_pairs]

    return {
        "reaction_type": preset.name,
        "k_form": k_form,
        "k_broken": k_broken,
        "r_broken": r_broken,
        "max_relax_steps": max_relax_steps,
        "r_form_targets": r_form_targets,
    }
```

- [ ] **Step 4: `_cmd_run` を multi-bond 化**

`reactx/cli.py` の以下の箇所を書き換える:

```python
# 旧 lines 209-219:
formed_pair = bond_changes.formed[0]   # Task 1 でこう書き換えていた
broken_pair = bond_changes.broken[0]
syms_r = [a.GetSymbol() for a in r_h.GetAtoms()]
eff = _resolve_effective_params(args, syms_r, formed_pair)
r_form_target = eff["r_form"]
log.info(
    "preset=%s effective: k_form=%.2f k_broken=%.2f r_broken=%.2f "
    "max_relax_steps=%d r_form=%.3f",
    eff["reaction_type"], eff["k_form"], eff["k_broken"],
    eff["r_broken"], eff["max_relax_steps"], eff["r_form"],
)
# 新:
formed_pairs = list(bond_changes.formed)
broken_pairs = list(bond_changes.broken)
syms_r = [a.GetSymbol() for a in r_h.GetAtoms()]
eff = _resolve_effective_params(args, syms_r, formed_pairs)
r_form_targets = eff["r_form_targets"]
log.info(
    "preset=%s effective: k_form=%.2f k_broken=%.2f r_broken=%.2f "
    "max_relax_steps=%d r_form_targets=%s",
    eff["reaction_type"], eff["k_form"], eff["k_broken"],
    eff["r_broken"], eff["max_relax_steps"],
    [f"{x:.3f}" for x in r_form_targets] if r_form_targets else "[]",
)
```

prescreen 呼び出し:
```python
# 旧 lines 263-271:
pre = prescreen_trials(
    atoms_list=atoms_for_prescreen,
    mol_h_template=mol_h_template,
    formed=[formed_pair], broken=[broken_pair],
    r_form_target=r_form_target,
    ...
)
# 新:
pre = prescreen_trials(
    atoms_list=atoms_for_prescreen,
    mol_h_template=mol_h_template,
    formed=formed_pairs, broken=broken_pairs,
    r_form_target=r_form_targets[0] if r_form_targets else 1.6,
    ...
)
```

(注: `prescreen_trials` の `r_form_target` は scalar 1 つ; 内部で全 formed bond に同値 broadcast している。Phase 3 では scalar 渡しの API を維持。SN1 dissoc の `r_form_targets=[]` の場合は default scalar を渡すが、formed=[] なので Hookean 拘束は 0 本作られて scalar は使われない。)

build_restraints 呼び出し:
```python
# 旧 lines 289-297:
restraints = build_restraints(
    atoms_init,
    formed=[formed_pair],
    broken=[broken_pair],
    r_form=r_form_target,
    ...
)
# 新:
restraints = build_restraints(
    atoms_init,
    formed=formed_pairs,
    broken=broken_pairs,
    r_form=r_form_targets[0] if r_form_targets else None,
    ...
)
```

(注: `build_restraints` は `r_form: float | None`; scalar 渡しは「全 formed bond に同値」を意味する。E2 では formed が 1 本なので scalar=1 値で同義。)

reached_product:
```python
# 旧 lines 313-319:
ok = reached_product(
    frames[-1],
    formed=[formed_pair],
    broken=[broken_pair],
    r_form_targets=[r_form_target],
    r_broken_target=eff["r_broken"],
)
# 新:
ok = reached_product(
    frames[-1],
    formed=formed_pairs,
    broken=broken_pairs,
    r_form_targets=r_form_targets,
    r_broken_target=eff["r_broken"],
)
```

`_write_outputs_and_exit` の `effective_params` 出力:
```python
# 旧 line 437-440:
"effective_params": (
    {
        k: effective[k]
        for k in ("k_form", "k_broken", "r_broken", "max_relax_steps", "r_form")
    }
    if effective is not None else None
),
# 新:
"effective_params": (
    {
        k: effective[k]
        for k in ("k_form", "k_broken", "r_broken", "max_relax_steps", "r_form_targets")
    }
    if effective is not None else None
),
```

`--reaction-type` choices 拡張は `presets.PRESETS` の自動展開で既に対応済 (line 49-52 の `sorted(_PRESETS)` を使っているため、Task 3 の preset 追加で自動的に reflect される)。

- [ ] **Step 5: 既存 SN2/PT/Menshutkin テストを `r_form_targets` list に追従**

`tests/test_re1_sn2.py` および `tests/test_re1_proton_transfer.py`, `tests/test_re1_menshutkin.py` で `meta["effective_params"]["r_form"]` を assert している箇所を `r_form_targets` (1 要素 list) に書き換える。具体的には:

```python
# 旧 (もし存在すれば):
assert isinstance(meta["effective_params"]["r_form"], float)
# 新:
assert isinstance(meta["effective_params"]["r_form_targets"], list)
assert len(meta["effective_params"]["r_form_targets"]) == 1
```

- [ ] **Step 6: 全テスト実行 (slow 除く) → PASS**

Run: `pytest -m "not slow and not blender" -v`
Expected: 全 pass

- [ ] **Step 7: SN2 slow integration を 1 件流して回帰確認**

Run: `pytest tests/test_re1_sn2.py -v -m slow` (UMA 必要)
Expected: pass、selected_trial >= 0、reached_product のトライアルが 1 件以上

- [ ] **Step 8: コミット**

```bash
git add reactx/cli.py tests/test_cli.py tests/test_re1_sn2.py tests/test_re1_proton_transfer.py tests/test_re1_menshutkin.py
git commit -m "$(cat <<'EOF'
feat(cli): generalize formed/broken to lists + r_form_targets per-bond

_resolve_effective_params returns r_form_targets: list[float] (per formed
bond) instead of r_form: float. Pipeline calls (build_restraints,
prescreen_trials, reached_product) feed bond_changes.formed/broken の全 bond.
meta.json.effective_params.r_form (scalar) -> r_form_targets (list) は
breaking change だが phase-3 ブランチで許容。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: CLI: unimolecular auto-clamp

**Files:**
- Modify: `reactx/cli.py`
- Create: `tests/test_cli_unimolecular.py`

- [ ] **Step 1: 失敗するテストを書く**

```python
"""Tests for unimolecular reaction handling: --n-angles auto-clamp + prescreen skip."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from reactx.cli import main


@pytest.fixture()
def fake_unimolecular_pipeline(monkeypatch, tmp_path):
    """Patch the UMA pipeline to a 1-fragment fake reaction.

    We don't need a real UMA call; intercept calculator + relax to return
    immediately, then inspect meta.json behaviour.
    """
    from reactx import calculators, embed3d, path_relax
    import numpy as np
    from ase import Atoms

    def fake_calc(*args, **kwargs):
        from ase.calculators.lj import LennardJones
        return LennardJones()

    def fake_embed(mol, **kw):
        # Return fake 5-atom single-fragment Atoms
        atoms = Atoms("CCCCC", positions=np.array([
            [0, 0, 0], [1.5, 0, 0], [3.0, 0, 0], [-1.5, 0, 0], [0, 1.5, 0],
        ]))
        return atoms

    def fake_relax(atoms_init, restraints, calc, **kw):
        return [atoms_init], [0.0]

    monkeypatch.setattr(calculators, "make_calculator", fake_calc)
    monkeypatch.setattr(embed3d, "embed_mol_to_atoms", fake_embed)
    monkeypatch.setattr(path_relax, "relax_with_restraints", fake_relax)


def test_unimolecular_n_angles_clamped_to_one(fake_unimolecular_pipeline, tmp_path):
    """When reactant has 1 fragment, --n-angles N is clamped to 1 and prescreen skipped."""
    rxn = Path("examples/sn1_dissoc.rxn")  # created in Task 8 — for now use a placeholder
    if not rxn.exists():
        pytest.skip("examples/sn1_dissoc.rxn not yet created (Task 8)")
    out = tmp_path / "out"
    rc = main([
        "run", str(rxn), "-o", str(out),
        "--backend", "lj",
        "--reaction-type", "sn1_dissoc",
        "--n-angles", "8",
    ])
    assert rc == 0
    meta = json.loads((out / "meta.json").read_text())
    assert len(meta["trials"]) == 1, f"expected 1 trial after clamp, got {len(meta['trials'])}"
    assert meta["prescreen"]["enabled"] is False or meta["prescreen"]["kept"] is None
```

- [ ] **Step 2: テスト実行 → SKIP (sn1_dissoc.rxn まだ無い) — Task 8 完了後に PASS する想定**

Run: `pytest tests/test_cli_unimolecular.py -v`
Expected: SKIP

- [ ] **Step 3: `reactx/cli.py` に unimolecular ガードを追加**

`_cmd_run` の `rotations = sample_attack_rotations(...)` 直前に挿入:

```python
n_frags_reactant = len(Chem.GetMolFrags(r_h))
effective_n_angles = args.n_angles
if n_frags_reactant == 1 and args.n_angles > 1:
    log.info(
        "unimolecular reaction (1 reactant fragment); "
        "n_angles forced from %d to 1, prescreen skipped",
        args.n_angles,
    )
    effective_n_angles = 1

rotations = sample_attack_rotations(
    n=effective_n_angles, cone_half_deg=args.cone_half_deg, seed=args.seed,
)
```

unimolecular で prescreen を skip する分岐:

```python
# 旧 prescreen 分岐 (line 251 以降):
if args.no_mmff_prescreen:
    log.info("prescreen: disabled (--no-mmff-prescreen)")
    keep_trial_indices = sorted(embedded_by_idx.keys())
    prescreen_meta = {...}
elif embedded_by_idx:
    ...
# 新 (先頭に unimolecular 分岐を追加):
if n_frags_reactant == 1:
    log.info("prescreen: skipped (single trial / unimolecular)")
    keep_trial_indices = sorted(embedded_by_idx.keys())
    prescreen_meta = {
        "enabled": False, "kept": None, "skipped": None,
        "mmff_failed": None, "wall_clock_seconds": 0.0,
    }
elif args.no_mmff_prescreen:
    ...
elif embedded_by_idx:
    ...
```

- [ ] **Step 4: テスト実行 → SKIP (まだ sn1_dissoc.rxn 無し)、SN2 回帰は PASS**

Run: `pytest -m "not slow and not blender" -v`
Expected: 全 pass (test_cli_unimolecular は SKIP)

- [ ] **Step 5: コミット**

```bash
git add reactx/cli.py tests/test_cli_unimolecular.py
git commit -m "$(cat <<'EOF'
feat(cli): auto-clamp --n-angles to 1 for unimolecular reactions

reactant fragment 数が 1 のとき multi-angle 試行に意味が無いので n_angles=1
にクランプし prescreen も skip。SN1 step1 (dissociation) の wall-clock を抑える。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: CLI: --neb-refine guard for non-1+1 reactions

**Files:**
- Modify: `reactx/cli.py`
- Create: `tests/test_cli_neb_refine_guard.py`

**意図**: `bond_changes_product` を構築するための swap ロジックは 1+1 反応にしか妥当でない。Phase 3 では `--neb-refine` が指定されたとき、`len(formed) != 1 or len(broken) != 1` で argparse error にする。

- [ ] **Step 1: 失敗するテストを書く**

```python
"""Test that --neb-refine is rejected for multi-bond reactions in Phase 3."""
import subprocess
import sys
from pathlib import Path

import pytest


def test_neb_refine_rejected_for_e2_reaction(tmp_path):
    rxn = Path("examples/e2.rxn")
    if not rxn.exists():
        pytest.skip("examples/e2.rxn not yet created (Task 7)")
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, "-m", "reactx.cli", "run", str(rxn), "-o", str(out),
         "--backend", "lj", "--neb-refine"],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "Phase 3" in (proc.stdout + proc.stderr)
    assert "1 formed + 1 broken" in (proc.stdout + proc.stderr)


def test_neb_refine_accepted_for_sn2(sn2_rxn_path, tmp_path, monkeypatch):
    """1+1 反応 (SN2) では --neb-refine が CLI レベルで rejected されない。
    実際の NEB 実行は UMA 必須なので mock する。"""
    from reactx import cli
    out = tmp_path / "out"
    # Just check the CLI argparse layer doesn't reject. We mock the heavy bits.
    monkeypatch.setattr(cli, "_check_hf_auth", lambda: 0)

    # If we get past argparse and reach calculator init, that's enough for this test.
    # Use --backend lj to skip UMA entirely.
    rc = cli.main([
        "run", str(sn2_rxn_path), "-o", str(out),
        "--backend", "lj", "--n-angles", "1", "--max-relax-steps", "5",
        "--neb-refine",
    ])
    # rc may be non-zero due to LJ being unable to do NEB convergence — but
    # we only care that we got *past* the early CLI guard.
    # Specifically, meta.json should exist (we got at least to embed phase).
    assert (out / "meta.json").exists(), "expected to pass CLI guard and reach pipeline"
```

- [ ] **Step 2: テスト実行 → SKIP/FAIL**

Run: `pytest tests/test_cli_neb_refine_guard.py -v`
Expected: 1 件 SKIP (e2 まだ無い)、1 件 FAIL (まだ guard 未実装で rc=0 のまま)

- [ ] **Step 3: `reactx/cli.py` に NEB refine ガードを追加**

`_cmd_run` の `if args.neb_refine and len(final_frames) >= 2:` ブロックの直前に:

```python
if args.neb_refine and (len(bond_changes.formed) != 1 or len(bond_changes.broken) != 1):
    log.error(
        "--neb-refine is only supported for 1 formed + 1 broken bond reactions "
        "in Phase 3 (got formed=%d, broken=%d). Multi-bond NEB endpoint "
        "construction is Phase 4+. Re-run without --neb-refine.",
        len(bond_changes.formed), len(bond_changes.broken),
    )
    return 2
```

これは `bond_changes` 計算後に置く必要があるので、`bond_changes = compute_bond_changes(...)` の **直後** が最も早く正しいタイミング (= UMA load 前)。具体的には `r_h = Chem.AddHs(r_mol); p_h = Chem.AddHs(p_mol); bond_changes = compute_bond_changes(...)` の次行に挿入する。

```python
bond_changes = compute_bond_changes(r_h, p_h, mapping)
if args.neb_refine and (len(bond_changes.formed) != 1 or len(bond_changes.broken) != 1):
    log.error(
        "--neb-refine is only supported for 1 formed + 1 broken bond reactions "
        "in Phase 3 (got formed=%d, broken=%d). Multi-bond NEB endpoint "
        "construction is Phase 4+. Re-run without --neb-refine.",
        len(bond_changes.formed), len(bond_changes.broken),
    )
    return 2
```

- [ ] **Step 4: テスト実行 → 1 件 SKIP (e2 まだ無い)、SN2 ケース PASS**

Run: `pytest tests/test_cli_neb_refine_guard.py -v`
Expected: 1 件 SKIP, 1 件 PASS

- [ ] **Step 5: 全テスト regression → PASS**

Run: `pytest -m "not slow and not blender" -v`
Expected: 全 pass

- [ ] **Step 6: コミット**

```bash
git add reactx/cli.py tests/test_cli_neb_refine_guard.py
git commit -m "$(cat <<'EOF'
feat(cli): reject --neb-refine for non-1+1 reactions in Phase 3

bond_changes_product 構築の swap ロジックは 1+1 反応にしか妥当でないため、
それ以外 (E2 / SN1 dissoc) では --neb-refine を CLI レベルで rejecting し
exit code 2 を返す。multi-bond NEB endpoint 構築は Phase 4+。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: examples/e2.rxn + integration test

**Files:**
- Create: `examples/e2.rxn`
- Create: `tests/test_re3_e2.py`
- Modify: `tests/conftest.py` (e2_rxn_path fixture 追加)
- Modify: `tests/test_artificial_force.py` (multi-bond regression 1 件追加)
- Modify: `tests/test_scoring.py` (per-bond r_form_targets 1 件追加)

- [ ] **Step 1: `examples/e2.rxn` を作成**

CH₃CH₂Cl + OH⁻ → CH₂=CH₂ + Cl⁻ + H₂O。原子マップ:
- map 1 = α-C (持つ Cl)
- map 2 = β-C (持つ抽象される H)
- map 3 = Cl
- map 4 = O (base)
- map 5 = β-H (抽象される; reactant では C2 上、product では water 中)
- map 6 = O 上の H (base, reactant)、product でも water 中

```
$RXN

      RDKit

  2  3
$MOL

     RDKit          2D

  4  3  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  1  0  0
   -1.5000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  2  0  0
    1.5000    0.0000    0.0000 Cl  0  0  0  0  0  0  0  0  0  3  0  0
   -2.5000    0.7000    0.0000 H   0  0  0  0  0  0  0  0  0  5  0  0
  1  2  1  0
  1  3  1  0
  2  4  1  0
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

  2  1  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  1  0  0
   -1.3000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  2  0  0
  1  2  2  0
M  END
$MOL

     RDKit          2D

  1  0  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 Cl  0  0  0  0  0  0  0  0  0  3  0  0
M  CHG  1   1  -1
M  END
$MOL

     RDKit          2D

  3  2  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  4  0  0
   -0.8000    0.5000    0.0000 H   0  0  0  0  0  0  0  0  0  5  0  0
    0.8000    0.5000    0.0000 H   0  0  0  0  0  0  0  0  0  6  0  0
  1  2  1  0
  1  3  1  0
M  END
```

- [ ] **Step 2: e2.rxn のパースを smoke test**

Run:
```python
python -c "
from reactx.rxn_parser import parse_rxn
from reactx.bond_changes import compute_bond_changes
from rdkit import Chem
r, p, m = parse_rxn('examples/e2.rxn')
r_h = Chem.AddHs(r)
p_h = Chem.AddHs(p)
bc = compute_bond_changes(r_h, p_h, m)
print('formed:', bc.formed, 'broken:', bc.broken)
print('atom counts:', r_h.GetNumAtoms(), p_h.GetNumAtoms())
"
```

Expected: `formed` が 1 件 (O–H_β 形成), `broken` が 2 件 (C–Cl, C–H_β)。原子数は両側とも 10。

もし atom count mismatch / implicit H mismatch 等が出たら .rxn の H/charge を調整。

- [ ] **Step 3: `tests/conftest.py` に fixture 追加**

```python
@pytest.fixture()
def e2_rxn_path(examples_dir: Path) -> Path:
    return examples_dir / "e2.rxn"


@pytest.fixture()
def sn1_dissoc_rxn_path(examples_dir: Path) -> Path:
    return examples_dir / "sn1_dissoc.rxn"
```

- [ ] **Step 4: `tests/test_artificial_force.py` に multi-bond 線形性テストを追加**

```python
def test_build_restraints_scales_linearly_with_bond_count():
    """Phase 3 multi-bond regression: 2 formed + 1 broken -> 3 constraints."""
    atoms = Atoms("CHFNN", positions=[
        [0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0],
    ])
    cs = build_restraints(
        atoms,
        formed=[(0, 1), (2, 3)],
        broken=[(0, 4)],
        r_form=1.5, r_broken=4.0,
    )
    assert len(cs) == 3  # 2 Hookeans + 1 PullApart
```

- [ ] **Step 5: `tests/test_scoring.py` に per-bond r_form_targets テストを追加**

```python
def test_reached_product_per_bond_r_form_targets():
    """Phase 3: r_form_targets list で formed bond ごとに別個に判定する。"""
    a = Atoms("OFCN", positions=[
        [0, 0, 0], [1.5, 0, 0], [-3.0, 0, 0], [-3.0, 1.05, 0],
    ])
    # Two formed bonds with different targets (O-F: 1.5, C-N: 1.05).
    # Both within target+tol → reached_product True.
    assert reached_product(
        a, formed=[(0, 1), (2, 3)], broken=[],
        r_form_targets=[1.5, 1.05], r_broken_target=4.0,
    ) is True

    # Make first one too far.
    a.set_positions([[0, 0, 0], [3.0, 0, 0], [-3.0, 0, 0], [-3.0, 1.05, 0]])
    assert reached_product(
        a, formed=[(0, 1), (2, 3)], broken=[],
        r_form_targets=[1.5, 1.05], r_broken_target=4.0,
    ) is False


def test_reached_product_mismatched_targets_length_raises():
    a = Atoms("OF", positions=[[0, 0, 0], [1.5, 0, 0]])
    with pytest.raises(ValueError, match="r_form_targets"):
        reached_product(
            a, formed=[(0, 1)], broken=[],
            r_form_targets=[1.5, 1.05],   # 2 targets but only 1 formed
            r_broken_target=4.0,
        )
```

- [ ] **Step 6: 高速ユニットテスト → PASS**

Run: `pytest -m "not slow and not blender" -v`
Expected: 全 pass。bond_changes / artificial_force / scoring の新規テスト 4 件追加。

- [ ] **Step 7: 統合テスト `tests/test_re3_e2.py` を書く**

```python
"""End-to-end E2 elimination test using UMA. Marked slow."""
import json
from pathlib import Path

import pytest
from ase.io import read

from reactx.cli import main


@pytest.mark.slow
def test_re3_e2_end_to_end(tmp_path: Path, e2_rxn_path: Path):
    out = tmp_path / "e2"
    rc = main([
        "run", str(e2_rxn_path), "-o", str(out),
        "--backend", "uma",
        "--reaction-type", "e2",
        "--n-angles", "4",
    ])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())
    assert meta["selected_trial"] >= 0, f"no trial selected: {meta}"
    assert any(t["reached_product"] for t in meta["trials"]), (
        f"no E2 trial reached product: {meta['trials']}"
    )
    # Phase 3 schema: r_form_targets is list[float]
    assert isinstance(meta["effective_params"]["r_form_targets"], list)
    assert len(meta["effective_params"]["r_form_targets"]) == 1   # E2 has 1 formed bond
    assert meta["reaction_type"] == "e2"

    frames = read(str(out / "trajectory.xyz"), index=":")
    assert len(frames) >= 3

    # E2 geometry assertions: identify atoms by symbol + connectivity.
    syms = frames[0].get_chemical_symbols()
    cl_idx = syms.index("Cl")
    o_idx = syms.index("O")
    # Find the C bonded to Cl in initial frame (= α-carbon)
    c_atoms = [i for i, s in enumerate(syms) if s == "C"]
    pos0 = frames[0].positions
    d_to_cl = [(i, float(((pos0[i] - pos0[cl_idx]) ** 2).sum() ** 0.5)) for i in c_atoms]
    c_alpha = min(d_to_cl, key=lambda kv: kv[1])[0]
    # β-C is the other one
    c_beta = next(i for i in c_atoms if i != c_alpha)

    # C-Cl distance grows
    d_ccl_first = frames[0].get_distance(c_alpha, cl_idx)
    d_ccl_last = frames[-1].get_distance(c_alpha, cl_idx)
    assert d_ccl_last > d_ccl_first + 0.5, (
        f"C-Cl should grow: {d_ccl_first:.2f} -> {d_ccl_last:.2f}"
    )

    # Some O–H distance should shrink (any H on β-C in initial frame ends up
    # close to O at the end)
    h_atoms = [i for i, s in enumerate(syms) if s == "H"]
    # H bonded to β-C in initial frame: closest H to c_beta
    d_to_cbeta = [(i, frames[0].get_distance(c_beta, i)) for i in h_atoms]
    h_beta = min(d_to_cbeta, key=lambda kv: kv[1])[0]
    d_oh_first = frames[0].get_distance(o_idx, h_beta)
    d_oh_last = frames[-1].get_distance(o_idx, h_beta)
    assert d_oh_last < d_oh_first - 1.0, (
        f"O-H_β should shrink: {d_oh_first:.2f} -> {d_oh_last:.2f}"
    )
```

- [ ] **Step 8: 統合テスト実行 (UMA 必須、~3-5 分)**

Run: `pytest tests/test_re3_e2.py -v -m slow`

Expected: pass。もし `reached_product` が 1 件も True にならない場合は preset を調整 (`k_broken=1.5` に上げる、`r_broken=4.5` に伸ばす、`max_relax_steps=300` に増やす等)。

- [ ] **Step 9: 残テスト regression**

Run: `pytest -m "not slow and not blender" -v`
Expected: 全 pass。test_cli_unimolecular は SN1 dissoc がまだ無いので SKIP のまま。test_cli_neb_refine_guard の e2 case が SKIP から PASS に変わる (e2.rxn 作成済み)。

- [ ] **Step 10: コミット**

```bash
git add examples/e2.rxn tests/test_re3_e2.py tests/conftest.py tests/test_artificial_force.py tests/test_scoring.py
git commit -m "$(cat <<'EOF'
feat(re3): add E2 example + end-to-end integration test

examples/e2.rxn: CH3CH2Cl + OH- -> CH2=CH2 + Cl- + H2O。原子マップに
β-H と base-H を explicit に含めて compute_bond_changes が 1 formed
(O-H) + 2 broken (C-Cl, C-H) を返すようにする。

slow integration: --reaction-type e2 で end-to-end 実行、reached_product
の少なくとも 1 件を assert、C-Cl 距離増加 + O-H_β 距離減少を verify。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: examples/sn1_dissoc.rxn + integration test

**Files:**
- Create: `examples/sn1_dissoc.rxn`
- Create: `tests/test_re3_sn1_dissoc.py`

- [ ] **Step 1: `examples/sn1_dissoc.rxn` を作成**

(CH₃)₃CBr → (CH₃)₃C⁺ + Br⁻。原子マップ:
- map 1 = 中心 C
- map 2, 3, 4 = 3 つのメチル C
- map 5 = Br

```
$RXN

      RDKit

  1  2
$MOL

     RDKit          2D

  5  4  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  1  0  0
    1.5000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  2  0  0
   -0.7500    1.3000    0.0000 C   0  0  0  0  0  0  0  0  0  3  0  0
   -0.7500   -1.3000    0.0000 C   0  0  0  0  0  0  0  0  0  4  0  0
    0.0000    0.0000    1.5000 Br  0  0  0  0  0  0  0  0  0  5  0  0
  1  2  1  0
  1  3  1  0
  1  4  1  0
  1  5  1  0
M  END
$MOL

     RDKit          2D

  4  3  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  1  0  0
    1.5000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  2  0  0
   -0.7500    1.3000    0.0000 C   0  0  0  0  0  0  0  0  0  3  0  0
   -0.7500   -1.3000    0.0000 C   0  0  0  0  0  0  0  0  0  4  0  0
  1  2  1  0
  1  3  1  0
  1  4  1  0
M  CHG  1   1   1
M  END
$MOL

     RDKit          2D

  1  0  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 Br  0  0  0  0  0  0  0  0  0  5  0  0
M  CHG  1   1  -1
M  END
```

- [ ] **Step 2: パース smoke test**

```bash
python -c "
from reactx.rxn_parser import parse_rxn
from reactx.bond_changes import compute_bond_changes
from rdkit import Chem
r, p, m = parse_rxn('examples/sn1_dissoc.rxn')
r_h = Chem.AddHs(r); p_h = Chem.AddHs(p)
bc = compute_bond_changes(r_h, p_h, m)
print('formed:', bc.formed, 'broken:', bc.broken)
"
```

Expected: `formed=()`, `broken=((c_central, br),)` (1 broken bond), `formed=()`。

- [ ] **Step 3: 統合テスト `tests/test_re3_sn1_dissoc.py`**

```python
"""End-to-end SN1 step 1 (heterolytic dissociation) test using UMA. Marked slow."""
import json
from pathlib import Path

import pytest
from ase.io import read

from reactx.cli import main


@pytest.mark.slow
def test_re3_sn1_dissoc_end_to_end(tmp_path: Path, sn1_dissoc_rxn_path: Path):
    out = tmp_path / "sn1d"
    rc = main([
        "run", str(sn1_dissoc_rxn_path), "-o", str(out),
        "--backend", "uma",
        "--reaction-type", "sn1_dissoc",
        "--n-angles", "8",   # will be auto-clamped to 1
    ])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())
    assert len(meta["trials"]) == 1, (
        f"unimolecular auto-clamp expected 1 trial; got {len(meta['trials'])}"
    )
    assert meta["selected_trial"] == 0
    assert meta["effective_params"]["r_form_targets"] == []  # no formed bonds
    assert meta["reaction_type"] == "sn1_dissoc"

    # prescreen should be skipped for unimolecular
    pre = meta["prescreen"]
    assert pre["enabled"] is False or pre["kept"] is None

    frames = read(str(out / "trajectory.xyz"), index=":")
    assert len(frames) >= 3

    syms = frames[0].get_chemical_symbols()
    br_idx = syms.index("Br")
    c_atoms = [i for i, s in enumerate(syms) if s == "C"]
    pos0 = frames[0].positions
    d_to_br = [(i, float(((pos0[i] - pos0[br_idx]) ** 2).sum() ** 0.5)) for i in c_atoms]
    c_central = min(d_to_br, key=lambda kv: kv[1])[0]

    d_first = frames[0].get_distance(c_central, br_idx)
    d_last = frames[-1].get_distance(c_central, br_idx)
    assert d_last > d_first + 1.5, (
        f"C-Br should grow appreciably: {d_first:.2f} -> {d_last:.2f}"
    )
    assert d_last >= 4.5, f"final C-Br should be at least 4.5 A, got {d_last:.2f}"
```

- [ ] **Step 4: 統合テスト実行 (UMA 必須、~1-2 分)**

Run: `pytest tests/test_re3_sn1_dissoc.py -v -m slow`
Expected: pass。C–Br が ≥4.5 Å まで伸びる。

- [ ] **Step 5: 残テスト regression (test_cli_unimolecular もこの段で PASS する)**

Run: `pytest -m "not slow and not blender" -v`
Expected: 全 pass。test_cli_unimolecular の SKIP が解消され PASS。

- [ ] **Step 6: コミット**

```bash
git add examples/sn1_dissoc.rxn tests/test_re3_sn1_dissoc.py
git commit -m "$(cat <<'EOF'
feat(re3): add SN1 step 1 example + unimolecular end-to-end test

(CH3)3CBr -> (CH3)3C+ + Br-. unimolecular なので --n-angles 8 でも
auto-clamp で 1 trial に収まる。slow integration で C-Br 距離が単調
増加し ≥4.5 A まで伸びることを assert。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Blender smoke test (e2) + README updates

**Files:**
- Modify: `tests/test_blender_smoke.py`
- Modify: `README.md`

- [ ] **Step 1: `tests/test_blender_smoke.py` の parametrize に E2 追加**

`test_blender_smoke.py` の `@pytest.mark.parametrize(...)` テーブルに `("e2", "e2.rxn", "e2")` の行を追加 (実装は既存の SN2/PT パターンを踏襲)。

```python
# 既存のように:
@pytest.mark.parametrize(
    "label,rxn_filename,reaction_type",
    [
        ("sn2", "sn2.rxn", "sn2_anion"),
        ("pt", "proton_transfer.rxn", "proton_transfer"),
        ("e2", "e2.rxn", "e2"),  # ← Phase 3 追加
    ],
)
```

(注: 既存ファイルの parametrize 形式を確認してから追記する; カラム名は既存に合わせる)

- [ ] **Step 2: blender smoke 実行 (Blender 必須、ローカル環境のみ)**

Run: `pytest tests/test_blender_smoke.py -v -m blender`
Expected: e2 ケースが pass。`out/<tmp>/e2/scene.blend` を Blender で開いて C–H と C–Cl の同時切断 + base 接近を視認可能。

- [ ] **Step 3: `README.md` を更新**

`README.md` の「Reaction-type presets」表に行を追加:

```
| `e2` | 1.0 | 1.0 | 4.0 | 200 | 元素表 (典型: O–H 0.97 / N–H 1.01) | E2 elimination (例: CH₃CH₂Cl + OH⁻) |
| `sn1_dissoc` | 0.0 | 2.0 | 6.0 | 200 | — (formed=0) | SN1 step 1 解離 (例: (CH₃)₃CBr → t-Bu⁺ + Br⁻) |
```

「Phase Re1 動作確認」セクションのタイトルを「Phase Re1 + Phase 3 動作確認」に変更し、以下を追記:

```
5. `reactx run examples/e2.rxn -o out/e2/ --reaction-type e2 --backend uma --render` を実行
   → `meta.json` で `selected_trial >= 0`, `reached_product=True` の trial が ≥1 件、`out/e2/scene.blend` で C–H と C–Cl の同時切断 + base (OH⁻) 接近を視認
6. `reactx run examples/sn1_dissoc.rxn -o out/sn1d/ --reaction-type sn1_dissoc --backend uma` を実行
   → `meta.json.trials` が 1 件 (unimolecular auto-clamp), `trajectory.xyz` で C–Br 距離が ≥4.5 Å まで伸びる
7. `pytest -m slow` で `test_re3_e2` + `test_re3_sn1_dissoc` + 既存 SN2/PT/Menshutkin が全 pass
```

「Phase Re1 の方針と限界」セクションを「Phase 3 の方針と限界」に書き換え (節は維持しつつ Phase 3 の追加情報を入れる):

```
- 対応反応は形成 0/1 + 切断 1/2 の elementary step (SN2 / proton transfer / Menshutkin / **E2 / SN1 step 1**)。SN1 step 2 / cycloaddition / metathesis / Diels-Alder などは Phase 4+
- `--neb-refine` は 1 formed + 1 broken 反応のみ対応 (E2 / SN1 dissoc では CLI レベルで reject)
- multi-bond NEB endpoint construction は Phase 4+
- 詳細仕様: `docs/superpowers/specs/2026-05-03-phase-3-multibond-design.md`
```

「Wall-clock (実測)」表に行を追加 (実測値は Step 4 で測定):

```
| E2 (`examples/e2.rxn`) | — (新規) | **TBD** | 実測未取得 |
| SN1 dissoc (`examples/sn1_dissoc.rxn`) | — (新規) | **TBD** | unimolecular, n_angles=1 強制 |
```

(注: Step 4 の実測後にこの 2 行の値を埋める)

- [ ] **Step 4: 実測 wall-clock を測定して README を更新**

```bash
time reactx run examples/e2.rxn -o /tmp/e2_bench/ --reaction-type e2 --backend uma
time reactx run examples/sn1_dissoc.rxn -o /tmp/sn1d_bench/ --reaction-type sn1_dissoc --backend uma
```

`meta.json.wall_clock_seconds` を読んで秒単位の値を README の「TBD」欄に書く。

- [ ] **Step 5: 全テスト regression (slow + blender 含む) 最終確認**

```bash
pytest -m "not slow and not blender" -v        # 高速ユニット
pytest -m slow -v                              # SN2/PT/Menshutkin/E2/SN1d 統合
pytest -m blender -v                           # Blender smoke (ローカルのみ)
```

Expected: 全 pass

- [ ] **Step 6: コミット**

```bash
git add tests/test_blender_smoke.py README.md
git commit -m "$(cat <<'EOF'
docs(readme): document Phase 3 multi-bond support (E2 + SN1 step 1)

preset 表に e2 / sn1_dissoc を追加、DoD 手順 を Phase 3 反応 2 種に拡張、
適用限界を Phase 4+ ネクストアクションに更新、wall-clock 表に実測値を追加。
blender smoke の parametrize に e2 を追加。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## 実装後の DoD 確認

1. `pytest -m "not slow and not blender"` 全 pass (~30 sec)
2. `pytest -m slow` で SN2 / PT / Menshutkin / E2 / SN1 dissoc 統合テスト全 pass (~10–15 min)
3. `reactx run examples/e2.rxn -o out/e2/ --reaction-type e2 --backend uma --render` を実行し `out/e2/scene.blend` を Blender で開いて視認
4. `reactx run examples/sn1_dissoc.rxn -o out/sn1d/ --reaction-type sn1_dissoc --backend uma` を実行し `trajectory.xyz` で C–Br 解離を確認
5. `--neb-refine` を E2 .rxn に渡して exit code 2 + エラーメッセージで reject されることを確認
