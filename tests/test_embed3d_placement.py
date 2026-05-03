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
    mol, frags = _make_mol_with_frags("CC(Cl).[OH-]")
    substrate = frags[0]
    other = frags[1]
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
    found = _find_substrate_fragment(frags, ((a, b),))
    assert found is None


def test_place_fragments_dispatch_directional_for_sn2_shape(sn2_atoms_setup):
    mol_h, frag_indices, positions, bond_changes = sn2_atoms_setup
    out = _place_fragments(
        mol_h, frag_indices, positions.copy(), bond_changes,
        rotation_perturbation=None,
    )
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    o_idx = syms.index("O")
    c_idx = syms.index("C")
    cl_idx = syms.index("Cl")
    v_co = out[o_idx] - out[c_idx]
    v_ccl = out[cl_idx] - out[c_idx]
    cos_angle = float(np.dot(v_co, v_ccl) / (
        np.linalg.norm(v_co) * np.linalg.norm(v_ccl) + 1e-12
    ))
    assert cos_angle < -0.5, f"expected backside placement, got cos(angle)={cos_angle}"


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


def test_place_fragments_dispatches_tier2_for_broken_zero_bimolecular(sn1_recomb_atoms_setup):
    """Tier 2 dispatch: broken=() の bimolecular で _planar_face_placement に流す。

    Phase 3 では NotImplementedError を投げていたが、Phase 4 で実装したので
    placement が成功し、Cl の位置が plane normal 方向に動くことを確認する。
    """
    from reactx.embed3d import _place_fragments, FRAGMENT_SEPARATION

    mol_h, frag_indices, positions, bc = sn1_recomb_atoms_setup
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    cl_idx = syms.index("Cl")
    central = next(
        i for i, a in enumerate(mol_h.GetAtoms())
        if a.GetSymbol() == "C" and a.GetFormalCharge() == 1
    )

    out = _place_fragments(
        mol_h, frag_indices, positions.copy(), bc,
        rotation_perturbation=None,
    )
    d = float(np.linalg.norm(out[cl_idx] - out[central]))
    np.testing.assert_allclose(d, FRAGMENT_SEPARATION, atol=0.5)


def test_place_fragments_still_raises_for_multi_substrate_metathesis_after_tier2():
    """multi-substrate metathesis (broken bonds が複数 frag に跨る) は Tier 2 でも reject。"""
    from reactx.embed3d import _place_fragments
    mol = Chem.AddHs(Chem.MolFromSmiles("CC.OO"))
    frags = Chem.GetMolFrags(mol)
    a = frags[0][0]
    b = frags[1][0]
    bc = BondChanges(formed=((frags[0][1], frags[1][1]),), broken=((a, b),))
    with pytest.raises(NotImplementedError, match="multi-substrate"):
        _place_fragments(
            mol, frags, np.zeros((mol.GetNumAtoms(), 3)),
            bc, rotation_perturbation=None,
        )


def test_place_fragments_raises_for_multi_substrate_metathesis():
    mol = Chem.AddHs(Chem.MolFromSmiles("CC.OO"))
    frags = Chem.GetMolFrags(mol)
    a = frags[0][0]
    b = frags[1][0]
    bc = BondChanges(formed=((frags[0][1], frags[1][1]),), broken=((a, b),))
    with pytest.raises(NotImplementedError, match="multi-substrate|centroid"):
        _place_fragments(
            mol, frags, np.zeros((mol.GetNumAtoms(), 3)),
            bc, rotation_perturbation=None,
        )


def test_place_fragments_raises_for_multi_base_on_single_anchor():
    mol = Chem.AddHs(Chem.MolFromSmiles("CC.[F-].[Cl-]"))
    frags = Chem.GetMolFrags(mol)
    c0, c1 = frags[0][0], frags[0][1]
    f_idx = frags[1][0]
    cl_idx = frags[2][0]
    bc = BondChanges(
        formed=((c0, f_idx), (c0, cl_idx)),
        broken=((c0, c1),),
    )
    with pytest.raises(NotImplementedError, match="multi-base"):
        _place_fragments(
            mol, frags, np.zeros((mol.GetNumAtoms(), 3)),
            bc, rotation_perturbation=None,
        )


def test_place_fragments_bystander_fragment_raises():
    """Non-substrate fragment が formed bond で substrate と橋渡しされていない場合は ValueError。

    例: 3 fragment 系で、substrate (broken intra) + 連結された base + bystander (どの bond にも関与しない)。
    """
    mol = Chem.AddHs(Chem.MolFromSmiles("CC.[F-].[Cl-]"))
    frags = Chem.GetMolFrags(mol)
    c0, c1 = frags[0][0], frags[0][1]
    f_idx = frags[1][0]
    cl_idx = frags[2][0]
    # formed C-F は substrate-base 橋渡し、broken C-C は substrate intra。
    # Cl- は formed/broken いずれにも関与しない bystander。
    bc = BondChanges(
        formed=((c0, f_idx),),
        broken=((c0, c1),),
    )
    with pytest.raises(ValueError, match="bridging"):
        _place_fragments(
            mol, frags, np.zeros((mol.GetNumAtoms(), 3)),
            bc, rotation_perturbation=None,
        )


def test_embed_mol_to_atoms_unimolecular_skips_dispatcher():
    """unimolecular (1 fragment) では _place_fragments が呼ばれず、bond_changes も不要。"""
    from reactx.embed3d import embed_mol_to_atoms
    mol = Chem.MolFromSmiles("CCO")  # 1 fragment, no bond_changes needed
    atoms = embed_mol_to_atoms(mol, calculator=None, seed=42)
    # Embedded successfully without bond_changes (which would be required for multi-fragment).
    assert len(atoms) > 0


def test_find_substrate_by_size_picks_larger_heavy_count():
    """Tier 2: 重原子数最大の fragment を substrate として返す。"""
    from reactx.embed3d import _find_substrate_by_size
    mol, frags = _make_mol_with_frags("[C+](C)(C)C.[Cl-]")
    found = _find_substrate_by_size(frags)
    # tBu+ fragment は 4 heavy (1 C+ + 3 methyl C), Cl- は 1 heavy。
    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    heavy_in_found = sum(1 for i in found if syms[i] != "H")
    assert heavy_in_found == 4


def test_find_substrate_by_size_tie_break_smallest_atom_index():
    """同じ heavy 数の場合、最小 atom index を含む fragment を選ぶ。"""
    from reactx.embed3d import _find_substrate_by_size
    # Cl- (heavy=1) と F- (heavy=1) の tie。frags[0] が smaller idx なので選ばれる。
    mol, frags = _make_mol_with_frags("[Cl-].[F-]")
    found = _find_substrate_by_size(frags)
    assert found == frags[0], "tie-break should pick fragment with smallest atom index"


def test_plane_normal_at_anchor_planar_three_neighbors():
    """3 substrate 隣接が xy 平面に乗っている場合、法線は ±z 方向 (符号は +z 寄り)。"""
    from reactx.embed3d import _plane_normal_at_anchor

    mol = Chem.AddHs(Chem.MolFromSmiles("[C+](C)(C)C"))
    central = next(
        i for i, a in enumerate(mol.GetAtoms())
        if a.GetSymbol() == "C" and a.GetFormalCharge() == 1
    )
    methyl_carbons = [
        n.GetIdx() for n in mol.GetAtomWithIdx(central).GetNeighbors()
        if n.GetSymbol() == "C"
    ]
    assert len(methyl_carbons) == 3, f"expected 3 methyl C neighbors, got {len(methyl_carbons)}"

    n = mol.GetNumAtoms()
    positions = np.zeros((n, 3))
    positions[central] = (0.0, 0.0, 0.0)
    # 3 methyl C を xy 平面の三角形に置く
    for k, m in enumerate(methyl_carbons):
        theta = 2 * np.pi * k / 3
        positions[m] = (np.cos(theta) * 1.5, np.sin(theta) * 1.5, 0.0)

    substrate = tuple(range(n))  # 全 atom が同じ fragment
    direction = _plane_normal_at_anchor(positions, central, mol, substrate)

    assert direction.shape == (3,)
    np.testing.assert_allclose(np.linalg.norm(direction), 1.0, atol=1e-6)
    # 法線は ±z で、符号は +z 寄り
    assert direction[2] > 0, f"sign disambiguation failed: direction={direction}"
    assert abs(direction[0]) < 1e-3 and abs(direction[1]) < 1e-3, (
        f"direction should be along z, got {direction}"
    )


def test_plane_normal_at_anchor_sign_flip_when_svd_returns_minus_z():
    """SVD が -z を返す配置でも sign disambiguation で +z 寄りに統一される。"""
    from reactx.embed3d import _plane_normal_at_anchor

    mol = Chem.AddHs(Chem.MolFromSmiles("[C+](C)(C)C"))
    central = next(
        i for i, a in enumerate(mol.GetAtoms())
        if a.GetSymbol() == "C" and a.GetFormalCharge() == 1
    )
    methyl_carbons = [
        n.GetIdx() for n in mol.GetAtomWithIdx(central).GetNeighbors()
        if n.GetSymbol() == "C"
    ]

    n = mol.GetNumAtoms()
    positions = np.zeros((n, 3))
    positions[central] = (0.0, 0.0, 0.0)
    # 3 methyl C を、わずかに -z へ傾けた xy 平面の三角形に置く。
    # こうすると SVD の最小特異値方向が ±z 近傍になり、実装によっては -z を返す。
    # sign-flip 後は必ず +z 寄りであることを確認する。
    for k, m in enumerate(methyl_carbons):
        theta = 2 * np.pi * k / 3
        positions[m] = (np.cos(theta) * 1.5, np.sin(theta) * 1.5, -0.01)

    substrate = tuple(range(n))
    direction = _plane_normal_at_anchor(positions, central, mol, substrate)

    # disambiguation 後は +z 寄り (direction[2] >= 0)。
    assert direction[2] >= 0, (
        f"sign disambiguation should force +z hemisphere, got {direction}"
    )
    np.testing.assert_allclose(np.linalg.norm(direction), 1.0, atol=1e-6)


def test_plane_normal_at_anchor_two_neighbors_falls_back_to_anti_mean():
    """隣接 2 個の場合、direction = -unit(mean_neighbor - anchor)。"""
    from reactx.embed3d import _plane_normal_at_anchor

    # CH2=CH+ (vinyl cation): 中心 C+ に C 隣接 1 + H 隣接 1
    mol = Chem.AddHs(Chem.MolFromSmiles("[CH+]=C"))
    central = next(
        i for i, a in enumerate(mol.GetAtoms())
        if a.GetSymbol() == "C" and a.GetFormalCharge() == 1
    )
    n = mol.GetNumAtoms()
    positions = np.zeros((n, 3))
    positions[central] = (0.0, 0.0, 0.0)
    # 2 substrate 隣接 (C と H) を +x 方向に置く (mean が +x)
    neighbors = [
        nb.GetIdx() for nb in mol.GetAtomWithIdx(central).GetNeighbors()
    ]
    assert len(neighbors) == 2, f"expected 2 neighbors, got {len(neighbors)}"
    positions[neighbors[0]] = (1.5, 0.5, 0.0)
    positions[neighbors[1]] = (1.5, -0.5, 0.0)

    substrate = tuple(range(n))
    direction = _plane_normal_at_anchor(positions, central, mol, substrate)

    np.testing.assert_allclose(np.linalg.norm(direction), 1.0, atol=1e-6)
    # mean_neighbor = (1.5, 0.0, 0.0), -unit = (-1.0, 0.0, 0.0)
    np.testing.assert_allclose(direction, [-1.0, 0.0, 0.0], atol=1e-3)


def test_plane_normal_at_anchor_non_planar_three_neighbors_falls_back():
    """3 隣接でも平面 fit 残差が大きい (sp³-like) 場合は fallback。"""
    from reactx.embed3d import _plane_normal_at_anchor

    mol = Chem.AddHs(Chem.MolFromSmiles("[C+](C)(C)C"))
    central = next(
        i for i, a in enumerate(mol.GetAtoms())
        if a.GetSymbol() == "C" and a.GetFormalCharge() == 1
    )
    methyl_carbons = [
        nb.GetIdx() for nb in mol.GetAtomWithIdx(central).GetNeighbors()
        if nb.GetSymbol() == "C"
    ]
    n = mol.GetNumAtoms()
    positions = np.zeros((n, 3))
    positions[central] = (0.0, 0.0, 0.0)
    # 3 methyl C を sp3 風に配置 (z 方向に大きな散らばり; 平面 fit 残差が大きい)
    sp3_dirs = np.array([
        [1.0, 0.0, 1.0],
        [-0.5, 0.866, 1.0],
        [-0.5, -0.866, 1.0],
    ])
    sp3_dirs /= np.linalg.norm(sp3_dirs, axis=1, keepdims=True)
    for m, d in zip(methyl_carbons, sp3_dirs, strict=True):
        positions[m] = d * 1.5

    substrate = tuple(range(n))
    direction = _plane_normal_at_anchor(positions, central, mol, substrate)

    np.testing.assert_allclose(np.linalg.norm(direction), 1.0, atol=1e-6)
    # mean of 3 sp3 dirs is in +z direction → -unit(mean) is -z direction
    assert direction[2] < 0, f"non-planar fallback should point -z, got {direction}"


def test_plane_normal_at_anchor_degenerate_uses_z_fallback(caplog, monkeypatch):
    """隣接 0 個の場合 (anchor が単独 atom) は [0, 0, 1] + warning。"""
    import logging
    from reactx.embed3d import _plane_normal_at_anchor

    # 他のテスト (cli.py の _configure_reactx_logging) で reactx logger の
    # propagate=False がセットされていると caplog が拾えないため、明示的に
    # propagation を有効化する。test 終了時に monkeypatch が元に戻す。
    reactx_logger = logging.getLogger("reactx")
    monkeypatch.setattr(reactx_logger, "propagate", True)

    # Cl- 単独: 隣接 0
    mol = Chem.AddHs(Chem.MolFromSmiles("[Cl-]"))
    n = mol.GetNumAtoms()
    positions = np.zeros((n, 3))
    substrate = tuple(range(n))

    caplog.set_level("WARNING", logger="reactx.embed3d")
    direction = _plane_normal_at_anchor(positions, 0, mol, substrate)

    np.testing.assert_allclose(direction, [0.0, 0.0, 1.0], atol=1e-9)
    assert any("plane-normal" in rec.getMessage() for rec in caplog.records), (
        "expected a warning log for degenerate anchor"
    )


def test_sn1_recomb_fixture_shape(sn1_recomb_atoms_setup):
    """fixture の形状確認: 2 fragments, formed=1, broken=0, tBu+ planar at origin."""
    mol_h, frag_indices, positions, bc = sn1_recomb_atoms_setup
    assert len(frag_indices) == 2
    assert len(bc.formed) == 1
    assert len(bc.broken) == 0
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    central = next(
        i for i, a in enumerate(mol_h.GetAtoms())
        if a.GetSymbol() == "C" and a.GetFormalCharge() == 1
    )
    np.testing.assert_allclose(positions[central], [0.0, 0.0, 0.0], atol=1e-9)
    cl_idx = syms.index("Cl")
    assert (central, cl_idx) == bc.formed[0] or (cl_idx, central) == bc.formed[0]


def test_planar_face_placement_places_cl_along_plane_normal(sn1_recomb_atoms_setup):
    """Tier 2: Cl の最終位置が anchor + plane_normal * FRAGMENT_SEPARATION。"""
    from reactx.embed3d import _planar_face_placement, FRAGMENT_SEPARATION

    mol_h, frag_indices, positions, bc = sn1_recomb_atoms_setup
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    central = next(
        i for i, a in enumerate(mol_h.GetAtoms())
        if a.GetSymbol() == "C" and a.GetFormalCharge() == 1
    )
    cl_idx = syms.index("Cl")

    substrate = next(f for f in frag_indices if central in f)
    out = _planar_face_placement(
        mol_h, frag_indices, positions.copy(), bc, substrate,
        rotation_perturbation=None,
    )

    # tBu+ は xy 平面、anchor=central=(0,0,0), plane normal は ±z (sign +z)。
    # → target = (0, 0, FRAGMENT_SEPARATION) = (0, 0, 3.5)
    expected = np.array([0.0, 0.0, FRAGMENT_SEPARATION])
    actual = out[cl_idx]
    np.testing.assert_allclose(actual, expected, atol=0.5)


def test_planar_face_placement_respects_rotation_perturbation(sn1_recomb_atoms_setup):
    """rotation_perturbation で Cl の位置が回転されることを確認。"""
    from reactx.embed3d import _planar_face_placement, FRAGMENT_SEPARATION

    mol_h, frag_indices, positions, bc = sn1_recomb_atoms_setup
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    central = next(
        i for i, a in enumerate(mol_h.GetAtoms())
        if a.GetSymbol() == "C" and a.GetFormalCharge() == 1
    )
    cl_idx = syms.index("Cl")
    substrate = next(f for f in frag_indices if central in f)

    out_id = _planar_face_placement(
        mol_h, frag_indices, positions.copy(), bc, substrate,
        rotation_perturbation=None,
    )

    # 90° rotation around x axis: z → y
    R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    out_rot = _planar_face_placement(
        mol_h, frag_indices, positions.copy(), bc, substrate,
        rotation_perturbation=R,
    )

    # Cl が異なる位置に置かれること
    assert not np.allclose(out_id[cl_idx], out_rot[cl_idx], atol=0.1), (
        f"rotation_perturbation should change Cl position; "
        f"identity={out_id[cl_idx]}, rotated={out_rot[cl_idx]}"
    )
    # 距離は同じ (回転は等距変換)
    d_id = float(np.linalg.norm(out_id[cl_idx] - out_id[central]))
    d_rot = float(np.linalg.norm(out_rot[cl_idx] - out_rot[central]))
    np.testing.assert_allclose(d_id, d_rot, atol=0.1)
    np.testing.assert_allclose(d_rot, FRAGMENT_SEPARATION, atol=0.5)


def test_planar_face_placement_rejects_multi_base_on_single_anchor():
    """Tier 2 で 1 anchor に複数 nucleophile が共有する場合は NotImplementedError。"""
    from reactx.embed3d import _place_fragments

    # [C+](C)(C)C.[F-].[Cl-] で formed=((C+, F), (C+, Cl)) — 2 base が同じ anchor 共有
    mol = Chem.AddHs(Chem.MolFromSmiles("[C+](C)(C)C.[F-].[Cl-]"))
    frags = Chem.GetMolFrags(mol)
    central = next(
        i for i, a in enumerate(mol.GetAtoms())
        if a.GetSymbol() == "C" and a.GetFormalCharge() == 1
    )
    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    f_idx = syms.index("F")
    cl_idx = syms.index("Cl")
    bc = BondChanges(
        formed=((central, f_idx), (central, cl_idx)),
        broken=(),
    )
    with pytest.raises(NotImplementedError, match="multi-base"):
        _place_fragments(
            mol, frags, np.zeros((mol.GetNumAtoms(), 3)),
            bc, rotation_perturbation=None,
        )
