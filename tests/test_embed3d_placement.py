"""Unit tests for embed3d placement dispatcher (Phase 3)."""
import numpy as np
import pytest
from rdkit import Chem

from reactx.bond_changes import BondChanges
from reactx.embed3d import (
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
    from reactx.embed3d import FRAGMENT_SEPARATION, _place_fragments

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


def test_place_fragments_still_rejects_3frag_or_asymmetric_after_tier3():
    """Tier 3 後も formed=1/broken=1 across-fragment metathesis-like は reject される。

    Phase 4 までは「broken bonds が複数 frag を跨ぐ」全ケースを reject していたが、
    Phase 5 で 2-frag formed=2/broken=2 のみ Tier 3 が拾うようになった。Tier 3 の
    条件を満たさない (formed=1/broken=1) は依然として最終 raise に到達する。
    """
    from reactx.embed3d import _place_fragments
    mol = Chem.AddHs(Chem.MolFromSmiles("CC.OO"))
    frags = Chem.GetMolFrags(mol)
    a = frags[0][0]
    b = frags[1][0]
    # formed=1 + broken=1 (両方 cross-fragment) — Tier 3 は formed=2/broken=2 のみ
    bc = BondChanges(formed=((frags[0][1], frags[1][1]),), broken=((a, b),))
    with pytest.raises(NotImplementedError, match="Phase 5|Phase 6|multi-substrate"):
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
    # formed C-F は substrate-base 橋渡し、broken C-C は substrate intra。
    # frags[2] (Cl-) は formed/broken いずれにも関与しない bystander。
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


def test_embed_in_place_uses_etkdg_when_mmff_fails(caplog, monkeypatch):
    """MMFF94 が parameterize できない fragment (例: LiBr) は ETKDG 結果のみで進行。"""
    import logging

    from rdkit import Chem
    from rdkit.Chem import AllChem

    from reactx.embed3d import _embed_in_place

    # 他のテストの propagate=False 影響を回避
    reactx_logger = logging.getLogger("reactx")
    monkeypatch.setattr(reactx_logger, "propagate", True)

    mol = Chem.AddHs(Chem.MolFromSmiles("[Li]Br"))
    Chem.SanitizeMol(mol)

    caplog.set_level("WARNING", logger="reactx.embed3d")
    _embed_in_place(mol, seed=42)

    # ETKDG conformer が残っているはず
    assert mol.GetNumConformers() == 1
    # MMFF94 警告が出ているはず
    assert any("MMFF94 cannot parameterize" in rec.getMessage() for rec in caplog.records)


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
    from reactx.embed3d import FRAGMENT_SEPARATION, _planar_face_placement

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
    from reactx.embed3d import FRAGMENT_SEPARATION, _planar_face_placement

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


def test_place_fragments_rejects_multi_formed_on_shared_anchor():
    """formed=2 (1 anchor に複数 nucleophile 共有) は dispatcher で Phase 5+ として reject。"""
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
    with pytest.raises(NotImplementedError, match="Phase 5|Phase 6"):
        _place_fragments(
            mol, frags, np.zeros((mol.GetNumAtoms(), 3)),
            bc, rotation_perturbation=None,
        )


def test_place_fragments_rejects_cycloaddition_pattern():
    """formed=2 を 2 fragments の別 anchor で形成する Diels-Alder 型は Phase 5+ として reject。"""
    from reactx.embed3d import _place_fragments

    # CC.CC で formed=((0, 2), (1, 3)) — 2 つの formed bond が別 anchor を介して 2 frag を繋ぐ
    mol = Chem.AddHs(Chem.MolFromSmiles("CC.CC"))
    frags = Chem.GetMolFrags(mol)
    a0, a1 = frags[0][0], frags[0][1]
    b0, b1 = frags[1][0], frags[1][1]
    bc = BondChanges(formed=((a0, b0), (a1, b1)), broken=())
    with pytest.raises(NotImplementedError, match="Phase 5|Phase 6"):
        _place_fragments(
            mol, frags, np.zeros((mol.GetNumAtoms(), 3)),
            bc, rotation_perturbation=None,
        )


def test_kabsch_rigid_transform_recovers_known_rotation_translation():
    """既知の (R, t) を src に適用した dst から、Kabsch が同じ (R, t) を回復する。"""
    from reactx.embed3d import _kabsch_rigid_transform

    rng = np.random.default_rng(42)
    src = rng.normal(size=(4, 3))
    # 既知の回転 (z 軸周り 30°) と translation
    theta = np.deg2rad(30.0)
    R_true = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0,            0.0,           1.0],
    ])
    t_true = np.array([1.5, -2.0, 0.5])
    dst = (R_true @ src.T).T + t_true

    R, t = _kabsch_rigid_transform(src, dst)
    np.testing.assert_allclose(R, R_true, atol=1e-9)
    np.testing.assert_allclose(t, t_true, atol=1e-9)


def test_kabsch_rigid_transform_rejects_reflection():
    """rotoinversion を解にしてしまう対応点でも det(R) >= 0 の rotation を返す。"""
    from reactx.embed3d import _kabsch_rigid_transform

    # 鏡面反転を要求する明示的な対応点 (xy 平面で z 反転を要求)
    src = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    # dst は src の x, y は同じだが z を反転 (= 鏡面反転)
    dst = np.array([
        [1.0, 0.0,  0.0],
        [0.0, 1.0,  0.0],
        [0.0, 0.0, -1.0],
    ])
    R, _t = _kabsch_rigid_transform(src, dst)
    det = float(np.linalg.det(R))
    np.testing.assert_allclose(det, 1.0, atol=1e-9, err_msg=f"expected proper rotation det=+1, got det={det}")


def test_perpendicular_face_dir_normal_case():
    """offset が axis に垂直成分を持つとき、その方向に正規化された unit vector を返す。"""
    from reactx.embed3d import _perpendicular_face_dir
    axis = np.array([1.0, 0.0, 0.0])
    offset = np.array([0.0, 2.0, 0.0])
    perp = _perpendicular_face_dir(axis, offset)
    np.testing.assert_allclose(perp, [0.0, 1.0, 0.0], atol=1e-9)


def test_perpendicular_face_dir_axis_aligned_offset_falls_back_to_z():
    """offset が axis と平行 (垂直成分なし) のとき +z fallback (axis が +x のとき)。"""
    from reactx.embed3d import _perpendicular_face_dir
    axis = np.array([1.0, 0.0, 0.0])
    offset = np.array([2.0, 0.0, 0.0])  # axis と平行
    perp = _perpendicular_face_dir(axis, offset)
    # +z は axis (=+x) と直交するので採用される
    np.testing.assert_allclose(perp, [0.0, 0.0, 1.0], atol=1e-9)


def test_perpendicular_face_dir_axis_z_skips_z_uses_y_fallback():
    """axis が +z のとき、世界基底 +z は使えないので +y にフォールバック。"""
    from reactx.embed3d import _perpendicular_face_dir
    axis = np.array([0.0, 0.0, 1.0])
    offset = np.array([0.0, 0.0, 2.0])  # axis と平行
    perp = _perpendicular_face_dir(axis, offset)
    # +z は axis と平行 → スキップ、+y は axis と直交 → 採用
    np.testing.assert_allclose(perp, [0.0, 1.0, 0.0], atol=1e-9)


def test_perpendicular_face_dir_rejects_zero_axis():
    """norm(axis) < 1e-12 で ValueError を投げる。"""
    from reactx.embed3d import _perpendicular_face_dir
    with pytest.raises(ValueError, match="non-zero"):
        _perpendicular_face_dir(np.zeros(3), np.array([1.0, 0.0, 0.0]))


def test_metathesis_fixture_shape(metathesis_atoms_setup):
    """fixture の形状確認: 2 fragments, formed=2, broken=2, anchor pair on x axis."""
    mol_h, frag_indices, positions, bc = metathesis_atoms_setup
    assert len(frag_indices) == 2
    assert len(bc.formed) == 2
    assert len(bc.broken) == 2
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    c_idx = syms.index("C")
    cl_idx = syms.index("Cl")
    li_idx = syms.index("Li")
    br_idx = syms.index("Br")
    np.testing.assert_allclose(positions[cl_idx], [0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(positions[c_idx], [1.78, 0.0, 0.0], atol=1e-9)
    # broken bonds 各 fragment 内に閉じる
    cccl = next(i for i, b in enumerate(bc.broken) if c_idx in b and cl_idx in b)
    libr = next(i for i, b in enumerate(bc.broken) if li_idx in b and br_idx in b)
    assert cccl != libr  # 別々の broken bond


def test_kabsch_alignment_respects_rotation_perturbation(metathesis_atoms_setup):
    """rotation_perturbation で moving fragment が回転される (相対距離は不変)。"""
    from reactx.embed3d import _kabsch_alignment

    mol_h, frag_indices, positions, bc = metathesis_atoms_setup
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    li_idx = syms.index("Li")
    br_idx = syms.index("Br")

    out_id = _kabsch_alignment(
        mol_h, frag_indices, positions.copy(), bc,
        rotation_perturbation=None,
    )

    # 90° rotation around z axis (incoming axis Br-Li is along x in this fixture,
    # so x rotation would not move the atoms — must rotate around y or z).
    R = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ])
    out_rot = _kabsch_alignment(
        mol_h, frag_indices, positions.copy(), bc,
        rotation_perturbation=R,
    )

    # Br と Li の絶対位置が変わる
    assert not np.allclose(out_id[br_idx], out_rot[br_idx], atol=0.1), (
        f"rotation_perturbation should change Br position; "
        f"identity={out_id[br_idx]}, rotated={out_rot[br_idx]}"
    )
    # 相対距離 (Br-Li) は剛体変換で不変
    d_id = float(np.linalg.norm(out_id[br_idx] - out_id[li_idx]))
    d_rot = float(np.linalg.norm(out_rot[br_idx] - out_rot[li_idx]))
    np.testing.assert_allclose(d_id, d_rot, atol=0.05)


def test_kabsch_alignment_creates_4center_geometry(metathesis_atoms_setup):
    """Tier 3 主路: CH3Cl + LiBr fixture で 4-center geometry を達成。

    Assertions (spec §7 (q)):
      - C-Br ≤ 2.5 Å (formed bond)
      - Li-Cl ≤ 2.5 Å (formed bond)
      - anchor 軸 (C-Cl) と incoming 軸 (Br-Li) が概並行 (cos angle > 0.7)
      - moving 重心が anchor 軸から FRAGMENT_SEPARATION/2 ± 0.5 Å 離れる
    """
    from reactx.embed3d import FRAGMENT_SEPARATION, _kabsch_alignment

    mol_h, frag_indices, positions, bc = metathesis_atoms_setup
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    c_idx = syms.index("C")
    cl_idx = syms.index("Cl")
    li_idx = syms.index("Li")
    br_idx = syms.index("Br")

    positions_in = positions.copy()

    out = _kabsch_alignment(
        mol_h, frag_indices, positions_in.copy(), bc,
        rotation_perturbation=None,
    )

    # reference fragment (CH3Cl) is fixed
    np.testing.assert_array_equal(out[c_idx], positions_in[c_idx])
    np.testing.assert_array_equal(out[cl_idx], positions_in[cl_idx])

    d_c_br = float(np.linalg.norm(out[c_idx] - out[br_idx]))
    d_li_cl = float(np.linalg.norm(out[li_idx] - out[cl_idx]))
    assert d_c_br <= 2.5, f"C-Br should be <=2.5 A (formed bond), got {d_c_br:.2f}"
    assert d_li_cl <= 2.5, f"Li-Cl should be <=2.5 A (formed bond), got {d_li_cl:.2f}"

    axis_anchor = out[c_idx] - out[cl_idx]
    axis_anchor /= np.linalg.norm(axis_anchor)
    axis_incoming = out[br_idx] - out[li_idx]
    axis_incoming /= np.linalg.norm(axis_incoming)
    cos_angle = abs(float(np.dot(axis_anchor, axis_incoming)))
    assert cos_angle > 0.7, f"anchor and incoming axes should be ~parallel, got cos={cos_angle:.3f}"

    moving_frag = next(f for f in frag_indices if li_idx in f)
    moving_centroid = out[list(moving_frag)].mean(axis=0)
    anchor_midpoint = (out[c_idx] + out[cl_idx]) / 2
    perp_dist = float(np.linalg.norm(
        (moving_centroid - anchor_midpoint)
        - np.dot(moving_centroid - anchor_midpoint, axis_anchor) * axis_anchor
    ))
    expected = FRAGMENT_SEPARATION / 2
    assert abs(perp_dist - expected) <= 0.1, (
        f"moving centroid should be {expected:.2f} A above anchor axis (perp), got {perp_dist:.2f}"
    )


def test_kabsch_alignment_rejects_broken_spanning_fragments():
    """broken bonds が両 fragment を跨ぐ (= 純粋 metathesis 以外) で NotImplementedError。"""
    from reactx.embed3d import _kabsch_alignment

    # 2 fragments で broken bond が両 frag を跨ぐ (cross-fragment broken)
    mol = Chem.AddHs(Chem.MolFromSmiles("CC.OO"))
    frags = Chem.GetMolFrags(mol)
    a = frags[0][0]  # 左 fragment の C
    b = frags[1][0]  # 右 fragment の O
    # broken=2 だが両方 cross-fragment → broken_within_* がいずれも 0
    bc = BondChanges(
        formed=((frags[0][1], frags[1][1]), (a, b)),
        broken=((a, b), (frags[0][1], frags[1][1])),
    )
    with pytest.raises(NotImplementedError, match="1\\+1 within-fragment"):
        _kabsch_alignment(
            mol, frags, np.zeros((mol.GetNumAtoms(), 3)),
            bc, rotation_perturbation=None,
        )


def test_kabsch_alignment_rejects_multi_bond_from_single_anchor(metathesis_atoms_setup):
    """同 anchor から 2 formed bond で NotImplementedError。"""
    from reactx.embed3d import _kabsch_alignment

    mol_h, frag_indices, positions, bc = metathesis_atoms_setup
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    c_idx = syms.index("C")
    cl_idx = syms.index("Cl")
    li_idx = syms.index("Li")
    br_idx = syms.index("Br")
    # 両 formed bond の reference 端を C にする (anchor=C 共有)
    bad_bc = BondChanges(
        formed=((c_idx, br_idx), (c_idx, li_idx)),
        broken=((c_idx, cl_idx), (li_idx, br_idx)),
    )
    with pytest.raises(NotImplementedError, match="multi-bond from single anchor"):
        _kabsch_alignment(
            mol_h, frag_indices, positions.copy(), bad_bc,
            rotation_perturbation=None,
        )


def test_place_fragments_dispatches_tier3_for_metathesis(metathesis_atoms_setup):
    """Tier 3 dispatch: 2-fragment formed=2/broken=2/multi-substrate で _kabsch_alignment に流れる。"""
    from reactx.embed3d import _place_fragments

    mol_h, frag_indices, positions, bc = metathesis_atoms_setup
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    c_idx = syms.index("C")
    br_idx = syms.index("Br")

    out = _place_fragments(
        mol_h, frag_indices, positions.copy(), bc,
        rotation_perturbation=None,
    )
    # Tier 3 を経由して 4-center 配置になっていれば C-Br が中程度の距離
    d_c_br = float(np.linalg.norm(out[c_idx] - out[br_idx]))
    assert 0.5 < d_c_br < 3.0, (
        f"expected Tier 3 placement to put Br near C (0.5-3.0 A), got {d_c_br:.2f}"
    )


def test_place_fragments_rejects_3_fragment_metathesis():
    """3 fragments + broken bond が fragments を跨ぐ場合は Phase 6+ reject。

    Tier 1 dispatch には乗らず (substrate=None: broken が単一 fragment に閉じない)、
    Tier 3 にも乗らない (frag_count==2 が条件) ので最終 raise に到達。
    """
    from reactx.embed3d import _place_fragments

    # 3 fragments: CC, OO, NN。broken bond が CC ↔ OO を跨ぐ → substrate=None。
    # 3 fragments なので Tier 3 dispatch (frag_count==2) も満たさず final raise。
    mol = Chem.AddHs(Chem.MolFromSmiles("CC.OO.NN"))
    frags = Chem.GetMolFrags(mol)
    c0 = frags[0][0]
    o0 = frags[1][0]
    n0 = frags[2][0]
    bc = BondChanges(
        formed=((c0, n0),),
        broken=((c0, o0),),
    )
    with pytest.raises(NotImplementedError, match="Phase 5|Phase 6|multi-substrate"):
        _place_fragments(
            mol, frags, np.zeros((mol.GetNumAtoms(), 3)),
            bc, rotation_perturbation=None,
        )


def test_place_fragments_rejects_asymmetric_metathesis():
    """非対称 metathesis (formed=2, broken=1, 両 fragment 跨ぎ) で Phase 6+ reject。"""
    from reactx.embed3d import _place_fragments

    # 2 fragments で broken=1 が両 frag を跨ぐ (substrate=None かつ broken count != formed count)
    mol = Chem.AddHs(Chem.MolFromSmiles("CC.OO"))
    frags = Chem.GetMolFrags(mol)
    a = frags[0][0]
    b = frags[1][0]
    bc = BondChanges(
        formed=((frags[0][1], frags[1][1]), (a, b)),
        broken=((a, b),),
    )
    with pytest.raises(NotImplementedError, match="Phase 5|Phase 6"):
        _place_fragments(
            mol, frags, np.zeros((mol.GetNumAtoms(), 3)),
            bc, rotation_perturbation=None,
        )


def test_kabsch_alignment_raises_when_anchor_pair_coincident(metathesis_atoms_setup):
    """anchor_a と anchor_b が同じ位置に来た場合 RuntimeError を投げる
    (= broken_within_reference の bond が既に切れている異常状態)。"""
    from reactx.embed3d import _kabsch_alignment

    mol_h, frag_indices, positions, bc = metathesis_atoms_setup
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    c_idx = syms.index("C")
    cl_idx = syms.index("Cl")
    # C と Cl を同じ位置に重ねる
    bad_positions = positions.copy()
    bad_positions[cl_idx] = bad_positions[c_idx]

    with pytest.raises(RuntimeError, match="anchor pair coincident"):
        _kabsch_alignment(
            mol_h, frag_indices, bad_positions, bc,
            rotation_perturbation=None,
        )
