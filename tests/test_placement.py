"""Unit tests for reactx.placement (Phase 7)."""
import numpy as np
import pytest
from ase import Atoms
from rdkit import Chem

from reactx.bond_changes import BondChanges
from reactx.placement import (
    PlacementResult,
    _find_bridging_formed,
    _identify_substrate,
    build_atoms_from_positions,
    compute_d_min,
    evaluate_direction,
    sample_sphere_directions,
    valid_placements,
)


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
    bridges = _find_bridging_formed(formed, sub, frag)
    assert bridges == [(0, 5)]


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


def test_valid_placements_termolecular_not_implemented():
    # 3 fragments (CH3Cl + 2 nucleophiles) → 1 substrate + 2 non-substrate
    # → out of scope for Phase 7.
    mol = Chem.MolFromSmiles("CCl.[OH-].[F-]")
    mol_h = Chem.AddHs(mol)
    frags = Chem.GetMolFrags(mol_h)
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    c_idx = syms.index("C")
    cl_idx = syms.index("Cl")
    o_idx = syms.index("O")
    f_idx = syms.index("F")
    # formed: both OH- and F- attempt to attack C; broken: C-Cl
    bc = BondChanges(formed=((c_idx, o_idx), (c_idx, f_idx)), broken=((c_idx, cl_idx),))
    n = mol_h.GetNumAtoms()
    positions = np.zeros((n, 3))
    with pytest.raises(NotImplementedError, match="termolecular"):
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


def test_placement_trial_default_orientation_is_single():
    import numpy as np

    from reactx.placement import PlacementTrial
    t = PlacementTrial(
        direction=np.array([0.0, 0.0, 1.0]),
        d_min=0.0,
        positions=np.zeros((3, 3)),
    )
    assert t.orientation == "single"


def test_placement_trial_orientation_can_be_set_to_endo_exo_achiral():
    import numpy as np

    from reactx.placement import PlacementTrial
    for label in ("single", "endo", "exo", "achiral"):
        t = PlacementTrial(
            direction=np.array([0.0, 0.0, 1.0]),
            d_min=0.0,
            positions=np.zeros((3, 3)),
            orientation=label,
        )
        assert t.orientation == label


def test_placement_result_placement_kind_default_single_anchor():
    from reactx.placement import PlacementResult
    r = PlacementResult(
        trials=[],
        n_candidates=0,
        n_blocked=0,
        blocked_reasons=[],
    )
    assert r.placement_kind == "single_anchor"


def test_placement_result_kind_multi_anchor_accepted():
    from reactx.placement import PlacementResult
    r = PlacementResult(
        trials=[],
        n_candidates=0,
        n_blocked=0,
        blocked_reasons=[],
        placement_kind="multi_anchor",
    )
    assert r.placement_kind == "multi_anchor"


def test_valid_placements_unimolecular_returns_single_anchor_kind():
    """unimolecular passthrough は placement_kind='single_anchor'、orientation='single'。"""
    import numpy as np
    from rdkit import Chem

    from reactx.bond_changes import BondChanges
    from reactx.placement import valid_placements

    mol = Chem.MolFromSmiles("C")
    mol = Chem.AddHs(mol)
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
    ], dtype=float)
    frag_indices = ((0, 1, 2, 3, 4),)
    # Use intramolecular formed bond — unimolecular passthrough doesn't require bridging
    bond_changes = BondChanges(formed=((0, 1),), broken=())

    result = valid_placements(mol, frag_indices, positions, bond_changes, n_candidates=1)
    assert result.placement_kind == "single_anchor"
    assert len(result.trials) == 1
    assert result.trials[0].orientation == "single"


def test_valid_placements_existing_sn2_path_returns_single_anchor_kind():
    """既存 SN2 path (bridges == 1, bimolecular) も placement_kind='single_anchor' を返す。"""
    from rdkit import Chem

    from reactx.bond_changes import BondChanges
    from reactx.embed3d import embed_fragments_to_positions
    from reactx.placement import valid_placements

    mol = Chem.MolFromSmiles("[O-].CCl")
    mol_h, frag_indices, positions = embed_fragments_to_positions(mol, seed=0)
    bond_changes = BondChanges(formed=((0, 1),), broken=((1, 2),))
    result = valid_placements(mol_h, frag_indices, positions, bond_changes, n_candidates=8, seed=0)
    assert result.placement_kind == "single_anchor"
    assert all(t.orientation == "single" for t in result.trials)


def test_rotate_atoms_around_z_axis_90deg():
    import numpy as np

    from reactx.placement import _rotate_atoms
    positions = np.array([
        [1.0, 0.0, 0.0],   # to be rotated
        [0.0, 0.0, 0.0],   # at center
        [5.0, 5.0, 5.0],   # NOT in indices
    ])
    indices = (0, 1)
    new_pos = _rotate_atoms(
        positions, indices=indices,
        axis=np.array([0.0, 0.0, 1.0]),
        center=np.array([0.0, 0.0, 0.0]),
        angle=np.pi / 2,
    )
    np.testing.assert_allclose(new_pos[0], [0.0, 1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(new_pos[1], [0.0, 0.0, 0.0], atol=1e-9)
    # idx 2 not in indices: unchanged
    np.testing.assert_allclose(new_pos[2], [5.0, 5.0, 5.0])


def test_rotate_atoms_180deg_inverts_perpendicular_component():
    import numpy as np

    from reactx.placement import _rotate_atoms
    positions = np.array([[1.0, 0.0, 0.0]])
    new_pos = _rotate_atoms(
        positions, indices=(0,),
        axis=np.array([0.0, 0.0, 1.0]),
        center=np.zeros(3),
        angle=np.pi,
    )
    np.testing.assert_allclose(new_pos[0], [-1.0, 0.0, 0.0], atol=1e-9)


def test_rotate_atoms_zero_angle_identity():
    import numpy as np

    from reactx.placement import _rotate_atoms
    positions = np.random.default_rng(0).standard_normal((4, 3))
    new_pos = _rotate_atoms(
        positions, indices=(0, 1, 2, 3),
        axis=np.array([1.0, 0.0, 0.0]),
        center=np.zeros(3),
        angle=0.0,
    )
    np.testing.assert_allclose(new_pos, positions, atol=1e-9)


def test_rotate_atoms_off_axis_center():
    """center != origin の場合の検証。"""
    import numpy as np

    from reactx.placement import _rotate_atoms
    # rotate point (2, 0, 0) around z-axis at center (1, 0, 0) by 90°
    # → original offset from center = (1, 0, 0) → rotated = (0, 1, 0) → final = (1, 1, 0)
    positions = np.array([[2.0, 0.0, 0.0]])
    new_pos = _rotate_atoms(
        positions, indices=(0,),
        axis=np.array([0.0, 0.0, 1.0]),
        center=np.array([1.0, 0.0, 0.0]),
        angle=np.pi / 2,
    )
    np.testing.assert_allclose(new_pos[0], [1.0, 1.0, 0.0], atol=1e-9)


def test_rotate_atoms_zero_axis_raises():
    """zero-length axis should raise ValueError."""
    import numpy as np
    import pytest

    from reactx.placement import _rotate_atoms
    with pytest.raises(ValueError, match="zero length"):
        _rotate_atoms(
            np.zeros((1, 3)), indices=(0,),
            axis=np.zeros(3),
            center=np.zeros(3),
            angle=np.pi / 4,
        )


def test_rotate_atoms_does_not_mutate_input():
    import numpy as np

    from reactx.placement import _rotate_atoms
    positions = np.array([[1.0, 0.0, 0.0]])
    original = positions.copy()
    _rotate_atoms(
        positions, indices=(0,),
        axis=np.array([0.0, 0.0, 1.0]),
        center=np.zeros(3),
        angle=np.pi / 2,
    )
    np.testing.assert_array_equal(positions, original)


def test_find_bridging_formed_returns_list_for_one_bridge():
    from reactx.placement import _find_bridging_formed
    bridges = _find_bridging_formed(
        formed=((0, 5),),
        substrate={0, 1, 2, 3},
        fragment={5, 6},
    )
    assert bridges == [(0, 5)]


def test_find_bridging_formed_returns_list_for_two_bridges():
    from reactx.placement import _find_bridging_formed
    bridges = _find_bridging_formed(
        formed=((0, 5), (3, 6)),
        substrate={0, 1, 2, 3},
        fragment={5, 6},
    )
    assert bridges == [(0, 5), (3, 6)]


def test_find_bridging_formed_normalizes_substrate_side_first():
    """When formed is given as (fragment_atom, substrate_atom), normalize order."""
    from reactx.placement import _find_bridging_formed
    bridges = _find_bridging_formed(
        formed=((5, 0), (6, 3)),  # fragment first
        substrate={0, 1, 2, 3},
        fragment={5, 6},
    )
    # After normalization: substrate atom is first
    assert bridges == [(0, 5), (3, 6)]


def test_find_bridging_formed_three_bridges_raises_not_implemented():
    import pytest

    from reactx.placement import _find_bridging_formed
    with pytest.raises(NotImplementedError, match="N>=3 bridges"):
        _find_bridging_formed(
            formed=((0, 5), (3, 6), (1, 7)),
            substrate={0, 1, 2, 3},
            fragment={5, 6, 7},
        )


def test_find_bridging_formed_zero_bridges_raises_value_error():
    import pytest

    from reactx.placement import _find_bridging_formed
    with pytest.raises(ValueError, match="no formed bond bridging"):
        _find_bridging_formed(
            formed=((0, 1),),
            substrate={0, 1, 2, 3},
            fragment={5, 6},
        )


def test_multi_anchor_placement_basic_translation_kabsch():
    """Synthetic 6-atom geometry: 4-atom substrate + 2-atom incoming.
    Verify _multi_anchor_placement returns surviving trials with correct
    placement (M_inc lands at M_sub + d * d_min, u_inc aligned with u_sub)."""
    import numpy as np

    from reactx.placement import _multi_anchor_placement

    # substrate: 4 C atoms along x-axis (anchor pair: A1=0, A2=3)
    # incoming: 2 C atoms along y-axis (initial), anchor pair: I1=4, I2=5
    positions = np.array([
        [0.0, 0.0, 0.0],     # A1 (substrate)
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],     # A2 (substrate)
        [0.0, 5.0, 0.0],     # I1 (incoming)
        [1.5, 5.0, 0.0],     # I2 (incoming, initially in xy plane)
    ])
    syms = ["C", "C", "C", "C", "C", "C"]
    substrate_set = {0, 1, 2, 3}
    fragment_set = {4, 5}
    bridges = [(0, 4), (3, 5)]  # substrate-first normalized

    trials, blocked_reasons = _multi_anchor_placement(
        positions, syms, substrate_set, fragment_set, bridges,
        n_candidates=8, seed=0,
    )

    # Geometry chosen so no direction is blocked → all 8 candidates produce a trial.
    assert len(trials) == 8
    assert len(blocked_reasons) == 8
    # All blocked_reasons should be None (no blocking in this task)
    assert all(r is None for r in blocked_reasons)

    # 2-atom incoming with no substituents is C2-symmetric around its bond axis
    # → endo/exo collapse to a single "achiral" trial per surviving direction
    for t in trials:
        assert t.orientation == "achiral"
        assert t.positions.shape == positions.shape

    # Numerical post-conditions of multi-anchor placement.
    M_sub = (positions[0] + positions[3]) / 2.0
    v_sub = positions[3] - positions[0]
    u_sub = v_sub / np.linalg.norm(v_sub)
    L_inc_orig = float(np.linalg.norm(positions[5] - positions[4]))

    for t in trials:
        I1_new = t.positions[4]
        I2_new = t.positions[5]
        M_inc_new = (I1_new + I2_new) / 2.0

        # (1) Translation correctness: M_inc lands at M_sub + d * d_min
        target = M_sub + t.direction * t.d_min
        np.testing.assert_allclose(M_inc_new, target, atol=1e-9)

        # (2) Alignment correctness: u_inc_new == u_sub
        v_inc_new = I2_new - I1_new
        L_inc_new = float(np.linalg.norm(v_inc_new))
        u_inc_new = v_inc_new / L_inc_new
        np.testing.assert_allclose(u_inc_new, u_sub, atol=1e-9)

        # (3) Rigid-body preservation: incoming bond length unchanged
        np.testing.assert_allclose(L_inc_new, L_inc_orig, atol=1e-9)

        # (4) Substrate atoms unchanged
        np.testing.assert_allclose(t.positions[:4], positions[:4], atol=1e-12)


def test_multi_anchor_placement_coincident_anchors_raises():
    """When I1 == I2 (coincident incoming anchors), raise ValueError."""
    import numpy as np
    import pytest

    from reactx.placement import _multi_anchor_placement

    positions = np.array([
        [0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 5.0, 0.0],   # coincident with idx 2
    ])
    syms = ["C", "C", "C", "C"]
    with pytest.raises(ValueError, match="(coincident|incoming anchor pair)"):
        _multi_anchor_placement(
            positions, syms, {0, 1}, {2, 3}, [(0, 2), (1, 3)],
            n_candidates=4, seed=0,
        )


def test_multi_anchor_placement_substrate_anchors_coincident_raises():
    """When A1 == A2, raise ValueError."""
    import numpy as np
    import pytest

    from reactx.placement import _multi_anchor_placement

    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],   # coincident with idx 0
        [0.0, 5.0, 0.0],
        [1.5, 5.0, 0.0],
    ])
    syms = ["C", "C", "C", "C"]
    with pytest.raises(ValueError, match="substrate anchor pair"):
        _multi_anchor_placement(
            positions, syms, {0, 1}, {2, 3}, [(0, 2), (1, 3)],
            n_candidates=4, seed=0,
        )


def test_multi_anchor_placement_uinc_antiparallel_to_usub_handles_180deg():
    """When u_inc is antiparallel to u_sub, the Kabsch rotation must be
    180° around an axis perpendicular to u_sub. Check that placement still
    produces valid trials."""
    import numpy as np

    from reactx.placement import _multi_anchor_placement

    # u_sub along +x, u_inc along -x (antiparallel)
    positions = np.array([
        [0.0, 0.0, 0.0], [3.0, 0.0, 0.0],   # A1, A2 (u_sub = +x)
        [3.0, 5.0, 0.0], [0.0, 5.0, 0.0],   # I1, I2 (u_inc = -x, antiparallel)
    ])
    syms = ["C", "C", "C", "C"]
    trials, blocked_reasons = _multi_anchor_placement(
        positions, syms, {0, 1}, {2, 3}, [(0, 2), (1, 3)],
        n_candidates=4, seed=0,
    )
    # Should not crash and produce 4 trials
    assert len(trials) == 4

    # Antiparallel case: after rotation, u_inc must be aligned (not antiparallel) to u_sub
    u_sub = (positions[1] - positions[0]) / np.linalg.norm(positions[1] - positions[0])
    for t in trials:
        v_inc_new = t.positions[3] - t.positions[2]
        u_inc_new = v_inc_new / np.linalg.norm(v_inc_new)
        # Check u_inc_new aligned with u_sub (cos > 0.99)
        cos_align = float(np.dot(u_inc_new, u_sub))
        assert cos_align > 0.99, f"u_inc not aligned with u_sub: cos={cos_align}"


def test_multi_anchor_placement_asymmetric_dual_anchor_blocked():
    """L_sub and L_inc with large mismatch produce asymmetric bond distances
    that exceed the threshold for many directions; at least one trial should
    be blocked with 'asymmetric_dual_anchor'."""
    import numpy as np

    from reactx.placement import _multi_anchor_placement

    # butadiene-like substrate (L_sub=3.6) + ethylene-like incoming (L_inc=1.34)
    positions = np.array([
        [0.0, 0.0, 0.0],     # A1
        [3.6, 0.0, 0.0],     # A2 (L_sub = 3.6)
        [0.0, 5.0, 0.0],     # I1
        [1.34, 5.0, 0.0],    # I2 (L_inc = 1.34, much shorter)
    ])
    syms = ["C", "C", "C", "C"]
    trials, blocked_reasons = _multi_anchor_placement(
        positions, syms, {0, 1}, {2, 3}, [(0, 2), (1, 3)],
        n_candidates=64, seed=0,
    )
    # Some directions should be blocked with asymmetric_dual_anchor
    assert any(
        r is not None and "asymmetric_dual_anchor" in r
        for r in blocked_reasons
    ), f"expected asymmetric_dual_anchor in blocked_reasons; got: {blocked_reasons}"
    # blocked_reasons must have length n_candidates
    assert len(blocked_reasons) == 64


def test_multi_anchor_placement_perpendicular_directions_survive():
    """⊥ u_sub directions should give b1 ≈ b2 → not blocked by asymmetry."""
    import numpy as np

    from reactx.placement import _multi_anchor_placement

    positions = np.array([
        [0.0, 0.0, 0.0], [3.6, 0.0, 0.0],
        [0.0, 5.0, 0.0], [1.34, 5.0, 0.0],
    ])
    syms = ["C", "C", "C", "C"]
    trials, blocked_reasons = _multi_anchor_placement(
        positions, syms, {0, 1}, {2, 3}, [(0, 2), (1, 3)],
        n_candidates=64, seed=0,
    )
    # At least 1 direction perpendicular enough to pass blocking
    survivors = [t for t in trials]
    assert len(survivors) >= 1


def test_multi_anchor_placement_unreachable_dual_anchor_blocked():
    """When d_min computed by compute_d_min would exceed d_min_ceiling, the
    direction is blocked with 'unreachable_dual_anchor' (post-placement bond
    distance > ceiling)."""
    import numpy as np

    from reactx.placement import _multi_anchor_placement

    # Geometry where some directions force the placement very far away
    # (highly anisotropic substrate)
    positions = np.array([
        [0.0, 0.0, 0.0], [3.0, 0.0, 0.0],
        # A bulky substrate atom blocking +z direction (forces large d_min)
        [1.5, 0.0, 0.5],
        # Incoming pair
        [0.0, 5.0, 0.0], [1.5, 5.0, 0.0],
    ])
    syms = ["C", "C", "C", "C", "C"]
    # bridges: A1=0, I1=3; A2=1, I2=4
    trials, blocked_reasons = _multi_anchor_placement(
        positions, syms, {0, 1, 2}, {3, 4}, [(0, 3), (1, 4)],
        n_candidates=32, seed=0, d_min_ceiling=4.0,  # tight ceiling
    )
    # With tight ceiling, some directions should be blocked by unreachable
    # OR asymmetric. Just check the totals are consistent:
    assert len(blocked_reasons) == 32
    assert len(trials) + sum(1 for r in blocked_reasons if r is not None) == 32
    # Explicit assertion: at least one direction must be blocked by the
    # unreachable_dual_anchor path (post-placement bond distance > ceiling).
    assert any(
        r is not None and "unreachable_dual_anchor" in r
        for r in blocked_reasons
    ), f"expected unreachable_dual_anchor in blocked_reasons; got: {blocked_reasons}"


def test_multi_anchor_placement_blocked_reasons_string_format():
    """Blocked reasons should include numeric values for debugging."""
    import numpy as np

    from reactx.placement import _multi_anchor_placement

    positions = np.array([
        [0.0, 0.0, 0.0], [3.6, 0.0, 0.0],
        [0.0, 5.0, 0.0], [1.34, 5.0, 0.0],
    ])
    syms = ["C", "C", "C", "C"]
    trials, blocked_reasons = _multi_anchor_placement(
        positions, syms, {0, 1}, {2, 3}, [(0, 2), (1, 3)],
        n_candidates=64, seed=0,
    )
    # Find an asymmetric blocked reason
    asym_reasons = [r for r in blocked_reasons if r and "asymmetric_dual_anchor" in r]
    if asym_reasons:
        # Format should be: asymmetric_dual_anchor:b1=X.XX,b2=Y.YY
        assert "b1=" in asym_reasons[0]
        assert "b2=" in asym_reasons[0]


def test_multi_anchor_endo_exo_distinct_for_asymmetric_fragment():
    """非対称 fragment では endo と exo が別 trial として残る。"""
    import numpy as np

    from reactx.placement import _multi_anchor_placement

    # incoming fragment is asymmetric: extra atom (F) above the I1-I2 axis
    positions = np.array([
        [0.0, 0.0, 0.0],     # A1 (substrate)
        [3.0, 0.0, 0.0],     # A2 (substrate)
        [0.0, 5.0, 0.0],     # I1 (incoming)
        [3.0, 5.0, 0.0],     # I2 (incoming)
        [1.5, 5.0, 1.5],     # F (asymmetric substituent on incoming)
    ])
    syms = ["C", "C", "C", "C", "F"]
    trials, blocked_reasons = _multi_anchor_placement(
        positions, syms, {0, 1}, {2, 3, 4}, [(0, 2), (1, 3)],
        n_candidates=8, seed=0,
    )
    orientations = {t.orientation for t in trials}
    assert "endo" in orientations
    assert "exo" in orientations
    # No achiral expected (asymmetric fragment)
    assert "achiral" not in orientations


def test_multi_anchor_achiral_collapse_for_symmetric_fragment():
    """ethylene-like incoming with C2 symmetry: 180° rotation around u_sub
    permutes H atoms, so permutation-aware RMSD ~0 → achiral."""
    import numpy as np

    from reactx.placement import _multi_anchor_placement

    # ethylene-like incoming: 2 C + 4 H, planar at z=5, with H atoms placed
    # symmetrically around the C=C axis.
    # After Kabsch (u_inc → u_sub = +x), the C=C axis is along +x.
    # 180° rotation around +x maps (x, y, z) -> (x, -y, -z), so any H at
    # (x_h, y_h, z_h) is mapped to (x_h, -y_h, -z_h). For a real ethylene-like
    # H₂C=CH₂, the H's are at (x_C ± dx, ±dy, 0) — same x as their parent C
    # but offset in y.
    #
    # Initial geometry: C-C along y at y=5, z=0 (will be rotated to +x by Kabsch)
    positions = np.array([
        [0.0, 0.0, 0.0], [3.0, 0.0, 0.0],   # A1, A2 (substrate, u_sub = +x)
        [0.0, 5.0, 0.0], [1.5, 5.0, 0.0],   # I1, I2 (incoming, u_inc = +x already)
        # 4 H atoms placed symmetrically around the I1-I2 axis (z=0):
        # H atoms in (x, y_offset, 0) and (x, -y_offset, 0) pairs but along z.
        # Actually we want C2 around the I1-I2 axis (= +x).
        # 180° around +x maps (x, y, z) -> (x, -y, -z).
        # So if we have an H at (x_h, dy, dz), its C2-image is (x_h, -dy, -dz).
        # To pass the symmetry test, the fragment must contain BOTH (x_h, dy, dz)
        # AND (x_h, -dy, -dz). Place them so they ARE C2 images:
        [-0.3, 5.3, 0.4],   # H near I1, +y +z side
        [-0.3, 4.7, -0.4],  # H near I1, -y -z side (C2 image of above)
        [1.8, 5.3, 0.4],    # H near I2, +y +z side
        [1.8, 4.7, -0.4],   # H near I2, -y -z side (C2 image)
    ])
    syms = ["C", "C", "C", "C", "H", "H", "H", "H"]
    trials, blocked_reasons = _multi_anchor_placement(
        positions, syms, {0, 1}, {2, 3, 4, 5, 6, 7}, [(0, 2), (1, 3)],
        n_candidates=8, seed=0,
    )
    # All trials should be achiral (C2-symmetric ethylene-like fragment)
    for t in trials:
        assert t.orientation == "achiral", (
            f"expected achiral for ethylene-like fragment, got {t.orientation}"
        )


def test_multi_anchor_achiral_collapse_for_axial_atoms():
    """Trivial case: when all fragment atoms lie on the rotation axis,
    RMSD is exactly 0 regardless of element matching."""
    import numpy as np

    from reactx.placement import _multi_anchor_placement

    positions = np.array([
        [0.0, 0.0, 0.0], [3.0, 0.0, 0.0],
        [0.0, 5.0, 0.0], [1.5, 5.0, 0.0],
        [-0.5, 5.0, 0.0], [2.0, 5.0, 0.0],   # all on z=0, on axis
    ])
    syms = ["C", "C", "C", "C", "H", "H"]
    trials, blocked_reasons = _multi_anchor_placement(
        positions, syms, {0, 1}, {2, 3, 4, 5}, [(0, 2), (1, 3)],
        n_candidates=8, seed=0,
    )
    for t in trials:
        assert t.orientation == "achiral"


def test_permutation_aware_rmsd_index_wise_match():
    import numpy as np

    from reactx.placement import _permutation_aware_rmsd
    # When poses are identical, RMSD = 0
    pos_a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    syms = ["C", "C"]
    assert _permutation_aware_rmsd(pos_a, pos_a, syms) == 0.0


def test_permutation_aware_rmsd_swap_same_element():
    import numpy as np

    from reactx.placement import _permutation_aware_rmsd
    # 2 H atoms swapped: index-wise RMSD large, permutation-aware RMSD = 0
    pos_a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    pos_b = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])  # swapped
    syms = ["H", "H"]
    assert _permutation_aware_rmsd(pos_a, pos_b, syms) == 0.0


def test_permutation_aware_rmsd_different_elements_no_match():
    import numpy as np

    from reactx.placement import _permutation_aware_rmsd
    # H and C swapped — different elements, no match available, falls back to index-wise
    pos_a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    pos_b = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    syms = ["H", "C"]
    rmsd = _permutation_aware_rmsd(pos_a, pos_b, syms)
    # Index-wise comparison: |0-1| at each → RMSD = sqrt(mean(1, 1)) = 1.0
    assert abs(rmsd - 1.0) < 1e-9


def test_multi_anchor_orientation_field_set_correctly():
    """For each surviving direction, exactly one orientation label is assigned."""
    import numpy as np

    from reactx.placement import _multi_anchor_placement

    positions = np.array([
        [0.0, 0.0, 0.0], [3.0, 0.0, 0.0],
        [0.0, 5.0, 0.0], [1.5, 5.0, 0.0],
        [0.5, 5.5, 0.5],   # asymmetric substituent
    ])
    syms = ["C", "C", "C", "C", "F"]
    trials, blocked_reasons = _multi_anchor_placement(
        positions, syms, {0, 1}, {2, 3, 4}, [(0, 2), (1, 3)],
        n_candidates=4, seed=0,
    )
    valid_orientations = {"single", "endo", "exo", "achiral"}
    for t in trials:
        assert t.orientation in valid_orientations
        # In multi-anchor path, "single" should never appear
        assert t.orientation != "single"


def test_multi_anchor_n_trials_relationship_to_blocked_reasons():
    """For asymmetric fragment, len(survivors) ≤ 2 * (n_candidates - n_blocked)."""
    import numpy as np

    from reactx.placement import _multi_anchor_placement

    positions = np.array([
        [0.0, 0.0, 0.0], [3.0, 0.0, 0.0],
        [0.0, 5.0, 0.0], [1.5, 5.0, 0.0],
        [0.5, 5.5, 0.5],
    ])
    syms = ["C", "C", "C", "C", "F"]
    trials, blocked_reasons = _multi_anchor_placement(
        positions, syms, {0, 1}, {2, 3, 4}, [(0, 2), (1, 3)],
        n_candidates=8, seed=0,
    )
    n_blocked = sum(1 for r in blocked_reasons if r is not None)
    n_directions_survived = 8 - n_blocked
    # Each surviving direction contributes either 1 (achiral) or 2 (endo+exo) trials
    assert len(trials) >= n_directions_survived
    assert len(trials) <= 2 * n_directions_survived
    # blocked_reasons still has length n_candidates
    assert len(blocked_reasons) == 8


def test_valid_placements_dispatches_to_multi_anchor_for_two_bridges():
    """bridges == 2 → placement_kind='multi_anchor', orientations from {endo,exo,achiral}."""
    from rdkit import Chem

    from reactx.bond_changes import BondChanges
    from reactx.embed3d import embed_fragments_to_positions
    from reactx.placement import valid_placements

    # butadiene + ethylene (the canonical DA case, ethylene is symmetric → achiral)
    mol = Chem.MolFromSmiles("C=CC=C.C=C")
    mol_h, frag_indices, positions = embed_fragments_to_positions(mol, seed=0)
    # atom 0,3 are diene ends; atoms 4, 5 are ethylene
    bond_changes = BondChanges(formed=((0, 4), (3, 5)), broken=())
    result = valid_placements(
        mol_h, frag_indices, positions, bond_changes,
        n_candidates=16, seed=0,
    )
    assert result.placement_kind == "multi_anchor"
    assert len(result.trials) >= 1
    for t in result.trials:
        assert t.orientation in ("endo", "exo", "achiral")
        assert t.orientation != "single"


def test_valid_placements_existing_sn2_path_kind_single_anchor():
    """SN2 (bridges == 1) → placement_kind='single_anchor', orientation='single'. Regression."""
    from rdkit import Chem

    from reactx.bond_changes import BondChanges
    from reactx.embed3d import embed_fragments_to_positions
    from reactx.placement import valid_placements

    mol = Chem.MolFromSmiles("[O-].CCl")
    mol_h, frag_indices, positions = embed_fragments_to_positions(mol, seed=0)
    bond_changes = BondChanges(formed=((0, 1),), broken=((1, 2),))
    result = valid_placements(mol_h, frag_indices, positions, bond_changes, n_candidates=8, seed=0)
    assert result.placement_kind == "single_anchor"
    assert all(t.orientation == "single" for t in result.trials)


def test_valid_placements_two_bridges_no_longer_raises_not_implemented():
    """The earlier-stage placeholder NotImplementedError for bridges == 2 has been replaced by real multi-anchor dispatch."""
    from rdkit import Chem

    from reactx.bond_changes import BondChanges
    from reactx.embed3d import embed_fragments_to_positions
    from reactx.placement import valid_placements

    mol = Chem.MolFromSmiles("C=CC=C.C=C")
    mol_h, frag_indices, positions = embed_fragments_to_positions(mol, seed=0)
    bond_changes = BondChanges(formed=((0, 4), (3, 5)), broken=())
    # Should NOT raise
    result = valid_placements(
        mol_h, frag_indices, positions, bond_changes,
        n_candidates=4, seed=0,
    )
    assert result is not None


def test_valid_placements_three_bridges_raises_not_implemented():
    """bridges >= 3 still raises (general cycloaddition is Phase 9+)."""
    import pytest
    from rdkit import Chem

    from reactx.bond_changes import BondChanges
    from reactx.embed3d import embed_fragments_to_positions
    from reactx.placement import valid_placements

    # 3-bridge case: contrived. Use cyclopentadiene + nitrogen-aromatic with 3 close C-C contacts
    # Easier: just construct manually with synthetic atom indices.
    # Use butadiene + propene-like with 3 mock formed bonds.
    mol = Chem.MolFromSmiles("C=CC=C.C=CC")
    mol_h, frag_indices, positions = embed_fragments_to_positions(mol, seed=0)
    # 3 formed bonds between diene fragment (atoms 0-3) and other fragment (atoms 4-6)
    # All three bridge the two fragments
    bond_changes = BondChanges(formed=((0, 4), (1, 5), (3, 6)), broken=())
    with pytest.raises(NotImplementedError, match="N>=3"):
        valid_placements(
            mol_h, frag_indices, positions, bond_changes,
            n_candidates=4, seed=0,
        )
