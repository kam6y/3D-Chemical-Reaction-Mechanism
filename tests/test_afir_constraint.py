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
