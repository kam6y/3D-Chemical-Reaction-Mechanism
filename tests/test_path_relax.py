"""Unit tests for reactx.path_relax (no UMA dependency)."""
import numpy as np
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.constraints import Hookean

from reactx.path_relax import relax_with_restraints


def _two_atom_system(initial_distance: float) -> Atoms:
    return Atoms("ArAr", positions=[[0, 0, 0], [initial_distance, 0, 0]])


def test_relax_pulls_atoms_toward_target_distance():
    atoms = _two_atom_system(initial_distance=5.0)
    atoms.calc = LennardJones()
    cs = [Hookean(a1=0, a2=1, rt=2.0, k=10.0)]

    frames, energies, _ = relax_with_restraints(
        atoms, cs, atoms.calc, max_steps=200, fmax=0.05, traj_stride=10,
    )

    final_d = float(np.linalg.norm(frames[-1].positions[1] - frames[-1].positions[0]))
    assert final_d < 3.0, f"final distance {final_d:.3f} not pulled toward target"


def test_traj_stride_controls_frame_count():
    atoms = _two_atom_system(initial_distance=5.0)
    atoms.calc = LennardJones()
    cs = [Hookean(a1=0, a2=1, rt=2.0, k=10.0)]

    frames, energies, _ = relax_with_restraints(
        atoms, cs, atoms.calc, max_steps=50, fmax=1e-9, traj_stride=10,
    )
    assert 4 <= len(frames) <= 8
    assert len(energies) == len(frames)


def test_max_steps_terminates_loop():
    atoms = _two_atom_system(initial_distance=5.0)
    atoms.calc = LennardJones()
    cs = [Hookean(a1=0, a2=1, rt=2.0, k=10.0)]

    frames, _, _ = relax_with_restraints(
        atoms, cs, atoms.calc, max_steps=5, fmax=1e-12, traj_stride=1,
    )
    assert len(frames) <= 7


def test_first_frame_is_initial():
    atoms = _two_atom_system(initial_distance=5.0)
    init_pos = atoms.positions.copy()
    atoms.calc = LennardJones()
    cs = [Hookean(a1=0, a2=1, rt=2.0, k=10.0)]

    frames, _, _ = relax_with_restraints(
        atoms, cs, atoms.calc, max_steps=10, fmax=0.1, traj_stride=5,
    )
    np.testing.assert_allclose(frames[0].positions, init_pos)


def test_energies_are_finite_floats():
    atoms = _two_atom_system(initial_distance=5.0)
    atoms.calc = LennardJones()
    cs = [Hookean(a1=0, a2=1, rt=2.0, k=10.0)]

    _, energies, _ = relax_with_restraints(
        atoms, cs, atoms.calc, max_steps=10, fmax=0.1, traj_stride=5,
    )
    for e in energies:
        assert isinstance(e, float)
        assert np.isfinite(e)


# Phase 9 additions
from reactx.artificial_force import AFIRConstraint


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
