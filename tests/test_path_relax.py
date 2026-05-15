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


# Phase 10: pre-relax (unbiased Stage A before AFIR Stage B)

def test_pre_relax_zero_omits_state_key():
    a = Atoms("CC", positions=[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    c = AFIRConstraint(
        formed=[(0, 1)], broken=[],
        alpha_formed=[0.5], alpha_broken=[],
        formed_thresholds=[1.5], broken_thresholds=[],
    )
    _, _, state = relax_with_restraints(
        a, [c], LennardJones(),
        pre_relax_steps=0,
        max_steps=10, fmax=0.5, traj_stride=2,
    )
    assert "frame_after_pre_relax" not in state


def test_pre_relax_positive_records_intermediate_frame():
    a = Atoms("CC", positions=[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    c = AFIRConstraint(
        formed=[(0, 1)], broken=[],
        alpha_formed=[0.5], alpha_broken=[],
        formed_thresholds=[1.5], broken_thresholds=[],
    )
    _, _, state = relax_with_restraints(
        a, [c], LennardJones(),
        pre_relax_steps=5,
        max_steps=10, fmax=0.5, traj_stride=2,
    )
    assert "frame_after_pre_relax" in state
    pre_frame = state["frame_after_pre_relax"]
    assert isinstance(pre_frame, Atoms)
    assert len(pre_frame) == 2
    assert len(pre_frame.constraints) == 0


def test_pre_relax_frames_concatenate_with_stage_b():
    """Stage A frames must be concatenated with Stage B frames, not replaced.

    Regression guard for spec §5.2: if Stage A's `opt_a.attach(_record_a, ...)`
    is removed (or the interval changes silently), Stage A frames go missing
    while existing single-Stage tests still pass. This test runs the same
    system with and without pre-relax and asserts the pre-relax run produces
    strictly more frames.
    """
    init_pos = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    cs_for = lambda: [Hookean(a1=0, a2=1, rt=2.0, k=10.0)]

    a_no_pre = Atoms("ArAr", positions=init_pos)
    frames_no_pre, _, _ = relax_with_restraints(
        a_no_pre, cs_for(), LennardJones(),
        pre_relax_steps=0,
        max_steps=20, fmax=1e-6, traj_stride=5,
    )

    a_with_pre = Atoms("ArAr", positions=init_pos)
    frames_with_pre, _, state = relax_with_restraints(
        a_with_pre, cs_for(), LennardJones(),
        pre_relax_steps=20,
        max_steps=20, fmax=1e-6, traj_stride=5,
    )

    assert len(frames_with_pre) > len(frames_no_pre), (
        f"pre-relax should concatenate Stage A frames on top of Stage B; "
        f"got with_pre={len(frames_with_pre)}, no_pre={len(frames_no_pre)}. "
        f"Stage A _record_a attach may be missing."
    )
    assert "frame_after_pre_relax" in state


def test_pre_relax_phase_does_not_apply_afir_force():
    a = Atoms("CC", positions=[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    c = AFIRConstraint(
        formed=[(0, 1)], broken=[],
        alpha_formed=[10.0], alpha_broken=[],
        formed_thresholds=[0.1], broken_thresholds=[],
    )
    _, _, state = relax_with_restraints(
        a, [c], LennardJones(),
        pre_relax_steps=80,
        max_steps=0,
        fmax=1e-6, traj_stride=10,
    )
    pre_frame = state["frame_after_pre_relax"]
    d = float(np.linalg.norm(pre_frame.positions[1] - pre_frame.positions[0]))
    assert d > 0.9, (
        f"distance after pre-relax {d:.3f} too close — looks like AFIR force "
        f"was applied during Stage A"
    )
