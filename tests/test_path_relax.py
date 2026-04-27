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

    frames, energies = relax_with_restraints(
        atoms, cs, atoms.calc, max_steps=200, fmax=0.05, traj_stride=10,
    )

    final_d = float(np.linalg.norm(frames[-1].positions[1] - frames[-1].positions[0]))
    assert final_d < 3.0, f"final distance {final_d:.3f} not pulled toward target"


def test_traj_stride_controls_frame_count():
    atoms = _two_atom_system(initial_distance=5.0)
    atoms.calc = LennardJones()
    cs = [Hookean(a1=0, a2=1, rt=2.0, k=10.0)]

    frames, energies = relax_with_restraints(
        atoms, cs, atoms.calc, max_steps=50, fmax=1e-9, traj_stride=10,
    )
    assert 4 <= len(frames) <= 8
    assert len(energies) == len(frames)


def test_max_steps_terminates_loop():
    atoms = _two_atom_system(initial_distance=5.0)
    atoms.calc = LennardJones()
    cs = [Hookean(a1=0, a2=1, rt=2.0, k=10.0)]

    frames, _ = relax_with_restraints(
        atoms, cs, atoms.calc, max_steps=5, fmax=1e-12, traj_stride=1,
    )
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
