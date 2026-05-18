"""Tests for reactx.endpoint_relax (Phase 11)."""
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones

from reactx.endpoint_relax import relax_endpoint


def _diatomic_too_close() -> Atoms:
    # LJ minimum is at r = 2^(1/6) * sigma. Default sigma=1 -> r_min ~ 1.122.
    # Start at r=1.0 (compressed), relax should push atoms toward 1.122.
    return Atoms("Ar2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])


def test_relax_endpoint_fire_converges_lj_diatomic():
    atoms = _diatomic_too_close()
    relaxed, info = relax_endpoint(
        atoms,
        LennardJones(),
        fmax=0.001,
        max_steps=200,
        optimizer="FIRE",
    )
    assert info["converged"] is True
    r = float(((relaxed.positions[1] - relaxed.positions[0]) ** 2).sum() ** 0.5)
    assert 1.10 < r < 1.13, f"r={r} not near LJ minimum 1.122"
    assert info["final_fmax"] <= 0.001
    assert info["n_steps"] > 0
    assert info["energy"] < 0.0


def test_relax_endpoint_bfgs_converges_lj_diatomic():
    atoms = _diatomic_too_close()
    relaxed, info = relax_endpoint(
        atoms,
        LennardJones(),
        fmax=0.001,
        max_steps=200,
        optimizer="BFGS",
    )
    assert info["converged"] is True


def test_relax_endpoint_returns_copy_not_mutated_input():
    atoms = _diatomic_too_close()
    original_positions = atoms.positions.copy()
    relax_endpoint(atoms, LennardJones(), fmax=0.001, max_steps=50)
    assert (atoms.positions == original_positions).all()


def test_relax_endpoint_unknown_optimizer_raises():
    atoms = _diatomic_too_close()
    with pytest.raises(ValueError, match="optimizer"):
        relax_endpoint(atoms, LennardJones(), optimizer="NEWTON")


def test_relax_endpoint_non_converged_returns_info_flag():
    atoms = _diatomic_too_close()
    relaxed, info = relax_endpoint(
        atoms,
        LennardJones(),
        fmax=1e-9,
        max_steps=1,
        optimizer="FIRE",
    )
    assert info["converged"] is False
    assert info["n_steps"] >= 1
