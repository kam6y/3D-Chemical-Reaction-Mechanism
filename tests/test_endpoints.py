"""Endpoint un-restrained relax for CI-NEB."""
from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.calculators.lj import LennardJones

from reactx.endpoints import relax_endpoint


def test_relax_endpoint_lj_dimer_moves_toward_minimum():
    # LJ minimum is at r = 2^(1/6) * sigma; default sigma=1, so r_min ≈ 1.122.
    atoms = Atoms("Ar2", positions=[[0, 0, 0], [3.0, 0, 0]])
    relaxed = relax_endpoint(atoms, LennardJones(), fmax=0.01, max_steps=200)
    r = float(np.linalg.norm(relaxed.positions[1] - relaxed.positions[0]))
    assert 1.0 < r < 1.5


def test_relax_endpoint_does_not_raise_on_max_steps():
    # fmax impossible in 1 step -> still returns Atoms, no exception.
    atoms = Atoms("Ar2", positions=[[0, 0, 0], [3.0, 0, 0]])
    relaxed = relax_endpoint(atoms, LennardJones(), fmax=1e-12, max_steps=1)
    assert relaxed is not None
    assert len(relaxed) == 2


def test_relax_endpoint_returns_independent_copy():
    atoms = Atoms("Ar2", positions=[[0, 0, 0], [3.0, 0, 0]])
    original_positions = atoms.positions.copy()
    _ = relax_endpoint(atoms, LennardJones(), fmax=0.01, max_steps=20)
    np.testing.assert_array_equal(atoms.positions, original_positions)


def test_relax_endpoint_calc_detached_from_result():
    atoms = Atoms("Ar2", positions=[[0, 0, 0], [3.0, 0, 0]])
    relaxed = relax_endpoint(atoms, LennardJones(), fmax=0.01, max_steps=20)
    assert relaxed.calc is None
