"""Unit tests for reactx.artificial_force."""
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.lj import LennardJones

from reactx.artificial_force import (
    DEFAULT_R_FORM,
    build_restraints,
    lookup_r_form,
)


def test_default_r_form_has_common_pairs():
    assert ("C", "F") in DEFAULT_R_FORM or ("F", "C") in DEFAULT_R_FORM
    assert ("N", "H") in DEFAULT_R_FORM or ("H", "N") in DEFAULT_R_FORM


def test_lookup_r_form_symmetric():
    assert lookup_r_form("C", "F") == lookup_r_form("F", "C")
    assert lookup_r_form("N", "H") == lookup_r_form("H", "N")


def test_lookup_r_form_known_values():
    assert abs(lookup_r_form("C", "F") - 1.39) < 1e-6
    assert abs(lookup_r_form("C", "Cl") - 1.78) < 1e-6
    assert abs(lookup_r_form("N", "H") - 1.01) < 1e-6


def test_lookup_r_form_unknown_pair_returns_default():
    assert lookup_r_form("Ge", "Te", default=1.6) == 1.6


def test_hookean_pulls_when_distance_above_rt():
    """Hookean (ASE built-in) pulls atoms together when r > rt."""
    atoms = Atoms("CF", positions=[[0, 0, 0], [3.0, 0, 0]])
    cs = build_restraints(atoms, formed=[(0, 1)], broken=[],
                          r_form=1.5, k_form=5.0)
    atoms.set_constraint(cs)
    atoms.calc = LennardJones()
    f = atoms.get_forces()
    # Atom 1 should be pulled toward atom 0 → negative x force
    assert f[1, 0] < -1.0, f"expected attractive force, got fx={f[1, 0]}"
    # Atom 0 should be pulled toward atom 1 → positive x force
    assert f[0, 0] > 1.0, f"expected attractive force on atom 0, got fx={f[0, 0]}"


def test_pullapart_pushes_when_distance_below_rt():
    """PullApart pushes atoms apart when r < rt."""
    atoms = Atoms("CCl", positions=[[0, 0, 0], [2.0, 0, 0]])
    cs = build_restraints(atoms, formed=[], broken=[(0, 1)],
                          r_broken=4.0, k_broken=3.0)
    atoms.set_constraint(cs)
    atoms.calc = LennardJones()
    f = atoms.get_forces()
    # Atom 1 should be pushed away from atom 0 → positive x force
    assert f[1, 0] > 1.0, f"expected repulsive force, got fx={f[1, 0]}"
    assert f[0, 0] < -1.0, f"expected repulsive force on atom 0, got fx={f[0, 0]}"


def test_pullapart_silent_when_distance_above_rt():
    """PullApart contributes zero force when r >= rt."""
    atoms = Atoms("CCl", positions=[[0, 0, 0], [5.0, 0, 0]])
    cs = build_restraints(atoms, formed=[], broken=[(0, 1)],
                          r_broken=4.0, k_broken=3.0)
    atoms.set_constraint(cs)

    class ZeroCalc(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=()):
            super().calculate(atoms, properties, system_changes)
            self.results = {
                "energy": 0.0,
                "forces": np.zeros((len(atoms), 3)),
            }

    atoms.calc = ZeroCalc()
    f = atoms.get_forces()
    np.testing.assert_allclose(f, 0.0, atol=1e-10)


def test_build_restraints_empty_lists():
    atoms = Atoms("HH", positions=[[0, 0, 0], [1, 0, 0]])
    cs = build_restraints(atoms, formed=[], broken=[])
    assert cs == []
