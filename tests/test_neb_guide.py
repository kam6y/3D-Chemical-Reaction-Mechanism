"""Tests for bond-change NEB guide calculator."""
from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from reactx.neb_guide import BondDistanceGuideCalculator


class ZeroCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        self.results["energy"] = 0.0
        self.results["forces"] = np.zeros((len(atoms), 3))


def test_bond_distance_guide_adds_harmonic_energy_and_restoring_forces():
    atoms = Atoms("HH", positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    atoms.calc = BondDistanceGuideCalculator(
        ZeroCalculator(),
        targets={(0, 1): 1.5},
        guide_k=4.0,
    )

    assert atoms.get_potential_energy() == 0.5
    forces = atoms.get_forces()
    np.testing.assert_allclose(forces[0], [2.0, 0.0, 0.0])
    np.testing.assert_allclose(forces[1], [-2.0, 0.0, 0.0])


def test_bond_distance_guide_is_zero_safe_for_overlapping_pair():
    atoms = Atoms("HH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    atoms.calc = BondDistanceGuideCalculator(
        ZeroCalculator(),
        targets={(0, 1): 1.0},
        guide_k=3.0,
    )

    assert atoms.get_potential_energy() == 1.5
    np.testing.assert_allclose(atoms.get_forces(), np.zeros((2, 3)))
