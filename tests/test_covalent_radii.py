import logging

import numpy as np
from ase import Atoms

from reactx.covalent_radii import (
    CORDERO_2008,
    cordero_radii_for_atoms,
    cordero_radius,
)


def test_cordero_radius_known_elements():
    assert cordero_radius("H") == 0.31
    assert cordero_radius("C") == 0.76
    assert cordero_radius("Cl") == 1.02
    assert cordero_radius("Br") == 1.20


def test_cordero_radius_unknown_returns_default(caplog):
    with caplog.at_level(logging.WARNING, logger="reactx.covalent_radii"):
        assert cordero_radius("Uuq", default=1.5) == 1.5
        assert cordero_radius("Po") == 1.5
    # Two warnings emitted (one per unknown symbol)
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 2


def test_cordero_table_covers_z1_to_z83():
    expected = {"H", "He", "C", "N", "O", "F", "Cl", "Br", "Bi"}
    assert expected.issubset(CORDERO_2008.keys())
    assert "Po" not in CORDERO_2008
    assert "Fr" not in CORDERO_2008


def test_cordero_radii_for_atoms():
    a = Atoms("CHCl", positions=np.zeros((3, 3)))
    radii = cordero_radii_for_atoms(a)
    np.testing.assert_allclose(radii, [0.76, 0.31, 1.02])
