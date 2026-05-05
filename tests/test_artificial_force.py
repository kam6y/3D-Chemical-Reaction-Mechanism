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


def test_build_restraints_scales_linearly_with_bond_count():
    """Phase 3 multi-bond regression: 2 formed + 1 broken -> 3 constraints."""
    atoms = Atoms("CHFNN", positions=[
        [0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0],
    ])
    cs = build_restraints(
        atoms,
        formed=[(0, 1), (2, 3)],
        broken=[(0, 4)],
        r_form=1.5, r_broken=4.0,
    )
    assert len(cs) == 3  # 2 Hookeans + 1 PullApart


def test_build_restraints_with_per_bond_k_form_list():
    from ase.constraints import Hookean
    atoms = Atoms("CCCC", positions=[[0, 0, 0], [1.5, 0, 0], [3, 0, 0], [4.5, 0, 0]])
    cs = build_restraints(
        atoms,
        formed=[(0, 2), (1, 3)],
        broken=[],
        r_form=[1.5, 1.5],
        k_form=[2.0, 5.0],
    )
    hookean_cs = [c for c in cs if isinstance(c, Hookean)]
    assert len(hookean_cs) == 2
    # Hookean k attribute is `spring` in ASE 3.x; fall back to `k` if needed
    def _hookean_k(c):
        return getattr(c, "spring", getattr(c, "k", None))
    ks = sorted(_hookean_k(c) for c in hookean_cs)
    assert ks == [2.0, 5.0]


def test_build_restraints_with_per_bond_k_broken_and_r_broken_lists():
    atoms = Atoms("CClCH", positions=[[0, 0, 0], [2, 0, 0], [4, 0, 0], [5, 0, 0]])
    cs = build_restraints(
        atoms,
        formed=[],
        broken=[(0, 1), (2, 3)],
        k_broken=[2.0, 4.0],
        r_broken=[3.0, 5.0],
    )
    assert len(cs) == 2
    rts = sorted(c.rt for c in cs)
    ks = sorted(c.k for c in cs)
    assert rts == [3.0, 5.0]
    assert ks == [2.0, 4.0]


def test_build_restraints_k_broken_list_length_mismatch_raises():
    import pytest
    atoms = Atoms("CClCH", positions=[[0, 0, 0], [2, 0, 0], [4, 0, 0], [5, 0, 0]])
    with pytest.raises(ValueError, match="k_broken"):
        build_restraints(
            atoms,
            formed=[],
            broken=[(0, 1), (2, 3)],
            k_broken=[1.0],
            r_broken=4.0,
        )


def test_build_restraints_k_form_list_length_mismatch_raises():
    import pytest
    atoms = Atoms("CCCC", positions=[[0, 0, 0], [1.5, 0, 0], [3, 0, 0], [4.5, 0, 0]])
    with pytest.raises(ValueError, match="k_form"):
        build_restraints(
            atoms,
            formed=[(0, 1)],
            broken=[],
            r_form=1.5,
            k_form=[1.0, 2.0],
        )


def test_build_restraints_r_form_list_passes_through():
    from ase.constraints import Hookean
    atoms = Atoms("CCCC", positions=[[0, 0, 0], [1.5, 0, 0], [3, 0, 0], [4.5, 0, 0]])
    cs = build_restraints(
        atoms,
        formed=[(0, 2), (1, 3)],
        broken=[],
        r_form=[1.5, 1.4],
        k_form=1.0,
    )
    hookean_cs = [c for c in cs if isinstance(c, Hookean)]
    rts = sorted(getattr(c, "threshold", getattr(c, "rt", None)) for c in hookean_cs)
    assert rts == [1.4, 1.5]
