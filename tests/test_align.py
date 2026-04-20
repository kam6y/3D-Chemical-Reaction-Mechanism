import numpy as np
import pytest
from ase import Atoms

from reactx.align import align_product_to_reactant


def _build_pair():
    # reactant: C, Cl, F, H, H, H  (indices 0..5)
    r = Atoms(
        symbols=["C", "Cl", "F", "H", "H", "H"],
        positions=[(0, 0, 0), (1.8, 0, 0), (-3.5, 0, 0),
                   (0.3, 1.0, 0), (0.3, -0.5, 0.9), (0.3, -0.5, -0.9)],
    )
    # product: Cl, F, C, H, H, H (indices 0..5) — heavy atoms shuffled
    p = Atoms(
        symbols=["Cl", "F", "C", "H", "H", "H"],
        positions=[(3.8, 0, 0), (-1.5, 0, 0), (0, 0, 0),
                   (-0.3, 1.0, 0), (-0.3, -0.5, 0.9), (-0.3, -0.5, -0.9)],
    )
    # heavy mapping reactant_heavy -> product_heavy:
    # reactant 0(C)->product 2(C), 1(Cl)->0(Cl), 2(F)->1(F)
    mapping = {0: 2, 1: 0, 2: 1}
    heavy_h_groups_reactant = {0: [3, 4, 5]}  # C at idx 0 owns 3 Hs
    heavy_h_groups_product = {2: [3, 4, 5]}   # C at idx 2 owns 3 Hs
    return r, p, mapping, heavy_h_groups_reactant, heavy_h_groups_product


def test_align_reorders_heavy_atoms_to_reactant_order():
    r, p, mapping, rH, pH = _build_pair()
    aligned = align_product_to_reactant(
        reactant=r, product=p, heavy_mapping=mapping,
        reactant_h_groups=rH, product_h_groups=pH,
    )
    assert aligned.get_chemical_symbols() == r.get_chemical_symbols()

    # Expected permutation: reactant[0]=C -> product[2], [1]=Cl->[0], [2]=F->[1],
    # plus the three Hs which map identity 3,4,5.
    expected_positions = p.positions[[2, 0, 1, 3, 4, 5]]
    assert np.allclose(aligned.positions, expected_positions)


def test_align_preserves_atom_count():
    r, p, mapping, rH, pH = _build_pair()
    aligned = align_product_to_reactant(r, p, mapping, rH, pH)
    assert len(aligned) == len(p) == len(r)


def test_align_h_count_per_heavy_must_match():
    r, p, mapping, rH, pH = _build_pair()
    pH_bad = {2: [3, 4]}  # only 2 Hs on product carbon — mismatch
    with pytest.raises(ValueError, match="hydrogen count"):
        align_product_to_reactant(r, p, mapping, rH, pH_bad)


def test_align_atom_count_mismatch_raises():
    r = Atoms(symbols=["C", "H"], positions=[(0, 0, 0), (1, 0, 0)])
    p = Atoms(symbols=["C"], positions=[(0, 0, 0)])
    with pytest.raises(ValueError, match="Atom count mismatch"):
        align_product_to_reactant(
            reactant=r, product=p, heavy_mapping={0: 0},
            reactant_h_groups={0: [1]}, product_h_groups={0: []},
        )


def test_align_reactant_heavy_not_in_mapping_raises():
    r = Atoms(symbols=["C", "Cl"], positions=[(0, 0, 0), (1.8, 0, 0)])
    p = Atoms(symbols=["Cl", "C"], positions=[(0, 0, 0), (1.8, 0, 0)])
    with pytest.raises(ValueError, match="not in mapping"):
        align_product_to_reactant(
            reactant=r, product=p, heavy_mapping={0: 1},  # missing reactant idx 1
            reactant_h_groups={0: [], 1: []}, product_h_groups={1: []},
        )
