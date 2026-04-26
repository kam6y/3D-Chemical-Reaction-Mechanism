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
    # full atom mapping (reactant idx -> product idx) including Hs
    atom_mapping = {0: 2, 1: 0, 2: 1, 3: 3, 4: 4, 5: 5}
    swappable_h_groups = [[3, 4, 5]]
    return r, p, atom_mapping, swappable_h_groups


def test_align_reorders_heavy_atoms_to_reactant_order():
    r, p, mapping, swap = _build_pair()
    aligned = align_product_to_reactant(r, p, mapping, swap)
    assert aligned.get_chemical_symbols() == r.get_chemical_symbols()

    # Expected permutation (before rigid-body alignment): centroid-subtracted
    # positions must match, since align_product_to_reactant also applies
    # rigid rotation+translation to minimize RMSD against reactant.
    expected = p.positions[[2, 0, 1, 3, 4, 5]]
    assert np.allclose(
        aligned.positions - aligned.positions.mean(axis=0),
        expected - expected.mean(axis=0),
        atol=1e-6,
    )


def test_align_preserves_atom_count():
    r, p, mapping, swap = _build_pair()
    aligned = align_product_to_reactant(r, p, mapping, swap)
    assert len(aligned) == len(p) == len(r)


def test_align_atom_count_mismatch_raises():
    r = Atoms(symbols=["C", "H"], positions=[(0, 0, 0), (1, 0, 0)])
    p = Atoms(symbols=["C"], positions=[(0, 0, 0)])
    with pytest.raises(ValueError, match="Atom count mismatch"):
        align_product_to_reactant(r, p, {0: 0, 1: 0}, [])


def test_align_missing_atom_in_mapping_raises():
    r = Atoms(symbols=["C", "Cl"], positions=[(0, 0, 0), (1.8, 0, 0)])
    p = Atoms(symbols=["Cl", "C"], positions=[(0, 0, 0), (1.8, 0, 0)])
    with pytest.raises(ValueError, match="must cover all reactant atoms"):
        align_product_to_reactant(r, p, {0: 1}, [])  # missing reactant idx 1


def test_align_with_no_swappable_groups_skips_h_permutation():
    r, p, mapping, _ = _build_pair()
    aligned = align_product_to_reactant(r, p, mapping, swappable_h_groups=None)
    assert aligned.get_chemical_symbols() == r.get_chemical_symbols()


def test_build_swappable_h_groups_excludes_atom_mapped_h():
    from rdkit import Chem
    from reactx.align import build_swappable_h_groups
    # methane with one explicit H carrying an atom map number; that H must
    # NOT appear in the swappable group.
    mol = Chem.MolFromSmiles("C")
    mol = Chem.AddHs(mol)
    # mark atom 1 (an H bonded to C at atom 0) as a migrating H
    assert mol.GetAtomWithIdx(1).GetSymbol() == "H"
    mol.GetAtomWithIdx(1).SetAtomMapNum(99)
    groups = build_swappable_h_groups(mol)
    assert len(groups) == 1
    assert 1 not in groups[0]
    # Only the 3 unmapped Hs are swappable
    assert len(groups[0]) == 3
