"""Unit tests for reactx.bond_changes."""

import pytest
from rdkit import Chem

from reactx.bond_changes import SimpleBondChanges, compute_simple_bond_changes
from reactx.rxn_parser import parse_rxn


def _atom_index_by_symbol(mol_h: Chem.Mol, sym: str) -> int:
    """Find first atom of given element symbol (helper for assertions)."""
    for atom in mol_h.GetAtoms():
        if atom.GetSymbol() == sym:
            return atom.GetIdx()
    raise AssertionError(f"no {sym} atom in mol")


def test_sn2_bond_changes():
    r_mol, p_mol, mapping = parse_rxn("examples/sn2.rxn")
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)

    bc = compute_simple_bond_changes(r_h, p_h, mapping)
    assert isinstance(bc, SimpleBondChanges)

    c_idx = _atom_index_by_symbol(r_h, "C")
    cl_idx = _atom_index_by_symbol(r_h, "Cl")
    o_idx = _atom_index_by_symbol(r_h, "O")

    assert set(bc.formed) == {c_idx, o_idx}
    assert set(bc.broken) == {c_idx, cl_idx}


def test_proton_transfer_bond_changes():
    r_mol, p_mol, mapping = parse_rxn("examples/proton_transfer.rxn")
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)

    bc = compute_simple_bond_changes(r_h, p_h, mapping)

    n_idx = _atom_index_by_symbol(r_h, "N")
    cl_idx = _atom_index_by_symbol(r_h, "Cl")
    # The migrating proton has atom map=1; find it via map number
    proton_idx = next(a.GetIdx() for a in r_h.GetAtoms() if a.GetAtomMapNum() == 1)

    assert set(bc.formed) == {n_idx, proton_idx}
    assert set(bc.broken) == {proton_idx, cl_idx}
    assert bc.shared_atom == proton_idx  # central atom is the moving H


def test_e2_like_raises_not_implemented():
    """Two formed + two broken bonds (E2-like) should raise NotImplementedError."""
    r = Chem.MolFromSmiles("[CH3:1][CH2:2][Cl:3]")
    p = Chem.MolFromSmiles("[CH2:1]=[CH2:2].[Cl:3]")
    r_h = Chem.AddHs(r)
    p_h = Chem.AddHs(p)
    r_map = {a.GetAtomMapNum(): a.GetIdx() for a in r_h.GetAtoms() if a.GetAtomMapNum()}
    p_map = {a.GetAtomMapNum(): a.GetIdx() for a in p_h.GetAtoms() if a.GetAtomMapNum()}
    mapping = {r_map[m]: p_map[m] for m in r_map if m in p_map}

    # Implicit H counts will differ (CH3CH2Cl has 5 Hs, CH2=CH2 has 4 Hs).
    # Either condition triggers the error.
    with pytest.raises((NotImplementedError, ValueError)):
        compute_simple_bond_changes(r_h, p_h, mapping)
