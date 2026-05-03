"""Unit tests for reactx.bond_changes."""
import pytest
from rdkit import Chem

from reactx.bond_changes import BondChanges, compute_bond_changes
from reactx.rxn_parser import parse_rxn


def _atom_index_by_symbol(mol_h: Chem.Mol, sym: str) -> int:
    for atom in mol_h.GetAtoms():
        if atom.GetSymbol() == sym:
            return atom.GetIdx()
    raise AssertionError(f"no {sym} atom in mol")


def test_sn2_bond_changes_returns_singleton_lists():
    r_mol, p_mol, mapping = parse_rxn("examples/sn2.rxn")
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)

    bc = compute_bond_changes(r_h, p_h, mapping)
    assert isinstance(bc, BondChanges)
    assert len(bc.formed) == 1
    assert len(bc.broken) == 1

    c_idx = _atom_index_by_symbol(r_h, "C")
    cl_idx = _atom_index_by_symbol(r_h, "Cl")
    o_idx = _atom_index_by_symbol(r_h, "O")
    assert set(bc.formed[0]) == {c_idx, o_idx}
    assert set(bc.broken[0]) == {c_idx, cl_idx}


def test_proton_transfer_bond_changes():
    r_mol, p_mol, mapping = parse_rxn("examples/proton_transfer.rxn")
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)

    bc = compute_bond_changes(r_h, p_h, mapping)
    assert len(bc.formed) == 1
    assert len(bc.broken) == 1

    n_idx = _atom_index_by_symbol(r_h, "N")
    cl_idx = _atom_index_by_symbol(r_h, "Cl")
    proton_idx = next(a.GetIdx() for a in r_h.GetAtoms() if a.GetAtomMapNum() == 1)

    assert set(bc.formed[0]) == {n_idx, proton_idx}
    assert set(bc.broken[0]) == {proton_idx, cl_idx}


def test_bond_changes_rejects_self_loop():
    with pytest.raises(ValueError, match="self-loop"):
        BondChanges(formed=((0, 0),), broken=((1, 2),))


def test_bond_changes_rejects_duplicate():
    with pytest.raises(ValueError, match="duplicate"):
        BondChanges(formed=((0, 1), (1, 0)), broken=())


def test_bond_changes_rejects_empty_total():
    with pytest.raises(ValueError, match="at least one"):
        BondChanges(formed=(), broken=())


def test_bond_changes_accepts_multi_bond_topologies():
    # E2 形状: 1 formed + 2 broken
    bc = BondChanges(formed=((4, 5),), broken=((0, 2), (1, 4)))
    assert len(bc.formed) == 1
    assert len(bc.broken) == 2

    # SN1 step 1 形状: 0 formed + 1 broken
    bc = BondChanges(formed=(), broken=((0, 1),))
    assert len(bc.formed) == 0
    assert len(bc.broken) == 1
