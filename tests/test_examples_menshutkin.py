"""Smoke test that examples/menshutkin.rxn parses + atom map covers all atoms."""
from pathlib import Path

from reactx.bond_changes import BondChanges
from reactx.rxn_parser import atom_map_to_reactant_idx, parse_rxn


def test_menshutkin_rxn_parses(menshutkin_rxn_path: Path):
    r_mol, p_mol, mapping = parse_rxn(menshutkin_rxn_path)
    assert len(mapping) == 9
    assert r_mol.GetNumAtoms() == 9
    assert p_mol.GetNumAtoms() == 9
    r_map_nums = {a.GetAtomMapNum() for a in r_mol.GetAtoms()}
    p_map_nums = {a.GetAtomMapNum() for a in p_mol.GetAtoms()}
    assert r_map_nums == {1, 2, 3, 4, 5, 6, 7, 8, 9}
    assert p_map_nums == {1, 2, 3, 4, 5, 6, 7, 8, 9}


def test_menshutkin_n_c_formed_c_cl_broken_via_toml_pairs(menshutkin_rxn_path: Path):
    """TOML claims formed=[[1,5]] broken=[[5,9]] — translating gives the right symbols."""
    r_mol, _, _ = parse_rxn(menshutkin_rxn_path)
    m2i = atom_map_to_reactant_idx(r_mol)
    bc = BondChanges.from_atom_map_pairs(
        formed_map=[(1, 5)],
        broken_map=[(5, 9)],
        atom_map_to_idx=m2i,
    )
    syms = [a.GetSymbol() for a in r_mol.GetAtoms()]
    assert sorted([syms[bc.formed[0][0]], syms[bc.formed[0][1]]]) == ["C", "N"]
    assert sorted([syms[bc.broken[0][0]], syms[bc.broken[0][1]]]) == ["C", "Cl"]
