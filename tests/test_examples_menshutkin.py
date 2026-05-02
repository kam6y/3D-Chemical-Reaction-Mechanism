"""Smoke test that examples/menshutkin.rxn parses + bond changes are correct."""
from pathlib import Path

from rdkit import Chem

from reactx.bond_changes import compute_simple_bond_changes
from reactx.rxn_parser import parse_rxn


def test_menshutkin_rxn_parses(menshutkin_rxn_path: Path):
    r_mol, p_mol, mapping = parse_rxn(menshutkin_rxn_path)
    # 9 atoms total on each side: N(1), 3H(2-4), C(5), 3H(6-8), Cl(9).
    # `mapping` is {reactant_idx: product_idx} for every mapped atom.
    assert len(mapping) == 9
    assert r_mol.GetNumAtoms() == 9
    assert p_mol.GetNumAtoms() == 9
    # All atom map numbers 1..9 must appear on both sides.
    r_map_nums = {a.GetAtomMapNum() for a in r_mol.GetAtoms()}
    p_map_nums = {a.GetAtomMapNum() for a in p_mol.GetAtoms()}
    assert r_map_nums == {1, 2, 3, 4, 5, 6, 7, 8, 9}
    assert p_map_nums == {1, 2, 3, 4, 5, 6, 7, 8, 9}


def test_menshutkin_bond_changes_n_c_formed_c_cl_broken(menshutkin_rxn_path: Path):
    r_mol, p_mol, mapping = parse_rxn(menshutkin_rxn_path)
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)
    bc = compute_simple_bond_changes(r_h, p_h, mapping)
    # Symbols on reactant-side at the bond endpoints
    syms = [a.GetSymbol() for a in r_h.GetAtoms()]
    formed_syms = sorted([syms[bc.formed[0]], syms[bc.formed[1]]])
    broken_syms = sorted([syms[bc.broken[0]], syms[bc.broken[1]]])
    assert formed_syms == ["C", "N"]
    assert broken_syms == ["C", "Cl"]
