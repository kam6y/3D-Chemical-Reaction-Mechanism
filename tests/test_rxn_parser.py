from pathlib import Path

import pytest
from rdkit import Chem

from reactx.rxn_parser import parse_rxn


def test_parse_sn2_rxn_returns_combined_mols(sn2_rxn_path: Path):
    reactant, product, mapping = parse_rxn(sn2_rxn_path)

    assert isinstance(reactant, Chem.Mol)
    assert isinstance(product, Chem.Mol)
    assert isinstance(mapping, dict)

    reactant_symbols = sorted(a.GetSymbol() for a in reactant.GetAtoms())
    product_symbols = sorted(a.GetSymbol() for a in product.GetAtoms())
    assert reactant_symbols == ["C", "Cl", "O"]
    assert product_symbols == ["C", "Cl", "O"]


def test_parse_sn2_rxn_has_two_fragments_each_side(sn2_rxn_path: Path):
    reactant, product, _ = parse_rxn(sn2_rxn_path)
    assert len(Chem.GetMolFrags(reactant)) == 2
    assert len(Chem.GetMolFrags(product)) == 2


def test_parse_sn2_rxn_mapping_covers_all_heavy_atoms(sn2_rxn_path: Path):
    reactant, product, mapping = parse_rxn(sn2_rxn_path)
    assert len(mapping) == reactant.GetNumHeavyAtoms() == 3
    for r_idx, p_idx in mapping.items():
        r_sym = reactant.GetAtomWithIdx(r_idx).GetSymbol()
        p_sym = product.GetAtomWithIdx(p_idx).GetSymbol()
        assert r_sym == p_sym, f"Mapping {r_idx}->{p_idx} crosses element: {r_sym}/{p_sym}"


def test_parse_rxn_raises_when_atom_map_missing(tmp_path: Path):
    bad = tmp_path / "nomap.rxn"
    bad.write_text(
        "$RXN\n\n   bad\n\n  1  1\n$MOL\n\n     RDKit          2D\n\n  1  0  0  0  0  0  0  0  0  0999 V2000\n"
        "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\nM  END\n$MOL\n\n"
        "     RDKit          2D\n\n  1  0  0  0  0  0  0  0  0  0999 V2000\n"
        "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\nM  END\n"
    )
    with pytest.raises(ValueError, match="has no atom map number"):
        parse_rxn(bad)


def test_heavy_to_hydrogen_groups_on_methane():
    from reactx.rxn_parser import heavy_to_hydrogen_groups
    mol = Chem.MolFromSmiles("C")
    mol_h = Chem.AddHs(mol)
    groups = heavy_to_hydrogen_groups(mol_h)
    assert groups == {0: [1, 2, 3, 4]}


def test_parse_sn1_recomb_rxn(sn1_recomb_rxn_path):
    """examples/sn1_recomb.rxn が parse でき、formed=1 / broken=0 が抽出される。"""
    from reactx.bond_changes import compute_bond_changes

    r_mol, p_mol, mapping = parse_rxn(sn1_recomb_rxn_path)
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)
    bc = compute_bond_changes(r_h, p_h, mapping)

    assert len(bc.formed) == 1, f"expected 1 formed bond, got {bc.formed}"
    assert len(bc.broken) == 0, f"expected 0 broken bond, got {bc.broken}"
    assert len(Chem.GetMolFrags(r_h)) == 2  # reactant: cation + nucleophile
    assert len(Chem.GetMolFrags(p_h)) == 1  # product: recombined
    a, b = bc.formed[0]
    syms = [at.GetSymbol() for at in r_h.GetAtoms()]
    assert {syms[a], syms[b]} == {"C", "Cl"}, (
        f"expected C-Cl formed, got {syms[a]}-{syms[b]}"
    )


def test_parse_metathesis_4center_rxn(metathesis_rxn_path):
    """examples/metathesis_4center.rxn が parse でき、formed=2 / broken=2 が抽出される。"""
    from rdkit import Chem

    from reactx.bond_changes import compute_bond_changes
    from reactx.rxn_parser import parse_rxn

    r_mol, p_mol, mapping = parse_rxn(metathesis_rxn_path)
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)
    bc = compute_bond_changes(r_h, p_h, mapping)

    assert len(bc.formed) == 2, f"expected 2 formed bonds, got {bc.formed}"
    assert len(bc.broken) == 2, f"expected 2 broken bonds, got {bc.broken}"
    assert len(Chem.GetMolFrags(r_h)) == 2, "reactant should have 2 fragments"
    assert len(Chem.GetMolFrags(p_h)) == 2, "product should have 2 fragments"

    syms = [a.GetSymbol() for a in r_h.GetAtoms()]
    formed_pair_syms = {frozenset({syms[a], syms[b]}) for a, b in bc.formed}
    broken_pair_syms = {frozenset({syms[a], syms[b]}) for a, b in bc.broken}
    assert frozenset({"C", "Br"}) in formed_pair_syms, f"expected C-Br formed; got {formed_pair_syms}"
    assert frozenset({"Li", "Cl"}) in formed_pair_syms, f"expected Li-Cl formed; got {formed_pair_syms}"
    assert frozenset({"C", "Cl"}) in broken_pair_syms, f"expected C-Cl broken; got {broken_pair_syms}"
    assert frozenset({"Li", "Br"}) in broken_pair_syms, f"expected Li-Br broken; got {broken_pair_syms}"
