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
    assert reactant_symbols == ["C", "Cl", "F"]
    assert product_symbols == ["C", "Cl", "F"]


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
