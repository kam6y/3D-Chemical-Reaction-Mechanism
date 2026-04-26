"""Tests for reactx.reaction_topology module."""

from pathlib import Path

from rdkit import Chem

from reactx.reaction_topology import BondChange, BondChanges, compute_bond_changes
from reactx.rxn_parser import parse_rxn


def test_bondchange_dataclass_construction():
    bc = BondChange(a=0, b=1, order_before=1.0, order_after=0.0)
    assert bc.a == 0
    assert bc.b == 1
    assert bc.order_before == 1.0
    assert bc.order_after == 0.0


def test_bondchanges_default_lists():
    changes = BondChanges(broken=[], formed=[])
    assert changes.broken == []
    assert changes.formed == []


def _pair_syms(bc: BondChange, mol_h: Chem.Mol) -> tuple[str, str]:
    return tuple(
        sorted(
            [
                mol_h.GetAtomWithIdx(bc.a).GetSymbol(),
                mol_h.GetAtomWithIdx(bc.b).GetSymbol(),
            ]
        )
    )


def test_sn2_bond_changes(sn2_rxn_path: Path):
    r_mol, p_mol, mapping = parse_rxn(sn2_rxn_path)
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)
    changes = compute_bond_changes(r_h, p_h, mapping)

    assert len(changes.broken) == 1
    assert len(changes.formed) == 1
    assert _pair_syms(changes.broken[0], r_h) == ("C", "Cl")
    assert _pair_syms(changes.formed[0], r_h) == ("C", "F")
    assert changes.broken[0].order_before == 1.0
    assert changes.broken[0].order_after == 0.0
    assert changes.formed[0].order_before == 0.0
    assert changes.formed[0].order_after == 1.0


def test_sn1_step1_bond_changes(sn1_step1_rxn_path: Path):
    r_mol, p_mol, mapping = parse_rxn(sn1_step1_rxn_path)
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)
    changes = compute_bond_changes(r_h, p_h, mapping)

    assert len(changes.broken) == 1
    assert len(changes.formed) == 0
    assert _pair_syms(changes.broken[0], r_h) == ("Br", "C")


def test_proton_transfer_bond_changes(proton_transfer_rxn_path: Path):
    r_mol, p_mol, mapping = parse_rxn(proton_transfer_rxn_path)
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)
    changes = compute_bond_changes(r_h, p_h, mapping)

    assert len(changes.broken) == 1
    assert len(changes.formed) == 1
    assert _pair_syms(changes.broken[0], r_h) == ("Cl", "H")
    assert _pair_syms(changes.formed[0], r_h) == ("H", "N")


def test_e2_bond_changes(e2_rxn_path: Path):
    r_mol, p_mol, mapping = parse_rxn(e2_rxn_path)
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)
    changes = compute_bond_changes(r_h, p_h, mapping)

    assert len(changes.broken) == 2
    assert len(changes.formed) == 2

    broken_pairs = sorted(_pair_syms(b, r_h) for b in changes.broken)
    formed_pairs = sorted(_pair_syms(f, r_h) for f in changes.formed)
    assert ("Br", "C") in broken_pairs
    assert ("C", "H") in broken_pairs
    assert ("H", "O") in formed_pairs
    cc_formed = [f for f in changes.formed if _pair_syms(f, r_h) == ("C", "C")]
    assert len(cc_formed) == 1
    assert cc_formed[0].order_before == 1.0
    assert cc_formed[0].order_after == 2.0


def test_e1_step2_bond_changes(e1_step2_rxn_path: Path):
    r_mol, p_mol, mapping = parse_rxn(e1_step2_rxn_path)
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)
    changes = compute_bond_changes(r_h, p_h, mapping)

    assert len(changes.broken) == 1
    assert len(changes.formed) == 1
    assert _pair_syms(changes.broken[0], r_h) == ("C", "H")
    formed = changes.formed[0]
    assert _pair_syms(formed, r_h) == ("C", "C")
    assert formed.order_before == 1.0
    assert formed.order_after == 2.0
