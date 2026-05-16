"""Tests for Phase 11 simple_placement."""
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from reactx.bond_changes import BondChanges
from reactx.embed3d import embed_fragments_to_positions
from reactx.placement import build_atoms_from_positions, simple_placement
from reactx.rxn_parser import parse_rxn

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _r_inputs(rxn_name: str):
    r_mol, p_mol, heavy_mapping = parse_rxn(EXAMPLES / f"{rxn_name}.rxn")
    mol_h, _, base_positions = embed_fragments_to_positions(r_mol, seed=0)
    bc = BondChanges.from_reaction_diff(r_mol, p_mol)
    return mol_h, base_positions, bc, heavy_mapping


def test_simple_placement_unimolecular_passthrough():
    mol_h, base_positions, bc, _ = _r_inputs("sn1_dissoc")
    out = simple_placement(
        mol_h,
        base_positions,
        bc,
        initial_separation=4.0,
        side="reactant",
        orientation="default",
    )
    assert np.allclose(out, base_positions)


def test_simple_placement_bimolecular_separation_at_least_target():
    mol_h, base_positions, bc, _ = _r_inputs("sn2")
    out = simple_placement(
        mol_h,
        base_positions,
        bc,
        initial_separation=4.0,
        side="reactant",
        orientation="default",
    )
    a, b = bc.formed[0]
    r = float(np.linalg.norm(out[b] - out[a]))
    assert r >= 4.0 - 1e-6, f"anchor distance {r} < target 4.0"

    frag_indices = Chem.GetMolFrags(mol_h)
    assert len(frag_indices) == 2
    com0 = out[list(frag_indices[0])].mean(axis=0)
    com1 = out[list(frag_indices[1])].mean(axis=0)
    assert float(np.linalg.norm(com1 - com0)) > 2.0


def test_simple_placement_product_uses_broken_anchor():
    _, _, bc, heavy_mapping = _r_inputs("sn2")
    _, p_mol, _ = parse_rxn(EXAMPLES / "sn2.rxn")
    mol_h_p, _, base_positions_p = embed_fragments_to_positions(p_mol, seed=0)
    out = simple_placement(
        mol_h_p,
        base_positions_p,
        bc,
        initial_separation=4.0,
        side="product",
        orientation="default",
        heavy_mapping=heavy_mapping,
    )
    a_r, b_r = bc.broken[0]
    a_p = heavy_mapping[a_r]
    b_p = heavy_mapping[b_r]
    r = float(np.linalg.norm(out[b_p] - out[a_p]))
    assert r >= 4.0 - 1e-6


def test_simple_placement_invalid_side_raises():
    mol_h, base_positions, bc, _ = _r_inputs("sn2")
    with pytest.raises(ValueError, match="side"):
        simple_placement(
            mol_h,
            base_positions,
            bc,
            initial_separation=4.0,
            side="other",
            orientation="default",
        )


def test_simple_placement_product_without_heavy_mapping_raises():
    mol_h, base_positions, bc, _ = _r_inputs("sn2")
    with pytest.raises(ValueError, match="heavy_mapping"):
        simple_placement(
            mol_h,
            base_positions,
            bc,
            initial_separation=4.0,
            side="product",
            orientation="default",
        )


def test_build_atoms_from_positions_preserves_symbols():
    mol_h, base_positions, _, _ = _r_inputs("sn2")
    atoms = build_atoms_from_positions(mol_h, base_positions)
    assert len(atoms) == mol_h.GetNumAtoms()
    assert atoms.get_chemical_symbols() == [a.GetSymbol() for a in mol_h.GetAtoms()]
