"""Tests for Phase 11 placement candidate generation."""
from pathlib import Path

import numpy as np
from rdkit import Chem

import reactx.placement as placement
from reactx.bond_changes import BondChanges
from reactx.embed3d import embed_fragments_to_positions
from reactx.placement import build_atoms_from_positions, valid_placements
from reactx.rxn_parser import parse_rxn

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _r_inputs(rxn_name: str):
    r_mol, p_mol, heavy_mapping = parse_rxn(EXAMPLES / f"{rxn_name}.rxn")
    mol_h, _, base_positions = embed_fragments_to_positions(r_mol, seed=0)
    bc = BondChanges.from_reaction_diff(r_mol, p_mol)
    return mol_h, base_positions, bc, heavy_mapping


def test_legacy_simple_placement_api_removed():
    assert not hasattr(placement, "simple_placement")


def test_valid_placements_sn2_backside_candidates_survive():
    mol_h, base_positions, bc, _ = _r_inputs("sn2")
    result = valid_placements(
        mol_h,
        Chem.GetMolFrags(mol_h),
        base_positions,
        bc,
        n_candidates=64,
        seed=0,
    )

    assert result.n_candidates == 64
    assert len(result.trials) > 0
    assert result.n_blocked > 0

    c_idx, cl_idx = bc.broken[0]
    c_idx_formed, o_idx = bc.formed[0]
    assert c_idx == c_idx_formed

    angles = []
    for trial in result.trials:
        v_ccl = trial.positions[cl_idx] - trial.positions[c_idx]
        v_co = trial.positions[o_idx] - trial.positions[c_idx]
        cos_theta = (v_ccl @ v_co) / (np.linalg.norm(v_ccl) * np.linalg.norm(v_co))
        angles.append(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))

    assert max(angles) >= 150.0


def test_build_atoms_from_positions_preserves_symbols():
    mol_h, base_positions, _, _ = _r_inputs("sn2")
    atoms = build_atoms_from_positions(mol_h, base_positions)
    assert len(atoms) == mol_h.GetNumAtoms()
    assert atoms.get_chemical_symbols() == [a.GetSymbol() for a in mol_h.GetAtoms()]
