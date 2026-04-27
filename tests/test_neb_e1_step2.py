"""E1 step 2 (β-H elimination from carbocation) end-to-end NEB test."""

from pathlib import Path

import pytest
from ase.io import read
from rdkit import Chem

from reactx.align import align_product_to_reactant, build_swappable_h_groups
from reactx.calculators import make_calculator
from reactx.embed3d import embed_mol_to_atoms
from reactx.neb import run_neb
from reactx.reaction_topology import compute_bond_changes, expanded_atom_mapping
from reactx.rxn_parser import parse_rxn


@pytest.mark.slow
def test_e1_step2_neb_extracts_beta_H(tmp_path: Path, e1_step2_rxn_path: Path):
    pytest.importorskip("fairchem.core")

    r_mol, p_mol, mapping = parse_rxn(e1_step2_rxn_path)
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)
    bc = compute_bond_changes(r_h, p_h, mapping)
    expanded = expanded_atom_mapping(r_h, p_h, mapping)

    calc = make_calculator("uma")
    reactant = embed_mol_to_atoms(
        r_mol,
        calculator=calc,
        seed=1,
        bond_changes=bc,
        side="reactant",
    )
    product_raw = embed_mol_to_atoms(
        p_mol,
        calculator=calc,
        seed=2,
        bond_changes=bc,
        side="product",
        index_translation=expanded,
    )
    product = align_product_to_reactant(
        reactant,
        product_raw,
        expanded,
        swappable_h_groups=build_swappable_h_groups(r_h),
    )

    out = tmp_path / "traj.xyz"
    run_neb(
        reactant=reactant,
        product=product,
        calculator=calc,
        n_images=13,
        output_xyz=out,
        fmax=0.05,
        max_steps=200,
        pad_frames=0,
    )
    frames = read(str(out), index=":")
    c_alpha = next(a.GetIdx() for a in r_h.GetAtoms() if a.GetAtomMapNum() == 1)
    c_beta = next(a.GetIdx() for a in r_h.GetAtoms() if a.GetAtomMapNum() == 2)
    o = next(a.GetIdx() for a in r_h.GetAtoms() if a.GetAtomMapNum() == 5)
    h_beta = next(a.GetIdx() for a in r_h.GetAtoms() if a.GetAtomMapNum() == 6)

    cab_r = frames[0].get_distance(c_alpha, c_beta)
    cab_p = frames[-1].get_distance(c_alpha, c_beta)
    cbh_r = frames[0].get_distance(c_beta, h_beta)
    cbh_p = frames[-1].get_distance(c_beta, h_beta)
    oh_r = frames[0].get_distance(o, h_beta)
    oh_p = frames[-1].get_distance(o, h_beta)
    assert cab_p < cab_r, f"Cα-Cβ should shorten (alkene formation): {cab_r:.2f} -> {cab_p:.2f}"
    assert cbh_p > cbh_r * 1.5, f"Cβ-H should break: {cbh_r:.2f} -> {cbh_p:.2f}"
    assert oh_p < oh_r, f"O-H should form: {oh_r:.2f} -> {oh_p:.2f}"
