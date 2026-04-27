"""SN1 hydrolysis end-to-end NEB test.

Concerted (CH3)3CBr + H2O -> (CH3)3COH + HBr — 2 broken / 2 formed.
"""

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
def test_sn1_hydrolysis_neb(tmp_path: Path, sn1_rxn_path: Path):
    pytest.importorskip("fairchem.core")

    r_mol, p_mol, mapping = parse_rxn(sn1_rxn_path)
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
        n_images=15,
        output_xyz=out,
        fmax=0.05,
        max_steps=400,
        pad_frames=0,
    )
    frames = read(str(out), index=":")
    c = next(a.GetIdx() for a in r_h.GetAtoms() if a.GetAtomMapNum() == 1)
    br = next(a.GetIdx() for a in r_h.GetAtoms() if a.GetAtomMapNum() == 2)
    o = next(a.GetIdx() for a in r_h.GetAtoms() if a.GetAtomMapNum() == 6)
    h_mig = next(a.GetIdx() for a in r_h.GetAtoms() if a.GetAtomMapNum() == 8)
    cbr_r = frames[0].get_distance(c, br)
    cbr_p = frames[-1].get_distance(c, br)
    co_r = frames[0].get_distance(c, o)
    co_p = frames[-1].get_distance(c, o)
    hbr_p = frames[-1].get_distance(h_mig, br)
    assert cbr_p > cbr_r * 1.4, f"C-Br should lengthen: {cbr_r:.2f} -> {cbr_p:.2f}"
    assert co_p < co_r, f"C-O should form: {co_r:.2f} -> {co_p:.2f}"
    assert hbr_p < 2.0, f"H-Br should form: {hbr_p:.2f} (expected ~1.4)"
