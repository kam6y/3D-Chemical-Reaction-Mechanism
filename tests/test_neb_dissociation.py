"""SN1 step 1 (heterolytic dissociation) end-to-end NEB test."""

from pathlib import Path

import numpy as np
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
def test_dissociation_neb_separates_C_and_Br(tmp_path: Path, sn1_step1_rxn_path: Path):
    pytest.importorskip("fairchem.core")

    r_mol, p_mol, mapping = parse_rxn(sn1_step1_rxn_path)
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
    meta = run_neb(
        reactant=reactant,
        product=product,
        calculator=calc,
        n_images=11,
        output_xyz=out,
        fmax=0.05,
        max_steps=200,
        pad_frames=0,
    )
    frames = read(str(out), index=":")
    syms = frames[0].get_chemical_symbols()
    c = next(a.GetIdx() for a in r_h.GetAtoms() if a.GetAtomMapNum() == 1)
    br = syms.index("Br")
    d_start = frames[0].get_distance(c, br)
    d_end = frames[-1].get_distance(c, br)
    assert d_start < 2.3, f"reactant C-Br too long: {d_start:.2f}"
    assert d_end > 3.0, f"product C-Br too short: {d_end:.2f}"
    energies = np.array(meta["image_energies"])
    assert energies[-1] > energies[0] - 0.5
