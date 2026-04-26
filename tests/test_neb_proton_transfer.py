"""Proton transfer end-to-end NEB test (HCl + NH3 -> Cl- + NH4+)."""
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
def test_proton_transfer_neb_H_moves_from_Cl_to_N(
    tmp_path: Path, proton_transfer_rxn_path: Path,
):
    pytest.importorskip("fairchem.core")

    r_mol, p_mol, mapping = parse_rxn(proton_transfer_rxn_path)
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)
    bc = compute_bond_changes(r_h, p_h, mapping)
    expanded = expanded_atom_mapping(r_h, p_h, mapping)

    calc = make_calculator("uma")
    reactant = embed_mol_to_atoms(
        r_mol, calculator=calc, seed=1, bond_changes=bc, side="reactant",
    )
    product_raw = embed_mol_to_atoms(
        p_mol, calculator=calc, seed=2, bond_changes=bc, side="product",
        index_translation=expanded,
    )
    product = align_product_to_reactant(
        reactant, product_raw, expanded,
        swappable_h_groups=build_swappable_h_groups(r_h),
    )

    out = tmp_path / "traj.xyz"
    meta = run_neb(
        reactant=reactant, product=product, calculator=calc,
        n_images=13, output_xyz=out, fmax=0.05, max_steps=200, pad_frames=0,
    )
    frames = read(str(out), index=":")
    syms = frames[0].get_chemical_symbols()
    cl = syms.index("Cl")
    n = syms.index("N")
    h_mig = next(a.GetIdx() for a in r_h.GetAtoms() if a.GetAtomMapNum() == 1)

    h_to_cl_r = frames[0].get_distance(h_mig, cl)
    h_to_n_r = frames[0].get_distance(h_mig, n)
    h_to_cl_p = frames[-1].get_distance(h_mig, cl)
    h_to_n_p = frames[-1].get_distance(h_mig, n)
    assert h_to_cl_r < 1.5
    assert h_to_n_p < 1.3
    assert h_to_cl_p > h_to_cl_r
    assert h_to_n_p < h_to_n_r

    energies = np.array(meta["image_energies"])
    ts_idx = int(np.argmax(energies))
    assert 0 < ts_idx < len(frames) - 1
