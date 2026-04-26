"""E2 elimination end-to-end NEB test."""
from pathlib import Path

import numpy as np
import pytest
from ase.io import read
from rdkit import Chem

from reactx.align import align_product_to_reactant, build_swappable_h_groups
from reactx.calculators import make_calculator
from reactx.cli import recommend_n_images
from reactx.embed3d import embed_mol_to_atoms
from reactx.neb import run_neb
from reactx.reaction_topology import compute_bond_changes, expanded_atom_mapping
from reactx.rxn_parser import parse_rxn


@pytest.mark.slow
def test_e2_neb_breaks_CBr_and_CbetaH_in_concert(tmp_path: Path, e2_rxn_path: Path):
    pytest.importorskip("fairchem.core")

    r_mol, p_mol, mapping = parse_rxn(e2_rxn_path)
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
        n_images=recommend_n_images(bc),
        output_xyz=out, fmax=0.05, max_steps=300, pad_frames=0,
    )
    frames = read(str(out), index=":")
    energies = np.array(meta["image_energies"])
    ts_idx = int(np.argmax(energies))
    assert 0 < ts_idx < len(frames) - 1

    syms = frames[0].get_chemical_symbols()
    c_alpha = next(a.GetIdx() for a in r_h.GetAtoms() if a.GetAtomMapNum() == 1)
    c_beta = next(a.GetIdx() for a in r_h.GetAtoms() if a.GetAtomMapNum() == 2)
    br = syms.index("Br")
    h_beta = next(a.GetIdx() for a in r_h.GetAtoms() if a.GetAtomMapNum() == 6)

    ts = frames[ts_idx]
    cbr_ts = ts.get_distance(c_alpha, br)
    cbh_ts = ts.get_distance(c_beta, h_beta)
    cbr_r = frames[0].get_distance(c_alpha, br)
    cbh_r = frames[0].get_distance(c_beta, h_beta)
    assert cbr_ts > cbr_r * 1.15, f"C-Br at TS not elongated: {cbr_ts:.2f} vs {cbr_r:.2f}"
    assert cbh_ts > cbh_r * 1.15, f"Cβ-H at TS not elongated: {cbh_ts:.2f} vs {cbh_r:.2f}"
