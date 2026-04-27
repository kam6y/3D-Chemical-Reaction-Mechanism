"""Tests for reactx.placement."""

from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from reactx.placement import place_fragments_generic
from reactx.reaction_topology import compute_bond_changes, expanded_atom_mapping
from reactx.rxn_parser import parse_rxn


def _embed_for_test(mol_h, seed: int = 42):
    """Embed each fragment in isolation (no inter-fragment placement).

    Returns (frag_indices, positions) ready to feed to place_fragments_generic.
    """
    frag_mols = Chem.GetMolFrags(mol_h, asMols=True, sanitizeFrags=True)
    frag_indices = Chem.GetMolFrags(mol_h)
    positions = np.zeros((mol_h.GetNumAtoms(), 3))
    for i, (frag, idxs) in enumerate(zip(frag_mols, frag_indices, strict=True)):
        params = AllChem.ETKDGv3()
        params.randomSeed = seed + i
        AllChem.EmbedMolecule(frag, params)
        if frag.GetNumHeavyAtoms() > 1:
            AllChem.MMFFOptimizeMolecule(frag, maxIters=200)
        conf = frag.GetConformer()
        for j, orig in enumerate(idxs):
            p = conf.GetAtomPosition(j)
            positions[orig] = (p.x, p.y, p.z)
    return frag_indices, positions


def test_sn2_reactant_placement_puts_F_on_backside_of_C(sn2_rxn_path: Path):
    r_mol, p_mol, mapping = parse_rxn(sn2_rxn_path)
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)
    bc = compute_bond_changes(r_h, p_h, mapping)
    frag_indices, positions = _embed_for_test(r_h)

    placed = place_fragments_generic(
        r_h,
        frag_indices,
        positions,
        bc,
        side="reactant",
    )

    syms = [a.GetSymbol() for a in r_h.GetAtoms()]
    c = syms.index("C")
    cl = syms.index("Cl")
    f = syms.index("F")

    c_to_cl = placed[cl] - placed[c]
    c_to_f = placed[f] - placed[c]
    cos_theta = np.dot(c_to_cl, c_to_f) / (np.linalg.norm(c_to_cl) * np.linalg.norm(c_to_f))
    assert cos_theta < -0.7, f"F not on backside of C-Cl: cos(theta)={cos_theta:.3f}"

    cf_dist = float(np.linalg.norm(c_to_f))
    assert 2.5 <= cf_dist <= 4.5, f"|C-F| = {cf_dist:.3f} Å, expected ~3.5±1.0"


def test_proton_transfer_reactant_places_NH3_near_HCl_H(
    proton_transfer_rxn_path: Path,
):
    r_mol, p_mol, mapping = parse_rxn(proton_transfer_rxn_path)
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)
    bc = compute_bond_changes(r_h, p_h, mapping)
    frag_indices, positions = _embed_for_test(r_h)

    placed = place_fragments_generic(
        r_h,
        frag_indices,
        positions,
        bc,
        side="reactant",
    )
    syms = [a.GetSymbol() for a in r_h.GetAtoms()]
    n = syms.index("N")
    h_hcl = next(
        a.GetIdx() for a in r_h.GetAtoms() if a.GetSymbol() == "H" and a.GetAtomMapNum() == 1
    )

    nh_dist = float(np.linalg.norm(placed[n] - placed[h_hcl]))
    assert 2.5 <= nh_dist <= 4.5, f"|N-H(Cl)| = {nh_dist:.2f} Å, expected ~3.5±1.0"


def test_e2_reactant_places_OH_near_beta_H(e2_rxn_path: Path):
    r_mol, p_mol, mapping = parse_rxn(e2_rxn_path)
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)
    bc = compute_bond_changes(r_h, p_h, mapping)
    frag_indices, positions = _embed_for_test(r_h)

    placed = place_fragments_generic(
        r_h,
        frag_indices,
        positions,
        bc,
        side="reactant",
    )
    o = next(a.GetIdx() for a in r_h.GetAtoms() if a.GetAtomMapNum() == 4)
    h_beta = next(a.GetIdx() for a in r_h.GetAtoms() if a.GetAtomMapNum() == 6)
    oh_dist = float(np.linalg.norm(placed[o] - placed[h_beta]))
    assert 2.5 <= oh_dist <= 4.5, f"|O-Hβ| = {oh_dist:.2f} Å, expected ~3.5±1.0"


def test_sn1_reactant_places_water_near_C(sn1_rxn_path: Path):
    """SN1 hydrolysis: water O approaches Cα backside of Cα-Br in reactant."""
    r_mol, p_mol, mapping = parse_rxn(sn1_rxn_path)
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)
    bc = compute_bond_changes(r_h, p_h, mapping)
    frag_indices, positions = _embed_for_test(r_h)

    placed = place_fragments_generic(
        r_h,
        frag_indices,
        positions,
        bc,
        side="reactant",
    )
    c_central = next(a.GetIdx() for a in r_h.GetAtoms() if a.GetAtomMapNum() == 1)
    o = next(a.GetIdx() for a in r_h.GetAtoms() if a.GetAtomMapNum() == 6)
    co = float(np.linalg.norm(placed[c_central] - placed[o]))
    assert 2.5 <= co <= 4.5, f"|C-O| reactant = {co:.2f} Å, expected ~3.5±1.0"


def test_e1_step2_reactant_places_OH_near_beta_H(e1_step2_rxn_path: Path):
    """E1 step 2 with OH-: migrating beta-H is captured by OH-."""
    r_mol, p_mol, mapping = parse_rxn(e1_step2_rxn_path)
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)
    bc = compute_bond_changes(r_h, p_h, mapping)
    frag_indices, positions = _embed_for_test(r_h)
    placed = place_fragments_generic(
        r_h,
        frag_indices,
        positions,
        bc,
        side="reactant",
    )
    o = next(a.GetIdx() for a in r_h.GetAtoms() if a.GetAtomMapNum() == 5)
    h_beta = next(a.GetIdx() for a in r_h.GetAtoms() if a.GetAtomMapNum() == 6)
    oh_dist = float(np.linalg.norm(placed[o] - placed[h_beta]))
    assert 2.5 <= oh_dist <= 4.5, f"|O-Hβ| reactant = {oh_dist:.2f} Å, expected ~3.5±1.0"


def test_e1_step2_product_separates_H2O_from_alkene(e1_step2_rxn_path: Path):
    """Product: H2O fragment placed away from isobutene."""
    r_mol, p_mol, mapping = parse_rxn(e1_step2_rxn_path)
    r_h = Chem.AddHs(r_mol)
    p_h = Chem.AddHs(p_mol)
    bc = compute_bond_changes(r_h, p_h, mapping)
    expanded = expanded_atom_mapping(r_h, p_h, mapping)
    frag_indices, positions = _embed_for_test(p_h)
    placed = place_fragments_generic(
        p_h,
        frag_indices,
        positions,
        bc,
        side="product",
        index_translation=expanded,
    )
    h_migrated = next(a.GetIdx() for a in p_h.GetAtoms() if a.GetAtomMapNum() == 6)
    cb = next(a.GetIdx() for a in p_h.GetAtoms() if a.GetAtomMapNum() == 2)
    hcb = float(np.linalg.norm(placed[h_migrated] - placed[cb]))
    assert 3.0 <= hcb <= 5.0, f"|H(H2O)-Cβ| in product = {hcb:.2f} Å, expected ~4.0±1.0"
