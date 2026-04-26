"""Tests for reactx.placement."""
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from reactx.placement import place_fragments_generic
from reactx.reaction_topology import compute_bond_changes
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
        r_h, frag_indices, positions, bc, side="reactant",
    )

    syms = [a.GetSymbol() for a in r_h.GetAtoms()]
    c = syms.index("C")
    cl = syms.index("Cl")
    f = syms.index("F")

    c_to_cl = placed[cl] - placed[c]
    c_to_f = placed[f] - placed[c]
    cos_theta = (
        np.dot(c_to_cl, c_to_f)
        / (np.linalg.norm(c_to_cl) * np.linalg.norm(c_to_f))
    )
    assert cos_theta < -0.7, f"F not on backside of C-Cl: cos(theta)={cos_theta:.3f}"

    cf_dist = float(np.linalg.norm(c_to_f))
    assert 2.0 <= cf_dist <= 4.0, f"|C-F| = {cf_dist:.3f} Å, expected ~3.0±1.0"
