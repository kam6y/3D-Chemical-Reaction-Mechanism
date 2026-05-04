"""2D RDKit Mol -> per-fragment 3D coordinates.

Phase 7: this module's responsibility is reduced to the per-fragment 3D
embed only. Multi-fragment placement (Tier 1 / Tier 2 dispatch) moved to
reactx.placement.valid_placements. The Atoms construction step moved to
reactx.placement.build_atoms_from_positions.

Output coordinates may have multiple fragments overlapping in space - the
caller is expected to call placement.valid_placements next.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

MAX_EMBED_RETRIES = 5


def embed_fragments_to_positions(
    mol: Chem.Mol,
    *,
    seed: int = 0xC0FFEE,
) -> tuple[Chem.Mol, tuple[tuple[int, ...], ...], np.ndarray]:
    """Run ETKDGv3 + MMFF94 per fragment; return mol_h, frag_indices, positions.

    Atom ordering of the returned mol_h is `Chem.AddHs(mol)` order; the
    positions array maps directly onto its atom indices.
    """
    mol_h = Chem.AddHs(mol)
    n_atoms = mol_h.GetNumAtoms()
    frag_indices = Chem.GetMolFrags(mol_h)
    frag_mols = Chem.GetMolFrags(mol_h, asMols=True, sanitizeFrags=True)

    positions = np.zeros((n_atoms, 3))
    for i, (indices, frag) in enumerate(zip(frag_indices, frag_mols, strict=True)):
        _embed_in_place(frag, seed=seed + i * MAX_EMBED_RETRIES)
        conf = frag.GetConformer()
        for j, orig_idx in enumerate(indices):
            p = conf.GetAtomPosition(j)
            positions[orig_idx] = (p.x, p.y, p.z)

    return mol_h, frag_indices, positions


def _embed_in_place(frag: Chem.Mol, *, seed: int) -> None:
    params = AllChem.ETKDGv3()
    for attempt in range(MAX_EMBED_RETRIES):
        params.randomSeed = seed + attempt
        status = AllChem.EmbedMolecule(frag, params)
        if status == 0:
            break
    else:
        elems = ", ".join(sorted({a.GetSymbol() for a in frag.GetAtoms()}))
        raise RuntimeError(
            f"RDKit failed to embed fragment ({frag.GetNumAtoms()} atoms: {elems}) "
            f"after {MAX_EMBED_RETRIES} attempts. Check input structure and RDKit version."
        )

    if frag.GetNumHeavyAtoms() > 1:
        result = AllChem.MMFFOptimizeMolecule(frag, maxIters=500)
        if result == -1:
            raise RuntimeError(
                f"MMFF94 force field could not be constructed for fragment "
                f"({frag.GetNumAtoms()} atoms). Check element coverage."
            )
