"""Convert 2D RDKit Mol to 3D ase.Atoms via RDKit ETKDG + MMFF (+ optional UMA).

Preserves the atom ordering of Chem.AddHs(mol) so that downstream consumers
(align, NEB) can correlate atom indices with heavy_to_hydrogen_groups lookups
on the same AddHs(mol) result.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize import BFGS
from rdkit import Chem
from rdkit.Chem import AllChem

MAX_EMBED_RETRIES = 5
FRAGMENT_SEPARATION = 3.5  # Å — attack distance for multi-fragment placement


def embed_mol_to_atoms(
    mol: Chem.Mol,
    *,
    calculator: Optional[Calculator] = None,
    seed: int = 0xC0FFEE,
    fmax: float = 0.05,
    max_opt_steps: int = 200,
) -> Atoms:
    """Embed a 2D Mol into 3D and return an ase.Atoms with implicit Hs added.

    Atom order in the returned Atoms matches Chem.AddHs(mol).GetAtoms() exactly.
    For disconnected fragments, each is embedded independently and then shifted
    along +x so fragments are separated by FRAGMENT_SEPARATION between their
    nearest x-extents, giving UMA a sensible encounter complex to relax.
    """
    mol_h = Chem.AddHs(mol)
    n_atoms = mol_h.GetNumAtoms()

    frag_indices = Chem.GetMolFrags(mol_h)
    frag_mols = Chem.GetMolFrags(mol_h, asMols=True, sanitizeFrags=True)

    positions = np.zeros((n_atoms, 3))
    for i, (indices, frag) in enumerate(zip(frag_indices, frag_mols)):
        _embed_in_place(frag, seed=seed + i * MAX_EMBED_RETRIES)
        conf = frag.GetConformer()
        for j, orig_idx in enumerate(indices):
            p = conf.GetAtomPosition(j)
            positions[orig_idx] = (p.x, p.y, p.z)

    if len(frag_indices) > 1:
        prev_group = list(frag_indices[0])
        for i in range(1, len(frag_indices)):
            curr_group = list(frag_indices[i])
            prev_max_x = positions[prev_group, 0].max()
            curr_min_x = positions[curr_group, 0].min()
            offset = prev_max_x - curr_min_x + FRAGMENT_SEPARATION
            positions[curr_group, 0] += offset
            prev_group = curr_group

    symbols = [a.GetSymbol() for a in mol_h.GetAtoms()]
    charges = [a.GetFormalCharge() for a in mol_h.GetAtoms()]
    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.set_initial_charges(charges)

    if calculator is not None:
        atoms.calc = calculator
        BFGS(atoms, logfile=None).run(fmax=fmax, steps=max_opt_steps)

    return atoms


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
