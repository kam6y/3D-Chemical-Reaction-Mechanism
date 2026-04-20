"""Convert 2D RDKit Mol to 3D ase.Atoms via RDKit ETKDG + MMFF (+ optional UMA)."""
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

    If the Mol is disconnected (multiple fragments), each fragment is embedded
    independently and then concatenated along the +x axis with FRAGMENT_SEPARATION
    spacing so that UMA can later relax into a sensible encounter complex.
    """
    mol_h = Chem.AddHs(mol)
    frags = Chem.GetMolFrags(mol_h, asMols=True, sanitizeFrags=True)

    atoms_per_frag = [_embed_single_fragment(f, seed=seed + i)
                      for i, f in enumerate(frags)]
    atoms = _stack_fragments(atoms_per_frag)

    if calculator is not None:
        atoms.calc = calculator
        BFGS(atoms, logfile=None).run(fmax=fmax, steps=max_opt_steps)

    return atoms


def _embed_single_fragment(frag: Chem.Mol, *, seed: int) -> Atoms:
    params = AllChem.ETKDGv3()
    for attempt in range(MAX_EMBED_RETRIES):
        params.randomSeed = seed + attempt
        status = AllChem.EmbedMolecule(frag, params)
        if status == 0:
            break
    else:
        raise RuntimeError(
            f"RDKit failed to embed fragment after {MAX_EMBED_RETRIES} attempts. "
            "Check the input structure and RDKit version."
        )

    if frag.GetNumHeavyAtoms() > 1:
        AllChem.MMFFOptimizeMolecule(frag, maxIters=500)

    return _rdkit_to_atoms(frag)


def _rdkit_to_atoms(mol: Chem.Mol) -> Atoms:
    conf = mol.GetConformer()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    positions = np.array([[conf.GetAtomPosition(i).x,
                           conf.GetAtomPosition(i).y,
                           conf.GetAtomPosition(i).z]
                          for i in range(mol.GetNumAtoms())])
    charges = [a.GetFormalCharge() for a in mol.GetAtoms()]
    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.set_initial_charges(charges)
    return atoms


def _stack_fragments(frags: list[Atoms]) -> Atoms:
    if len(frags) == 1:
        return frags[0]
    stacked = frags[0].copy()
    for f in frags[1:]:
        offset = stacked.positions[:, 0].max() - f.positions[:, 0].min() + FRAGMENT_SEPARATION
        f2 = f.copy()
        f2.positions[:, 0] += offset
        stacked += f2
    return stacked
