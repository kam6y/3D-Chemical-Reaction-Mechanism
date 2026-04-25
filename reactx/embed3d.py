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
    fmax: float = 0.01,
    max_opt_steps: int = 300,
) -> Atoms:
    """Embed a 2D Mol into 3D and return an ase.Atoms with implicit Hs added.

    Atom order in the returned Atoms matches Chem.AddHs(mol).GetAtoms() exactly.
    For disconnected fragments, each is embedded independently and then placed
    for backside attack geometry: the first fragment defines the substrate, and
    subsequent fragments (nucleophile/leaving group) are placed on the -x side
    of the first fragment so they approach anti to the leaving group (which MMFF
    typically places along +x). This gives UMA a correct SN2 encounter complex.
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
        positions = _place_nucleophile_backside(mol_h, frag_indices, positions)

    symbols = [a.GetSymbol() for a in mol_h.GetAtoms()]
    charges = [a.GetFormalCharge() for a in mol_h.GetAtoms()]
    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.set_initial_charges(charges)

    atoms.info["charge"] = int(sum(charges))
    atoms.info["spin"] = 1  # Phase 0: assume closed-shell singlet

    if calculator is not None:
        atoms.calc = calculator
        BFGS(atoms, logfile=None).run(fmax=fmax, steps=max_opt_steps)

    return atoms


def _find_c_lg_bond(mol_h: Chem.Mol, substrate_indices: list[int]) -> tuple[int, int] | None:
    """Locate the C and leaving-group (LG) atoms in the substrate fragment.

    LG heuristic: heavy atom (Z > 1) bonded to sp3 carbon with highest atomic
    number (halogens F < Cl < Br < I; also works for O, N, S). Returns
    (c_idx, lg_idx) in mol_h coordinates, or None if no C-LG bond is found.
    """
    best: tuple[int, int, int] | None = None  # (Z, c_idx, lg_idx)
    substrate_set = set(substrate_indices)
    for atom in mol_h.GetAtoms():
        if atom.GetIdx() not in substrate_set or atom.GetSymbol() != "C":
            continue
        for nb in atom.GetNeighbors():
            if nb.GetIdx() not in substrate_set or nb.GetAtomicNum() <= 1:
                continue
            if nb.GetSymbol() == "C":
                continue
            z = nb.GetAtomicNum()
            if best is None or z > best[0]:
                best = (z, atom.GetIdx(), nb.GetIdx())
    if best is None:
        return None
    return best[1], best[2]


def _place_nucleophile_backside(
    mol_h: Chem.Mol,
    frag_indices: tuple[tuple[int, ...], ...],
    positions: np.ndarray,
) -> np.ndarray:
    """Place nucleophile fragment(s) along -(C->LG) from the attacked carbon.

    Uses the geometry of the substrate (fragment 0) to find the C-LG bond
    direction, then places each nucleophile fragment's centroid at
    ``C + (-unit(C->LG)) * FRAGMENT_SEPARATION``.

    Falls back to placing nucleophile on substrate's -x side when no C-LG
    bond can be identified (e.g., non-SN2 substrate without halogen/heteroatom
    leaving group).
    """
    substrate = list(frag_indices[0])
    bond = _find_c_lg_bond(mol_h, substrate)
    if bond is None:
        sub_min_x = positions[substrate, 0].min()
        for i in range(1, len(frag_indices)):
            nuc = list(frag_indices[i])
            nuc_max_x = positions[nuc, 0].max()
            positions[nuc, 0] += sub_min_x - FRAGMENT_SEPARATION - nuc_max_x
        return positions

    c_idx, lg_idx = bond
    c_pos = positions[c_idx]
    lg_pos = positions[lg_idx]
    c_lg = lg_pos - c_pos
    c_lg_norm = float(np.linalg.norm(c_lg))
    if c_lg_norm < 1e-6:
        return positions
    backside = -c_lg / c_lg_norm

    for i in range(1, len(frag_indices)):
        nuc = list(frag_indices[i])
        nuc_centroid = positions[nuc].mean(axis=0)
        target = c_pos + backside * FRAGMENT_SEPARATION
        positions[nuc] += target - nuc_centroid
    return positions


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
