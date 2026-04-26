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


def _find_c_lg_bond(mol_h: Chem.Mol, substrate_indices: list[int]) -> tuple[int, int]:
    """Locate the C and leaving-group (LG) atoms in the substrate fragment.

    Phase 0 assumes an SN2-style substrate (CH3X). The heuristic picks the
    C–heteroatom bond with the highest-Z heteroatom (F < Cl < Br < I; also
    works for O, N, S). Hybridization is not enforced because Phase 0 only
    ever sees sp3 substrates; broader inputs are out of scope (see spec §11).
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
        raise RuntimeError(
            "No C–leaving-group bond found in substrate fragment. Phase 0 expects "
            "an SN2-style substrate (sp3 C bonded to a halogen or heteroatom)."
        )
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
    """
    substrate = list(frag_indices[0])
    c_idx, lg_idx = _find_c_lg_bond(mol_h, substrate)
    c_pos = positions[c_idx]
    lg_pos = positions[lg_idx]
    c_lg = lg_pos - c_pos
    c_lg_norm = float(np.linalg.norm(c_lg))
    if c_lg_norm < 1e-6:
        raise RuntimeError("C and LG atoms coincide after MMFF — embedding is broken")
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
