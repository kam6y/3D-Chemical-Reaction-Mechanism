"""Convert 2D RDKit Mol to 3D ase.Atoms via RDKit ETKDG + MMFF (+ optional UMA).

Preserves the atom ordering of Chem.AddHs(mol) so that downstream consumers
(align, NEB) can correlate atom indices with heavy_to_hydrogen_groups lookups
on the same AddHs(mol) result.

Multi-fragment placement is delegated to reactx.placement.place_fragments_generic,
driven by bond_changes computed in cli.py from the .rxn atom mapping.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize import BFGS
from rdkit import Chem
from rdkit.Chem import AllChem

from reactx.placement import place_fragments_generic
from reactx.reaction_topology import BondChanges

log = logging.getLogger(__name__)

MAX_EMBED_RETRIES = 5


def embed_mol_to_atoms(
    mol: Chem.Mol,
    *,
    calculator: Calculator | None = None,
    seed: int = 0xC0FFEE,
    fmax: float = 0.01,
    max_opt_steps: int = 300,
    bond_changes: BondChanges | None = None,
    side: Literal["reactant", "product"] = "reactant",
    index_translation: dict[int, int] | None = None,
    d_form: float | None = None,
    d_dissoc: float | None = None,
) -> Atoms:
    """Embed a 2D Mol into 3D and return an ase.Atoms with implicit Hs added.

    Atom order in the returned Atoms matches Chem.AddHs(mol).GetAtoms() exactly.
    For multi-fragment input, fragment placement is delegated to
    placement.place_fragments_generic, driven by bond_changes (broken/formed
    bond list extracted from the .rxn atom mapping).

    bond_changes uses reactant indexing by convention. When ``side="product"``,
    pass ``index_translation`` (typically reactx.reaction_topology
    .expanded_atom_mapping) so that the placement engine can project the
    reactant-indexed bond_changes onto the product mol_h indexing.

    bond_changes=None falls back to a naive +x displacement, sufficient for
    unit tests that don't have a paired reactant/product available.
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

    if len(frag_indices) > 1:
        bc = bond_changes if bond_changes is not None else BondChanges(broken=[], formed=[])
        place_kwargs = {"side": side, "index_translation": index_translation}
        if d_form is not None:
            place_kwargs["d_form"] = d_form
        if d_dissoc is not None:
            place_kwargs["d_dissoc"] = d_dissoc
        positions = place_fragments_generic(
            mol_h,
            frag_indices,
            positions,
            bc,
            **place_kwargs,
        )

    symbols = [a.GetSymbol() for a in mol_h.GetAtoms()]
    charges = [a.GetFormalCharge() for a in mol_h.GetAtoms()]
    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.set_initial_charges(charges)

    atoms.info["charge"] = int(sum(charges))
    atoms.info["spin"] = 1  # Phase 1: closed-shell only (UMA omol task)

    if calculator is not None:
        atoms.calc = calculator
        BFGS(atoms, logfile=None).run(fmax=fmax, steps=max_opt_steps)

    return atoms


def _embed_in_place(frag: Chem.Mol, *, seed: int) -> None:
    # Clear chirality flags first: the .rxn parser may infer a spurious
    # stereo center on charged atoms with identical substituents (e.g. N+ in
    # NH4+ becomes [N@+]), which makes distance geometry unsatisfiable.
    # Phase 1 only computes 3D from connectivity, not from R/S labels, so this
    # is safe to drop for all inputs.
    for atom in frag.GetAtoms():
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    for bond in frag.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)

    # Two stages: first try ETKDGv3 (curated knowledge), then fall back to
    # random coords. The fallback rescues highly-symmetric small species like
    # NH4+ where ETKDGv3 starting layout is degenerate.
    for use_random in (False, True):
        params = AllChem.ETKDGv3()
        params.useRandomCoords = use_random
        for attempt in range(MAX_EMBED_RETRIES):
            params.randomSeed = seed + attempt + (1000 if use_random else 0)
            if AllChem.EmbedMolecule(frag, params) == 0:
                if frag.GetNumHeavyAtoms() > 1:
                    AllChem.MMFFOptimizeMolecule(frag, maxIters=500)
                return
    elems = ", ".join(sorted({a.GetSymbol() for a in frag.GetAtoms()}))
    raise RuntimeError(
        f"RDKit failed to embed fragment ({frag.GetNumAtoms()} atoms: {elems}) "
        f"after {MAX_EMBED_RETRIES} attempts (with and without random coords). "
        "Check input structure and RDKit version."
    )
