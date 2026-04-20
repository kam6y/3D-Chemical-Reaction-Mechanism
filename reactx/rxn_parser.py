"""Parse MDL .rxn files into combined RDKit Mols and atom mapping."""
from __future__ import annotations

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem


def parse_rxn(rxn_path: str | Path) -> tuple[Chem.Mol, Chem.Mol, dict[int, int]]:
    """Parse an MDL Rxn file and return combined reactant/product Mols + mapping.

    The returned reactant Mol contains all reactant fragments combined via
    Chem.CombineMols (preserves atom order across fragments). The mapping dict
    maps each reactant heavy-atom index to the corresponding product heavy-atom
    index based on the AtomMapNum field present in the .rxn file.

    Raises ValueError when any heavy atom on either side is missing an atom
    map number.
    """
    rxn = AllChem.ReactionFromRxnFile(str(rxn_path))
    if rxn is None:
        raise ValueError(f"Failed to parse .rxn file: {rxn_path}")

    reactant = _combine_fragments(list(rxn.GetReactants()))
    product = _combine_fragments(list(rxn.GetProducts()))

    r_map = _collect_atom_map_numbers(reactant, side="reactant")
    p_map = _collect_atom_map_numbers(product, side="product")

    mapping: dict[int, int] = {}
    for mapnum, r_idx in r_map.items():
        if mapnum not in p_map:
            raise ValueError(
                f"Atom map number {mapnum} present in reactant but not in product"
            )
        mapping[r_idx] = p_map[mapnum]

    for mapnum in p_map:
        if mapnum not in r_map:
            raise ValueError(
                f"Atom map number {mapnum} present in product but not in reactant"
            )

    return reactant, product, mapping


def _combine_fragments(mols: list[Chem.Mol]) -> Chem.Mol:
    if not mols:
        raise ValueError("No fragments found on one side of the reaction")
    combined = mols[0]
    for m in mols[1:]:
        combined = Chem.CombineMols(combined, m)
    return combined


def _collect_atom_map_numbers(mol: Chem.Mol, *, side: str) -> dict[int, int]:
    result: dict[int, int] = {}
    for atom in mol.GetAtoms():
        mapnum = atom.GetAtomMapNum()
        if mapnum == 0:
            raise ValueError(
                f"{side} atom {atom.GetIdx()} ({atom.GetSymbol()}) has no atom map number"
            )
        if mapnum in result:
            raise ValueError(
                f"Duplicate atom map number {mapnum} on {side} side"
            )
        result[mapnum] = atom.GetIdx()
    return result


def heavy_to_hydrogen_groups(mol_with_h: Chem.Mol) -> dict[int, list[int]]:
    """Return {heavy_atom_idx: [bonded_h_idx, ...]} for a Mol with explicit Hs.

    Use after Chem.AddHs so that hydrogen indices correspond to positions in the
    Atoms object produced by embed3d._rdkit_to_atoms (heavy atoms + attached Hs
    in the original mol_h ordering).
    """
    groups: dict[int, list[int]] = {}
    for atom in mol_with_h.GetAtoms():
        if atom.GetSymbol() == "H":
            continue
        heavy_idx = atom.GetIdx()
        hs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetSymbol() == "H"]
        groups[heavy_idx] = hs
    return groups
