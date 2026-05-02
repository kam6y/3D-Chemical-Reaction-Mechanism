"""Unit tests for embed3d placement dispatcher (Phase 3)."""
import numpy as np
import pytest
from rdkit import Chem

from reactx.bond_changes import BondChanges
from reactx.embed3d import (
    _directional_placement,
    _find_substrate_fragment,
    _place_fragments,
    embed_mol_to_atoms,
)


def _make_mol_with_frags(smiles: str) -> tuple[Chem.Mol, tuple[tuple[int, ...], ...]]:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    return mol, Chem.GetMolFrags(mol)


def test_find_substrate_fragment_unique():
    mol, frags = _make_mol_with_frags("CC(Cl).[OH-]")
    substrate = frags[0]
    other = frags[1]
    broken = ((substrate[0], substrate[1]), (substrate[1], substrate[2]))
    found = _find_substrate_fragment(frags, broken)
    assert found == substrate


def test_find_substrate_fragment_returns_none_when_broken_empty():
    mol, frags = _make_mol_with_frags("[CH3+].[Br-]")
    found = _find_substrate_fragment(frags, ())
    assert found is None


def test_find_substrate_fragment_returns_none_when_split_across_fragments():
    mol, frags = _make_mol_with_frags("CC.OO")
    a = frags[0][0]
    b = frags[1][0]
    found = _find_substrate_fragment(frags, ((a, b),))
    assert found is None


def test_place_fragments_dispatch_directional_for_sn2_shape(sn2_atoms_setup):
    mol_h, frag_indices, positions, bond_changes = sn2_atoms_setup
    out = _place_fragments(
        mol_h, frag_indices, positions.copy(), bond_changes,
        rotation_perturbation=None,
    )
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    o_idx = syms.index("O")
    c_idx = syms.index("C")
    cl_idx = syms.index("Cl")
    v_co = out[o_idx] - out[c_idx]
    v_ccl = out[cl_idx] - out[c_idx]
    cos_angle = float(np.dot(v_co, v_ccl) / (
        np.linalg.norm(v_co) * np.linalg.norm(v_ccl) + 1e-12
    ))
    assert cos_angle < -0.5, f"expected backside placement, got cos(angle)={cos_angle}"


def test_place_fragments_e2_shape_directional_anchors_on_h(e2_atoms_setup):
    mol_h, frag_indices, positions, bond_changes = e2_atoms_setup
    formed_atoms = set(bond_changes.formed[0])
    h_anchor = next(
        next(iter(formed_atoms & {a, b}))
        for a, b in bond_changes.broken
        if formed_atoms & {a, b}
    )
    out = _place_fragments(
        mol_h, frag_indices, positions.copy(), bond_changes,
        rotation_perturbation=None,
    )
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    base_frag = next(
        f for f in frag_indices
        if all(syms[i] in {"O", "H"} for i in f)
    )
    o_idx = next(i for i in base_frag if syms[i] == "O")
    d_oh = float(np.linalg.norm(out[o_idx] - out[h_anchor]))
    assert d_oh < 4.5, f"O should be close to anchor H, got {d_oh:.2f} A"


def test_place_fragments_raises_for_broken_zero_bimolecular():
    mol = Chem.AddHs(Chem.MolFromSmiles("[CH3+].[OH-]"))
    frags = Chem.GetMolFrags(mol)
    bc = BondChanges(formed=((frags[0][0], frags[1][0]),), broken=())
    with pytest.raises(NotImplementedError, match="centroid"):
        _place_fragments(
            mol, frags, np.zeros((mol.GetNumAtoms(), 3)),
            bc, rotation_perturbation=None,
        )


def test_place_fragments_raises_for_multi_substrate_metathesis():
    mol = Chem.AddHs(Chem.MolFromSmiles("CC.OO"))
    frags = Chem.GetMolFrags(mol)
    a = frags[0][0]
    b = frags[1][0]
    bc = BondChanges(formed=((frags[0][1], frags[1][1]),), broken=((a, b),))
    with pytest.raises(NotImplementedError, match="multi-substrate|centroid"):
        _place_fragments(
            mol, frags, np.zeros((mol.GetNumAtoms(), 3)),
            bc, rotation_perturbation=None,
        )


def test_place_fragments_raises_for_multi_base_on_single_anchor():
    mol = Chem.AddHs(Chem.MolFromSmiles("CC.[F-].[Cl-]"))
    frags = Chem.GetMolFrags(mol)
    c0, c1 = frags[0][0], frags[0][1]
    f_idx = frags[1][0]
    cl_idx = frags[2][0]
    bc = BondChanges(
        formed=((c0, f_idx), (c0, cl_idx)),
        broken=((c0, c1),),
    )
    with pytest.raises(NotImplementedError, match="multi-base"):
        _place_fragments(
            mol, frags, np.zeros((mol.GetNumAtoms(), 3)),
            bc, rotation_perturbation=None,
        )
