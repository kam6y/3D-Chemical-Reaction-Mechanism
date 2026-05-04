import numpy as np
from rdkit import Chem

from reactx.embed3d import embed_fragments_to_positions


def _ch3cl() -> Chem.Mol:
    return Chem.MolFromSmiles("CCl")


def test_embed_fragments_unimolecular_returns_atom_count_positions():
    mol_h, frags, positions = embed_fragments_to_positions(_ch3cl(), seed=42)
    assert mol_h.GetNumAtoms() == 5
    assert len(frags) == 1
    assert positions.shape == (5, 3)


def test_embed_fragments_unimolecular_has_reasonable_c_cl_bond():
    mol_h, _, positions = embed_fragments_to_positions(_ch3cl(), seed=42)
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    c_idx = syms.index("C")
    cl_idx = syms.index("Cl")
    d = float(np.linalg.norm(positions[c_idx] - positions[cl_idx]))
    assert 1.6 < d < 2.0, f"C-Cl distance out of range: {d:.3f} A"


def test_embed_fragments_multifragment_each_independent():
    mol = Chem.MolFromSmiles("CCl.[F-]")
    mol_h, frags, positions = embed_fragments_to_positions(mol, seed=42)
    assert len(frags) == 2
    assert positions.shape == (mol_h.GetNumAtoms(), 3)
    # 各 fragment の重心は別座標 (内部 ETKDG が独立に embed したため、
    # 平均原点近くに集まる; 重なりは valid_placements 側で動かす)。


def test_embed_fragments_seed_determinism():
    mol_h_a, frags_a, pos_a = embed_fragments_to_positions(_ch3cl(), seed=7)
    mol_h_b, frags_b, pos_b = embed_fragments_to_positions(_ch3cl(), seed=7)
    np.testing.assert_array_equal(pos_a, pos_b)
