import pytest
from ase import Atoms
from rdkit import Chem

from reactx.embed3d import embed_mol_to_atoms


def _ch3cl() -> Chem.Mol:
    mol = Chem.MolFromSmiles("CCl")
    return mol


def test_embed_ch3cl_returns_atoms_with_all_atoms():
    atoms = embed_mol_to_atoms(_ch3cl(), calculator=None, seed=42)
    assert isinstance(atoms, Atoms)
    # C + Cl + 3 H = 5
    assert len(atoms) == 5
    syms = sorted(atoms.get_chemical_symbols())
    assert syms == ["C", "Cl", "H", "H", "H"]


def test_embed_ch3cl_has_reasonable_c_cl_bond():
    atoms = embed_mol_to_atoms(_ch3cl(), calculator=None, seed=42)
    syms = atoms.get_chemical_symbols()
    c_idx = syms.index("C")
    cl_idx = syms.index("Cl")
    d = atoms.get_distance(c_idx, cl_idx)
    assert 1.6 < d < 2.0, f"C-Cl distance out of range: {d:.3f} Å"


def test_embed_ch3cl_has_reasonable_hch_angles():
    atoms = embed_mol_to_atoms(_ch3cl(), calculator=None, seed=42)
    syms = atoms.get_chemical_symbols()
    c_idx = syms.index("C")
    h_idxs = [i for i, s in enumerate(syms) if s == "H"]
    assert len(h_idxs) == 3
    angles = [
        atoms.get_angle(h_idxs[i], c_idx, h_idxs[j]) for i in range(3) for j in range(i + 1, 3)
    ]
    for a in angles:
        assert 100 < a < 120, f"H-C-H angle out of range: {a:.1f}°"


def test_embed_multifragment_places_fragments_apart():
    mol = Chem.MolFromSmiles("CCl.[F-]")
    atoms = embed_mol_to_atoms(mol, calculator=None, seed=42)
    syms = atoms.get_chemical_symbols()
    assert "F" in syms
    f_idx = syms.index("F")
    c_idx = syms.index("C")
    d = atoms.get_distance(c_idx, f_idx)
    assert d > 2.5, f"Fragments too close: C-F distance {d:.3f} Å"


def test_embed_failure_raises_runtime_error(monkeypatch):
    from reactx import embed3d as mod

    def always_fail(*_a, **_kw):
        return -1  # RDKit embed failure code

    monkeypatch.setattr(mod.AllChem, "EmbedMolecule", always_fail)
    with pytest.raises(RuntimeError, match="embed"):
        embed_mol_to_atoms(_ch3cl(), calculator=None, seed=1)


def test_embed_multifragment_preserves_addhs_ordering():
    """Critical invariant: embed output atom order must match Chem.AddHs(mol)
    so downstream align + heavy_to_hydrogen_groups can share indices."""
    mol = Chem.MolFromSmiles("CCl.[F-]")
    mol_h = Chem.AddHs(mol)
    expected_symbols = [a.GetSymbol() for a in mol_h.GetAtoms()]

    atoms = embed_mol_to_atoms(mol, calculator=None, seed=42)
    assert atoms.get_chemical_symbols() == expected_symbols


def test_embed_sets_total_charge_and_spin_in_info():
    mol = Chem.MolFromSmiles("CCl.[F-]")
    atoms = embed_mol_to_atoms(mol, calculator=None, seed=42)
    assert atoms.info["charge"] == -1
    assert atoms.info["spin"] == 1


def test_embed_neutral_molecule_has_zero_charge():
    mol = Chem.MolFromSmiles("CCl")
    atoms = embed_mol_to_atoms(mol, calculator=None, seed=42)
    assert atoms.info["charge"] == 0
    assert atoms.info["spin"] == 1
