import numpy as np
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
    angles = [atoms.get_angle(h_idxs[i], c_idx, h_idxs[j])
              for i in range(3) for j in range(i + 1, 3)]
    for a in angles:
        assert 100 < a < 120, f"H-C-H angle out of range: {a:.1f}°"


def test_embed_multifragment_places_fragments_apart():
    from reactx.bond_changes import BondChanges
    mol = Chem.MolFromSmiles("CCl.[F-]")
    mol_h = Chem.AddHs(mol)
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    c_idx = syms.index("C")
    cl_idx = syms.index("Cl")
    f_idx = syms.index("F")
    bc = BondChanges(formed=((c_idx, f_idx),), broken=((c_idx, cl_idx),))

    atoms = embed_mol_to_atoms(mol, calculator=None, seed=42, bond_changes=bc)
    syms_out = atoms.get_chemical_symbols()
    f_out = syms_out.index("F")
    c_out = syms_out.index("C")
    d = atoms.get_distance(c_out, f_out)
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
    from reactx.bond_changes import BondChanges
    mol = Chem.MolFromSmiles("CCl.[F-]")
    mol_h = Chem.AddHs(mol)
    expected_symbols = [a.GetSymbol() for a in mol_h.GetAtoms()]
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    bc = BondChanges(
        formed=((syms.index("C"), syms.index("F")),),
        broken=((syms.index("C"), syms.index("Cl")),),
    )
    atoms = embed_mol_to_atoms(mol, calculator=None, seed=42, bond_changes=bc)
    assert atoms.get_chemical_symbols() == expected_symbols


def test_embed_sets_total_charge_and_spin_in_info():
    from reactx.bond_changes import BondChanges
    mol = Chem.MolFromSmiles("CCl.[F-]")
    mol_h = Chem.AddHs(mol)
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    bc = BondChanges(
        formed=((syms.index("C"), syms.index("F")),),
        broken=((syms.index("C"), syms.index("Cl")),),
    )
    atoms = embed_mol_to_atoms(mol, calculator=None, seed=42, bond_changes=bc)
    assert atoms.info["charge"] == -1
    assert atoms.info["spin"] == 1


def test_embed_neutral_molecule_has_zero_charge():
    mol = Chem.MolFromSmiles("CCl")
    atoms = embed_mol_to_atoms(mol, calculator=None, seed=42)
    assert atoms.info["charge"] == 0
    assert atoms.info["spin"] == 1


def test_embed_rotation_perturbation_changes_nucleophile_position():
    """A non-identity rotation_perturbation should move the nucleophile fragment
    relative to the identity case."""
    from reactx.bond_changes import BondChanges
    mol = Chem.MolFromSmiles("CCl.[F-]")
    mol_h = Chem.AddHs(mol)
    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    bc = BondChanges(
        formed=((syms.index("C"), syms.index("F")),),
        broken=((syms.index("C"), syms.index("Cl")),),
    )
    a_id = embed_mol_to_atoms(mol, calculator=None, seed=42, bond_changes=bc)
    # 30° rotation around y-axis
    theta = np.deg2rad(30.0)
    R = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)],
    ])
    a_pert = embed_mol_to_atoms(
        mol, calculator=None, seed=42, bond_changes=bc, rotation_perturbation=R,
    )
    # F position should differ between identity and perturbed
    f_idx = a_id.get_chemical_symbols().index("F")
    diff = np.linalg.norm(a_id.positions[f_idx] - a_pert.positions[f_idx])
    assert diff > 0.5, f"F position barely moved ({diff:.3f} Å) under 30° perturbation"


def test_embed_proton_transfer_substrate_is_hcl_fragment():
    """For HCl + NH3 → Cl- + NH4+, the broken bond is H-Cl. The substrate
    fragment must be HCl (containing both H and Cl), even though NH3 has
    more atoms after AddHs."""
    from reactx.bond_changes import BondChanges
    from reactx.rxn_parser import parse_rxn
    r_mol, _, _ = parse_rxn("examples/proton_transfer.rxn")
    r_h = Chem.AddHs(r_mol)
    syms = [a.GetSymbol() for a in r_h.GetAtoms()]
    n_idx = syms.index("N")
    cl_idx = syms.index("Cl")
    proton_idx = next(
        a.GetIdx() for a in r_h.GetAtoms() if a.GetAtomMapNum() == 1
    )
    bc = BondChanges(formed=((n_idx, proton_idx),), broken=((proton_idx, cl_idx),))
    atoms = embed_mol_to_atoms(r_mol, calculator=None, seed=42, bond_changes=bc)

    # After placement, N should be on the opposite side of Cl from the proton
    # (= backside of H from Cl). Roughly: vec(H->N) ≈ -vec(H->Cl)
    p_h = atoms.positions[proton_idx]
    p_cl = atoms.positions[cl_idx]
    p_n = atoms.positions[n_idx]
    h_to_cl = p_cl - p_h
    h_to_n = p_n - p_h
    cos_theta = float(np.dot(h_to_cl, h_to_n) / (
        np.linalg.norm(h_to_cl) * np.linalg.norm(h_to_n)
    ))
    assert cos_theta < -0.7, (
        f"N is not on backside of H from Cl: cos(angle Cl-H-N)={cos_theta:.3f}"
    )
