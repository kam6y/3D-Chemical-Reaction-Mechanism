"""Shared pytest fixtures for reactx tests."""
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from reactx.bond_changes import BondChanges


@pytest.fixture()
def examples_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture()
def sn2_rxn_path(examples_dir: Path) -> Path:
    return examples_dir / "sn2.rxn"


@pytest.fixture()
def menshutkin_rxn_path(examples_dir: Path) -> Path:
    return examples_dir / "menshutkin.rxn"


@pytest.fixture()
def e2_rxn_path(examples_dir: Path) -> Path:
    return examples_dir / "e2.rxn"


@pytest.fixture()
def sn1_dissoc_rxn_path(examples_dir: Path) -> Path:
    return examples_dir / "sn1_dissoc.rxn"


@pytest.fixture()
def sn2_atoms_setup():
    """3-fragment-style SN2 setup for placement dispatcher test (CH3Cl + OH-)."""
    mol = Chem.AddHs(Chem.MolFromSmiles("C(Cl).[OH-]"))
    Chem.SanitizeMol(mol)
    frag_indices = Chem.GetMolFrags(mol)
    n = mol.GetNumAtoms()
    positions = np.zeros((n, 3))
    for i in range(n):
        positions[i] = (float(i) * 0.5, 0.0, 0.0)
    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    c_idx = syms.index("C")
    cl_idx = syms.index("Cl")
    o_idx = syms.index("O")
    bc = BondChanges(formed=((c_idx, o_idx),), broken=((c_idx, cl_idx),))
    return mol, frag_indices, positions, bc


@pytest.fixture()
def e2_atoms_setup():
    """E2 setup: CH3CH2Cl + OH-, with explicit β-H atom."""
    smi = "[H][CH2:1][CH2:2][Cl:3].[O:4][H:5]"
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    Chem.SanitizeMol(mol)
    frag_indices = Chem.GetMolFrags(mol)
    n = mol.GetNumAtoms()
    positions = np.zeros((n, 3))
    for i in range(n):
        positions[i] = (float(i) * 0.5, float(i % 2), 0.0)

    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    map_to_idx = {a.GetAtomMapNum(): a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum()}
    c_alpha = map_to_idx[1]
    c_beta = map_to_idx[2]
    cl = map_to_idx[3]
    o = map_to_idx[4]
    h_anchor = next(
        n.GetIdx() for n in mol.GetAtomWithIdx(c_beta).GetNeighbors()
        if n.GetSymbol() == "H"
    )
    bc = BondChanges(
        formed=((o, h_anchor),),
        broken=((c_alpha, cl), (c_beta, h_anchor)),
    )
    return mol, frag_indices, positions, bc
