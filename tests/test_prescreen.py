"""Unit tests for MMFF prescreen."""
from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem

from reactx.prescreen import (
    MMFFParameterizationError,
    PrescreenResult,
    RDKitMMFFCalculator,
    make_mol_from_atoms,
    prescreen_trials,
    select_top_k_indices,
)
from reactx.scoring import TrialResult


def _build_h2o_mol() -> Chem.Mol:
    mol = Chem.AddHs(Chem.MolFromSmiles("O"))
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def _atoms_from_mol(mol: Chem.Mol) -> Atoms:
    conf = mol.GetConformer()
    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    pos = np.array([
        [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
        for i in range(mol.GetNumAtoms())
    ])
    return Atoms(symbols=syms, positions=pos)


def _make_trial(idx: int, reached: bool, peak: float) -> TrialResult:
    return TrialResult(
        trial_idx=idx, rotation_deg=0.0,
        frames=[], energies=[],
        reached_product=reached, peak_energy=peak, n_steps=0,
    )


def test_mmff_calc_water_energy_finite():
    mol = _build_h2o_mol()
    atoms = _atoms_from_mol(mol)
    atoms.calc = RDKitMMFFCalculator(mol)
    e = atoms.get_potential_energy()
    assert np.isfinite(e)
    assert abs(e) < 5.0, f"H2O MMFF energy out of expected range: {e:.3f} eV"


def test_mmff_calc_water_forces_shape():
    mol = _build_h2o_mol()
    atoms = _atoms_from_mol(mol)
    atoms.calc = RDKitMMFFCalculator(mol)
    f = atoms.get_forces()
    assert f.shape == (3, 3)
    assert np.all(np.isfinite(f))
    assert np.linalg.norm(f) < 50.0, (
        f"Optimized H2O forces unexpectedly large: max={np.abs(f).max():.3f} eV/Å"
    )


def test_mmff_calc_raises_when_props_none(monkeypatch):
    """RDKitMMFFCalculator raises MMFFParameterizationError when MMFF cannot parameterize."""
    from reactx import prescreen

    monkeypatch.setattr(prescreen, "MMFFGetMoleculeProperties", lambda m: None)
    mol = Chem.AddHs(Chem.MolFromSmiles("O"))
    AllChem.EmbedMolecule(mol)
    with pytest.raises(MMFFParameterizationError):
        RDKitMMFFCalculator(mol)


def test_make_mol_from_atoms_preserves_topology():
    template = Chem.AddHs(Chem.MolFromSmiles("O"))
    AllChem.EmbedMolecule(template, randomSeed=11)
    atoms = Atoms("OH2", positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.5, 0.8, 0.0]])
    mol = make_mol_from_atoms(template, atoms)
    assert mol.GetNumAtoms() == 3
    assert mol.GetNumBonds() == 2
    conf = mol.GetConformer()
    assert conf.GetAtomPosition(0).x == pytest.approx(0.0)
    assert conf.GetAtomPosition(1).x == pytest.approx(0.96)
    assert conf.GetAtomPosition(2).y == pytest.approx(0.8)


def test_make_mol_from_atoms_count_mismatch_raises():
    template = Chem.AddHs(Chem.MolFromSmiles("O"))
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="Atom count mismatch"):
        make_mol_from_atoms(template, atoms)


def test_select_top_k_reached_first():
    trials = [
        _make_trial(0, True, 5.0),
        _make_trial(1, False, 1.0),
        _make_trial(2, True, 3.0),
        _make_trial(3, True, 4.0),
    ]
    # k=3: all 3 reached, ordered by peak_energy ascending
    assert select_top_k_indices(trials, 3) == [2, 3, 0]


def test_select_top_k_mixed_when_few_reached():
    trials = [
        _make_trial(0, True, 5.0),
        _make_trial(1, False, 1.0),
        _make_trial(2, False, 2.0),
        _make_trial(3, False, 8.0),
    ]
    # k=3: 1 reached + top 2 unreached by peak_energy
    assert select_top_k_indices(trials, 3) == [0, 1, 2]


def test_select_top_k_all_unreached():
    trials = [
        _make_trial(0, False, 5.0),
        _make_trial(1, False, 1.0),
        _make_trial(2, False, 3.0),
    ]
    assert select_top_k_indices(trials, 2) == [1, 2]


def test_prescreen_short_circuit_when_k_ge_n():
    """k_keep > n: short-circuit returns all indices, never invokes MMFF."""
    junk_mol = Chem.MolFromSmiles("O")  # no conformer; would crash if MMFF were invoked
    result = prescreen_trials(
        atoms_list=[Atoms("H"), Atoms("H")],
        mol_h_template=junk_mol,
        formed=[], broken=[],
        r_form_target=1.5, r_broken_target=4.0,
        k_form=0.5, k_broken=1.0,
        k_keep=5,  # > n=2
    )
    assert isinstance(result, PrescreenResult)
    assert result.enabled
    assert result.kept == [0, 1]
    assert result.skipped == []
    assert not result.mmff_failed
    assert result.wall_clock_seconds == 0.0


def test_prescreen_short_circuit_k_equal_n():
    junk_mol = Chem.MolFromSmiles("O")
    result = prescreen_trials(
        atoms_list=[Atoms("H"), Atoms("H"), Atoms("H")],
        mol_h_template=junk_mol,
        formed=[], broken=[],
        r_form_target=1.5, r_broken_target=4.0,
        k_form=0.5, k_broken=1.0,
        k_keep=3,  # == n
    )
    assert result.kept == [0, 1, 2]
    assert result.wall_clock_seconds == 0.0


def test_prescreen_handles_mmff_failure(monkeypatch):
    """When MMFF cannot parameterize, fall back to keeping all trial indices."""
    from reactx import prescreen

    monkeypatch.setattr(prescreen, "MMFFGetMoleculeProperties", lambda m: None)

    template = Chem.AddHs(Chem.MolFromSmiles("O"))
    AllChem.EmbedMolecule(template, randomSeed=11)
    atoms_list = [_atoms_from_mol(template) for _ in range(3)]
    result = prescreen_trials(
        atoms_list=atoms_list,
        mol_h_template=template,
        formed=[(0, 1)], broken=[(0, 2)],
        r_form_target=1.5, r_broken_target=4.0,
        k_form=0.5, k_broken=1.0,
        k_keep=1,
    )
    assert result.mmff_failed
    assert result.kept == [0, 1, 2]
    assert result.skipped == []
    assert result.wall_clock_seconds >= 0.0


def test_prescreen_real_mmff_water_short_relax():
    """End-to-end smoke: 2 H2O trials through real MMFF prescreen, k_keep=1."""
    template = Chem.AddHs(Chem.MolFromSmiles("O"))
    AllChem.EmbedMolecule(template, randomSeed=11)
    AllChem.MMFFOptimizeMolecule(template)
    base_atoms = _atoms_from_mol(template)
    # Trial 0: equilibrium. Trial 1: stretched O-H bond (higher initial energy).
    a0 = base_atoms.copy()
    a1 = base_atoms.copy()
    p1 = a1.get_positions()
    p1[1] += np.array([0.5, 0.0, 0.0])  # stretch first O-H
    a1.set_positions(p1)

    result = prescreen_trials(
        atoms_list=[a0, a1],
        mol_h_template=template,
        formed=[], broken=[],
        r_form_target=1.5, r_broken_target=4.0,
        k_form=0.0, k_broken=0.0,
        max_steps=10, fmax=0.5, traj_stride=5,
        k_keep=1,
    )
    assert result.enabled
    assert not result.mmff_failed
    assert len(result.kept) == 1
    assert len(result.skipped) == 1
    assert sorted(result.kept + result.skipped) == [0, 1]
