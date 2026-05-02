"""MMFF94 prescreen for multi-angle trials.

Phase Re1 default runs N=8 trials through UMA at ~10 sec each on a GPU.
Most trials end up with reached_product=True and similar trajectories, so
running all 8 wastes wall-clock. This module wraps RDKit MMFF94 in an ASE
Calculator so we can run the SAME relax_with_restraints pipeline cheaply
(~0.1 sec/trial), then select the top-K to pass to UMA.

When MMFF94 cannot parameterize the system (ion pair, exotic atom types),
prescreen_trials transparently returns "keep all" so the caller falls back
to full UMA — wall-clock identical to Phase Re1 default.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import (
    MMFFGetMoleculeForceField,
    MMFFGetMoleculeProperties,
)

from reactx.artificial_force import build_restraints
from reactx.path_relax import relax_with_restraints
from reactx.scoring import TrialResult, reached_product

log = logging.getLogger(__name__)

# CODATA 2018: 1 kcal/mol = 0.0433641153 eV. Same scalar applies to forces
# (kcal/mol/Å -> eV/Å), since gradient and potential share the energy unit.
KCAL_PER_MOL_TO_EV = 0.0433641153


class MMFFParameterizationError(RuntimeError):
    """RDKit MMFF94 could not parameterize the molecule (e.g. exotic atom type)."""


class RDKitMMFFCalculator(Calculator):
    """ASE Calculator backed by RDKit MMFF94.

    Bound to a fixed RDKit Mol topology at construction. Each `calculate`
    call updates the embedded conformer's positions from the ASE Atoms,
    re-evaluates energy + gradient via MMFF, and converts kcal/mol -> eV.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, rdkit_mol: Chem.Mol):
        super().__init__()
        props = MMFFGetMoleculeProperties(rdkit_mol)
        if props is None:
            raise MMFFParameterizationError(
                "MMFFGetMoleculeProperties returned None"
            )
        self._mol = Chem.Mol(rdkit_mol)
        if self._mol.GetNumConformers() == 0:
            conf = Chem.Conformer(self._mol.GetNumAtoms())
            self._mol.AddConformer(conf, assignId=True)
        self._props = props

    def calculate(
        self, atoms=None, properties=("energy",), system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        positions = atoms.get_positions()
        conf = self._mol.GetConformer()
        for i in range(self._mol.GetNumAtoms()):
            conf.SetAtomPosition(i, (
                float(positions[i, 0]),
                float(positions[i, 1]),
                float(positions[i, 2]),
            ))
        ff = MMFFGetMoleculeForceField(self._mol, self._props)
        if ff is None:
            raise MMFFParameterizationError(
                "MMFFGetMoleculeForceField returned None at evaluation"
            )
        e_kcal = ff.CalcEnergy()
        grad = np.asarray(ff.CalcGrad(), dtype=float).reshape(-1, 3)
        self.results = {
            "energy": float(e_kcal) * KCAL_PER_MOL_TO_EV,
            "forces": -grad * KCAL_PER_MOL_TO_EV,
        }


def make_mol_from_atoms(template_mol_h: Chem.Mol, atoms: Atoms) -> Chem.Mol:
    """Copy template_mol_h and overwrite its conformer with `atoms` positions.

    Atom ordering must match (embed_mol_to_atoms preserves the AddHs(mol)
    ordering, so this holds for the canonical caller path).
    """
    if template_mol_h.GetNumAtoms() != len(atoms):
        raise ValueError(
            f"Atom count mismatch: template_mol_h has {template_mol_h.GetNumAtoms()}, "
            f"atoms has {len(atoms)}"
        )
    mol = Chem.Mol(template_mol_h)
    mol.RemoveAllConformers()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, p in enumerate(atoms.get_positions()):
        conf.SetAtomPosition(i, (float(p[0]), float(p[1]), float(p[2])))
    mol.AddConformer(conf, assignId=True)
    return mol


@dataclass(frozen=True)
class PrescreenResult:
    enabled: bool
    kept: list[int]
    skipped: list[int]
    mmff_failed: bool
    wall_clock_seconds: float


def select_top_k_indices(trials: list[TrialResult], k: int) -> list[int]:
    """Top-K trial indices: reached_product first, then lowest peak_energy.

    Spans both groups when fewer than K trials reached product (so that
    promising-but-not-yet-reached trials still make it to UMA).
    """
    reached = sorted(
        (t for t in trials if t.reached_product),
        key=lambda t: t.peak_energy,
    )
    not_reached = sorted(
        (t for t in trials if not t.reached_product),
        key=lambda t: t.peak_energy,
    )
    return [t.trial_idx for t in (reached + not_reached)[:k]]


def prescreen_trials(
    atoms_list: list[Atoms],
    mol_h_template: Chem.Mol,
    formed: list[tuple[int, int]],
    broken: list[tuple[int, int]],
    *,
    r_form_target: float,
    r_broken_target: float,
    k_form: float,
    k_broken: float,
    max_steps: int = 30,
    fmax: float = 0.5,
    traj_stride: int = 5,
    k_keep: int = 3,
) -> PrescreenResult:
    """Run a short MMFF restraint relaxation per trial; return top-K indices.

    On any MMFF parameterize failure, fall back to keeping all trial indices
    (mmff_failed=True; warning logged). Short-circuit without invoking MMFF
    when k_keep >= len(atoms_list).
    """
    n = len(atoms_list)
    if k_keep >= n:
        return PrescreenResult(
            enabled=True, kept=list(range(n)), skipped=[],
            mmff_failed=False, wall_clock_seconds=0.0,
        )

    t0 = time.monotonic()
    trials: list[TrialResult] = []
    try:
        for i, atoms_i in enumerate(atoms_list):
            mol_i = make_mol_from_atoms(mol_h_template, atoms_i)
            calc = RDKitMMFFCalculator(mol_i)
            restraints = build_restraints(
                atoms_i, formed=formed, broken=broken,
                r_form=r_form_target, r_broken=r_broken_target,
                k_form=k_form, k_broken=k_broken,
            )
            frames, energies = relax_with_restraints(
                atoms_i, restraints, calc,
                max_steps=max_steps, fmax=fmax, traj_stride=traj_stride,
            )
            ok = (
                reached_product(
                    frames[-1], formed=formed, broken=broken,
                    r_form_targets=[r_form_target] * len(formed),
                    r_broken_target=r_broken_target,
                )
                if formed or broken
                else False
            )
            trials.append(TrialResult(
                trial_idx=i, rotation_deg=0.0,
                frames=frames, energies=energies,
                reached_product=ok,
                peak_energy=float(max(energies)) if energies else float("inf"),
                n_steps=len(frames),
            ))
    except MMFFParameterizationError as err:
        log.warning(
            "MMFF prescreen failed (%s): falling back to full UMA on all %d trials.",
            err, n,
        )
        return PrescreenResult(
            enabled=True, kept=list(range(n)), skipped=[],
            mmff_failed=True,
            wall_clock_seconds=float(time.monotonic() - t0),
        )

    kept = select_top_k_indices(trials, k_keep)
    skipped = [i for i in range(n) if i not in kept]
    log.info(
        "prescreen: %d trials -> %d kept (idx=%s) in %.1fs",
        n, len(kept), kept, time.monotonic() - t0,
    )
    return PrescreenResult(
        enabled=True, kept=kept, skipped=skipped, mmff_failed=False,
        wall_clock_seconds=float(time.monotonic() - t0),
    )
