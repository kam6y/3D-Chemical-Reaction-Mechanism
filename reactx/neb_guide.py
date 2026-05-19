"""Bond-distance guide calculator for reaction-coordinate-biased NEB."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from reactx.bond_changes import BondChanges

BondKind = Literal["formed", "broken"]


@dataclass(frozen=True)
class GuidedBond:
    atoms: tuple[int, int]
    kind: BondKind
    reactant_distance: float
    product_distance: float

    def target_at(self, image_index: int, n_images: int) -> float:
        t = image_index / (n_images - 1)
        return (1.0 - t) * self.reactant_distance + t * self.product_distance

    def as_metadata(self) -> dict:
        return {
            "atoms": [int(self.atoms[0]), int(self.atoms[1])],
            "kind": self.kind,
            "reactant_distance": float(self.reactant_distance),
            "product_distance": float(self.product_distance),
        }


class BondDistanceGuideCalculator(Calculator):
    """Calculator wrapper adding harmonic restraints to selected pair distances."""

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        base_calculator: Calculator,
        *,
        targets: dict[tuple[int, int], float],
        guide_k: float,
        zero_distance_tol: float = 1.0e-12,
    ) -> None:
        super().__init__()
        if not math.isfinite(float(guide_k)) or float(guide_k) <= 0.0:
            raise ValueError("guide_k must be a finite positive number")
        self.base_calculator = base_calculator
        self.targets = {
            _canonical_pair(pair): float(target) for pair, target in targets.items()
        }
        self.guide_k = float(guide_k)
        self.zero_distance_tol = float(zero_distance_tol)

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        base_energy = float(self.base_calculator.get_potential_energy(atoms))
        base_forces = np.array(self.base_calculator.get_forces(atoms), dtype=float)

        guide_energy = 0.0
        guide_forces = np.zeros_like(base_forces)
        positions = atoms.get_positions()
        for (i, j), target in self.targets.items():
            pair_vector = positions[j] - positions[i]
            distance = float(np.linalg.norm(pair_vector))
            delta = distance - target
            guide_energy += 0.5 * self.guide_k * delta * delta
            if distance <= self.zero_distance_tol:
                continue
            direction = pair_vector / distance
            force = self.guide_k * delta * direction
            guide_forces[i] += force
            guide_forces[j] -= force

        self.results["energy"] = base_energy + guide_energy
        self.results["forces"] = base_forces + guide_forces


def build_guided_bonds(
    reactant: Atoms,
    product: Atoms,
    bond_changes: BondChanges,
) -> list[GuidedBond]:
    guided: list[GuidedBond] = []
    for kind, pairs in (
        ("formed", bond_changes.formed),
        ("broken", bond_changes.broken),
    ):
        for pair in pairs:
            i, j = _canonical_pair(pair)
            guided.append(
                GuidedBond(
                    atoms=(i, j),
                    kind=kind,
                    reactant_distance=_distance(reactant, i, j),
                    product_distance=_distance(product, i, j),
                )
            )
    return guided


def targets_for_image(
    guided_bonds: list[GuidedBond],
    *,
    image_index: int,
    n_images: int,
) -> dict[tuple[int, int], float]:
    return {
        bond.atoms: bond.target_at(image_index, n_images) for bond in guided_bonds
    }


def _distance(atoms: Atoms, i: int, j: int) -> float:
    return float(atoms.get_distance(i, j, mic=False))


def _canonical_pair(pair: tuple[int, int]) -> tuple[int, int]:
    i, j = pair
    return (int(i), int(j)) if i <= j else (int(j), int(i))
