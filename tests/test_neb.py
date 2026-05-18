"""Tests for reactx.neb.run_neb return value extension (Phase 11)."""
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.lj import LennardJones
from ase.io import read

import reactx.neb as neb_module
from reactx.bond_changes import BondChanges
from reactx.neb import run_neb


def _lj_endpoint(r: float) -> Atoms:
    return Atoms("Ar2", positions=[[0.0, 0.0, 0.0], [r, 0.0, 0.0]])


class PairDistanceCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)
        distance = float(atoms.get_distance(0, 1, mic=False))
        self.results["energy"] = distance
        self.results["forces"] = np.zeros((len(atoms), 3))


def test_run_neb_returns_image_atoms(tmp_path: Path):
    reactant = _lj_endpoint(1.2)
    product = _lj_endpoint(1.4)
    xyz = tmp_path / "traj.xyz"
    info = run_neb(
        reactant,
        product,
        calculator=LennardJones(),
        n_images=5,
        output_xyz=xyz,
        fmax=0.5,
        max_steps=10,
    )
    assert "image_atoms" in info
    assert isinstance(info["image_atoms"], list)
    assert len(info["image_atoms"]) == 5
    for img in info["image_atoms"]:
        assert isinstance(img, Atoms)
        assert len(img) == 2


def test_run_neb_passes_configured_neb_options_to_each_neb(monkeypatch, tmp_path: Path):
    captured: list[tuple[bool, float, str, bool]] = []

    class FakeNEB:
        def __init__(
            self,
            images,
            *,
            k: float,
            climb: bool,
            allow_shared_calculator: bool,
            method: str,
            remove_rotation_and_translation: bool,
        ):
            self.images = images
            self.energies = [0.0] * len(images)
            captured.append((climb, k, method, remove_rotation_and_translation))

        def interpolate(self, *, method: str) -> None:
            assert method == "idpp"

        def get_residual(self) -> float:
            return 0.0

    class FakeFIRE:
        def __init__(self, neb, *, logfile: str):
            self.neb = neb
            assert logfile == "-"

        def run(self, *, fmax: float, steps: int) -> bool:
            return True

    monkeypatch.setattr(neb_module, "NEB", FakeNEB)
    monkeypatch.setattr(neb_module, "FIRE", FakeFIRE)
    monkeypatch.setattr(neb_module, "write", lambda *args, **kwargs: None)

    info = run_neb(
        _lj_endpoint(1.2),
        _lj_endpoint(1.4),
        calculator=LennardJones(),
        n_images=4,
        output_xyz=tmp_path / "traj.xyz",
        max_steps=2,
        k=2.75,
        method="eb",
        remove_rotation_and_translation=True,
    )

    assert captured == [(False, 2.75, "eb", True), (True, 2.75, "eb", True)]
    assert info["k"] == 2.75
    assert info["method"] == "eb"
    assert info["remove_rotation_and_translation"] is True


def test_run_neb_applies_guide_to_internal_images_and_reports_unbiased_energy(
    monkeypatch,
    tmp_path: Path,
):
    calc = LennardJones()
    captured_internal_calcs = []

    class FakeNEB:
        def __init__(
            self,
            images,
            *,
            k: float,
            climb: bool,
            allow_shared_calculator: bool,
            method: str,
            remove_rotation_and_translation: bool,
        ):
            self.images = images
            self.energies = [10.0, 11.0, 12.0, 13.0, 14.0]
            captured_internal_calcs.append([type(img.calc).__name__ for img in images])

        def interpolate(self, *, method: str) -> None:
            assert method == "idpp"

        def get_residual(self) -> float:
            return 0.0

    class FakeFIRE:
        def __init__(self, neb, *, logfile: str):
            self.neb = neb

        def run(self, *, fmax: float, steps: int) -> bool:
            return True

    monkeypatch.setattr(neb_module, "NEB", FakeNEB)
    monkeypatch.setattr(neb_module, "FIRE", FakeFIRE)
    monkeypatch.setattr(neb_module, "write", lambda *args, **kwargs: None)

    info = run_neb(
        _lj_endpoint(1.0),
        _lj_endpoint(2.0),
        calculator=calc,
        n_images=5,
        output_xyz=tmp_path / "traj.xyz",
        max_steps=2,
        guide_bond_changes=True,
        guide_k=0.25,
        bond_changes=BondChanges(formed=((0, 1),), broken=()),
    )

    assert captured_internal_calcs[0] == [
        "LennardJones",
        "BondDistanceGuideCalculator",
        "BondDistanceGuideCalculator",
        "BondDistanceGuideCalculator",
        "LennardJones",
    ]
    assert info["biased_optimization"] is True
    assert info["guide_bond_changes"] is True
    assert info["guide_k"] == 0.25
    assert info["guided_bonds"] == [
        {"atoms": [0, 1], "kind": "formed", "reactant_distance": 1.0, "product_distance": 2.0}
    ]
    assert info["image_energies"] == [10.0, 11.0, 12.0, 13.0, 14.0]
    assert len(info["unbiased_image_energies"]) == 5
    assert all(isinstance(e, float) for e in info["unbiased_image_energies"])


def test_run_neb_real_guide_separates_biased_unbiased_and_omits_extxyz_results(
    tmp_path: Path,
):
    reactant = Atoms("Ar2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    product = Atoms("Ar2", positions=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    xyz = tmp_path / "traj.xyz"

    info = run_neb(
        reactant,
        product,
        calculator=PairDistanceCalculator(),
        n_images=3,
        output_xyz=xyz,
        fmax=1.0e9,
        max_steps=1,
        climb=False,
        remove_rotation_and_translation=False,
        guide_bond_changes=True,
        guide_k=2.0,
        bond_changes=BondChanges(formed=((0, 1),), broken=()),
    )

    assert info["biased_optimization"] is True
    assert info["unbiased_image_energies"] == pytest.approx(
        [1.0, 1.0456114614533338, 1.0]
    )
    assert info["image_energies"] == pytest.approx(
        [1.0, 1.0476918668692428, 1.0]
    )
    assert info["image_energies"][1] > info["unbiased_image_energies"][1]

    frames = read(str(xyz), index=":")
    assert len(frames) == 3
    for frame in frames:
        assert frame.calc is None
        assert "energy" not in frame.info
        assert "forces" not in frame.arrays
