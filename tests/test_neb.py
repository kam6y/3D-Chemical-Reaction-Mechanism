"""Tests for reactx.neb.run_neb return value extension (Phase 11)."""
from pathlib import Path

from ase import Atoms
from ase.calculators.lj import LennardJones

import reactx.neb as neb_module
from reactx.neb import run_neb


def _lj_endpoint(r: float) -> Atoms:
    return Atoms("Ar2", positions=[[0.0, 0.0, 0.0], [r, 0.0, 0.0]])


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
