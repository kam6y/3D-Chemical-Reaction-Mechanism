"""Tests for reactx.neb.run_neb return value extension (Phase 11)."""
from pathlib import Path

from ase import Atoms
from ase.calculators.lj import LennardJones

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
