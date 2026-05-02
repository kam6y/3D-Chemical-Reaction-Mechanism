"""Tests for unimolecular reaction handling: --n-angles auto-clamp + prescreen skip."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from reactx.cli import main


@pytest.fixture()
def fake_unimolecular_pipeline(monkeypatch, tmp_path):
    """Patch the UMA pipeline to a 1-fragment fake reaction.

    We don't need a real UMA call; intercept calculator + relax to return
    immediately, then inspect meta.json behaviour.
    """
    from reactx import calculators, embed3d, path_relax
    import numpy as np
    from ase import Atoms

    def fake_calc(*args, **kwargs):
        from ase.calculators.lj import LennardJones
        return LennardJones()

    def fake_embed(mol, **kw):
        atoms = Atoms("CCCCC", positions=np.array([
            [0, 0, 0], [1.5, 0, 0], [3.0, 0, 0], [-1.5, 0, 0], [0, 1.5, 0],
        ]))
        return atoms

    def fake_relax(atoms_init, restraints, calc, **kw):
        return [atoms_init], [0.0]

    monkeypatch.setattr(calculators, "make_calculator", fake_calc)
    monkeypatch.setattr(embed3d, "embed_mol_to_atoms", fake_embed)
    monkeypatch.setattr(path_relax, "relax_with_restraints", fake_relax)


def test_unimolecular_n_angles_clamped_to_one(fake_unimolecular_pipeline, tmp_path):
    """When reactant has 1 fragment, --n-angles N is clamped to 1 and prescreen skipped."""
    rxn = Path("examples/sn1_dissoc.rxn")
    if not rxn.exists():
        pytest.skip("examples/sn1_dissoc.rxn not yet created (Task 8)")
    out = tmp_path / "out"
    rc = main([
        "run", str(rxn), "-o", str(out),
        "--backend", "lj",
        "--reaction-type", "sn1_dissoc",
        "--n-angles", "8",
    ])
    assert rc == 0
    meta = json.loads((out / "meta.json").read_text())
    assert len(meta["trials"]) == 1, f"expected 1 trial after clamp, got {len(meta['trials'])}"
    assert meta["prescreen"]["enabled"] is False or meta["prescreen"]["kept"] is None
