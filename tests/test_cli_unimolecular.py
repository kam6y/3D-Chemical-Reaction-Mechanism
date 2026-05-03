"""Tests for unimolecular reaction handling: n_angles auto-clamp + prescreen skip."""
import json
from pathlib import Path

import pytest

_SN1_DISSOC_FAST = """\
description = "sn1_dissoc fast"
formed = []
broken = [[1, 5]]
[restraints]
k_form = 0.0
k_broken = 2.0
r_broken = 6.0
max_relax_steps = 5
[sampling]
n_angles = 8
"""


@pytest.fixture()
def fake_unimolecular_pipeline(monkeypatch):
    import numpy as np
    from ase import Atoms

    from reactx import calculators, embed3d, path_relax

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


def test_unimolecular_n_angles_clamped_to_one(
    fake_unimolecular_pipeline, tmp_path: Path, tmp_rxn_with_toml,
):
    """When reactant has 1 fragment and TOML claims n_angles=8, clamp to 1."""
    rxn = tmp_rxn_with_toml("sn1_dissoc", toml_body=_SN1_DISSOC_FAST)
    out = tmp_path / "out"
    from reactx.cli import main
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "lj"])
    assert rc == 0
    meta = json.loads((out / "meta.json").read_text())
    assert len(meta["trials"]) == 1
    assert meta["prescreen"]["enabled"] is False
    assert meta["prescreen"]["kept"] is None
