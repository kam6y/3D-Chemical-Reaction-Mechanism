"""Tests for unimolecular reaction handling: n_candidates auto-clamp to 1."""
import json
from pathlib import Path

import pytest

_SN1_DISSOC_FAST = """\
description = "sn1_dissoc fast"
formed = []
broken = [[1, 5]]

[afir]
alpha_broken = 1.0
max_relax_steps = 5

[scoring]
r_broken_threshold = 6.0

[sampling]
n_candidates = 8
"""


@pytest.fixture()
def fake_unimolecular_pipeline(monkeypatch):
    import numpy as np
    from rdkit import Chem

    from reactx import calculators, embed3d, path_relax

    def fake_calc(*args, **kwargs):
        from ase.calculators.lj import LennardJones
        return LennardJones()

    def fake_embed(mol, *, seed=0):
        mol_h = Chem.AddHs(mol)
        n = mol_h.GetNumAtoms()
        positions = np.zeros((n, 3))
        # Lay atoms on a line so LJ has finite gradients.
        for i in range(n):
            positions[i] = (i * 1.5, 0.0, 0.0)
        return mol_h, Chem.GetMolFrags(mol_h), positions

    def fake_relax(atoms_init, restraints, calc, **kw):
        return [atoms_init], [0.0]

    monkeypatch.setattr(calculators, "make_calculator", fake_calc)
    monkeypatch.setattr(embed3d, "embed_fragments_to_positions", fake_embed)
    monkeypatch.setattr(path_relax, "relax_with_restraints", fake_relax)


def test_unimolecular_n_candidates_clamped_to_one(
    fake_unimolecular_pipeline, tmp_path: Path, tmp_rxn_with_toml,
):
    """When reactant has 1 fragment and TOML claims n_candidates=8, clamp to 1."""
    rxn = tmp_rxn_with_toml("sn1_dissoc", toml_body=_SN1_DISSOC_FAST)
    out = tmp_path / "out"
    from reactx.cli import main
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "lj"])
    assert rc == 0
    meta = json.loads((out / "meta.json").read_text())
    assert len(meta["trials"]) == 1
    assert meta["placement"]["n_candidates"] == 1
    assert meta["placement"]["n_valid"] == 1
    assert meta["placement"]["n_blocked"] == 0
    assert "prescreen" not in meta
