"""End-to-end Diels-Alder (butadiene + ethylene) with UMA. Marked slow."""
import json
from pathlib import Path

import pytest
from ase.io import read

from reactx.cli import main


@pytest.mark.slow
def test_diels_alder_simple_end_to_end(tmp_path: Path):
    rxn = Path(__file__).resolve().parent.parent / "examples" / "diels_alder_simple.rxn"
    out = tmp_path / "da_simple"
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "uma"])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())
    assert meta["selected_trial"] >= 0, f"no trial selected: {meta}"
    assert meta["placement"]["placement_kind"] == "multi_anchor"
    assert any(t["reached_product"] for t in meta["trials"]), (
        f"no DA trial reached product: {meta['trials']}"
    )

    # Symmetric ethylene -> expected to collapse to achiral
    orientations = {t["orientation"] for t in meta["trials"]}
    assert orientations <= {"achiral", "endo", "exo"}
    # In particular, there should be at least one achiral or matched endo/exo pair
    assert "achiral" in orientations or {"endo", "exo"}.issubset(orientations)

    frames = read(str(out / "trajectory.xyz"), index=":")
    assert len(frames) >= 3

    # Verify sigma bond formation: |C1-C5| and |C4-C6| <= 1.8 A in final frame.
    # atom_map_to_reactant_idx gives heavy-atom indices for maps 1,4,5,6.
    # For cyclohexene formation, the C-C sigma bonds should be <= 1.8 A.
    last = frames[-1]
    from reactx.rxn_parser import atom_map_to_reactant_idx, parse_rxn
    r_mol, _, _ = parse_rxn(rxn)
    idx_map = atom_map_to_reactant_idx(r_mol)
    c1 = idx_map[1]
    c4 = idx_map[4]
    c5 = idx_map[5]
    c6 = idx_map[6]
    d_15 = last.get_distance(c1, c5)
    d_46 = last.get_distance(c4, c6)
    assert d_15 < 1.8, f"C1-C5 not formed: {d_15:.2f} A"
    assert d_46 < 1.8, f"C4-C6 not formed: {d_46:.2f} A"
