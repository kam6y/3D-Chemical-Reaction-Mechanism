"""End-to-end Diels-Alder (cyclopentadiene + maleic anhydride) with UMA.

Marked slow. Verifies endo / exo trial expansion and that the pipeline
produces both orientations as separate trials."""
import json
from pathlib import Path

import pytest
from ase.io import read

from reactx.cli import main


@pytest.mark.slow
def test_diels_alder_endo_end_to_end(tmp_path: Path):
    rxn = Path(__file__).resolve().parent.parent / "examples" / "diels_alder_endo.rxn"
    out = tmp_path / "da_endo"
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "uma"])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())
    assert meta["selected_trial"] >= 0
    assert meta["placement"]["placement_kind"] == "multi_anchor"

    # MA is asymmetric -> both endo and exo trials should be present
    orientations = {t["orientation"] for t in meta["trials"]}
    assert "endo" in orientations, f"expected endo in orientations: {orientations}"
    assert "exo" in orientations, f"expected exo in orientations: {orientations}"

    # selected trial should be endo or exo (not single, not achiral)
    selected = next(t for t in meta["trials"] if t["trial"] == meta["selected_trial"])
    assert selected["orientation"] in ("endo", "exo"), (
        f"selected trial orientation is unexpected: {selected['orientation']}"
    )

    frames = read(str(out / "trajectory.xyz"), index=":")
    assert len(frames) >= 3

    # sigma bond check: |C1-C5| <= 1.9 A and |C4-C6| <= 1.9 A in final frame.
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
    assert d_15 < 1.9, f"C1-C5 not formed: {d_15:.2f} A (tolerance 1.9 for endo strain)"
    assert d_46 < 1.9, f"C4-C6 not formed: {d_46:.2f} A"
