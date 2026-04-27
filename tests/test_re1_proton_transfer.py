"""End-to-end proton transfer test (HCl + NH3 -> Cl- + NH4+)."""
import json
from pathlib import Path

import pytest
from ase.io import read

from reactx.cli import main


@pytest.fixture()
def proton_transfer_rxn_path(examples_dir: Path) -> Path:
    return examples_dir / "proton_transfer.rxn"


@pytest.mark.slow
def test_re1_proton_transfer_end_to_end(
    tmp_path: Path, proton_transfer_rxn_path: Path,
):
    out = tmp_path / "proton_transfer"
    rc = main([
        "run", str(proton_transfer_rxn_path), "-o", str(out),
        "--backend", "uma",
        "--n-angles", "4",
        "--max-relax-steps", "50",
        "--r-form", "1.05",  # N-H equilibrium
    ])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())
    assert meta["selected_trial"] >= 0
    assert any(t["reached_product"] for t in meta["trials"])

    frames = read(str(out / "trajectory.xyz"), index=":")
    syms = frames[0].get_chemical_symbols()
    cl_idx = syms.index("Cl")
    n_idx = syms.index("N")
    h_indices = [i for i, s in enumerate(syms) if s == "H"]
    initial_h_to_cl = [frames[0].get_distance(i, cl_idx) for i in h_indices]
    proton_idx = h_indices[initial_h_to_cl.index(min(initial_h_to_cl))]

    d_h_cl_first = frames[0].get_distance(proton_idx, cl_idx)
    d_h_cl_last = frames[-1].get_distance(proton_idx, cl_idx)
    d_h_n_first = frames[0].get_distance(proton_idx, n_idx)
    d_h_n_last = frames[-1].get_distance(proton_idx, n_idx)

    assert d_h_cl_last > d_h_cl_first + 0.3, (
        f"H-Cl should elongate: {d_h_cl_first:.2f} -> {d_h_cl_last:.2f}"
    )
    assert d_h_n_last < d_h_n_first - 0.5, (
        f"H-N should shrink: {d_h_n_first:.2f} -> {d_h_n_last:.2f}"
    )
