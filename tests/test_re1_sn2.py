"""End-to-end SN2 test using UMA. Marked slow, requires HF auth + GPU."""
import json
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from reactx.cli import main


@pytest.mark.slow
def test_re1_sn2_end_to_end(tmp_path: Path, sn2_rxn_path: Path):
    out = tmp_path / "sn2"
    rc = main([
        "run", str(sn2_rxn_path), "-o", str(out),
        "--backend", "uma",
        "--n-angles", "4",          # smaller for test speed
        "--max-relax-steps", "50",
    ])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())
    assert meta["selected_trial"] >= 0
    # At least one trial reached product
    assert any(t["reached_product"] for t in meta["trials"]), (
        f"No trial reached product. trials={meta['trials']}"
    )

    frames = read(str(out / "trajectory.xyz"), index=":")
    assert len(frames) >= 3

    syms = frames[0].get_chemical_symbols()
    c_idx = syms.index("C")
    cl_idx = syms.index("Cl")
    f_idx = syms.index("F")

    # Walden inversion check #1: F-C-Cl angle peaks ≥ 120° somewhere on the path
    angles = []
    for f in frames:
        v_cf = f.positions[f_idx] - f.positions[c_idx]
        v_ccl = f.positions[cl_idx] - f.positions[c_idx]
        cos_t = float(np.dot(v_cf, v_ccl) / (
            np.linalg.norm(v_cf) * np.linalg.norm(v_ccl)
        ))
        angles.append(float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0)))))
    assert max(angles) >= 120.0, (
        f"F-C-Cl angle never reached 120° on the trajectory: max={max(angles):.1f}°"
    )

    # Walden inversion check #2: C-F shrinks, C-Cl grows (compare endpoints)
    d_cf_first = frames[0].get_distance(c_idx, f_idx)
    d_cf_last = frames[-1].get_distance(c_idx, f_idx)
    d_ccl_first = frames[0].get_distance(c_idx, cl_idx)
    d_ccl_last = frames[-1].get_distance(c_idx, cl_idx)
    assert d_cf_last < d_cf_first - 0.5, (
        f"C-F should shrink: {d_cf_first:.2f} -> {d_cf_last:.2f}"
    )
    assert d_ccl_last > d_ccl_first + 0.5, (
        f"C-Cl should grow: {d_ccl_first:.2f} -> {d_ccl_last:.2f}"
    )
