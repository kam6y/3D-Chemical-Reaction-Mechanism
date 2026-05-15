"""End-to-end SN2 test using UMA. Marked slow, requires HF auth + GPU."""
import json
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from reactx.cli import main

_SN2_FAST = """\
description = "SN2 fast"
formed = [[1, 3]]
broken = [[1, 2]]

[afir]
alpha_formed = 4.0
alpha_broken = 2.5
max_relax_steps = 300
pre_relax_steps = 15

[scoring]
r_broken_threshold = 3.0

[sampling]
n_candidates = 32
"""


@pytest.mark.slow
def test_re1_sn2_end_to_end(tmp_path: Path, tmp_rxn_with_toml):
    rxn = tmp_rxn_with_toml("sn2", toml_body=_SN2_FAST)
    out = tmp_path / "sn2"
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "uma"])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())
    assert meta["description"] == "SN2 fast"
    assert meta["selected_trial"] >= 0
    assert meta["effective_params"]["pre_relax_steps"] == 15
    pl = meta["placement"]
    assert pl["n_candidates"] >= 1
    assert pl["n_valid"] >= 1
    assert pl["n_blocked"] == pl["n_candidates"] - pl["n_valid"]
    assert len(meta["trials"]) == pl["n_valid"]
    assert any(t["reached_product"] for t in meta["trials"])

    frames = read(str(out / "trajectory.xyz"), index=":")
    assert len(frames) >= 3

    syms = frames[0].get_chemical_symbols()
    c_idx = syms.index("C")
    cl_idx = syms.index("Cl")
    o_idx = syms.index("O")

    angles = []
    for f in frames:
        v_co = f.positions[o_idx] - f.positions[c_idx]
        v_ccl = f.positions[cl_idx] - f.positions[c_idx]
        cos_t = float(np.dot(v_co, v_ccl) / (
            np.linalg.norm(v_co) * np.linalg.norm(v_ccl)
        ))
        angles.append(float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0)))))
    # Phase 10: pre-relax (15 steps for SN2, tuned via sweep) orients OH- to
    # Walden back-side via CH3Cl ion-dipole attraction. Walden inversion
    # ⇒ Cl-C-O ≈ 180°. Sweep at pre=15 produced max=173.83°, min=162.90°;
    # the 160° threshold leaves a small margin for UMA noise / seed variance.
    assert max(angles) >= 160.0, (
        f"max Cl-C-O angle {max(angles):.2f}° below Walden threshold; "
        f"pre-relax should orient OH- to back-side attack"
    )

    d_co_first = frames[0].get_distance(c_idx, o_idx)
    d_co_last = frames[-1].get_distance(c_idx, o_idx)
    d_ccl_first = frames[0].get_distance(c_idx, cl_idx)
    d_ccl_last = frames[-1].get_distance(c_idx, cl_idx)
    assert d_co_last < d_co_first - 0.5
    assert d_ccl_last > d_ccl_first + 0.5


_SN2_SAFETY = """\
description = "SN2 safety check"
formed = [[1, 3]]
broken = [[1, 2]]

[afir]
alpha_formed = 4.0
alpha_broken = 2.5
max_relax_steps = 300

[scoring]
r_broken_threshold = 3.0
"""


@pytest.mark.slow
def test_re1_sn2_min_nonbonded_distance(
    tmp_path: Path, tmp_rxn_with_toml, assert_min_nonbonded_distance_ok,
):
    """Spec §6.4: AFIR-driven trajectory must not produce non-bonded
    atom pairs closer than 0.5 Å (UMA off-manifold detection).

    sn2.rxn atom-map: 1->idx 0 (C), 2->idx 1 (Cl), 3->idx 2 (O).
    formed=[[1,3]] -> (0,2); broken=[[1,2]] -> (0,1).
    """
    rxn = tmp_rxn_with_toml("sn2", toml_body=_SN2_SAFETY)
    out = tmp_path / "sn2_safety"
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "uma"])
    assert rc == 0

    frames = read(str(out / "trajectory.xyz"), index=":")
    assert_min_nonbonded_distance_ok(
        frames, formed=[(0, 2)], broken=[(0, 1)],
    )
