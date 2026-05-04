"""Wall-clock sanity check for SN2 default settings under Phase 8.

Phase 8 promotes CI-NEB to the main path engine, adding a NEB stage on top of
the artificial-force screening. We loosen the upper bound to 900s (15 min) to
absorb the extra UMA workload while still flagging gross regressions on the
user's RTX 5070 Ti reference setup.
"""
import json
from pathlib import Path

import pytest

from reactx.cli import main

_SN2_WALLCLOCK = """\
description = "SN2 wallclock"
formed = [[1, 3]]
broken = [[1, 2]]
[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 100
[sampling]
n_candidates = 16
[neb]
top_k = 2
n_images = 5
max_steps = 30
[parallel]
screening_workers = 2
neb_workers = 2
"""


@pytest.mark.slow
def test_re1_sn2_wallclock_below_900s(tmp_path: Path, tmp_rxn_with_toml):
    rxn = tmp_rxn_with_toml("sn2", toml_body=_SN2_WALLCLOCK)
    out = tmp_path / "sn2_wc"
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "uma"])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())
    wc = meta["wall_clock_seconds"]
    assert wc > 0, f"wall_clock_seconds not recorded: {wc}"
    assert wc < 900.0, f"wall_clock_seconds={wc:.1f}s exceeds 900s upper bound"
    # Phase 8 also reports per-stage breakdown.
    breakdown = meta["wall_clock_breakdown"]
    assert "screening" in breakdown
    assert "neb" in breakdown
