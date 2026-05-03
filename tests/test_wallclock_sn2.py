"""Wall-clock sanity check for SN2 default settings with prescreen ON.

With MMFF prescreen reducing UMA runs from 8 to 3, the dev-machine target
is ~25-30s. The hard upper bound is set at 60s to leave headroom for
hardware variation while still flagging gross regressions.
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
n_angles = 8
"""


@pytest.mark.slow
def test_re1_sn2_wallclock_below_60s(tmp_path: Path, tmp_rxn_with_toml):
    rxn = tmp_rxn_with_toml("sn2", toml_body=_SN2_WALLCLOCK)
    out = tmp_path / "sn2_wc"
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "uma"])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())
    wc = meta["wall_clock_seconds"]
    assert wc > 0, f"wall_clock_seconds not recorded: {wc}"
    assert wc < 60.0, f"wall_clock_seconds={wc:.1f}s exceeds 60s upper bound"
