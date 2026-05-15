"""Wall-clock sanity check for SN2 default settings under Phase 7.

Phase 7 removed the MMFF prescreen, so UMA runs on every blocking-survivor
candidate. The hard upper bound is loosened to 180s to absorb the larger
UMA workload and still flag gross regressions.
"""
import json
from pathlib import Path

import pytest

from reactx.cli import main

_SN2_WALLCLOCK = """\
description = "SN2 wallclock"
formed = [[1, 3]]
broken = [[1, 2]]

[afir]
alpha_formed = 0.5
alpha_broken = 1.0
max_relax_steps = 100

[scoring]
r_broken_threshold = 4.0

[sampling]
n_candidates = 16
"""


@pytest.mark.slow
def test_re1_sn2_wallclock_below_180s(tmp_path: Path, tmp_rxn_with_toml):
    rxn = tmp_rxn_with_toml("sn2", toml_body=_SN2_WALLCLOCK)
    out = tmp_path / "sn2_wc"
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "uma"])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())
    wc = meta["wall_clock_seconds"]
    assert wc > 0, f"wall_clock_seconds not recorded: {wc}"
    assert wc < 180.0, f"wall_clock_seconds={wc:.1f}s exceeds 180s upper bound"
