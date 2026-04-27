"""Wall-clock sanity check for Phase Re1 SN2 default settings.

Phase 0 NEB baseline on the same hardware is the comparison target; this test
records the wall-clock for the new pipeline so the README can quote it.
The hard upper bound is intentionally loose (≤ 300s) to stay environment-
agnostic; a dev-machine target ≤ 60s is documented but not asserted.
"""
import json
from pathlib import Path

import pytest

from reactx.cli import main


@pytest.mark.slow
def test_re1_sn2_wallclock_below_300s(tmp_path: Path, sn2_rxn_path: Path):
    out = tmp_path / "sn2_wc"
    rc = main([
        "run", str(sn2_rxn_path), "-o", str(out),
        "--backend", "uma",
    ])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())
    wc = meta["wall_clock_seconds"]
    assert wc > 0, f"wall_clock_seconds not recorded: {wc}"
    # Loose absolute bound; tighten in README based on actual measurement
    assert wc < 300.0, f"wall_clock_seconds={wc:.1f}s exceeds 300s upper bound"
