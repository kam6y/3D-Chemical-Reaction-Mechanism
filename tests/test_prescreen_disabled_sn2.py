"""Verify prescreen.enabled=false in TOML restores the all-trials-through-UMA path.

Slow: requires UMA + GPU. Pinned to a small n_angles to keep wall-clock
reasonable while exercising the disabled-prescreen branch end-to-end.
"""
import json
from pathlib import Path

import pytest

from reactx.cli import main

_SN2_NO_PRESCREEN = """\
description = "SN2 no-prescreen"
formed = [[1, 3]]
broken = [[1, 2]]
[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 30
[sampling]
n_angles = 3
[prescreen]
enabled = false
"""


@pytest.mark.slow
def test_re1_sn2_prescreen_disabled(tmp_path: Path, tmp_rxn_with_toml):
    rxn = tmp_rxn_with_toml("sn2", toml_body=_SN2_NO_PRESCREEN)
    out = tmp_path / "sn2_no_pre"
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "uma"])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())
    pre = meta["prescreen"]
    assert pre["enabled"] is False
    assert pre["kept"] is None
    assert pre["skipped"] is None
    assert pre["mmff_failed"] is None
    assert pre["wall_clock_seconds"] == 0.0
    assert len(meta["trials"]) == 3
