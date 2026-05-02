"""Verify --no-mmff-prescreen restores Phase Re1 default behaviour.

Slow: requires UMA + GPU. Pinned to a small n_angles to keep wall-clock
reasonable while exercising the disabled-prescreen branch end-to-end.
"""
import json
from pathlib import Path

import pytest

from reactx.cli import main


@pytest.mark.slow
def test_re1_sn2_prescreen_disabled(tmp_path: Path, sn2_rxn_path: Path):
    out = tmp_path / "sn2_no_pre"
    rc = main([
        "run", str(sn2_rxn_path), "-o", str(out),
        "--backend", "uma",
        "--n-angles", "3",
        "--max-relax-steps", "30",
        "--no-mmff-prescreen",
    ])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())
    pre = meta["prescreen"]
    assert pre["enabled"] is False
    assert pre["kept"] is None
    assert pre["skipped"] is None
    assert pre["mmff_failed"] is None
    assert pre["wall_clock_seconds"] == 0.0
    # All n_angles trials went through UMA when prescreen is disabled.
    assert len(meta["trials"]) == 3
