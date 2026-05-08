"""Smoke: each example TOML must validate after Phase 9 migration."""
from pathlib import Path

import pytest

from reactx.config import load_config_v9 as load_config

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.mark.parametrize("rxn_name", [
    "sn2.rxn", "proton_transfer.rxn", "menshutkin.rxn", "e2.rxn",
    "sn1_dissoc.rxn", "sn1_recomb.rxn",
    "diels_alder_simple.rxn", "diels_alder_endo.rxn",
])
def test_example_loads(rxn_name: str):
    cfg = load_config(EXAMPLES / rxn_name)
    assert cfg.description
    assert cfg.afir.max_relax_steps > 0
