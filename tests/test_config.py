"""Tier A (.rxn.toml schema) validation tests for reactx.config."""
from pathlib import Path

import pytest

from reactx.config import (
    PrescreenConfig,
    ReactionConfig,
    RestraintConfig,
    SamplingConfig,
    load_config,
    resolve_r_form_targets,
)


def _write(tmp_path: Path, body: str) -> Path:
    rxn = tmp_path / "x.rxn"
    rxn.write_text("$RXN\n")  # body irrelevant for Tier A
    (tmp_path / "x.rxn.toml").write_text(body)
    return rxn


def test_load_minimal_required_keys(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "test"
formed = [[1, 2]]
broken = []
[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 100
""")
    cfg = load_config(rxn)
    assert isinstance(cfg, ReactionConfig)
    assert cfg.description == "test"
    assert cfg.formed == ((1, 2),)
    assert cfg.broken == ()
    assert cfg.restraints == RestraintConfig(
        k_form=0.5, k_broken=1.0, r_broken=4.0,
        max_relax_steps=100, r_form=None,
    )
    assert cfg.sampling == SamplingConfig(n_angles=8, cone_half_deg=30.0)
    assert cfg.prescreen == PrescreenConfig(enabled=True, keep=3, steps=30)


def test_load_full_keys(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "full"
formed = [[1, 5], [2, 6]]
broken = [[3, 4]]
[restraints]
k_form = 1.0
k_broken = 2.0
r_broken = 5.0
max_relax_steps = 200
r_form = [1.78, 1.05]
[sampling]
n_angles = 1
cone_half_deg = 45.0
[prescreen]
enabled = false
keep = 5
steps = 50
""")
    cfg = load_config(rxn)
    assert cfg.formed == ((1, 5), (2, 6))
    assert cfg.broken == ((3, 4),)
    assert cfg.restraints.r_form == (1.78, 1.05)
    assert cfg.sampling.n_angles == 1
    assert cfg.sampling.cone_half_deg == 45.0
    assert cfg.prescreen.enabled is False
    assert cfg.prescreen.keep == 5
    assert cfg.prescreen.steps == 50


def test_load_scalar_r_form(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "pt"
formed = [[1, 3]]
broken = [[1, 2]]
[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 100
r_form = 1.05
""")
    cfg = load_config(rxn)
    assert cfg.restraints.r_form == 1.05


def test_missing_sidecar_raises_filenotfound(tmp_path: Path):
    rxn = tmp_path / "missing.rxn"
    rxn.write_text("$RXN\n")
    with pytest.raises(FileNotFoundError, match="sidecar TOML not found"):
        load_config(rxn)


def test_unknown_top_level_key_rejected(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "x"
formed = [[1, 2]]
broken = []
foo = 1
[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 100
""")
    with pytest.raises(ValueError, match="unknown config key.*foo"):
        load_config(rxn)


def test_unknown_restraints_key_rejected(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "x"
formed = [[1, 2]]
broken = []
[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 100
kept = 9
""")
    with pytest.raises(ValueError, match="unknown config key.*kept"):
        load_config(rxn)


def test_missing_required_restraint_key_rejected(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "x"
formed = [[1, 2]]
broken = []
[restraints]
k_form = 0.5
r_broken = 4.0
max_relax_steps = 100
""")
    with pytest.raises(ValueError, match="missing required.*k_broken"):
        load_config(rxn)


def test_empty_formed_and_broken_rejected(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "x"
formed = []
broken = []
[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 100
""")
    with pytest.raises(ValueError, match="at least one of 'formed' or 'broken'"):
        load_config(rxn)


def test_negative_k_form_rejected(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "x"
formed = [[1, 2]]
broken = []
[restraints]
k_form = -0.1
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 100
""")
    with pytest.raises(ValueError, match="k_form.*>= 0"):
        load_config(rxn)


def test_zero_n_angles_rejected(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "x"
formed = [[1, 2]]
broken = []
[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 100
[sampling]
n_angles = 0
""")
    with pytest.raises(ValueError, match="n_angles.*> 0"):
        load_config(rxn)


def test_r_form_list_length_mismatch_rejected(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "x"
formed = [[1, 2]]
broken = []
[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 100
r_form = [1.0, 2.0]
""")
    with pytest.raises(ValueError, match="r_form.*list length 2.*formed.*=1"):
        load_config(rxn)


def test_non_table_section_rejected(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "x"
formed = [[1, 2]]
broken = []
restraints = 5
""")
    with pytest.raises(ValueError, match="'restraints' must be a table"):
        load_config(rxn)


def test_formed_pair_must_be_two_ints(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "x"
formed = [[1, 2, 3]]
broken = []
[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 100
""")
    with pytest.raises(ValueError, match=r"formed\[0\].*\[int,int\]"):
        load_config(rxn)


def test_resolve_r_form_targets_lookup_when_none():
    cfg = ReactionConfig(
        description="x",
        formed=((1, 2),),
        broken=(),
        restraints=RestraintConfig(
            k_form=0.5, k_broken=1.0, r_broken=4.0,
            max_relax_steps=100, r_form=None,
        ),
    )
    syms = ["C", "O"]
    out = resolve_r_form_targets(cfg, syms, [(0, 1)])
    assert out == [pytest.approx(1.43)]  # element-table C-O


def test_resolve_r_form_targets_scalar_broadcast():
    cfg = ReactionConfig(
        description="x",
        formed=((1, 2), (3, 4)),
        broken=(),
        restraints=RestraintConfig(
            k_form=0.5, k_broken=1.0, r_broken=4.0,
            max_relax_steps=100, r_form=1.10,
        ),
    )
    out = resolve_r_form_targets(cfg, ["C", "C", "C", "C"], [(0, 1), (2, 3)])
    assert out == [1.10, 1.10]


def test_resolve_r_form_targets_list_passthrough():
    cfg = ReactionConfig(
        description="x",
        formed=((1, 2), (3, 4)),
        broken=(),
        restraints=RestraintConfig(
            k_form=0.5, k_broken=1.0, r_broken=4.0,
            max_relax_steps=100, r_form=(1.78, 1.05),
        ),
    )
    out = resolve_r_form_targets(cfg, ["C", "C", "C", "C"], [(0, 1), (2, 3)])
    assert out == [1.78, 1.05]


def test_resolve_r_form_targets_empty_formed():
    cfg = ReactionConfig(
        description="x",
        formed=(),
        broken=((1, 2),),
        restraints=RestraintConfig(
            k_form=0.5, k_broken=1.0, r_broken=4.0,
            max_relax_steps=100, r_form=None,
        ),
    )
    assert resolve_r_form_targets(cfg, ["C", "Br"], []) == []
