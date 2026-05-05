"""Tier A (.rxn.toml schema) validation tests for reactx.config."""
from pathlib import Path

import pytest

from reactx.config import (
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
    assert cfg.sampling == SamplingConfig(n_candidates=64)
    assert not hasattr(cfg, "prescreen")


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
n_candidates = 32
""")
    cfg = load_config(rxn)
    assert cfg.formed == ((1, 5), (2, 6))
    assert cfg.broken == ((3, 4),)
    assert cfg.restraints.r_form == (1.78, 1.05)
    assert cfg.sampling.n_candidates == 32


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


def test_whitespace_only_description_rejected(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "   "
formed = [[1, 2]]
broken = []
[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 100
""")
    with pytest.raises(ValueError, match="'description' must be a non-empty string"):
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


def test_zero_n_candidates_rejected(tmp_path: Path):
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
n_candidates = 0
""")
    with pytest.raises(ValueError, match=r"n_candidates.*> 0"):
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


def test_legacy_n_angles_key_rejected(tmp_path: Path):
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
n_angles = 8
""")
    with pytest.raises(ValueError, match="n_angles"):
        load_config(rxn)


def test_legacy_cone_half_deg_key_rejected(tmp_path: Path):
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
n_candidates = 16
cone_half_deg = 30.0
""")
    with pytest.raises(ValueError, match="cone_half_deg"):
        load_config(rxn)


def test_legacy_prescreen_section_rejected(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "x"
formed = [[1, 2]]
broken = []
[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 100
[prescreen]
enabled = true
keep = 3
steps = 30
""")
    with pytest.raises(ValueError, match=r"\[prescreen\] section is removed in Phase 7"):
        load_config(rxn)


def test_default_n_candidates_is_64(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "x"
formed = [[1, 2]]
broken = []
[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 100
""")
    cfg = load_config(rxn)
    assert cfg.sampling.n_candidates == 64


def test_load_k_form_as_list_of_two(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "DA"
formed = [[1, 5], [4, 6]]
broken = []
[restraints]
k_form = [1.0, 1.5]
k_broken = 0.0
r_broken = 4.0
max_relax_steps = 200
""")
    cfg = load_config(rxn)
    assert cfg.restraints.k_form == (1.0, 1.5)


def test_k_form_list_length_must_match_formed(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "bad"
formed = [[1, 5], [4, 6]]
broken = []
[restraints]
k_form = [1.0]
k_broken = 0.0
r_broken = 4.0
max_relax_steps = 200
""")
    with pytest.raises(ValueError, match="k_form.*list length"):
        load_config(rxn)


def test_k_form_list_negative_value_rejected(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "bad"
formed = [[1, 5], [4, 6]]
broken = []
[restraints]
k_form = [1.0, -0.1]
k_broken = 0.0
r_broken = 4.0
max_relax_steps = 200
""")
    with pytest.raises(ValueError, match="k_form.*must be >= 0"):
        load_config(rxn)


def test_load_k_broken_as_list_of_two_for_e2_pattern(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "E2 per-bond k"
formed = [[4, 5]]
broken = [[2, 5], [1, 3]]
[restraints]
k_form = 1.0
k_broken = [1.0, 2.0]
r_broken = 4.0
max_relax_steps = 200
""")
    cfg = load_config(rxn)
    assert cfg.restraints.k_broken == (1.0, 2.0)


def test_k_broken_list_length_must_match_broken(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "bad"
formed = [[4, 5]]
broken = [[2, 5], [1, 3]]
[restraints]
k_form = 1.0
k_broken = [1.0]
r_broken = 4.0
max_relax_steps = 200
""")
    with pytest.raises(ValueError, match="k_broken.*list length"):
        load_config(rxn)


def test_k_broken_scalar_with_empty_broken_filler_ok(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "DA broken empty"
formed = [[1, 5], [4, 6]]
broken = []
[restraints]
k_form = [1.0, 1.0]
k_broken = 0.0
r_broken = 4.0
max_relax_steps = 200
""")
    cfg = load_config(rxn)
    assert cfg.restraints.k_broken == 0.0


def test_load_r_broken_as_list_for_e2(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "E2 per-bond r_broken"
formed = [[4, 5]]
broken = [[2, 5], [1, 3]]
[restraints]
k_form = 1.0
k_broken = 1.0
r_broken = [3.0, 5.0]
max_relax_steps = 200
""")
    cfg = load_config(rxn)
    assert cfg.restraints.r_broken == (3.0, 5.0)


def test_r_broken_list_must_be_positive(tmp_path: Path):
    rxn = _write(tmp_path, """\
description = "bad"
formed = [[4, 5]]
broken = [[2, 5], [1, 3]]
[restraints]
k_form = 1.0
k_broken = 1.0
r_broken = [3.0, 0.0]
max_relax_steps = 200
""")
    with pytest.raises(ValueError, match="r_broken.*must be > 0"):
        load_config(rxn)
