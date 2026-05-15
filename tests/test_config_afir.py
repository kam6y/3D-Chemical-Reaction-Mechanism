"""Phase 9 [afir] + [scoring] schema (additive — coexists with old [restraints]).

Spec: docs/superpowers/specs/2026-05-08-afir-force-design.md §4.4, §6.2
"""
from pathlib import Path

import pytest

from reactx.config import ConfigError, load_config


def _write(tmp_path: Path, body: str) -> Path:
    rxn = tmp_path / "x.rxn"
    rxn.write_text("placeholder", encoding="utf-8")
    (tmp_path / "x.rxn.toml").write_text(body, encoding="utf-8")
    return rxn


def test_load_minimal_sn2(tmp_path: Path):
    rxn = _write(tmp_path, """
description = "SN2"
formed = [[1, 3]]
broken = [[1, 2]]

[afir]
alpha_formed = 0.7
alpha_broken = 0.5
max_relax_steps = 100

[scoring]
r_broken_threshold = 4.0
""")
    cfg = load_config(rxn)
    assert cfg.afir.alpha_formed == 0.7
    assert cfg.afir.alpha_broken == 0.5
    assert cfg.afir.max_relax_steps == 100
    assert cfg.scoring.r_broken_threshold == 4.0
    assert cfg.scoring.r_formed_threshold is None


def test_load_e2_per_bond_lists(tmp_path: Path):
    rxn = _write(tmp_path, """
description = "E2"
formed = [[4, 5]]
broken = [[2, 5], [1, 3]]

[afir]
alpha_formed = 1.5
alpha_broken = [1.0, 1.5]
max_relax_steps = 200

[scoring]
r_broken_threshold = [3.0, 4.0]
""")
    cfg = load_config(rxn)
    assert cfg.afir.alpha_broken == (1.0, 1.5)
    assert cfg.scoring.r_broken_threshold == (3.0, 4.0)


def test_load_da_with_empty_broken_alpha_broken_omitted(tmp_path: Path):
    rxn = _write(tmp_path, """
description = "DA"
formed = [[1, 5], [4, 6]]
broken = []

[afir]
alpha_formed = [2.5, 2.5]
max_relax_steps = 200
""")
    cfg = load_config(rxn)
    assert cfg.broken == ()
    assert cfg.afir.alpha_broken == ()


def test_load_da_with_empty_broken_alpha_broken_zero(tmp_path: Path):
    rxn = _write(tmp_path, """
description = "DA"
formed = [[1, 5], [4, 6]]
broken = []

[afir]
alpha_formed = [2.5, 2.5]
alpha_broken = 0.0
max_relax_steps = 200
""")
    cfg = load_config(rxn)
    assert cfg.broken == ()


def test_load_sn1_dissoc_with_empty_formed(tmp_path: Path):
    rxn = _write(tmp_path, """
description = "SN1 dissoc"
formed = []
broken = [[1, 5]]

[afir]
alpha_broken = 1.5
max_relax_steps = 200

[scoring]
r_broken_threshold = 6.0
""")
    cfg = load_config(rxn)
    assert cfg.formed == ()
    assert cfg.afir.alpha_formed == ()


def test_old_restraints_section_rejected(tmp_path: Path):
    rxn = _write(tmp_path, """
description = "old"
formed = [[1, 2]]
broken = []

[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 100
""")
    with pytest.raises(ConfigError, match=r"restraints"):
        load_config(rxn)


def test_alpha_zero_rejected_for_nonempty_pair(tmp_path: Path):
    rxn = _write(tmp_path, """
description = "α=0"
formed = [[1, 2]]
broken = [[3, 4]]

[afir]
alpha_formed = 0.0
alpha_broken = 1.0
max_relax_steps = 100

[scoring]
r_broken_threshold = 4.0
""")
    with pytest.raises(ConfigError, match="alpha_formed"):
        load_config(rxn)


def test_alpha_negative_rejected(tmp_path: Path):
    rxn = _write(tmp_path, """
description = "α<0"
formed = [[1, 2]]
broken = []

[afir]
alpha_formed = -1.0
max_relax_steps = 100
""")
    with pytest.raises(ConfigError):
        load_config(rxn)


def test_r_broken_threshold_required_when_broken_nonempty(tmp_path: Path):
    rxn = _write(tmp_path, """
description = "missing r_broken"
formed = []
broken = [[1, 2]]

[afir]
alpha_broken = 1.0
max_relax_steps = 100
""")
    with pytest.raises(ConfigError, match="r_broken_threshold"):
        load_config(rxn)


def test_nan_alpha_rejected(tmp_path: Path):
    rxn = _write(tmp_path, """
description = "nan"
formed = [[1, 2]]
broken = []

[afir]
alpha_formed = nan
max_relax_steps = 100
""")
    with pytest.raises(ConfigError, match="finite"):
        load_config(rxn)


def test_inf_alpha_rejected(tmp_path: Path):
    rxn = _write(tmp_path, """
description = "inf"
formed = [[1, 2]]
broken = []

[afir]
alpha_formed = inf
max_relax_steps = 100
""")
    with pytest.raises(ConfigError, match="finite"):
        load_config(rxn)


def test_alpha_length_mismatch_rejected(tmp_path: Path):
    rxn = _write(tmp_path, """
description = "mismatch"
formed = [[1, 2]]
broken = [[3, 4]]

[afir]
alpha_formed = [1.0, 2.0]
alpha_broken = 0.5
max_relax_steps = 100

[scoring]
r_broken_threshold = 4.0
""")
    with pytest.raises(ConfigError, match="alpha_formed"):
        load_config(rxn)


def test_r_formed_threshold_list_passthrough(tmp_path: Path):
    rxn = _write(tmp_path, """
description = "list r_formed"
formed = [[1, 2], [3, 4]]
broken = []

[afir]
alpha_formed = [1.0, 2.0]
max_relax_steps = 100

[scoring]
r_formed_threshold = [1.5, 1.8]
""")
    cfg = load_config(rxn)
    assert cfg.scoring.r_formed_threshold == (1.5, 1.8)


def test_max_relax_steps_must_be_positive(tmp_path: Path):
    rxn = _write(tmp_path, """
description = "zero"
formed = [[1, 2]]
broken = []

[afir]
alpha_formed = 1.0
max_relax_steps = 0
""")
    with pytest.raises(ConfigError, match="max_relax_steps"):
        load_config(rxn)


def test_obsolete_top_level_keys_rejected(tmp_path: Path):
    for old_key in ["k_form", "k_broken", "r_broken", "r_form"]:
        body = f"""
description = "old key"
formed = [[1, 2]]
broken = []
{old_key} = 0.5

[afir]
alpha_formed = 1.0
max_relax_steps = 100
"""
        rxn = _write(tmp_path, body)
        with pytest.raises(ConfigError):
            load_config(rxn)
