"""Tests for Phase 11 config schema."""
from pathlib import Path

import pytest

from reactx.config import (
    ConfigError,
    EndpointRelaxSection,
    NEBSection,
    PlacementSection,
    load_config,
)


def _write_rxn_and_toml(tmp_path: Path, body: str) -> Path:
    rxn = tmp_path / "x.rxn"
    rxn.write_text("$RXN\n\n  RDKit\n\n  0  0\n")
    (tmp_path / "x.rxn.toml").write_text(body)
    return rxn


def test_minimal_config_only_description(tmp_path):
    rxn = _write_rxn_and_toml(tmp_path, 'description = "minimal"\n')
    cfg = load_config(rxn)
    assert cfg.description == "minimal"
    assert cfg.placement == PlacementSection()
    assert cfg.endpoint_relax == EndpointRelaxSection()
    assert cfg.neb == NEBSection()


def test_default_values(tmp_path):
    rxn = _write_rxn_and_toml(tmp_path, 'description = "x"\n')
    cfg = load_config(rxn)
    assert cfg.placement.initial_separation == 4.0
    assert cfg.placement.orientation == "default"
    assert cfg.endpoint_relax.fmax == 0.01
    assert cfg.endpoint_relax.max_steps == 500
    assert cfg.endpoint_relax.optimizer == "FIRE"
    assert cfg.neb.n_images == 11
    assert cfg.neb.fmax == 0.05
    assert cfg.neb.max_steps == 200
    assert cfg.neb.k == 1.0
    assert cfg.neb.climb is True
    assert cfg.neb.pad_frames == 0


def test_all_sections_explicit(tmp_path):
    body = '''description = "all"

[placement]
initial_separation = 5.5
orientation = "endo"

[endpoint_relax]
fmax = 0.005
max_steps = 800
optimizer = "BFGS"

[neb]
n_images = 13
fmax = 0.03
max_steps = 250
k = 0.5
climb = false
pad_frames = 2
'''
    rxn = _write_rxn_and_toml(tmp_path, body)
    cfg = load_config(rxn)
    assert cfg.placement.initial_separation == 5.5
    assert cfg.placement.orientation == "endo"
    assert cfg.endpoint_relax.fmax == 0.005
    assert cfg.endpoint_relax.max_steps == 800
    assert cfg.endpoint_relax.optimizer == "BFGS"
    assert cfg.neb.n_images == 13
    assert cfg.neb.fmax == 0.03
    assert cfg.neb.max_steps == 250
    assert cfg.neb.k == 0.5
    assert cfg.neb.climb is False
    assert cfg.neb.pad_frames == 2


@pytest.mark.parametrize(
    "obsolete",
    [
        "[afir]\nalpha_formed = 1.0\nalpha_broken = 1.0\nmax_relax_steps = 100\n",
        "[scoring]\nr_broken_threshold = 3.0\n",
        "[sampling]\nn_candidates = 32\n",
        "formed = [[1, 2]]\n",
        "broken = [[1, 2]]\n",
        "[restraints]\nk_form = 1.0\n",
        "[prescreen]\nn_angles = 8\n",
    ],
)
def test_obsolete_keys_rejected(tmp_path, obsolete):
    body = 'description = "x"\n' + obsolete
    rxn = _write_rxn_and_toml(tmp_path, body)
    with pytest.raises(ConfigError, match="Phase 11"):
        load_config(rxn)


@pytest.mark.parametrize(
    "bad_val,scope,key",
    [
        (0.0, "placement", "initial_separation"),
        (-1.0, "placement", "initial_separation"),
        ("invalid_orient", "placement", "orientation"),
        (0.0, "endpoint_relax", "fmax"),
        (-0.1, "endpoint_relax", "fmax"),
        (0, "endpoint_relax", "max_steps"),
        ("Newton", "endpoint_relax", "optimizer"),
        (2, "neb", "n_images"),
        (0.0, "neb", "fmax"),
        (0, "neb", "max_steps"),
        (0.0, "neb", "k"),
        (-1, "neb", "pad_frames"),
    ],
)
def test_invalid_values_rejected(tmp_path, bad_val, scope, key):
    body = f'description = "x"\n[{scope}]\n{key} = {bad_val!r}\n'
    rxn = _write_rxn_and_toml(tmp_path, body)
    with pytest.raises(ConfigError):
        load_config(rxn)


def test_missing_description_rejected(tmp_path):
    rxn = _write_rxn_and_toml(tmp_path, "[placement]\ninitial_separation = 3.0\n")
    with pytest.raises(ConfigError, match="description"):
        load_config(rxn)


def test_missing_toml_raises_file_not_found(tmp_path):
    rxn = tmp_path / "x.rxn"
    rxn.write_text("$RXN\n")
    with pytest.raises(FileNotFoundError):
        load_config(rxn)


def test_unknown_top_level_key_rejected(tmp_path):
    body = 'description = "x"\nweird_key = 42\n'
    rxn = _write_rxn_and_toml(tmp_path, body)
    with pytest.raises(ConfigError, match="unknown"):
        load_config(rxn)
