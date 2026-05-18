"""Tests for explicit endpoint config schema."""
from pathlib import Path

import pytest

from reactx.config import ConfigError, EndpointRelaxSection, NEBSection, load_config


def _write_rxn_and_toml(tmp_path: Path, body: str) -> Path:
    rxn = tmp_path / "x.rxn"
    rxn.write_text("$RXN\n\n  RDKit\n\n  0  0\n", encoding="utf-8")
    (tmp_path / "x.rxn.toml").write_text(body, encoding="utf-8")
    return rxn


def test_minimal_config_requires_endpoint_paths(tmp_path):
    rxn = _write_rxn_and_toml(
        tmp_path,
        """description = "minimal"
reactant_structure = "x.reactant.xyz"
product_structure = "x.product.xyz"
""",
    )
    cfg = load_config(rxn)
    assert cfg.description == "minimal"
    assert cfg.reactant_structure == "x.reactant.xyz"
    assert cfg.product_structure == "x.product.xyz"
    assert cfg.endpoint_relax == EndpointRelaxSection()
    assert cfg.neb == NEBSection()


def test_default_values(tmp_path):
    rxn = _write_rxn_and_toml(
        tmp_path,
        """description = "x"
reactant_structure = "r.xyz"
product_structure = "p.xyz"
""",
    )
    cfg = load_config(rxn)
    assert cfg.endpoint_relax.fmax == 0.01
    assert cfg.endpoint_relax.max_steps == 500
    assert cfg.endpoint_relax.optimizer == "FIRE"
    assert cfg.neb.n_images == 11
    assert cfg.neb.fmax == 0.05
    assert cfg.neb.max_steps == 200
    assert cfg.neb.k == 1.0
    assert cfg.neb.method == "eb"
    assert cfg.neb.remove_rotation_and_translation is True
    assert cfg.neb.climb is True
    assert cfg.neb.pad_frames == 0
    assert cfg.neb.guide_bond_changes is True
    assert cfg.neb.guide_k == 0.25


def test_all_sections_explicit(tmp_path):
    body = """description = "all"
reactant_structure = "r.xyz"
product_structure = "p.xyz"

[endpoint_relax]
fmax = 0.005
max_steps = 800
optimizer = "BFGS"

[neb]
n_images = 13
fmax = 0.03
max_steps = 250
k = 0.5
method = "improvedtangent"
remove_rotation_and_translation = false
climb = false
pad_frames = 2
guide_bond_changes = false
guide_k = 0.75
"""
    rxn = _write_rxn_and_toml(tmp_path, body)
    cfg = load_config(rxn)
    assert cfg.reactant_structure == "r.xyz"
    assert cfg.product_structure == "p.xyz"
    assert cfg.endpoint_relax.fmax == 0.005
    assert cfg.endpoint_relax.max_steps == 800
    assert cfg.endpoint_relax.optimizer == "BFGS"
    assert cfg.neb.n_images == 13
    assert cfg.neb.fmax == 0.03
    assert cfg.neb.max_steps == 250
    assert cfg.neb.k == 0.5
    assert cfg.neb.method == "improvedtangent"
    assert cfg.neb.remove_rotation_and_translation is False
    assert cfg.neb.climb is False
    assert cfg.neb.pad_frames == 2
    assert cfg.neb.guide_bond_changes is False
    assert cfg.neb.guide_k == 0.75


@pytest.mark.parametrize(
    "body,match",
    [
        ('reactant_structure = "r.xyz"\nproduct_structure = "p.xyz"\n', "description"),
        ('description = "x"\nproduct_structure = "p.xyz"\n', "reactant_structure"),
        ('description = "x"\nreactant_structure = "r.xyz"\n', "product_structure"),
        ('description = "x"\nreactant_structure = ""\nproduct_structure = "p.xyz"\n', "reactant_structure"),
        ('description = "x"\nreactant_structure = "r.xyz"\nproduct_structure = ""\n', "product_structure"),
        ('description = "x"\nreactant_structure = "   "\nproduct_structure = "p.xyz"\n', "reactant_structure"),
        ('description = "x"\nreactant_structure = "r.xyz"\nproduct_structure = "   "\n', "product_structure"),
    ],
)
def test_required_values_rejected(tmp_path, body, match):
    rxn = _write_rxn_and_toml(tmp_path, body)
    with pytest.raises(ConfigError, match=match):
        load_config(rxn)


@pytest.mark.parametrize(
    "obsolete",
    [
        "[placement]\norientation = \"endo\"\n",
        "[placement]\nn_candidates = 32\n",
        "[afir]\nalpha_formed = 1.0\n",
        "[scoring]\nr_broken_threshold = 3.0\n",
        "[sampling]\nn_candidates = 32\n",
        "formed = [[1, 2]]\n",
        "broken = [[1, 2]]\n",
        "[restraints]\nk_form = 1.0\n",
        "[prescreen]\nn_angles = 8\n",
    ],
)
def test_obsolete_keys_rejected(tmp_path, obsolete):
    body = """description = "x"
reactant_structure = "r.xyz"
product_structure = "p.xyz"
""" + obsolete
    rxn = _write_rxn_and_toml(tmp_path, body)
    with pytest.raises(ConfigError, match="explicit endpoint"):
        load_config(rxn)


@pytest.mark.parametrize(
    "bad_val,scope,key",
    [
        (0.0, "endpoint_relax", "fmax"),
        (-0.1, "endpoint_relax", "fmax"),
        (0, "endpoint_relax", "max_steps"),
        ("Newton", "endpoint_relax", "optimizer"),
        (2, "neb", "n_images"),
        (0.0, "neb", "fmax"),
        (0, "neb", "max_steps"),
        (0.0, "neb", "k"),
        ("badmethod", "neb", "method"),
        ("true", "neb", "remove_rotation_and_translation"),
        (-1, "neb", "pad_frames"),
        ("yes", "neb", "guide_bond_changes"),
        (0.0, "neb", "guide_k"),
        (-0.1, "neb", "guide_k"),
    ],
)
def test_invalid_values_rejected(tmp_path, bad_val, scope, key):
    body = f"""description = "x"
reactant_structure = "r.xyz"
product_structure = "p.xyz"
[{scope}]
{key} = {bad_val!r}
"""
    rxn = _write_rxn_and_toml(tmp_path, body)
    with pytest.raises(ConfigError):
        load_config(rxn)


def test_missing_toml_raises_file_not_found(tmp_path):
    rxn = tmp_path / "x.rxn"
    rxn.write_text("$RXN\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_config(rxn)


def test_unknown_top_level_key_rejected(tmp_path):
    body = """description = "x"
reactant_structure = "r.xyz"
product_structure = "p.xyz"
weird_key = 42
"""
    rxn = _write_rxn_and_toml(tmp_path, body)
    with pytest.raises(ConfigError, match="unknown"):
        load_config(rxn)
