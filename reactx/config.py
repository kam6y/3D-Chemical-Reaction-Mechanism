"""Per-reaction sidecar TOML config for explicit endpoint inputs."""
from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path


class ConfigError(ValueError):
    """Raised when `<rxn_path>.toml` violates the Phase 11 schema."""


@dataclass(frozen=True)
class EndpointRelaxSection:
    fmax: float = 0.01
    max_steps: int = 500
    optimizer: str = "FIRE"


@dataclass(frozen=True)
class NEBSection:
    n_images: int = 11
    fmax: float = 0.05
    max_steps: int = 200
    k: float = 1.0
    method: str = "eb"
    remove_rotation_and_translation: bool = True
    climb: bool = True
    pad_frames: int = 0
    guide_bond_changes: bool = True
    guide_k: float = 5.0


@dataclass(frozen=True)
class ReactionConfig:
    description: str
    reactant_structure: str
    product_structure: str
    endpoint_relax: EndpointRelaxSection
    neb: NEBSection


_TOP_LEVEL_KEYS = {
    "description",
    "reactant_structure",
    "product_structure",
    "endpoint_relax",
    "neb",
}
_TOP_LEVEL_REQUIRED = {"description", "reactant_structure", "product_structure"}
_OBSOLETE_TOP_LEVEL = {
    "afir",
    "alpha_broken",
    "alpha_formed",
    "broken",
    "formed",
    "k_broken",
    "k_form",
    "max_relax_steps",
    "placement",
    "prescreen",
    "r_broken",
    "r_broken_threshold",
    "r_form",
    "r_formed",
    "r_formed_threshold",
    "restraints",
    "sampling",
    "scoring",
}
_ENDPOINT_KEYS = {"fmax", "max_steps", "optimizer"}
_NEB_KEYS = {
    "n_images",
    "fmax",
    "max_steps",
    "k",
    "method",
    "remove_rotation_and_translation",
    "climb",
    "pad_frames",
    "guide_bond_changes",
    "guide_k",
}
_VALID_OPTIMIZERS = {"FIRE", "BFGS"}
_VALID_NEB_METHODS = {"aseneb", "improvedtangent", "eb", "spline", "string"}


def sidecar_path(rxn_path: Path) -> Path:
    """`examples/sn2.rxn` -> `examples/sn2.rxn.toml`."""
    return rxn_path.parent / (rxn_path.name + ".toml")


def load_config(rxn_path: Path) -> ReactionConfig:
    """Load `<rxn_path>.toml` and return a validated ReactionConfig."""
    toml_path = sidecar_path(rxn_path)
    if not toml_path.is_file():
        raise FileNotFoundError(
            f"sidecar TOML not found: expected {toml_path} alongside {rxn_path}"
        )
    raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    return _validate(raw, source=str(toml_path))


def _validate(raw: dict, *, source: str) -> ReactionConfig:
    for key in _OBSOLETE_TOP_LEVEL:
        if key in raw:
            raise ConfigError(
                f"{source}: '{key}' is obsolete with explicit endpoint inputs. "
                "Use reactant_structure and product_structure instead."
            )

    _check_keys(raw, _TOP_LEVEL_KEYS, _TOP_LEVEL_REQUIRED, scope="<top>", source=source)

    description = raw["description"]
    if not isinstance(description, str) or not description.strip():
        raise ConfigError(f"{source}: 'description' must be a non-empty string")
    reactant_structure = _endpoint_path(
        raw["reactant_structure"],
        key="reactant_structure",
        source=source,
    )
    product_structure = _endpoint_path(
        raw["product_structure"],
        key="product_structure",
        source=source,
    )

    endpoint_relax = _build_endpoint_relax(
        raw.get("endpoint_relax", {}),
        source=source,
    )
    neb = _build_neb(raw.get("neb", {}), source=source)

    return ReactionConfig(
        description=description,
        reactant_structure=reactant_structure,
        product_structure=product_structure,
        endpoint_relax=endpoint_relax,
        neb=neb,
    )


def _build_endpoint_relax(raw: dict, *, source: str) -> EndpointRelaxSection:
    _check_keys(raw, _ENDPOINT_KEYS, set(), scope="[endpoint_relax]", source=source)
    fmax = _positive_float(
        raw.get("fmax", 0.01),
        key="[endpoint_relax].fmax",
        source=source,
    )
    max_steps = _positive_int(
        raw.get("max_steps", 500),
        key="[endpoint_relax].max_steps",
        source=source,
    )
    optimizer = raw.get("optimizer", "FIRE")
    if optimizer not in _VALID_OPTIMIZERS:
        raise ConfigError(
            f"{source}: '[endpoint_relax].optimizer' must be one of "
            f"{sorted(_VALID_OPTIMIZERS)}, got {optimizer!r}"
        )
    return EndpointRelaxSection(
        fmax=fmax,
        max_steps=max_steps,
        optimizer=optimizer,
    )


def _build_neb(raw: dict, *, source: str) -> NEBSection:
    _check_keys(raw, _NEB_KEYS, set(), scope="[neb]", source=source)
    n_images = _positive_int(raw.get("n_images", 11), key="[neb].n_images", source=source)
    if n_images < 3:
        raise ConfigError(f"{source}: '[neb].n_images' must be >= 3")
    fmax = _positive_float(raw.get("fmax", 0.05), key="[neb].fmax", source=source)
    max_steps = _positive_int(
        raw.get("max_steps", 200),
        key="[neb].max_steps",
        source=source,
    )
    k = _positive_float(raw.get("k", 1.0), key="[neb].k", source=source)
    method = raw.get("method", "eb")
    if method not in _VALID_NEB_METHODS:
        raise ConfigError(
            f"{source}: '[neb].method' must be one of "
            f"{sorted(_VALID_NEB_METHODS)}, got {method!r}"
        )
    remove_rotation_and_translation = raw.get("remove_rotation_and_translation", True)
    if not isinstance(remove_rotation_and_translation, bool):
        raise ConfigError(
            f"{source}: '[neb].remove_rotation_and_translation' must be boolean"
        )
    climb = raw.get("climb", True)
    if not isinstance(climb, bool):
        raise ConfigError(f"{source}: '[neb].climb' must be boolean")
    pad_frames = _non_negative_int(
        raw.get("pad_frames", 0),
        key="[neb].pad_frames",
        source=source,
    )
    guide_bond_changes = raw.get("guide_bond_changes", True)
    if not isinstance(guide_bond_changes, bool):
        raise ConfigError(f"{source}: '[neb].guide_bond_changes' must be boolean")
    # Bond-distance guide strength. This intentionally biases optimization when
    # enabled; unbiased energies are written separately for interpretation.
    guide_k = _positive_float(raw.get("guide_k", 5.0), key="[neb].guide_k", source=source)
    return NEBSection(
        n_images=n_images,
        fmax=fmax,
        max_steps=max_steps,
        k=k,
        method=method,
        remove_rotation_and_translation=remove_rotation_and_translation,
        climb=climb,
        pad_frames=pad_frames,
        guide_bond_changes=guide_bond_changes,
        guide_k=guide_k,
    )


def _check_keys(
    raw: dict,
    allowed: set[str],
    required: set[str],
    *,
    scope: str,
    source: str,
) -> None:
    if not isinstance(raw, dict):
        raise ConfigError(f"{source}: {scope} must be a table")
    missing = required - raw.keys()
    if missing:
        raise ConfigError(f"{source}: {scope} missing required keys: {sorted(missing)}")
    extra = set(raw.keys()) - allowed
    if extra:
        raise ConfigError(f"{source}: {scope} contains unknown keys: {sorted(extra)}")


def _positive_float(value, *, key: str, source: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{source}: '{key}' must be a positive number")
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ConfigError(f"{source}: '{key}' must be a finite positive number")
    return float(value)


def _positive_int(value, *, key: str, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{source}: '{key}' must be a positive integer")
    return int(value)


def _endpoint_path(value, *, key: str, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{source}: '{key}' must be a non-empty string")
    return value


def _non_negative_int(value, *, key: str, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"{source}: '{key}' must be a non-negative integer")
    return int(value)
