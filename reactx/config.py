"""Per-reaction sidecar TOML config (`<rxn_path>.toml`).

Loads and validates the schema described in
`docs/superpowers/specs/2026-05-03-rxn-config-sidecar-design.md` §3.2.

Tier A (this module): schema + range validation, runs without parsing the
.rxn file. Tier B (atom-map presence in the .rxn) lives in
reactx.bond_changes.from_atom_map_pairs.
"""
from __future__ import annotations

import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from reactx.artificial_force import lookup_r_form


@dataclass(frozen=True)
class RestraintConfig:
    k_form: float | tuple[float, ...]
    k_broken: float | tuple[float, ...]
    r_broken: float | tuple[float, ...]
    max_relax_steps: int
    r_form: float | tuple[float, ...] | None = None


@dataclass(frozen=True)
class SamplingConfig:
    n_candidates: int = 64


@dataclass(frozen=True)
class ReactionConfig:
    description: str
    formed: tuple[tuple[int, int], ...]
    broken: tuple[tuple[int, int], ...]
    restraints: RestraintConfig
    sampling: SamplingConfig = field(default_factory=SamplingConfig)


_TOP_LEVEL_KEYS = {"description", "formed", "broken", "restraints", "sampling"}
_RESTRAINTS_KEYS = {"k_form", "k_broken", "r_broken", "max_relax_steps", "r_form"}
_RESTRAINTS_REQUIRED = {"k_form", "k_broken", "r_broken", "max_relax_steps"}
_SAMPLING_KEYS = {"n_candidates"}
_TOP_LEVEL_REQUIRED = {"description", "formed", "broken", "restraints"}


def sidecar_path(rxn_path: Path) -> Path:
    """`examples/sn2.rxn` -> `examples/sn2.rxn.toml`."""
    return rxn_path.parent / (rxn_path.name + ".toml")


def load_config(rxn_path: Path) -> ReactionConfig:
    """Load `<rxn_path>.toml` and return a validated ReactionConfig.

    Raises:
        FileNotFoundError: when the sidecar TOML is missing.
        ValueError: on schema violations (Tier A).
    """
    toml_path = sidecar_path(rxn_path)
    if not toml_path.is_file():
        raise FileNotFoundError(
            f"sidecar TOML not found: expected {toml_path} alongside {rxn_path}"
        )
    raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    return _validate(raw, source=str(toml_path))


def _validate(raw: dict, *, source: str) -> ReactionConfig:
    if "prescreen" in raw:
        raise ValueError(
            f"{source}: [prescreen] section is removed in Phase 7; "
            f"delete it from the TOML"
        )
    _check_keys(raw, _TOP_LEVEL_KEYS, _TOP_LEVEL_REQUIRED, scope="<top>", source=source)

    description = raw["description"]
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"{source}: 'description' must be a non-empty string")

    formed = _to_pair_tuple(raw["formed"], key="formed", source=source)
    broken = _to_pair_tuple(raw["broken"], key="broken", source=source)
    if not formed and not broken:
        raise ValueError(
            f"{source}: at least one of 'formed' or 'broken' must be non-empty"
        )

    restraints = _build_restraints(
        raw["restraints"],
        formed_count=len(formed),
        broken_count=len(broken),
        source=source,
    )
    sampling = _build_sampling(raw.get("sampling", {}), source=source)

    return ReactionConfig(
        description=description,
        formed=formed,
        broken=broken,
        restraints=restraints,
        sampling=sampling,
    )


def _check_keys(
    raw: dict, allowed: set[str], required: set[str], *, scope: str, source: str,
) -> None:
    if not isinstance(raw, dict):
        raise ValueError(
            f"{source}: '{scope}' must be a table, got {type(raw).__name__}"
        )
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(
            f"{source}: unknown config key(s) under {scope}: "
            f"{sorted(unknown)}; allowed: {sorted(allowed)}"
        )
    missing = required - set(raw)
    if missing:
        raise ValueError(
            f"{source}: missing required key(s) under {scope}: {sorted(missing)}"
        )


def _to_pair_tuple(
    raw: object, *, key: str, source: str,
) -> tuple[tuple[int, int], ...]:
    if not isinstance(raw, list):
        raise ValueError(f"{source}: '{key}' must be a list of [int,int] pairs")
    pairs: list[tuple[int, int]] = []
    for i, p in enumerate(raw):
        if (
            not isinstance(p, list)
            or len(p) != 2
            or not all(isinstance(x, int) and not isinstance(x, bool) for x in p)
        ):
            raise ValueError(
                f"{source}: '{key}[{i}]' must be a [int,int] pair, got {p!r}"
            )
        pairs.append((int(p[0]), int(p[1])))
    return tuple(pairs)


def _build_restraints(raw: dict, *, formed_count: int, broken_count: int, source: str) -> RestraintConfig:
    _check_keys(raw, _RESTRAINTS_KEYS, _RESTRAINTS_REQUIRED,
                scope="restraints", source=source)
    k_form = _normalize_scalar_or_list(
        raw["k_form"], n_bonds=formed_count,
        key="restraints.k_form", bonds_label="formed",
        source=source, strict=False,
    )
    k_broken = _normalize_scalar_or_list(
        raw["k_broken"], n_bonds=broken_count,
        key="restraints.k_broken", bonds_label="broken",
        source=source, strict=False,
    )
    r_broken = _normalize_scalar_or_list(
        raw["r_broken"], n_bonds=broken_count,
        key="restraints.r_broken", bonds_label="broken",
        source=source, strict=True,
    )
    max_steps = _as_int(raw["max_relax_steps"], "restraints.max_relax_steps", source, positive=True)
    r_form = _normalize_scalar_or_list(
        raw.get("r_form"), n_bonds=formed_count,
        key="restraints.r_form", bonds_label="formed",
        source=source, strict=True, allow_none=True,
    )
    return RestraintConfig(
        k_form=k_form, k_broken=k_broken, r_broken=r_broken,
        max_relax_steps=max_steps, r_form=r_form,
    )


def _normalize_scalar_or_list(
    raw: object,
    *,
    n_bonds: int,
    key: str,
    bonds_label: str,
    source: str,
    strict: bool,
    allow_none: bool = False,
) -> float | tuple[float, ...] | None:
    """Validate a TOML restraint field that accepts either a scalar or a per-bond list.

    `strict=True`  -> require value > 0 (for r_form / r_broken).
    `strict=False` -> require value >= 0 (for k_form / k_broken).
    `allow_none=True` -> accept None (for r_form fallback to element table).
    """
    bound_msg = "must be > 0" if strict else "must be >= 0"

    def _violates(v: float) -> bool:
        return v <= 0.0 if strict else v < 0.0

    if raw is None:
        if allow_none:
            return None
        raise ValueError(f"{source}: '{key}' must be number or list")
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        v = float(raw)
        if _violates(v):
            raise ValueError(f"{source}: '{key}' {bound_msg}")
        return v
    if isinstance(raw, list):
        if len(raw) != n_bonds:
            raise ValueError(
                f"{source}: '{key}' list length {len(raw)} "
                f"must match len({bonds_label})={n_bonds}"
            )
        out: list[float] = []
        for i, x in enumerate(raw):
            if not isinstance(x, (int, float)) or isinstance(x, bool):
                raise ValueError(f"{source}: '{key}[{i}]' must be a number")
            v = float(x)
            if _violates(v):
                raise ValueError(f"{source}: '{key}[{i}]' {bound_msg}")
            out.append(v)
        return tuple(out)
    raise ValueError(f"{source}: '{key}' must be number or list")


def _build_sampling(raw: dict, *, source: str) -> SamplingConfig:
    _check_keys(raw, _SAMPLING_KEYS, set(), scope="sampling", source=source)
    n_candidates = _as_int(
        raw.get("n_candidates", 64), "sampling.n_candidates", source, positive=True,
    )
    return SamplingConfig(n_candidates=n_candidates)


def _as_float(
    raw: object, key: str, source: str, *,
    non_negative: bool = False, positive: bool = False,
) -> float:
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        raise ValueError(f"{source}: '{key}' must be a number")
    v = float(raw)
    if positive and v <= 0.0:
        raise ValueError(f"{source}: '{key}' must be > 0")
    if non_negative and v < 0.0:
        raise ValueError(f"{source}: '{key}' must be >= 0")
    return v


def _as_int(
    raw: object, key: str, source: str, *, positive: bool = False,
) -> int:
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise ValueError(f"{source}: '{key}' must be an int")
    if positive and raw <= 0:
        raise ValueError(f"{source}: '{key}' must be > 0")
    return int(raw)


def resolve_r_form_targets(
    cfg: ReactionConfig,
    syms: Sequence[str],
    formed_idx_pairs: Sequence[tuple[int, int]],
) -> list[float]:
    """Expand `cfg.restraints.r_form` into per-formed-bond target distances.

    The 0-based heavy-atom indices in `formed_idx_pairs` come from
    `BondChanges.from_atom_map_pairs(...)`, NOT from cfg.formed (which is
    atom-map numbers).
    """
    r = cfg.restraints.r_form
    if r is None:
        return [lookup_r_form(syms[a], syms[b]) for a, b in formed_idx_pairs]
    if isinstance(r, float):
        return [r] * len(formed_idx_pairs)
    return list(r)


def _resolve_per_bond(
    value: float | tuple[float, ...] | None,
    n_bonds: int,
) -> list[float]:
    """Broadcast scalar to length n_bonds; pass tuple through as list.

    Returns empty list when n_bonds == 0 regardless of value.
    """
    if n_bonds == 0:
        return []
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (int, float)):
        return [float(value)] * n_bonds
    raise TypeError(f"value must be float or tuple, got {type(value).__name__}")


def resolve_k_form_targets(
    cfg: ReactionConfig,
    formed_idx_pairs: Sequence[tuple[int, int]],
) -> list[float]:
    """Expand cfg.restraints.k_form into per-formed-bond k values."""
    return _resolve_per_bond(cfg.restraints.k_form, len(formed_idx_pairs))


def resolve_k_broken_targets(
    cfg: ReactionConfig,
    broken_idx_pairs: Sequence[tuple[int, int]],
) -> list[float]:
    """Expand cfg.restraints.k_broken into per-broken-bond k values."""
    return _resolve_per_bond(cfg.restraints.k_broken, len(broken_idx_pairs))


def resolve_r_broken_targets(
    cfg: ReactionConfig,
    broken_idx_pairs: Sequence[tuple[int, int]],
) -> list[float]:
    """Expand cfg.restraints.r_broken into per-broken-bond r target values."""
    return _resolve_per_bond(cfg.restraints.r_broken, len(broken_idx_pairs))


# ===== Phase 9 schema (additive, coexists with v8 [restraints]) =====

import math


class ConfigError(ValueError):
    """Raised when `<rxn_path>.toml` violates the Phase 9 schema."""


@dataclass(frozen=True)
class AFIRSection:
    alpha_formed: float | tuple[float, ...]
    alpha_broken: float | tuple[float, ...]
    max_relax_steps: int


@dataclass(frozen=True)
class ScoringSection:
    r_broken_threshold: float | tuple[float, ...] | None = None
    r_formed_threshold: float | tuple[float, ...] | None = None


@dataclass(frozen=True)
class ReactionConfigV9:
    description: str
    formed: tuple[tuple[int, int], ...]
    broken: tuple[tuple[int, int], ...]
    afir: AFIRSection
    scoring: ScoringSection
    sampling: SamplingConfig = field(default_factory=SamplingConfig)


_V9_TOP_LEVEL_KEYS = {"description", "formed", "broken", "afir", "scoring", "sampling"}
_V9_TOP_LEVEL_REQUIRED = {"description", "formed", "broken", "afir"}
_V9_OBSOLETE_TOP_LEVEL = {"restraints", "prescreen", "k_form", "k_broken",
                          "r_broken", "r_form"}
_V9_AFIR_KEYS = {"alpha_formed", "alpha_broken", "max_relax_steps"}
_V9_AFIR_REQUIRED = {"max_relax_steps"}
_V9_SCORING_KEYS = {"r_broken_threshold", "r_formed_threshold"}


def load_config_v9(rxn_path: Path) -> ReactionConfigV9:
    """Phase 9 loader. Coexists with v8 `load_config` until cli.py switches."""
    toml_path = sidecar_path(rxn_path)
    if not toml_path.is_file():
        raise FileNotFoundError(
            f"sidecar TOML not found: expected {toml_path} alongside {rxn_path}"
        )
    raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    return _validate_v9(raw, source=str(toml_path))


def _validate_v9(raw: dict, *, source: str) -> ReactionConfigV9:
    for k in _V9_OBSOLETE_TOP_LEVEL:
        if k in raw:
            raise ConfigError(
                f"{source}: '[{k}]' or '{k}' is removed in Phase 9; "
                f"migrate to '[afir]' / '[scoring]' (see spec §4.4)"
            )
    _v9_check_keys(raw, _V9_TOP_LEVEL_KEYS, _V9_TOP_LEVEL_REQUIRED,
                   scope="<top>", source=source)

    description = raw["description"]
    if not isinstance(description, str) or not description.strip():
        raise ConfigError(f"{source}: 'description' must be a non-empty string")

    formed = _v9_to_pair_tuple(raw["formed"], key="formed", source=source)
    broken = _v9_to_pair_tuple(raw["broken"], key="broken", source=source)
    if not formed and not broken:
        raise ConfigError(
            f"{source}: at least one of 'formed' or 'broken' must be non-empty"
        )

    afir = _v9_build_afir(
        raw["afir"], formed_count=len(formed), broken_count=len(broken),
        source=source,
    )
    scoring = _v9_build_scoring(
        raw.get("scoring", {}), formed_count=len(formed), broken_count=len(broken),
        source=source,
    )
    sampling = _build_sampling(raw.get("sampling", {}), source=source)

    return ReactionConfigV9(
        description=description, formed=formed, broken=broken,
        afir=afir, scoring=scoring, sampling=sampling,
    )


def _v9_build_afir(raw: dict, *, formed_count: int, broken_count: int,
                   source: str) -> AFIRSection:
    _v9_check_keys(raw, _V9_AFIR_KEYS, _V9_AFIR_REQUIRED, scope="[afir]", source=source)
    alpha_formed = _v9_normalize_alpha(
        raw.get("alpha_formed", 0.0), expected_count=formed_count,
        key="alpha_formed", source=source,
    )
    alpha_broken = _v9_normalize_alpha(
        raw.get("alpha_broken", 0.0), expected_count=broken_count,
        key="alpha_broken", source=source,
    )
    max_steps = raw["max_relax_steps"]
    if not isinstance(max_steps, int) or max_steps <= 0:
        raise ConfigError(
            f"{source}: '[afir].max_relax_steps' must be a positive integer "
            f"(got {max_steps!r})"
        )
    return AFIRSection(
        alpha_formed=alpha_formed, alpha_broken=alpha_broken,
        max_relax_steps=int(max_steps),
    )


def _v9_normalize_alpha(value, *, expected_count: int, key: str,
                        source: str) -> float | tuple[float, ...]:
    def _check_finite(v, ctx: str):
        if not isinstance(v, (int, float)):
            raise ConfigError(
                f"{source}: '[afir].{key}'{ctx} must be numeric (got {v!r})"
            )
        if not math.isfinite(v):
            raise ConfigError(
                f"{source}: '[afir].{key}'{ctx} must be finite (got {v!r})"
            )

    if isinstance(value, list):
        for idx, v in enumerate(value):
            _check_finite(v, f"[{idx}]")
    else:
        _check_finite(value, "")

    if expected_count == 0:
        return tuple()

    if isinstance(value, list):
        if len(value) != expected_count:
            raise ConfigError(
                f"{source}: '[afir].{key}' list length {len(value)} != "
                f"expected {expected_count}"
            )
        for v in value:
            if v <= 0:
                raise ConfigError(
                    f"{source}: '[afir].{key}' must be > 0 for non-empty "
                    f"pair set (spec §4.4 forbids α=0); got {v}"
                )
        return tuple(float(v) for v in value)

    if value <= 0:
        raise ConfigError(
            f"{source}: '[afir].{key}' must be > 0 for non-empty pair set "
            f"(spec §4.4 forbids α=0); got {value}"
        )
    return float(value)


def _v9_build_scoring(raw: dict, *, formed_count: int, broken_count: int,
                      source: str) -> ScoringSection:
    _v9_check_keys(raw, _V9_SCORING_KEYS, set(), scope="[scoring]", source=source)
    r_broken = _v9_normalize_threshold(
        raw.get("r_broken_threshold"), expected_count=broken_count,
        key="r_broken_threshold", source=source,
        required_when_pairs_present=True,
    )
    r_formed = _v9_normalize_threshold(
        raw.get("r_formed_threshold"), expected_count=formed_count,
        key="r_formed_threshold", source=source,
        required_when_pairs_present=False,
    )
    return ScoringSection(
        r_broken_threshold=r_broken, r_formed_threshold=r_formed,
    )


def _v9_normalize_threshold(value, *, expected_count: int, key: str, source: str,
                            required_when_pairs_present: bool):
    if expected_count == 0:
        return None
    if value is None:
        if required_when_pairs_present:
            raise ConfigError(
                f"{source}: '[scoring].{key}' is required when there are "
                f"non-empty pairs"
            )
        return None
    if isinstance(value, list):
        if len(value) != expected_count:
            raise ConfigError(
                f"{source}: '[scoring].{key}' list length {len(value)} != "
                f"expected {expected_count}"
            )
        for v in value:
            if not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                raise ConfigError(
                    f"{source}: '[scoring].{key}' must contain finite, "
                    f"positive values; got {v!r}"
                )
        return tuple(float(v) for v in value)
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ConfigError(
            f"{source}: '[scoring].{key}' must be a finite, positive "
            f"number (got {value!r})"
        )
    return float(value)


def _v9_check_keys(raw: dict, allowed: set[str], required: set[str], *,
                   scope: str, source: str) -> None:
    missing = required - raw.keys()
    if missing:
        raise ConfigError(
            f"{source}: {scope} missing required keys: {sorted(missing)}"
        )
    extra = set(raw.keys()) - allowed
    if extra:
        raise ConfigError(
            f"{source}: {scope} contains unknown keys: {sorted(extra)}"
        )


def _v9_to_pair_tuple(value, *, key: str, source: str) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list):
        raise ConfigError(f"{source}: '{key}' must be a list of [i, j] pairs")
    out: list[tuple[int, int]] = []
    for pair in value:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(isinstance(x, int) for x in pair)
        ):
            raise ConfigError(
                f"{source}: '{key}' entries must be [int, int] pairs (got {pair!r})"
            )
        out.append((int(pair[0]), int(pair[1])))
    return tuple(out)


def resolve_alpha_formed_v9(cfg: ReactionConfigV9) -> list[float]:
    return _v9_broadcast(cfg.afir.alpha_formed, len(cfg.formed))


def resolve_alpha_broken_v9(cfg: ReactionConfigV9) -> list[float]:
    return _v9_broadcast(cfg.afir.alpha_broken, len(cfg.broken))


def _v9_broadcast(value, n: int) -> list[float]:
    if n == 0:
        return []
    if isinstance(value, tuple):
        return [float(v) for v in value]
    return [float(value)] * n
