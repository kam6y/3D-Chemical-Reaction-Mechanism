"""Per-reaction sidecar TOML config (`<rxn_path>.toml`).

Phase 9 schema: [afir] + [scoring] sections.
Spec: docs/superpowers/specs/2026-05-08-afir-force-design.md §4.4
"""
from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


class ConfigError(ValueError):
    """Raised when `<rxn_path>.toml` violates the Phase 9 schema."""


@dataclass(frozen=True)
class SamplingConfig:
    n_candidates: int = 64


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
class ReactionConfig:
    description: str
    formed: tuple[tuple[int, int], ...]
    broken: tuple[tuple[int, int], ...]
    afir: AFIRSection
    scoring: ScoringSection
    sampling: SamplingConfig = field(default_factory=SamplingConfig)


_TOP_LEVEL_KEYS = {"description", "formed", "broken", "afir", "scoring", "sampling"}
_TOP_LEVEL_REQUIRED = {"description", "formed", "broken", "afir"}
_OBSOLETE_TOP_LEVEL = {"restraints", "prescreen", "k_form", "k_broken",
                       "r_broken", "r_form"}
_AFIR_KEYS = {"alpha_formed", "alpha_broken", "max_relax_steps"}
_AFIR_REQUIRED = {"max_relax_steps"}
_SCORING_KEYS = {"r_broken_threshold", "r_formed_threshold"}
_SAMPLING_KEYS = {"n_candidates"}


def sidecar_path(rxn_path: Path) -> Path:
    """`examples/sn2.rxn` -> `examples/sn2.rxn.toml`."""
    return rxn_path.parent / (rxn_path.name + ".toml")


def load_config(rxn_path: Path) -> ReactionConfig:
    """Load `<rxn_path>.toml` and return a validated ReactionConfig.

    Raises:
        FileNotFoundError: when the sidecar TOML is missing.
        ConfigError: on schema violations.
    """
    toml_path = sidecar_path(rxn_path)
    if not toml_path.is_file():
        raise FileNotFoundError(
            f"sidecar TOML not found: expected {toml_path} alongside {rxn_path}"
        )
    raw = tomllib.loads(toml_path.read_text(encoding="utf-8"))
    return _validate(raw, source=str(toml_path))


def _validate(raw: dict, *, source: str) -> ReactionConfig:
    for k in _OBSOLETE_TOP_LEVEL:
        if k in raw:
            raise ConfigError(
                f"{source}: '[{k}]' or '{k}' is removed in Phase 9; "
                f"migrate to '[afir]' / '[scoring]' (see spec §4.4)"
            )
    _check_keys(raw, _TOP_LEVEL_KEYS, _TOP_LEVEL_REQUIRED,
                scope="<top>", source=source)

    description = raw["description"]
    if not isinstance(description, str) or not description.strip():
        raise ConfigError(f"{source}: 'description' must be a non-empty string")

    formed = _to_pair_tuple(raw["formed"], key="formed", source=source)
    broken = _to_pair_tuple(raw["broken"], key="broken", source=source)
    if not formed and not broken:
        raise ConfigError(
            f"{source}: at least one of 'formed' or 'broken' must be non-empty"
        )

    afir = _build_afir(
        raw["afir"], formed_count=len(formed), broken_count=len(broken),
        source=source,
    )
    scoring = _build_scoring(
        raw.get("scoring", {}), formed_count=len(formed), broken_count=len(broken),
        source=source,
    )
    sampling = _build_sampling(raw.get("sampling", {}), source=source)

    return ReactionConfig(
        description=description, formed=formed, broken=broken,
        afir=afir, scoring=scoring, sampling=sampling,
    )


def _build_afir(raw: dict, *, formed_count: int, broken_count: int,
                source: str) -> AFIRSection:
    _check_keys(raw, _AFIR_KEYS, _AFIR_REQUIRED, scope="[afir]", source=source)
    alpha_formed = _normalize_alpha(
        raw.get("alpha_formed", 0.0), expected_count=formed_count,
        key="alpha_formed", source=source,
    )
    alpha_broken = _normalize_alpha(
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


def _normalize_alpha(value, *, expected_count: int, key: str,
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


def _build_scoring(raw: dict, *, formed_count: int, broken_count: int,
                   source: str) -> ScoringSection:
    _check_keys(raw, _SCORING_KEYS, set(), scope="[scoring]", source=source)
    r_broken = _normalize_threshold(
        raw.get("r_broken_threshold"), expected_count=broken_count,
        key="r_broken_threshold", source=source,
        required_when_pairs_present=True,
    )
    r_formed = _normalize_threshold(
        raw.get("r_formed_threshold"), expected_count=formed_count,
        key="r_formed_threshold", source=source,
        required_when_pairs_present=False,
    )
    return ScoringSection(
        r_broken_threshold=r_broken, r_formed_threshold=r_formed,
    )


def _normalize_threshold(value, *, expected_count: int, key: str, source: str,
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


def _check_keys(raw: dict, allowed: set[str], required: set[str], *,
                scope: str, source: str) -> None:
    if not isinstance(raw, dict):
        raise ConfigError(
            f"{source}: '{scope}' must be a table, got {type(raw).__name__}"
        )
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


def _to_pair_tuple(value, *, key: str, source: str) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list):
        raise ConfigError(f"{source}: '{key}' must be a list of [i, j] pairs")
    out: list[tuple[int, int]] = []
    for pair in value:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or not all(isinstance(x, int) and not isinstance(x, bool) for x in pair)
        ):
            raise ConfigError(
                f"{source}: '{key}' entries must be [int, int] pairs (got {pair!r})"
            )
        out.append((int(pair[0]), int(pair[1])))
    return tuple(out)


def _build_sampling(raw: dict, *, source: str) -> SamplingConfig:
    _check_keys(raw, _SAMPLING_KEYS, set(), scope="sampling", source=source)
    n_candidates = raw.get("n_candidates", 64)
    if not isinstance(n_candidates, int) or isinstance(n_candidates, bool) or n_candidates <= 0:
        raise ConfigError(
            f"{source}: 'sampling.n_candidates' must be a positive integer "
            f"(got {n_candidates!r})"
        )
    return SamplingConfig(n_candidates=int(n_candidates))


def resolve_alpha_formed(cfg: ReactionConfig) -> list[float]:
    return _broadcast(cfg.afir.alpha_formed, len(cfg.formed))


def resolve_alpha_broken(cfg: ReactionConfig) -> list[float]:
    return _broadcast(cfg.afir.alpha_broken, len(cfg.broken))


def _broadcast(value, n: int) -> list[float]:
    if n == 0:
        return []
    if isinstance(value, tuple):
        return [float(v) for v in value]
    return [float(value)] * n
