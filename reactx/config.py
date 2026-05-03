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
    k_form: float
    k_broken: float
    r_broken: float
    max_relax_steps: int
    r_form: float | tuple[float, ...] | None = None


@dataclass(frozen=True)
class SamplingConfig:
    n_angles: int = 8
    cone_half_deg: float = 30.0


@dataclass(frozen=True)
class PrescreenConfig:
    enabled: bool = True
    keep: int = 3
    steps: int = 30


@dataclass(frozen=True)
class ReactionConfig:
    description: str
    formed: tuple[tuple[int, int], ...]
    broken: tuple[tuple[int, int], ...]
    restraints: RestraintConfig
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    prescreen: PrescreenConfig = field(default_factory=PrescreenConfig)


_TOP_LEVEL_KEYS = {"description", "formed", "broken", "restraints", "sampling", "prescreen"}
_RESTRAINTS_KEYS = {"k_form", "k_broken", "r_broken", "max_relax_steps", "r_form"}
_RESTRAINTS_REQUIRED = {"k_form", "k_broken", "r_broken", "max_relax_steps"}
_SAMPLING_KEYS = {"n_angles", "cone_half_deg"}
_PRESCREEN_KEYS = {"enabled", "keep", "steps"}
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
    _check_keys(raw, _TOP_LEVEL_KEYS, _TOP_LEVEL_REQUIRED, scope="<top>", source=source)

    description = raw["description"]
    if not isinstance(description, str) or not description:
        raise ValueError(f"{source}: 'description' must be a non-empty string")

    formed = _to_pair_tuple(raw["formed"], key="formed", source=source)
    broken = _to_pair_tuple(raw["broken"], key="broken", source=source)
    if not formed and not broken:
        raise ValueError(
            f"{source}: at least one of 'formed' or 'broken' must be non-empty"
        )

    restraints = _build_restraints(raw["restraints"], formed_count=len(formed), source=source)
    sampling = _build_sampling(raw.get("sampling", {}), source=source)
    prescreen = _build_prescreen(raw.get("prescreen", {}), source=source)

    return ReactionConfig(
        description=description,
        formed=formed,
        broken=broken,
        restraints=restraints,
        sampling=sampling,
        prescreen=prescreen,
    )


def _check_keys(
    raw: dict, allowed: set[str], required: set[str], *, scope: str, source: str,
) -> None:
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


def _build_restraints(raw: dict, *, formed_count: int, source: str) -> RestraintConfig:
    _check_keys(raw, _RESTRAINTS_KEYS, _RESTRAINTS_REQUIRED,
                scope="restraints", source=source)
    k_form = _as_float(raw["k_form"], "restraints.k_form", source, non_negative=True)
    k_broken = _as_float(raw["k_broken"], "restraints.k_broken", source, non_negative=True)
    r_broken = _as_float(raw["r_broken"], "restraints.r_broken", source, positive=True)
    max_steps = _as_int(raw["max_relax_steps"], "restraints.max_relax_steps", source, positive=True)
    r_form_raw = raw.get("r_form", None)
    r_form = _normalize_r_form(r_form_raw, formed_count=formed_count, source=source)
    return RestraintConfig(
        k_form=k_form, k_broken=k_broken, r_broken=r_broken,
        max_relax_steps=max_steps, r_form=r_form,
    )


def _normalize_r_form(
    raw: object, *, formed_count: int, source: str,
) -> float | tuple[float, ...] | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        v = float(raw)
        if v <= 0.0:
            raise ValueError(f"{source}: 'restraints.r_form' must be positive")
        return v
    if isinstance(raw, list):
        if len(raw) != formed_count:
            raise ValueError(
                f"{source}: 'restraints.r_form' list length {len(raw)} "
                f"must match len(formed)={formed_count}"
            )
        out: list[float] = []
        for i, x in enumerate(raw):
            if not isinstance(x, (int, float)) or isinstance(x, bool):
                raise ValueError(
                    f"{source}: 'restraints.r_form[{i}]' must be a number"
                )
            v = float(x)
            if v <= 0.0:
                raise ValueError(
                    f"{source}: 'restraints.r_form[{i}]' must be positive"
                )
            out.append(v)
        return tuple(out)
    raise ValueError(f"{source}: 'restraints.r_form' must be number, list, or absent")


def _build_sampling(raw: dict, *, source: str) -> SamplingConfig:
    _check_keys(raw, _SAMPLING_KEYS, set(), scope="sampling", source=source)
    n_angles = _as_int(raw.get("n_angles", 8), "sampling.n_angles", source, positive=True)
    cone = _as_float(raw.get("cone_half_deg", 30.0), "sampling.cone_half_deg",
                     source, positive=True)
    return SamplingConfig(n_angles=n_angles, cone_half_deg=cone)


def _build_prescreen(raw: dict, *, source: str) -> PrescreenConfig:
    _check_keys(raw, _PRESCREEN_KEYS, set(), scope="prescreen", source=source)
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"{source}: 'prescreen.enabled' must be bool")
    keep = _as_int(raw.get("keep", 3), "prescreen.keep", source, positive=True)
    steps = _as_int(raw.get("steps", 30), "prescreen.steps", source, positive=True)
    return PrescreenConfig(enabled=enabled, keep=keep, steps=steps)


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
