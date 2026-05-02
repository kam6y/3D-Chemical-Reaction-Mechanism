"""Built-in reaction-type presets for the reactx CLI.

Each preset bundles a tested set of artificial-force restraint parameters
(k_form, k_broken, r_broken, max_relax_steps, optional r_form override) for
a class of reactions. Preset values applied via `--reaction-type` can still
be overridden by individual CLI flags.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReactionPreset:
    name: str
    k_form: float
    k_broken: float
    r_broken: float
    max_relax_steps: int
    r_form: float | None = None  # None = element-pair table lookup


PRESETS: dict[str, ReactionPreset] = {
    "sn2_anion": ReactionPreset(
        name="sn2_anion",
        k_form=0.5,
        k_broken=1.0,
        r_broken=4.0,
        max_relax_steps=100,
    ),
    "proton_transfer": ReactionPreset(
        name="proton_transfer",
        k_form=0.5,
        k_broken=1.0,
        r_broken=4.0,
        max_relax_steps=100,
        r_form=1.05,
    ),
    "menshutkin": ReactionPreset(
        name="menshutkin",
        k_form=2.0,
        k_broken=2.0,
        r_broken=5.0,
        max_relax_steps=200,
    ),
}


def get_preset(name: str) -> ReactionPreset:
    """Return the built-in preset by name. Raises ValueError if unknown."""
    if name not in PRESETS:
        raise ValueError(
            f"Unknown reaction-type preset {name!r}. "
            f"Available: {sorted(PRESETS)}"
        )
    return PRESETS[name]
