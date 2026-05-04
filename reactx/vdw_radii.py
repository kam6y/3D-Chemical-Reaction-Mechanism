"""Shared van der Waals radius table for placement and rendering.

Alvarez (2013) "A cartography of the van der Waals territories"
Dalton Trans. 42, 8617. Values in Angstrom for Z=1..83 (H..Bi),
matching OMol25 / UMA omol task element coverage exactly.

Z > 83 (Po, At, Rn, Fr, Ra, all actinides) は UMA omol25 訓練外なので
実用上発火しないが、安全のため Alvarez 中央値近傍 (1.50 Å) で fallback する。
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

VDW_RADII_ANGSTROM: dict[str, float] = {
    "H": 1.20, "He": 1.43,
    "Li": 2.12, "Be": 1.98, "B": 1.91, "C": 1.77, "N": 1.66, "O": 1.50,
    "F": 1.46, "Ne": 1.58,
    "Na": 2.50, "Mg": 2.51, "Al": 2.25, "Si": 2.19, "P": 1.90, "S": 1.89,
    "Cl": 1.82, "Ar": 1.83,
    "K": 2.73, "Ca": 2.62, "Sc": 2.58, "Ti": 2.46, "V": 2.42, "Cr": 2.45,
    "Mn": 2.45, "Fe": 2.44, "Co": 2.40, "Ni": 2.40, "Cu": 2.38, "Zn": 2.39,
    "Ga": 2.32, "Ge": 2.29, "As": 1.88, "Se": 1.82, "Br": 1.86, "Kr": 2.25,
    "Rb": 3.21, "Sr": 2.84, "Y": 2.75, "Zr": 2.52, "Nb": 2.56, "Mo": 2.45,
    "Tc": 2.44, "Ru": 2.46, "Rh": 2.44, "Pd": 2.15, "Ag": 2.53, "Cd": 2.49,
    "In": 2.43, "Sn": 2.42, "Sb": 2.47, "Te": 1.99, "I": 2.04, "Xe": 2.06,
    "Cs": 3.48, "Ba": 3.03,
    "La": 2.98, "Ce": 2.88, "Pr": 2.92, "Nd": 2.95, "Pm": 2.93, "Sm": 2.90,
    "Eu": 2.87, "Gd": 2.83, "Tb": 2.79, "Dy": 2.87, "Ho": 2.81, "Er": 2.83,
    "Tm": 2.79, "Yb": 2.80, "Lu": 2.74,
    "Hf": 2.63, "Ta": 2.53, "W": 2.57, "Re": 2.49, "Os": 2.48, "Ir": 2.41,
    "Pt": 2.29, "Au": 2.32, "Hg": 2.45, "Tl": 2.47, "Pb": 2.60, "Bi": 2.54,
}

FALLBACK_RADIUS: float = 1.50


def vdw_radius(symbol: str) -> float:
    """Alvarez 2013 vdW radius (Å) for a chemical element symbol.

    Unknown symbols (Z > 83, lanthanide/actinide gaps, garbled) → log warning
    and return FALLBACK_RADIUS (1.50 Å).
    """
    r = VDW_RADII_ANGSTROM.get(symbol)
    if r is None:
        log.warning(
            "vdw_radius: symbol %r not in Alvarez 2013 table (Z=1..83); "
            "using fallback %.2f Å", symbol, FALLBACK_RADIUS,
        )
        return FALLBACK_RADIUS
    return r
