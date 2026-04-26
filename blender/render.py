"""Blender headless script: trajectory.xyz -> ball-and-stick animated .blend.

Invoke:
    blender --background --python blender/render.py -- <trajectory.xyz> <out.blend>
"""
import math
import sys
from pathlib import Path

import bpy  # type: ignore[import-not-found]

import os


# Alvarez (2013) "A cartography of the van der Waals territories"
# Dalton Trans. 42, 8617. Values in Angstrom for Z=1..83 (H..Bi),
# matching OMol25 / UMA omol task element coverage exactly.
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

# atomic-blender-pdb-xyz uses element full names as ball prefixes
# (e.g. "Hydrogen_ball"). Note "Aluminium"/"Caesium"/"Sulfur" follow the
# spelling in the add-on's ELEMENTS_DEFAULT.
_ELEMENT_NAME_TO_SYMBOL: dict[str, str] = {
    "Hydrogen": "H", "Helium": "He",
    "Lithium": "Li", "Beryllium": "Be", "Boron": "B", "Carbon": "C",
    "Nitrogen": "N", "Oxygen": "O", "Fluorine": "F", "Neon": "Ne",
    "Sodium": "Na", "Magnesium": "Mg", "Aluminium": "Al", "Silicon": "Si",
    "Phosphorus": "P", "Sulfur": "S", "Chlorine": "Cl", "Argon": "Ar",
    "Potassium": "K", "Calcium": "Ca", "Scandium": "Sc", "Titanium": "Ti",
    "Vanadium": "V", "Chromium": "Cr", "Manganese": "Mn", "Iron": "Fe",
    "Cobalt": "Co", "Nickel": "Ni", "Copper": "Cu", "Zinc": "Zn",
    "Gallium": "Ga", "Germanium": "Ge", "Arsenic": "As", "Selenium": "Se",
    "Bromine": "Br", "Krypton": "Kr",
    "Rubidium": "Rb", "Strontium": "Sr", "Yttrium": "Y", "Zirconium": "Zr",
    "Niobium": "Nb", "Molybdenum": "Mo", "Technetium": "Tc", "Ruthenium": "Ru",
    "Rhodium": "Rh", "Palladium": "Pd", "Silver": "Ag", "Cadmium": "Cd",
    "Indium": "In", "Tin": "Sn", "Antimony": "Sb", "Tellurium": "Te",
    "Iodine": "I", "Xenon": "Xe",
    "Caesium": "Cs", "Barium": "Ba",
    "Lanthanum": "La", "Cerium": "Ce", "Praseodymium": "Pr", "Neodymium": "Nd",
    "Promethium": "Pm", "Samarium": "Sm", "Europium": "Eu", "Gadolinium": "Gd",
    "Terbium": "Tb", "Dysprosium": "Dy", "Holmium": "Ho", "Erbium": "Er",
    "Thulium": "Tm", "Ytterbium": "Yb", "Lutetium": "Lu",
    "Hafnium": "Hf", "Tantalum": "Ta", "Tungsten": "W", "Rhenium": "Re",
    "Osmium": "Os", "Iridium": "Ir", "Platinum": "Pt", "Gold": "Au",
    "Mercury": "Hg", "Thallium": "Tl", "Lead": "Pb", "Bismuth": "Bi",
}

DEFAULT_VDW_SCALE = 0.25


FPS = 24


def _parse_args(argv: list[str]) -> tuple[Path, Path]:
    if "--" not in argv:
        raise SystemExit("Usage: blender --background --python render.py -- <xyz> <out.blend>")
    extra = argv[argv.index("--") + 1:]
    if len(extra) < 2:
        raise SystemExit("Need <xyz> <out.blend>")
    return Path(extra[0]), Path(extra[1])


def _resolve_vdw_scale() -> float:
    raw = os.environ.get("REACTX_VDW_SCALE")
    if raw is None:
        return DEFAULT_VDW_SCALE
    try:
        return float(raw)
    except ValueError:
        print(f"[reactx] vdw-rescale: bad REACTX_VDW_SCALE={raw!r}, "
              f"using {DEFAULT_VDW_SCALE}")
        return DEFAULT_VDW_SCALE


def _rescale_atoms_to_vdw() -> None:
    scale = _resolve_vdw_scale()
    suffix = "_ball"
    for obj in bpy.data.objects:
        if not obj.name.endswith(suffix):
            continue
        element_name = obj.name[: -len(suffix)]
        symbol = _ELEMENT_NAME_TO_SYMBOL.get(element_name)
        radius = VDW_RADII_ANGSTROM.get(symbol) if symbol else None
        if radius is None:
            print(f"[reactx] vdw-rescale: skip {obj.name} (no entry)")
            continue
        new_scale = radius * scale
        obj.scale = (new_scale, new_scale, new_scale)
        print(f"[reactx] vdw-rescale: {element_name} scale={new_scale:.6f}")


def _reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = FPS


def _import_trajectory(xyz: Path) -> None:
    candidates = (
        "bl_ext.blender_org.atomic_blender_pdb_xyz",
        "atomic_blender_pdb_xyz",
        "atomic_blender_xyz",
    )
    last_err: Exception | None = None
    for name in candidates:
        try:
            bpy.ops.preferences.addon_enable(module=name)
            print(f"Enabled add-on: {name}")
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    else:
        raise RuntimeError(
            f"Could not enable atomic-blender-pdb-xyz add-on. Tried: {candidates}. "
            f"Last error: {last_err}. Install via Edit > Preferences > Get Extensions."
        )
    bpy.ops.import_mesh.xyz(filepath=str(xyz), use_frames=True)


def _add_three_point_lighting() -> None:
    def _light(name: str, energy: float, location: tuple[float, float, float]) -> None:
        bpy.ops.object.light_add(type="AREA", location=location)
        light = bpy.context.object
        light.name = name
        light.data.energy = energy
        light.data.size = 2.0

    _light("Key", 800, (5, -5, 5))
    _light("Fill", 300, (-5, -3, 3))
    _light("Rim", 500, (0, 5, 4))


def _add_camera_looking_at_origin() -> None:
    bpy.ops.object.camera_add(location=(0, -8, 2),
                              rotation=(math.radians(80), 0, 0))
    bpy.context.scene.camera = bpy.context.object


def _count_xyz_frames(xyz: Path) -> int:
    """Number of frames in an extxyz trajectory (each frame = n_atoms + 2 lines)."""
    lines = xyz.read_text().splitlines()
    if not lines:
        return 0
    n_atoms = int(lines[0].strip())
    return len(lines) // (n_atoms + 2)


def _set_timeline_to_trajectory(xyz: Path) -> None:
    n_frames = _count_xyz_frames(xyz)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = max(1, n_frames)


def main(argv: list[str]) -> int:
    xyz, out = _parse_args(argv)
    _reset_scene()
    _import_trajectory(xyz)
    _rescale_atoms_to_vdw()
    _add_three_point_lighting()
    _add_camera_looking_at_origin()
    _set_timeline_to_trajectory(xyz)
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out))
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(list(sys.argv)))
