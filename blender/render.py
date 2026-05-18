"""Blender headless script: trajectory.xyz -> ball-and-stick animated .blend.

Invoke:
    blender --background --python blender/render.py -- <trajectory.xyz> <out.blend>
"""
import math
import os
import sys
from pathlib import Path

import bpy  # type: ignore[import-not-found]
import mathutils  # type: ignore[import-not-found]
import numpy as np

# Alvarez (2013) vdW radii come from reactx.vdw_radii.
# Blender's bundled Python may not have the project root on sys.path, so we
# vendor a fallback copy that must stay byte-identical to reactx/vdw_radii.py.
try:
    from reactx.vdw_radii import VDW_RADII_ANGSTROM  # noqa: F401
except ImportError:
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

# Cordero (2008) Dalton Trans. 2832. Canonical table is `reactx/covalent_radii.py`.
# Falls back to a vendored copy when this script runs inside Blender's bundled
# Python without reactx on sys.path.
try:
    from reactx.covalent_radii import CORDERO_2008 as COVALENT_RADII_ANGSTROM  # noqa: F401
except ImportError:
    COVALENT_RADII_ANGSTROM: dict[str, float] = {
        "H": 0.31, "He": 0.28,
        "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66,
        "F": 0.57, "Ne": 0.58,
        "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05,
        "Cl": 1.02, "Ar": 1.06,
        "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39,
        "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
        "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16,
        "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54,
        "Tc": 1.47, "Ru": 1.46, "Rh": 1.42, "Pd": 1.39, "Ag": 1.45, "Cd": 1.44,
        "In": 1.42, "Sn": 1.39, "Sb": 1.39, "Te": 1.38, "I": 1.39, "Xe": 1.40,
        "Cs": 2.44, "Ba": 2.15,
        "La": 2.07, "Ce": 2.04, "Pr": 2.03, "Nd": 2.01, "Pm": 1.99, "Sm": 1.98,
        "Eu": 1.98, "Gd": 1.96, "Tb": 1.94, "Dy": 1.92, "Ho": 1.92, "Er": 1.89,
        "Tm": 1.90, "Yb": 1.87, "Lu": 1.87,
        "Hf": 1.75, "Ta": 1.70, "W": 1.62, "Re": 1.51, "Os": 1.44, "Ir": 1.41,
        "Pt": 1.36, "Au": 1.36, "Hg": 1.32, "Tl": 1.45, "Pb": 1.46, "Bi": 1.48,
    }

# A pair (i, j) is bonded if dist <= (rcov_i + rcov_j) * BOND_TOLERANCE.
BOND_TOLERANCE = 1.1
# Skin modifier cross-section radius for bond cylinders (A).
BOND_RADIUS = 0.10


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


def _parse_xyz_trajectory(xyz: Path) -> tuple[list[str], np.ndarray]:
    """Parse an extxyz file into (symbols, positions) where positions has shape (F, N, 3).

    All frames must share the same atom count and order — true for trajectory.xyz.
    """
    lines = xyz.read_text().splitlines()
    symbols: list[str] = []
    frames_pos: list[list[tuple[float, float, float]]] = []
    i = 0
    while i < len(lines):
        head = lines[i].strip()
        if not head or not head.lstrip("-").isdigit():
            i += 1
            continue
        n = int(head)
        atoms_pos: list[tuple[float, float, float]] = []
        atoms_sym: list[str] = []
        for j in range(2, 2 + n):
            if i + j >= len(lines):
                break
            parts = lines[i + j].split()
            if len(parts) < 4:
                continue
            atoms_sym.append(parts[0])
            atoms_pos.append((float(parts[1]), float(parts[2]), float(parts[3])))
        if len(atoms_pos) == n:
            if not symbols:
                symbols = atoms_sym
            frames_pos.append(atoms_pos)
        i += n + 2
    if not frames_pos:
        return [], np.empty((0, 0, 3))
    return symbols, np.asarray(frames_pos, dtype=np.float64)


def _center_positions(positions: np.ndarray) -> np.ndarray:
    """Subtract each frame's geometric mean to match the addon's put_to_center_all=True default."""
    if positions.size == 0:
        return positions
    return positions - positions.mean(axis=1, keepdims=True)


def _compute_bond_mask(
    symbols: list[str], positions: np.ndarray
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Detect bonds across all frames in a single NumPy pass.

    Returns (bond_pairs, bonded_mask) where bond_pairs is the union over frames
    of (i, j) pairs that are bonded in at least one frame, and bonded_mask has
    shape (F, len(bond_pairs)) marking per-frame visibility.
    """
    if positions.size == 0:
        return [], np.empty((0, 0), dtype=bool)
    n_frames, n_atoms, _ = positions.shape

    rcov = np.array(
        [COVALENT_RADII_ANGSTROM.get(s, math.nan) for s in symbols], dtype=np.float64
    )
    threshold_sq = ((rcov[:, None] + rcov[None, :]) * BOND_TOLERANCE) ** 2
    # NaNs (unknown elements) propagate to NaN thresholds — comparisons return False.

    deltas = positions[:, :, None, :] - positions[:, None, :, :]
    dist_sq = np.einsum("fija,fija->fij", deltas, deltas)

    iu, ju = np.triu_indices(n_atoms, k=1)
    pair_dist_sq = dist_sq[:, iu, ju]
    pair_threshold_sq = threshold_sq[iu, ju]
    pair_mask = pair_dist_sq <= pair_threshold_sq  # (F, n_pairs)

    union = pair_mask.any(axis=0)
    bond_pairs = list(zip(iu[union].tolist(), ju[union].tolist(), strict=True))
    bonded_mask = pair_mask[:, union]
    return bond_pairs, bonded_mask


def _make_bond_material():
    mat = bpy.data.materials.new(name="Bond")
    mat.use_nodes = True
    for n in mat.node_tree.nodes:
        if n.type == "BSDF_PRINCIPLED":
            n.inputs["Base Color"].default_value = (0.4, 0.4, 0.4, 1.0)
            n.inputs["Roughness"].default_value = 0.5
            break
    return mat


def _build_bonds(
    positions: np.ndarray,
    bond_pairs: list[tuple[int, int]],
    bonded_mask: np.ndarray,
) -> None:
    """Create one cylinder per candidate bond whose transform tracks the two
    atoms across frames, and whose hide_* visibility toggles per frame so bonds
    form/break dynamically."""
    if positions.size == 0:
        print("[reactx] bonds: no frames parsed")
        return
    n_frames = positions.shape[0]
    if not bond_pairs:
        print("[reactx] bonds: no bonds detected")
        return
    print(f"[reactx] bonds: {len(bond_pairs)} bond(s) across {n_frames} frame(s) "
          f"(per-frame visibility)")

    bond_mat = _make_bond_material()

    for bond_idx, (i, j) in enumerate(bond_pairs):
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=32, radius=BOND_RADIUS, depth=2.0,
            enter_editmode=False, location=(0, 0, 0),
        )
        cyl = bpy.context.object
        cyl.name = f"Bond_{bond_idx}_{i}_{j}"
        cyl.data.materials.append(bond_mat)
        # Smooth shading on the side faces gives a rounded silhouette without
        # adding geometry. Top/bottom caps (normal aligned with Z) stay flat
        # so the cap-side seam keeps a clean edge.
        for poly in cyl.data.polygons:
            poly.use_smooth = abs(poly.normal.z) < 0.5

        p1 = positions[:, i, :]
        p2 = positions[:, j, :]
        mids = 0.5 * (p1 + p2)
        vecs = p2 - p1
        lengths = np.linalg.norm(vecs, axis=1)

        for k in range(n_frames):
            cyl.location = mids[k].tolist()
            if lengths[k] > 1e-6:
                cyl.rotation_euler = mathutils.Vector(vecs[k].tolist()).to_track_quat(
                    "Z", "Y"
                ).to_euler()
                cyl.scale = (1.0, 1.0, lengths[k] / 2.0)
            else:
                cyl.scale = (1.0, 1.0, 1e-6)
            bonded = bool(bonded_mask[k, bond_idx])
            cyl.hide_viewport = not bonded
            cyl.hide_render = not bonded
            cyl.keyframe_insert("location", frame=k)
            cyl.keyframe_insert("rotation_euler", frame=k)
            cyl.keyframe_insert("scale", frame=k)
            cyl.keyframe_insert("hide_viewport", frame=k)
            cyl.keyframe_insert("hide_render", frame=k)

        # CONSTANT interpolation makes visibility flip instantly rather than ease.
        if cyl.animation_data and cyl.animation_data.action:
            for fc in cyl.animation_data.action.fcurves:
                if fc.data_path in ("hide_viewport", "hide_render"):
                    for kp in fc.keyframe_points:
                        kp.interpolation = "CONSTANT"


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


def _set_timeline_to_trajectory(n_frames: int) -> None:
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = max(1, n_frames)


def main(argv: list[str]) -> int:
    xyz, out = _parse_args(argv)
    _reset_scene()
    _import_trajectory(xyz)
    _rescale_atoms_to_vdw()
    symbols, positions = _parse_xyz_trajectory(xyz)
    positions = _center_positions(positions)
    bond_pairs, bonded_mask = _compute_bond_mask(symbols, positions)
    _build_bonds(positions, bond_pairs, bonded_mask)
    _add_three_point_lighting()
    _add_camera_looking_at_origin()
    _set_timeline_to_trajectory(positions.shape[0] if positions.size else 0)
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out))
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(list(sys.argv)))
