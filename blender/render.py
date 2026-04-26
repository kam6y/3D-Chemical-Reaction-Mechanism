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

# Cordero et al. (2008) Dalton Trans. 2832. Covalent radii in Angstrom for
# Z=1..83. Used purely for distance-based bond detection.
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
BOND_TOLERANCE = 1.3
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


_AtomFrame = list[tuple[str, tuple[float, float, float]]]


def _parse_xyz_trajectory(xyz: Path) -> list[_AtomFrame]:
    """Parse an extxyz file into a list of frames, each a list of (symbol, (x, y, z))."""
    lines = xyz.read_text().splitlines()
    frames: list[_AtomFrame] = []
    i = 0
    while i < len(lines):
        head = lines[i].strip()
        if not head:
            i += 1
            continue
        try:
            n = int(head)
        except ValueError:
            i += 1
            continue
        atoms: _AtomFrame = []
        for j in range(2, 2 + n):
            if i + j >= len(lines):
                break
            parts = lines[i + j].split()
            if len(parts) < 4:
                continue
            atoms.append((parts[0], (float(parts[1]), float(parts[2]), float(parts[3]))))
        if len(atoms) == n:
            frames.append(atoms)
        i += n + 2
    return frames


def _center_frames(frames: list[_AtomFrame]) -> list[_AtomFrame]:
    """Subtract each frame's geometric mean to mirror the addon's put_to_center_all=True default."""
    out: list[_AtomFrame] = []
    for frame in frames:
        n = len(frame)
        if n == 0:
            out.append(frame)
            continue
        cx = sum(p[0] for _, p in frame) / n
        cy = sum(p[1] for _, p in frame) / n
        cz = sum(p[2] for _, p in frame) / n
        out.append([(s, (p[0] - cx, p[1] - cy, p[2] - cz)) for s, p in frame])
    return out


def _compute_bond_pairs(frames: list[_AtomFrame]) -> list[tuple[int, int]]:
    """Bonded atom-index pairs (union over all frames) by covalent radius criterion."""
    if not frames:
        return []
    n = len(frames[0])
    bonds: set[tuple[int, int]] = set()
    for frame in frames:
        for i in range(n):
            sym_i, pos_i = frame[i]
            rcov_i = COVALENT_RADII_ANGSTROM.get(sym_i)
            if rcov_i is None:
                continue
            for j in range(i + 1, n):
                sym_j, pos_j = frame[j]
                rcov_j = COVALENT_RADII_ANGSTROM.get(sym_j)
                if rcov_j is None:
                    continue
                threshold = (rcov_i + rcov_j) * BOND_TOLERANCE
                dx = pos_i[0] - pos_j[0]
                dy = pos_i[1] - pos_j[1]
                dz = pos_i[2] - pos_j[2]
                if dx * dx + dy * dy + dz * dz <= threshold * threshold:
                    bonds.add((i, j))
    return sorted(bonds)


def _animate_bond_shape_keys(bond_obj, n_frames: int, frame_delta: int = 1) -> None:
    """Mirror xyz_import.py's per-frame shape-key value keyframing pattern."""
    blocks = bond_obj.data.shape_keys.key_blocks
    scene = bpy.context.scene

    if n_frames < 1:
        return
    if n_frames == 1:
        scene.frame_current = 0
        blocks[1].value = 1.0
        blocks[1].keyframe_insert("value")
        return

    scene.frame_current = 0
    blocks[1].value = 1.0
    blocks[2].value = 0.0
    blocks[1].keyframe_insert("value")
    blocks[2].keyframe_insert("value")

    for number in range(2, n_frames):
        scene.frame_current += frame_delta
        blocks[number - 1].value = 0.0
        blocks[number].value = 1.0
        blocks[number + 1].value = 0.0
        blocks[number - 1].keyframe_insert("value")
        blocks[number].keyframe_insert("value")
        blocks[number + 1].keyframe_insert("value")

    scene.frame_current += frame_delta
    blocks[n_frames].value = 1.0
    blocks[n_frames - 1].value = 0.0
    blocks[n_frames].keyframe_insert("value")
    blocks[n_frames - 1].keyframe_insert("value")


def _build_bonds(xyz: Path) -> None:
    """Create a Bonds mesh whose vertices follow atom positions via shape keys.

    Edges are drawn between atom pairs detected as bonded in any frame. A Skin
    modifier thickens edges into rectangular tubes; a Subsurf modifier rounds them.
    """
    frames = _center_frames(_parse_xyz_trajectory(xyz))
    if not frames:
        print("[reactx] bonds: no frames parsed")
        return
    bond_pairs = _compute_bond_pairs(frames)
    if not bond_pairs:
        print("[reactx] bonds: no bonds detected")
        return
    print(f"[reactx] bonds: {len(bond_pairs)} bond(s) across {len(frames)} frame(s)")

    verts0 = [pos for _, pos in frames[0]]
    edges = [tuple(p) for p in bond_pairs]

    bond_mesh = bpy.data.meshes.new("Bonds_mesh")
    bond_mesh.from_pydata(verts0, edges, [])
    bond_mesh.update()

    bond_obj = bpy.data.objects.new("Bonds", bond_mesh)
    bpy.context.scene.collection.objects.link(bond_obj)

    bpy.ops.object.select_all(action="DESELECT")
    bond_obj.select_set(True)
    bpy.context.view_layer.objects.active = bond_obj

    bpy.ops.object.shape_key_add(from_mix=False)  # Basis
    for frame_idx, frame in enumerate(frames):
        key = bond_obj.shape_key_add(name=f"Frame_{frame_idx}", from_mix=False)
        key.value = 0.0
        for v_idx, (_, pos) in enumerate(frame):
            key.data[v_idx].co = pos

    _animate_bond_shape_keys(bond_obj, len(frames))

    skin_mod = bond_obj.modifiers.new(name="Skin", type="SKIN")
    skin_mod.use_smooth_shade = True
    if bond_mesh.skin_vertices:
        skin_data = bond_mesh.skin_vertices[0].data
        for sv in skin_data:
            sv.radius = (BOND_RADIUS, BOND_RADIUS)
        skin_data[0].use_root = True

    subsurf = bond_obj.modifiers.new(name="Subsurf", type="SUBSURF")
    subsurf.levels = 1
    subsurf.render_levels = 2


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
    _build_bonds(xyz)
    _add_three_point_lighting()
    _add_camera_looking_at_origin()
    _set_timeline_to_trajectory(xyz)
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out))
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(list(sys.argv)))
