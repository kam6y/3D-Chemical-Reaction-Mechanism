"""Blender headless script: trajectory.xyz -> ball-and-stick animated .blend.

Invoke:
    blender --background --python blender/render.py -- <trajectory.xyz> <out.blend>
"""
import math
import sys
from pathlib import Path

import bpy  # type: ignore[import-not-found]


FPS = 24


def _parse_args(argv: list[str]) -> tuple[Path, Path]:
    if "--" not in argv:
        raise SystemExit("Usage: blender --background --python render.py -- <xyz> <out.blend>")
    extra = argv[argv.index("--") + 1:]
    if len(extra) < 2:
        raise SystemExit("Need <xyz> <out.blend>")
    return Path(extra[0]), Path(extra[1])


def _reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = FPS


def _import_trajectory(xyz: Path) -> None:
    try:
        bpy.ops.preferences.addon_enable(module="atomic_blender_pdb_xyz")
    except Exception:
        bpy.ops.preferences.addon_enable(module="atomic_blender_xyz")
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


def _stretch_timeline() -> None:
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = max(scene.frame_end, int(FPS * 10))


def main(argv: list[str]) -> int:
    xyz, out = _parse_args(argv)
    _reset_scene()
    _import_trajectory(xyz)
    _add_three_point_lighting()
    _add_camera_looking_at_origin()
    _stretch_timeline()
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out))
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(list(sys.argv)))
