"""--neb-refine flag exercises the optional NEB refinement on the best trial."""
import json
from pathlib import Path

import pytest
from ase.io import read

from reactx.cli import main


@pytest.mark.slow
def test_neb_refine_writes_refined_trajectory(
    tmp_path: Path, sn2_rxn_path: Path,
):
    out = tmp_path / "sn2_neb"
    rc = main([
        "run", str(sn2_rxn_path), "-o", str(out),
        "--backend", "uma",
        "--n-angles", "2",
        "--max-relax-steps", "30",
        "--neb-refine",
        "--neb-images", "5",
    ])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())
    assert meta["neb_refined"] is True

    frames = read(str(out / "trajectory.xyz"), index=":")
    # NEB with 5 images produces exactly 5 frames (no padding)
    assert len(frames) == 5
