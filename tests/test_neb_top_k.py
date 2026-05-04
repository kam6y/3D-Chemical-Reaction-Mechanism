"""Stage 2 NEB orchestration: run top-K NEB jobs (slow, requires UMA)."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.slow
def test_run_neb_top_k_sn2_workers_1(tmp_path: Path, screening_result_factory):
    from reactx.neb import run_neb_top_k
    top_k = screening_result_factory("sn2", n_results=1)
    out = run_neb_top_k(
        top_k,
        backend="uma",
        neb_model="uma-s-1p2",
        workers=1,
        n_images=5,
        fmax=0.1,
        max_steps=20,
        pad_frames=0,
        output_dir=tmp_path,
    )
    assert len(out) == 1
    assert "peak_energy" in out[0]
    assert (tmp_path / out[0]["xyz_path"]).exists()
