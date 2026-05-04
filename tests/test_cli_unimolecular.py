"""Tests for unimolecular reaction handling: n_candidates auto-clamp to 1."""
import json

from reactx.cli import main


def test_cli_unimolecular_clamps_top_k_to_screening_count(tmp_path, tmp_rxn_with_toml):
    rxn = tmp_rxn_with_toml("sn1_dissoc", toml_body=
        'description = "sn1 unimolecular"\n'
        'formed = []\n'
        'broken = [[1, 5]]\n'
        '[restraints]\nk_form=0\nk_broken=0.5\nr_broken=4.0\nmax_relax_steps=3\n'
        '[neb]\ntop_k=4\nn_images=3\nmax_steps=2\n'
        '[parallel]\nscreening_workers=1\nneb_workers=1\n'
    )
    out = tmp_path / "sn1d"
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "lj"])
    assert rc == 0
    meta = json.loads((out / "meta.json").read_text())
    # n_candidates=1 (auto-clamp) -> screening 1 件 -> top-K 1 件
    assert len(meta["screening_trials"]) == 1
    assert len(meta["top_k_indices"]) == 1
    assert len(meta["neb_results"]) == 1
