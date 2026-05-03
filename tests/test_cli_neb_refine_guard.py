"""Test that --neb-refine is rejected for multi-bond reactions in Phase 3.

The bond_changes_product swap logic in cli._cmd_run is only valid for
1 formed + 1 broken bond reactions (canonical SN2-like). For multi-bond
reactions (e.g. E2: 1 formed + 2 broken), the product-side endpoint
construction is not yet defined, so --neb-refine must be rejected at the
CLI level with exit code 2.
"""
from pathlib import Path

import pytest

from reactx import cli


def test_neb_refine_rejected_for_e2_reaction(tmp_path, monkeypatch, caplog):
    """E2 (1 formed + 2 broken) must be rejected with exit code 2."""
    rxn = Path("examples/e2.rxn")
    if not rxn.exists():
        pytest.skip("examples/e2.rxn not yet created (Task 7)")

    out = tmp_path / "out"
    monkeypatch.setattr(cli, "_check_hf_auth", lambda: 0)
    # Enable log propagation so caplog can capture reactx messages.
    # _configure_reactx_logging() sets propagate=False, so neutralize it and
    # restore propagate after.
    monkeypatch.setattr(cli, "_configure_reactx_logging", lambda: None)
    cli.log.propagate = True
    caplog.set_level("INFO", logger="reactx")

    rc = cli.main([
        "run", str(rxn), "-o", str(out),
        "--backend", "lj", "--n-angles", "1", "--max-relax-steps", "5",
        "--neb-refine",
    ])
    assert rc == 2, f"expected exit code 2 for multi-bond + --neb-refine, got {rc}"
    # The error message should mention Phase 3 and the 1+1 constraint.
    msgs = " ".join(rec.getMessage() for rec in caplog.records)
    assert "Phase 3" in msgs
    assert "1 formed" in msgs and "1 broken" in msgs


def test_neb_refine_rejected_for_sn1_recomb_reaction(tmp_path, monkeypatch, caplog):
    """SN1 step 2 (formed=1, broken=0) も 1+1 以外なので reject される。"""
    rxn = Path("examples/sn1_recomb.rxn")
    if not rxn.exists():
        pytest.skip("examples/sn1_recomb.rxn not yet created (Task 10)")

    out = tmp_path / "out"
    monkeypatch.setattr(cli, "_check_hf_auth", lambda: 0)
    monkeypatch.setattr(cli, "_configure_reactx_logging", lambda: None)
    cli.log.propagate = True
    caplog.set_level("INFO", logger="reactx")

    rc = cli.main([
        "run", str(rxn), "-o", str(out),
        "--backend", "lj", "--n-angles", "1", "--max-relax-steps", "5",
        "--neb-refine",
    ])
    assert rc == 2, (
        f"expected exit code 2 for SN1 step 2 + --neb-refine, got {rc}"
    )
    msgs = " ".join(rec.getMessage() for rec in caplog.records)
    assert "Phase 3" in msgs
    assert "1 formed" in msgs and "1 broken" in msgs


def test_neb_refine_accepted_for_sn2(sn2_rxn_path, tmp_path, monkeypatch):
    """1+1 reactions (SN2) pass the CLI guard.

    NEB itself may fail on the LJ backend (no realistic energies), but the
    CLI guard must let the pipeline reach the embed/relax stage so meta.json
    gets written.
    """
    out = tmp_path / "out"
    monkeypatch.setattr(cli, "_check_hf_auth", lambda: 0)

    cli.main([
        "run", str(sn2_rxn_path), "-o", str(out),
        "--backend", "lj", "--n-angles", "1", "--max-relax-steps", "5",
        "--neb-refine",
    ])
    # rc may be nonzero (LJ backend won't produce a sensible NEB), but the
    # important invariant is that we got past the CLI guard far enough to
    # write meta.json.
    assert (out / "meta.json").exists(), (
        "expected to pass CLI guard and reach pipeline (meta.json missing)"
    )


def test_neb_refine_rejected_for_metathesis_reaction(tmp_path, monkeypatch, caplog):
    """metathesis (formed=2, broken=2) も 1+1 以外なので reject される。"""
    rxn = Path("examples/metathesis_4center.rxn")
    if not rxn.exists():
        pytest.skip("examples/metathesis_4center.rxn not yet created (Task 9)")

    out = tmp_path / "out"
    monkeypatch.setattr(cli, "_check_hf_auth", lambda: 0)
    monkeypatch.setattr(cli, "_configure_reactx_logging", lambda: None)
    cli.log.propagate = True
    caplog.set_level("INFO", logger="reactx")

    rc = cli.main([
        "run", str(rxn), "-o", str(out),
        "--backend", "lj", "--n-angles", "1", "--max-relax-steps", "5",
        "--neb-refine",
    ])
    assert rc == 2, (
        f"expected exit code 2 for metathesis + --neb-refine, got {rc}"
    )
    msgs = " ".join(rec.getMessage() for rec in caplog.records)
    assert "Phase 3" in msgs
    assert "1 formed" in msgs and "1 broken" in msgs
