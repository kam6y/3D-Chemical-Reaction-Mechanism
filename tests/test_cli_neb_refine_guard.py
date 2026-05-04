"""Test that --neb-refine is rejected for multi-bond reactions in Phase 3."""

from reactx import cli

_E2_TOML = """\
description = "e2 fast"
formed = [[4, 5]]
broken = [[2, 5], [1, 3]]
[restraints]
k_form = 1.0
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 5
[sampling]
n_candidates = 1
"""

_SN1_RECOMB_TOML = """\
description = "sn1_recomb fast"
formed = [[1, 5]]
broken = []
[restraints]
k_form = 1.0
k_broken = 0.0
r_broken = 4.0
max_relax_steps = 5
[sampling]
n_candidates = 1
"""

_SN2_TOML = """\
description = "sn2 fast"
formed = [[1, 3]]
broken = [[1, 2]]
[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 5
[sampling]
n_candidates = 8
"""


def test_neb_refine_rejected_for_e2_reaction(tmp_path, monkeypatch, caplog, tmp_rxn_with_toml):
    rxn = tmp_rxn_with_toml("e2", toml_body=_E2_TOML)
    out = tmp_path / "out"
    monkeypatch.setattr(cli, "_check_hf_auth", lambda: 0)
    monkeypatch.setattr(cli, "_configure_reactx_logging", lambda: None)
    cli.log.propagate = True
    caplog.set_level("INFO", logger="reactx")

    rc = cli.main([
        "run", str(rxn), "-o", str(out),
        "--backend", "lj", "--neb-refine",
    ])
    assert rc == 2
    msgs = " ".join(rec.getMessage() for rec in caplog.records)
    assert "Phase 3" in msgs
    assert "1 formed" in msgs and "1 broken" in msgs


def test_neb_refine_rejected_for_sn1_recomb_reaction(tmp_path, monkeypatch, caplog, tmp_rxn_with_toml):
    rxn = tmp_rxn_with_toml("sn1_recomb", toml_body=_SN1_RECOMB_TOML)
    out = tmp_path / "out"
    monkeypatch.setattr(cli, "_check_hf_auth", lambda: 0)
    monkeypatch.setattr(cli, "_configure_reactx_logging", lambda: None)
    cli.log.propagate = True
    caplog.set_level("INFO", logger="reactx")

    rc = cli.main([
        "run", str(rxn), "-o", str(out),
        "--backend", "lj", "--neb-refine",
    ])
    assert rc == 2
    msgs = " ".join(rec.getMessage() for rec in caplog.records)
    assert "Phase 3" in msgs
    assert "1 formed" in msgs and "1 broken" in msgs


def test_neb_refine_accepted_for_sn2(tmp_path, monkeypatch, tmp_rxn_with_toml):
    rxn = tmp_rxn_with_toml("sn2", toml_body=_SN2_TOML)
    out = tmp_path / "out"
    monkeypatch.setattr(cli, "_check_hf_auth", lambda: 0)
    cli.main([
        "run", str(rxn), "-o", str(out),
        "--backend", "lj", "--neb-refine",
    ])
    assert (out / "meta.json").exists()
