"""Smoke test for the AFIR force module - substantive tests live in
`tests/test_afir_constraint.py`."""
from reactx.artificial_force import AFIRConstraint, build_afir_constraint


def test_module_exports_public_api():
    assert AFIRConstraint is not None
    assert callable(build_afir_constraint)
