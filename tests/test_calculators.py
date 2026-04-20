import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator

from reactx.calculators import make_calculator


def test_make_calculator_lj_returns_working_calculator():
    calc = make_calculator("lj")
    assert isinstance(calc, Calculator)

    h2 = Atoms("H2", positions=[(0, 0, 0), (0, 0, 0.74)])
    h2.calc = calc
    energy = h2.get_potential_energy()
    assert isinstance(energy, float)


def test_make_calculator_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown calculator"):
        make_calculator("no-such-backend")


def test_make_calculator_uma_requires_fairchem(monkeypatch):
    """UMA path defers import of fairchem; when unavailable we get ImportError."""
    import reactx.calculators as mod

    def fake_import(*_args, **_kwargs):
        raise ImportError("fairchem-core not installed")

    monkeypatch.setattr(mod, "_build_uma_calculator", lambda **_: fake_import())
    with pytest.raises(ImportError):
        make_calculator("uma")
