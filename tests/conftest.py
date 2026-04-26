"""Shared pytest fixtures for reactx tests."""
from pathlib import Path

import pytest


@pytest.fixture()
def examples_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture()
def sn2_rxn_path(examples_dir: Path) -> Path:
    return examples_dir / "sn2.rxn"
