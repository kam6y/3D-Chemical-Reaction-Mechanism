"""Shared pytest fixtures for reactx tests."""

from pathlib import Path

import pytest


@pytest.fixture()
def examples_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture()
def sn2_rxn_path(examples_dir: Path) -> Path:
    return examples_dir / "sn2.rxn"


@pytest.fixture()
def sn1_rxn_path(examples_dir: Path) -> Path:
    return examples_dir / "sn1.rxn"


@pytest.fixture()
def e2_rxn_path(examples_dir: Path) -> Path:
    return examples_dir / "e2.rxn"


@pytest.fixture()
def e1_step2_rxn_path(examples_dir: Path) -> Path:
    return examples_dir / "e1_step2.rxn"


@pytest.fixture()
def proton_transfer_rxn_path(examples_dir: Path) -> Path:
    return examples_dir / "proton_transfer.rxn"
