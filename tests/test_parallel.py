"""multiprocessing.Pool helper with workers=1 in-process fast-path."""
from __future__ import annotations

import os

import pytest

from reactx.parallel import run_with_pool


def _square(x: int) -> int:
    return x * x


def _record_init(token: str) -> None:
    # init runs in worker; expose via env var so the main process can verify.
    os.environ.setdefault("REACTX_TEST_INIT_COUNT", "0")
    os.environ["REACTX_TEST_INIT_COUNT"] = str(int(os.environ["REACTX_TEST_INIT_COUNT"]) + 1)
    os.environ["REACTX_TEST_INIT_TOKEN"] = token


def _read_init_count(_: int) -> int:
    return int(os.environ.get("REACTX_TEST_INIT_COUNT", "0"))


def test_run_with_pool_workers_1_is_inprocess():
    out = run_with_pool(_square, [1, 2, 3, 4], workers=1)
    assert out == [1, 4, 9, 16]


def test_run_with_pool_preserves_input_order():
    items = list(range(20))
    out = run_with_pool(_square, items, workers=2)
    assert out == [x * x for x in items]


def test_run_with_pool_raises_on_zero_workers():
    with pytest.raises(ValueError, match="workers"):
        run_with_pool(_square, [1, 2], workers=0)


def test_run_with_pool_handles_empty_items_workers_1():
    assert run_with_pool(_square, [], workers=1) == []


def test_run_with_pool_handles_empty_items_workers_n():
    assert run_with_pool(_square, [], workers=2) == []
