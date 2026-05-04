"""multiprocessing.Pool helper with workers=1 in-process fast-path.

Use spawn context unconditionally (Windows safety + CUDA reinit safety).
"""
from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")
R = TypeVar("R")

_CACHED_CALCULATOR: Any = None


def run_with_pool(
    fn: Callable[[T], R],
    items: list[T],
    *,
    workers: int,
    initializer: Callable[..., None] | None = None,
    initargs: tuple = (),
) -> list[R]:
    """Run fn over items, optionally in parallel via multiprocessing.Pool (spawn).

    workers == 1 -> in-process map (no Pool overhead, no model reload).
    workers >= 2 -> spawn Pool; each worker runs initializer once at startup.

    Output order matches input order (Pool.map semantics).
    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if not items:
        return []
    if workers == 1:
        if initializer is not None:
            initializer(*initargs)
        return [fn(it) for it in items]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, initializer=initializer, initargs=initargs) as pool:
        return list(pool.map(fn, items))


def init_uma_worker(model_name: str, task_name: str = "omol") -> None:
    """Build a UMA Calculator once per worker process and cache module-globally.

    Called from run_with_pool's initializer at worker startup. Per-task fns
    pull the cached calculator via get_cached_calculator().
    """
    global _CACHED_CALCULATOR
    from reactx.calculators import make_calculator

    _CACHED_CALCULATOR = make_calculator(
        "uma", model_name=model_name, task_name=task_name,
    )


def init_lj_worker() -> None:
    """LJ backend init for tests; cheap and avoids fairchem import in CI."""
    global _CACHED_CALCULATOR
    from reactx.calculators import make_calculator

    _CACHED_CALCULATOR = make_calculator("lj")


def get_cached_calculator():
    """Return the worker-cached Calculator. Raises if not yet initialized."""
    if _CACHED_CALCULATOR is None:
        raise RuntimeError(
            "get_cached_calculator() called before init_uma_worker / init_lj_worker"
        )
    return _CACHED_CALCULATOR
