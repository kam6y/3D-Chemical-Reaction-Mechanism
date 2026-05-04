# Phase 8 — CI-NEB Unified Path Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Phase 7 の人工力 relax をスクリーニング段階に降格し、`top_k=4` 件だけ CI-NEB で本評価する 2-stage パイプラインに統一。`uma-s-1p2` を default model にし、screening / NEB を process pool で並列化。endpoint は screening trajectory の first/last frame 由来 (拘束無し短 relax) で、E2 / SN1 含む全反応で CI-NEB を走らせる。

**Architecture:** 新規 `reactx/parallel.py` (Pool helper, spawn 強制), `reactx/endpoints.py` (拘束無し再 relax), `reactx/screening.py` (Stage 1 並列実行) を追加。`reactx/neb.py` に `run_neb_for_trial` / `run_neb_top_k` を追加。`reactx/config.py` に `[model]` `[neb]` `[parallel]` セクション追加。`reactx/scoring.py` で `TrialResult` → `ScreeningTrialResult` rename + `top_k_trials` 追加。`reactx/cli.py` を全書き換え。`reactx/align.py` 削除 (P endpoint が trajectory 由来でアトム順序が同じなので不要)。

**Tech Stack:** Python 3.11+, ASE (FIRE / BFGS / NEB), fairchem-core (UMA), RDKit, NumPy, multiprocessing (spawn), pytest。

**Spec:** `docs/superpowers/specs/2026-05-05-cineb-unified-design.md`

---

## File Structure

### New files
- `reactx/parallel.py` — `run_with_pool` (workers=1 で in-process, >=2 で multiprocessing.Pool / spawn) + `init_uma_worker` / `get_cached_calculator`
- `reactx/endpoints.py` — `relax_endpoint` (拘束無し BFGS 短 relax)
- `reactx/screening.py` — `ScreeningTrialResult` (rename from `TrialResult`)、`screen_all_trials`、`_run_one_trial` worker fn
- `tests/test_parallel.py`
- `tests/test_endpoints.py`
- `tests/test_screening.py`
- `tests/test_neb_top_k.py`

### Modified files
- `reactx/config.py` — `ModelConfig` / `NebConfig` / `ParallelConfig` 追加、validation
- `reactx/scoring.py` — `TrialResult` → `ScreeningTrialResult` rename、`top_k_trials` 追加、`select_best_trial` を wrapper 化
- `reactx/neb.py` — `run_neb_for_trial` / `run_neb_top_k` 追加、既存 `run_neb` はそのまま
- `reactx/cli.py` — orchestration 全書き換え (Stage1 → Top-K → Stage2)、`--neb-refine` / `--neb-images` / `--model` 削除
- `reactx/calculators.py` — default model name を `"uma-s-1p2"` に
- `tests/test_config.py` — 新セクション validation テスト追加
- `tests/test_scoring.py` — `top_k_trials` テスト追加、`TrialResult` 参照を `ScreeningTrialResult` に
- `tests/test_cli.py` — Phase 8 schema、`--neb-refine` 系 assert 削除
- `tests/test_cli_unimolecular.py` — Phase 8 schema
- `tests/test_re1_sn2.py` — CI-NEB 結果 assert
- `tests/test_re1_proton_transfer.py` — 同上
- `tests/test_re1_menshutkin.py` — 同上
- `tests/test_re3_e2.py` — 同上
- `tests/test_re3_sn1_dissoc.py` — 同上
- `tests/test_re4_sn1_recomb.py` — 同上
- `tests/test_examples_menshutkin.py` — 同上
- `tests/test_wallclock_sn2.py` — Phase 8 wall-clock しきい値
- `tests/test_path_relax.py` — `TrialResult` 参照変更があれば更新
- `tests/test_artificial_force.py` — 同上 (rename 影響なら)
- `examples/sn1_dissoc.rxn.toml` — `[neb] top_k = 1` 追加
- `examples/menshutkin.rxn.toml` — `[neb] pad_frames = 3` 任意追加
- `README.md` — アーキテクチャ図、対応反応表、CLI フラグ、Wall-clock 表、方針と限界

### Deleted files
- `reactx/align.py`
- `tests/test_align.py`
- `tests/test_neb_refine_sn2.py`
- `tests/test_cli_neb_refine_guard.py`

---

## Task 1: Add ModelConfig / NebConfig / ParallelConfig to reactx/config.py

**Files:**
- Modify: `reactx/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing tests in tests/test_config.py**

末尾に追記:

```python
def test_config_default_model_section_uma_s(tmp_path: Path):
    rxn = tmp_path / "x.rxn"
    rxn.write_text("$RXN\n", encoding="utf-8")
    (tmp_path / "x.rxn.toml").write_text(
        'description="x"\nformed=[[1,2]]\nbroken=[]\n'
        '[restraints]\nk_form=1\nk_broken=1\nr_broken=4\nmax_relax_steps=10\n',
        encoding="utf-8",
    )
    cfg = load_config(rxn)
    assert cfg.model.screening_model == "uma-s-1p2"
    assert cfg.model.neb_model == "uma-s-1p2"


def test_config_default_neb_section(tmp_path: Path):
    rxn = tmp_path / "x.rxn"
    rxn.write_text("$RXN\n", encoding="utf-8")
    (tmp_path / "x.rxn.toml").write_text(
        'description="x"\nformed=[[1,2]]\nbroken=[]\n'
        '[restraints]\nk_form=1\nk_broken=1\nr_broken=4\nmax_relax_steps=10\n',
        encoding="utf-8",
    )
    cfg = load_config(rxn)
    assert cfg.neb.top_k == 4
    assert cfg.neb.n_images == 7
    assert cfg.neb.fmax == 0.05
    assert cfg.neb.max_steps == 200
    assert cfg.neb.pad_frames == 0


def test_config_default_parallel_section(tmp_path: Path):
    rxn = tmp_path / "x.rxn"
    rxn.write_text("$RXN\n", encoding="utf-8")
    (tmp_path / "x.rxn.toml").write_text(
        'description="x"\nformed=[[1,2]]\nbroken=[]\n'
        '[restraints]\nk_form=1\nk_broken=1\nr_broken=4\nmax_relax_steps=10\n',
        encoding="utf-8",
    )
    cfg = load_config(rxn)
    assert cfg.parallel.screening_workers == 3
    assert cfg.parallel.neb_workers == 3


def test_config_neb_top_k_must_be_positive(tmp_path: Path):
    rxn = tmp_path / "x.rxn"
    rxn.write_text("$RXN\n", encoding="utf-8")
    (tmp_path / "x.rxn.toml").write_text(
        'description="x"\nformed=[[1,2]]\nbroken=[]\n'
        '[restraints]\nk_form=1\nk_broken=1\nr_broken=4\nmax_relax_steps=10\n'
        '[neb]\ntop_k=0\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="neb.top_k"):
        load_config(rxn)


def test_config_neb_n_images_min_3(tmp_path: Path):
    rxn = tmp_path / "x.rxn"
    rxn.write_text("$RXN\n", encoding="utf-8")
    (tmp_path / "x.rxn.toml").write_text(
        'description="x"\nformed=[[1,2]]\nbroken=[]\n'
        '[restraints]\nk_form=1\nk_broken=1\nr_broken=4\nmax_relax_steps=10\n'
        '[neb]\nn_images=2\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="neb.n_images"):
        load_config(rxn)


def test_config_parallel_workers_must_be_positive(tmp_path: Path):
    rxn = tmp_path / "x.rxn"
    rxn.write_text("$RXN\n", encoding="utf-8")
    (tmp_path / "x.rxn.toml").write_text(
        'description="x"\nformed=[[1,2]]\nbroken=[]\n'
        '[restraints]\nk_form=1\nk_broken=1\nr_broken=4\nmax_relax_steps=10\n'
        '[parallel]\nscreening_workers=0\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="parallel.screening_workers"):
        load_config(rxn)


def test_config_model_screening_must_be_nonempty(tmp_path: Path):
    rxn = tmp_path / "x.rxn"
    rxn.write_text("$RXN\n", encoding="utf-8")
    (tmp_path / "x.rxn.toml").write_text(
        'description="x"\nformed=[[1,2]]\nbroken=[]\n'
        '[restraints]\nk_form=1\nk_broken=1\nr_broken=4\nmax_relax_steps=10\n'
        '[model]\nscreening_model=""\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="model.screening_model"):
        load_config(rxn)


def test_config_unknown_keys_in_neb_rejected(tmp_path: Path):
    rxn = tmp_path / "x.rxn"
    rxn.write_text("$RXN\n", encoding="utf-8")
    (tmp_path / "x.rxn.toml").write_text(
        'description="x"\nformed=[[1,2]]\nbroken=[]\n'
        '[restraints]\nk_form=1\nk_broken=1\nr_broken=4\nmax_relax_steps=10\n'
        '[neb]\nbogus=1\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown config key"):
        load_config(rxn)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py -v -k "model_section or neb_section or parallel_section or top_k or n_images or workers or model_screening or unknown_keys_in_neb"`
Expected: FAIL — ModelConfig / NebConfig / ParallelConfig not in cfg

- [ ] **Step 3: Add new dataclasses and validation in reactx/config.py**

`reactx/config.py` 上部 (既存 dataclass の隣):

```python
@dataclass(frozen=True)
class ModelConfig:
    screening_model: str = "uma-s-1p2"
    neb_model: str = "uma-s-1p2"


@dataclass(frozen=True)
class NebConfig:
    top_k: int = 4
    n_images: int = 7
    fmax: float = 0.05
    max_steps: int = 200
    pad_frames: int = 0


@dataclass(frozen=True)
class ParallelConfig:
    screening_workers: int = 3
    neb_workers: int = 3
```

`ReactionConfig` にフィールド追加:

```python
@dataclass(frozen=True)
class ReactionConfig:
    description: str
    formed: tuple[tuple[int, int], ...]
    broken: tuple[tuple[int, int], ...]
    restraints: RestraintConfig
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    neb: NebConfig = field(default_factory=NebConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
```

`_TOP_LEVEL_KEYS` を更新:

```python
_TOP_LEVEL_KEYS = {"description", "formed", "broken", "restraints", "sampling", "model", "neb", "parallel"}
_MODEL_KEYS = {"screening_model", "neb_model"}
_NEB_KEYS = {"top_k", "n_images", "fmax", "max_steps", "pad_frames"}
_PARALLEL_KEYS = {"screening_workers", "neb_workers"}
```

`_validate` に builder 呼び出し追加 (既存 `sampling = _build_sampling(...)` の後):

```python
    model = _build_model(raw.get("model", {}), source=source)
    neb = _build_neb(raw.get("neb", {}), source=source)
    parallel = _build_parallel(raw.get("parallel", {}), source=source)

    return ReactionConfig(
        description=description,
        formed=formed,
        broken=broken,
        restraints=restraints,
        sampling=sampling,
        model=model,
        neb=neb,
        parallel=parallel,
    )
```

新 builder 関数を `_build_sampling` の下に追加:

```python
def _build_model(raw: dict, *, source: str) -> ModelConfig:
    _check_keys(raw, _MODEL_KEYS, set(), scope="model", source=source)
    screening = _as_str(raw.get("screening_model", "uma-s-1p2"), "model.screening_model", source)
    neb = _as_str(raw.get("neb_model", "uma-s-1p2"), "model.neb_model", source)
    return ModelConfig(screening_model=screening, neb_model=neb)


def _build_neb(raw: dict, *, source: str) -> NebConfig:
    _check_keys(raw, _NEB_KEYS, set(), scope="neb", source=source)
    top_k = _as_int(raw.get("top_k", 4), "neb.top_k", source, positive=True)
    n_images = _as_int(raw.get("n_images", 7), "neb.n_images", source, positive=True)
    if n_images < 3:
        raise ValueError(f"{source}: 'neb.n_images' must be >= 3 (got {n_images})")
    fmax = _as_float(raw.get("fmax", 0.05), "neb.fmax", source, positive=True)
    max_steps = _as_int(raw.get("max_steps", 200), "neb.max_steps", source, positive=True)
    pad_frames_raw = raw.get("pad_frames", 0)
    if not isinstance(pad_frames_raw, int) or isinstance(pad_frames_raw, bool):
        raise ValueError(f"{source}: 'neb.pad_frames' must be an int")
    if pad_frames_raw < 0:
        raise ValueError(f"{source}: 'neb.pad_frames' must be >= 0 (got {pad_frames_raw})")
    return NebConfig(
        top_k=top_k, n_images=n_images, fmax=fmax,
        max_steps=max_steps, pad_frames=int(pad_frames_raw),
    )


def _build_parallel(raw: dict, *, source: str) -> ParallelConfig:
    _check_keys(raw, _PARALLEL_KEYS, set(), scope="parallel", source=source)
    screening = _as_int(raw.get("screening_workers", 3), "parallel.screening_workers", source, positive=True)
    neb = _as_int(raw.get("neb_workers", 3), "parallel.neb_workers", source, positive=True)
    return ParallelConfig(screening_workers=screening, neb_workers=neb)


def _as_str(raw: object, key: str, source: str) -> str:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{source}: '{key}' must be a non-empty string")
    return raw
```

- [ ] **Step 4: Run config tests**

Run: `pytest tests/test_config.py -v`
Expected: ALL PASS (既存 + 新規 8 件)

- [ ] **Step 5: Commit**

```bash
git add reactx/config.py tests/test_config.py
git commit -m "feat(config): add [model] [neb] [parallel] sections with validation"
```

---

## Task 2: Create reactx/parallel.py with Pool helper

**Files:**
- Create: `reactx/parallel.py`
- Test: `tests/test_parallel.py`

- [ ] **Step 1: Write failing tests in tests/test_parallel.py**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_parallel.py -v`
Expected: FAIL — `reactx.parallel` not importable

- [ ] **Step 3: Implement reactx/parallel.py**

```python
"""multiprocessing.Pool helper with workers=1 in-process fast-path.

Use spawn context unconditionally (Windows safety + CUDA reinit safety).
"""
from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable, Iterable
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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_parallel.py -v`
Expected: PASS (5/5)

- [ ] **Step 5: Commit**

```bash
git add reactx/parallel.py tests/test_parallel.py
git commit -m "feat(parallel): add run_with_pool helper with spawn-only multiprocessing"
```

---

## Task 3: Rename TrialResult → ScreeningTrialResult and add top_k_trials

**Files:**
- Modify: `reactx/scoring.py`
- Modify: `tests/test_scoring.py`
- Modify (rename refs only): `reactx/cli.py`, `tests/test_path_relax.py` (if any)

- [ ] **Step 1: Survey TrialResult usage**

Run: `git grep -n "TrialResult" -- reactx tests`
Expected: hits in `reactx/scoring.py`, `reactx/cli.py`, `tests/test_scoring.py`

- [ ] **Step 2: Write failing tests in tests/test_scoring.py**

末尾に追記:

```python
def _make_trial(idx: int, *, reached: bool, peak: float) -> "ScreeningTrialResult":
    from reactx.scoring import ScreeningTrialResult
    return ScreeningTrialResult(
        trial_idx=idx, direction=np.array([0.0, 0.0, 1.0]),
        frames=[], energies=[], reached_product=reached,
        peak_energy=peak, n_steps=0,
    )


def test_top_k_trials_returns_k_when_enough_reached():
    from reactx.scoring import top_k_trials
    trials = [
        _make_trial(0, reached=True, peak=2.0),
        _make_trial(1, reached=True, peak=1.0),
        _make_trial(2, reached=True, peak=3.0),
        _make_trial(3, reached=False, peak=0.5),
    ]
    out = top_k_trials(trials, 2)
    assert [t.trial_idx for t in out] == [1, 0]


def test_top_k_trials_falls_back_to_unreached_to_fill_k():
    from reactx.scoring import top_k_trials
    trials = [
        _make_trial(0, reached=True, peak=2.0),
        _make_trial(1, reached=False, peak=1.0),
        _make_trial(2, reached=False, peak=0.5),
    ]
    out = top_k_trials(trials, 3)
    # reached=True 群 (1 件) を最優先 -> reached=False 群を peak 昇順で補完
    assert [t.trial_idx for t in out] == [0, 2, 1]


def test_top_k_trials_returns_all_when_fewer_than_k():
    from reactx.scoring import top_k_trials
    trials = [
        _make_trial(0, reached=True, peak=1.0),
        _make_trial(1, reached=False, peak=2.0),
    ]
    out = top_k_trials(trials, 5)
    assert len(out) == 2


def test_top_k_trials_empty_raises():
    from reactx.scoring import top_k_trials
    with pytest.raises(ValueError):
        top_k_trials([], 1)


def test_top_k_trials_zero_k_raises():
    from reactx.scoring import top_k_trials
    with pytest.raises(ValueError, match="k must be"):
        top_k_trials([_make_trial(0, reached=True, peak=1.0)], 0)
```

既存テストの `TrialResult` 参照をすべて `ScreeningTrialResult` に置換 (test_scoring.py, test_path_relax.py 内、import 行と dataclass 構築箇所)。

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_scoring.py -v`
Expected: FAIL — `ScreeningTrialResult` 未定義 / `top_k_trials` 未実装

- [ ] **Step 4: Refactor reactx/scoring.py**

`TrialResult` を `ScreeningTrialResult` に rename し、`top_k_trials` 追加:

```python
"""Trial scoring + final-best selection for screening trials.

Phase 8: TrialResult renamed to ScreeningTrialResult to reflect its role
as Stage 1 (screening) output. Adds top_k_trials for the screening->NEB
hand-off: pick top-K by reached_product first, then by peak_energy
ascending; if fewer than K reached, fill from unreached pool by peak.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms


@dataclass
class ScreeningTrialResult:
    """Outcome of one Stage 1 (artificial-force) relax trial.

    `direction` is the unit vector used for sphere-based fragment placement
    (placeholder +z for unimolecular passthrough).
    `error` is the relax exception string when frames=[]/energies=[];
    None on success.
    """

    trial_idx: int
    direction: np.ndarray
    frames: list[Atoms]
    energies: list[float]
    reached_product: bool
    peak_energy: float
    n_steps: int
    error: str | None = None


def reached_product(
    final_atoms: Atoms,
    formed: list[tuple[int, int]],
    broken: list[tuple[int, int]],
    *,
    r_form_targets: list[float],
    r_broken_target: float,
    form_tol: float = 0.3,
    broken_tol: float = 0.5,
) -> bool:
    """True iff every formed bond is within r_form + form_tol AND every broken
    bond is at least r_broken - broken_tol apart.
    """
    if len(formed) != len(r_form_targets):
        raise ValueError(
            f"formed ({len(formed)}) must match r_form_targets ({len(r_form_targets)})"
        )
    p = final_atoms.positions
    for (a, b), rt in zip(formed, r_form_targets, strict=True):
        d = float(np.linalg.norm(p[a] - p[b]))
        if d > rt + form_tol:
            return False
    for a, b in broken:
        d = float(np.linalg.norm(p[a] - p[b]))
        if d < r_broken_target - broken_tol:
            return False
    return True


def top_k_trials(
    results: list[ScreeningTrialResult],
    k: int,
) -> list[ScreeningTrialResult]:
    """Return up to k results ranked by (reached_product desc, peak_energy asc).

    1. reached_product=True 群を peak_energy 昇順で並べる
    2. reached_product=False 群を peak_energy 昇順で並べる
    3. 連結して先頭から k 件
    """
    if not results:
        raise ValueError("top_k_trials called with empty results list")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    reached = sorted(
        (r for r in results if r.reached_product), key=lambda r: r.peak_energy,
    )
    unreached = sorted(
        (r for r in results if not r.reached_product), key=lambda r: r.peak_energy,
    )
    return (reached + unreached)[:k]


def score_trials(results: list[ScreeningTrialResult]) -> ScreeningTrialResult:
    """Return the best ScreeningTrialResult (top_k_trials(..., 1)[0])."""
    return top_k_trials(results, 1)[0]


def select_best_trial(trials: list[ScreeningTrialResult]) -> int:
    """Return the trial_idx of the best result."""
    return score_trials(trials).trial_idx
```

`reactx/cli.py` で `from reactx.scoring import TrialResult` を `from reactx.scoring import ScreeningTrialResult as TrialResult` に **しない** (alias は腐るので)。直接 import 名を更新する:

`reactx/cli.py`:

```python
# 旧
from reactx.scoring import TrialResult, reached_product, score_trials
# 新
from reactx.scoring import ScreeningTrialResult, reached_product, score_trials
```

`TrialResult(` の参照を `ScreeningTrialResult(` に全置換。`list[TrialResult]` も `list[ScreeningTrialResult]` に。

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_scoring.py tests/test_cli.py -v -k "not slow"`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add reactx/scoring.py reactx/cli.py tests/test_scoring.py tests/test_path_relax.py
git commit -m "refactor(scoring): rename TrialResult->ScreeningTrialResult, add top_k_trials"
```

---

## Task 4: Switch default model to uma-s-1p2

**Files:**
- Modify: `reactx/calculators.py`
- Modify: `tests/test_calculators.py` (if asserts default)

- [ ] **Step 1: Inspect current default**

Run: `git grep -n "uma-m-1p1" -- reactx`
Expected: hits in `reactx/calculators.py`, possibly README + tests

- [ ] **Step 2: Update default in reactx/calculators.py**

```python
# 旧
def _build_uma_calculator(
    *,
    model_name: str = "uma-m-1p1",
    ...
)
# 新
def _build_uma_calculator(
    *,
    model_name: str = "uma-s-1p2",
    ...
)
```

- [ ] **Step 3: Run unit tests**

Run: `pytest tests/test_calculators.py -v`
Expected: PASS (LJ backend なら影響無し、UMA 系は slow なので未実行)

- [ ] **Step 4: Commit**

```bash
git add reactx/calculators.py
git commit -m "feat(calculators): switch UMA default model to uma-s-1p2"
```

---

## Task 5: Create reactx/endpoints.py

**Files:**
- Create: `reactx/endpoints.py`
- Test: `tests/test_endpoints.py`

- [ ] **Step 1: Write failing tests in tests/test_endpoints.py**

```python
"""Endpoint un-restrained relax for CI-NEB."""
from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.calculators.lj import LennardJones

from reactx.endpoints import relax_endpoint


def test_relax_endpoint_lj_dimer_moves_toward_minimum():
    # LJ minimum is at r = 2^(1/6) * sigma; default sigma=1, so r_min ≈ 1.122.
    atoms = Atoms("Ar2", positions=[[0, 0, 0], [3.0, 0, 0]])
    relaxed = relax_endpoint(atoms, LennardJones(), fmax=0.01, max_steps=200)
    r = float(np.linalg.norm(relaxed.positions[1] - relaxed.positions[0]))
    assert 1.0 < r < 1.5


def test_relax_endpoint_does_not_raise_on_max_steps():
    # fmax impossible in 1 step -> still returns Atoms, no exception.
    atoms = Atoms("Ar2", positions=[[0, 0, 0], [3.0, 0, 0]])
    relaxed = relax_endpoint(atoms, LennardJones(), fmax=1e-12, max_steps=1)
    assert relaxed is not None
    assert len(relaxed) == 2


def test_relax_endpoint_returns_independent_copy():
    # Input atoms must NOT be mutated; relaxed is a separate Atoms.
    atoms = Atoms("Ar2", positions=[[0, 0, 0], [3.0, 0, 0]])
    original_positions = atoms.positions.copy()
    _ = relax_endpoint(atoms, LennardJones(), fmax=0.01, max_steps=20)
    np.testing.assert_array_equal(atoms.positions, original_positions)


def test_relax_endpoint_calc_detached_from_result():
    # Result Atoms.calc must be None so it serializes cleanly.
    atoms = Atoms("Ar2", positions=[[0, 0, 0], [3.0, 0, 0]])
    relaxed = relax_endpoint(atoms, LennardJones(), fmax=0.01, max_steps=20)
    assert relaxed.calc is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_endpoints.py -v`
Expected: FAIL — `reactx.endpoints` not importable

- [ ] **Step 3: Implement reactx/endpoints.py**

```python
"""NEB endpoint preparation: short un-restrained relax of R / P sides.

The screening trajectory's first/last frames are constrained relax outputs,
not local minima of the bare PES. CI-NEB requires both endpoints at minima
to converge to a meaningful saddle. relax_endpoint runs a short BFGS pass
without restraints to settle each endpoint into a nearby minimum.

Tolerant of non-convergence (returns whatever the optimizer reaches).
"""
from __future__ import annotations

import logging

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize import BFGS

log = logging.getLogger(__name__)


def relax_endpoint(
    atoms: Atoms,
    calc: Calculator,
    *,
    fmax: float = 0.05,
    max_steps: int = 50,
) -> Atoms:
    """Un-restrained BFGS relax to settle an endpoint at a local minimum.

    Returns a new Atoms with calc detached. Input atoms is not mutated.
    """
    work = atoms.copy()
    work.calc = calc
    try:
        BFGS(work, logfile=None).run(fmax=fmax, steps=max_steps)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "relax_endpoint BFGS raised %s: %s; returning current geometry",
            type(exc).__name__, exc,
        )
    out = work.copy()
    out.calc = None
    return out
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_endpoints.py -v`
Expected: PASS (4/4)

- [ ] **Step 5: Commit**

```bash
git add reactx/endpoints.py tests/test_endpoints.py
git commit -m "feat(endpoints): add un-restrained BFGS endpoint relax helper for NEB"
```

---

## Task 6: Create reactx/screening.py with parallel Stage 1 orchestration

**Files:**
- Create: `reactx/screening.py`
- Test: `tests/test_screening.py`

- [ ] **Step 1: Write failing tests in tests/test_screening.py**

```python
"""Stage 1 screening: parallel artificial-force relax of all placement survivors."""
from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.lj import LennardJones
from rdkit import Chem

from reactx.bond_changes import BondChanges
from reactx.config import ReactionConfig, RestraintConfig, SamplingConfig
from reactx.placement import PlacementResult, PlacementTrial
from reactx.screening import screen_all_trials


def _trivial_placement_two_trials() -> tuple[Chem.Mol, BondChanges, PlacementResult]:
    # H + H system (covers vdw_radius lookup + Atoms build), 2 placement trials.
    mol = Chem.MolFromSmiles("[H].[H]")
    mol_h = Chem.AddHs(mol)
    Chem.SanitizeMol(mol_h)
    pos1 = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    pos2 = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    trials = [
        PlacementTrial(direction=np.array([1.0, 0, 0]), d_min=3.0, positions=pos1),
        PlacementTrial(direction=np.array([-1.0, 0, 0]), d_min=4.0, positions=pos2),
    ]
    pl = PlacementResult(trials=trials, n_candidates=2, n_blocked=0,
                         blocked_reasons=[None, None])
    bc = BondChanges(formed=((0, 1),), broken=())
    return mol_h, bc, pl


def _make_cfg() -> ReactionConfig:
    return ReactionConfig(
        description="t",
        formed=((1, 2),),
        broken=tuple(),
        restraints=RestraintConfig(
            k_form=0.5, k_broken=0.0, r_broken=4.0, max_relax_steps=10,
        ),
        sampling=SamplingConfig(n_candidates=2),
    )


def test_screen_all_trials_workers_1_returns_results_in_order():
    mol_h, bc, pl = _trivial_placement_two_trials()
    cfg = _make_cfg()

    results = screen_all_trials(
        pl, mol_h, bc, cfg,
        backend="lj",
        screening_model="",
        workers=1,
        relax_fmax=0.5,
        traj_stride=5,
        seed=0,
    )
    assert [r.trial_idx for r in results] == [0, 1]
    assert all(len(r.frames) >= 1 for r in results)


def test_screen_all_trials_records_error_for_failed_trial(monkeypatch):
    mol_h, bc, pl = _trivial_placement_two_trials()
    cfg = _make_cfg()

    from reactx import screening

    real_relax = screening.relax_with_restraints
    call_count = {"n": 0}

    def flaky_relax(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("boom")
        return real_relax(*args, **kwargs)

    monkeypatch.setattr(screening, "relax_with_restraints", flaky_relax)

    results = screen_all_trials(
        pl, mol_h, bc, cfg,
        backend="lj",
        screening_model="",
        workers=1,
        relax_fmax=0.5,
        traj_stride=5,
        seed=0,
    )
    assert results[0].error is not None
    assert "boom" in results[0].error
    assert results[0].frames == []
    assert results[1].error is None


def test_screen_all_trials_empty_placement_returns_empty():
    mol = Chem.AddHs(Chem.MolFromSmiles("[H].[H]"))
    pl = PlacementResult(trials=[], n_candidates=0, n_blocked=0, blocked_reasons=[])
    bc = BondChanges(formed=((0, 1),), broken=())
    cfg = _make_cfg()
    out = screen_all_trials(
        pl, mol, bc, cfg,
        backend="lj", screening_model="", workers=1,
        relax_fmax=0.5, traj_stride=5, seed=0,
    )
    assert out == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_screening.py -v`
Expected: FAIL — `reactx.screening` not importable

- [ ] **Step 3: Implement reactx/screening.py**

```python
"""Stage 1: artificial-force relax of all placement survivors (parallel).

Each placement trial is relaxed independently under Hookean / PullApart
restraints, producing a (frames, energies, peak_energy, reached_product)
record packed in ScreeningTrialResult. Pool-parallel via reactx.parallel
when workers >= 2.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from rdkit import Chem

from reactx.artificial_force import build_restraints
from reactx.bond_changes import BondChanges
from reactx.calculators import make_calculator
from reactx.config import ReactionConfig, resolve_r_form_targets
from reactx.parallel import (
    get_cached_calculator,
    init_lj_worker,
    init_uma_worker,
    run_with_pool,
)
from reactx.path_relax import relax_with_restraints
from reactx.placement import PlacementResult, PlacementTrial, build_atoms_from_positions
from reactx.scoring import ScreeningTrialResult, reached_product

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ScreenJob:
    """Pickle-friendly per-trial input for the worker function."""
    trial_idx: int
    direction: np.ndarray
    positions: np.ndarray
    syms: list[str]
    charges: list[int]
    formed: list[tuple[int, int]]
    broken: list[tuple[int, int]]
    r_form_targets: list[float]
    r_broken: float
    k_form: float
    k_broken: float
    max_relax_steps: int
    relax_fmax: float
    traj_stride: int
    backend: str
    screening_model: str


def screen_all_trials(
    placement: PlacementResult,
    mol_h: Chem.Mol,
    bond_changes: BondChanges,
    cfg: ReactionConfig,
    *,
    backend: str,
    screening_model: str,
    workers: int,
    relax_fmax: float,
    traj_stride: int,
    seed: int,
) -> list[ScreeningTrialResult]:
    """Run artificial-force relax for every placement trial; return per-trial results.

    workers >= 2 -> spawn Pool; each worker loads one screening calculator.
    workers == 1 -> in-process; calculator created once via init_*_worker.
    """
    if not placement.trials:
        return []

    syms = [a.GetSymbol() for a in mol_h.GetAtoms()]
    charges = [a.GetFormalCharge() for a in mol_h.GetAtoms()]
    formed_pairs = list(bond_changes.formed)
    broken_pairs = list(bond_changes.broken)
    r_form_targets = resolve_r_form_targets(cfg, syms, formed_pairs)

    jobs = [
        _ScreenJob(
            trial_idx=i,
            direction=t.direction,
            positions=t.positions,
            syms=syms,
            charges=charges,
            formed=formed_pairs,
            broken=broken_pairs,
            r_form_targets=r_form_targets,
            r_broken=cfg.restraints.r_broken,
            k_form=cfg.restraints.k_form,
            k_broken=cfg.restraints.k_broken,
            max_relax_steps=cfg.restraints.max_relax_steps,
            relax_fmax=relax_fmax,
            traj_stride=traj_stride,
            backend=backend,
            screening_model=screening_model,
        )
        for i, t in enumerate(placement.trials)
    ]

    initializer, initargs = _select_initializer(backend, screening_model)
    return run_with_pool(
        _run_one_screen, jobs,
        workers=workers, initializer=initializer, initargs=initargs,
    )


def _select_initializer(
    backend: str, screening_model: str,
) -> tuple[callable, tuple]:
    if backend == "uma":
        return init_uma_worker, (screening_model,)
    if backend == "lj":
        return init_lj_worker, ()
    raise ValueError(f"unsupported backend: {backend!r}")


def _run_one_screen(job: _ScreenJob) -> ScreeningTrialResult:
    """Execute one trial's relax. Runs in worker process when pool>1."""
    try:
        calc = get_cached_calculator()
    except RuntimeError:
        # workers=1 path didn't run initializer (run_with_pool sequential branch
        # invokes initializer when present; defensive fallback for direct calls)
        calc = make_calculator(
            job.backend, **({"model_name": job.screening_model} if job.backend == "uma" else {}),
        )

    atoms_init = Atoms(symbols=job.syms, positions=job.positions)
    atoms_init.set_initial_charges(job.charges)
    atoms_init.info["charge"] = int(sum(job.charges))
    atoms_init.info["spin"] = 1
    restraints = build_restraints(
        atoms_init,
        formed=job.formed, broken=job.broken,
        r_form=job.r_form_targets[0] if job.r_form_targets else None,
        r_broken=job.r_broken,
        k_form=job.k_form, k_broken=job.k_broken,
    )

    try:
        frames, energies = relax_with_restraints(
            atoms_init, restraints, calc,
            max_steps=job.max_relax_steps,
            fmax=job.relax_fmax,
            traj_stride=job.traj_stride,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("trial %d relax failed: %s: %s",
                    job.trial_idx, type(exc).__name__, exc)
        return ScreeningTrialResult(
            trial_idx=job.trial_idx, direction=job.direction,
            frames=[], energies=[], reached_product=False,
            peak_energy=float("inf"), n_steps=0,
            error=f"{type(exc).__name__}: {exc}",
        )

    ok = reached_product(
        frames[-1], formed=job.formed, broken=job.broken,
        r_form_targets=job.r_form_targets,
        r_broken_target=job.r_broken,
    )
    peak = max(energies) if energies else float("inf")
    return ScreeningTrialResult(
        trial_idx=job.trial_idx, direction=job.direction,
        frames=frames, energies=energies,
        reached_product=ok, peak_energy=float(peak),
        n_steps=len(frames),
        error=None,
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_screening.py -v`
Expected: PASS (3/3)

- [ ] **Step 5: Commit**

```bash
git add reactx/screening.py tests/test_screening.py
git commit -m "feat(screening): add Stage 1 parallel artificial-force orchestration"
```

---

## Task 7: Add run_neb_for_trial / run_neb_top_k to reactx/neb.py

**Files:**
- Modify: `reactx/neb.py`
- Test: `tests/test_neb_top_k.py`

- [ ] **Step 1: Write failing slow test in tests/test_neb_top_k.py**

```python
"""Stage 2 NEB orchestration: run top-K NEB jobs (slow, requires UMA)."""
from __future__ import annotations

from pathlib import Path

import pytest

# imports kept module-local to keep the file importable without UMA at collect time


@pytest.mark.slow
def test_run_neb_top_k_sn2_workers_1(tmp_path: Path):
    from reactx.neb import run_neb_top_k
    from reactx.cli import _build_screening_results_for_test  # see helper note below

    # Helper builds a minimal 1-trial screening result for SN2; see Task 8.
    top_k = _build_screening_results_for_test("sn2", n_results=1)
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
```

テスト fixture は `tests/conftest.py` に直接実装する (cli.py の本体には触らない、テスト側で完結させる方針):

`tests/conftest.py` 末尾に追記:

```python
@pytest.fixture()
def screening_result_factory(examples_dir, tmp_path):
    """Build a minimal ScreeningTrialResult list for a given example reaction.

    Runs the placement + 1 short artificial-force relax under LJ to produce
    realistic frames quickly (no UMA load).
    """
    def _factory(stem: str, n_results: int = 1):
        from dataclasses import replace
        from reactx.bond_changes import BondChanges
        from reactx.config import load_config
        from reactx.embed3d import embed_fragments_to_positions
        from reactx.placement import valid_placements
        from reactx.rxn_parser import atom_map_to_reactant_idx, parse_rxn
        from reactx.screening import screen_all_trials

        rxn_path = examples_dir / f"{stem}.rxn"
        cfg = load_config(rxn_path)
        cfg = replace(
            cfg,
            sampling=replace(cfg.sampling, n_candidates=n_results),
            parallel=replace(cfg.parallel, screening_workers=1),
        )
        r_mol, _, _ = parse_rxn(rxn_path)
        bond_changes = BondChanges.from_atom_map_pairs(
            formed_map=cfg.formed,
            broken_map=cfg.broken,
            atom_map_to_idx=atom_map_to_reactant_idx(r_mol),
        )
        mol_h, frag_indices, positions = embed_fragments_to_positions(
            r_mol, seed=0,
        )
        placement = valid_placements(
            mol_h, frag_indices, positions, bond_changes,
            n_candidates=cfg.sampling.n_candidates, seed=0,
        )
        return screen_all_trials(
            placement, mol_h, bond_changes, cfg,
            backend="lj", screening_model="",
            workers=1, relax_fmax=0.5, traj_stride=5, seed=0,
        )
    return _factory
```

(slow test 1 件のために fixture を一つ生やす。本体コードは触らない。)

`tests/test_neb_top_k.py` を上の fixture を使う形に書き換える:

```python
@pytest.mark.slow
def test_run_neb_top_k_sn2_workers_1(tmp_path, screening_result_factory):
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
    assert (tmp_path / out[0]["xyz_path"]).exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_neb_top_k.py -v -m slow --collect-only`
Expected: collection error (`run_neb_top_k` 未定義) — unit test レベルで非 slow 部分は無いので collect-only 確認に留める

- [ ] **Step 3: Implement run_neb_for_trial / run_neb_top_k in reactx/neb.py**

`reactx/neb.py` 末尾に追加:

```python
from dataclasses import dataclass

from reactx.endpoints import relax_endpoint
from reactx.parallel import (
    get_cached_calculator,
    init_lj_worker,
    init_uma_worker,
    run_with_pool,
)


@dataclass(frozen=True)
class _NebJob:
    """Pickle-friendly per-trial input for NEB worker."""
    trial_idx: int
    reactant_atoms: Atoms
    product_atoms: Atoms
    n_images: int
    fmax: float
    max_steps: int
    pad_frames: int
    output_xyz: Path
    backend: str
    neb_model: str


def run_neb_for_trial(
    reactant_atoms: Atoms,
    product_atoms: Atoms,
    *,
    calc: Calculator,
    n_images: int,
    fmax: float,
    max_steps: int,
    pad_frames: int,
    output_xyz: Path,
) -> dict:
    """Run un-restrained endpoint relax + 2-phase NEB for one trial.

    Returns a dict {trial_idx, converged, n_images, image_energies, peak_energy,
    final_fmax, xyz_path}. xyz_path is the basename (relative to output_xyz parent).
    """
    R = relax_endpoint(reactant_atoms, calc, fmax=0.05, max_steps=50)
    P = relax_endpoint(product_atoms, calc, fmax=0.05, max_steps=50)
    info = run_neb(
        reactant=R, product=P, calculator=calc,
        n_images=n_images, output_xyz=output_xyz,
        fmax=fmax, max_steps=max_steps, pad_frames=pad_frames,
    )
    image_energies = info.get("image_energies") or []
    peak = max((e for e in image_energies if e == e), default=float("nan"))
    return {
        "converged": info.get("converged", False),
        "n_images": info.get("n_images", n_images),
        "image_energies": image_energies,
        "peak_energy": float(peak) if peak == peak else float("nan"),
        "final_fmax": info.get("final_fmax", float("nan")),
        "xyz_path": output_xyz.name,
    }


def _run_one_neb(job: _NebJob) -> dict:
    try:
        calc = get_cached_calculator()
    except RuntimeError:
        from reactx.calculators import make_calculator
        calc = make_calculator(
            job.backend,
            **({"model_name": job.neb_model} if job.backend == "uma" else {}),
        )
    result = run_neb_for_trial(
        job.reactant_atoms, job.product_atoms, calc=calc,
        n_images=job.n_images, fmax=job.fmax, max_steps=job.max_steps,
        pad_frames=job.pad_frames, output_xyz=job.output_xyz,
    )
    result["trial_idx"] = job.trial_idx
    return result


def run_neb_top_k(
    top_k_results: list,           # list[ScreeningTrialResult]
    *,
    backend: str,
    neb_model: str,
    workers: int,
    n_images: int,
    fmax: float,
    max_steps: int,
    pad_frames: int,
    output_dir: Path,
) -> list[dict]:
    """Run NEB on each top-K screening result, parallel via Pool when workers>=2.

    Returns list of NEB result dicts in the same order as top_k_results
    (trial_idx field preserved). NEB exceptions for any one trial are logged
    and that trial is dropped from the result list.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[_NebJob] = []
    for r in top_k_results:
        if not r.frames:
            log.warning("skipping NEB for trial %d: no frames", r.trial_idx)
            continue
        out_xyz = output_dir / f"trajectory_neb_trial_{r.trial_idx}.xyz"
        jobs.append(_NebJob(
            trial_idx=r.trial_idx,
            reactant_atoms=r.frames[0],
            product_atoms=r.frames[-1],
            n_images=n_images, fmax=fmax, max_steps=max_steps,
            pad_frames=pad_frames, output_xyz=out_xyz,
            backend=backend, neb_model=neb_model,
        ))
    if not jobs:
        return []

    if backend == "uma":
        initializer, initargs = init_uma_worker, (neb_model,)
    elif backend == "lj":
        initializer, initargs = init_lj_worker, ()
    else:
        raise ValueError(f"unsupported backend: {backend!r}")

    raw = run_with_pool(
        _run_one_neb, jobs,
        workers=workers, initializer=initializer, initargs=initargs,
    )
    # Drop None entries (NEB-level exceptions caught upstream return dict;
    # use a marker if downstream chooses to mark error). Currently every
    # _run_one_neb returns a dict; if a worker process crashed Pool would
    # raise — that propagates to the CLI which treats it as fatal.
    return raw
```

ファイル先頭の import を整理:

```python
from pathlib import Path
```

(既にあれば skip)

- [ ] **Step 4: Run unit tests (non-slow)**

Run: `pytest tests/ -v -k "not slow"`
Expected: 既存テストすべて PASS、NEB 系の slow テストは未実行

- [ ] **Step 5: Commit**

```bash
git add reactx/neb.py tests/test_neb_top_k.py tests/conftest.py
git commit -m "feat(neb): add run_neb_for_trial / run_neb_top_k for Stage 2 orchestration"
```

---

## Task 8: Rewrite reactx/cli.py orchestration (Stage 1 → top-K → Stage 2)

**Files:**
- Modify: `reactx/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_cli_unimolecular.py`
- Delete: `tests/test_cli_neb_refine_guard.py`
- Delete: `tests/test_neb_refine_sn2.py`

- [ ] **Step 1: Write the new tests for tests/test_cli.py**

既存 `test_cli.py` の内容を確認 (`git show HEAD:tests/test_cli.py | head -100`) し、`--neb-refine` / `--neb-images` / `--model` を使っているテストを削除。代わりに以下を追加:

```python
def test_cli_phase8_meta_json_schema(tmp_path: Path, tmp_rxn_with_toml):
    """meta.json contains screening_trials, top_k_indices, neb_results."""
    rxn = tmp_rxn_with_toml("sn2", toml_body=
        'description = "sn2 lj fast"\n'
        'formed = [[1, 3]]\n'
        'broken = [[1, 2]]\n'
        '[restraints]\nk_form=0.1\nk_broken=0.1\nr_broken=4.0\nmax_relax_steps=3\n'
        '[sampling]\nn_candidates=2\n'
        '[neb]\ntop_k=1\nn_images=3\nmax_steps=2\n'
        '[parallel]\nscreening_workers=1\nneb_workers=1\n'
    )
    out = tmp_path / "sn2"
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "lj"])
    # LJ backend で UMA-quality NEB は出ないが、cli 経路全体が完走することを確認
    assert rc == 0
    meta = json.loads((out / "meta.json").read_text())
    assert "screening_trials" in meta
    assert "top_k_indices" in meta
    assert "neb_results" in meta
    assert meta["effective_params"]["screening_model"]
    assert meta["effective_params"]["neb_model"]
    assert isinstance(meta["wall_clock_breakdown"], dict)
    assert {"placement", "screening", "neb"} <= set(meta["wall_clock_breakdown"])
```

`tests/test_cli_unimolecular.py` の既存テストを更新 (top_k=1 自動 clamp の確認):

```python
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
```

- [ ] **Step 2: Delete obsolete tests**

```bash
git rm tests/test_neb_refine_sn2.py tests/test_cli_neb_refine_guard.py
```

- [ ] **Step 3: Rewrite reactx/cli.py**

```python
"""CLI entry point: reactx run <rxn> -o <outdir> [options]."""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

from ase.io import read, write
from rdkit import Chem

from reactx.bond_changes import BondChanges
from reactx.config import ReactionConfig, load_config, resolve_r_form_targets
from reactx.embed3d import embed_fragments_to_positions
from reactx.neb import run_neb_top_k
from reactx.placement import (
    PlacementResult,
    valid_placements,
)
from reactx.rxn_parser import atom_map_to_reactant_idx, parse_rxn
from reactx.screening import screen_all_trials
from reactx.scoring import ScreeningTrialResult, top_k_trials

log = logging.getLogger("reactx")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="reactx")
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="Run full pipeline on a .rxn file (with sidecar .rxn.toml)")
    run.add_argument("rxn_path", type=Path)
    run.add_argument("-o", "--output", type=Path, required=True)
    run.add_argument("--backend", choices=["uma", "lj"], default="uma")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--relax-fmax", type=float, default=0.1)
    run.add_argument("--traj-stride", type=int, default=5)
    run.add_argument("--render", action="store_true",
                     help="Also invoke blender/render.py after pipeline")
    run.add_argument("--blender-exe", type=str, default="blender")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return _cmd_run(args)


def _configure_reactx_logging() -> None:
    if log.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[reactx] %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False


def _sanitize_for_json(obj):
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _check_hf_auth() -> int:
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import LocalTokenNotFoundError
    except ImportError as exc:
        log.error("UMA backend requires huggingface_hub: %s", exc)
        return 1
    try:
        HfApi().whoami()
    except LocalTokenNotFoundError:
        log.error(
            "Hugging Face token not found. Run `hf auth login` first "
            "(UMA models are gated and require an authorized account)."
        )
        return 1
    except Exception as exc:  # noqa: BLE001
        log.error(
            "Hugging Face authentication check failed (%s: %s). "
            "Run `hf auth login` and ensure UMA model access is approved.",
            type(exc).__name__, exc,
        )
        return 1
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    _configure_reactx_logging()

    if not args.rxn_path.exists():
        log.error("Error: .rxn not found: %s", args.rxn_path)
        return 1
    try:
        cfg = load_config(args.rxn_path)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 2
    except ValueError as exc:
        log.error("Invalid sidecar TOML: %s", exc)
        return 2

    if args.backend == "uma":
        rc = _check_hf_auth()
        if rc != 0:
            return rc

    args.output.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()
    breakdown: dict[str, float] = {}

    r_mol, _, _ = parse_rxn(args.rxn_path)
    r_h = Chem.AddHs(r_mol)
    try:
        bond_changes = BondChanges.from_atom_map_pairs(
            formed_map=cfg.formed,
            broken_map=cfg.broken,
            atom_map_to_idx=atom_map_to_reactant_idx(r_mol),
        )
    except (KeyError, ValueError) as exc:
        log.error("Invalid bond_changes: %s", exc)
        return 2

    formed_pairs = list(bond_changes.formed)
    syms_r = [a.GetSymbol() for a in r_h.GetAtoms()]
    r_form_targets = resolve_r_form_targets(cfg, syms_r, formed_pairs)
    log.info(
        "description=%s effective: k_form=%.2f k_broken=%.2f r_broken=%.2f "
        "max_relax_steps=%d r_form_targets=%s",
        cfg.description,
        cfg.restraints.k_form, cfg.restraints.k_broken,
        cfg.restraints.r_broken, cfg.restraints.max_relax_steps,
        [f"{x:.3f}" for x in r_form_targets] if r_form_targets else "[]",
    )

    # --- placement
    t_placement = time.monotonic()
    n_frags_reactant = len(Chem.GetMolFrags(r_h))
    effective_n_candidates = cfg.sampling.n_candidates
    if n_frags_reactant == 1 and effective_n_candidates > 1:
        log.info("unimolecular reaction; n_candidates clamped %d -> 1",
                 effective_n_candidates)
        effective_n_candidates = 1
    try:
        mol_h_r, frag_indices_r, base_positions = embed_fragments_to_positions(
            r_mol, seed=args.seed,
        )
    except RuntimeError as exc:
        log.error("per-fragment embed failed: %s", exc)
        return 1
    try:
        placement = valid_placements(
            mol_h_r, frag_indices_r, base_positions, bond_changes,
            n_candidates=effective_n_candidates, seed=args.seed,
        )
    except NotImplementedError as exc:
        log.error("placement not supported: %s", exc)
        return 2
    except (RuntimeError, ValueError) as exc:
        log.error("placement failed: %s", exc)
        return 1
    log.info("placement: %d/%d survived (%d blocked)",
             len(placement.trials), placement.n_candidates, placement.n_blocked)
    breakdown["placement"] = time.monotonic() - t_placement

    # --- Stage 1: screening
    t_screen = time.monotonic()
    screen_results = screen_all_trials(
        placement, mol_h_r, bond_changes, cfg,
        backend=args.backend,
        screening_model=cfg.model.screening_model,
        workers=cfg.parallel.screening_workers,
        relax_fmax=args.relax_fmax,
        traj_stride=args.traj_stride,
        seed=args.seed,
    )
    breakdown["screening"] = time.monotonic() - t_screen

    if not any(r.frames for r in screen_results):
        log.error("All screening trials failed; see meta.json")
        return _write_outputs(
            args, cfg, screen_results, top_k_indices=[], neb_results=[],
            placement=placement, t_start=t_start, breakdown=breakdown,
            r_form_targets=r_form_targets, rc=1,
        )

    # --- top-K
    top_k_list = top_k_trials(screen_results, cfg.neb.top_k)
    top_k_indices = [r.trial_idx for r in top_k_list]
    log.info("top-%d trials by screening score: %s",
             cfg.neb.top_k, top_k_indices)

    # --- Stage 2: NEB
    t_neb = time.monotonic()
    try:
        neb_results = run_neb_top_k(
            top_k_list,
            backend=args.backend,
            neb_model=cfg.model.neb_model,
            workers=cfg.parallel.neb_workers,
            n_images=cfg.neb.n_images,
            fmax=cfg.neb.fmax,
            max_steps=cfg.neb.max_steps,
            pad_frames=cfg.neb.pad_frames,
            output_dir=args.output,
        )
    except Exception as exc:  # noqa: BLE001
        log.error("Stage 2 NEB failed: %s: %s", type(exc).__name__, exc)
        return _write_outputs(
            args, cfg, screen_results, top_k_indices=top_k_indices, neb_results=[],
            placement=placement, t_start=t_start, breakdown=breakdown,
            r_form_targets=r_form_targets, rc=1,
        )
    breakdown["neb"] = time.monotonic() - t_neb

    if not neb_results:
        log.error("All Stage 2 NEB jobs failed; see meta.json")
        return _write_outputs(
            args, cfg, screen_results, top_k_indices=top_k_indices, neb_results=[],
            placement=placement, t_start=t_start, breakdown=breakdown,
            r_form_targets=r_form_targets, rc=1,
        )

    final = min(neb_results, key=lambda r: r["peak_energy"])
    final_xyz_src = args.output / final["xyz_path"]
    final_frames = read(str(final_xyz_src), index=":")
    final_xyz = args.output / "trajectory.xyz"
    write(str(final_xyz), final_frames, format="extxyz")
    log.info("selected trial=%d (NEB peak=%.4f)",
             final["trial_idx"], final["peak_energy"])

    rc = _write_outputs(
        args, cfg, screen_results, top_k_indices=top_k_indices,
        neb_results=neb_results,
        placement=placement, t_start=t_start, breakdown=breakdown,
        r_form_targets=r_form_targets, rc=0, selected_trial=final["trial_idx"],
    )
    if rc != 0:
        return rc

    if args.render:
        rc_render = _invoke_blender(args, final_xyz)
        if rc_render != 0:
            return rc_render
    log.info("OK: wrote %s", final_xyz)
    return 0


def _write_outputs(
    args, cfg: ReactionConfig,
    screen_results: list[ScreeningTrialResult],
    *,
    top_k_indices: list[int],
    neb_results: list[dict],
    placement: PlacementResult | None,
    t_start: float,
    breakdown: dict[str, float],
    r_form_targets: list[float],
    rc: int,
    selected_trial: int = -1,
) -> int:
    placement_meta = (
        {"n_candidates": placement.n_candidates,
         "n_blocked": placement.n_blocked,
         "n_valid": len(placement.trials)}
        if placement is not None
        else {"n_candidates": 0, "n_blocked": 0, "n_valid": 0}
    )
    meta = {
        "backend": args.backend,
        "description": cfg.description,
        "selected_trial": selected_trial,
        "placement": placement_meta,
        "screening_trials": [
            {
                "trial_idx": r.trial_idx,
                "reached_product": r.reached_product,
                "peak_energy": float(r.peak_energy)
                    if math.isfinite(r.peak_energy) else None,
                "n_steps": r.n_steps,
                "direction": [float(x) for x in r.direction],
                "error": r.error,
            }
            for r in screen_results
        ],
        "top_k_indices": list(top_k_indices),
        "neb_results": [
            {k: v for k, v in res.items() if k != "image_energies" or True}
            for res in neb_results
        ],
        "wall_clock_seconds": float(time.monotonic() - t_start),
        "wall_clock_breakdown": {k: float(v) for k, v in breakdown.items()},
        "effective_params": {
            "k_form": cfg.restraints.k_form,
            "k_broken": cfg.restraints.k_broken,
            "r_broken": cfg.restraints.r_broken,
            "max_relax_steps": cfg.restraints.max_relax_steps,
            "r_form_targets": list(r_form_targets) if r_form_targets else [],
            "n_candidates": cfg.sampling.n_candidates,
            "screening_model": cfg.model.screening_model,
            "neb_model": cfg.model.neb_model,
            "top_k": cfg.neb.top_k,
            "n_images": cfg.neb.n_images,
            "screening_workers": cfg.parallel.screening_workers,
            "neb_workers": cfg.parallel.neb_workers,
        },
    }
    (args.output / "meta.json").write_text(
        json.dumps(_sanitize_for_json(meta), indent=2)
    )
    if neb_results:
        best = min(neb_results, key=lambda r: r["peak_energy"])
        (args.output / "energies.json").write_text(json.dumps(best["image_energies"]))
    return rc


def _invoke_blender(args, xyz: Path) -> int:
    import shutil
    import subprocess
    if shutil.which(args.blender_exe) is None:
        log.error("Blender executable not found on PATH: %s", args.blender_exe)
        return 1
    script = Path(__file__).resolve().parent.parent / "blender" / "render.py"
    blend = args.output / "scene.blend"
    cmd = [args.blender_exe, "--background", "--python", str(script),
           "--", str(xyz), str(blend)]
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        log.error("blender exited with code %d", result.returncode)
        return 1
    return 0
```

- [ ] **Step 4: Run all non-slow tests**

Run: `pytest tests/ -v -k "not slow"`
Expected: ALL PASS (config / scoring / parallel / endpoints / screening / cli unit)

- [ ] **Step 5: Commit**

```bash
git add reactx/cli.py tests/test_cli.py tests/test_cli_unimolecular.py
git rm tests/test_neb_refine_sn2.py tests/test_cli_neb_refine_guard.py
git commit -m "refactor(cli): rewrite for Stage1->top-K->Stage2 CI-NEB pipeline"
```

---

## Task 9: Delete reactx/align.py and tests/test_align.py

**Files:**
- Delete: `reactx/align.py`
- Delete: `tests/test_align.py`

- [ ] **Step 1: Verify no remaining imports**

Run: `git grep -n "from reactx.align\|import reactx.align\|align_product_to_reactant" -- reactx tests`
Expected: 0 hits (Task 8 で全削除済み)。残っていれば Task 8 漏れなので戻って修正する。

Run: `git grep -n "heavy_to_hydrogen_groups" -- reactx tests`
Expected: `reactx/rxn_parser.py` の定義のみ。Task 8 後は呼び出し側が消えている。残れば dead code として削除可 (オプション、Step 5 で扱う)。

- [ ] **Step 2: Delete files**

```bash
git rm reactx/align.py tests/test_align.py
```

- [ ] **Step 3: Run all non-slow tests**

Run: `pytest tests/ -v -k "not slow"`
Expected: ALL PASS

- [ ] **Step 4: (optional) Remove dead heavy_to_hydrogen_groups**

`reactx/rxn_parser.py` から `heavy_to_hydrogen_groups` を削除 (Task 8 後に唯一の呼び出し元 cli.py が消えるため):

```bash
# rxn_parser.py から関数定義を削除 (Step 1 で残存呼び出しが無いことを確認済み)
```

Run: `pytest tests/ -v -k "not slow"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add reactx/rxn_parser.py
git commit -m "chore(align,rxn_parser): remove align module + heavy_to_hydrogen_groups (Phase 8 endpoints come from trajectory)"
```

---

## Task 10: Migrate slow integration tests to Phase 8 schema

**Files:**
- Modify: `tests/test_re1_sn2.py`
- Modify: `tests/test_re1_proton_transfer.py`
- Modify: `tests/test_re1_menshutkin.py`
- Modify: `tests/test_re3_e2.py`
- Modify: `tests/test_re3_sn1_dissoc.py`
- Modify: `tests/test_re4_sn1_recomb.py`
- Modify: `tests/test_examples_menshutkin.py`
- Modify: `tests/test_wallclock_sn2.py`

- [ ] **Step 1: Update tests/test_re1_sn2.py**

```python
"""SN2 end-to-end with CI-NEB. Slow, requires UMA."""
import json
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

from reactx.cli import main


_SN2_FAST = """\
description = "SN2 fast"
formed = [[1, 3]]
broken = [[1, 2]]
[restraints]
k_form = 0.5
k_broken = 1.0
r_broken = 4.0
max_relax_steps = 50
[sampling]
n_candidates = 8
[neb]
top_k = 2
n_images = 5
max_steps = 30
[parallel]
screening_workers = 2
neb_workers = 2
"""


@pytest.mark.slow
def test_re1_sn2_end_to_end_with_cineb(tmp_path: Path, tmp_rxn_with_toml):
    rxn = tmp_rxn_with_toml("sn2", toml_body=_SN2_FAST)
    out = tmp_path / "sn2"
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "uma"])
    assert rc == 0

    meta = json.loads((out / "meta.json").read_text())
    assert meta["description"] == "SN2 fast"
    assert meta["selected_trial"] >= 0
    assert len(meta["screening_trials"]) >= 1
    assert len(meta["top_k_indices"]) >= 1
    assert len(meta["neb_results"]) >= 1
    assert all("peak_energy" in r for r in meta["neb_results"])
    assert meta["effective_params"]["screening_model"]
    assert meta["effective_params"]["neb_model"]

    frames = read(str(out / "trajectory.xyz"), index=":")
    # CI-NEB images count = n_images + 2 * pad_frames (default pad=0)
    assert len(frames) == 5

    syms = frames[0].get_chemical_symbols()
    c_idx = syms.index("C")
    cl_idx = syms.index("Cl")
    o_idx = syms.index("O")
    angles = []
    for f in frames:
        v_co = f.positions[o_idx] - f.positions[c_idx]
        v_ccl = f.positions[cl_idx] - f.positions[c_idx]
        cos_t = float(np.dot(v_co, v_ccl) / (
            np.linalg.norm(v_co) * np.linalg.norm(v_ccl)
        ))
        angles.append(float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0)))))
    assert max(angles) >= 120.0

    d_co_first = frames[0].get_distance(c_idx, o_idx)
    d_co_last = frames[-1].get_distance(c_idx, o_idx)
    d_ccl_first = frames[0].get_distance(c_idx, cl_idx)
    d_ccl_last = frames[-1].get_distance(c_idx, cl_idx)
    assert d_co_last < d_co_first - 0.5
    assert d_ccl_last > d_ccl_first + 0.5
```

- [ ] **Step 2: Update tests/test_re1_proton_transfer.py / tests/test_re1_menshutkin.py / tests/test_re3_e2.py / tests/test_re3_sn1_dissoc.py / tests/test_re4_sn1_recomb.py / tests/test_examples_menshutkin.py**

各ファイルで:
1. TOML body に `[neb]` `[parallel]` セクション追記 (sn1_dissoc は `top_k=1`)
2. `meta["selected_trial"]` の前後で `meta["screening_trials"]` / `meta["top_k_indices"]` / `meta["neb_results"]` を assert
3. trajectory.xyz の frames 数を `cfg.neb.n_images + 2 * pad_frames` に書き換え
4. 既存の geometric assert (距離 / 角度) はそのまま維持

例: `tests/test_re3_sn1_dissoc.py`:

```python
_SN1_DISSOC_FAST = """\
description = "SN1 dissoc fast"
formed = []
broken = [[1, 5]]
[restraints]
k_form = 0.0
k_broken = 2.0
r_broken = 6.0
max_relax_steps = 100
[sampling]
n_candidates = 1
[neb]
top_k = 1
n_images = 5
max_steps = 30
[parallel]
screening_workers = 1
neb_workers = 1
"""

@pytest.mark.slow
def test_re3_sn1_dissoc_end_to_end_with_cineb(...):
    ...
    assert len(meta["neb_results"]) == 1
    assert len(meta["top_k_indices"]) == 1
    frames = read(str(out / "trajectory.xyz"), index=":")
    assert len(frames) == 5
    # 既存の C-Br 距離 assert を維持
```

- [ ] **Step 3: Update tests/test_wallclock_sn2.py**

しきい値を Phase 8 の見積もり (~5-10 分) に合わせて緩和:

```python
@pytest.mark.slow
def test_wallclock_sn2_under_900s(tmp_path, tmp_rxn_with_toml):
    rxn = tmp_rxn_with_toml("sn2", toml_body=_SN2_FAST_FOR_WALLCLOCK)
    out = tmp_path / "sn2"
    t0 = time.monotonic()
    rc = main(["run", str(rxn), "-o", str(out), "--backend", "uma"])
    elapsed = time.monotonic() - t0
    assert rc == 0
    # Phase 8 SN2 は CI-NEB 経由で ~5-10 分 (5070 Ti + uma-s-1p2)。
    # CI 的な余裕として 15 分 (900 s) を上限。
    assert elapsed < 900
```

- [ ] **Step 4: Run slow integration tests (requires UMA)**

Run: `pytest -m slow tests/test_re1_sn2.py -v`
Expected: PASS (CI-NEB が走り peak_energy が finite)

他反応も順次:
```
pytest -m slow tests/test_re1_proton_transfer.py
pytest -m slow tests/test_re1_menshutkin.py
pytest -m slow tests/test_re3_e2.py
pytest -m slow tests/test_re3_sn1_dissoc.py
pytest -m slow tests/test_re4_sn1_recomb.py
pytest -m slow tests/test_examples_menshutkin.py
pytest -m slow tests/test_wallclock_sn2.py
```

すべて PASS することを確認。失敗があれば該当反応の TOML default を再調整 (k_form / max_relax_steps / pad_frames など)。

- [ ] **Step 5: Commit**

```bash
git add tests/test_re1_*.py tests/test_re3_*.py tests/test_re4_*.py \
        tests/test_examples_menshutkin.py tests/test_wallclock_sn2.py
git commit -m "test(slow): migrate integration tests to Phase 8 CI-NEB schema"
```

---

## Task 11: Update example .rxn.toml files (only where defaults are insufficient)

**Files:**
- Modify: `examples/sn1_dissoc.rxn.toml`
- Modify: `examples/menshutkin.rxn.toml`

- [ ] **Step 1: Update examples/sn1_dissoc.rxn.toml**

`[neb] top_k = 1` を明示 (default 4 だと unimolecular で混乱する):

```toml
description = "SN1 step 1: tBuBr -> tBu+ + Br- (dissociation)"
formed = []
broken = [[1, 5]]

[restraints]
k_form = 0.0
k_broken = 2.0
r_broken = 6.0
max_relax_steps = 200

[sampling]
n_candidates = 1

[neb]
top_k = 1
```

- [ ] **Step 2: Update examples/menshutkin.rxn.toml**

`[neb] pad_frames = 3` を追加 (Menshutkin は image 数が短いと render 視認性が落ちるため):

```toml
description = "Menshutkin: NH3 + CH3Cl -> NH3CH3+ Cl- (neutral -> ion pair)"
formed = [[1, 5]]
broken = [[5, 9]]

[restraints]
k_form = 2.0
k_broken = 2.0
r_broken = 5.0
max_relax_steps = 200

[neb]
pad_frames = 3
```

- [ ] **Step 3: Run slow tests for these reactions**

Run: `pytest -m slow tests/test_re3_sn1_dissoc.py tests/test_re1_menshutkin.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add examples/sn1_dissoc.rxn.toml examples/menshutkin.rxn.toml
git commit -m "chore(examples): add [neb] overrides for sn1_dissoc and menshutkin"
```

---

## Task 12: Update README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Rewrite the architecture diagram**

`## アーキテクチャ` セクションの ASCII 図を以下に書き換え:

```
.rxn + .rxn.toml ─> rxn_parser + load_config ─> ReactionConfig
                                                       │
                                                       ▼
                                               BondChanges (formed/broken)
                                                       │
                                                       ▼
                                       embed_fragments_to_positions
                                                       │
                                                       ▼
                                       placement.valid_placements
                                       (4π sr Fibonacci + 角度シャドウ + d_min ceiling)
                                                       │
                                                       ▼
                                  ┌── Stage 1 screening (parallel) ──┐
                                  │  人工力 relax で peak_energy 評価 │
                                  └─────────────┬────────────────────┘
                                                │
                                                ▼
                                      top_k_trials (K=4 default)
                                                │
                                                ▼
                                  ┌── Stage 2 CI-NEB (parallel) ─────┐
                                  │  endpoints relax + warmup + climb │
                                  └─────────────┬────────────────────┘
                                                │
                                                ▼
                                      best (lowest peak_energy)
                                                │
                                                ▼
                                trajectory.xyz → blender/render.py → .blend
```

- [ ] **Step 2: Update 「対応反応」表に CI-NEB 列**

| `.rxn` | description | formed (map) | broken (map) | k_form | k_broken | r_broken | max_relax_steps | r_form | n_candidates | CI-NEB |
|---|---|---|---|---|---|---|---|---|---|---|
| sn2.rxn | SN2 anion | `[[1,3]]` | `[[1,2]]` | 0.5 | 1.0 | 4.0 | 100 | 元素表 | 64 | 全反応で適用 (top_k=4) |
| ... (他反応も同様) ||||||||||| 全反応で適用 |

- [ ] **Step 3: Update 「使い方」セクション**

`--neb-refine` / `--neb-images` / `--model` の例を削除。CLI フラグの一覧を:

```
- `--backend {uma,lj}` (default: `uma`)
- `--seed 0` (sampling 再現性デバッグ)
- `--relax-fmax 0.1`, `--traj-stride 5` (出力品質)
- `--render` + `--blender-exe blender`
```

モデル指定 / NEB パラメータ / 並列度はすべて `[model]` `[neb]` `[parallel]` TOML セクションに移動した旨を明記。

- [ ] **Step 4: Add 「Per-reaction `.rxn.toml` config」 sections for [model] [neb] [parallel]**

最小例 (`examples/sn2.rxn.toml`) のあとに追記:

```toml
# 全部省略可 (default 値が埋まる)
[model]
screening_model = "uma-s-1p2"   # default
neb_model = "uma-s-1p2"         # default; "uma-m-1p1" にすれば NEB 精度↑時間↑

[neb]
top_k = 4                       # screening から NEB に流す trial 数
n_images = 7
fmax = 0.05
max_steps = 200
pad_frames = 0                  # render 視認性のため 1-3 推奨

[parallel]
screening_workers = 3           # GPU VRAM 16 GB / 小モデル想定
neb_workers = 3
```

- [ ] **Step 5: Update 「方針と限界」**

追加項目:

- 反応経路探索は CI-NEB に統一 (Phase 8)。screening 段階の人工力 relax は endpoint 生成器として残置
- ML モデルは default `uma-s-1p2` (小型)、`[model]` セクションで `uma-m-1p1` 等に切り替え可能
- 並列化は trial / NEB job レベルで `multiprocessing.Pool` (spawn 強制); image-level の MPI 並列 NEB は VRAM 制約で非対応
- E2 / SN1 dissoc / SN1 recomb も CI-NEB で本評価される (Phase 7 までの 1+1 限定 guard は撤廃)

- [ ] **Step 6: Run docs lint (optional)**

Run: `git diff README.md | head -200`
README diff を目視確認。

- [ ] **Step 7: Commit**

```bash
git add README.md
git commit -m "docs(readme): rewrite for Phase 8 CI-NEB unified pipeline"
```

---

## Task 13: Re-measure wall-clock and update README table

**Files:**
- Modify: `README.md` (Wall-clock 表のみ)

- [ ] **Step 1: Run all 6 reactions on RTX 5070 Ti**

```bash
for rxn in sn2 proton_transfer menshutkin e2 sn1_dissoc sn1_recomb; do
  rm -rf out/$rxn
  time reactx run examples/$rxn.rxn -o out/$rxn/ --backend uma --render \
    2>&1 | tee out/$rxn.log
done
```

`out/<rxn>/meta.json` から以下を集計:
- `wall_clock_seconds`
- `wall_clock_breakdown.{placement, screening, neb, render}`
- `placement.{n_valid, n_blocked}`
- `len(neb_results)`
- `selected_trial`

- [ ] **Step 2: Update Wall-clock 表**

`README.md` の `## Wall-clock (実測)` セクションを Phase 8 数値で書き換え:

| Reaction | wall-clock | screening | neb | n_valid | top_k | NEB peak (eV) | 備考 |
|---|---|---|---|---|---|---|---|
| SN2 | T s | T_s s | T_n s | N | 4 | E | ... |
| ... ||||||||

(実測後に表を埋める)

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(readme): record Phase 8 wall-clock measurements"
```

---

## Final Verification

- [ ] **Step 1: Run all non-slow tests**

Run: `pytest tests/ -v -k "not slow"`
Expected: PASS, no collection errors

- [ ] **Step 2: Run all slow tests**

Run: `pytest -m slow tests/ -v`
Expected: 全 6 反応 + wall-clock + neb_top_k で PASS

- [ ] **Step 3: Lint**

Run: `ruff check reactx tests`
Expected: clean

Run: `ruff format --check reactx tests`
Expected: clean (or auto-format with `ruff format reactx tests` + commit)

- [ ] **Step 4: Smoke render**

Run: `reactx run examples/sn2.rxn -o out/smoke/ --backend uma --render`
Expected: `out/smoke/scene.blend` が生成される + Blender で開いて Walden 反転が見える

- [ ] **Step 5: Push and open PR**

```bash
git push -u origin phase-8
gh pr create --base develop --title "Phase 8: CI-NEB unified path engine + parallel screening" --body "..."
```

PR body には spec へのリンク (`docs/superpowers/specs/2026-05-05-cineb-unified-design.md`) と 6 反応の wall-clock 表を含める。
