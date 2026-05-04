# reactx Phase 8 — CI-NEB Unified Path Engine + Screening Filter + Parallelization

- Status: Draft (awaiting user spec review)
- Date: 2026-05-05
- Owner: @kam6y
- Branch: `phase-8` (from `develop` after Phase 7 merge)
- 前提仕様: `docs/superpowers/specs/2026-05-04-generic-placement-design.md`
- 後方互換性: **完全に放棄する**。TOML schema・`meta.json`・CLI フラグ・内部 API すべて breaking change を許容。

## 1. 目的

Phase 7 までの reaction path engine は 2 系統が併存している:

- **メイン**: 人工力 (Hookean attraction + PullApart repulsion) を用いた拘束 relax で trajectory を生成。peak energy はその拘束付き経路の最大値であり、真の TS エネルギーではない。
- **オプション**: `--neb-refine` で CI-NEB を 1 formed + 1 broken 反応に限定して追加適用。E2 / SN1 dissoc / SN1 recomb は使えない。

Phase 8 で:

- **CI-NEB を全反応の本評価手法に昇格** させる。最終的な trajectory.xyz と peak energy は CI-NEB が出すものに統一。
- 人工力 relax は **screening 兼 endpoint 生成器** に役割を変更する (廃止しない)。
- 全 placement-survivor を NEB に流すのは非現実的なので、**人工力 relax の peak_energy で top-K=4 に絞ってから CI-NEB を回す**。
- 機械学習ポテンシャル (UMA) は **小型モデル `uma-s-1p2` をデフォルト** にし、screening / NEB のモデルを TOML で個別指定可能にする。
- Screening 段階と NEB 段階を **process pool で並列化** する。RTX 5070 Ti (16 GB VRAM) + 小モデル (~3-4 GB) の前提で同時 3 worker。

## 2. Non-goals

明示的に **このフェーズではやらない** こと:

- **Cycloaddition / metathesis 対応**: Phase 7 で `NotImplementedError` にしている placement の制限 (multi-anchor / fragment-bridging broken bond) は引き続きそのまま。
- **NEB 入力以外への multi-GPU 対応**: 単一 GPU 前提 (5070 Ti)。マルチ GPU 環境への自動分散は将来課題。
- **MPI ベースの ASE parallel NEB (image-level)**: VRAM 制約で 7 images × 3.5 GB が乗らないため不採用。並列化は process-level (trial / NEB job) のみ。
- **大型モデル (`uma-l-*`) 対応**: 5070 Ti では VRAM 超過。CLI 経由で manual 指定はできるが、TOML default や validation は s/m のみ前提。
- **新しい反応型の追加**: 現状の 6 反応 (SN2 / Proton transfer / Menshutkin / E2 / SN1 dissoc / SN1 recomb) を全て CI-NEB 対応にすることがゴール。新反応の検証は別フェーズ。
- **screening の人工力モデル交換**: 拘束は引き続き Hookean + PullApart。restraints の k / r パラメータも現行 TOML を継承。

## 3. アーキテクチャ

### 3.1 新パイプライン

```
.rxn + .rxn.toml
  → rxn_parser + load_config → ReactionConfig
  → BondChanges (formed / broken, 0-indexed)
            │
            ▼
[各 fragment を個別 3D 化] (Phase 7 と同じ)
   embed_fragments_to_positions
            │
            ▼
[Generic placement] (Phase 7 と同じ)
   valid_placements → N 個の PlacementTrial (典型 20-42 件)
            │
            ▼
[Stage 1: Screening — 人工力 relax 全 survivor 並列実行]
   for trial in placement.trials (parallel, screening_workers):
     atoms_init = build_atoms_from_positions(...)
     restraints = build_restraints(...)
     frames, energies = relax_with_restraints(atoms_init, restraints, calc_screen, ...)
     score: peak_energy + reached_product
            │
            ▼
[Top-K 絞り] score_trials の優先順位そのまま、上位 K=4 件だけ通す
            │
            ▼
[Stage 2: NEB — top-K 並列実行]
   for trial in top_k (parallel, neb_workers):
     R_endpoint = relax_no_restraints(frames[0], calc_neb)
     P_endpoint = relax_no_restraints(frames[-1], calc_neb)
     neb_frames, neb_energies = run_neb(R, P, calc_neb, ...)
     final_score: max(neb_energies)
            │
            ▼
[Final select] NEB 後の peak energy 最低 trial を選択
   trajectory.xyz (NEB 経路) / energies.json / meta.json
```

### 3.2 Phase 7 → Phase 8 主要変更

| 項目 | Phase 7 | Phase 8 |
|---|---|---|
| 主スコア | 人工力 relax の peak energy | CI-NEB の peak energy |
| trajectory.xyz の中身 | 人工力 relax の frames | CI-NEB の N images (+ pad) |
| CI-NEB 対応反応 | 1 formed + 1 broken のみ | 全反応 (endpoints は人工力経路から取得) |
| Screening 段階 | 無 (全 survivor を本評価) | 人工力 relax で top-K=4 に絞り |
| 並列化 | 無 (sequential) | screening + NEB を process pool で並列 |
| ML モデル | 単一 (`uma-m-1p1` default) | screening / NEB の 2 系統指定可、default は両方 `uma-s-1p2` |
| `--neb-refine` フラグ | 任意のオプション | **削除**。CI-NEB 常時実行が default。CI-NEB を skip したい場合は `--no-neb` で artificial-force 経路を維持 (テスト用ホットパス) |
| `--neb-images` フラグ | 任意 | **削除**。`[neb] n_images` を TOML 化 |
| `align_product_to_reactant` | 1+1 反応の P endpoint アラインに使用 | **不要・削除** (P endpoint は人工力 trajectory の最終フレーム由来でアトム順序が同じ) |

### 3.3 ファイル構成

新規:
- `reactx/screening.py` — Stage 1 全体を司る関数群 (人工力 relax の並列実行 + 結果集約)
- `reactx/parallel.py` — process pool ヘルパ (worker init で UMA load 1 回、子プロセス間で reuse)
- `reactx/endpoints.py` — NEB 用に R/P endpoints を非拘束で短く relax する helper

改変:
- `reactx/cli.py` — orchestration を Stage 1 → Top-K → Stage 2 に書き換え
- `reactx/config.py` — `[neb]` `[parallel]` `[model]` セクション追加、validation
- `reactx/neb.py` — top-K NEB を並列実行できるよう entry を function-level に整理 (現状の `run_neb` は image-level の中核としてそのまま)
- `reactx/scoring.py` — `score_trials` を `top_k_trials(K)` に拡張 (現行 `score_trials` は `top_k_trials(1)[0]` の wrapper として残す)
- `reactx/calculators.py` — calculator 名 default を `uma-s-1p2` に、make_calculator は単に factory として動作 (load の責任は parallel 側に移管しない)

削除:
- `reactx/align.py` — `align_product_to_reactant` は P endpoint が trajectory 由来になるので不要
- `reactx/calculators.py` の `--neb-refine` 系 CLI 分岐 (cli.py 側のみ)

### 3.4 削除される CLI フラグ

- `--neb-refine` — 削除 (CI-NEB が default になる)
- `--neb-images` — 削除 (`[neb] n_images` で TOML 化)

新規 CLI フラグ:
- `--no-neb` — Stage 2 を skip し、Stage 1 best trial をそのまま `trajectory.xyz` に書き出す。テスト用 / デバッグ用。`meta.json.neb_used = false`。

### 3.5 削除される TOML キー

(`prescreen` / `n_angles` / `cone_half_deg` は Phase 7 で削除済み。Phase 8 では新規追加のみ。)

## 4. コンポーネント詳細

### 4.1 新規モジュール: `reactx/screening.py`

```python
@dataclass(frozen=True)
class ScreeningTrialResult:
    """Stage 1 の各 trial の出力。"""
    trial_idx: int
    direction: np.ndarray
    frames: list[Atoms]              # 人工力 relax の trajectory
    energies: list[float]            # 各 frame の (拘束込み) potential energy
    reached_product: bool
    peak_energy: float
    n_steps: int
    error: str | None = None         # relax 中に例外が出たときの reason、それ以外 None

def screen_all_trials(
    placement: PlacementResult,
    mol_h: Chem.Mol,
    bond_changes: BondChanges,
    cfg: ReactionConfig,
    *,
    backend: str,                    # "uma" | "lj"
    screening_model: str,            # e.g. "uma-s-1p2"
    workers: int,                    # parallel.screening_workers
    relax_fmax: float,
    traj_stride: int,
    seed: int,
) -> list[ScreeningTrialResult]:
    """全 placement-survivor について人工力 relax を並列実行。

    workers >= 2 のときは reactx.parallel.run_with_pool を使い、各 worker で
    1 回だけ calculator を load する pool を立てる。workers == 1 のときは
    in-process sequential 実行 (CI / LJ backend / debug 用)。

    relax 中に例外が出た trial は ScreeningTrialResult.error にメッセージを
    詰めて frames=[]、reached_product=False、peak_energy=inf で返す
    (Phase 7 の cli.py 内のロジックを抽出)。
    """
```

### 4.2 新規モジュール: `reactx/parallel.py`

```python
def run_with_pool(
    fn: Callable[[T], R],            # 各 worker で実行する純関数
    items: list[T],
    *,
    workers: int,
    initializer: Callable[..., None] | None = None,
    initargs: tuple = (),
    context: str = "spawn",          # Windows でも安全な spawn を default
) -> list[R]:
    """multiprocessing.Pool を立てて items を並列処理し、入力順序を維持して返す。

    workers == 1 の場合は Pool を起こさず in-process map 実行 (テスト容易性 +
    LJ backend での無駄なオーバーヘッド回避)。

    initializer は worker プロセス起動時に 1 回だけ呼ばれる (UMA model load
    のような重い初期化を子プロセスごとに 1 回で済ませるため)。
    """

def init_uma_worker(model_name: str, task_name: str = "omol") -> None:
    """worker プロセス起動時に UMA Calculator を 1 回だけ作って module-global
    にキャッシュする。各 trial 関数は get_cached_calculator() で参照する。
    """

def get_cached_calculator() -> Calculator:
    """init_uma_worker でセットされた calculator を取り出す。worker 外で
    呼ばれたら RuntimeError。"""
```

### 4.3 新規モジュール: `reactx/endpoints.py`

```python
def relax_endpoint(
    atoms: Atoms,
    calc: Calculator,
    *,
    fmax: float = 0.05,
    max_steps: int = 50,
) -> Atoms:
    """拘束無しの BFGS relax で endpoint を局所最小に落とす。

    NEB は両端が真の minimum であることを前提にしているが、screening 直後の
    frames[0] / frames[-1] は拘束付き relax の結果なので拘束無しの local min
    ではない。ここで短く再 relax する。

    max_steps=50 は経験則: 拘束を外した状態でも分子が大きく動くケースは
    少ないので 50 step あれば fmax=0.05 で大半が収束する。converge しない
    場合も例外は出さず、その時点の Atoms を返す (NEB は noisy endpoint を
    許容する)。
    """
```

### 4.4 改変: `reactx/neb.py`

`run_neb` 関数自体は Phase 7 のまま (warmup + climb 2-phase) を維持。Stage 2 の orchestration を担う新関数を追加:

```python
def run_neb_for_trial(
    screen_result: ScreeningTrialResult,
    *,
    calc: Calculator,
    n_images: int,
    fmax: float,
    max_steps: int,
    pad_frames: int,
    output_xyz: Path,
) -> dict:
    """単一 screening trial の trajectory から CI-NEB を実行し、
    (peak_energy, image_energies, frames, converged) を返す。

    内部:
      1. R = relax_endpoint(screen_result.frames[0], calc)
      2. P = relax_endpoint(screen_result.frames[-1], calc)
      3. run_neb(R, P, calc, ...) — 既存実装
      4. peak_energy = max(image_energies)
    """

def run_neb_top_k(
    top_k_results: list[ScreeningTrialResult],
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
    """K 件の trial について CI-NEB を並列実行。

    workers >= 2 のときは reactx.parallel.run_with_pool で各 worker に
    別 UMA calculator を load。workers == 1 で in-process sequential。

    各 NEB 結果は output_dir / f"trajectory_neb_trial_{trial_idx}.xyz" に
    書き出される。final_select はこのリストから最低 peak_energy を選ぶ。
    """
```

### 4.5 改変: `reactx/scoring.py`

```python
def top_k_trials(
    results: list[ScreeningTrialResult],   # Phase 8: TrialResult から rename + frames 等を持つ
    k: int,
) -> list[ScreeningTrialResult]:
    """既存 score_trials の優先順位 (reached_product=True 群を最低
    peak_energy 順 → False 群を最低 peak_energy 順で補完) で上位 k 件返す。

    入力が k 件未満なら全件返す。空入力は ValueError。
    """

def score_trials(results) -> ScreeningTrialResult:
    """top_k_trials(results, 1)[0] の薄い wrapper。後方互換用に残す
    (CLI 側は top_k_trials を直接呼ぶ)。"""
```

`TrialResult` dataclass は `ScreeningTrialResult` に rename。`select_best_trial` は `top_k_trials(results, 1)[0].trial_idx` の wrapper として残す。

### 4.6 改変: `reactx/config.py`

新セクション追加:

```toml
# 既存 (Phase 7 から)
description = "..."
formed = [...]
broken = [...]

[restraints]
k_form = ...
k_broken = ...
r_broken = ...
max_relax_steps = ...

[sampling]
n_candidates = 64

# Phase 8 で追加
[model]                      # ML potential のモデル指定 (新規)
                             # CLI の --backend uma|lj とは別概念 (こちらは UMA 内のモデル名選択)
screening_model = "uma-s-1p2"
neb_model = "uma-s-1p2"      # = screening_model なら 1 model load で済む

[neb]                        # CI-NEB hyperparameters (新規)
top_k = 4                    # screening → NEB に渡す trial 数
n_images = 7
fmax = 0.05
max_steps = 200
pad_frames = 0

[parallel]                   # process pool 並列度 (新規)
screening_workers = 3
neb_workers = 3
```

`[model]` / `[neb]` / `[parallel]` 全てオプション (省略時は default 値)。default 値は Phase 8 のメインターゲット (RTX 5070 Ti + 小モデル) を念頭に決めた数値。

新 dataclass:

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

Validation 追加:
- `neb.top_k >= 1`、`neb.n_images >= 3`、`neb.fmax > 0`、`neb.max_steps >= 1`、`neb.pad_frames >= 0`
- `parallel.screening_workers >= 1`、`parallel.neb_workers >= 1`
- `screening_model` / `neb_model` は文字列で非空 (中身の検証は fairchem-core に委ねる)

### 4.7 改変: `reactx/cli.py`

Orchestration を再構成:

```python
def _cmd_run(args):
    cfg = load_config(args.rxn_path)
    bond_changes = ...   # 既存
    mol_h, frag_indices, base_positions = embed_fragments_to_positions(...)
    placement = valid_placements(...)

    # Stage 1: screening (parallel)
    screen_results = screen_all_trials(
        placement, mol_h, bond_changes, cfg,
        backend=args.backend,
        screening_model=cfg.model.screening_model,
        workers=cfg.parallel.screening_workers,
        relax_fmax=args.relax_fmax,
        traj_stride=args.traj_stride,
        seed=args.seed,
    )

    # Top-K 絞り
    if args.no_neb:
        # NEB skip path: best 1 件をそのまま採用
        best = top_k_trials(screen_results, 1)[0]
        write_outputs(best.frames, best.energies, ...)
        return 0

    top_k = top_k_trials(screen_results, cfg.neb.top_k)

    # Stage 2: NEB (parallel)
    neb_results = run_neb_top_k(
        top_k,
        backend=args.backend,
        neb_model=cfg.model.neb_model,
        workers=cfg.parallel.neb_workers,
        n_images=cfg.neb.n_images,
        fmax=cfg.neb.fmax,
        max_steps=cfg.neb.max_steps,
        pad_frames=cfg.neb.pad_frames,
        output_dir=args.output,
    )

    # Final select
    final = min(neb_results, key=lambda r: r["peak_energy"])
    final_frames = read(final["xyz_path"], index=":")
    write(args.output / "trajectory.xyz", final_frames, format="extxyz")

    write_meta_json(...)
    if args.render:
        invoke_blender(...)
    return 0
```

CLI フラグの追加 / 削除:
- 削除: `--neb-refine`, `--neb-images`
- 追加: `--no-neb` (Stage 2 を skip)
- 残置: `--backend`, `--seed`, `--relax-fmax`, `--traj-stride`, `--render`, `--blender-exe`
- `--model` は残置するが、指定時は **screening と NEB の両方** に上書き適用 (TOML を override する単純動作。screening / NEB 個別の override は TOML でやってくれ、で押し通す)

## 5. データフロー (SN2 を例に)

```
入力: examples/sn2.rxn + examples/sn2.rxn.toml (Phase 8 schema)
  formed=[[1,3]], broken=[[1,2]]
  [model] screening_model="uma-s-1p2", neb_model="uma-s-1p2"
  [neb] top_k=4, n_images=7
  [parallel] screening_workers=3, neb_workers=3

→ placement: 64 candidates → blocking 後 ~20 survivor

→ Stage 1: 20 trial を 3 worker で並列に人工力 relax
  各 worker で uma-s-1p2 を 1 回 load (~10-15 s × 3 = ~15 s 並列)
  20 trial × ~5-10 s/trial / 3 worker ≒ 35-70 s (Phase 7 比 ~3× 高速化)
  → 20 ScreeningTrialResult、reached_product=True が ~6-10 件期待

→ top_k_trials(K=4):
  reached=True 群から peak_energy 昇順 4 件

→ Stage 2: 4 NEB jobs を 3 worker で並列
  各 worker で uma-s-1p2 を 1 回 load
  各 NEB: 2 × endpoint relax (~10 s) + warmup (~50 step) + climb (~100 step)
       ≒ ~120 s/NEB
  4 jobs / 3 worker ≒ ~240 s 合計 (sequential なら ~480 s)

→ Final: 4 NEB の peak_energy 最低を選択
  trajectory.xyz = 7 NEB images (+ pad)

合計 wall-clock 目標: ~5-7 分 (Phase 7 SN2 ≒ 4 分 + NEB 数値はそれよりかなり高品質)
```

## 6. エラーハンドリング

| 失敗モード | 検知箇所 | 挙動 |
|---|---|---|
| Stage 1 で全 trial が relax 例外 | `cli._cmd_run` | meta.json に `error` 込みで dump、`rc=1` |
| Stage 1 で reached_product=True が 0 件 | `top_k_trials` | 警告ログ + False 群から K 件補完 (現行挙動継承) |
| top-K 数 > screening survivors 数 | 同上 | 全件返す + 警告ログ (`top_k_trials` 内で clamp) |
| Stage 2 で NEB が converge しない | `run_neb_for_trial` | converged=False で続行、その trial の peak_energy は使う (NaN ではなく実値) |
| Stage 2 で NEB が例外 | `run_neb_for_trial` | その trial を結果から除外 + 警告ログ。残り trial で final select |
| Stage 2 で全 NEB が例外 | `cli._cmd_run` | meta.json に dump、`rc=1` (Stage 1 結果は残す) |
| `parallel.screening_workers > 物理 GPU メモリ許容` | runtime (CUDA OOM) | fairchem 側で例外。worker 数を減らせと user に促す警告 |
| `[model].screening_model` が fairchem に無いモデル名 | worker init | fairchem 側で例外、worker 起動失敗 → pool エラーで cli が rc=1 |
| Windows で `multiprocessing` 起動エラー | `parallel.run_with_pool` | spawn 強制 + `if __name__ == "__main__"` を CLI 側で確実にガード |

## 7. テスト計画

### 7.1 新規テストファイル

**`tests/test_screening.py`**:
- `test_screen_all_trials_sequential_workers_1` — workers=1 で in-process 実行、結果が ScreeningTrialResult のリスト
- `test_screen_all_trials_handles_relax_exceptions` — 1 trial の relax で例外を投げる mock calculator → 該当 trial の error フィールドにメッセージ、他は正常完了
- `test_screen_all_trials_preserves_input_order` — 並列実行でも trial_idx 順を維持

**`tests/test_parallel.py`**:
- `test_run_with_pool_workers_1_is_inprocess` — workers=1 では Pool を作らない (mock で確認)
- `test_run_with_pool_preserves_order` — 入力順を維持
- `test_run_with_pool_initializer_runs_once_per_worker` — initializer 呼び出し回数 == workers

**`tests/test_endpoints.py`**:
- `test_relax_endpoint_lj_simple_dimer` — LJ calculator で 2 原子系を relax、距離が r_min に近づく
- `test_relax_endpoint_does_not_raise_on_max_steps` — fmax=1e-9 / max_steps=2 で converge しなくても例外無し

**`tests/test_neb_top_k.py`** (slow):
- `test_run_neb_top_k_sn2_workers_1` — SN2 で K=2 sequential、2 dict が返る、いずれも `peak_energy` finite
- `test_run_neb_top_k_sn2_workers_2_parallel` — workers=2 で同じ反応、結果が wall-clock で sequential の 1.5× 以上速い (ハード依存なので tolerance 大きめ)

### 7.2 変更テストファイル

**`tests/test_cli.py`** (Phase 8 schema へ migration):
- `meta.json.neb_used` などの新フィールド assert
- `--neb-refine` / `--neb-images` を使っているテストは削除
- `--no-neb` の挙動 (NEB skip → trajectory.xyz は screening 由来) を追加

**`tests/test_config.py`** (新セクションの validation):
- `test_config_default_neb_section` — `[neb]` 省略時の default 値
- `test_config_neb_top_k_must_be_positive`
- `test_config_neb_n_images_min_3`
- `test_config_parallel_workers_must_be_positive`
- `test_config_model_section_default_uma_s`

**`tests/test_scoring.py`** (top_k への拡張):
- `test_top_k_trials_returns_k_when_enough_reached`
- `test_top_k_trials_falls_back_to_unreached_to_fill_k`
- `test_top_k_trials_returns_all_when_fewer_than_k`
- `test_top_k_trials_empty_raises`

**`tests/test_re1_*.py` / `tests/test_re3_*.py` / `tests/test_re4_*.py`** (slow integration tests):
- 全反応で trajectory.xyz が **NEB の image 数 (default 7)** であることを assert (Phase 7 では人工力 frames 数だった)
- 全反応で `meta.json.neb_used == True` 確認
- 全反応で `meta.json.neb_results` (新フィールド) が K 件存在
- E2 / SN1 dissoc / SN1 recomb も今回から CI-NEB が走る → 各反応の `meta.json.selected_neb_peak_energy` に finite 値
- SN1 dissoc は `n_candidates=1` のまま、Stage 2 は NEB 1 job のみ実行
- wall-clock テスト (`test_wallclock_sn2.py`) はしきい値再調整 (Phase 8 の目標値に合わせる)

### 7.3 削除テストファイル

- `tests/test_neb_refine_sn2.py` — `--neb-refine` 削除に伴い役割消滅、内容を `tests/test_re1_sn2.py` に統合 (CI-NEB が default で走るので独立 file 不要)
- `tests/test_align.py` — `align_product_to_reactant` 削除に伴いファイルごと削除
- `tests/test_cli_neb_refine_guard.py` — `--neb-refine` の 1+1 制限 guard が消えるので削除

### 7.4 Slow integration tests (`pytest -m slow`)

全 6 反応について Phase 8 schema で再 pass を確認。各反応で:
- `meta.json.neb_used == True`
- `len(trajectory.xyz frames) == cfg.neb.n_images + 2 * cfg.neb.pad_frames`
- `meta.json.neb_results` が `cfg.neb.top_k` 件 (clamping 発生時は `<= top_k`)
- `selected_trial` が `meta.json.neb_results` 内のいずれかと一致
- 反応固有 geometric assert (現状の Phase 7 assert を維持: SN2 の OCCl 角度、Menshutkin の C-N 距離 etc.)

### 7.5 Blender smoke test (`pytest -m blender`)

NEB images で trajectory.xyz が短くなる (典型 7-9 frames vs 人工力 ~30-40 frames) ので render 結果の動画 frame 数が減る。アニメーション時間が短くなるが視認性は問題ない (NEB images は path に沿って意味のある間隔で配置されている)。

`pad_frames` を default 0 のままにすると初手 / 末尾の停止時間が無くなる。視認性向上のため `[neb] pad_frames = 3` を例 TOML で推奨設定にする (default は 0)。

## 8. TOML schema 変更 (新セクションのみ)

### 8.1 既存セクション

不変 (`description` / `formed` / `broken` / `[restraints]` / `[sampling]`)。

### 8.2 新セクション

```toml
[model]
screening_model = "uma-s-1p2"
neb_model = "uma-s-1p2"

[neb]
top_k = 4
n_images = 7
fmax = 0.05
max_steps = 200
pad_frames = 0

[parallel]
screening_workers = 3
neb_workers = 3
```

全 6 example file (`examples/*.rxn.toml`) は新セクション無しでも動く (default 値で全部埋まる)。ただし以下 2 つは既存配置のために追記が必要:

- `examples/sn1_dissoc.rxn.toml`: unimolecular で `n_candidates=1` のため `[neb] top_k = 1` を明示 (auto-clamp される動作を依存にしない方針)。
- `examples/menshutkin.rxn.toml`: reached が 0 件で peak_energy 比較になるので `[neb] top_k = 4` のままで OK だが、`pad_frames = 3` を視認性向上のため追加。

## 9. `meta.json` schema 変更

```jsonc
// 旧 (Phase 7)
{
  "selected_trial": 0,
  "placement": {"n_candidates": 64, "n_blocked": 33, "n_valid": 31},
  "trials": [{"trial_idx": 0, "direction": [...], "reached_product": true, "peak_energy": ..., "n_steps": ...}],
  "neb_refined": false,
  "wall_clock_seconds": ...
}

// 新 (Phase 8)
{
  "selected_trial": 0,                    // top_k から最終選択された trial_idx
  "placement": {"n_candidates": 64, "n_blocked": 33, "n_valid": 31},
  "screening_trials": [
    {"trial_idx": 0, "direction": [...], "reached_product": true,
     "peak_energy": ..., "n_steps": ..., "error": null}
  ],
  "top_k_indices": [3, 7, 11, 0],        // screening_trials のうち NEB に流したもの (peak 順)
  "neb_used": true,
  "neb_results": [                        // top_k_indices と同順
    {
      "trial_idx": 3,
      "converged": true,
      "n_images": 7,
      "image_energies": [...],
      "peak_energy": ...,
      "final_fmax": ...,
      "xyz_path": "trajectory_neb_trial_3.xyz"  // output dir 相対
    }
  ],
  "wall_clock_seconds": ...,
  "wall_clock_breakdown": {                // debug 用に screening / NEB 区切りの時間
    "placement": 0.5,
    "screening": 35.2,
    "neb": 240.3,
    "render": 12.0
  },
  "effective_params": {
    "screening_model": "uma-s-1p2",
    "neb_model": "uma-s-1p2",
    "top_k": 4,
    "n_images": 7,
    "screening_workers": 3,
    "neb_workers": 3,
    /* 既存の k_form / k_broken / r_broken / max_relax_steps / r_form_targets / n_candidates */
  }
}
```

主要変更:
- `trials` → `screening_trials` (内容は ScreeningTrialResult に対応)
- `top_k_indices` 追加
- `neb_used` (bool) と `neb_results` (list) 追加
- `wall_clock_breakdown` 追加
- `neb_refined` 削除 (`neb_used` が後継、`--no-neb` で false)
- `effective_params` に model / neb / parallel フィールドを増やす

## 10. README 更新

- 「アーキテクチャ」ASCII 図を Phase 8 の 2-stage に書き換え
- 「対応反応」表に **CI-NEB 列** を追加し、6 反応すべてが本評価で CI-NEB を通ることを明記
- 「使い方」セクションの `--neb-refine` 例を削除
- 「Per-reaction `.rxn.toml` config」に `[model]` `[neb]` `[parallel]` セクションの説明追加
- 「Wall-clock (実測)」表を Phase 8 値で更新 (実装後)
- 「方針と限界」に「default は uma-s-1p2、`[model]` セクションで uma-m-1p1 等への切り替え可」を明記
- 「テスト」セクションは無変更

## 11. パラメータ default 値 + 根拠

| パラメータ | default | 根拠 |
|---|---|---|
| `model.screening_model` | `"uma-s-1p2"` | 小モデルで screening のスループット最大化。精度は本評価 (NEB) で担保 |
| `model.neb_model` | `"uma-s-1p2"` | 小モデルで wall-clock 短縮。精度を上げたいときは `"uma-m-1p1"` に上書き |
| `neb.top_k` | `4` | 5070 Ti + 並列 3 worker で 1-2 wave で完了する数。screening が ~20-30 件残るので 4 件抽出は充分な余裕 |
| `neb.n_images` | `7` | reactant + 5 middle + product。Phase 7 default を踏襲。少なすぎると saddle 周辺の解像度が落ち、多すぎると wall-clock 増 |
| `neb.fmax` | `0.05` | Phase 7 `run_neb` default を踏襲 |
| `neb.max_steps` | `200` | warmup 100 + climb 100 程度の budget。Phase 7 で実測 SN2 が ~7 分の経験則 |
| `neb.pad_frames` | `0` | render 視認性のための padding は user 側で TOML 指定 |
| `parallel.screening_workers` | `3` | RTX 5070 Ti = 16 GB / `uma-s-1p2` ~3.5 GB ≒ 4.5 instance 上限、安全側で 3 |
| `parallel.neb_workers` | `3` | 同上。Stage 2 は 1 worker あたり ~5-7 image を sequential 評価するので、4 並列でも瞬間的な VRAM ピークは 1 instance ぶん |

## 12. パフォーマンス見積もり (RTX 5070 Ti + uma-s-1p2)

Phase 7 の SN2 (uma-m-1p1, sequential, 232 s) を Phase 8 default で再実行した場合の概算:

| Stage | Phase 7 | Phase 8 (見積もり) | 高速化要因 |
|---|---|---|---|
| placement | ~0.5 s | ~0.5 s | 不変 |
| Screening (20 trial) | ~200 s | ~50 s | 小モデル ~2× + 3 並列 ~3× = ~6× |
| Top-K NEB | 0 (skip) | ~120 s (4 NEB / 3 worker) | 新規追加 |
| 合計 | ~232 s | ~170 s | screening 短縮分が NEB 追加分を吸収 |

CI-NEB 経由で peak_energy が **真の TS エネルギー近似** に切り替わるメリットの方が wall-clock の差より重要 (人工力 peak は拘束項込みで不正確だった)。

`uma-m-1p1` を NEB に使う構成 (TOML B 案) では Phase 8 ~330 s 程度になる見込み (NEB だけ 2× 重くなる)。

## 13. 実装順 (writing-plans への申し送り)

おおむね以下の依存順:

1. `reactx/parallel.py` 新規 + `tests/test_parallel.py` (依存無し、Pool helper のみ)
2. `reactx/config.py` 拡張 (`ModelConfig` / `NebConfig` / `ParallelConfig`) + 既存 `test_config.py` に新セクション validation 追加
3. `reactx/scoring.py` 拡張 (`top_k_trials`、`TrialResult` → `ScreeningTrialResult` rename) + `test_scoring.py` 更新
4. `reactx/endpoints.py` 新規 + `tests/test_endpoints.py`
5. `reactx/screening.py` 新規 (Stage 1 並列実行) + `tests/test_screening.py`
6. `reactx/neb.py` 拡張 (`run_neb_for_trial` / `run_neb_top_k`) + `tests/test_neb_top_k.py`
7. `reactx/cli.py` 改変 (orchestration 全書き換え、`--neb-refine` 削除、`--no-neb` 追加、meta.json schema 更新)
8. `reactx/align.py` 削除 + `tests/test_align.py` 削除 (CI-NEB が trajectory 由来 endpoint を使う設計のため不要)
9. `tests/test_neb_refine_sn2.py` / `tests/test_cli_neb_refine_guard.py` 削除
10. `examples/*.rxn.toml` 全 6 件に `[neb] top_k = N` を必要に応じて追記 (sn1_dissoc は 1)
11. Slow integration tests (test_re1_*.py / test_re3_*.py / test_re4_*.py) を Phase 8 schema に migration、CI-NEB 結果の geometric / energetic assert 追加
12. README 全面更新 (アーキテクチャ図、対応反応表、Wall-clock 表)
13. Wall-clock 再実測 → README 数値更新

各ステップで unit test pass を確認しつつ進める。Stage 1 + Stage 2 を独立にテストしてから cli 統合を行う方針 (cli 統合時に integration test が初めて全部走る)。
