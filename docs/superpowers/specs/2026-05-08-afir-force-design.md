# reactx Phase 9 — Per-pair AFIR Force + Two-stage Relax (v2, post-codex-review)

- Status: Draft (post external review revision; awaiting user spec review)
- Date: 2026-05-08
- Owner: @kam6y
- Branch: `phase-9` (from `develop` after Phase 8 merge)
- 前提仕様: `docs/superpowers/specs/2026-05-04-generic-placement-design.md`、`docs/superpowers/specs/2026-05-05-phase-8-cycloaddition-design.md`
- 後方互換性: **完全に放棄する**。`.rxn.toml` schema・内部 API すべて breaking change を許容。既存 8 example は再 tune する。
- 前版 (v1) との差分: 数式の中心仮定 (multi-pair collective with ω weighting + 自然減衰による cutoff 不要) が外部レビューで誤りと判明したため、per-pair AFIR (each pair independent, F=α·d̂) + two-stage relax (driven Stage A + unbiased Stage B) + early-stop に再設計した。詳細は §10 改訂履歴参照。

## 1. 目的

Phase Re1 〜 Phase 8 (再) で使ってきた経験的力場 (`Hookean attraction` + `PullApart repulsion`) を、概念的に統一された **per-pair AFIR (single-pair-collective Artificial Force Induced Reaction)** + **two-stage relax** に置き換え、TOML 設定を簡素化しつつ「product 到達後に力が消えない」という per-pair AFIR 固有のリスクを Stage 構造で吸収する。

具体的には:

- 各反応の力場ハイパラを **`alpha_formed` / `alpha_broken` の 2 種** (per-pair リスト or スカラー) に統一。`r_form` / `r_broken` は force パラメータからは消滅し、scoring 専用閾値 (`[scoring].r_broken_threshold`) のみ残る。
- per-pair 力 `F = α · d̂` (一定 magnitude) の特性により、reaction を「product 到達まで一定速度で押し続ける」アニメーション向けの駆動を実現。**力は距離依存しないため product 到達後も押し続ける** が、これは early-stop で吸収する。
- 既存の Phase 8 (再) で使われている per-bond list (`k_form = [3.0, 3.0]`) の表現力を保持 (各 pair に個別 α)。
- relax を **Stage A (AFIR 駆動 + per-step `reached_product` 監視で即停止)** と **Stage B (AFIR 外して unbiased FIRE/BFGS で短く relax)** の 2 段に分ける。trajectory は両 Stage の concat、scoring の peak_energy は Stage A frames のみで計算 (Stage B は relax で energy が単調減少するため peak は Stage A 内に落ちる)。

phase-8 ブランチで挫折した「CI-NEB 汎用化 + screening + 並列化」の 3 点同時変更は本フェーズには含めない。force model の刷新と Stage 構造のみ。

**重要な精神的注意**: 本フェーズで生成される trajectory は **biased-force によって駆動された path** であり、真の minimum energy path や transition state を反映するものではない。peak_energy も approximate な指標であって障壁の正確な値ではない。仕様書全体で「product-like animation の生成」を目的として一貫して扱い、「mechanism の科学的同定」を主張しない。CI-NEB による mechanism refine は Phase 10 で別途扱う。

## 2. Non-goals

明示的に **このフェーズではやらない** こと:

- **CI-NEB の汎用化**: `--neb-refine` は今のとおり 1 formed + 1 broken 反応専用のオプション (`align_product_to_reactant` 経由)。E2 / SN1 / DA で NEB 必要なら future phase。
- **Screening / top-K / 並列化**: Phase 8 (旧) の Stage1→top-K→Stage2 構造は導入しない。全 placement-survivor を sequential に AFIR-relax する従来構造を維持。
- **Multi-pair collective AFIR (Maeda 原典の ω-weighted ρ)**: 本フェーズの per-pair AFIR は **single-pair-collective AFIR を pair ごとに独立適用** する形であり、Maeda 2010 原典の multi-pair ω-weighted collective は採用しない (理由は §10.1)。
- **MC-AFIR (reaction discovery mode)**: ユーザーが `formed`/`broken` を **書かない** で AFIR が自動的に bond change を発見する Multi-Component AFIR は対象外。本フェーズはあくまで「formed/broken は手動指定」前提の force-replacement。
- **新反応の追加**: 8 反応 (SN2 / PT / Menshutkin / E2 / SN1 dissoc / SN1 recomb / DA simple / DA endo) を全て新方式で動かすことがゴール。新反応は別フェーズ。
- **biased PES の peak energy 報告**: AFIR-biased energy ではなく **unbiased UMA energy** を `meta.json` / `energies.json` に記録する (既存仕様維持)。AFIR は ASE FixConstraint パターンで力のみ加えるため、`atoms.get_potential_energy()` には影響しない。
- **Stage A の force ramp-up/ramp-down**: 一定 magnitude `α` で固定。ramp は導入しない (early-stop で十分機能するため)。

## 3. アーキテクチャ

### 3.1 パイプライン全体図

```
.rxn + .rxn.toml ─> rxn_parser + load_config ─> ReactionConfig
                                                       │
                                                       ▼
                                               BondChanges (formed/broken)
                                                       │
                                                       ▼
                                       embed_fragments_to_positions   (Phase 7 と同じ)
                                                       │
                                                       ▼
                       placement.valid_placements                     (Phase 7/8 と同じ)
                       ├─ unimolecular passthrough
                       ├─ single-anchor (bridges == 1)
                       └─ multi-anchor (bridges == 2)
                                                       │
                                                       ▼
            ┌── trial 1 ──┐
            ├── trial 2 ──┤  build_afir_constraint(formed, broken, α_f[], α_b[])
            ├── ...        │      │
            └── trial K ──┘      │
                                 ▼
                  ┌─── Stage A (driven) ───┐
                  │  FIRE + AFIRConstraint │  per-step check: reached_product?
                  │  ↓ early-stop on True  │  ─── frames_A, energies_A
                  └────────────────────────┘
                                 │
                                 ▼
                  ┌─── Stage B (release) ──┐
                  │  FIRE without AFIR     │  fmax=0.1, max_steps=50
                  │  on last frame of A    │  ─── frames_B, energies_B
                  └────────────────────────┘
                                 │
                                 ▼
                  trajectory = frames_A + frames_B
                  peak_energy = max(energies_A)         ← Stage A only
                  reached_product = (Stage A early-stop fired) OR (final B passes check)
                                                       │
                                                       ▼
                                          scoring.score_trials      (Phase 7 と同じ)
                                                       │
                                                       ▼
                                          best trial → trajectory.xyz
                                                       │
                                              (optional --neb-refine、1+1 のみ)
                                                       │
                                                       ▼
                                          blender/render.py → .blend
```

### 3.2 Phase 8 (再) → Phase 9 主要変更

| 項目 | Phase 8 (再) | Phase 9 |
|---|---|---|
| 力場 (formed) | `Hookean(k, rt)` per-bond (`r > rt` 時 attractive、力は `(r-rt)` で減衰) | `AFIRPair(α)` per-bond (常に attractive、`F = α · d̂`、距離不変) |
| 力場 (broken) | `PullApart(k, rt)` per-bond (`r < rt` 時 repulsive、`r ≥ rt` でゼロ) | `AFIRPair(α)` per-bond (常に repulsive、`F = α · d̂`、距離不変) |
| ω weighting | なし | **採用しない** (per-pair 独立、Maeda 原典の ω-weighted ρ は §10.1 の理由で破棄) |
| 半径基底 | `DEFAULT_R_FORM` 元素表 + `r_broken` 手動 | force 計算では半径不要 (`d̂` のみ)。scoring の閾値で Cordero (2008) 共有半径を共通使用 |
| TOML keys (force) | `[restraints] k_form k_broken r_broken r_form max_relax_steps` | `[afir] alpha_formed alpha_broken max_relax_steps` |
| `alpha_*` の per-bond | (Phase 8 (再) で `k_form`/`k_broken` のみ list 対応) | `alpha_formed: float \| list[float]`, `alpha_broken: float \| list[float]` 両対応 (各 pair 独立 α、Phase 8 表現力保持) |
| TOML keys (scoring) | (`r_broken` を force と兼用) | `[scoring] r_broken_threshold: float \| list[float]` (per-pair 必須)、`r_formed_threshold: float \| list[float]` (任意、default = `Rsum + 0.4 Å`) |
| 力の self-extinguishing | Hookean (`r ≤ rt` でゼロ)、PullApart (`r ≥ rt` でゼロ) | **なし**: AFIR は product 到達後も押し続ける。**Stage A の per-step `reached_product` 監視 + early-stop で吸収** |
| relax 構造 | 1 stage (FIRE + Hookean+PullApart) | **2 stage**: Stage A (FIRE + AFIR + early-stop)、Stage B (FIRE 単独、`max_steps=50`、`fmax=0.1`) |
| trajectory.xyz | Stage 1 frames | Stage A frames + Stage B frames concat |
| `peak_energy` の対象 | 全 frames | **Stage A frames のみ** (Stage B は monotone decrease なので peak が Stage B にあるはずがなく安全) |
| `reached_product` 評価 | 最終 frame のみ | Stage A 中の per-step + Stage B 最終 frame 最終確認 |
| `r_form` lookup | `DEFAULT_R_FORM` 元素表 (force と兼用) | scoring の formed 判定にだけ使用、default `Rsum_covalent + 0.4 Å` |
| FIRE 設定 | `maxstep=0.1, dt=0.05, dtmax=0.2` (Hookean self-extinguish 前提) | **AFIR 用に再 tune** (`maxstep=0.2-0.4`、`dt=0.05` 維持、`dtmax=0.2` 維持あたりから探索)、最終値は slow test で決定し本仕様にも反映 |

### 3.3 ファイル構成

書き換え:
- `reactx/artificial_force.py` — `Hookean`/`PullApart`/`build_restraints`/`DEFAULT_R_FORM`/`lookup_r_form` を全削除し、`AFIRConstraint` クラスと `build_afir_constraint` 関数のみ残す
- `reactx/path_relax.py` — `relax_with_restraints` を **`relax_two_stage`** に書き換え (Stage A + Stage B + early-stop)。`reached_product` callback を受け取る
- `reactx/config.py` — `[restraints]` セクション削除、`[afir]` 追加、`[scoring]` 拡張。`resolve_*_targets` ヘルパは AFIR 用に簡素化 (`α_formed`/`α_broken` の broadcasting と `r_*_threshold` の broadcasting のみ)
- `reactx/scoring.py` — `reached_product` を `r_formed_threshold` / `r_broken_threshold` ベースの per-pair 閾値で再実装。AFIR Stage A 内の per-step 評価でも同じ関数を使う
- `reactx/cli.py` — `build_restraints` 呼び出しを `build_afir_constraint` に置換、orchestration を two-stage に
- `examples/*.rxn.toml` × 8 — 新 schema に書き換え (各 reaction で α / r_*_threshold を tune)
- `tests/test_*.py` — Hookean/PullApart 系を削除、AFIR 系を追加、slow integration の期待値を再 tune
- `README.md` — Phase 9 セクション、新 schema 表、wall-clock 再測定

新規:
- `reactx/covalent_radii.py` — Cordero (2008) 共有半径 lookup (現在 `blender/render.py` 内に埋め込まれている表を切り出す。`reactx/scoring.py` の閾値計算と Blender の bond drawing で **同一テーブル** を使う)
- `tests/test_afir_constraint.py` — 解析勾配 vs 有限差分の一致、空 set の no-op、力の方向、対称性、r→0 の fail-fast

削除候補:
- `reactx/align.py` (Phase 8 旧で削除予定だった残骸の最終撤去 — `align_product_to_reactant` を 1+1 NEB がまだ使うなら **保留**、未使用なら削除)。実装時に `grep -r align_product_to_reactant` で確認後決定。

## 4. コンポーネント詳細

### 4.1 `reactx/artificial_force.py` — `AFIRConstraint`

per-pair の独立 AFIR 力。各 pair が **single-pair AFIR collective** を持ち、ρ_pair = r_ij だから ∂ρ/∂r = 1、F_pair = ±α · d̂_ij (一定 magnitude)。ω weighting は使わず、各 pair の α は独立。

```python
"""Per-pair AFIR force.

For each (i, j) pair specified in `formed` (or `broken`):

    F_AFIR_on_j = sign · α_pair · d̂_ij,  d̂_ij = (x_j - x_i) / r_ij
    F_AFIR_on_i = -F_AFIR_on_j           (Newton's third law)

with sign = -1 for formed pairs (compress) and sign = +1 for broken pairs
(expand). The force magnitude is **constant** in r — it does not vanish at
the equilibrium bond length, which is intentional for AFIR-style driving
but requires the early-stop logic in `relax_two_stage` to terminate Stage A
once the target structure is reached.

This matches the single-pair AFIR collective (ρ = r_ij, ∂ρ/∂r = 1) of
Maeda & Morokuma 2010, applied independently to each user-specified pair.
The original multi-pair ω-weighted collective is *not* used here; see
spec §10.1 for the rationale (it can produce sign-flipped forces on
co-formed bonds in asymmetric Diels-Alder).
"""

from __future__ import annotations
import numpy as np
from ase.atoms import Atoms
from ase.constraints import FixConstraint


class AFIRConstraint(FixConstraint):
    def __init__(
        self,
        formed: list[tuple[int, int]],
        broken: list[tuple[int, int]],
        *,
        alpha_formed: list[float],   # length == len(formed); >= 0
        alpha_broken: list[float],   # length == len(broken); >= 0
    ):
        if len(alpha_formed) != len(formed):
            raise ValueError(
                f"alpha_formed length {len(alpha_formed)} != n_formed {len(formed)}")
        if len(alpha_broken) != len(broken):
            raise ValueError(
                f"alpha_broken length {len(alpha_broken)} != n_broken {len(broken)}")
        if any(a < 0 for a in alpha_formed) or any(a < 0 for a in alpha_broken):
            raise ValueError("alpha_* must be non-negative")
        # store as plain Python tuples to keep ASE serialization simple
        self.formed = [(int(a), int(b)) for a, b in formed]
        self.broken = [(int(a), int(b)) for a, b in broken]
        self.alpha_formed = [float(x) for x in alpha_formed]
        self.alpha_broken = [float(x) for x in alpha_broken]

    def adjust_positions(self, atoms, newpositions):
        return  # force-only constraint

    def adjust_forces(self, atoms, forces):
        pos = atoms.positions
        for (i, j), alpha in zip(self.formed, self.alpha_formed, strict=True):
            if alpha == 0.0:
                continue
            self._apply_pair(pos, forces, i, j, sign=-1.0, alpha=alpha)
        for (i, j), alpha in zip(self.broken, self.alpha_broken, strict=True):
            if alpha == 0.0:
                continue
            self._apply_pair(pos, forces, i, j, sign=+1.0, alpha=alpha)

    @staticmethod
    def _apply_pair(pos, forces, i, j, *, sign, alpha):
        v = pos[j] - pos[i]
        r = float(np.linalg.norm(v))
        if r < 1e-6:
            # Coincident atoms indicate a placement bug — fail fast rather
            # than silently apply a meaningless direction.
            raise ValueError(
                f"AFIRConstraint: atoms {i} and {j} coincide (r={r:.2e}); "
                f"check fragment placement geometry"
            )
        d_hat = v / r
        # F_on_j = sign · α · d̂      (sign=-1 → toward i, sign=+1 → away from i)
        force_on_j = sign * alpha * d_hat
        forces[j] += force_on_j
        forces[i] -= force_on_j

    def get_indices(self):
        return sorted({a for pair in (self.formed + self.broken) for a in pair})

    def todict(self):
        return {
            "name": "AFIRConstraint",
            "kwargs": {
                "formed": self.formed,
                "broken": self.broken,
                "alpha_formed": self.alpha_formed,
                "alpha_broken": self.alpha_broken,
            },
        }


def build_afir_constraint(
    atoms: Atoms,
    formed: list[tuple[int, int]],
    broken: list[tuple[int, int]],
    *,
    alpha_formed: float | list[float],
    alpha_broken: float | list[float],
) -> list[AFIRConstraint]:
    """Broadcast scalar α values to per-pair lists, return 0- or 1-element list."""
    if not formed and not broken:
        return []
    af = _broadcast(alpha_formed, len(formed), key="alpha_formed")
    ab = _broadcast(alpha_broken, len(broken), key="alpha_broken")
    return [AFIRConstraint(formed, broken, alpha_formed=af, alpha_broken=ab)]


def _broadcast(value: float | list[float], n: int, *, key: str) -> list[float]:
    if isinstance(value, list):
        if len(value) != n:
            raise ValueError(f"{key} list length {len(value)} != n_pairs {n}")
        return [float(v) for v in value]
    return [float(value)] * n
```

**重要**:
- per-pair の force は **距離に依存せず** `α · d̂` 一定。これにより Stage A は product 到達まで一定速度で押し続ける。Maeda 原典 (multi-pair ω-collective) との違いは §10.1 を参照。
- `α = 0` の pair は skip (`broken=[]` の DA で `alpha_broken=[0.0,...]` が来ても問題なし)。
- `r < 1e-6` (1 µÅ) で **fail fast**。原子が重なっているのは placement バグであり、relax で「自然に離れる」を期待するのは危険。
- `adjust_forces` は energy には触れない (ASE FixConstraint 規約)。`atoms.get_potential_energy()` は素の UMA energy のまま。

### 4.2 `reactx/path_relax.py` — `relax_two_stage`

```python
"""Two-stage relaxation: AFIR-driven Stage A + unbiased Stage B.

Stage A drives the system toward the product manifold under AFIRConstraint,
checking `reached_product(atoms)` after every optimizer step. As soon as the
check returns True, Stage A halts (early-stop) — preventing AFIR from
continuing to push past the product geometry.

Stage B removes the AFIR constraint and runs a short unbiased FIRE relax
from Stage A's last frame, settling the system into a local minimum near
the product. Stage B is bounded (`max_steps=50`, `fmax=0.1`) and is purely
cosmetic for the animation; scoring uses Stage A's energies only.

If Stage A fails to reach the product within `max_steps_a`, the trial is
still recorded (with `reached_product=False`); Stage B still runs from
the last A frame, and `score_trials`'s least-bad fallback handles selection.
"""

def relax_two_stage(
    atoms: Atoms,
    afir_constraints: list,         # [] or [AFIRConstraint]
    calc: Calculator,
    *,
    reached_product_fn,             # callable(Atoms) -> bool
    max_steps_a: int,               # from cfg.afir.max_relax_steps
    max_steps_b: int = 50,
    fmax_a: float = 0.1,
    fmax_b: float = 0.1,
    traj_stride: int = 5,
) -> tuple[list[Atoms], list[float], list[Atoms], list[float], bool]:
    """
    Returns (frames_a, energies_a, frames_b, energies_b, early_stopped).

    `early_stopped=True` iff Stage A halted because reached_product_fn
    returned True (not because max_steps was hit or fmax was met).
    """
    ...
```

実装ポイント:
- Stage A: `atoms.set_constraint(afir_constraints)`、FIRE で `attach(record_and_check, interval=1)` (per-step check)。`reached_product_fn(atoms)` が True ならその場で `opt.converged` を short-circuit (例えば `raise StopIteration` を catch する pattern、あるいは ASE の `FIRE.run` を loop で呼んで条件付き break)。具体的な実装は test 駆動で決定。
- Stage B: `atoms.set_constraint([])`、新規 FIRE optimizer (`maxstep=0.1, dt=0.05` Hookean時代の値で OK)、`max_steps=50`。
- frames は両 Stage で別 list として返し、cli 側で concat する (peak_energy 計算が Stage A だけに限定できる)。

FIRE 設定 (Stage A 用、AFIR 一定力に対応):
- `maxstep=0.2-0.4` (現行 0.1 では一定力に対する step が小さすぎ収束遅い、実装時に slow test で確定)
- `dt=0.05`、`dtmax=0.2` 維持で出発 (overshoot が見られたら再 tune)
- `fmax=0.1`: AFIR 力が常に存在するため、`fmax<=0.1` で「収束」する条件は実質「AFIR 残力 + UMA 残力が打ち消す配置」になる。これは not necessarily product configuration なので、Stage A は **early-stop 主導**で終了し、`fmax` 収束は補助的。`max_steps_a` 上限到達も terminate 条件。

### 4.3 `reactx/covalent_radii.py` — Cordero 表の切り出し

```python
"""Cordero (2008) covalent radii in Å.

Used by:
- `reactx/scoring.py` for reached_product threshold defaults
- `blender/render.py` for bond drawing distance threshold

Reference: Cordero et al., Dalton Trans. 2008, 2832.
"""

from __future__ import annotations
import numpy as np
from ase.atoms import Atoms

CORDERO_2008: dict[str, float] = {
    # Z=1..83 per OMol25 / UMA training range
    "H": 0.31, "He": 0.28,
    "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66,
    "F": 0.57, "Ne": 0.58,
    # ... (full table identical to existing blender/render.py COVALENT_RADII_ANGSTROM)
}

def cordero_radius(symbol: str, *, default: float = 1.5) -> float:
    return CORDERO_2008.get(symbol, default)

def cordero_radii_for_atoms(atoms: Atoms) -> np.ndarray:
    return np.array([cordero_radius(s) for s in atoms.get_chemical_symbols()])
```

`blender/render.py` も import に切り替えて重複辞書を排除。Blender 環境で `reactx` パッケージが import できない場合 (現状の `try/except ImportError` パターン) は本ファイルの inline コピーに fallback。

### 4.4 `reactx/config.py` — TOML schema 変更

新 schema (`examples/sn2.rxn.toml`):

```toml
description = "SN2 anion: CH3Cl + OH- -> CH3OH + Cl-"
formed = [[1, 3]]
broken = [[1, 2]]

[afir]
alpha_formed = 0.7      # eV/Å, scalar (positive: 縮める方向、内部で sign=-1)
alpha_broken = 0.5      # eV/Å, scalar (positive: 引き離す方向、内部で sign=+1)
max_relax_steps = 100   # Stage A の上限ステップ数。Stage B は固定 50

[scoring]
r_broken_threshold = 4.0   # 必須、reached_product 判定用 (per-bond list 可)
# r_formed_threshold = ...  # 任意、未指定時は (R_i+R_j)_covalent + 0.4 Å (per-bond)
```

E2 (per-bond α、per-bond r_broken_threshold):

```toml
description = "E2 elimination..."
formed = [[4, 5]]
broken = [[2, 5], [1, 3]]

[afir]
alpha_formed = 1.5
alpha_broken = [1.0, 1.5]   # C-H と C-Cl で別 magnitude
max_relax_steps = 200

[scoring]
r_broken_threshold = [3.0, 4.0]   # C-H ≥ 3.0 Å, C-Cl ≥ 4.0 Å
```

DA (両方の formed bond に同じ α、broken なし):

```toml
[afir]
alpha_formed = [2.5, 2.5]
alpha_broken = 0.0          # 使われない (broken=[])
max_relax_steps = 200

[scoring]
r_broken_threshold = 4.0    # broken=[] なら使われない、ダミーでも可
```

Dataclass:

```python
@dataclass(frozen=True)
class AFIRSection:
    alpha_formed: float | list[float]
    alpha_broken: float | list[float]
    max_relax_steps: int

@dataclass(frozen=True)
class ScoringSection:
    r_broken_threshold: float | list[float]
    r_formed_threshold: float | list[float] | None = None  # None = covalent default

@dataclass(frozen=True)
class ReactionConfig:
    description: str
    formed: list[tuple[int, int]]
    broken: list[tuple[int, int]]
    afir: AFIRSection
    scoring: ScoringSection
    sampling: SamplingSection         # 既存維持
```

バリデーション:
- `alpha_formed` / `alpha_broken`: scalar `>= 0` または `list[float]` で全要素 `>= 0` かつ length が `formed` / `broken` と一致
- `max_relax_steps > 0`
- `r_broken_threshold`: scalar `> 0` または `list[float]` で全要素 `> 0` かつ length が `broken` と一致
  - 例外: `broken=[]` の DA / SN1 recomb では `r_broken_threshold` 値は使われないが、TOML 記述は **必須** (placeholder でも書く運用、validation で「broken=[] なら任意」として穴を開けない方針)。理由: 後で broken を追加する変更で TOML が静かに壊れない。
- `r_formed_threshold`: scalar `> 0` または `list[float]` で全要素 `> 0` かつ length が `formed` と一致、または `None` で default 使用
- 旧 keys (`k_form`, `k_broken`, `r_broken`, `r_form`、 `[restraints]` セクション) が出現したら `ConfigError` で reject

### 4.5 `reactx/scoring.py` — `reached_product`

```python
COVALENT_FORMED_SLACK_ANGSTROM = 0.4   # default formed threshold = Rsum + 0.4 Å


def resolve_formed_threshold(atoms, formed, override) -> list[float]:
    """Per-pair formed thresholds (Å). None override → Rsum_covalent + 0.4 Å."""
    cov = cordero_radii_for_atoms(atoms)
    if override is None:
        return [cov[i] + cov[j] + COVALENT_FORMED_SLACK_ANGSTROM
                for (i, j) in formed]
    if isinstance(override, list):
        return [float(x) for x in override]
    return [float(override)] * len(formed)


def resolve_broken_threshold(atoms, broken, override) -> list[float]:
    """Per-pair broken thresholds (Å). Required (no covalent default)."""
    if isinstance(override, list):
        return [float(x) for x in override]
    return [float(override)] * len(broken)


def reached_product(
    atoms: Atoms,
    formed: list[tuple[int, int]],
    broken: list[tuple[int, int]],
    formed_thresholds: list[float],     # already resolved per-pair
    broken_thresholds: list[float],
) -> bool:
    pos = atoms.positions
    for (i, j), thr in zip(formed, formed_thresholds, strict=True):
        d = float(np.linalg.norm(pos[j] - pos[i]))
        if d > thr:
            return False
    for (i, j), thr in zip(broken, broken_thresholds, strict=True):
        d = float(np.linalg.norm(pos[j] - pos[i]))
        if d < thr:
            return False
    return True
```

`score_trials` の他の部分 (peak_energy 最小化、least-bad fallback) は変更なし。**ただし peak_energy は Stage A frames のみで計算する** (Stage B は energy が monotone decrease するため、Stage A peak が overall peak になる。ただし Stage A 駆動中に over-extension で過大評価されるリスクは early-stop で軽減済)。

### 4.6 `reactx/cli.py` — orchestration

per-trial flow:

```python
afir_cs = build_afir_constraint(atoms_init, formed, broken,
                                alpha_formed=cfg.afir.alpha_formed,
                                alpha_broken=cfg.afir.alpha_broken)
ft = resolve_formed_threshold(atoms_init, formed, cfg.scoring.r_formed_threshold)
bt = resolve_broken_threshold(atoms_init, broken, cfg.scoring.r_broken_threshold)

def reached(a):
    return reached_product(a, formed, broken, ft, bt)

frames_a, eners_a, frames_b, eners_b, early = relax_two_stage(
    atoms_init, afir_cs, calc,
    reached_product_fn=reached,
    max_steps_a=cfg.afir.max_relax_steps,
    max_steps_b=50,
    fmax_a=relax_fmax,
    fmax_b=0.1,
    traj_stride=traj_stride,
)
trajectory = frames_a + frames_b
energies = eners_a + eners_b           # Stage B energies are stored but...
peak_energy = max(eners_a) if eners_a else float("inf")  # ...peak is from A only
final_reached = reached(frames_b[-1] if frames_b else frames_a[-1])
```

`meta.json` per-trial エントリ更新:
- 削除: `r_broken_target`, `r_form_target`, `k_form`, `k_broken`
- 追加: `alpha_formed: list[float]`, `alpha_broken: list[float]`, `r_formed_threshold: list[float]`, `r_broken_threshold: list[float]`, `early_stopped: bool`, `n_steps_a: int`, `n_steps_b: int`

NEB refine path (`--neb-refine`、1+1 のみ) は変更なし。AFIR endpoints (Stage A 最終 + Stage B 最終) を NEB の入力として使うが、現状の `align_product_to_reactant` が原子順序保存のため必要なら継続使用、不要なら削除。実装時に決定。

## 5. 8 example の AFIR 初期 tuning 値

per-pair AFIR は **力が距離不変** で、Hookean (`r-rt` で減衰) や PullApart (`rt-r` で減衰) と次元が違うため、旧 `k_form`/`k_broken` からの数値換算はできない。下表は「step あたりの押し速度 = `α × dt × maxstep`」を Hookean 時代と同程度にする経験値の **初期 guess**。実装後 slow test の `pytest -m slow` で各反応の `reached_product` 到達と Stage A の peak_energy 妥当性を見ながら 1-2 反復で再調整。

| Reaction | α_formed | α_broken | max_relax_steps | r_broken_threshold | r_formed_threshold | 備考 |
|---|---|---|---|---|---|---|
| sn2 | 0.7 | 0.7 | 100 | 4.0 | (default) | C-Cl 切断 |
| proton_transfer | 1.5 | 1.0 | 100 | 4.0 | 1.5 (短いO-H用) | H-Cl 切断、O-H 形成 |
| menshutkin | 4.0 | 1.5 | 200 | 5.0 | (default) | gas-phase repulsive PES |
| e2 | 1.5 | [1.0, 1.5] | 200 | [3.0, 4.0] | (default) | 2 broken: C-H, C-Cl |
| sn1_dissoc | 0.0 | 1.5 | 200 | 6.0 | (default、formed=[] で使われない) | unimolecular |
| sn1_recomb | 1.5 | 0.0 | 200 | 4.0 (placeholder、broken=[]) | (default) | bimolecular formation |
| diels_alder_simple | [2.5, 2.5] | 0.0 | 200 | 4.0 (placeholder) | (default) | symmetric DA |
| diels_alder_endo | [2.5, 2.5] | 0.0 | 250 | 4.0 (placeholder) | (default) | endo/exo CP+MA |

注: Phase 8 (再) の `k_form = [3.0, 3.0]` (per-bond list) → α_formed = `[2.5, 2.5]` で per-bond list を保持。symmetric な分子 (butadiene + ethylene) では同値で、対称性の崩れは生じない。

## 6. テスト戦略

### 6.1 新規 unit tests (`tests/test_afir_constraint.py`)

- **解析勾配 vs 有限差分**: 任意の n-atom 配置 + ランダム formed/broken pair set で、`AFIRConstraint.adjust_forces` が返す力 = `Σ_pairs (sign · α · d̂)` を有限差分との一致 (1e-6 相対誤差) で確認 (per-pair 力は閉形式なので有限差分と完全一致するはず)
- **単一 pair 縮約**: `formed=[(0,1)], alpha_formed=[α]` のとき force_on_1 = `-α · d̂_01`、force_on_0 = `+α · d̂_01`
- **空 set no-op**: `formed=[], broken=[]` で `build_afir_constraint` が `[]` を返す。`alpha_*=[]` で force 適用なし
- **per-pair 独立性**: 2 pair で別々の α を指定したとき、各 pair の force は他 pair に依存しない (single-pair-collective の独立性)
- **対称性**: `(i, j)` と `(j, i)` で同じ力 (順序不変性)
- **r→0 fail-fast**: 同位置の 2 atom で `ValueError`
- **負 α 拒否**: `alpha_formed=[-0.1]` で `ValueError`
- **broadcasting**: scalar `alpha_formed=0.7` と `[0.7]*n` が同等

### 6.2 新規 unit tests (`tests/test_relax_two_stage.py`)

- Stage A early-stop: `reached_product_fn` が False→False→True と返す mock で、3 step 目で early_stop
- Stage A max_steps 到達 (reached_product 常 False) で `early_stopped=False`、`max_steps_a` step 完走
- Stage B: AFIR が外れた後の relax で energy が減少 (LJ backend で確認)
- Stage A frames + Stage B frames が独立に取れる、cli concat と一致

### 6.3 新規 unit tests (`tests/test_config_afir.py`)

- 新 schema parse 成功 (sn2 / proton_transfer / e2 / dummy minimal)
- 旧 schema (`[restraints]`, `k_form`, `k_broken`, `r_broken`, `r_form`) を含む TOML が `ConfigError` で reject
- `alpha_formed` scalar / list 両方 parse 成功
- `r_broken_threshold` scalar / list 両方 parse 成功
- `r_formed_threshold` 省略時 None
- length mismatch (e.g., `formed=[[1,3]]` で `alpha_formed=[1, 2]`) で `ConfigError`

### 6.4 既存 slow integration tests (`pytest -m slow`)

8 反応すべての統合テストを retune:
- `selected_trial >= 0` であり、かつ `reached_product=True` が trials 中 1 件以上
- `meta.json.early_stopped=True` の trial が大半 (Stage A が想定通り終了)
- `peak_energy` が反応によって典型的な範囲 (実装後実測で更新)
- **新規 assertion**:
  - selected_trial の **最終 frame (Stage B 最終)** で `reached_product=True`
  - selected_trial の最終 frame で全 broken pair の距離が `r_broken_threshold` 以上、全 formed pair の距離が `r_formed_threshold` 以下
  - selected_trial の Stage A frames の `min` 非結合距離が `0.5 Å` 以上 (UMA off-manifold 検出)

### 6.5 削除する tests

- `tests/test_pull_apart.py` (PullApart クラス消滅)
- `tests/test_hookean_*.py` (Hookean 直接使用箇所消滅)
- `tests/test_resolve_*targets.py` (古い resolve_*_targets ヘルパ消滅) — 該当ファイル存在確認後
- 既存 `tests/test_artificial_force.py` の旧 schema 部分 → 全削除し AFIR 用に書き直し

### 6.6 Blender smoke

`tests/test_blender_render.py` の SN2 / proton_transfer parametrized は既存通り pass を期待。Cordero 表の `blender/render.py` ↔ `reactx/covalent_radii.py` 整合性を unit test で別途確認。

## 7. 実装順序

1. **共有半径切り出し**: `reactx/covalent_radii.py` 新規、`blender/render.py` から表を移管 + import 化。重複辞書削除確認の unit test。
2. **AFIRConstraint**: `tests/test_afir_constraint.py` を **先に** TDD で書き、`reactx/artificial_force.py` を全面書き換えて pass。Hookean/PullApart/DEFAULT_R_FORM/lookup_r_form 全削除。
3. **relax_two_stage**: `tests/test_relax_two_stage.py` (LJ backend で Stage A early-stop / Stage B relax を mock 検証)。`reactx/path_relax.py` 書き換え。
4. **config 刷新**: `tests/test_config_afir.py`、`reactx/config.py` で TDD。
5. **scoring 刷新**: `reached_product` を per-pair threshold に。`tests/test_scoring.py` 更新。
6. **cli 配線**: `build_restraints` → `build_afir_constraint` 置換、`relax_with_restraints` → `relax_two_stage` 置換、`meta.json` schema 更新。
7. **example 8 個一括書き換え** (§5 表の初期値)。
8. **`pytest`** (unit + non-slow) green。
9. **`pytest -m slow`** で 1 反応ずつ走らせて α / r_*_threshold tuning。順序: sn1_dissoc → sn2 → proton_transfer → sn1_recomb → menshutkin → e2 → da_simple → da_endo。各反応で:
   - selected_trial の `early_stopped=True` を期待
   - selected_trial の最終 frame で `reached_product=True`
   - Stage A の最小非結合距離 ≥ 0.5 Å
   - Stage A の peak_energy が reasonable (前 phase 値 ± 50% 程度)
   いずれか fail なら α / r_*_threshold / FIRE param (`maxstep`, `dt`, `dtmax`) を再 tune。
10. **FIRE 設定確定**: Stage A の `maxstep` 最終値を全 8 反応で動く値に固定し、`reactx/path_relax.py` 既定値として書き込む。
11. **align.py 残骸チェック**: `grep -r align_product_to_reactant`、未使用なら削除、1+1 NEB が依存しているなら保留 (Phase 10 で扱う)。
12. **README 全面書き換え**: Phase 9 セクション、`[afir]` + `[scoring]` schema 表、wall-clock 再測定 (RTX 5070 Ti + UMA 環境で 8 反応)。
13. **Phase 9 PR**: develop ベース、TDD 由来の commit history を維持。

## 8. リスクと緩和

### 8.1 致命的に近い (Phase 9 で必ず緩和)

- **AFIR 力が product 到達後も消えない**: per-pair AFIR は距離不変。これを放置すると broken は無限に引き離され、formed は無限に縮められ、unbiased UMA energy で frames の後半が artificial に汚染される。
  **緩和**: §3.1 の Stage A early-stop (per-step `reached_product` 監視で即停止) + Stage B unbiased relax。`peak_energy` は Stage A frames のみ。slow test で `early_stopped=True` を全 selected_trial に要求。
- **per-step `reached_product` チェックが過渡的に True を返す false positive**: relax 中に bond 距離が target を一瞬越えた直後に振り戻すケース。Stage A が過渡状態で停止し、Stage B が逆向きに relax して product 未達になる可能性。
  **緩和**: ① `reached_product` 判定を **k 連続フレームで True** に厳格化 (例: k=3、`traj_stride=5` なら 15 step 安定が必要)、② Stage B 後の最終 `reached_product` 再判定で fail なら `reached_product=False` フラグで scoring に渡す (least-bad fallback で吸収)。本実装では ① を採用 (`stable_window_steps=3` を `relax_two_stage` に追加 param)。
- **Multi-pair 同時形成で片方が先行するリスク (DA asymmetric)**: per-pair で各 α 独立としたため、symmetric な α でも初期幾何の非対称性 (placement 揺らぎ) でどちらかの bond が先に形成される。先行 bond の Stage A early-stop が遅れる bond を見捨てる可能性。
  **緩和**: `reached_product` は **全 formed pair が threshold 以下、全 broken pair が threshold 以上** の AND 判定。片方だけ達成した中間状態は False を返すため early-stop しない。Stage A は遅れる bond が追いつくまで継続。
- **UMA off-manifold での NaN / 過短結合 / 意図しない新結合形成**: AFIR 一定力で押し続けると、UMA が訓練分布外の構造で異常 force を返す可能性。
  **緩和**: ① Stage A の per-step で **min 非結合距離 < 0.5 Å** の場合は trial を `error="off_manifold"` で record して abort、② slow test で `selected_trial` の Stage A frames の min 非結合距離 ≥ 0.5 Å を assert、③ UMA 例外発生時の trial-level catch (既存)。

### 8.2 中程度

- **FIRE の `maxstep` / `dt` / `dtmax` が AFIR 一定力に対して未調整**: 現行値 (`maxstep=0.1, dt=0.05, dtmax=0.2`) は Hookean のゼロ化前提で tune されている。AFIR では収束しない、または overshoot で結合形成構造を行き過ぎる可能性。
  **緩和**: 実装手順 §7 の step 9 で各反応の slow test 内で確認、step 10 で確定値を `path_relax.py` に書き込む。早期に LJ backend で試して見当をつける (UMA call の cost を避けて param sweep)。
- **scoring threshold (`r_broken_threshold`, `r_formed_threshold`) と Blender bond detection (`1.1 × covalent`) の不一致**: 例えば `r_formed_threshold = Rsum + 0.4 Å` で formed と判定されたペアが Blender 表示閾値 `1.1 × Rsum` (≒ Rsum + 0.1-0.2 Å) を超えていれば「scoring 上は形成」だが「animation 上は結合線が表示されない」状態が起き得る。
  **緩和**: ① Cordero 表を共有モジュールにして Blender / scoring で同じテーブル使用、② `BOND_TOLERANCE` (現状 1.1) と `COVALENT_FORMED_SLACK_ANGSTROM` (新規 0.4) の関係を README に明記、③ 必要なら `r_formed_threshold` の default を `1.15 × Rsum` 等の **Blender bond-tol に揃えた値** に変更 (実装時要決定)。
- **`r_broken_threshold` が `broken=[]` でも必須化される使い勝手**: DA / SN1 recomb で `r_broken_threshold = 4.0` (placeholder) を書かせるのは余計な記述。
  **緩和**: `broken=[]` のとき TOML 省略可能とする validation を入れる (実装で決定)。
- **per-pair AFIR は Maeda 原典 (multi-pair ω-collective) と異なる**: 本仕様の "AFIR" は **single-pair AFIR を pair ごとに独立適用** したもの。Maeda 2010 の原典手法とは異なるため、文献引用時の表現に注意。
  **緩和**: 仕様書 §1 / §10.1 / README で「per-pair single-pair AFIR」と明示、「Maeda-Morokuma AFIR を直接実装したものではない」と注記。

### 8.3 軽微

- **Stage B の `max_steps=50` が短すぎる場合**: Stage A 終了時の structure が product から少し離れていると、50 step では完全には relax しきらない。アニメーション末尾が「ほぼ product だが少し歪み」になる。
  **緩和**: 実装後 slow test で確認、必要なら `max_steps_b` を TOML 化 (`[afir] max_relax_steps_b = 50`)。本仕様では hardcode で開始。
- **`α=0` の no-op 機構**: `broken=[]` の DA で `alpha_broken=0.0` または `alpha_broken=[]`。前者は AFIRConstraint に格納され adjust_forces で skip、後者は broadcast で空 list。両方とも動作するが、明示的に `α=0.0` を置くか省略可能にするか統一すべき。
  **緩和**: 規約として「`broken=[]` なら `alpha_broken=0.0` を置く」を README に明記。validation は緩く (両方許容)。

## 9. 今後のロードマップ

- **Phase 10**: CI-NEB を E2 / SN1 / DA に拡張 (1+1 制約解除)、`align_product_to_reactant` を AFIR Stage A endpoint 由来に置換、`--neb-refine` を default on か否か検討
- **Phase 11**: Stage1 (AFIR screening) → top-K → Stage2 (CI-NEB) の 2 段階パイプライン、process pool 並列化
- **Phase 12**: MC-AFIR モード (`formed`/`broken` を書かなくても reaction discovery)、cycloaddition `bridges >= 3` (1,3-dipolar など)、cheletropic
- **Phase 13 候補**: per-pair AFIR を Maeda 原典の multi-pair ω-collective AFIR に **再昇格** する option を提供 (cheletropic で多 bond 同時形成の coupling が必要なら)

## 10. 改訂履歴

### 10.1 v2 改訂理由 (post-codex-review)

v1 (initial draft) は **Maeda 原典の multi-pair ω-weighted collective AFIR** (`ρ = Σ ω_ij r_ij / Σ ω_ij`) を採用していたが、external technical review で以下 2 点の中心仮定の誤りが指摘された:

1. **「`ω ∝ r⁻⁶` で自然減衰するから cutoff 不要」**: single-pair の場合 `ρ = r`、`∂ρ/∂r = 1`、force = α (定数)。多 pair でも近 pair 支配時に同様。 product 到達後も力は消えず、broken は無限に引き離され、formed は無限に縮められる。
2. **「multi-pair collective なら Phase 8 (再) の `k_form=[3.0, 3.0]` の表現力を保持できる」**: p=6 の ω weighting は近 pair が完全支配し、遠 pair の `coef = (ω/W) [(1-p) + pρ/r]` が小さく、場合によっては **負** になる。例: DA で片側 C-C が r=1.5 Å (近)、もう片側が r=3.0 Å (遠) のとき、遠 pair の coef は (small) × (-5 + 6×1.5/3.0) = (small) × (-2) で負。「form 用」のはずの力が **引き離す方向** に作用する。symmetric DA でも初期非対称が増幅される。

これらは spec の中心仮定 (§1, §3.2) を崩すため、v2 で:

- **per-pair AFIR (single-pair AFIR を pair ごとに独立適用)** に変更: 各 pair が独立に F=α·d̂、ω weighting なし。multi-bond の coupling がなくなるため遠 pair 支配や sign-flip の問題が消える。
- **Stage A early-stop (per-step `reached_product` 監視で即停止) + Stage B unbiased relax**: AFIR 力が消えない問題を Stage 構造で吸収。`peak_energy` も Stage A only で取得して artificial inflation を防ぐ。
- **`r_broken_threshold` を `[scoring]` セクションに必須化、per-bond list 対応**: E2 で C-H と C-Cl の閾値を独立指定可能に (auto-derive `(R_i+R_j)*1.5/2.5` の固定 multiplier では false positive のため)。
- **formed 閾値を `Rsum_covalent + 0.4 Å` 絶対 slack** (was `Rsum × 1.3`): Blender の `1.1 × Rsum` bond-tol と桁感が揃い、C-C 1.98 Å vs C-N 1.91 Å のような重原子結合での「緩すぎる」問題を緩和。
- **FIRE param の AFIR 用 re-tune を実装計画に明記**: `maxstep` を 0.2-0.4 で探索、最終値を `path_relax.py` に書き込む。
- **`p` exponent の TOML 露出を完全削除**: per-pair で ω なしのため p は登場しない。
- **`r→0` を fail-fast に**: 楽観的 skip ではなく `ValueError` raise (placement バグなら早期検出が正)。
- **科学的正当性の主張を tone-down**: 「product-like animation の生成」を一貫した目的とし、「mechanism の正確な再現」は CI-NEB 拡張 (Phase 10+) に委ねる。

v1 で書いた `AFIRConstraint._apply` の解析勾配 (`coef = (ω/W)[(1-p) + p ρ/r]`) は数式としては正しかったが、上記 2 点の設計判断ミスにより使われなくなる。v2 では per-pair で `F = α · d̂` の単純式のみ。

### 10.2 v1 (rejected, 2026-05-08)

- multi-pair ω-weighted collective ρ
- single `α_formed` / `α_broken` scalar (per-bond list なし)
- 1-stage relax (early-stop なし)
- `r_broken_threshold` を覆覆 covalent radii × 2.5 で auto-derive
- formed 判定 `Rsum × 1.3`
- p=6 が schema に任意 key として露出
- `r < 1e-10` skip (fail-fast でなく)
