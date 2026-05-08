# reactx Phase 9 — Per-pair AFIR with Sticky Latch + 1-stage Relax (v3)

- Status: Draft (post v2 codex review revision; awaiting user spec review)
- Date: 2026-05-08
- Owner: @kam6y
- Branch: `phase-9` (from `develop` after Phase 8 merge)
- 前提仕様: `docs/superpowers/specs/2026-05-04-generic-placement-design.md`、`docs/superpowers/specs/2026-05-05-phase-8-cycloaddition-design.md`
- 後方互換性: **完全に放棄する**。`.rxn.toml` schema・内部 API すべて breaking change を許容。既存 8 example は再 tune する。
- 改訂履歴: §10 (v1 = multi-pair ω-collective、v2 = per-pair + two-stage、**v3 = per-pair + sticky latch + 1-stage**)。

## 1. 目的

Phase Re1 〜 Phase 8 (再) で使ってきた経験的力場 (`Hookean attraction` + `PullApart repulsion`) を、**per-pair AFIR with sticky latch** に置き換え、概念統一・TOML 簡素化・Phase 8 (再) の per-bond 表現力保持・"product 到達後に押し続けない" self-extinguishing を一気に達成する。

具体的には:

- 各反応の力場ハイパラを `alpha_formed` / `alpha_broken` の **per-pair list (or scalar broadcast)** 一本に統一。
- per-pair 力 `F = α · d̂` (一定 magnitude、距離不変) を **sticky latch** で gate する: 各 pair が自分の threshold (`r_formed_threshold[k]` または `r_broken_threshold[k]`) を初めて越えた時点で latch ON、以降そのペアの AFIR force は 0 (sticky = 一度 ON なら relax 終了まで OFF にならない)。
- 全 pair が latch ON になれば AFIR 力は完全消失し、自然に unbiased な FIRE relax 状態へ移行する。Stage A/B のような 2 段構造は **不要**。relax_with_restraints (1-stage) のまま、constraint を AFIR に差し替えるだけ。
- threshold は **AFIR と scoring で共有**: `[scoring].r_formed_threshold` / `[scoring].r_broken_threshold` の値を AFIRConstraint と `reached_product` の両方が読む。これにより「scoring が True と判定する条件」=「AFIR が latch ON になる条件」となり、矛盾なく一致する。

phase-8 ブランチで挫折した「CI-NEB 汎用化 + screening + 並列化」の 3 点同時変更は本フェーズには含めない。force model + scoring 閾値の刷新のみ。

**重要な注意**: 本フェーズで生成される trajectory は AFIR-biased force によって駆動された path であり、真の MEP/TS を反映するものではない。peak_energy も approximate な指標。仕様全体で「product-like animation の生成」を一貫した目的とし、「mechanism の正確な同定」は CI-NEB 拡張 (Phase 10+) に委ねる。

## 2. Non-goals

- **CI-NEB の汎用化**: `--neb-refine` は今のとおり 1 formed + 1 broken 反応専用のオプション。E2 / SN1 / DA で NEB 必要なら future phase。
- **Screening / top-K / 並列化**: Phase 8 (旧) の Stage1→top-K→Stage2 構造、process pool 並列化は導入しない。
- **Multi-pair collective AFIR (Maeda 原典の ω-weighted ρ)**: §10.1 で破棄 (sign-flip on lagging bonds、self-extinguish 不能)。
- **MC-AFIR (reaction discovery mode)**: ユーザーが `formed`/`broken` を書かない reaction discovery は対象外。
- **新反応の追加**: 8 反応 (SN2 / PT / Menshutkin / E2 / SN1 dissoc / SN1 recomb / DA simple / DA endo) を新方式で動かすことがゴール。
- **biased PES の peak energy 報告**: AFIR は ASE FixConstraint パターンで力のみ加え、`atoms.get_potential_energy()` には影響しない。peak_energy は引き続き unbiased UMA energy。
- **FIRE param の AFIR 用 retune**: latch 機構により全 pair latch 後は AFIR 力が消えるため、Phase Re1 で UMA に対して tune された既存 FIRE 設定 (`maxstep=0.1, dt=0.05, dtmax=0.2`) のまま運用する。AFIR が active な間も constant α は ~ 0.5-4 eV/Å で UMA force と同オーダなので overshoot リスクは限定的。実装後の slow test で confirm のみ。
- **Two-stage relax (Stage A driven + Stage B unbiased)**: v2 で導入したが latch で代替できるため廃止。

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
                                                       │
                                                       ▼
            ┌── trial 1 ──┐  build_afir_constraint(formed, broken,
            ├── trial 2 ──┤                          alpha_formed[], alpha_broken[],
            ├── ...        │                          formed_thresholds[], broken_thresholds[])
            └── trial K ──┘
                                 │
                                 ▼
                  ┌─── 1-stage relax ────────────────────────┐
                  │  FIRE + AFIRConstraint (sticky latch)    │
                  │   - 各 step で per-pair の現在 r を chk  │  ── frames, energies
                  │   - threshold 越えたら latch ON、force=0 │
                  │  fmax=0.1, max_steps=cfg.afir.max_relax  │
                  │  AFIR と UMA は ASE constraint で合成    │
                  └──────────────────────────────────────────┘
                                 │
                                 ▼
              trajectory = frames                  (Stage 構造なし、単一 list)
              peak_energy = max(energies)
              reached_product = (final frame で全 pair が threshold 達成)
                                                       │
                                                       ▼
                                 scoring.score_trials      (最良 trial 選択)
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
| 力場 (formed) | `Hookean(k, rt)` per-bond (`r > rt` 時 attractive、`r ≤ rt` でゼロ) | `AFIRPair(α, threshold)` per-bond (`r > threshold` 時 `F = α·d̂`、`r ≤ threshold` で latch ON 以降 0) |
| 力場 (broken) | `PullApart(k, rt)` per-bond (`r < rt` 時 repulsive、`r ≥ rt` でゼロ) | `AFIRPair(α, threshold)` per-bond (`r < threshold` 時 `F = α·d̂`、`r ≥ threshold` で latch ON 以降 0) |
| Self-extinguish | あり (`(r-rt)` で減衰してゼロ) | あり (sticky latch でゼロ) |
| Force shape | linear in `(r - rt)` | constant magnitude `α` (距離不変) |
| TOML keys (force) | `[restraints] k_form k_broken r_broken r_form max_relax_steps` | `[afir] alpha_formed alpha_broken max_relax_steps` |
| `alpha_*` の per-bond | (Phase 8 (再) で `k_form`/`k_broken` のみ list 対応) | `alpha_formed: float \| list[float]`, `alpha_broken: float \| list[float]` 両対応 |
| TOML keys (scoring) | (`r_broken` を force と兼用) | `[scoring] r_broken_threshold (broken≠[] なら必須)`, `r_formed_threshold (任意、default = 1.15 × Rsum_covalent)` 両方 `float \| list[float]` |
| threshold の AFIR ↔ scoring 共有 | なし (force の `rt` と scoring の `r_broken_target` は別物) | **あり**: `[scoring]` の閾値が AFIRConstraint と `reached_product` 両方で使われる。同じ閾値 = 同じ判定。 |
| relax 構造 | 1 stage (`relax_with_restraints`) | 1 stage (同上、constraint だけ AFIRConstraint に差し替え)。Stage A/B 構造は v2 で検討したが v3 で破棄 (latch で代替) |
| `r_form` lookup | `DEFAULT_R_FORM` 元素表 (force と兼用) | scoring の formed 閾値にだけ使用、default `1.15 × Rsum_covalent` (Cordero 2008) |
| FIRE 設定 | `maxstep=0.1, dt=0.05, dtmax=0.2` | **変更なし**: latch で力消失するため Hookean 時代の値で問題なし。slow test で confirm |
| `peak_energy` の対象 | 全 frames | 全 frames (1-stage なので分割なし) |
| `reached_product` 評価 | 最終 frame のみ | 最終 frame のみ (1-stage の自然な定義) |

### 3.3 ファイル構成

書き換え:
- `reactx/artificial_force.py` — `Hookean`/`PullApart`/`build_restraints`/`DEFAULT_R_FORM`/`lookup_r_form` を全削除し、`AFIRConstraint` クラスと `build_afir_constraint` 関数のみ残す
- `reactx/path_relax.py` — **構造維持**。`relax_with_restraints` の API はそのまま、AFIRConstraint を `restraints` 引数に渡せるようにする (constraint base class が同じ ASE FixConstraint なので変更不要)
- `reactx/config.py` — `[restraints]` セクション削除、`[afir]` 追加、`[scoring]` 拡張。`resolve_*_targets` ヘルパは AFIR 用に簡素化
- `reactx/scoring.py` — `reached_product` を per-pair threshold ベースに、`product_distance_residual` 関数追加 (least-bad fallback 用、§4.5)
- `reactx/cli.py` — `build_restraints` 呼び出しを `build_afir_constraint` に置換、threshold 解決を AFIR 構築前に行う
- `examples/*.rxn.toml` × 8 — 新 schema に書き換え
- `tests/test_*.py` — Hookean/PullApart 系を削除、AFIR 系を追加、slow integration の期待値を再 tune
- `README.md` — Phase 9 セクション、新 schema 表、wall-clock 再測定

新規:
- `reactx/covalent_radii.py` — Cordero (2008) 共有半径 lookup (現在 `blender/render.py` 内の表を切り出す。`reactx/scoring.py` の閾値計算と Blender bond drawing で **同一テーブル** を使う)
- `tests/test_afir_constraint.py` — 解析勾配 vs 有限差分 (V=±α·r 人工 potential)、latch state 遷移、空 set の no-op、対称性、r→0 の fail-fast

削除候補:
- `reactx/align.py` の未使用部分 (Phase 8 旧で削除予定だった残骸)。`align_product_to_reactant` を 1+1 NEB が使い続けるなら **保留**。実装時に `grep -r align_product_to_reactant` で確認後決定。

## 4. コンポーネント詳細

### 4.1 `reactx/artificial_force.py` — `AFIRConstraint`

per-pair AFIR に **sticky latch** を組み合わせた ASE FixConstraint。各 pair が自分の threshold を持ち、現在距離が threshold を越えた時点で **永続的に** latch ON となり、以降その pair の AFIR 力は 0 になる。

```python
"""Per-pair AFIR force with sticky per-pair latch.

For each (i, j) pair specified in `formed` (or `broken`):

    if formed and r_ij > threshold[k] and not latched[k]:
        F_AFIR_on_j = -α[k] · d̂_ij,  d̂_ij = (x_j - x_i) / r_ij
        F_AFIR_on_i = +α[k] · d̂_ij
    elif broken and r_ij < threshold[k] and not latched[k]:
        F_AFIR_on_j = +α[k] · d̂_ij
        F_AFIR_on_i = -α[k] · d̂_ij
    else:
        # latch ON (sticky): force is zero from now until end of relax.
        latched[k] = True
        F = 0

The latch is **sticky**: once activated, it stays activated for the
remainder of the constraint's lifetime (one relax invocation). This
prevents oscillation around the threshold and provides a clean
self-extinguishing behavior — when all pairs are latched, the AFIR
contribution to forces is identically zero, and the system continues
under unbiased UMA (or LJ) potential alone.

This is *not* the multi-pair ω-weighted collective of Maeda & Morokuma
2010; see §10.1 for rationale. Each pair is an independent single-pair
AFIR collective with ρ = r_ij and ∂ρ/∂r = 1, gated by a per-pair latch.

Energy is *not* modified (ASE FixConstraint convention); peak_energy
along the trajectory therefore reflects unbiased UMA energy throughout.
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
        alpha_formed: list[float],         # length == len(formed); >= 0
        alpha_broken: list[float],         # length == len(broken); >= 0
        formed_thresholds: list[float],    # length == len(formed); > 0
        broken_thresholds: list[float],    # length == len(broken); > 0
    ):
        # length validation
        if len(alpha_formed) != len(formed):
            raise ValueError(
                f"alpha_formed length {len(alpha_formed)} != n_formed {len(formed)}")
        if len(alpha_broken) != len(broken):
            raise ValueError(
                f"alpha_broken length {len(alpha_broken)} != n_broken {len(broken)}")
        if len(formed_thresholds) != len(formed):
            raise ValueError(
                f"formed_thresholds length {len(formed_thresholds)} != n_formed")
        if len(broken_thresholds) != len(broken):
            raise ValueError(
                f"broken_thresholds length {len(broken_thresholds)} != n_broken")
        if any(a < 0 for a in alpha_formed) or any(a < 0 for a in alpha_broken):
            raise ValueError("alpha_* must be non-negative")
        if any(t <= 0 for t in formed_thresholds + broken_thresholds):
            raise ValueError("thresholds must be positive")

        self.formed = [(int(a), int(b)) for a, b in formed]
        self.broken = [(int(a), int(b)) for a, b in broken]
        self.alpha_formed = [float(x) for x in alpha_formed]
        self.alpha_broken = [float(x) for x in alpha_broken]
        self.formed_thresholds = [float(t) for t in formed_thresholds]
        self.broken_thresholds = [float(t) for t in broken_thresholds]
        # mutable per-pair latch state (initial: all False)
        self.formed_latched = [False] * len(formed)
        self.broken_latched = [False] * len(broken)

    def adjust_positions(self, atoms, newpositions):
        return  # force-only constraint

    def adjust_forces(self, atoms, forces):
        pos = atoms.positions
        # formed: compress (sign=-1) until r <= threshold, then latch
        for k, (i, j) in enumerate(self.formed):
            if self.formed_latched[k]:
                continue
            if self.alpha_formed[k] == 0.0:
                continue
            r, d_hat = self._geom(pos, i, j)
            if r <= self.formed_thresholds[k]:
                self.formed_latched[k] = True
                continue
            # apply force toward i
            f_on_j = -self.alpha_formed[k] * d_hat
            forces[j] += f_on_j
            forces[i] -= f_on_j

        # broken: expand (sign=+1) until r >= threshold, then latch
        for k, (i, j) in enumerate(self.broken):
            if self.broken_latched[k]:
                continue
            if self.alpha_broken[k] == 0.0:
                continue
            r, d_hat = self._geom(pos, i, j)
            if r >= self.broken_thresholds[k]:
                self.broken_latched[k] = True
                continue
            # apply force away from i
            f_on_j = +self.alpha_broken[k] * d_hat
            forces[j] += f_on_j
            forces[i] -= f_on_j

    @staticmethod
    def _geom(pos, i, j) -> tuple[float, np.ndarray]:
        v = pos[j] - pos[i]
        r = float(np.linalg.norm(v))
        if r < 1e-6:
            # Coincident atoms indicate a placement bug — fail fast.
            raise ValueError(
                f"AFIRConstraint: atoms {i} and {j} coincide (r={r:.2e}); "
                f"check fragment placement geometry"
            )
        return r, v / r

    def all_latched(self) -> bool:
        return all(self.formed_latched) and all(self.broken_latched)

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
                "formed_thresholds": self.formed_thresholds,
                "broken_thresholds": self.broken_thresholds,
            },
        }


def build_afir_constraint(
    atoms: Atoms,
    formed: list[tuple[int, int]],
    broken: list[tuple[int, int]],
    *,
    alpha_formed: float | list[float],
    alpha_broken: float | list[float],
    formed_thresholds: list[float],   # already resolved per-pair, see scoring.py
    broken_thresholds: list[float],
) -> list[AFIRConstraint]:
    """Broadcast scalar α to per-pair lists, return 0- or 1-element list."""
    if not formed and not broken:
        return []
    af = _broadcast(alpha_formed, len(formed), key="alpha_formed")
    ab = _broadcast(alpha_broken, len(broken), key="alpha_broken")
    return [AFIRConstraint(
        formed, broken,
        alpha_formed=af, alpha_broken=ab,
        formed_thresholds=formed_thresholds,
        broken_thresholds=broken_thresholds,
    )]


def _broadcast(value: float | list[float], n: int, *, key: str) -> list[float]:
    if isinstance(value, list):
        if len(value) != n:
            raise ValueError(f"{key} list length {len(value)} != n_pairs {n}")
        return [float(v) for v in value]
    return [float(value)] * n
```

**重要**:
- `α = 0` の pair は force=0 (latch とは独立)。これにより `broken=[]` の DA で `alpha_broken=0.0` (scalar broadcast → `[]`) または冗長に `alpha_broken=[]` を渡しても動作する。
- `r < 1e-6` (1 µÅ) で **fail fast**。原子が重なっているのは placement バグ。
- `adjust_forces` は energy には触れない (ASE FixConstraint 規約) → unbiased UMA peak_energy が保たれる。
- `formed_latched` / `broken_latched` は **AFIRConstraint instance の lifetime に紐付く** (1 trial の relax セッション中で sticky)。次の trial では新規 instance を生成するためリセットされる。
- threshold は scoring と完全に **共有**: AFIR が latch ON する条件 = scoring の `reached_product` が True と判定する条件。同じ値を使うため矛盾なし。

### 4.2 `reactx/path_relax.py` — `relax_with_restraints` (構造維持)

v2 では `relax_two_stage` への書き換えを提案したが、v3 では latch 機構が AFIR 力の self-extinguishing を実現するため Stage 構造は不要。Phase Re1 以来の `relax_with_restraints` をそのまま使う。

```python
def relax_with_restraints(
    atoms: Atoms,
    restraints: list,           # [] or [AFIRConstraint] (length 0 or 1)
    calc: Calculator,
    *,
    max_steps: int = 100,
    fmax: float = 0.1,
    traj_stride: int = 5,
) -> tuple[list[Atoms], list[float]]:
    ...   # 既存実装そのまま
```

実装上の変更点:
- AFIRConstraint も ASE FixConstraint なので、`atoms.set_constraint(restraints)` がそのまま動作 (新規対応コードなし)。
- 全 pair 用 latch ON で AFIR force=0 になり、UMA force のみで relax が継続。`fmax=0.1` を満たすと FIRE が converged で停止する。`max_steps` 上限は安全弁。

### 4.3 `reactx/covalent_radii.py` — Cordero 表の切り出し

```python
"""Cordero (2008) covalent radii in Å.

Used by:
- `reactx/scoring.py` for reached_product threshold defaults
- `reactx/cli.py` for resolving formed thresholds from scoring config
- `blender/render.py` for bond drawing distance threshold

Reference: Cordero et al., Dalton Trans. 2008, 2832.
"""

from __future__ import annotations
import numpy as np
from ase.atoms import Atoms

CORDERO_2008: dict[str, float] = {
    "H": 0.31, "He": 0.28,
    "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66,
    "F": 0.57, "Ne": 0.58,
    # ... (full table identical to existing blender/render.py COVALENT_RADII_ANGSTROM,
    #     up to Bi Z=83)
}

def cordero_radius(symbol: str, *, default: float = 1.5) -> float:
    return CORDERO_2008.get(symbol, default)

def cordero_radii_for_atoms(atoms: Atoms) -> np.ndarray:
    return np.array([cordero_radius(s) for s in atoms.get_chemical_symbols()])
```

`blender/render.py` も import に切り替えて重複辞書を排除。Blender 環境で `reactx` パッケージが import できない場合の `try/except ImportError` fallback (現状パターン) は維持。

### 4.4 `reactx/config.py` — TOML schema 変更

新 schema (`examples/sn2.rxn.toml`):

```toml
description = "SN2 anion: CH3Cl + OH- -> CH3OH + Cl-"
formed = [[1, 3]]
broken = [[1, 2]]

[afir]
alpha_formed = 0.7      # eV/Å, scalar (positive: 縮める方向、内部で sign=-1)
alpha_broken = 0.5      # eV/Å, scalar (positive: 引き離す方向、内部で sign=+1)
max_relax_steps = 100

[scoring]
r_broken_threshold = 4.0   # broken=[] なら省略可。AFIR latch と reached_product で共有
# r_formed_threshold = ...  # 省略時 default = 1.15 * Rsum_covalent (per-pair auto)
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
alpha_broken = 0.0          # broken=[] なので使われない (省略可)
max_relax_steps = 200

[scoring]
# broken=[] のため r_broken_threshold は省略 (記述するとエラーではないが、無視される)
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
    r_broken_threshold: float | list[float] | None = None
    r_formed_threshold: float | list[float] | None = None  # None = auto default

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
- `alpha_formed` / `alpha_broken`:
  - scalar `>= 0` ⇒ broadcast to `len(formed)` / `len(broken)`
  - list `>= 0` で length が `formed` / `broken` と一致 (`broken=[]` なら `[]` のみ valid)
  - **特例**: `broken=[]` のとき `alpha_broken = 0.0` scalar、`alpha_broken = []` list、`alpha_broken` 省略のいずれも valid (broadcast の結果同じく `[]`)
- `max_relax_steps > 0`
- `r_broken_threshold`:
  - `broken=[]` ⇒ 省略可、scalar / list を渡しても warning なしで無視
  - `broken≠[]` ⇒ **必須**。scalar `> 0` または list `> 0` で length が `broken` と一致
- `r_formed_threshold`:
  - 省略可。省略時は per-pair auto-default `1.15 * (R_i + R_j)_Cordero` (Blender bond-tol `1.1 × Rsum` よりわずかに緩めて transient stretch 許容)
  - 明示時は scalar `> 0` または list `> 0` で length が `formed` と一致
- 旧 keys (`k_form`, `k_broken`, `r_broken`, `r_form`、`[restraints]` セクション) が出現したら `ConfigError` で reject

### 4.5 `reactx/scoring.py` — `reached_product` + `product_distance_residual`

`reached_product` を per-pair threshold ベースに再実装:

```python
COVALENT_FORMED_TOLERANCE = 1.15   # default formed threshold = 1.15 * Rsum_Cordero


def resolve_formed_thresholds(atoms, formed, override) -> list[float]:
    """Per-pair formed thresholds (Å). None override → 1.15 * Rsum_covalent."""
    if not formed:
        return []
    cov = cordero_radii_for_atoms(atoms)
    if override is None:
        return [(cov[i] + cov[j]) * COVALENT_FORMED_TOLERANCE
                for (i, j) in formed]
    if isinstance(override, list):
        if len(override) != len(formed):
            raise ConfigError(
                f"r_formed_threshold list length {len(override)} != n_formed")
        return [float(x) for x in override]
    return [float(override)] * len(formed)


def resolve_broken_thresholds(atoms, broken, override) -> list[float]:
    """Per-pair broken thresholds (Å). Required when broken != []."""
    if not broken:
        return []
    if override is None:
        raise ConfigError(
            "r_broken_threshold required when broken bonds are specified")
    if isinstance(override, list):
        if len(override) != len(broken):
            raise ConfigError(
                f"r_broken_threshold list length {len(override)} != n_broken")
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


def product_distance_residual(
    atoms: Atoms,
    formed: list[tuple[int, int]],
    broken: list[tuple[int, int]],
    formed_thresholds: list[float],
    broken_thresholds: list[float],
) -> float:
    """Sum of squared violations from per-pair thresholds (Å²).

    Used as a tiebreaker in least-bad fallback when no trial reaches
    product. A trial closer to the product manifold gets a smaller
    residual. Residual = 0 iff reached_product=True.

        residual = Σ_formed max(d - thr, 0)² + Σ_broken max(thr - d, 0)²
    """
    pos = atoms.positions
    res = 0.0
    for (i, j), thr in zip(formed, formed_thresholds, strict=True):
        d = float(np.linalg.norm(pos[j] - pos[i]))
        res += max(d - thr, 0.0) ** 2
    for (i, j), thr in zip(broken, broken_thresholds, strict=True):
        d = float(np.linalg.norm(pos[j] - pos[i]))
        res += max(thr - d, 0.0) ** 2
    return res
```

`score_trials` 更新:
- 第一優先: `reached_product=True` の trial 内で `peak_energy` 最小
- 第二優先 (least-bad fallback、全 trial 未達): **`product_distance_residual` 最小** の trial。残差が同じなら `peak_energy` 最小。これにより「低 energy だが product から遠い」trial を選ぶリスクを軽減する。

`TrialResult` (v8 で `ScreeningTrialResult` に rename された物の系譜) に `product_distance_residual: float` フィールド追加。

### 4.6 `reactx/cli.py` — orchestration

per-trial flow:

```python
# threshold 解決 (scoring と AFIR で共有)
ft = resolve_formed_thresholds(atoms_init, formed, cfg.scoring.r_formed_threshold)
bt = resolve_broken_thresholds(atoms_init, broken, cfg.scoring.r_broken_threshold)

# AFIR constraint 構築 (threshold 込み)
afir_cs = build_afir_constraint(
    atoms_init, formed, broken,
    alpha_formed=cfg.afir.alpha_formed,
    alpha_broken=cfg.afir.alpha_broken,
    formed_thresholds=ft,
    broken_thresholds=bt,
)

# 既存の relax_with_restraints をそのまま呼ぶ
frames, energies = relax_with_restraints(
    atoms_init, afir_cs, calc,
    max_steps=cfg.afir.max_relax_steps,
    fmax=relax_fmax,
    traj_stride=traj_stride,
)

# 最終 frame で reached_product と residual 評価
final = frames[-1]
ok = reached_product(final, formed, broken, ft, bt)
residual = product_distance_residual(final, formed, broken, ft, bt)
peak_energy = max(energies)

trial = ScreeningTrialResult(
    ..., reached_product=ok,
    peak_energy=peak_energy,
    product_distance_residual=residual,
    n_steps=len(frames),
    ...,
)
```

`meta.json` per-trial エントリ更新:
- 削除: `r_broken_target`, `r_form_target`, `k_form`, `k_broken`
- 追加: `alpha_formed: list[float]`, `alpha_broken: list[float]`, `formed_thresholds: list[float]`, `broken_thresholds: list[float]`, `product_distance_residual: float`, `formed_latch_count: int`, `broken_latch_count: int` (relax 終了時の latch ON 数、デバッグ用)

NEB refine path (`--neb-refine`、1+1 のみ):
- **既存実装維持**: NEB endpoint は引き続き reactant 軌跡フレーム + embedded/aligned product から取得 (`align_product_to_reactant` が必要)。
- AFIR trajectory の最終 frame を product endpoint に直接使う変更は **本フェーズでは行わない** (Phase 10 で扱う)。
- NEB の入力 reactant frame は引き続き `frames[0]` を使う、product frame は別途 product-side の placement を `BondChanges(formed=broken_orig, broken=formed_orig)` で生成する既存パスを維持する。

## 5. 8 example の AFIR 初期 tuning 値

per-pair AFIR は「step あたりの押し速度 = `α × dt × maxstep`」が Hookean 時代と同オーダになる経験値。FIRE 設定 (`maxstep=0.1, dt=0.05`) を維持するため、α は eV/Å オーダで Hookean の `k × (typical r-rt) ~ k × 1` と同程度の値を目安にする。実装後 slow test で再調整。

| Reaction | α_formed | α_broken | max_relax_steps | r_broken_threshold | r_formed_threshold | 備考 |
|---|---|---|---|---|---|---|
| sn2 | 0.7 | 0.7 | 100 | 4.0 | (default) | C-Cl 切断 |
| proton_transfer | 1.5 | 1.0 | 100 | 4.0 | 1.5 (短い O-H 用) | H-Cl 切断、O-H 形成 |
| menshutkin | 4.0 | 1.5 | 200 | 5.0 | (default) | gas-phase repulsive PES |
| e2 | 1.5 | [1.0, 1.5] | 200 | [3.0, 4.0] | (default) | 2 broken: C-H, C-Cl |
| sn1_dissoc | 0.0 | 1.5 | 200 | 6.0 | (formed=[] のため default 不使用) | unimolecular |
| sn1_recomb | 1.5 | 0.0 | 200 | (broken=[] で省略) | (default) | bimolecular formation |
| diels_alder_simple | [2.5, 2.5] | 0.0 | 200 | (broken=[] で省略) | (default) | symmetric DA |
| diels_alder_endo | [2.5, 2.5] | 0.0 | 250 | (broken=[] で省略) | (default) | endo/exo CP+MA |

注:
- Phase 8 (再) の `k_form = [3.0, 3.0]` (per-bond list) → α_formed = `[2.5, 2.5]` で per-bond list を保持。symmetric な DA (butadiene + ethylene) では同値、初期幾何の対称性は維持されるが、UMA force・FIRE step の微小非対称が増幅される可能性は残る (latch 機構により先行 bond は早期 latch されるため、過圧縮による off-manifold 化は抑制される)。
- E2 の C-H broken は `r_broken_threshold = 3.0` (Cordero `1.07 × 2.5 = 2.68` より厳しめ)、C-Cl は `4.0` (Cordero `1.78 × 2.0 = 3.56` よりやや緩め) で経験値。
- DA の `broken=[]` は `r_broken_threshold` 省略可、`alpha_broken = 0.0` (scalar broadcast → `[]`) で AFIR には何も渡らない。

## 6. テスト戦略

### 6.1 新規 unit tests (`tests/test_afir_constraint.py`)

- **解析勾配 vs 有限差分 (V=±α·r 人工 potential)**: AFIRConstraint の `adjust_forces` 出力を、人工 potential `V_pair = (sign × α) × r_ij` (formed: sign=+1、broken: sign=-1) を有限差分した数値勾配と比較し、相対誤差 1e-6 以下で一致 (per-pair の閉形式は線形なので有限差分と完全一致するはず)。**ASE potential energy は使わない** (AFIRConstraint は energy には触れないため)。
- **単一 pair (formed)**: `formed=[(0,1)], alpha_formed=[α], formed_thresholds=[t]`、初期 `r > t` で force_on_1 = `-α · d̂_01`、force_on_0 = `+α · d̂_01`。
- **単一 pair (broken)**: 上の broken 版。
- **Latch transition (formed)**: `r > t` で force on、`r ≤ t` で latch ON、以降 force=0、再度 `r > t` に戻っても force=0 (sticky)。
- **Latch transition (broken)**: `r < t` で force on、`r ≥ t` で latch ON、sticky。
- **Latch initial state**: 初期 `r ≤ t` で latch 即 ON、初回 `adjust_forces` で force=0。
- **`α=0` の no-op**: latch 状態に関わらず force=0 (latch も ON にしない、無関係)。
- **空 set no-op**: `formed=[], broken=[]` で `build_afir_constraint` が `[]` を返す。
- **per-pair 独立性**: 2 pair で別々の α・threshold を指定したとき、pair 1 の latch が ON でも pair 2 は continued。
- **対称性**: `(i, j)` と `(j, i)` で同じ力 (順序不変性)。
- **r→0 fail-fast**: 同位置の 2 atom で `ValueError`。
- **負 α 拒否**: `alpha_formed=[-0.1]` で `ValueError`。
- **負/ゼロ threshold 拒否**: `formed_thresholds=[0]` または `[-1]` で `ValueError`。
- **broadcasting**: scalar `alpha_formed=0.7` と `[0.7]*n` が同等。
- **`all_latched()` helper**: 全 pair latch ON で True、いずれか OFF で False、空 set で True (空集合の AND)。
- **todict round-trip**: `AFIRConstraint(**c.todict()['kwargs'])` で同等の constraint が作れる (latch 状態は再構築時に reset される、これは仕様)。

### 6.2 新規 unit tests (`tests/test_config_afir.py`)

- 新 schema parse 成功 (sn2 / proton_transfer / e2 / da_simple / sn1_dissoc / dummy minimal)
- 旧 schema (`[restraints]`, `k_form`, `k_broken`, `r_broken`, `r_form`) を含む TOML が `ConfigError` で reject
- `alpha_formed` scalar / list 両方 parse 成功
- `r_broken_threshold` scalar / list 両方 parse 成功
- `broken=[]` で `r_broken_threshold` 省略可、明示 0 / list / scalar 渡しても warning なしで無視
- `broken≠[]` で `r_broken_threshold` 省略すると `ConfigError`
- `r_formed_threshold` 省略時は None (default 解決は scoring 側で)
- length mismatch (e.g., `formed=[[1,3]]` で `alpha_formed=[1, 2]`) で `ConfigError`
- `alpha_broken=0.0` (scalar) と `broken=[]` の組み合わせが valid

### 6.3 新規 unit tests (`tests/test_scoring.py` 追加)

- `resolve_formed_thresholds`: None override → `1.15 × Rsum`、scalar → broadcast、list → passthrough、length mismatch → `ConfigError`
- `resolve_broken_thresholds`: `broken=[]` → 空 list、`broken≠[]` で None override → `ConfigError`、scalar/list passthrough
- `reached_product`: per-pair threshold で True/False の境界テスト
- `product_distance_residual`: reached のとき 0、formed が一つ届かない時 `(d - thr)²`、broken が一つ離れない時 `(thr - d)²`、混在時の和

### 6.4 既存 slow integration tests (`pytest -m slow`)

8 反応すべての統合テストを retune:
- `selected_trial >= 0` で `reached_product=True` が 1 件以上
- selected_trial の **最終 frame で `reached_product=True`** (latch all ON)
- selected_trial の `meta.json.formed_latch_count == len(formed)` (全 formed pair が latch ON)
- selected_trial の `meta.json.broken_latch_count == len(broken)` (全 broken pair が latch ON)
- **新規 assertion**: selected_trial の trajectory 全 frame での **min 非結合 atom-pair 距離 ≥ 0.5 Å**。「非結合 pair」は以下を**除外**した残り: ① 初期 frame で `d_ij ≤ 1.1 × Rsum_Cordero` の pair (= initial bonded)、② `formed` / `broken` リストに含まれる pair、③ 同じ初期 fragment 内の任意 pair (placement で同 fragment は内部結合済とみなす)。実装ヘルパは `tests/conftest.py` に置く。

### 6.5 削除する tests

- `tests/test_pull_apart.py` (PullApart クラス消滅)
- `tests/test_hookean_*.py` (Hookean 直接使用箇所消滅)
- `tests/test_resolve_*targets.py` (古い resolve_*_targets ヘルパ消滅) — 該当ファイル存在確認後
- 既存 `tests/test_artificial_force.py` の旧 schema 部分 → 全削除し AFIR 用に書き直し

### 6.6 Blender smoke

`tests/test_blender_render.py` の SN2 / proton_transfer parametrized は既存通り pass を期待。Cordero 表の `blender/render.py` ↔ `reactx/covalent_radii.py` 整合性を unit test で別途確認 (`assert reactx.covalent_radii.CORDERO_2008 == blender.render.COVALENT_RADII_ANGSTROM` のような同値性テスト、または `blender/render.py` 側が `from reactx.covalent_radii import CORDERO_2008` で fallback なしの直接 import)。

## 7. 実装順序

1. **共有半径切り出し**: `reactx/covalent_radii.py` 新規、`blender/render.py` から表を移管 + import 化。重複辞書削除。
2. **AFIRConstraint**: `tests/test_afir_constraint.py` を **先に** TDD で書き、`reactx/artificial_force.py` を全面書き換えて pass。Hookean / PullApart / DEFAULT_R_FORM / lookup_r_form 全削除。**latch 機構の挙動 (sticky、per-pair 独立、α=0 の no-op、初期 latch、対称性)** を unit test で網羅。
3. **scoring 刷新**: `tests/test_scoring.py` 拡張、`reactx/scoring.py` で `resolve_formed_thresholds` / `resolve_broken_thresholds` / `reached_product` / `product_distance_residual` を実装 + `score_trials` の least-bad fallback に residual を組み込む。
4. **config 刷新**: `tests/test_config_afir.py`、`reactx/config.py` で TDD。旧 keys reject も含む。
5. **cli 配線**: `build_restraints` → `build_afir_constraint`、threshold 解決の追加、`meta.json` schema 更新 (`product_distance_residual`、`*_latch_count`)。
6. **example 8 個一括書き換え** (§5 表の初期値)。
7. **`pytest`** (unit + non-slow) green。
8. **`pytest -m slow`** で 1 反応ずつ走らせて α / r_*_threshold tuning。順序: sn1_dissoc → sn2 → proton_transfer → sn1_recomb → menshutkin → e2 → da_simple → da_endo。各反応で:
   - selected_trial の **最終 frame `reached_product=True`**
   - selected_trial の **全 latch ON** (= `*_latch_count` が full)
   - **min 非結合距離 ≥ 0.5 Å** (UMA off-manifold 検出)
   - peak_energy が reasonable (前 phase 値 ± 50% 程度)
   いずれか fail なら α / r_*_threshold を再 tune。FIRE param は基本変更しない (latch で力消失するため Hookean 時代の値で動く想定)、必要なら最終手段として `maxstep` 微調整。
9. **align.py 残骸チェック**: `grep -r align_product_to_reactant`、未使用なら削除、1+1 NEB が依存しているなら保留 (Phase 10 で扱う)。
10. **README 全面書き換え**: Phase 9 セクション、`[afir]` + `[scoring]` schema 表、wall-clock 再測定 (RTX 5070 Ti + UMA 環境で 8 反応)。
11. **Phase 9 PR**: develop ベース、TDD 由来の commit history を維持。

## 8. リスクと緩和

### 8.1 致命的に近い

- **Latch threshold 越え後、UMA force で再度 threshold 内に押し戻されるリスク**: 例えば broken bond が threshold で latch ON した直後、UMA の弱い vdW 引力で原子が threshold より近距離に戻る。Sticky latch なので AFIR は再駆動しないが、最終 frame で `reached_product` が False になる。
  **緩和**: ① threshold を保守的に (UMA vdW well より十分遠く) 設定、② slow test で各反応の selected_trial の `reached_product` を最終 frame で確認、③ 必要なら threshold を 0.5 Å 程度上に再 tune。formed 側は UMA bonding の minimum (e.g., 1.5 Å) より外で latch する (1.15 × Rsum ≈ 1.85 Å for C-C は適切)。

- **Multi-pair 同時形成で先行 bond が未 latch の遅い bond を待つ間に過圧縮するリスク**: v2 で重要視されたが、v3 では **latch がこれを直接解決**する (先行 bond は threshold 越えた瞬間に latch ON、以降 AFIR force = 0)。残るリスクは: 先行 bond が latch ON した後、**UMA bonding force だけ**で強い引力が継続し、bond 距離が threshold より大幅に短くなる可能性。
  **緩和**: ① latch 後の bond 距離は UMA の equilibrium length (e.g., C-C 1.5 Å) に向かう。これは threshold (`1.15 × 1.52 = 1.75 Å` for C-C) より短いが、UMA の anharmonic well が制限するため non-physical な過圧縮は起きにくい、② slow test で **全 trajectory frame** の min 非結合距離 ≥ 0.5 Å を assert (off-manifold 検出)。

- **DA で symmetric α でも初期幾何の僅かな非対称が増幅されるリスク**: placement の Fibonacci サンプリング誤差や FIRE step の数値誤差により、symmetric DA (butadiene + ethylene) でも片側 C-C が先に latch ON する可能性。これは latch 機構の設計上避けられず、結果として asymmetric concerted 様の trajectory になり得る。
  **緩和**: ① 動画として asymmetric concerted も physically な反応経路の一つ、必ずしも問題ではない、② symmetric な animation を強制したい場合は将来 phase で「全 formed pair が同時に latch するまで個別 latch を保留」する synchronized latch オプションを検討。本 phase では実装しない。

- **UMA off-manifold での NaN / 過短結合 / 意図しない新結合形成**: AFIR + UMA の非物理構造 (訓練分布外) で UMA が異常 force を返す可能性。
  **緩和**: ① 既存の trial-level UMA 例外 catch、② slow test の min 非結合距離 ≥ 0.5 Å assertion、③ 必要なら UMA force magnitude の sanity check (例: 50 eV/Å 超で trial を error 扱い) を `relax_with_restraints` 内で追加 (本 phase 必須ではない)。

### 8.2 中程度

- **`peak_energy` Stage A only と least-bad fallback の相性**: v3 では Stage 構造廃止のため peak_energy は全 frame で計算。問題は reached=False の trial で「low energy だが product から遠い」を選ぶリスク。
  **緩和**: §4.5 で `product_distance_residual` を導入し、least-bad fallback では **residual 最小** を第一基準、tied なら peak_energy 最小。これにより「product に近づいているが energy が高い」trial が「遠いが energy が低い」trial に勝つ。
- **Cordero `1.15 × Rsum` (新 default formed threshold) と Blender `1.1 × Rsum` (bond drawing tol) の不整合**: scoring で formed と判定された結合が Blender で表示されないケース。差は 0.05 × Rsum ≈ 0.05-0.1 Å で軽微だが、edge case でズレる。
  **緩和**: Cordero 表を共有モジュール化 (§4.3) し、`BOND_TOLERANCE = 1.1` と `COVALENT_FORMED_TOLERANCE = 1.15` の関係を README に明記。実装時に Blender 側を 1.15 に揃える (animation 上の bond 表示を scoring と一致させる) 選択も検討、ただし bond drawing が緩くなりすぎる懸念があるため slow test で確認後決定。
- **`α=0` の no-op と空 list の同義性**: `broken=[]` で `alpha_broken=0.0` (scalar broadcast → `[]`)、`alpha_broken=[]` (明示空 list)、`alpha_broken` 省略 (default 0) のいずれも valid。**実装と validation の両方で正規化** (`_broadcast(0.0, 0)` が `[]` を返す、`alpha_broken=[]` をそのまま受け付ける、省略時は `0.0` を default とする) する必要がある。
  **緩和**: `tests/test_config_afir.py` で 3 通り全てを test、`build_afir_constraint` の `_broadcast` を symmetric に書く (どの入力でも長さ `n` の list が返る)。
- **既存 `align_product_to_reactant` への依存と NEB refine 経路**: 現行 cli は NEB refine 時に reactant frame と embedded/aligned product を NEB endpoint としており、AFIR Stage A/B endpoint を直接使わない。
  **緩和**: 本 phase では既存実装を **そのまま維持**。AFIR endpoint を NEB に直接渡す変更は Phase 10 で扱う。`reactx/align.py` の削除は NEB 系がそれを使い続ける限り保留。

### 8.3 軽微

- **Latch 状態の serialization**: `AFIRConstraint.todict()` は latch 状態を含まない (kwargs のみ)。restore 時は latch リセット。問題は ASE の trajectory 書き戻しで constraint を保存・復元する場合のみで、本 phase では trajectory.xyz 出力は constraint なしの atoms のみなので問題なし。
- **空 list の AND は True (空集合論)**: `all_latched()` は `formed=[]` & `broken=[]` で True を返す。これは仕様上問題ないが、empty constraint は `build_afir_constraint` で `[]` を返すので呼ばれない経路。
- **Stage 構造廃止による未到達 trial の trajectory**: latch ON しないまま max_steps で終了した trial の trajectory は AFIR に押され続けたまま。peak_energy は inflate されるが、`product_distance_residual` で fallback 判定すれば問題なし。
- **`r_form` lookup 表 (`DEFAULT_R_FORM`) の削除**: Phase 8 (再) で formed bond 距離 default lookup に使われていた元素表は削除、Cordero 共有半径から動的計算に。`r_formed_threshold` の default が表 lookup ではなく `1.15 × Rsum_covalent` 計算に統一されることで、コードベースから tribal knowledge dictionary が一個減る。

## 9. 今後のロードマップ

- **Phase 10**: CI-NEB を E2 / SN1 / DA に拡張 (1+1 制約解除)、`align_product_to_reactant` の AFIR endpoint 直接利用への移行検討、`--neb-refine` を default on か否か検討
- **Phase 11**: Stage1 (AFIR screening) → top-K → Stage2 (CI-NEB) の 2 段階パイプライン、process pool 並列化
- **Phase 12**: MC-AFIR モード (formed/broken を書かなくても reaction discovery)、cycloaddition `bridges >= 3` (1,3-dipolar など)、cheletropic
- **Phase 13 候補**: synchronized latch (DA など symmetric concerted を強制したいときの「全 formed pair 同時 latch」オプション)、Maeda 原典の multi-pair ω-collective AFIR を **オプションとして** 提供 (cheletropic で多 bond 同時形成の coupling が必要なら)

## 10. 改訂履歴

### 10.1 v1 → v2 改訂理由 (post-codex-review)

v1 (initial draft) は **Maeda 原典の multi-pair ω-weighted collective AFIR** (`ρ = Σ ω_ij r_ij / Σ ω_ij`) を採用していたが、external technical review で以下 2 点の中心仮定の誤りが指摘された:

1. **「`ω ∝ r⁻⁶` で自然減衰するから cutoff 不要」**: single-pair の場合 `ρ = r`、`∂ρ/∂r = 1`、force = α (定数)。多 pair でも近 pair 支配時に同様。product 到達後も力は消えず、broken は無限に引き離され、formed は無限に縮められる。
2. **「multi-pair collective なら Phase 8 (再) の `k_form=[3.0, 3.0]` の表現力を保持できる」**: p=6 の ω weighting は近 pair が完全支配し、遠 pair の `coef = (ω/W) [(1-p) + pρ/r]` が小さく、場合によっては **負** になる。例: DA で片側 C-C が r=1.5 Å (近)、もう片側が r=3.0 Å (遠) のとき、遠 pair の coef は (small) × (-5 + 6×1.5/3.0) = (small) × (-2) で負。「form 用」のはずの力が **引き離す方向** に作用する。

v2 では以下に再設計した:
- per-pair AFIR (single-pair AFIR を pair ごとに独立適用)
- Stage A early-stop (per-step `reached_product` 監視) + Stage B unbiased relax の **2-stage 構造**
- `r_broken_threshold` を `[scoring]` 必須化、per-bond list 対応
- formed 閾値を `Rsum + 0.4 Å` 絶対 slack
- FIRE param の AFIR 用 retune を実装計画に明記
- p exponent 削除、r→0 fail-fast

### 10.2 v2 → v3 改訂理由 (second codex review)

v2 は v1 の中心的な誤りを解消したが、以下の残課題が指摘された:

- `reached_product` の最終判定が図 (Stage A early-stop OR final B passes) と §8.1 (Stage B 後に fail なら False) で矛盾
- `stable_window_steps=3` がリスク章にだけ書かれ、API/test に未反映
- `r_broken_threshold` `broken=[]` で必須/省略可が §4.4 ⟷ §8.2 で矛盾
- `alpha_broken=[0.0,...]` 許容するかが §4.1 ⟷ §4.4 で矛盾
- multi-pair 同時形成の緩和が弱い (AND 判定で early-stop は防げるが、遅い bond を待つ間に先行 bond が押されすぎて過短結合・UMA off-manifold)
- `peak_energy` Stage A only と least-bad fallback の相性悪く、未達時の選別が「低 energy だが product から遠い」を選びやすい
- Stage B energy 単調減少前提が強すぎ (FIRE は厳密 line-search でない)
- 「symmetric α なら対称性の崩れは生じない」は過剰主張
- `r_formed_threshold = Rsum + 0.4 Å` が Blender の `1.1 × Rsum` より緩く、scoring 上 formed でも bond line 非表示の可能性
- Stage A 最終 frame と Stage B 初期 frame の concat 重複
- 解析勾配 vs 有限差分 test の対象が ASE energy ではなく人工 potential `V=±α·r` であるべき
- min 非結合距離測定の対象 pair 未定義
- NEB refine 説明が既存 CLI と不整合

v3 では **per-pair sticky latch + 1-stage relax** に再設計し、以下を解決した:

- **Stage A/B 構造廃止**: latch が AFIR force の self-extinguishing を実現するため Stage 構造不要。Phase Re1 以来の `relax_with_restraints` (1-stage) のまま。
- **threshold を AFIR と scoring で共有**: `[scoring].r_*_threshold` の値を AFIRConstraint と `reached_product` 両方で使い、矛盾を構造的に排除。
- **multi-pair 過圧縮を latch で直接緩和**: 先行 bond が threshold 越えた瞬間に latch ON、以降 AFIR force = 0。残るのは UMA force だけで、UMA の anharmonic well により非物理的な過圧縮は抑制。
- **`product_distance_residual` を least-bad fallback に追加**: peak_energy 単独だと「低 energy だが product から遠い」を選ぶリスクを軽減。
- **FIRE retune 不要**: latch で力消失するため Phase Re1 の値で動く想定。実装後 confirm のみ。
- **`r_broken_threshold` `broken=[]` で省略可** に統一。
- **`r_formed_threshold` default を `1.15 × Rsum`** (Blender bond-tol 1.1× にやや余裕、ズレ 0.05 × Rsum 程度)。
- **`alpha_broken` の `broken=[]` 取り扱い** を 3 形式 (scalar 0、空 list、省略) すべて valid と明記。
- **解析勾配 test を `V=±α·r` 人工 potential に対する有限差分** と明記。
- **min 非結合距離 assertion の対象** を「初期 bonded、formed/broken、同 fragment」を除外した残り、と固定。
- **NEB refine 既存実装維持**: AFIR endpoint 直接利用は Phase 10 へ。
- 「symmetric α なら対称性崩れない」記述を削除。

### 10.3 v1 (rejected, 2026-05-08)

multi-pair ω-weighted collective ρ、single `α_formed`/`α_broken` scalar、1-stage relax (early-stop なし)、`r_broken_threshold` を covalent radii × 2.5 で auto-derive、formed 判定 `Rsum × 1.3`、p=6 が schema に任意 key として露出、`r < 1e-10` skip。

### 10.4 v2 (rejected, 2026-05-08)

per-pair AFIR、Stage A (early-stop) + Stage B (unbiased relax 50 steps)、`stable_window_steps=3` の言及のみ、`r_broken_threshold` 必須/省略可の矛盾、AFIR が Stage A 中に過圧縮する緩和不十分、`peak_energy` Stage A only と least-bad fallback の相性問題、frame 境界重複、Blender bond-tol との不一致 (`Rsum + 0.4 Å`)。
