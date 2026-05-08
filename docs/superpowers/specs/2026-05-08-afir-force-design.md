# reactx Phase 9 — AFIR Force Replacement (incremental, minimal scope)

- Status: Draft (awaiting user spec review)
- Date: 2026-05-08
- Owner: @kam6y
- Branch: `phase-9` (from `develop` after Phase 8 merge)
- 前提仕様: `docs/superpowers/specs/2026-05-04-generic-placement-design.md`、`docs/superpowers/specs/2026-05-05-phase-8-cycloaddition-design.md`
- 後方互換性: **完全に放棄する**。`.rxn.toml` schema・内部 API すべて breaking change を許容。既存 8 example は再 tune する。

## 1. 目的

Phase Re1 〜 Phase 8 (再) で使ってきた経験的力場 (`Hookean attraction` + `PullApart repulsion`) を、Maeda-Morokuma の **AFIR (Artificial Force Induced Reaction)** [Maeda, Morokuma, *J. Chem. Theory Comput.* **6**, 2734 (2010)] に置き換え、力場の科学的正当性とパラメータ簡素化を一気に達成する。

具体的には:

- 各反応につき 4 系統 (`k_form`, `k_broken`, `r_broken`, `r_form`) あった力場ハイパラを、**`alpha_formed` と `alpha_broken` の 2 個** (＋ p exponent default 6) に縮約する。
- per-bond list (`k_form = [3.0, 3.0]`) で擬似的に対応していた「複数結合の同時形成」を、AFIR が本質的に持つ **collective coordinate (weighted average distance)** で物理的に正当な扱いに格上げする。
- `r_broken` cutoff (PullApart で必要だった「どこまで repulsive 力をかけるか」) は AFIR の `ω ∝ r⁻⁶` で自動減衰するため **完全に消滅**。

phase-8 ブランチで挫折した「CI-NEB 汎用化 + screening + 並列化」の三点同時変更は本フェーズには含めない。force model だけ置換し、Stage 構造・並列化・CI-NEB 拡張は後続フェーズに切り出す。

## 2. Non-goals

明示的に **このフェーズではやらない** こと:

- **CI-NEB の汎用化**: `--neb-refine` は今のとおり 1 formed + 1 broken 反応専用のオプション (`align_product_to_reactant` 経由)。E2 / SN1 dissoc / SN1 recomb / DA で NEB 必要なら future phase。
- **Screening / top-K / 並列化**: Phase 8 (旧) の Stage1→top-K→Stage2 構造は導入しない。全 placement-survivor を sequential に AFIR-relax する従来構造を維持。
- **MC-AFIR (reaction discovery mode)**: ユーザーが `formed`/`broken` を **書かない** で AFIR が自動的に bond change を発見する、いわゆる Multi-Component AFIR は対象外。本フェーズはあくまで「formed/broken は手動指定」前提の force-replacement。
- **新反応の追加**: 8 反応 (SN2 / PT / Menshutkin / E2 / SN1 dissoc / SN1 recomb / DA simple / DA endo) を全て AFIR で動かすことがゴール。新反応は別フェーズ。
- **biased PES の peak energy 報告**: AFIR-biased energy ではなく **unbiased UMA energy** を `meta.json` / `energies.json` に記録し続ける (既存仕様維持)。
- **p exponent の TOML 露出**: `p = 6` は Maeda 2010 default 固定 (内部定数)。露出する必要が出たら future phase。

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
            ├── trial 2 ──┤  build_afir_constraint(formed, broken, α_f, α_b)
            ├── ...        │  → relax_with_restraints (FIRE on biased grad,
            └── trial K ──┘                              unbiased energy snapshots)
                                                       │
                                                       ▼
                                   scoring.reached_product           ← ★ 共有半径ベースで自動閾値
                                   scoring.score_trials              (Phase 7 と同じ)
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
| 力場 (formed) | `Hookean(k, rt)` per-bond | `AFIRConstraint` の `ρ_formed` collective × scalar `α_formed` |
| 力場 (broken) | `PullApart(k, rt)` per-bond | `AFIRConstraint` の `ρ_broken` collective × scalar `α_broken` (符号は内部固定) |
| 重み関数 | なし (Hookean は単一 pair の harmonic) | `ω_ij = ((R_i+R_j)/r_ij)^p`、p=6 (Maeda 2010) |
| 半径基底 | `DEFAULT_R_FORM` 元素表 + `r_broken` 手動 | Cordero (2008) **共有半径** (反応点だけでなく全 atom 用) |
| TOML keys (force) | `k_form` `k_broken` `r_broken` `r_form` `max_relax_steps` | `alpha_formed` `alpha_broken` `max_relax_steps` (任意 `p`) |
| per-bond override | `k_form = [3.0, 3.0]` 形式 list 対応 | **廃止** (collective coordinate に縮約。ω が自動で per-bond 重みを算出) |
| `reached_product` 閾値 | `r_broken` (TOML) と `r_form` (TOML / 元素表) | 共有半径 × tolerance (`broken=2.5`, `formed=1.3`) で自動、TOML 上書き任意 |
| `peak_energy` の中身 | unbiased UMA energy | **変更なし** (ASE FixConstraint パターン、energy には影響しない) |

### 3.3 ファイル構成

書き換え:
- `reactx/artificial_force.py` — `Hookean`/`PullApart`/`build_restraints`/`DEFAULT_R_FORM`/`lookup_r_form` を全削除し、`AFIRConstraint` クラスと `build_afir_constraint` 関数のみ残す
- `reactx/config.py` — `[restraints]` セクション削除、`[afir]` 追加。`resolve_*_targets` ヘルパも AFIR 用に簡素化
- `reactx/scoring.py` — `reached_product` を共有半径ベースの自動閾値に
- `reactx/cli.py` — `build_restraints` 呼び出しを `build_afir_constraint` に置換、resolved param の流れを刷新
- `examples/*.rxn.toml` × 8 — 新 schema に書き換え
- `tests/test_*.py` — Hookean/PullApart 系を削除、AFIR 系を追加、slow integration の期待値を再 tune
- `README.md` — Phase 9 セクション、新 schema 表、wall-clock 再測定

新規:
- `reactx/covalent_radii.py` — Cordero (2008) 共有半径 lookup (現在 `blender/render.py` 内に埋め込まれている表を切り出す)
- `tests/test_afir_constraint.py` — 解析勾配 vs 有限差分の一致、ρ 計算、空 set の no-op、力の方向

削除:
- `reactx/align.py` (Phase 8 旧で削除予定だった残骸の最終撤去 — 1+1 NEB が `align_product_to_reactant` を使い続けるなら **保留**、未使用なら削除)。`grep -r align_product_to_reactant` で確認後決定。
- `tests/test_pull_apart.py`、`tests/test_hookean_restraints.py`、その他既存力場専用 unit tests

## 4. コンポーネント詳細

### 4.1 `reactx/artificial_force.py` — `AFIRConstraint`

```python
"""Maeda-Morokuma AFIR force on (formed, broken) atom-pair sets.

Replaces Phase Re1's Hookean+PullApart hybrid with a single ASE FixConstraint
that drives the system along two collective coordinates:

    ρ_formed(Q) = Σ_{(i,j)∈formed} ω_ij r_ij  /  Σ_{(i,j)∈formed} ω_ij
    ρ_broken(Q) = Σ_{(i,j)∈broken} ω_ij r_ij  /  Σ_{(i,j)∈broken} ω_ij

with ω_ij = ((R_i + R_j) / r_ij)^p,  p=6,  R_i,R_j Cordero (2008) covalent
radii in Å.

Force on atom k:

    F_k = -α_formed · ∂ρ_formed/∂x_k  +  (+α_broken) · ∂ρ_broken/∂x_k

i.e. positive α_formed compresses the formed pairs, positive α_broken expands
the broken pairs (sign is fixed internally; users supply only magnitudes).

Reference: Maeda & Morokuma, J. Chem. Theory Comput. 6, 2734 (2010).
"""

from __future__ import annotations

import numpy as np
from ase.atoms import Atoms
from ase.constraints import FixConstraint

P_DEFAULT = 6


class AFIRConstraint(FixConstraint):
    def __init__(
        self,
        formed: list[tuple[int, int]],
        broken: list[tuple[int, int]],
        cov_radii: np.ndarray,         # shape=(n_atoms,), Å
        *,
        alpha_formed: float,           # eV/Å, magnitude (>= 0)
        alpha_broken: float,           # eV/Å, magnitude (>= 0)
        p: int = P_DEFAULT,
    ):
        if alpha_formed < 0 or alpha_broken < 0:
            raise ValueError("alpha_formed and alpha_broken must be non-negative")
        if p <= 0:
            raise ValueError("AFIR exponent p must be positive")
        self.formed = [(int(a), int(b)) for a, b in formed]
        self.broken = [(int(a), int(b)) for a, b in broken]
        self.cov = np.asarray(cov_radii, dtype=float).copy()
        self.alpha_formed = float(alpha_formed)
        self.alpha_broken = float(alpha_broken)
        self.p = int(p)

    def adjust_positions(self, atoms, newpositions):
        return  # force-only constraint; positions integrated by optimizer

    def adjust_forces(self, atoms, forces):
        if self.formed and self.alpha_formed > 0:
            self._apply(atoms.positions, forces, self.formed, sign=-1.0,
                        alpha=self.alpha_formed)
        if self.broken and self.alpha_broken > 0:
            self._apply(atoms.positions, forces, self.broken, sign=+1.0,
                        alpha=self.alpha_broken)

    def _apply(self, pos, forces, pairs, *, sign, alpha):
        """
        Add the AFIR force contribution from one pair set to `forces`.

        Energy convention:
            E_AFIR = sign_E · α · ρ      with sign_E = +1 (formed) or -1 (broken)
            F     = -∂E_AFIR/∂x = -sign_E · α · ∂ρ/∂x

        Equivalently, with `sign = -sign_E`  (i.e. -1 for formed, +1 for broken):
            F = sign · α · ∂ρ/∂x

        Derivation of ∂ρ/∂r:
            ρ = S/W, where S = Σ_l ω_l r_l, W = Σ_l ω_l, ω_l = ((R_i+R_j)/r_l)^p
            ∂ω_l/∂r_l = -p · ω_l / r_l
            ∂S/∂r_k   = ω_k (1 - p)         [only the k-th term depends on r_k]
            ∂W/∂r_k   = -p · ω_k / r_k
            ∂ρ/∂r_k   = (1/W) [∂S/∂r_k - ρ · ∂W/∂r_k]
                      = (ω_k / W) · [(1 - p) + p · ρ / r_k]
            ∂r_k/∂x_j =  d̂_k,    ∂r_k/∂x_i = -d̂_k,    d̂_k = (x_j - x_i)/r_k

        Single-pair sanity check (W=ω_0, S=ω_0·r_0, ρ=r_0):
            coef_0 = (ω_0/W) [(1-p) + p · r_0/r_0] = (1-p) + p = 1
            grad_0 = +d̂_0 (on atom j)
            For formed (sign=-1): F_j = -α · d̂_0  → j pushed toward i (compress) ✓
            For broken (sign=+1): F_j = +α · d̂_0  → j pushed away from i (expand) ✓
        """
        n = len(pairs)
        omega = np.zeros(n, dtype=float)
        r = np.zeros(n, dtype=float)
        d_hat = np.zeros((n, 3), dtype=float)
        for k, (i, j) in enumerate(pairs):
            v = pos[j] - pos[i]
            rij = float(np.linalg.norm(v))
            if rij < 1e-10:
                # degenerate: atoms coincident, skip; UMA repulsion will move
                # them apart on the next step.
                continue
            Rsum = self.cov[i] + self.cov[j]
            omega[k] = (Rsum / rij) ** self.p
            r[k] = rij
            d_hat[k] = v / rij
        W = omega.sum()
        if W < 1e-30:
            return
        S = float((omega * r).sum())
        rho = S / W
        safe_r = np.where(r > 0, r, 1.0)
        coef = omega / W * ((1 - self.p) + self.p * rho / safe_r)
        for k, (i, j) in enumerate(pairs):
            if r[k] == 0.0:
                continue
            grad_j = coef[k] * d_hat[k]            # ∂ρ/∂x_j
            force_on_j = sign * alpha * grad_j     # F = sign·α·∂ρ/∂x
            forces[j] += force_on_j
            forces[i] -= force_on_j                # Newton's third law

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
                "p": self.p,
            },
        }


def build_afir_constraint(
    atoms: Atoms,
    formed: list[tuple[int, int]],
    broken: list[tuple[int, int]],
    *,
    alpha_formed: float,
    alpha_broken: float,
    p: int = P_DEFAULT,
) -> list[AFIRConstraint]:
    """Return a 1- or 0-element list (empty when both pair sets are empty)."""
    if not formed and not broken:
        return []
    from reactx.covalent_radii import cordero_radii_for_atoms
    cov = cordero_radii_for_atoms(atoms)
    return [AFIRConstraint(
        formed, broken, cov,
        alpha_formed=alpha_formed,
        alpha_broken=alpha_broken,
        p=p,
    )]
```

**API 互換性**: `relax_with_restraints(atoms, restraints, calc, ...)` の `restraints` 引数は `list[Constraint]` 型のまま。`atoms.set_constraint(restraints)` が ASE 規約どおり動く (length 1 の AFIR list でも length 0 の空 list でも問題なし)。

### 4.2 `reactx/covalent_radii.py` — Cordero 表の切り出し

```python
"""Cordero (2008) covalent radii in Å, indexed by element symbol.

Reference: Cordero et al., Dalton Trans. 2008, 2832.
Used by:
- `AFIRConstraint` for ω_ij weights and reached_product threshold
- `blender/render.py` for bond drawing distance threshold
"""

from __future__ import annotations
import numpy as np
from ase.atoms import Atoms

CORDERO_2008: dict[str, float] = {
    "H": 0.31, "He": 0.28,
    "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66,
    "F": 0.57, "Ne": 0.58,
    # ... up to Bi (Z=83) per OMol25 / UMA training range
    # full table copied from blender/render.py existing source
}


def cordero_radius(symbol: str, *, default: float = 1.5) -> float:
    return CORDERO_2008.get(symbol, default)


def cordero_radii_for_atoms(atoms: Atoms) -> np.ndarray:
    return np.array([cordero_radius(s) for s in atoms.get_chemical_symbols()])
```

`blender/render.py` 側は import に切り替え (重複辞書を排除)。

### 4.3 `reactx/config.py` — TOML schema 変更

新 schema (`examples/sn2.rxn.toml` を例に):

```toml
description = "SN2 anion: CH3Cl + OH- -> CH3OH + Cl-"
formed = [[1, 3]]
broken = [[1, 2]]

[afir]
alpha_formed = 0.7      # eV/Å, scalar; positive → 縮める方向
alpha_broken = 0.5      # eV/Å, scalar; magnitude only (符号は内部固定 = 引き離し)
max_relax_steps = 100
# p = 6                 # 任意 (default=6)、通常省略

# [scoring] r_broken_threshold = 4.0   # 任意上書き (default = 共有半径×1.5)
```

`reactx/config.py` の dataclass:

```python
@dataclass(frozen=True)
class AFIRSection:
    alpha_formed: float
    alpha_broken: float
    max_relax_steps: int
    p: int = 6

@dataclass(frozen=True)
class ScoringSection:
    r_broken_threshold: float | None = None  # None = covalent-radius default

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
- `alpha_formed >= 0`、`alpha_broken >= 0`、`max_relax_steps > 0`、`p > 0`
- `formed=[]` のとき `alpha_formed` の値は何でも良い (使われない、warning も出さない)
- `broken=[]` のとき `alpha_broken` の値は何でも良い
- 既存 `[restraints]` セクションが TOML に出てきたら **`ConfigError`** で reject (旧 schema が誤って残っている検出)
- 旧 keys (`k_form`, `k_broken`, `r_broken`, `r_form`) の top-level / restraints 配下出現も `ConfigError`

### 4.4 `reactx/scoring.py` — `reached_product` の共有半径化

```python
COVALENT_FORMED_TOLERANCE = 1.3   # ≤ 1.3 × (R_i+R_j) → bond formed (covalent + 30% slack)
COVALENT_BROKEN_TOLERANCE = 2.5   # ≥ 2.5 × (R_i+R_j) → bond broken (well past vdW contact)


def reached_product(
    atoms: Atoms,
    formed: list[tuple[int, int]],
    broken: list[tuple[int, int]],
    *,
    override_r_broken: float | None = None,  # from [scoring].r_broken_threshold
) -> bool:
    cov = cordero_radii_for_atoms(atoms)
    pos = atoms.positions
    for (i, j) in formed:
        d = float(np.linalg.norm(pos[j] - pos[i]))
        if d > (cov[i] + cov[j]) * COVALENT_FORMED_TOLERANCE:
            return False
    for (i, j) in broken:
        d = float(np.linalg.norm(pos[j] - pos[i]))
        threshold = override_r_broken
        if threshold is None:
            threshold = (cov[i] + cov[j]) * COVALENT_BROKEN_TOLERANCE
        if d < threshold:
            return False
    return True
```

`score_trials` の他の部分 (peak_energy 最小化、least-bad fallback) は変更なし。

### 4.5 `reactx/cli.py` — orchestration の変更点

- `resolve_k_form_targets` / `resolve_k_broken_targets` / `resolve_r_broken_targets` / `resolve_r_form_targets` を **削除** (Phase 8 で導入された 4 系統 normalizer の DRY 統合は AFIR が collective coordinate であるため不要に)
- 各 trial の relax 直前に `build_afir_constraint(atoms, formed, broken, alpha_formed=cfg.afir.alpha_formed, alpha_broken=cfg.afir.alpha_broken, p=cfg.afir.p)` を呼ぶ
- `meta.json` の per-trial エントリから `r_broken_target` / `r_form_target` / `k_form` / `k_broken` を **削除**、代わりに `alpha_formed` / `alpha_broken` を記録
- NEB refine path はそのまま (`align_product_to_reactant` が必要なら残す、不要なら削除)

## 5. 8 example の AFIR 初期 tuning 値

| Reaction | α_formed | α_broken | max_relax_steps | 備考 |
|---|---|---|---|---|
| sn2 | 0.7 | 0.5 | 100 | 旧 k_form=0.5, k_broken=1.0 から AFIR の collective scale に概算換算 |
| proton_transfer | 1.5 | 1.0 | 100 | 旧 k_form=1.0, k_broken=2.0 |
| menshutkin | 4.0 | 1.5 | 200 | gas-phase repulsive PES を押し切る必要、旧 k_form=4.0 を踏襲 |
| e2 | 1.5 | 1.0 | 200 | 2 broken は ω 重み平均で自動分配。旧 k_form=1.0, k_broken=1.0 |
| sn1_dissoc | 0.0 | 1.5 | 200 | formed=[] なので α_formed=0 (無視される) |
| sn1_recomb | 1.5 | 0.0 | 200 | broken=[] なので α_broken=0 |
| diels_alder_simple | 2.5 | 0.0 | 200 | DA 2-bond collective 1 個、旧 k_form=[3.0, 3.0] を AFIR に圧縮 |
| diels_alder_endo | 2.5 | 0.0 | 250 | 同上 |

これらは「初期 guess」。実装後 `pytest -m slow` で各反応の `reached_product` を見ながら 1〜2 iteration で再調整する。

**`r_broken_threshold` 自動値の検証**: tolerance=2.5 で各反応の閾値は

| broken pair | 共有半径合計 | × 2.5 = auto threshold | 旧 TOML `r_broken` |
|---|---|---|---|
| sn2 C-Cl | 1.78 | 4.45 Å | 4.0 |
| proton_transfer N-H (実は H-Cl) | H 0.31 + Cl 1.02 = 1.33 | 3.33 Å | 4.0 |
| menshutkin C-Cl | 1.78 | 4.45 Å | 5.0 |
| e2 C-H (broken[0]) | 0.76+0.31=1.07 | 2.68 Å | 4.0 |
| e2 C-Cl (broken[1]) | 1.78 | 4.45 Å | 4.0 |
| sn1_dissoc C-Br | 0.76+1.20=1.96 | 4.90 Å | 6.0 |

E2 の C-H が 2.68 Å と若干厳しい (旧 4.0 より低い閾値で reached_product=True 判定が容易になる)。実装後 `pytest -m slow` で false positive (反応未達なのに判定 True) が起きていないか trajectory を visual で確認、必要なら `[scoring] r_broken_threshold = 3.5` を E2 にだけ明示追加する。SN1 dissoc は 4.90 vs 6.0 でほぼ同等。他は旧値とほぼ整合する。

## 6. テスト戦略

### 6.1 新規 unit tests (`tests/test_afir_constraint.py`)

- **解析勾配 vs 有限差分**: 任意の n-atom 配置 + ランダム formed/broken pair set で、`AFIRConstraint.adjust_forces` が返す力 = `ρ` の有限差分から計算した数値勾配 × (-α) と相対誤差 1e-5 以下で一致
- **単一 pair 縮約**: `formed=[(0,1)]` のとき `ρ_formed = r_01`、力は `-α_formed × d̂_01` (∇r が単位ベクトルなので) と一致
- **空 set no-op**: `formed=[]` または `broken=[]` で当該 collective が自動 skip、`forces` 配列が変更されない
- **両方空**: `build_afir_constraint(formed=[], broken=[])` が空 list を返す
- **対称性**: i,j を swap しても同じ力 (ペア順は無関係)
- **ω 重み**: 異種長距離 pair (`r=4.0 Å` と `r=2.0 Å` の同時 formed) で、ρ 計算時に r=2.0 の方が ω で支配的になることを assert
- **負の α 拒否**: `alpha_formed=-0.1` で `ValueError`

### 6.2 新規 unit tests (`tests/test_config_afir.py`)

- 新 schema parse 成功 (sn2 / proton_transfer / dummy minimal)
- 旧 schema (`[restraints]`, `k_form`, `k_broken`, `r_broken`, `r_form`) を含む TOML が `ConfigError` で reject
- `p` 省略時 default=6
- `alpha_formed` 省略時は `ConfigError` (任意化しない、明示要求)
- `[scoring] r_broken_threshold` 省略時 None
- `max_relax_steps <= 0` で `ConfigError`

### 6.3 既存 slow integration tests (`pytest -m slow`)

8 反応すべての統合テストを retune:
- `selected_trial >= 0` であり、かつ `reached_product=True` が trials 中 1 件以上
- `peak_energy` が反応によって典型的な範囲 (例: SN2 ~3-5 eV、Menshutkin ~5-8 eV、DA ~2-4 eV) — 値は実測で更新
- `meta.json.placement_kind` が既存値と一致 (placement レイヤは触っていないので)

### 6.4 削除する tests

- `tests/test_pull_apart.py` (PullApart クラス消滅)
- `tests/test_hookean_*.py` (Hookean 直接使用箇所消滅)
- `tests/test_resolve_*targets.py` (resolve_*_targets ヘルパ消滅) — 該当ファイルが存在する場合のみ
- 既存 `tests/test_artificial_force.py` の旧 schema 部分 → 全削除し AFIR 用に書き直し

### 6.5 Blender smoke

`tests/test_blender_render.py` の SN2 / proton_transfer parametrized は既存通り pass を期待 (アニメーションの中身は変わらないので)。

## 7. 実装順序

1. **共有半径切り出し**: `reactx/covalent_radii.py` 新規作成、`blender/render.py` から表を移管 + import 化。unit test (lookup と Atoms 配列化)。
2. **AFIRConstraint**: `tests/test_afir_constraint.py` を **先に** TDD で書き、`reactx/artificial_force.py` を全面書き換えて pass させる。
3. **config 刷新**: `[afir]` schema、validation。`tests/test_config_afir.py` で TDD。
4. **scoring 刷新**: `reached_product` を共有半径化。unit test を更新。
5. **cli 配線**: `build_restraints` → `build_afir_constraint` 置換、resolve_*_targets 削除、meta.json schema 更新。
6. **example 8 個一括書き換え** (`alpha_formed/alpha_broken/max_relax_steps` のみ、初期値は §5 の表)。
7. **`pytest`** (unit + non-slow) green を確認。
8. **`pytest -m slow`** で 1 反応ずつ走らせて α tuning。順序は短い方から長い方へ: sn1_dissoc → sn2 → proton_transfer → sn1_recomb → menshutkin → e2 → da_simple → da_endo。各反応で `reached_product` 1 件以上、selected_trial の trajectory が物理的に妥当。
9. **align.py 残骸チェック**: `grep -r align_product_to_reactant`、未使用なら削除、1+1 NEB が依存しているなら保留。
10. **README 全面書き換え**: Phase 9 セクション、`[afir]` schema 表、wall-clock 再測定 (RTX 5070 Ti + UMA 環境で 8 反応)。
11. **Phase 9 PR**: develop ベース、commit history は task-by-task で TDD 由来の順序。

## 8. リスクと緩和

- **AFIR force が UMA の ML potential と組み合わさったときの挙動が未検証**: Maeda の論文は DFT/HF が前提。UMA の OMol25 学習領域では artificial force による non-physical な構造に陥る可能性あり。**緩和**: 全 8 反応で slow test が pass するまで α を tune、駄目なら fallback として p exponent (現在 hardcoded 6) を 4 や 8 に振る選択肢を後から TOML 露出する option を残す (本フェーズでは露出しない)。
- **Diels-Alder の 2-bond collective が片方だけ形成される asymmetric concerted の表現力が落ちる**: per-bond k で Phase 8 が出していた `k_form = [3.0, 3.0]` 形式は AFIR の ω 自動重みに任せるが、CP+MA の electronic 非対称性は ω では捉えられない。**緩和**: 実装後 `diels_alder_endo` の trajectory を visual で確認、片側 bond だけ早く形成して selected_trial の reached_product が False になるなら、α_formed を増やす or future phase で per-collective AFIR を 2 個に分ける拡張を検討。
- **`r_broken_threshold` の自動値 (covalent×2.5) が反応によっては緩すぎる/厳しすぎる**: 大半の反応では旧 `r_broken=4.0-6.0 Å` と整合する (§5 表) が、E2 の C-H = 2.68 Å が旧 4.0 より緩く、false positive (relax 中盤の中間状態を product と誤判定) を起こす可能性がある。**緩和**: slow test 時に E2 trajectory を visual + `meta.json` `selected_trial` の最終 frame の bond distance で確認、必要なら `[scoring] r_broken_threshold = 3.5` を E2 example のみ明示追加。tolerance=2.5 自体が経験定数なので、実測で全 8 反応に効く値が見つかれば調整する余地がある (例: 2.0 でより厳しく)。
- **AFIR の `ω ∝ r⁻⁶` は近距離で発散**: 解析勾配の `coef = ω/W * ((1-p) + p ρ/r)` は r→0 で発散するが、`r < 1e-10` ガード (実装の `_apply` 冒頭) で skip。実用上は UMA の repulsion が r=0.3-0.5 Å で十分強いので発散領域には到達しない想定。**緩和**: unit test で `r=1e-12` 配置を投げ、NaN や inf が forces に入らないことを assert。

## 9. 今後のロードマップ (Phase 10+ の参考)

本フェーズで AFIR に統一されたあと、phase-8 (旧) が目指していた残課題は順次別フェーズで:
- **Phase 10**: CI-NEB を E2 / SN1 / DA に拡張 (1+1 制約解除)、`align_product_to_reactant` を AFIR 軌跡 endpoint 由来に置換
- **Phase 11**: Stage1 (AFIR screening) → top-K → Stage2 (CI-NEB) の 2 段階パイプライン、process pool 並列化
- **Phase 12**: MC-AFIR モード (`formed`/`broken` を書かなくても reaction discovery)、cycloaddition `bridges >= 3` (1,3-dipolar など)、cheletropic
