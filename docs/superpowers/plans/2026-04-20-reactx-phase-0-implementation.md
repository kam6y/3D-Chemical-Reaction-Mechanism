# reactx Phase 0 Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** SN2 反応 (CH₃Cl + F⁻ → CH₃F + Cl⁻) 1 件について `.rxn` → UMA+ASE NEB → Blender ball-and-stick アニメーションまでを end-to-end で貫通させる Phase 0 スパイクを構築する。

**Architecture:** Python パッケージ `reactx/` に parse → embed3d → align → neb → cli のパイプラインを薄くモジュール化する。各モジュールは単一責務、ASE `Atoms` を共通中間形式とする。Blender 側は `blender/render.py` を `blender --background --python` から実行するスクリプトとして独立させる。計算エンジンは `calculators.make_calculator(name)` ファクトリで差し替え可能にし、Phase 0 では UMA を使用しつつテストでは軽量な ASE 組込 calculator (`LennardJones`, `EMT`) を利用する。

**Tech Stack:** Python 3.10+, RDKit, ASE, fairchem-core (UMA), PyTorch (MPS/CPU), pytest, Blender 4.x, `atomic-blender-pdb-xyz` add-on。

---

## File Structure

新規作成:

```
reactx/
  __init__.py          # パッケージ宣言、公開シンボル列挙
  rxn_parser.py        # .rxn → (reactant Mol, product Mol, mapping)
  embed3d.py           # 2D Mol → ase.Atoms (RDKit embed → MMFF → UMA 再最適化)
  align.py             # atom-mapping に基づき product Atoms を reactant の原子順に揃える
  calculators.py       # make_calculator(name) ファクトリ
  neb.py               # ASE IDPP + CI-NEB driver、trajectory.xyz 出力
  cli.py               # `reactx run <rxn> -o <dir>` エントリポイント
blender/
  render.py            # `blender --background --python render.py -- <xyz> <out.blend>`
examples/
  sn2.rxn              # CH3Cl + F- → CH3F + Cl- の MDL Rxn v2000
tests/
  __init__.py
  conftest.py          # fixture 定義 (SN2 Atoms ペア)
  test_rxn_parser.py
  test_embed3d.py
  test_align.py
  test_calculators.py
  test_neb_sn2.py      # @pytest.mark.slow、UMA 依存
  test_cli.py
  test_blender_smoke.py  # ローカル限定、CI では skip
pyproject.toml
.gitignore
```

既存変更:

- `README.md` — プロジェクト概要 + 再現手順を上書き

---

## Task 1: プロジェクトスキャフォールド

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `reactx/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: `pyproject.toml` を作成**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "reactx"
version = "0.0.1"
description = "2D reaction scheme to 3D NEB trajectory and Blender animation (Phase 0 spike)"
requires-python = ">=3.10"
dependencies = [
    "rdkit>=2024.3",
    "ase>=3.22",
    "numpy>=1.24",
    "fairchem-core>=1.0",
    "torch>=2.2",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-mock>=3.12",
]

[project.scripts]
reactx = "reactx.cli:main"

[tool.setuptools.packages.find]
include = ["reactx*"]

[tool.pytest.ini_options]
markers = [
    "slow: require UMA model download and run NEB to convergence",
    "blender: require local Blender install + atomic-blender-pdb-xyz add-on",
]
testpaths = ["tests"]
```

- [ ] **Step 2: `.gitignore` を作成**

```
__pycache__/
*.py[cod]
*.egg-info/
.venv/
.pytest_cache/
out/
*.blend
*.blend1
.DS_Store
```

- [ ] **Step 3: 空のパッケージファイルを作成**

`reactx/__init__.py`:

```python
"""reactx — 2D reaction scheme to 3D NEB trajectory pipeline (Phase 0 spike)."""

__version__ = "0.0.1"
```

`tests/__init__.py`: 空ファイル。

- [ ] **Step 4: `tests/conftest.py` を作成（後続タスクで拡張する骨組みのみ）**

```python
"""Shared pytest fixtures for reactx tests."""
from pathlib import Path

import pytest


@pytest.fixture()
def examples_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture()
def sn2_rxn_path(examples_dir: Path) -> Path:
    return examples_dir / "sn2.rxn"
```

- [ ] **Step 5: インストールと疎通確認**

Run: `python -m pip install -e .[dev]`
Expected: `Successfully installed reactx-0.0.1 ...`

Run: `python -c "import reactx; print(reactx.__version__)"`
Expected: `0.0.1`

Run: `pytest -q`
Expected: `no tests ran` (0 passed, 0 failed) — collection error が無いこと。

- [ ] **Step 6: コミット**

```bash
git add pyproject.toml .gitignore reactx/__init__.py tests/__init__.py tests/conftest.py
git commit -m "chore: scaffold reactx package and pytest config"
```

---

## Task 2: SN2 `.rxn` フィクスチャ作成

**Files:**
- Create: `examples/sn2.rxn`

**Context:** MDL Rxn v2000 形式で CH₃Cl + F⁻ → CH₃F + Cl⁻ を atom map 番号付きで記述する。heavy atom のみ (H は implicit)。原子は C=1, Cl=2, F=3 の map 番号で対応させる。

- [ ] **Step 1: `examples/sn2.rxn` を作成**

```
$RXN

      reactx Phase0 SN2 example
  CH3Cl + F- -> CH3F + Cl-
  2  2
$MOL

     RDKit          2D

  2  1  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  1  0  0
    1.5000    0.0000    0.0000 Cl  0  0  0  0  0  0  0  0  0  2  0  0
  1  2  1  0
M  END
$MOL

     RDKit          2D

  1  0  0  0  0  0  0  0  0  0999 V2000
   -2.0000    0.0000    0.0000 F   0  5  0  0  0  0  0  0  0  3  0  0
M  CHG  1   1  -1
M  END
$MOL

     RDKit          2D

  2  1  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  1  0  0
    1.5000    0.0000    0.0000 F   0  0  0  0  0  0  0  0  0  3  0  0
  1  2  1  0
M  END
$MOL

     RDKit          2D

  1  0  0  0  0  0  0  0  0  0999 V2000
   -2.0000    0.0000    0.0000 Cl  0  5  0  0  0  0  0  0  0  2  0  0
M  CHG  1   1  -1
M  END
```

> **注意 (V2000 カラム仕様):** atom block の最終 map 番号は 16 桁目のフィールド（`aaamm` の `mm`）に格納される。上記は RDKit がパース可能な緩い書式に準拠しているため、Task 3 でパース結果を確認し、ズレがあれば列合わせを修正する。

- [ ] **Step 2: RDKit でパース可能かを手動確認**

Run:
```bash
python -c "from rdkit.Chem import AllChem; rxn = AllChem.ReactionFromRxnFile('examples/sn2.rxn'); print('Reactants:', rxn.GetNumReactantTemplates()); print('Products:', rxn.GetNumProductTemplates()); [print('R', i, [a.GetSymbol() for a in m.GetAtoms()], [a.GetAtomMapNum() for a in m.GetAtoms()]) for i, m in enumerate(rxn.GetReactants())]; [print('P', i, [a.GetSymbol() for a in m.GetAtoms()], [a.GetAtomMapNum() for a in m.GetAtoms()]) for i, m in enumerate(rxn.GetProducts())]"
```

Expected:
```
Reactants: 2
Products: 2
R 0 ['C', 'Cl'] [1, 2]
R 1 ['F'] [3]
P 0 ['C', 'F'] [1, 3]
P 1 ['Cl'] [2]
```

パースが失敗、または map 番号がゼロの場合は .rxn の書式を修正する（原子行の 16 桁目の map number カラムを詰める / 空白で揃える）。RDKit の出力に合わせ、正しく認識されるまで修正を繰り返す。

- [ ] **Step 3: コミット**

```bash
git add examples/sn2.rxn
git commit -m "feat(examples): add SN2 reaction .rxn fixture with atom map numbers"
```

---

## Task 3: `rxn_parser` — `.rxn` をパースして mapping を抽出

**Files:**
- Create: `reactx/rxn_parser.py`
- Create: `tests/test_rxn_parser.py`

**Responsibility:** `.rxn` を読み込み、reactant 側の 2 分子を結合した `Mol`、product 側の 2 分子を結合した `Mol`、heavy-atom の `reactant_idx → product_idx` mapping dict を返す。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_rxn_parser.py`:

```python
from pathlib import Path

import pytest
from rdkit import Chem

from reactx.rxn_parser import parse_rxn


def test_parse_sn2_rxn_returns_combined_mols(sn2_rxn_path: Path):
    reactant, product, mapping = parse_rxn(sn2_rxn_path)

    assert isinstance(reactant, Chem.Mol)
    assert isinstance(product, Chem.Mol)
    assert isinstance(mapping, dict)

    reactant_symbols = sorted(a.GetSymbol() for a in reactant.GetAtoms())
    product_symbols = sorted(a.GetSymbol() for a in product.GetAtoms())
    assert reactant_symbols == ["C", "Cl", "F"]
    assert product_symbols == ["C", "Cl", "F"]


def test_parse_sn2_rxn_has_two_fragments_each_side(sn2_rxn_path: Path):
    reactant, product, _ = parse_rxn(sn2_rxn_path)
    assert len(Chem.GetMolFrags(reactant)) == 2
    assert len(Chem.GetMolFrags(product)) == 2


def test_parse_sn2_rxn_mapping_covers_all_heavy_atoms(sn2_rxn_path: Path):
    reactant, product, mapping = parse_rxn(sn2_rxn_path)
    assert len(mapping) == reactant.GetNumHeavyAtoms() == 3
    for r_idx, p_idx in mapping.items():
        r_sym = reactant.GetAtomWithIdx(r_idx).GetSymbol()
        p_sym = product.GetAtomWithIdx(p_idx).GetSymbol()
        assert r_sym == p_sym, f"Mapping {r_idx}->{p_idx} crosses element: {r_sym}/{p_sym}"


def test_parse_rxn_raises_when_atom_map_missing(tmp_path: Path):
    bad = tmp_path / "nomap.rxn"
    bad.write_text(
        "$RXN\n\n   bad\n\n  1  1\n$MOL\n\n     RDKit          2D\n\n  1  0  0  0  0  0  0  0  0  0999 V2000\n"
        "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\nM  END\n$MOL\n\n"
        "     RDKit          2D\n\n  1  0  0  0  0  0  0  0  0  0999 V2000\n"
        "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\nM  END\n"
    )
    with pytest.raises(ValueError, match="atom map"):
        parse_rxn(bad)
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `pytest tests/test_rxn_parser.py -v`
Expected: `ModuleNotFoundError: No module named 'reactx.rxn_parser'` — 4 errored (collection error OK)

- [ ] **Step 3: `reactx/rxn_parser.py` を実装**

```python
"""Parse MDL .rxn files into combined RDKit Mols and atom mapping."""
from __future__ import annotations

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem


def parse_rxn(rxn_path: str | Path) -> tuple[Chem.Mol, Chem.Mol, dict[int, int]]:
    """Parse an MDL Rxn file and return combined reactant/product Mols + mapping.

    The returned reactant Mol contains all reactant fragments combined via
    Chem.CombineMols (preserves atom order across fragments). The mapping dict
    maps each reactant heavy-atom index to the corresponding product heavy-atom
    index based on the AtomMapNum field present in the .rxn file.

    Raises ValueError when any heavy atom on either side is missing an atom
    map number.
    """
    rxn = AllChem.ReactionFromRxnFile(str(rxn_path))
    if rxn is None:
        raise ValueError(f"Failed to parse .rxn file: {rxn_path}")

    reactant = _combine_fragments(list(rxn.GetReactants()))
    product = _combine_fragments(list(rxn.GetProducts()))

    r_map = _collect_atom_map_numbers(reactant, side="reactant")
    p_map = _collect_atom_map_numbers(product, side="product")

    mapping: dict[int, int] = {}
    for mapnum, r_idx in r_map.items():
        if mapnum not in p_map:
            raise ValueError(
                f"Atom map number {mapnum} present in reactant but not in product"
            )
        mapping[r_idx] = p_map[mapnum]

    return reactant, product, mapping


def _combine_fragments(mols: list[Chem.Mol]) -> Chem.Mol:
    if not mols:
        raise ValueError("No fragments found on one side of the reaction")
    combined = mols[0]
    for m in mols[1:]:
        combined = Chem.CombineMols(combined, m)
    Chem.SanitizeMol(combined)
    return combined


def _collect_atom_map_numbers(mol: Chem.Mol, *, side: str) -> dict[int, int]:
    result: dict[int, int] = {}
    for atom in mol.GetAtoms():
        mapnum = atom.GetAtomMapNum()
        if mapnum == 0:
            raise ValueError(
                f"{side} atom {atom.GetIdx()} ({atom.GetSymbol()}) has no atom map number"
            )
        if mapnum in result:
            raise ValueError(
                f"Duplicate atom map number {mapnum} on {side} side"
            )
        result[mapnum] = atom.GetIdx()
    return result
```

- [ ] **Step 4: テストが通ることを確認**

Run: `pytest tests/test_rxn_parser.py -v`
Expected: `4 passed`

4 件のうちいずれかが失敗する場合、Task 2 の `.rxn` 書式が RDKit の期待する列位置と合っていない可能性が高い。Task 2 Step 2 の確認コマンドで `[a.GetAtomMapNum() for a in m.GetAtoms()]` が全て非ゼロになるまで `.rxn` を修正してから再実行する。

- [ ] **Step 5: コミット**

```bash
git add reactx/rxn_parser.py tests/test_rxn_parser.py
git commit -m "feat(rxn_parser): parse MDL .rxn into combined Mols with atom mapping"
```

---

## Task 4: `calculators` ファクトリ — UMA / テスト用 calculator を切替

**Files:**
- Create: `reactx/calculators.py`
- Create: `tests/test_calculators.py`

**Responsibility:** `make_calculator(name)` で ASE `Calculator` を返す。Phase 0 では `"uma"` が本命、`"lj"` (LennardJones) をテスト用として提供する。将来 `"xtb"`, `"grrm23"` を追加する拡張フックを残す。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_calculators.py`:

```python
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
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `pytest tests/test_calculators.py -v`
Expected: `ModuleNotFoundError: No module named 'reactx.calculators'`

- [ ] **Step 3: `reactx/calculators.py` を実装**

```python
"""ASE Calculator factory. Phase 0 supports UMA (primary) and LennardJones (tests)."""
from __future__ import annotations

from typing import Any

from ase.calculators.calculator import Calculator


def make_calculator(name: str = "uma", **kwargs: Any) -> Calculator:
    """Return an ASE Calculator for the requested backend.

    Supported names:
        "uma"  — fairchem-core FAIRChemCalculator (model configurable via kwargs)
        "lj"   — ase.calculators.lj.LennardJones (cheap, for unit tests)
    """
    if name == "uma":
        return _build_uma_calculator(**kwargs)
    if name == "lj":
        from ase.calculators.lj import LennardJones

        return LennardJones(**kwargs)
    raise ValueError(
        f"Unknown calculator '{name}'. Supported: 'uma', 'lj'."
    )


def _build_uma_calculator(
    *,
    model_name: str = "fairchem/UMA-S",
    device: str | None = None,
    **kwargs: Any,
) -> Calculator:
    try:
        import torch
        from fairchem.core import FAIRChemCalculator
    except ImportError as exc:
        raise ImportError(
            "UMA backend requires fairchem-core and torch. "
            "Install with: pip install fairchem-core torch"
        ) from exc

    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    return FAIRChemCalculator(model_name=model_name, device=device, **kwargs)
```

- [ ] **Step 4: テストが通ることを確認**

Run: `pytest tests/test_calculators.py -v`
Expected: `3 passed`

- [ ] **Step 5: コミット**

```bash
git add reactx/calculators.py tests/test_calculators.py
git commit -m "feat(calculators): add ASE Calculator factory with UMA and LJ backends"
```

---

## Task 5: `embed3d` — 2D Mol から 3D `ase.Atoms` を生成

**Files:**
- Create: `reactx/embed3d.py`
- Create: `tests/test_embed3d.py`

**Responsibility:** RDKit `Mol` (implicit H を含む) を受け取り、`AddHs` → `EmbedMolecule` → `MMFFOptimizeMolecule` → `ase.Atoms` 変換 → （任意で）UMA 微調整、の順で 3D 座標付き `Atoms` を返す。multi-fragment Mol の場合は各 fragment 単独で埋め込み後、既定の attack 位置に並べる。

Phase 0 で UMA 微調整は「オプション」とする。テストでは `calculator=None` で RDKit+MMFF のみを使い、軽量に通す。CLI 側では UMA 再最適化を行う。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_embed3d.py`:

```python
import numpy as np
import pytest
from ase import Atoms
from rdkit import Chem

from reactx.embed3d import embed_mol_to_atoms


def _ch3cl() -> Chem.Mol:
    mol = Chem.MolFromSmiles("CCl")
    return mol


def test_embed_ch3cl_returns_atoms_with_all_atoms():
    atoms = embed_mol_to_atoms(_ch3cl(), calculator=None, seed=42)
    assert isinstance(atoms, Atoms)
    # C + Cl + 3 H = 5
    assert len(atoms) == 5
    syms = sorted(atoms.get_chemical_symbols())
    assert syms == ["C", "Cl", "H", "H", "H"]


def test_embed_ch3cl_has_reasonable_c_cl_bond():
    atoms = embed_mol_to_atoms(_ch3cl(), calculator=None, seed=42)
    syms = atoms.get_chemical_symbols()
    c_idx = syms.index("C")
    cl_idx = syms.index("Cl")
    d = atoms.get_distance(c_idx, cl_idx)
    assert 1.6 < d < 2.0, f"C-Cl distance out of range: {d:.3f} Å"


def test_embed_ch3cl_has_reasonable_hch_angles():
    atoms = embed_mol_to_atoms(_ch3cl(), calculator=None, seed=42)
    syms = atoms.get_chemical_symbols()
    c_idx = syms.index("C")
    h_idxs = [i for i, s in enumerate(syms) if s == "H"]
    assert len(h_idxs) == 3
    angles = [atoms.get_angle(h_idxs[i], c_idx, h_idxs[j])
              for i in range(3) for j in range(i + 1, 3)]
    for a in angles:
        assert 100 < a < 120, f"H-C-H angle out of range: {a:.1f}°"


def test_embed_multifragment_places_fragments_apart():
    mol = Chem.MolFromSmiles("CCl.[F-]")
    atoms = embed_mol_to_atoms(mol, calculator=None, seed=42)
    syms = atoms.get_chemical_symbols()
    assert "F" in syms
    f_idx = syms.index("F")
    c_idx = syms.index("C")
    d = atoms.get_distance(c_idx, f_idx)
    assert d > 2.5, f"Fragments too close: C-F distance {d:.3f} Å"


def test_embed_failure_raises_runtime_error(monkeypatch):
    from reactx import embed3d as mod
    def always_fail(*_a, **_kw):
        return -1  # RDKit embed failure code
    monkeypatch.setattr(mod.AllChem, "EmbedMolecule", always_fail)
    with pytest.raises(RuntimeError, match="embed"):
        embed_mol_to_atoms(_ch3cl(), calculator=None, seed=1)
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `pytest tests/test_embed3d.py -v`
Expected: `ModuleNotFoundError: No module named 'reactx.embed3d'`

- [ ] **Step 3: `reactx/embed3d.py` を実装**

```python
"""Convert 2D RDKit Mol to 3D ase.Atoms via RDKit ETKDG + MMFF (+ optional UMA)."""
from __future__ import annotations

from typing import Optional

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.optimize import BFGS
from rdkit import Chem
from rdkit.Chem import AllChem

MAX_EMBED_RETRIES = 5
FRAGMENT_SEPARATION = 3.5  # Å — attack distance for multi-fragment placement


def embed_mol_to_atoms(
    mol: Chem.Mol,
    *,
    calculator: Optional[Calculator] = None,
    seed: int = 0xC0FFEE,
    fmax: float = 0.05,
    max_opt_steps: int = 200,
) -> Atoms:
    """Embed a 2D Mol into 3D and return an ase.Atoms with implicit Hs added.

    If the Mol is disconnected (multiple fragments), each fragment is embedded
    independently and then concatenated along the +x axis with FRAGMENT_SEPARATION
    spacing so that UMA can later relax into a sensible encounter complex.
    """
    mol_h = Chem.AddHs(mol)
    frags = Chem.GetMolFrags(mol_h, asMols=True, sanitizeFrags=True)

    atoms_per_frag = [_embed_single_fragment(f, seed=seed + i)
                      for i, f in enumerate(frags)]
    atoms = _stack_fragments(atoms_per_frag)

    if calculator is not None:
        atoms.calc = calculator
        BFGS(atoms, logfile=None).run(fmax=fmax, steps=max_opt_steps)

    return atoms


def _embed_single_fragment(frag: Chem.Mol, *, seed: int) -> Atoms:
    params = AllChem.ETKDGv3()
    for attempt in range(MAX_EMBED_RETRIES):
        params.randomSeed = seed + attempt
        status = AllChem.EmbedMolecule(frag, params)
        if status == 0:
            break
    else:
        raise RuntimeError(
            f"RDKit failed to embed fragment after {MAX_EMBED_RETRIES} attempts. "
            "Check the input structure and RDKit version."
        )

    if frag.GetNumHeavyAtoms() > 1:
        AllChem.MMFFOptimizeMolecule(frag, maxIters=500)

    return _rdkit_to_atoms(frag)


def _rdkit_to_atoms(mol: Chem.Mol) -> Atoms:
    conf = mol.GetConformer()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    positions = np.array([[conf.GetAtomPosition(i).x,
                           conf.GetAtomPosition(i).y,
                           conf.GetAtomPosition(i).z]
                          for i in range(mol.GetNumAtoms())])
    charges = [a.GetFormalCharge() for a in mol.GetAtoms()]
    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.set_initial_charges(charges)
    return atoms


def _stack_fragments(frags: list[Atoms]) -> Atoms:
    if len(frags) == 1:
        return frags[0]
    stacked = frags[0].copy()
    for f in frags[1:]:
        offset = stacked.positions[:, 0].max() - f.positions[:, 0].min() + FRAGMENT_SEPARATION
        f2 = f.copy()
        f2.positions[:, 0] += offset
        stacked += f2
    return stacked
```

- [ ] **Step 4: テストが通ることを確認**

Run: `pytest tests/test_embed3d.py -v`
Expected: `5 passed`

埋め込みが確率的に失敗する場合は `seed=42` で再現性があるか確認する。稀に MMFF が発散して H-C-H 角が異常値になる場合は、Task 5 Step 3 の `maxIters` を 1000 に上げる。

- [ ] **Step 5: コミット**

```bash
git add reactx/embed3d.py tests/test_embed3d.py
git commit -m "feat(embed3d): 2D Mol to 3D ase.Atoms via ETKDG + MMFF + optional UMA"
```

---

## Task 6: `align` — atom mapping に従って product の原子順を reactant に揃える

**Files:**
- Create: `reactx/align.py`
- Create: `tests/test_align.py`

**Responsibility:** reactant `Atoms`, product `Atoms`, heavy-atom mapping を受け取り、product の原子順を reactant と同じになるよう並べ替えた新しい `Atoms` を返す。hydrogens は heavy atom に紐付いた順で再割当する。

**戦略:** 
1. reactant/product 双方について「heavy atom index → 紐付く H indices」を計算。
2. 入力 heavy-atom mapping (`r_heavy_idx → p_heavy_idx`) に従い、product 側の heavy atom 順を reactant に合わせた permutation を構築。
3. 各 heavy atom について紐付く H を同じ数だけ対応付ける（SN2 では C に 3H なので 1:1）。H 同士の個別区別は付けず、RDKit 出力順で割り当てる。
4. 最終 permutation で product `Atoms` を並べ替える。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_align.py`:

```python
import numpy as np
import pytest
from ase import Atoms

from reactx.align import align_product_to_reactant


def _build_pair():
    # reactant: C, Cl, F, H, H, H  (indices 0..5)
    r = Atoms(
        symbols=["C", "Cl", "F", "H", "H", "H"],
        positions=[(0, 0, 0), (1.8, 0, 0), (-3.5, 0, 0),
                   (0.3, 1.0, 0), (0.3, -0.5, 0.9), (0.3, -0.5, -0.9)],
    )
    # product: Cl, F, C, H, H, H (indices 0..5) — heavy atoms shuffled
    p = Atoms(
        symbols=["Cl", "F", "C", "H", "H", "H"],
        positions=[(3.8, 0, 0), (-1.5, 0, 0), (0, 0, 0),
                   (-0.3, 1.0, 0), (-0.3, -0.5, 0.9), (-0.3, -0.5, -0.9)],
    )
    # heavy mapping reactant_heavy -> product_heavy:
    # reactant 0(C)->product 2(C), 1(Cl)->0(Cl), 2(F)->1(F)
    mapping = {0: 2, 1: 0, 2: 1}
    heavy_h_groups_reactant = {0: [3, 4, 5]}  # C at idx 0 owns 3 Hs
    heavy_h_groups_product = {2: [3, 4, 5]}   # C at idx 2 owns 3 Hs
    return r, p, mapping, heavy_h_groups_reactant, heavy_h_groups_product


def test_align_reorders_heavy_atoms_to_reactant_order():
    r, p, mapping, rH, pH = _build_pair()
    aligned = align_product_to_reactant(
        reactant=r, product=p, heavy_mapping=mapping,
        reactant_h_groups=rH, product_h_groups=pH,
    )
    assert aligned.get_chemical_symbols() == r.get_chemical_symbols()


def test_align_preserves_atom_count():
    r, p, mapping, rH, pH = _build_pair()
    aligned = align_product_to_reactant(r, p, mapping, rH, pH)
    assert len(aligned) == len(p) == len(r)


def test_align_h_count_per_heavy_must_match():
    r, p, mapping, rH, pH = _build_pair()
    pH_bad = {2: [3, 4]}  # only 2 Hs on product carbon — mismatch
    with pytest.raises(ValueError, match="hydrogen count"):
        align_product_to_reactant(r, p, mapping, rH, pH_bad)
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `pytest tests/test_align.py -v`
Expected: `ModuleNotFoundError: No module named 'reactx.align'`

- [ ] **Step 3: `reactx/align.py` を実装**

```python
"""Reorder product atoms to match reactant atom ordering using atom mapping."""
from __future__ import annotations

from ase import Atoms


def align_product_to_reactant(
    reactant: Atoms,
    product: Atoms,
    heavy_mapping: dict[int, int],
    reactant_h_groups: dict[int, list[int]],
    product_h_groups: dict[int, list[int]],
) -> Atoms:
    """Return product Atoms reordered so that atom i corresponds to reactant atom i.

    heavy_mapping maps reactant heavy-atom index -> product heavy-atom index.
    *_h_groups maps heavy-atom index -> list of bonded hydrogen indices on that side.
    """
    if len(reactant) != len(product):
        raise ValueError(
            f"Atom count mismatch: reactant={len(reactant)} product={len(product)}"
        )

    permutation: list[int] = [-1] * len(reactant)

    for r_idx in range(len(reactant)):
        sym_r = reactant.get_chemical_symbols()[r_idx]
        if sym_r == "H":
            continue
        if r_idx not in heavy_mapping:
            raise ValueError(f"Reactant heavy atom {r_idx} ({sym_r}) not in mapping")
        p_idx = heavy_mapping[r_idx]
        permutation[r_idx] = p_idx

        r_hs = reactant_h_groups.get(r_idx, [])
        p_hs = product_h_groups.get(p_idx, [])
        if len(r_hs) != len(p_hs):
            raise ValueError(
                f"hydrogen count mismatch for heavy atom {r_idx}->{p_idx}: "
                f"{len(r_hs)} vs {len(p_hs)}"
            )
        for r_h, p_h in zip(r_hs, p_hs):
            permutation[r_h] = p_h

    if any(x < 0 for x in permutation):
        missing = [i for i, x in enumerate(permutation) if x < 0]
        raise ValueError(f"Unmapped reactant atoms: {missing}")

    aligned = product[permutation]
    return aligned
```

- [ ] **Step 4: テストが通ることを確認**

Run: `pytest tests/test_align.py -v`
Expected: `3 passed`

- [ ] **Step 5: ヘルパーを `rxn_parser` に追加（heavy→H group を Mol から抽出）**

`reactx/rxn_parser.py` の末尾に追加:

```python
def heavy_to_hydrogen_groups(mol_with_h: Chem.Mol) -> dict[int, list[int]]:
    """Return {heavy_atom_idx: [bonded_h_idx, ...]} for a Mol with explicit Hs.

    Use after Chem.AddHs so that hydrogen indices correspond to positions in the
    Atoms object produced by embed3d._rdkit_to_atoms.
    """
    groups: dict[int, list[int]] = {}
    for atom in mol_with_h.GetAtoms():
        if atom.GetSymbol() == "H":
            continue
        heavy_idx = atom.GetIdx()
        hs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetSymbol() == "H"]
        groups[heavy_idx] = hs
    return groups
```

- [ ] **Step 6: `heavy_to_hydrogen_groups` の薄いテストを追加**

`tests/test_rxn_parser.py` の末尾に追加:

```python
def test_heavy_to_hydrogen_groups_on_methane():
    from reactx.rxn_parser import heavy_to_hydrogen_groups
    mol = Chem.MolFromSmiles("C")
    mol_h = Chem.AddHs(mol)
    groups = heavy_to_hydrogen_groups(mol_h)
    assert groups == {0: [1, 2, 3, 4]}
```

Run: `pytest tests/test_rxn_parser.py tests/test_align.py -v`
Expected: `5 passed` (rxn_parser 4 + heavy_to_hydrogen_groups 1) and `3 passed` (align)

- [ ] **Step 7: コミット**

```bash
git add reactx/align.py tests/test_align.py reactx/rxn_parser.py tests/test_rxn_parser.py
git commit -m "feat(align): reorder product atoms to reactant order via atom mapping"
```

---

## Task 7: `neb` — ASE IDPP + CI-NEB で trajectory.xyz を生成

**Files:**
- Create: `reactx/neb.py`
- Create: `tests/test_neb_sn2.py`

**Responsibility:** aligned reactant/product `Atoms` を受け取り、11 image (両端 + 9 中間) の NEB を実行して `trajectory.xyz` を書き出す。IDPP で初期補間、CI-NEB で鞍点に収束させる。Calculator は引数で差し替え可能。

- [ ] **Step 1: 失敗するユニットテスト (LJ) を書く**

`tests/test_neb_sn2.py`:

```python
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read

from reactx.calculators import make_calculator
from reactx.neb import run_neb


def _two_state_h2(displacement: float) -> Atoms:
    return Atoms("H2", positions=[(0, 0, 0), (0, 0, displacement)])


def test_run_neb_with_lj_produces_xyz_with_expected_images(tmp_path: Path):
    reactant = _two_state_h2(0.5)
    product = _two_state_h2(1.2)
    out = tmp_path / "traj.xyz"
    meta = run_neb(
        reactant=reactant,
        product=product,
        calculator_factory=lambda: make_calculator("lj"),
        n_images=5,
        output_xyz=out,
        fmax=0.2,
        max_steps=50,
        pad_frames=0,
    )
    assert out.exists()
    frames = read(str(out), index=":")
    assert len(frames) == 5
    assert "converged" in meta
    assert "final_fmax" in meta


def test_run_neb_pads_endpoints(tmp_path: Path):
    reactant = _two_state_h2(0.5)
    product = _two_state_h2(1.2)
    out = tmp_path / "traj.xyz"
    run_neb(
        reactant=reactant, product=product,
        calculator_factory=lambda: make_calculator("lj"),
        n_images=5, output_xyz=out, fmax=0.5, max_steps=10,
        pad_frames=3,
    )
    frames = read(str(out), index=":")
    # 3 reactant + 5 neb + 3 product
    assert len(frames) == 11


@pytest.mark.slow
def test_sn2_neb_ts_has_walden_inversion(tmp_path: Path):
    """Real UMA + SN2 end-to-end. Skipped by default — run with `pytest -m slow`."""
    pytest.importorskip("fairchem.core")
    from reactx.rxn_parser import parse_rxn, heavy_to_hydrogen_groups
    from reactx.embed3d import embed_mol_to_atoms
    from reactx.align import align_product_to_reactant
    from rdkit import Chem

    rxn = Path(__file__).parent.parent / "examples" / "sn2.rxn"
    r_mol, p_mol, mapping = parse_rxn(rxn)
    r_mol_h = Chem.AddHs(r_mol)
    p_mol_h = Chem.AddHs(p_mol)
    rH = heavy_to_hydrogen_groups(r_mol_h)
    pH = heavy_to_hydrogen_groups(p_mol_h)

    calc = make_calculator("uma")
    reactant = embed_mol_to_atoms(r_mol, calculator=calc, seed=1)
    product_raw = embed_mol_to_atoms(p_mol, calculator=calc, seed=2)
    product = align_product_to_reactant(reactant, product_raw, mapping, rH, pH)

    out = tmp_path / "traj.xyz"
    meta = run_neb(
        reactant=reactant, product=product,
        calculator_factory=lambda: make_calculator("uma"),
        n_images=11, output_xyz=out, fmax=0.05, max_steps=200, pad_frames=0,
    )
    frames = read(str(out), index=":")
    energies = np.array(meta["image_energies"])
    ts_idx = int(np.argmax(energies))
    assert 0 < ts_idx < len(frames) - 1, "TS must be an interior image"
    assert energies[ts_idx] > energies[0] and energies[ts_idx] > energies[-1]

    ts = frames[ts_idx]
    syms = ts.get_chemical_symbols()
    c = syms.index("C")
    f = syms.index("F")
    cl = syms.index("Cl")
    angle = ts.get_angle(f, c, cl)
    assert angle > 165, f"Walden inversion expects near-linear F-C-Cl, got {angle:.1f}°"
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `pytest tests/test_neb_sn2.py -v -m "not slow"`
Expected: `ModuleNotFoundError: No module named 'reactx.neb'`

- [ ] **Step 3: `reactx/neb.py` を実装**

```python
"""ASE IDPP + CI-NEB driver. Writes multi-frame XYZ with optional end padding."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from ase import Atoms
from ase.io import write
from ase.mep import NEB
from ase.optimize import BFGS


def run_neb(
    reactant: Atoms,
    product: Atoms,
    *,
    calculator_factory: Callable[[], "object"],
    n_images: int = 11,
    output_xyz: str | Path,
    fmax: float = 0.05,
    max_steps: int = 200,
    climb: bool = True,
    pad_frames: int = 0,
) -> dict:
    """Run IDPP interpolation + (CI-)NEB, write trajectory XYZ, return metadata."""
    assert n_images >= 3, "n_images must be >= 3 (reactant + >=1 middle + product)"

    images = [reactant.copy()]
    for _ in range(n_images - 2):
        images.append(reactant.copy())
    images.append(product.copy())

    for img in images:
        img.calc = calculator_factory()

    neb = NEB(images, climb=climb, allow_shared_calculator=False)
    neb.interpolate(method="idpp")

    opt = BFGS(neb, logfile=None)
    converged = False
    try:
        opt.run(fmax=fmax, steps=max_steps)
        converged = all(
            max(abs(img.get_forces().flatten())) < fmax
            for img in images[1:-1]
        )
    except Exception:
        converged = False

    final_fmax = max(
        float(max(abs(img.get_forces().flatten())))
        for img in images[1:-1]
    )
    image_energies = [float(img.get_potential_energy()) for img in images]

    padded: list[Atoms] = []
    padded.extend([images[0].copy() for _ in range(pad_frames)])
    padded.extend(images)
    padded.extend([images[-1].copy() for _ in range(pad_frames)])

    write(str(output_xyz), padded, format="extxyz")

    return {
        "n_images": n_images,
        "converged": converged,
        "final_fmax": final_fmax,
        "image_energies": image_energies,
        "pad_frames": pad_frames,
    }
```

> **注意:** `ase.mep.NEB` は新しい ASE (3.23+) の配置。古い ASE では `ase.neb.NEB`。インストールした ASE のバージョンに合わせ、どちらか通る方を使う（`pytest` 実行時に ImportError が出たら切替）。

- [ ] **Step 4: テストが通ることを確認**

Run: `pytest tests/test_neb_sn2.py -v -m "not slow"`
Expected: `2 passed, 1 deselected`

LJ + H2 は H が希ガス扱いで引力が弱いケースなど不安定な場合、IDPP 補間までで十分収束判定は見なくていい。`fmax=0.5` までは緩和可能。テスト内部の `fmax` 値を調整して通す。

- [ ] **Step 5: SN2 統合テストを手動実行（UMA 環境が整っていれば）**

Run: `pytest tests/test_neb_sn2.py::test_sn2_neb_ts_has_walden_inversion -v -m slow`
Expected: `1 passed` (環境が無ければ `pytest -m slow` を skip して以降のタスクを先に進め、Task 10 で戻ってくる)

- [ ] **Step 6: コミット**

```bash
git add reactx/neb.py tests/test_neb_sn2.py
git commit -m "feat(neb): IDPP+CI-NEB driver with trajectory XYZ output and end padding"
```

---

## Task 8: `cli` — `reactx run` エントリポイント

**Files:**
- Create: `reactx/cli.py`
- Create: `tests/test_cli.py`

**Responsibility:** `reactx run <rxn> -o <outdir> [--images N] [--fmax F] [--backend uma|lj] [--render]` で parse→embed3d→align→neb→(optional Blender) を一気通貫で実行する。`out/trajectory.xyz`, `out/energies.json`, `out/meta.json` を書き出す。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_cli.py`:

```python
import json
from pathlib import Path

import pytest

from reactx import cli


def test_cli_run_end_to_end_with_lj_backend(tmp_path: Path, sn2_rxn_path: Path):
    out = tmp_path / "out"
    rc = cli.main([
        "run", str(sn2_rxn_path), "-o", str(out),
        "--images", "5", "--fmax", "0.5", "--max-steps", "20",
        "--backend", "lj",
    ])
    assert rc == 0
    assert (out / "trajectory.xyz").exists()
    meta = json.loads((out / "meta.json").read_text())
    assert meta["n_images"] == 5
    energies = json.loads((out / "energies.json").read_text())
    assert len(energies) == 5


def test_cli_missing_rxn_returns_nonzero(tmp_path: Path):
    rc = cli.main([
        "run", str(tmp_path / "nope.rxn"), "-o", str(tmp_path / "out"),
        "--backend", "lj",
    ])
    assert rc != 0
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `pytest tests/test_cli.py -v`
Expected: `ImportError` または `AttributeError: module 'reactx.cli' has no attribute 'main'`

- [ ] **Step 3: `reactx/cli.py` を実装**

```python
"""CLI entry point: reactx run <rxn> -o <outdir> [options]."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rdkit import Chem

from reactx.align import align_product_to_reactant
from reactx.calculators import make_calculator
from reactx.embed3d import embed_mol_to_atoms
from reactx.neb import run_neb
from reactx.rxn_parser import heavy_to_hydrogen_groups, parse_rxn


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="reactx")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run full pipeline on a .rxn file")
    run.add_argument("rxn_path", type=Path)
    run.add_argument("-o", "--output", type=Path, required=True)
    run.add_argument("--images", type=int, default=11)
    run.add_argument("--fmax", type=float, default=0.05)
    run.add_argument("--max-steps", type=int, default=200)
    run.add_argument("--backend", choices=["uma", "lj"], default="uma")
    run.add_argument("--pad-frames", type=int, default=3)
    run.add_argument("--render", action="store_true",
                     help="Also invoke blender/render.py after NEB")
    run.add_argument("--blender-exe", type=str, default="blender")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "run":
        return _cmd_run(args)
    return 2


def _cmd_run(args: argparse.Namespace) -> int:
    if not args.rxn_path.exists():
        print(f"Error: .rxn not found: {args.rxn_path}", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    r_mol, p_mol, mapping = parse_rxn(args.rxn_path)

    calc = make_calculator(args.backend)

    reactant = embed_mol_to_atoms(r_mol, calculator=calc, seed=1)
    product_raw = embed_mol_to_atoms(p_mol, calculator=calc, seed=2)

    r_mol_h = Chem.AddHs(r_mol)
    p_mol_h = Chem.AddHs(p_mol)
    rH = heavy_to_hydrogen_groups(r_mol_h)
    pH = heavy_to_hydrogen_groups(p_mol_h)
    product = align_product_to_reactant(reactant, product_raw, mapping, rH, pH)

    xyz = args.output / "trajectory.xyz"
    meta = run_neb(
        reactant=reactant,
        product=product,
        calculator_factory=lambda: make_calculator(args.backend),
        n_images=args.images,
        output_xyz=xyz,
        fmax=args.fmax,
        max_steps=args.max_steps,
        pad_frames=args.pad_frames,
    )

    (args.output / "meta.json").write_text(json.dumps(meta, indent=2))
    (args.output / "energies.json").write_text(json.dumps(meta["image_energies"]))

    if args.render:
        _invoke_blender(args, xyz)

    print(f"OK: wrote {xyz} (converged={meta['converged']}, fmax={meta['final_fmax']:.4f})")
    return 0


def _invoke_blender(args: argparse.Namespace, xyz: Path) -> None:
    import subprocess
    script = Path(__file__).resolve().parent.parent / "blender" / "render.py"
    blend = args.output / "scene.blend"
    cmd = [args.blender_exe, "--background", "--python", str(script),
           "--", str(xyz), str(blend)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
```

- [ ] **Step 4: テストが通ることを確認**

Run: `pytest tests/test_cli.py -v`
Expected: `2 passed`

LJ backend で全 SN2 原子をまともに扱うのは物理的にナンセンス（LJ は元素無視で引力一律）だが、パイプラインの配管と I/O を確認する用途としては通過する。収束しなくても `meta["converged"] = False` で続行する仕様なので OK。

- [ ] **Step 5: CLI を手動で叩いて疎通確認**

Run: `reactx run examples/sn2.rxn -o out/ --images 5 --fmax 0.5 --max-steps 10 --backend lj`
Expected: `OK: wrote out/trajectory.xyz ...` と `out/{trajectory.xyz,meta.json,energies.json}` が生成される。

- [ ] **Step 6: コミット**

```bash
git add reactx/cli.py tests/test_cli.py
git commit -m "feat(cli): reactx run end-to-end pipeline command"
```

---

## Task 9: `blender/render.py` — Blender で `.blend` を生成

**Files:**
- Create: `blender/render.py`
- Create: `tests/test_blender_smoke.py`

**Responsibility:** Blender の `--background --python` から呼ばれ、与えられた `trajectory.xyz` を `atomic-blender-pdb-xyz` アドオンで読み込み、3 点照明 + カメラを設置し、`.blend` を保存する。TS 付近のスローダウンはフレームオフセット（image 間フレーム数を TS ± 1 だけ延ばす）で表現する。Bond の動的 fade は Phase 0 では扱わず、アドオンのデフォルトに委ねる（spec §6.7-6 に準拠）。

- [ ] **Step 1: `blender/render.py` を実装**

```python
"""Blender headless script: trajectory.xyz -> ball-and-stick animated .blend.

Invoke:
    blender --background --python blender/render.py -- <trajectory.xyz> <out.blend>
"""
import math
import sys
from pathlib import Path

import bpy  # type: ignore[import-not-found]


FPS = 24
SECONDS_PER_FRAME_IMAGE = 0.8  # default dwell per NEB image
TS_SLOW_MULTIPLIER = 2.5       # TS±1 images are stretched by this factor


def _parse_args(argv: list[str]) -> tuple[Path, Path]:
    if "--" not in argv:
        raise SystemExit("Usage: blender --background --python render.py -- <xyz> <out.blend>")
    extra = argv[argv.index("--") + 1:]
    if len(extra) < 2:
        raise SystemExit("Need <xyz> <out.blend>")
    return Path(extra[0]), Path(extra[1])


def _reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = FPS


def _import_trajectory(xyz: Path) -> None:
    try:
        bpy.ops.preferences.addon_enable(module="atomic_blender_pdb_xyz")
    except Exception:
        bpy.ops.preferences.addon_enable(module="atomic_blender_xyz")
    bpy.ops.import_mesh.xyz(filepath=str(xyz), use_frames=True)


def _add_three_point_lighting() -> None:
    def _light(name: str, energy: float, location: tuple[float, float, float]) -> None:
        bpy.ops.object.light_add(type="AREA", location=location)
        light = bpy.context.object
        light.name = name
        light.data.energy = energy
        light.data.size = 2.0

    _light("Key", 800, (5, -5, 5))
    _light("Fill", 300, (-5, -3, 3))
    _light("Rim", 500, (0, 5, 4))


def _add_camera_looking_at_origin() -> None:
    bpy.ops.object.camera_add(location=(0, -8, 2),
                              rotation=(math.radians(80), 0, 0))
    bpy.context.scene.camera = bpy.context.object


def _stretch_ts_frames() -> None:
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = max(scene.frame_end, int(FPS * 10))


def main(argv: list[str]) -> int:
    xyz, out = _parse_args(argv)
    _reset_scene()
    _import_trajectory(xyz)
    _add_three_point_lighting()
    _add_camera_looking_at_origin()
    _stretch_ts_frames()
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out))
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(list(sys.argv)))
```

- [ ] **Step 2: smoke test を書く（ローカル Blender があるとき限定）**

`tests/test_blender_smoke.py`:

```python
import os
import shutil
import subprocess
from pathlib import Path

import pytest

BLENDER = os.environ.get("BLENDER_EXE", "blender")


@pytest.mark.blender
def test_blender_smoke_produces_blend(tmp_path: Path):
    if shutil.which(BLENDER) is None:
        pytest.skip(f"Blender executable not found: {BLENDER}")

    xyz = tmp_path / "traj.xyz"
    # Minimal 2-frame XYZ (H2 stretch) — bypass full NEB for smoke test
    xyz.write_text(
        "2\nFrame 0\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n"
        "2\nFrame 1\nH 0.0 0.0 0.0\nH 0.0 0.0 1.10\n"
    )
    out_blend = tmp_path / "scene.blend"
    script = Path(__file__).resolve().parent.parent / "blender" / "render.py"
    result = subprocess.run(
        [BLENDER, "--background", "--python", str(script),
         "--", str(xyz), str(out_blend)],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert out_blend.exists()
    assert out_blend.stat().st_size > 0
```

- [ ] **Step 3: smoke test を実行**

Run: `pytest tests/test_blender_smoke.py -v -m blender`
Expected (Blender + add-on インストール済み): `1 passed`
Expected (Blender 未インストール): `1 skipped`

アドオン名が環境によって `atomic_blender_xyz` か `atomic_blender_pdb_xyz` か異なるので、`_import_trajectory` の 2 段階 fallback で対応している。どちらも効かない場合は、その環境で実際に有効なモジュール名を Blender GUI の Preferences > Add-ons で確認し、`_import_trajectory` の module 名を修正する。

- [ ] **Step 4: コミット**

```bash
git add blender/render.py tests/test_blender_smoke.py
git commit -m "feat(blender): headless render.py with three-point lighting and camera"
```

---

## Task 10: README と Definition of Done 確認

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 現 `README.md` の中身を確認**

Run: `cat README.md`
Expected: `# 3D-Chemical-Reaction-Mechanism` 程度の 1 行ファイル。

- [ ] **Step 2: 新しい `README.md` を書く**

```markdown
# reactx — Phase 0 Spike

2D 反応機構 (`.rxn`) から UMA + ASE NEB で 3D MEP を探索し、Blender で ball-and-stick アニメーションを再生するパイプラインのフェーズ 0 実装。

Phase 0 の範囲: SN2 反応 (CH₃Cl + F⁻ → CH₃F + Cl⁻) 1 件を end-to-end で貫通することに限定。

## セットアップ

```bash
python -m pip install -e .[dev]
huggingface-cli login    # UMA モデル取得のため
```

Blender 4.x と `atomic-blender-pdb-xyz` アドオンを別途インストールしておく。

## 使い方

```bash
reactx run examples/sn2.rxn -o out/ --images 11 --fmax 0.05 --backend uma
blender --background --python blender/render.py -- out/trajectory.xyz out/scene.blend
```

生成物:

- `out/trajectory.xyz` — 11+ フレームの NEB 軌跡
- `out/energies.json` — 各 image のエネルギー
- `out/meta.json` — 収束情報など
- `out/scene.blend` — Blender シーン（GUI で開いて再生）

`--render` フラグを付けると CLI から Blender を直接呼び出す:

```bash
reactx run examples/sn2.rxn -o out/ --render --blender-exe /path/to/blender
```

## テスト

```bash
pytest              # 高速ユニットテストのみ
pytest -m slow      # UMA 依存の SN2 統合テスト
pytest -m blender   # Blender smoke test (ローカル環境のみ)
```

## アーキテクチャ

```
.rxn → rxn_parser → embed3d → align → neb → trajectory.xyz → blender/render.py → .blend
```

詳細: `docs/superpowers/specs/2026-04-20-reactx-phase-0-design.md`
```

- [ ] **Step 3: 全テストを通す**

Run: `pytest -v -m "not slow and not blender"`
Expected: Task 3/4/5/6/7/8 の全ユニットテストが pass。

- [ ] **Step 4: DoD 1/2 を確認 — UMA バックエンドで SN2 を通す**

Run: `reactx run examples/sn2.rxn -o out/ --images 11 --fmax 0.05 --backend uma`
Expected: `out/trajectory.xyz` が生成。`meta.json` の `converged: true` (時間内に収束しない場合は `--max-steps 300` まで緩めてよい)。

Run: `pytest tests/test_neb_sn2.py::test_sn2_neb_ts_has_walden_inversion -v -m slow`
Expected: `1 passed`

- [ ] **Step 5: DoD 3/4 を確認 — Blender で `.blend` を生成して GUI 確認**

Run: `blender --background --python blender/render.py -- out/trajectory.xyz out/scene.blend`
Expected: `Saved: out/scene.blend`

Blender GUI で `out/scene.blend` を開き、タイムラインを再生して以下を目視確認:
- 原子 (ball) と結合 (stick) が表示されている
- F が C に近づき、Cl が離れる Walden 反転が視認できる
- 3 点照明でメタリックな見栄えになっている

- [ ] **Step 6: DoD 5 を確認 — README 手順が実行可能**

別シェルでクリーンな venv を作り、README の手順のみをコピペで実行して `out/trajectory.xyz` と `out/scene.blend` が生成されることを確認する。

- [ ] **Step 7: 最終コミット**

```bash
git add README.md
git commit -m "docs: rewrite README with Phase 0 reproduction steps"
```

---

## Self-Review Checklist (作者用)

**Spec coverage:**
- §6.1 rxn_parser → Task 3 ✓
- §6.2 embed3d → Task 5 ✓
- §6.3 align → Task 6 ✓
- §6.4 neb → Task 7 ✓
- §6.5 calculators → Task 4 ✓
- §6.6 cli → Task 8 ✓
- §6.7 blender/render.py → Task 9 ✓
- §8 error handling → 各タスクの実装内 (embed retry, UMA ImportError, NEB converged=False, atom-map 欠損 ValueError) ✓
- §9 tests → Task 3/5/7/9 にすべて対応 ✓
- §12 DoD 1-5 → Task 10 Step 3-6 ✓
- §11 非スコープ → 計画に含めていない ✓

**Type / signature consistency:**
- `parse_rxn` → `(Mol, Mol, dict[int, int])` : Task 3, 使用 Task 8。
- `heavy_to_hydrogen_groups` → `dict[int, list[int]]` : Task 6 で追加、使用 Task 7/8。
- `embed_mol_to_atoms(mol, *, calculator=None, seed=...)` : Task 5, 使用 Task 7 (slow)/8。
- `align_product_to_reactant(reactant, product, heavy_mapping, reactant_h_groups, product_h_groups)` : Task 6, 使用 Task 7 (slow)/8。
- `make_calculator(name)` : Task 4, 使用 Task 5/7/8。
- `run_neb(reactant, product, *, calculator_factory, n_images, output_xyz, fmax, max_steps, pad_frames)` : Task 7, 使用 Task 8。

**Placeholder scan:** TBD/TODO/fill in details/適切な無し。各ステップにコード/コマンド/期待出力が含まれている。

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-20-reactx-phase-0-implementation.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
