# vdW 半径ベース原子球リスケール 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `blender/render.py` で生成される ball-and-stick アニメーションの原子球サイズを Alvarez (2013) van der Waals 半径 × 0.25 に揃え、UMA `omol` タスク (OMol25) が訓練対応する Z=1〜83 (H〜Bi) を全てカバーする。

**Architecture:** `atomic-blender-pdb-xyz` アドオンによる XYZ import 後の post-processing。シーン内の `<ElementName>_ball` オブジェクトを走査し、Alvarez 2013 の vdW 半径テーブルから引いた値で `obj.scale` を上書きする。設定値は環境変数 `REACTX_VDW_SCALE` で上書き可能。

**Tech Stack:** Blender 4.x の bpy API、Python 3.10+、pytest。新規依存なし。

**Spec:** `docs/superpowers/specs/2026-04-26-vdw-radii-design.md`

---

## File Structure

変更:

```
blender/
  render.py            # vdW テーブル + 名称→記号辞書 + リスケール関数を追加
tests/
  test_blender_smoke.py  # test_blender_vdw_rescale を追加
```

新規作成: なし

既存テストへの影響: なし (`test_blender_smoke_produces_blend` は引き続きパスする)

---

## Task 1: 失敗するスモークテストを追加

**Files:**
- Modify: `tests/test_blender_smoke.py`

- [ ] **Step 1: 失敗するテストを追記**

`tests/test_blender_smoke.py` の末尾に以下を追加:

```python
import re


@pytest.mark.blender
def test_blender_vdw_rescale(tmp_path: Path):
    if shutil.which(BLENDER) is None:
        pytest.skip(f"Blender executable not found: {BLENDER}")

    xyz = tmp_path / "traj.xyz"
    # 1-frame minimal CH (H + C) — exercises both Hydrogen_ball and Carbon_ball
    xyz.write_text(
        "2\nFrame 0\nH 0.0 0.0 0.0\nC 0.0 0.0 1.10\n"
    )
    out_blend = tmp_path / "scene.blend"
    script = Path(__file__).resolve().parent.parent / "blender" / "render.py"
    result = subprocess.run(
        [BLENDER, "--background", "--python", str(script),
         "--", str(xyz), str(out_blend)],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stderr

    pattern = re.compile(r"\[reactx\] vdw-rescale: (\S+) scale=(\S+)")
    scales: dict[str, float] = {}
    for line in result.stdout.splitlines():
        m = pattern.search(line)
        if m:
            scales[m.group(1)] = float(m.group(2))

    assert "Hydrogen" in scales, f"no Hydrogen rescale log; stdout was:\n{result.stdout}"
    assert "Carbon" in scales, f"no Carbon rescale log; stdout was:\n{result.stdout}"
    # Alvarez 2013 vdW: H=1.20, C=1.77; default scale=0.25.
    assert abs(scales["Hydrogen"] - 1.20 * 0.25) < 1e-3
    assert abs(scales["Carbon"] - 1.77 * 0.25) < 1e-3
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `pytest tests/test_blender_smoke.py::test_blender_vdw_rescale -v -m blender`

Expected: FAIL — `assert "Hydrogen" in scales` で AssertionError (現在の `render.py` は `[reactx] vdw-rescale:` ログを出さないため `scales` が空)。

Blender が無いマシンで実行する場合は `pytest.skip` でスキップされる — その場合は次タスクに進んで実装後に Blender を持つ環境で再実行する。

- [ ] **Step 3: コミットしない**

このタスクの最後ではコミットしない。Task 5 で実装と一緒にコミットする (TDD で red→green を 1 コミットにまとめる方針)。

---

## Task 2: vdW 半径テーブルと名称→記号辞書を追加

**Files:**
- Modify: `blender/render.py`

- [ ] **Step 1: 既存 import 直後に定数ブロックを追加**

`blender/render.py` の `import bpy` の **直後** (現状 11 行目以降) に以下を挿入:

```python
import os


# Alvarez (2013) "A cartography of the van der Waals territories"
# Dalton Trans. 42, 8617. Values in Angstrom for Z=1..83 (H..Bi),
# matching OMol25 / UMA omol task element coverage exactly.
VDW_RADII_ANGSTROM: dict[str, float] = {
    "H": 1.20, "He": 1.43,
    "Li": 2.12, "Be": 1.98, "B": 1.91, "C": 1.77, "N": 1.66, "O": 1.50,
    "F": 1.46, "Ne": 1.58,
    "Na": 2.50, "Mg": 2.51, "Al": 2.25, "Si": 2.19, "P": 1.90, "S": 1.89,
    "Cl": 1.82, "Ar": 1.83,
    "K": 2.73, "Ca": 2.62, "Sc": 2.58, "Ti": 2.46, "V": 2.42, "Cr": 2.45,
    "Mn": 2.45, "Fe": 2.44, "Co": 2.40, "Ni": 2.40, "Cu": 2.38, "Zn": 2.39,
    "Ga": 2.32, "Ge": 2.29, "As": 1.88, "Se": 1.82, "Br": 1.86, "Kr": 2.25,
    "Rb": 3.21, "Sr": 2.84, "Y": 2.75, "Zr": 2.52, "Nb": 2.56, "Mo": 2.45,
    "Tc": 2.44, "Ru": 2.46, "Rh": 2.44, "Pd": 2.15, "Ag": 2.53, "Cd": 2.49,
    "In": 2.43, "Sn": 2.42, "Sb": 2.47, "Te": 1.99, "I": 2.04, "Xe": 2.06,
    "Cs": 3.48, "Ba": 3.03,
    "La": 2.98, "Ce": 2.88, "Pr": 2.92, "Nd": 2.95, "Pm": 2.93, "Sm": 2.90,
    "Eu": 2.87, "Gd": 2.83, "Tb": 2.79, "Dy": 2.87, "Ho": 2.81, "Er": 2.83,
    "Tm": 2.79, "Yb": 2.80, "Lu": 2.74,
    "Hf": 2.63, "Ta": 2.53, "W": 2.57, "Re": 2.49, "Os": 2.48, "Ir": 2.41,
    "Pt": 2.29, "Au": 2.32, "Hg": 2.45, "Tl": 2.47, "Pb": 2.60, "Bi": 2.54,
}

# atomic-blender-pdb-xyz uses element full names as ball prefixes
# (e.g. "Hydrogen_ball"). Note "Aluminium"/"Caesium"/"Sulfur" follow the
# spelling in the add-on's ELEMENTS_DEFAULT.
_ELEMENT_NAME_TO_SYMBOL: dict[str, str] = {
    "Hydrogen": "H", "Helium": "He",
    "Lithium": "Li", "Beryllium": "Be", "Boron": "B", "Carbon": "C",
    "Nitrogen": "N", "Oxygen": "O", "Fluorine": "F", "Neon": "Ne",
    "Sodium": "Na", "Magnesium": "Mg", "Aluminium": "Al", "Silicon": "Si",
    "Phosphorus": "P", "Sulfur": "S", "Chlorine": "Cl", "Argon": "Ar",
    "Potassium": "K", "Calcium": "Ca", "Scandium": "Sc", "Titanium": "Ti",
    "Vanadium": "V", "Chromium": "Cr", "Manganese": "Mn", "Iron": "Fe",
    "Cobalt": "Co", "Nickel": "Ni", "Copper": "Cu", "Zinc": "Zn",
    "Gallium": "Ga", "Germanium": "Ge", "Arsenic": "As", "Selenium": "Se",
    "Bromine": "Br", "Krypton": "Kr",
    "Rubidium": "Rb", "Strontium": "Sr", "Yttrium": "Y", "Zirconium": "Zr",
    "Niobium": "Nb", "Molybdenum": "Mo", "Technetium": "Tc", "Ruthenium": "Ru",
    "Rhodium": "Rh", "Palladium": "Pd", "Silver": "Ag", "Cadmium": "Cd",
    "Indium": "In", "Tin": "Sn", "Antimony": "Sb", "Tellurium": "Te",
    "Iodine": "I", "Xenon": "Xe",
    "Caesium": "Cs", "Barium": "Ba",
    "Lanthanum": "La", "Cerium": "Ce", "Praseodymium": "Pr", "Neodymium": "Nd",
    "Promethium": "Pm", "Samarium": "Sm", "Europium": "Eu", "Gadolinium": "Gd",
    "Terbium": "Tb", "Dysprosium": "Dy", "Holmium": "Ho", "Erbium": "Er",
    "Thulium": "Tm", "Ytterbium": "Yb", "Lutetium": "Lu",
    "Hafnium": "Hf", "Tantalum": "Ta", "Tungsten": "W", "Rhenium": "Re",
    "Osmium": "Os", "Iridium": "Ir", "Platinum": "Pt", "Gold": "Au",
    "Mercury": "Hg", "Thallium": "Tl", "Lead": "Pb", "Bismuth": "Bi",
}

DEFAULT_VDW_SCALE = 0.25
```

- [ ] **Step 2: 構文チェック**

Run: `python -c "import ast; ast.parse(open('blender/render.py').read())"`

Expected: 標準出力に何も出ず終了コード 0 (構文エラー無し)。

- [ ] **Step 3: 辞書サイズの確認**

Run:
```bash
python -c "
import ast, sys
src = open('blender/render.py').read()
tree = ast.parse(src)
# Locate the two dict literals and count keys
for node in ast.walk(tree):
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        t = node.targets[0]
        if isinstance(t, ast.Name) and t.id in ('VDW_RADII_ANGSTROM', '_ELEMENT_NAME_TO_SYMBOL'):
            assert isinstance(node.value, ast.Dict), t.id
            print(f'{t.id}: {len(node.value.keys)} entries')
            assert len(node.value.keys) == 83, (t.id, len(node.value.keys))
print('OK')
"
```

Expected:
```
VDW_RADII_ANGSTROM: 83 entries
_ELEMENT_NAME_TO_SYMBOL: 83 entries
OK
```

- [ ] **Step 4: コミットしない**

Task 5 で実装一括コミット。

---

## Task 3: スケール解決とリスケール関数を追加

**Files:**
- Modify: `blender/render.py`

- [ ] **Step 1: `_reset_scene()` の直前に2つの helper 関数を追加**

`blender/render.py` 内、`def _reset_scene()` の **直前** に以下を挿入:

```python
def _resolve_vdw_scale() -> float:
    raw = os.environ.get("REACTX_VDW_SCALE")
    if raw is None:
        return DEFAULT_VDW_SCALE
    try:
        return float(raw)
    except ValueError:
        print(f"[reactx] vdw-rescale: bad REACTX_VDW_SCALE={raw!r}, "
              f"using {DEFAULT_VDW_SCALE}")
        return DEFAULT_VDW_SCALE


def _rescale_atoms_to_vdw() -> None:
    scale = _resolve_vdw_scale()
    suffix = "_ball"
    for obj in bpy.data.objects:
        if not obj.name.endswith(suffix):
            continue
        element_name = obj.name[: -len(suffix)]
        symbol = _ELEMENT_NAME_TO_SYMBOL.get(element_name)
        radius = VDW_RADII_ANGSTROM.get(symbol) if symbol else None
        if radius is None:
            print(f"[reactx] vdw-rescale: skip {obj.name} (no entry)")
            continue
        new_scale = radius * scale
        obj.scale = (new_scale, new_scale, new_scale)
        print(f"[reactx] vdw-rescale: {element_name} scale={new_scale:.6f}")
```

備考: `obj.name` は Blender が衝突回避のため `Hydrogen_ball.001` のように suffix を付けることがあるが、本仕様では各元素タイプ毎に1個だけ作られる代表ボールのみが `_ball` で正確に終わるため、`endswith("_ball")` で十分。アドオン挙動の観察に基づく。

- [ ] **Step 2: 構文チェック**

Run: `python -c "import ast; ast.parse(open('blender/render.py').read())"`

Expected: エラー無し。

- [ ] **Step 3: コミットしない**

Task 5 で実装一括コミット。

---

## Task 4: `main()` にリスケール呼び出しを組み込む

**Files:**
- Modify: `blender/render.py:87-97`

- [ ] **Step 1: `main()` の `_import_trajectory(xyz)` 直後に呼び出しを追加**

現状の `main()`:

```python
def main(argv: list[str]) -> int:
    xyz, out = _parse_args(argv)
    _reset_scene()
    _import_trajectory(xyz)
    _add_three_point_lighting()
    _add_camera_looking_at_origin()
    _set_timeline_to_trajectory(xyz)
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out))
    print(f"Saved: {out}")
    return 0
```

`_import_trajectory(xyz)` の直後に `_rescale_atoms_to_vdw()` を追加:

```python
def main(argv: list[str]) -> int:
    xyz, out = _parse_args(argv)
    _reset_scene()
    _import_trajectory(xyz)
    _rescale_atoms_to_vdw()
    _add_three_point_lighting()
    _add_camera_looking_at_origin()
    _set_timeline_to_trajectory(xyz)
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out))
    print(f"Saved: {out}")
    return 0
```

- [ ] **Step 2: 構文チェック**

Run: `python -c "import ast; ast.parse(open('blender/render.py').read())"`

Expected: エラー無し。

---

## Task 5: スモークテストを実行して合格を確認しコミット

**Files:**
- (実行のみ、`blender/render.py` と `tests/test_blender_smoke.py` の変更内容を一括コミット)

- [ ] **Step 1: vdW スモークテストを実行**

Run: `pytest tests/test_blender_smoke.py::test_blender_vdw_rescale -v -m blender`

Expected: PASS。
- Blender が PATH に無い環境では skip される → その環境では Step 4 のマニュアル検証 (Task 6) で代替確認となるため、**この時点では skip でも OK** (ただし最終的にどこかで実機テストすること)。
- Blender が動く環境では `Hydrogen scale=0.300000` と `Carbon scale=0.442500` が stdout に出力され、両方の assert が通ることを確認。

- [ ] **Step 2: 既存スモークテストも回帰しないことを確認**

Run: `pytest tests/test_blender_smoke.py -v -m blender`

Expected: 2 件 PASS (既存 `test_blender_smoke_produces_blend` + 新規 `test_blender_vdw_rescale`)。

- [ ] **Step 3: 高速ユニットテスト全体を実行 (回帰がないこと確認)**

Run: `pytest`

Expected: 既存テストが全件 PASS。`-m blender` と `-m slow` は除外されるため Blender / UMA 不要。

- [ ] **Step 4: 変更ファイルをステージしてコミット**

Run:
```bash
git add blender/render.py tests/test_blender_smoke.py
git commit -m "$(cat <<'EOF'
feat(blender): rescale atom balls to Alvarez 2013 vdW radii x 0.25

Adds a post-import rescaling pass to blender/render.py that walks
<ElementName>_ball objects produced by the atomic-blender-pdb-xyz
add-on and overrides their scale using a hard-coded Alvarez (2013)
vdW radius table covering Z=1..83 (matching OMol25 / UMA omol task).
The global multiplier defaults to 0.25 (ball-and-stick) and can be
overridden via REACTX_VDW_SCALE.

A new @pytest.mark.blender smoke test runs Blender on a minimal
H+C trajectory and asserts the logged scales for Hydrogen and
Carbon match the Alvarez values.

Spec: docs/superpowers/specs/2026-04-26-vdw-radii-design.md
EOF
)"
```

Expected: 1 件のコミットが作成される。

---

## Task 6: マニュアル E2E 検証 (任意・推奨)

**Files:** (実行のみ)

- [ ] **Step 1: SN2 サンプルでフルパイプラインを走らせる**

Run:
```bash
reactx run examples/sn2.rxn -o out/ --backend uma --render
```

Expected:
- `out/trajectory.xyz`, `out/scene.blend` が生成される
- Blender stdout に `[reactx] vdw-rescale: Hydrogen scale=...`, `Carbon scale=...`, `Fluorine scale=...`, `Chlorine scale=...` の 4 行が含まれる
- F (1.46) > Cl (1.82) なので、Cl の方が F より明確に大きく見える ことが reactant frame で目視確認できる

- [ ] **Step 2: スケール上書きを確認**

Run:
```bash
REACTX_VDW_SCALE=0.4 blender --background --python blender/render.py -- out/trajectory.xyz out/scene_cpk.blend
```

Expected: `Hydrogen scale=0.480000` (= 1.20 × 0.4) が stdout に出力され、`out/scene_cpk.blend` の球がより大きい。

- [ ] **Step 3: 不正な値のフォールバック挙動を確認**

Run:
```bash
REACTX_VDW_SCALE=foo blender --background --python blender/render.py -- out/trajectory.xyz out/scene_bad.blend
```

Expected: stdout に `[reactx] vdw-rescale: bad REACTX_VDW_SCALE='foo', using 0.25` が出て、デフォルト値で処理が続行する。

---

## Self-Review チェックリスト

Plan 全体を読み返した上で以下を確認:

- **Spec カバレッジ**:
  - § 3.2 vdW テーブル → Task 2
  - § 3.3 `_resolve_vdw_scale` → Task 3
  - § 3.4 `_rescale_atoms_to_vdw` → Task 3
  - § 3.5 main 統合 → Task 4
  - § 4 挙動仕様 (各セル) → Task 5 のテスト + Task 6 のマニュアル検証
  - § 5.1 新規テスト → Task 1 + Task 5
  - § 5.2 既存テスト非影響 → Task 5 Step 3
  - § 7 README 更新 → 既に commit `b1fd40d` で完了
- **プレースホルダ**: 無し (vdW テーブル全 83 元素、コード全文、コマンド全部記載済)
- **型整合性**: `VDW_RADII_ANGSTROM` (str→float)、`_ELEMENT_NAME_TO_SYMBOL` (str→str) の型は両関数で整合
- **関数名/シグネチャ**: `_resolve_vdw_scale() -> float`、`_rescale_atoms_to_vdw() -> None` は全タスク間で一貫
