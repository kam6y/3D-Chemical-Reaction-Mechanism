# Blender アニメーションの原子半径を van der Waals 半径比に揃える

- 作成日: 2026-04-26
- 関連: `docs/superpowers/specs/2026-04-20-reactx-phase-0-design.md`
- 対象範囲: Phase 0 (`blender/render.py`)

## 1. 背景と目的

`reactx run --render` で生成される `out/scene.blend` は、`atomic-blender-pdb-xyz` アドオンに同梱の元素データから半径を引いている。アドオンの既定値は **共有結合半径 (covalent radius)** に相当し、たとえば SN2 反応 (CH₃Cl + F⁻ → CH₃F + Cl⁻) では H=0.32, C=0.77, F=0.72, Cl=0.99 (Å) という比率で球が描画される。

化学的な視認性を高めるため、球サイズの **相対比** を実際の vdW 半径比へ揃える。これにより、たとえば F⁻ や Cl⁻ といった求核種・脱離基が対応する周辺原子と比べて適切な大きさで表示される。

UMA (`uma-m-1p1`) の分子タスク (`omol`) は OMol25 データセットで訓練されており、訓練元素は **Z=1 (H) から Z=83 (Bi) までの連続した 83 元素** (アクチノイドおよび Po, At, Rn, Fr, Ra は対象外)。本仕様の vdW テーブルもこれと同じ範囲 Z=1〜83 を一対一でカバーする。

OMol25 が含む 83 元素 (出典: ColabFit OMol25 Family ページ; Z=1〜83 と一致することを確認済):
```
Ag Al Ar As Au B Ba Be Bi Br C Ca Cd Ce Cl Co Cr Cs Cu Dy Er Eu F Fe Ga
Gd Ge H He Hf Hg Ho I In Ir K Kr La Li Lu Mg Mn Mo N Na Nb Nd Ne Ni O
Os P Pb Pd Pm Pr Pt Rb Re Rh Ru S Sb Sc Se Si Sm Sn Sr Ta Tb Tc Te Ti
Tl Tm V W Xe Y Yb Zn Zr
```

## 2. 設計上の判断

| 項目 | 採用案 | 検討した代替 | 採用理由 |
|---|---|---|---|
| 絶対サイズ | vdW 半径 × 0.25 (ball-and-stick) | C を現状 0.77 Å に固定 / vdW そのまま (CPK) | 結合棒が見え続けて Walden 反転が視認しやすく、現状の見た目とほぼ同スケール |
| データソース | Alvarez (2013) *Dalton Trans.* 42, 8617 (Z=1-96) | Bondi (1964) + Mantina (2009) 拡張 / RDKit `GetRvdw` | 単一論文で UMA 対応上限 Z=83 まで一貫した方法論。継ぎ接ぎ不要 |
| 差し替え方式 | Import 後の post-processing | アドオンの `datafile` 引数 / `ELEMENTS` の monkey-patch | コード量最少、アドオンの内部 API に依存しない、未知元素を素直にスキップできる |
| 設定方式 | 環境変数 `REACTX_VDW_SCALE` | CLI フラグ / モジュール分割 | Phase 0 では Blender 側だけで完結させ、CLI 配管に手を入れない |

## 3. 実装

### 3.1 配置

- 変更対象: `blender/render.py` のみ
- 新規依存: なし (Blender 同梱の Python 標準ライブラリのみ)

### 3.2 モジュール定数

```python
# Alvarez (2013) Dalton Trans. 42, 8617. Values in Å, Z=1..83 (OMol25 coverage).
VDW_RADII_ANGSTROM: dict[str, float] = {
    "H": 1.20, "He": 1.43, "Li": 2.12, ... "Bi": 2.07,
}
DEFAULT_VDW_SCALE = 0.25
```

- キーは元素記号 (大文字始まり)
- Z=1〜83 を **欠落なく** 埋める。OMol25 = Z=1〜83 の連続レンジと一致するため、辞書サイズもちょうど 83 エントリ
- Alvarez 2013 Table 9 (および Z=96 までの集計表) は対象元素全てに値を割り当てているので欠損補完は不要
- アドオン側のオブジェクト名は元素フルネーム (`"Hydrogen"`, `"Carbon"`, ...) なので、フルネーム → 記号の逆引き辞書 `_ELEMENT_NAME_TO_SYMBOL` を併設する (こちらも 83 エントリ)

### 3.3 スケール解決

```python
def _resolve_vdw_scale() -> float:
    raw = os.environ.get("REACTX_VDW_SCALE")
    if raw is None:
        return DEFAULT_VDW_SCALE
    try:
        return float(raw)
    except ValueError:
        print(f"[reactx] vdw-rescale: bad REACTX_VDW_SCALE={raw!r}, using {DEFAULT_VDW_SCALE}")
        return DEFAULT_VDW_SCALE
```

### 3.4 リスケール関数

```python
def _rescale_atoms_to_vdw() -> None:
    scale = _resolve_vdw_scale()
    for obj in bpy.data.objects:
        if not obj.name.endswith("_ball"):
            continue
        element_name = obj.name[: -len("_ball")]
        symbol = _ELEMENT_NAME_TO_SYMBOL.get(element_name)
        radius = VDW_RADII_ANGSTROM.get(symbol) if symbol else None
        if radius is None:
            print(f"[reactx] vdw-rescale: skip {obj.name} (no entry)")
            continue
        new_scale = radius * scale
        obj.scale = (new_scale, new_scale, new_scale)
        print(f"[reactx] vdw-rescale: {element_name} scale={new_scale:.6f}")
```

### 3.5 `main()` への組み込み

```python
def main(argv: list[str]) -> int:
    xyz, out = _parse_args(argv)
    _reset_scene()
    _import_trajectory(xyz)
    _rescale_atoms_to_vdw()       # ← 追加
    _add_three_point_lighting()
    _add_camera_looking_at_origin()
    _set_timeline_to_trajectory(xyz)
    ...
```

`_rescale_atoms_to_vdw()` の呼び出し位置は `_import_trajectory()` 後でカメラ/ライト追加前なら任意。シーン物質の追加と独立しているため副作用順序の制約はない。

## 4. 挙動仕様

| ケース | 期待挙動 |
|---|---|
| Z=1〜83 で `VDW_RADII_ANGSTROM` にエントリあり (OMol25 訓練対応) | `obj.scale` を `radius * scale` で完全上書き |
| アドオン由来のオブジェクト名が `Vacancy_cube` (vacancy) | suffix が `_cube` なのでマッチせずスキップ |
| アドオン由来の `Default_ball` / `Stick_ball` | 元素記号への逆引きで失敗 → スキップ + ログ |
| Z>83 (Po, At, Rn, Fr, Ra および全アクチノイド) | 逆引きで失敗 → スキップ + ログ。アドオン既定の半径がそのまま使われる。UMA 訓練範囲外なのでそもそも入力 XYZ に出現しない想定 |
| `REACTX_VDW_SCALE` 未設定 | `DEFAULT_VDW_SCALE = 0.25` を使用 |
| `REACTX_VDW_SCALE=0.4` | `0.4` を使用 |
| `REACTX_VDW_SCALE=foo` | デフォルトにフォールバック + 警告ログ |

## 5. テスト

### 5.1 新規追加: `tests/test_blender_smoke.py::test_blender_vdw_rescale`

- マーカー: `@pytest.mark.blender` (既存の smoke と同じ)
- 入力 XYZ: H と C を含む 1 フレーム最小構成 (`H 0 0 0\nC 0 0 1.1`)
- `subprocess.run(..., capture_output=True, text=True)` で Blender を実行
- stdout を行単位で走査し、`r"\[reactx\] vdw-rescale: (\S+) scale=(\S+)"` で `(element_name, scale_str)` を抽出
- アサーション (Alvarez 2013 値: H=1.20, C=1.77):
  - `Hydrogen` の scale が `1.20 * 0.25 = 0.30` に対し `abs(diff) < 1e-3`
  - `Carbon` の scale が `1.77 * 0.25 = 0.4425` に対し `abs(diff) < 1e-3`
  - これにより「比率が vdW 由来」と「グローバルスケールが 0.25」の両方を担保

### 5.2 影響を受けない既存テスト

- `test_neb_sn2.py`, `test_calculators.py`, `test_embed3d.py`, `test_align.py`, `test_rxn_parser.py`, `test_cli.py`, 既存 `test_blender_smoke_produces_blend` — いずれも本変更で挙動が変わらないことを確認

## 6. ロールアウト & 切り戻し

- 切り戻し: `blender/render.py` の `_rescale_atoms_to_vdw()` 呼び出し1行を削除すれば元のアドオン既定半径に戻る
- データ修正: vdW テーブルは単一辞書なので、Alvarez 表の誤読があれば該当 key の値を直す PR 1 件で対応可能

## 7. ドキュメント更新

- `README.md` の「Phase 0 の既知の制約」直前に短い節を追加し、以下を記載:
  - 球半径は Alvarez 2013 の vdW 半径 × 0.25 (`REACTX_VDW_SCALE` で上書き可能)
  - 対応元素は OMol25 訓練対応の **Z=1〜83 (H〜Bi)**。Po/At/Rn/Fr/Ra および全アクチノイドは UMA 訓練外
