"""Shared pytest fixtures for reactx tests."""
import logging
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from reactx.bond_changes import BondChanges


@pytest.fixture(autouse=True)
def reset_reactx_logger_after_test():
    yield
    logger = logging.getLogger("reactx")
    logger.handlers.clear()
    logger.propagate = True


@pytest.fixture()
def examples_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture()
def sn2_rxn_path(examples_dir: Path) -> Path:
    return examples_dir / "sn2.rxn"


@pytest.fixture()
def menshutkin_rxn_path(examples_dir: Path) -> Path:
    return examples_dir / "menshutkin.rxn"


@pytest.fixture()
def e2_rxn_path(examples_dir: Path) -> Path:
    return examples_dir / "e2.rxn"


@pytest.fixture()
def sn1_dissoc_rxn_path(examples_dir: Path) -> Path:
    return examples_dir / "sn1_dissoc.rxn"


@pytest.fixture()
def sn1_recomb_rxn_path(examples_dir: Path) -> Path:
    return examples_dir / "sn1_recomb.rxn"


@pytest.fixture()
def sn2_atoms_setup():
    """3-fragment-style SN2 setup for placement dispatcher test (CH3Cl + OH-)."""
    mol = Chem.AddHs(Chem.MolFromSmiles("C(Cl).[OH-]"))
    Chem.SanitizeMol(mol)
    frag_indices = Chem.GetMolFrags(mol)
    n = mol.GetNumAtoms()
    positions = np.zeros((n, 3))
    for i in range(n):
        positions[i] = (float(i) * 0.5, 0.0, 0.0)
    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    c_idx = syms.index("C")
    cl_idx = syms.index("Cl")
    o_idx = syms.index("O")
    bc = BondChanges(formed=((c_idx, o_idx),), broken=((c_idx, cl_idx),))
    return mol, frag_indices, positions, bc


@pytest.fixture()
def e2_atoms_setup():
    """E2 setup: CH3CH2Cl + OH-, with explicit β-H atom."""
    smi = "[H][CH2:1][CH2:2][Cl:3].[O:4][H:5]"
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    Chem.SanitizeMol(mol)
    frag_indices = Chem.GetMolFrags(mol)
    n = mol.GetNumAtoms()
    positions = np.zeros((n, 3))
    for i in range(n):
        positions[i] = (float(i) * 0.5, float(i % 2), 0.0)

    map_to_idx = {a.GetAtomMapNum(): a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum()}
    c_alpha = map_to_idx[1]
    c_beta = map_to_idx[2]
    cl = map_to_idx[3]
    o = map_to_idx[4]
    h_anchor = next(
        n.GetIdx() for n in mol.GetAtomWithIdx(c_beta).GetNeighbors()
        if n.GetSymbol() == "H"
    )
    bc = BondChanges(
        formed=((o, h_anchor),),
        broken=((c_alpha, cl), (c_beta, h_anchor)),
    )
    return mol, frag_indices, positions, bc


@pytest.fixture()
def sn1_recomb_atoms_setup():
    """SN1 step 2 setup: tBu+ + Cl- with formed=(C-Cl), broken=()."""
    mol = Chem.AddHs(Chem.MolFromSmiles("[C+](C)(C)C.[Cl-]"))
    Chem.SanitizeMol(mol)
    frag_indices = Chem.GetMolFrags(mol)
    n = mol.GetNumAtoms()
    positions = np.zeros((n, 3))
    # tBu+ を planar 三角形 + 中心 C+ に: 中心 C+ は (0,0,0)、3 methyl C は xy 平面
    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    central = next(
        i for i, a in enumerate(mol.GetAtoms())
        if a.GetSymbol() == "C" and a.GetFormalCharge() == 1
    )
    positions[central] = (0.0, 0.0, 0.0)
    methyl_carbons = [
        nb.GetIdx() for nb in mol.GetAtomWithIdx(central).GetNeighbors()
        if nb.GetSymbol() == "C"
    ]
    for k, m in enumerate(methyl_carbons):
        theta = 2 * np.pi * k / 3
        positions[m] = (np.cos(theta) * 1.5, np.sin(theta) * 1.5, 0.0)
    # 各 methyl C の H 隣接を適当な位置に (test 内で正確には使わない)
    for m in methyl_carbons:
        for h_nb in mol.GetAtomWithIdx(m).GetNeighbors():
            if h_nb.GetSymbol() == "H":
                # methyl C の周りに H を散らす
                positions[h_nb.GetIdx()] = positions[m] + np.array(
                    [0.5 * (h_nb.GetIdx() % 3 - 1), 0.5, 0.5 * ((h_nb.GetIdx() // 3) % 2)]
                )
    # Cl- を遠くに置く (Tier 2 placement で動かされる)
    cl_idx = syms.index("Cl")
    positions[cl_idx] = (10.0, 10.0, 10.0)

    bc = BondChanges(formed=((central, cl_idx),), broken=())
    return mol, frag_indices, positions, bc


@pytest.fixture()
def tmp_rxn_with_toml(tmp_path: Path):
    """Copy `examples/<stem>.rxn` to tmp_path, write a fresh sidecar TOML.

    Usage:
        rxn_path = tmp_rxn_with_toml("sn2", toml_body='''\\
            description = "sn2 fast"
            [endpoint_relax]
            max_steps = 1
            [neb]
            n_images = 3
            max_steps = 1
        ''')

    Returns the temp `.rxn` Path. The .rxn body is unchanged from
    examples/<stem>.rxn; only the sidecar TOML is configurable.
    """
    examples = Path(__file__).resolve().parent.parent / "examples"

    def _make(stem: str, *, toml_body: str) -> Path:
        src_rxn = examples / f"{stem}.rxn"
        dst_rxn = tmp_path / f"{stem}.rxn"
        dst_rxn.write_bytes(src_rxn.read_bytes())
        (tmp_path / f"{stem}.rxn.toml").write_text(toml_body, encoding="utf-8")
        return dst_rxn

    return _make


@pytest.fixture()
def assert_min_nonbonded_distance_ok():
    """Pytest fixture returning a callable that asserts across all `frames`
    no pair of atoms gets closer than `threshold` (Å) — except (a)
    initial-bonded pairs (any pair within 1.1 × covalent_sum at frame 0),
    (b) `formed` pairs, (c) `broken` pairs.

    Used in slow integration tests as a UMA off-manifold guard.
    """
    import numpy as np

    from reactx.covalent_radii import cordero_radii_for_atoms

    def _assert(frames, *, formed, broken, threshold: float = 0.5) -> None:
        if not frames:
            return
        first = frames[0]
        cov = cordero_radii_for_atoms(first)
        n = len(first)
        initial_bonded = set()
        pos0 = first.positions
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(pos0[j] - pos0[i]))
                if d <= 1.1 * (cov[i] + cov[j]):
                    initial_bonded.add((i, j))
        reactive = set()
        for (i, j) in formed:
            reactive.add((min(i, j), max(i, j)))
        for (i, j) in broken:
            reactive.add((min(i, j), max(i, j)))

        excluded = initial_bonded | reactive

        for frame_idx, frame in enumerate(frames):
            pos = frame.positions
            for i in range(n):
                for j in range(i + 1, n):
                    if (i, j) in excluded:
                        continue
                    d = float(np.linalg.norm(pos[j] - pos[i]))
                    assert d >= threshold, (
                        f"frame {frame_idx}: non-bonded pair ({i},{j}) "
                        f"distance {d:.3f} < {threshold} Å "
                        f"(UMA off-manifold?)"
                    )

    return _assert
