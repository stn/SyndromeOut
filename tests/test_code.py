"""Acceptance tests for M1 (spec section 10): code structure and logical operators."""

import itertools

import numpy as np
import pytest

from syndrome_out import LogicalEffect, Pauli, SurfaceCode, logical_effect, sample_depolarizing

DISTANCES = [3, 5, 7, 9]


@pytest.fixture(params=DISTANCES, ids=lambda d: f"d{d}")
def code(request: pytest.FixtureRequest) -> SurfaceCode:
    return SurfaceCode(request.param)


def test_face_counts(code: SurfaceCode) -> None:
    expected = (code.d**2 - 1) // 2
    assert len(code.x_faces) == expected
    assert len(code.z_faces) == expected
    for f in code.faces:
        assert len(f.qubits) == (2 if f.side else 4)


def test_boundary_faces_do_not_touch_same_type_interior(code: SurfaceCode) -> None:
    """Each boundary half-face shares its two qubits with exactly one interior face of the other type."""
    interior = [f for f in code.faces if f.side is None]
    for f in code.faces:
        if f.side is None:
            continue
        sharing = [g for g in interior if set(f.qubits) <= set(g.qubits)]
        assert len(sharing) == 1
        assert sharing[0].kind != f.kind


def test_stabilizers_commute(code: SurfaceCode) -> None:
    stabs = [code.stabilizer(i) for i in range(code.n_faces)]
    for a, b in itertools.combinations(stabs, 2):
        assert a.commutes_with(b)


def test_logical_operators(code: SurfaceCode) -> None:
    lx, lz = code.logical_x, code.logical_z
    assert lx.weight == code.d
    assert lz.weight == code.d
    assert not lx.commutes_with(lz)
    for i in range(code.n_faces):
        s = code.stabilizer(i)
        assert lx.commutes_with(s)
        assert lz.commutes_with(s)
    assert not code.syndrome(lx).any()
    assert not code.syndrome(lz).any()


def test_logical_effect_of_logicals(code: SurfaceCode) -> None:
    assert logical_effect(code, code.logical_x) is LogicalEffect.X
    assert logical_effect(code, code.logical_z) is LogicalEffect.Z
    assert logical_effect(code, code.logical_x * code.logical_z) is LogicalEffect.Y
    assert logical_effect(code, Pauli.identity(code.n)) is LogicalEffect.NONE


def test_stabilizer_products_are_trivial(code: SurfaceCode) -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        mask = rng.random(code.n_faces) < 0.5
        prod = Pauli.identity(code.n)
        for i in np.flatnonzero(mask):
            prod = prod * code.stabilizer(int(i))
        assert not code.syndrome(prod).any()
        assert logical_effect(code, prod) is LogicalEffect.NONE


def test_single_errors_light_one_or_two_faces(code: SurfaceCode) -> None:
    for q in range(code.n):
        sx = code.syndrome(Pauli.from_support(code.n, xs=[q]))[0]
        sz = code.syndrome(Pauli.from_support(code.n, zs=[q]))[0]
        # X errors light Z-type faces only, Z errors light X-type faces only.
        assert not sx[code.x_faces].any()
        assert not sz[code.z_faces].any()
        assert 1 <= sx.sum() <= 2
        assert 1 <= sz.sum() <= 2
        # Bulk qubits (not on the outer ring) light exactly two faces of each type.
        r, c = code.qubit_pos(q)
        if 0 < r < code.d - 1 and 0 < c < code.d - 1:
            assert sx.sum() == 2 and sz.sum() == 2


def test_syndrome_shape_has_round_axis(code: SurfaceCode) -> None:
    s = code.syndrome(Pauli.identity(code.n))
    assert s.shape == (1, code.n_faces)
    assert s.dtype == bool


def test_seed_reproducibility() -> None:
    code = SurfaceCode(7)
    a = sample_depolarizing(code, 0.1, seed=12345)
    b = sample_depolarizing(code, 0.1, seed=12345)
    c = sample_depolarizing(code, 0.1, seed=12346)
    assert a == b
    assert a != c


def test_depolarizing_marginals() -> None:
    from syndrome_out.noise import sample_depolarizing_batch

    code = SurfaceCode(5)
    x, z = sample_depolarizing_batch(code, 0.15, seed=1, shots=20000)
    total = x.size
    only_x = np.count_nonzero(x & ~z) / total
    only_z = np.count_nonzero(~x & z) / total
    both = np.count_nonzero(x & z) / total
    for rate in (only_x, only_z, both):
        assert abs(rate - 0.05) < 0.005


def test_pauli_algebra() -> None:
    a = Pauli.from_support(4, xs=[0, 1], zs=[1])
    b = Pauli.from_support(4, zs=[0], xs=[3])
    assert str(a) == "XYII"
    assert str(a * b) == "YYIX"
    assert a.symplectic(b) == 1
    assert (a * a).is_identity()
    assert a.weight == 2


def test_stabilizer_faces_decomposition(code: SurfaceCode) -> None:
    rng = np.random.default_rng(1)
    for _ in range(50):
        chosen = np.flatnonzero(rng.random(code.n_faces) < 0.4)
        prod = Pauli.identity(code.n)
        for i in chosen:
            prod = prod * code.stabilizer(int(i))
        faces = code.stabilizer_faces(prod)
        assert faces is not None
        assert sorted(int(f) for f in faces) == sorted(int(i) for i in chosen)
    assert code.stabilizer_faces(code.logical_x) is None
    assert code.stabilizer_faces(code.logical_z * code.stabilizer(0)) is None
    assert len(code.stabilizer_faces(Pauli.identity(code.n))) == 0
