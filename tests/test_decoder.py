"""Acceptance tests for M2 (spec section 10): MWPM corrects all errors up to floor((d-1)/2)."""

import itertools

import numpy as np
import pytest

from syndrome_out import LogicalEffect, Pauli, SurfaceCode, decode_mwpm, logical_effect
from syndrome_out.decoder import decode_mwpm_batch


def _corrects(code: SurfaceCode, error: Pauli) -> bool:
    correction = decode_mwpm(code, code.syndrome(error))
    residual = error * correction
    return (
        not code.syndrome(residual).any() and logical_effect(code, residual) is LogicalEffect.NONE
    )


@pytest.mark.parametrize("d", [3, 5])
def test_corrects_all_errors_up_to_half_distance_exhaustively(d: int) -> None:
    code = SurfaceCode(d)
    t = (d - 1) // 2
    for w in range(1, t + 1):
        for qubits in itertools.combinations(range(code.n), w):
            for kinds in itertools.product("XYZ", repeat=w):
                xs = [q for q, k in zip(qubits, kinds) if k in "XY"]
                zs = [q for q, k in zip(qubits, kinds) if k in "ZY"]
                error = Pauli.from_support(code.n, xs=xs, zs=zs)
                assert _corrects(code, error), f"d={d} failed on {error}"


def test_corrects_random_errors_d7() -> None:
    code = SurfaceCode(7)
    t = 3
    rng = np.random.default_rng(7)
    shots = 10_000
    x = np.zeros((shots, code.n), dtype=bool)
    z = np.zeros((shots, code.n), dtype=bool)
    for s in range(shots):
        w = rng.integers(1, t + 1)
        qubits = rng.choice(code.n, size=w, replace=False)
        kinds = rng.integers(0, 3, size=w)  # 0=X, 1=Y, 2=Z
        x[s, qubits[kinds <= 1]] = True
        z[s, qubits[kinds >= 1]] = True
    syndromes = np.zeros((shots, code.n_faces), dtype=bool)
    syndromes[:, code.z_faces] = (x.astype(np.uint8) @ code.h_z.T) & 1
    syndromes[:, code.x_faces] = (z.astype(np.uint8) @ code.h_x.T) & 1
    cx, cz = decode_mwpm_batch(code, syndromes)
    rx, rz = x ^ cx, z ^ cz
    # Residual must be syndrome-free and commute with both logicals.
    assert not (((rx.astype(np.uint8) @ code.h_z.T) & 1).any())
    assert not (((rz.astype(np.uint8) @ code.h_x.T) & 1).any())
    assert not ((rx[:, code.logical_z.z].sum(axis=1) & 1).any())
    assert not ((rz[:, code.logical_x.x].sum(axis=1) & 1).any())


def test_decoder_output_matches_syndrome() -> None:
    code = SurfaceCode(9)
    rng = np.random.default_rng(3)
    for _ in range(100):
        error = Pauli(rng.random(code.n) < 0.1, rng.random(code.n) < 0.1)
        s = code.syndrome(error)
        c = decode_mwpm(code, s)
        assert np.array_equal(code.syndrome(c), s)
        assert c.weight <= error.weight * 2  # X and Z parts each no heavier than the error's


def test_logical_error_as_input_is_undetected() -> None:
    code = SurfaceCode(5)
    s = code.syndrome(code.logical_x)
    assert not s.any()
    assert decode_mwpm(code, s).is_identity()
    assert logical_effect(code, code.logical_x) is LogicalEffect.X
