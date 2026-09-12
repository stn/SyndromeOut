"""Acceptance tests for M2 (spec section 10): the frontier decoder corrects every error up to
floor((d-1)/2), and its per-class minimum weights and likelihoods agree with brute force."""

import itertools
import math

import numpy as np
import pytest

from syndrome_out import LogicalEffect, Pauli, SurfaceCode, decode, logical_effect
from syndrome_out.decoder import _ratio, _sweeps, decode_classes, decode_min_weight_batch


def _corrects(code: SurfaceCode, error: Pauli) -> bool:
    correction = decode(code, code.syndrome(error), 0.1).min_weight
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
    shots = 2_000
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
    cx, cz = decode_min_weight_batch(code, syndromes)
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
        dec = decode(code, s, 0.1)
        for c in (dec.min_weight, dec.ml):
            assert np.array_equal(code.syndrome(c), s)
        # X and Z parts of the min-weight correction are each no heavier than the error's.
        assert dec.min_weight.x.sum() <= error.x.sum()
        assert dec.min_weight.z.sum() <= error.z.sum()


def test_logical_error_as_input_is_undetected() -> None:
    code = SurfaceCode(5)
    s = code.syndrome(code.logical_x)
    assert not s.any()
    dec = decode(code, s, 0.1)
    assert dec.min_weight.is_identity() and dec.ml.is_identity()
    assert logical_effect(code, code.logical_x) is LogicalEffect.X


# -- per-class results against brute force -------------------------------------------------


def _brute(sweep, errors, syndrome_bits: int, r: float) -> dict[int, tuple[int | None, float]]:
    """{class: (min weight, likelihood)} over the given error bitmasks."""
    best: dict[int, list] = {0: [None, 0.0], 1: [None, 0.0]}
    for e in errors:
        s = 0
        cls = 0
        w = 0
        for q in range(sweep.n):
            if (e >> q) & 1:
                s ^= sweep.qmask[q]
                cls ^= sweep.logical[q]
                w += 1
        if s != syndrome_bits:
            continue
        entry = best[cls]
        entry[1] += r**w
        if entry[0] is None or w < entry[0]:
            entry[0] = w
    return {c: (v[0], v[1]) for c, v in best.items()}


def _check_classes(sweep, results, expected) -> None:
    for cls, (w, lk) in expected.items():
        assert results[cls].weight == w
        assert math.isclose(results[cls].likelihood, lk, rel_tol=1e-9, abs_tol=1e-300)
        if w is not None:
            wit = results[cls].witness
            assert bin(wit).count("1") == w
            s = 0
            c = 0
            for q in range(sweep.n):
                if (wit >> q) & 1:
                    s ^= sweep.qmask[q]
                    c ^= sweep.logical[q]
            assert c == cls


def test_classes_match_brute_force_d3_all_syndromes() -> None:
    code = SurfaceCode(3)
    r = _ratio(0.1)
    errors = range(1 << code.n)
    for sweep in _sweeps(code):
        n_faces = len(sweep.face_indices)
        for s_bits in range(1 << n_faces):
            _check_classes(sweep, sweep.run(s_bits, r), _brute(sweep, errors, s_bits, r))


def test_classes_match_coset_enumeration_d5() -> None:
    """Every error with a given syndrome is a particular solution times a kernel element; the
    kernel is spanned by the faces of the other type plus the logical of the error's type."""
    code = SurfaceCode(5)
    r = _ratio(0.15)
    rng = np.random.default_rng(5)
    sweep_x, sweep_z = _sweeps(code)
    for sweep, kernel_faces, logical in (
        (sweep_x, code.x_faces, code.logical_x.x),
        (sweep_z, code.z_faces, code.logical_z.z),
    ):
        gens = [sum(1 << q for q in code.faces[fi].qubits) for fi in kernel_faces]
        gens.append(sum(1 << q for q in np.flatnonzero(logical)))
        for _ in range(8):
            s = code.syndrome(Pauli(rng.random(code.n) < 0.15, rng.random(code.n) < 0.15))
            s_bits = sweep.local_syndrome(s[-1])
            c0, c1 = sweep.run(s_bits, r)
            x0 = c0.witness if c0.weight is not None else c1.witness
            coset = []
            for mask in range(1 << len(gens)):
                e = x0
                for i, g in enumerate(gens):
                    if (mask >> i) & 1:
                        e ^= g
                coset.append(e)
            _check_classes(sweep, (c0, c1), _brute(sweep, coset, s_bits, r))


def test_ml_picks_most_likely_class() -> None:
    code = SurfaceCode(7)
    rng = np.random.default_rng(11)
    seen_disagreement = False
    for _ in range(300):
        s = code.syndrome(Pauli(rng.random(code.n) < 0.12, rng.random(code.n) < 0.12))
        (x0, x1), (z0, z1) = decode_classes(code, s, 0.12)
        dec = decode(code, s, 0.12)
        for (c0, c1), part, logical in (
            ((x0, x1), dec.ml.x, code.logical_z.z),
            ((z0, z1), dec.ml.z, code.logical_x.x),
        ):
            cls = int(np.count_nonzero(part & logical) & 1)
            assert cls == (1 if c1.likelihood > c0.likelihood else 0)
        if dec.ml != dec.min_weight:
            seen_disagreement = True
    assert seen_disagreement, "expected some boards where ML and min-weight disagree"
