"""Acceptance tests for M2 (spec section 10): the frontier decoder corrects every error up to
floor((d-1)/2), and its per-class minimum weights and likelihoods agree with brute force."""

import itertools
import math

import numpy as np
import pytest

from syndrome_out import LogicalEffect, Pauli, SurfaceCode, decode, logical_effect
from syndrome_out.decoder import (
    _joint_ratio,
    _joint_sweep,
    _ratio,
    _sweeps,
    decode_joint_classes,
    decode_min_weight_batch,
)


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


# -- joint (depolarizing) classes against brute force -------------------------------------


def _joint_class(code: SurfaceCode, p: Pauli) -> int:
    return int(np.count_nonzero(p.x & code.logical_z.z) & 1) | (
        int(np.count_nonzero(p.z & code.logical_x.x) & 1) << 1
    )


def _check_joint_classes(code: SurfaceCode, s, results, expected: dict[int, tuple]) -> None:
    """``expected`` maps class -> (min support weight or None, likelihood)."""
    sweep = _joint_sweep(code)
    for cls, (w, lk) in expected.items():
        assert results[cls].weight == w
        assert math.isclose(results[cls].likelihood, lk, rel_tol=1e-9, abs_tol=1e-300)
        if w is not None:
            witness = sweep.to_pauli(results[cls].x_witness, results[cls].z_witness)
            assert witness.weight == w
            assert np.array_equal(code.syndrome(witness), s)
            assert _joint_class(code, witness) == cls


def test_joint_classes_match_brute_force_d3_all_syndromes() -> None:
    """Every one of the 4^9 Paulis on the d=3 code, bucketed by syndrome and class."""
    code = SurfaceCode(3)
    p = 0.1
    r = _joint_ratio(p)
    sweep = _joint_sweep(code)
    n = code.n
    by_syndrome: dict[int, dict[int, list]] = {}
    for e in range(4**n):
        bits = 0
        cls = 0
        w = 0
        for q in range(n):
            op = (e >> (2 * q)) & 3  # x + 2z, as in the sweep
            if op:
                mask, flip = sweep.ops[q][op]
                bits ^= mask
                cls ^= flip
                w += 1
        classes = by_syndrome.setdefault(bits, {c: [None, 0.0] for c in range(4)})
        entry = classes[cls]
        entry[1] += r**w
        if entry[0] is None or w < entry[0]:
            entry[0] = w
    assert len(by_syndrome) == 1 << code.n_faces
    for bits, classes in by_syndrome.items():
        s = np.zeros((1, code.n_faces), dtype=bool)
        for j, fi in enumerate(sweep.face_indices):
            s[0, fi] = bool((bits >> j) & 1)
        results = decode_joint_classes(code, s, p)
        _check_joint_classes(code, s, results, {c: (v[0], v[1]) for c, v in classes.items()})


def test_joint_classes_match_coset_enumeration_d5() -> None:
    """The errors with the syndrome of E are E times a logical times a stabilizer; enumerate the
    2^24 stabilizers with numpy and bucket support weights per class."""
    code = SurfaceCode(5)
    p = 0.15
    r = _joint_ratio(p)
    rng = np.random.default_rng(5)
    n = code.n

    def group(face_indices) -> np.ndarray:
        members = np.zeros(1, dtype=np.uint64)
        for fi in face_indices:
            g = np.uint64(sum(1 << q for q in code.faces[fi].qubits))
            members = np.concatenate([members, members ^ g])
        return members

    def mask(arr) -> np.uint64:
        return np.uint64(sum(1 << int(q) for q in np.flatnonzero(arr)))

    x_stabs = group(code.x_faces)  # X-type faces multiply the X part
    z_stabs = group(code.z_faces)
    logical_x, logical_z = mask(code.logical_x.x), mask(code.logical_z.z)
    zero = np.uint64(0)
    for _ in range(4):
        error = Pauli(rng.random(n) < p, rng.random(n) < p)
        s = code.syndrome(error)
        ex, ez = mask(error.x), mask(error.z)
        expected = {}
        for flip, lx, lz in (
            (0, zero, zero),
            (1, logical_x, zero),
            (2, zero, logical_z),
            (3, logical_x, logical_z),
        ):
            xs = x_stabs ^ ex ^ lx
            hist = np.zeros(n + 1, dtype=np.int64)
            for zc in z_stabs ^ ez ^ lz:
                hist += np.bincount(np.bitwise_count(xs | zc), minlength=n + 1)
            weights = np.flatnonzero(hist)
            lk = float(np.sum(hist[weights] * r ** weights.astype(float)))
            expected[_joint_class(code, error) ^ flip] = (int(weights.min()), lk)
        _check_joint_classes(code, s, decode_joint_classes(code, s, p), expected)


def test_ml_picks_most_likely_joint_class() -> None:
    code = SurfaceCode(5)
    rng = np.random.default_rng(11)
    seen_disagreement = False
    for _ in range(300):
        s = code.syndrome(Pauli(rng.random(code.n) < 0.12, rng.random(code.n) < 0.12))
        classes = decode_joint_classes(code, s, 0.12)
        dec = decode(code, s, 0.12)
        best = max(range(4), key=lambda c: classes[c].likelihood)
        assert _joint_class(code, dec.ml) == best
        assert dec.ml.weight == classes[best].weight
        if _joint_class(code, dec.min_weight) != best:
            seen_disagreement = True
    assert seen_disagreement, "expected some boards where ML and min-weight disagree"


def test_ml_uses_y_correlations() -> None:
    """Board 551A6A: the true error has two Ys. Counting a Y as one X plus one Z (independent
    X/Z decoding) makes the class a logical Z away look likelier; the joint sweep does not."""
    code = SurfaceCode(5)
    error = Pauli.from_support(code.n, xs=[14, 18], zs=[5, 14, 18])
    s = code.syndrome(error)
    dec = decode(code, s, 0.10)
    assert logical_effect(code, error * dec.ml) is LogicalEffect.NONE
    assert logical_effect(code, error * dec.min_weight) is LogicalEffect.Z
    classes = decode_joint_classes(code, s, 0.10)
    true_cls = _joint_class(code, error)
    assert classes[true_cls].weight == 3
    assert classes[true_cls].likelihood > 20 * classes[true_cls ^ 2].likelihood
