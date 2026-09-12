"""Frontier-DP decoder: exact minimum-weight correction and per-class likelihood, pure Python.

X and Z errors are decoded independently (spec 5). For one error type the data qubits are
swept in row-major order; the DP state is the parity of every stabilizer face that has been
opened but not yet closed, plus one bit for the logical class of the partial assignment. A
face closes at its last qubit, where its parity is checked against the syndrome. This is the
frontier method behind the ZDD experiments, run once per syndrome instead of being reduced
into a diagram, with back-pointers so the minimum-weight witness can be read off.

The minimum-weight witness is what MWPM computes for the surface code; the per-class
likelihood sum (over every error consistent with the syndrome) additionally gives the
degeneracy-aware maximum-likelihood class, which MWPM cannot see.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

from syndrome_out.code import SurfaceCode, Syndrome
from syndrome_out.pauli import Pauli


@dataclass(frozen=True, slots=True)
class ClassResult:
    """Errors of one logical class consistent with a syndrome. ``weight`` is None if empty."""

    weight: int | None
    witness: int  # minimum-weight member as a qubit bitmask
    likelihood: float  # sum of r^|E| over the class, r = q / (1 - q)


@dataclass(frozen=True, slots=True)
class Decoding:
    min_weight: Pauli  # what an MWPM bot plays
    ml: Pauli  # lightest member of the most likely class


class _Sweep:
    """Row-major frontier sweep for one face type (X faces see Z errors and vice versa)."""

    def __init__(self, code: SurfaceCode, face_indices: np.ndarray, logical: np.ndarray) -> None:
        n = code.n
        self.n = n
        self.face_indices = face_indices
        self.qmask = [0] * n
        self.closing_mask = [0] * n
        self.logical = [bool(v) for v in logical]
        for j, fi in enumerate(face_indices):
            qubits = code.faces[fi].qubits
            for q in qubits:
                self.qmask[q] |= 1 << j
            self.closing_mask[max(qubits)] |= 1 << j

    def run(self, syndrome: int, r: float) -> tuple[ClassResult, ClassResult]:
        # State key = parity of open faces << 1 | logical class. Bit 0 is tried first and a
        # later equal weight never displaces it, so tie-breaking is deterministic.
        states: dict[int, list] = {0: [0, 1.0]}
        history: list[dict[int, tuple[int, int]]] = []
        for q in range(self.n):
            qmask, cm, lg = self.qmask[q], self.closing_mask[q], self.logical[q]
            want = syndrome & cm
            new: dict[int, list] = {}
            back: dict[int, tuple[int, int]] = {}
            for key, (w, lk) in states.items():
                for bit in (0, 1):
                    par = key >> 1
                    cls = key & 1
                    if bit:
                        par ^= qmask
                        cls ^= lg
                        w2, lk2 = w + 1, lk * r
                    else:
                        w2, lk2 = w, lk
                    if par & cm != want:
                        continue
                    k2 = ((par & ~cm) << 1) | cls
                    entry = new.get(k2)
                    if entry is None:
                        new[k2] = [w2, lk2]
                        back[k2] = (key, bit)
                    else:
                        entry[1] += lk2
                        if w2 < entry[0]:
                            entry[0] = w2
                            back[k2] = (key, bit)
            states = new
            history.append(back)

        def result(cls: int) -> ClassResult:
            entry = states.get(cls)
            if entry is None:
                return ClassResult(None, 0, 0.0)
            witness = 0
            key = cls
            for q in range(self.n - 1, -1, -1):
                key, bit = history[q][key]
                if bit:
                    witness |= 1 << q
            return ClassResult(entry[0], witness, entry[1])

        return result(0), result(1)

    def local_syndrome(self, s: np.ndarray) -> int:
        bits = 0
        for j, fi in enumerate(self.face_indices):
            if s[fi]:
                bits |= 1 << j
        return bits

    def to_array(self, witness: int) -> np.ndarray:
        return np.array([(witness >> q) & 1 for q in range(self.n)], dtype=bool)


@cache
def _sweeps(code: SurfaceCode) -> tuple[_Sweep, _Sweep]:
    """(sweep for X errors on Z faces, sweep for Z errors on X faces).

    The class bit of an X error is its overlap parity with logical Z (and vice versa),
    matching ``logical_effect``.
    """
    return (
        _Sweep(code, code.z_faces, code.logical_z.z),
        _Sweep(code, code.x_faces, code.logical_x.x),
    )


def _ratio(p: float) -> float:
    """r = q / (1 - q) where q = 2p/3 is the chance a depolarized qubit has an X (or Z) part."""
    q = 2.0 * p / 3.0
    return q / (1.0 - q)


def _pick(c0: ClassResult, c1: ClassResult) -> tuple[int, int]:
    """(min-weight witness, ML witness). Ties go to the min-weight class, then class 0."""
    w0 = c0.weight if c0.weight is not None else float("inf")
    w1 = c1.weight if c1.weight is not None else float("inf")
    mw = c0 if w0 <= w1 else c1
    if c0.likelihood == c1.likelihood:
        ml = mw
    else:
        ml = c0 if c0.likelihood > c1.likelihood else c1
    return mw.witness, ml.witness


def decode_classes(
    code: SurfaceCode, syndrome: Syndrome, p: float
) -> tuple[tuple[ClassResult, ClassResult], tuple[ClassResult, ClassResult]]:
    """Per-class results for (X errors, Z errors) from the last syndrome round."""
    s = np.asarray(syndrome, dtype=bool)[-1]
    r = _ratio(p)
    sx, sz = _sweeps(code)
    return sx.run(sx.local_syndrome(s), r), sz.run(sz.local_syndrome(s), r)


def decode(code: SurfaceCode, syndrome: Syndrome, p: float) -> Decoding:
    """Minimum-weight and maximum-likelihood corrections for depolarizing noise of rate ``p``."""
    (x0, x1), (z0, z1) = decode_classes(code, syndrome, p)
    x_mw, x_ml = _pick(x0, x1)
    z_mw, z_ml = _pick(z0, z1)
    sx, sz = _sweeps(code)
    return Decoding(
        Pauli(sx.to_array(x_mw), sz.to_array(z_mw)),
        Pauli(sx.to_array(x_ml), sz.to_array(z_ml)),
    )


def decode_min_weight_batch(
    code: SurfaceCode, syndromes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Batch variant for simulations: ``syndromes`` is (shots, n_faces); returns (x, z)."""
    s = np.asarray(syndromes, dtype=bool)
    sx, sz = _sweeps(code)
    x = np.zeros((s.shape[0], code.n), dtype=bool)
    z = np.zeros((s.shape[0], code.n), dtype=bool)
    for i in range(s.shape[0]):
        x_mw, _ = _pick(*sx.run(sx.local_syndrome(s[i]), 0.0))
        z_mw, _ = _pick(*sz.run(sz.local_syndrome(s[i]), 0.0))
        x[i] = sx.to_array(x_mw)
        z[i] = sz.to_array(z_mw)
    return x, z
