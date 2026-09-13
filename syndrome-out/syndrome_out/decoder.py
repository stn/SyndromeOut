"""Frontier-DP decoder: exact minimum-weight correction and per-class likelihood, pure Python.

The data qubits are swept in row-major order; the DP state is the parity of every stabilizer
face that has been opened but not yet closed, plus the logical class of the partial
assignment. A face closes at its last qubit, where its parity is checked against the
syndrome. Back-pointers let the minimum-weight witness be read off after the sweep.

Two sweeps are used (spec 5). ``_Sweep`` handles one error type at a time, so X and Z errors
are decoded independently; its minimum-weight witness is what MWPM computes for the surface
code. ``_JointSweep`` tries I/X/Z/Y on every qubit and tracks both face types and both class
bits at once; its per-class likelihood sum over every error consistent with the syndrome is
the true depolarizing posterior (a Y costs one factor of p/3, not two), which gives the
degeneracy-aware maximum-likelihood class that neither MWPM nor the independent sweeps see.
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
class JointClassResult:
    """Errors of one of the four logical classes consistent with a syndrome.

    ``weight`` is the minimum support weight (a Y counts once), None if the class is empty.
    ``likelihood`` is the sum of r^|E| over the class with r = (p/3) / (1 - p), i.e. the class
    probability up to the common factor (1 - p)^n.
    """

    weight: int | None
    x_witness: int  # X part of a minimum-weight member as a qubit bitmask
    z_witness: int  # Z part of the same member
    likelihood: float


@dataclass(frozen=True, slots=True)
class Decoding:
    min_weight: Pauli  # what an MWPM bot plays
    ml: Pauli  # lightest member of the most likely class under depolarizing noise


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


class _JointSweep:
    """Row-major frontier sweep over both face types with I/X/Z/Y tried on every qubit.

    Face bits: Z faces first (they see X parts), then X faces (they see Z parts). Class bits:
    bit 0 is the X part's overlap parity with logical Z, bit 1 the Z part's with logical X,
    matching ``logical_effect``. The frontier is about twice as wide as one ``_Sweep``'s, so
    at d=9 this holds a few thousand states and takes a few hundred ms.
    """

    def __init__(self, code: SurfaceCode) -> None:
        n = code.n
        self.n = n
        self.face_indices = [*code.z_faces, *code.x_faces]
        n_z = len(code.z_faces)
        x_mask = [0] * n
        z_mask = [0] * n
        self.closing_mask = [0] * n
        for j, fi in enumerate(self.face_indices):
            qubits = code.faces[fi].qubits
            mask = x_mask if j < n_z else z_mask
            for q in qubits:
                mask[q] |= 1 << j
            self.closing_mask[max(qubits)] |= 1 << j
        # Per qubit, the (face mask, class flip) of I, X, Z and Y, indexed by op = x + 2z.
        self.ops: list[tuple[tuple[int, int], ...]] = []
        for q in range(n):
            cx = int(code.logical_z.z[q])
            cz = int(code.logical_x.x[q]) << 1
            self.ops.append(
                ((0, 0), (x_mask[q], cx), (z_mask[q], cz), (x_mask[q] | z_mask[q], cx | cz))
            )

    def run(self, syndrome: int, r: float) -> tuple[JointClassResult, ...]:
        # State key = parity of open faces << 2 | class. Ops are tried in order I, X, Z, Y and
        # a later equal weight never displaces an earlier one, so tie-breaking is deterministic.
        states: dict[int, list] = {0: [0, 1.0]}
        history: list[dict[int, tuple[int, int]]] = []
        for q in range(self.n):
            cm = self.closing_mask[q]
            want = syndrome & cm
            ops = self.ops[q]
            new: dict[int, list] = {}
            back: dict[int, tuple[int, int]] = {}
            for key, (w, lk) in states.items():
                par = key >> 2
                cls = key & 3
                for op in range(4):
                    mask, flip = ops[op]
                    p2 = par ^ mask
                    if p2 & cm != want:
                        continue
                    k2 = ((p2 & ~cm) << 2) | (cls ^ flip)
                    if op:
                        w2, lk2 = w + 1, lk * r
                    else:
                        w2, lk2 = w, lk
                    entry = new.get(k2)
                    if entry is None:
                        new[k2] = [w2, lk2]
                        back[k2] = (key, op)
                    else:
                        entry[1] += lk2
                        if w2 < entry[0]:
                            entry[0] = w2
                            back[k2] = (key, op)
            states = new
            history.append(back)

        def result(cls: int) -> JointClassResult:
            entry = states.get(cls)
            if entry is None:
                return JointClassResult(None, 0, 0, 0.0)
            x_witness = z_witness = 0
            key = cls
            for q in range(self.n - 1, -1, -1):
                key, op = history[q][key]
                if op & 1:
                    x_witness |= 1 << q
                if op & 2:
                    z_witness |= 1 << q
            return JointClassResult(entry[0], x_witness, z_witness, entry[1])

        return tuple(result(cls) for cls in range(4))

    def local_syndrome(self, s: np.ndarray) -> int:
        bits = 0
        for j, fi in enumerate(self.face_indices):
            if s[fi]:
                bits |= 1 << j
        return bits

    def to_pauli(self, x_witness: int, z_witness: int) -> Pauli:
        return Pauli(
            [(x_witness >> q) & 1 for q in range(self.n)],
            [(z_witness >> q) & 1 for q in range(self.n)],
        )


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


@cache
def _joint_sweep(code: SurfaceCode) -> _JointSweep:
    return _JointSweep(code)


def _ratio(p: float) -> float:
    """r = q / (1 - q) where q = 2p/3 is the chance a depolarized qubit has an X (or Z) part."""
    q = 2.0 * p / 3.0
    return q / (1.0 - q)


def _joint_ratio(p: float) -> float:
    """r = (p/3) / (1 - p): the likelihood ratio of any single-qubit Pauli error to no error."""
    return (p / 3.0) / (1.0 - p)


def _pick(c0: ClassResult, c1: ClassResult) -> int:
    """Minimum-weight witness over both classes; ties go to class 0."""
    w0 = c0.weight if c0.weight is not None else float("inf")
    w1 = c1.weight if c1.weight is not None else float("inf")
    return c0.witness if w0 <= w1 else c1.witness


def _pick_ml(classes: tuple[JointClassResult, ...]) -> JointClassResult:
    """The most likely class; ties go to the lighter class, then the lower class index."""
    return max(
        classes,
        key=lambda c: (c.likelihood, -(c.weight if c.weight is not None else float("inf"))),
    )


def decode_classes(
    code: SurfaceCode, syndrome: Syndrome, p: float
) -> tuple[tuple[ClassResult, ClassResult], tuple[ClassResult, ClassResult]]:
    """Per-class results for (X errors, Z errors) from the last syndrome round, decoded
    independently; the likelihoods are the X/Z marginals, not the depolarizing posterior."""
    s = np.asarray(syndrome, dtype=bool)[-1]
    r = _ratio(p)
    sx, sz = _sweeps(code)
    return sx.run(sx.local_syndrome(s), r), sz.run(sz.local_syndrome(s), r)


def decode_joint_classes(
    code: SurfaceCode, syndrome: Syndrome, p: float
) -> tuple[JointClassResult, ...]:
    """Results for the four logical classes (indexed by X-class bit + 2 * Z-class bit) under
    depolarizing noise of rate ``p``, from the last syndrome round."""
    s = np.asarray(syndrome, dtype=bool)[-1]
    sweep = _joint_sweep(code)
    return sweep.run(sweep.local_syndrome(s), _joint_ratio(p))


def decode(code: SurfaceCode, syndrome: Syndrome, p: float) -> Decoding:
    """Minimum-weight and maximum-likelihood corrections for depolarizing noise of rate ``p``."""
    (x0, x1), (z0, z1) = decode_classes(code, syndrome, p)
    sx, sz = _sweeps(code)
    ml = _pick_ml(decode_joint_classes(code, syndrome, p))
    return Decoding(
        Pauli(sx.to_array(_pick(x0, x1)), sz.to_array(_pick(z0, z1))),
        _joint_sweep(code).to_pauli(ml.x_witness, ml.z_witness),
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
        x_mw = _pick(*sx.run(sx.local_syndrome(s[i]), 0.0))
        z_mw = _pick(*sz.run(sz.local_syndrome(s[i]), 0.0))
        x[i] = sx.to_array(x_mw)
        z[i] = sz.to_array(z_mw)
    return x, z
