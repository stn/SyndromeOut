"""Rotated surface code: faces, check matrices, logical operators."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np

from syndrome_out.pauli import Pauli

FaceKind = Literal["X", "Z"]
Side = Literal["top", "bottom", "left", "right"]

# Syndrome for one or more measurement rounds, shape (rounds, n_faces), bool.
Syndrome = np.ndarray


@dataclass(frozen=True, slots=True)
class Face:
    """A stabilizer face.

    ``center`` is in lattice coordinates (row, col) where data qubit (r, c) sits at (r, c);
    interior faces are centred at half-integers, boundary faces hang half a cell outside.
    ``side`` is None for interior faces, otherwise which edge the half-face is attached to.
    """

    kind: FaceKind
    qubits: tuple[int, ...]
    center: tuple[float, float]
    side: Side | None = None


class SurfaceCode:
    """Distance-``d`` rotated surface code on a d x d grid of data qubits.

    Layout convention (fixed, see spec 3.1 and the plan):
      * interior face with top-left qubit (i, j) is Z-type iff (i + j) is even;
      * top/bottom boundary half-faces are X-type, left/right ones are Z-type,
        each placed so it never shares an edge with an interior face of the same type;
      * logical Z = Z on row 0 (parallel to the X-type boundary), logical X = X on column 0.
    """

    def __init__(self, d: int) -> None:
        if d < 3 or d % 2 == 0:
            raise ValueError("d must be an odd integer >= 3")
        self.d = d
        self.n = d * d
        self.faces: tuple[Face, ...] = tuple(self._build_faces())
        self.x_faces = np.array([i for i, f in enumerate(self.faces) if f.kind == "X"])
        self.z_faces = np.array([i for i, f in enumerate(self.faces) if f.kind == "Z"])
        self.h_x = self._check_matrix(self.x_faces)
        self.h_z = self._check_matrix(self.z_faces)
        self.logical_x = Pauli.from_support(self.n, xs=[self.qubit_index(r, 0) for r in range(d)])
        self.logical_z = Pauli.from_support(self.n, zs=[self.qubit_index(0, c) for c in range(d)])

    def __hash__(self) -> int:
        return hash(self.d)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SurfaceCode) and other.d == self.d

    # -- geometry -------------------------------------------------------------------------

    def qubit_index(self, r: int, c: int) -> int:
        return r * self.d + c

    def qubit_pos(self, q: int) -> tuple[int, int]:
        return divmod(q, self.d)

    def _build_faces(self) -> list[Face]:
        d = self.d
        q = self.qubit_index
        faces: list[Face] = []
        for i in range(d - 1):
            for j in range(d - 1):
                kind: FaceKind = "Z" if (i + j) % 2 == 0 else "X"
                qubits = (q(i, j), q(i, j + 1), q(i + 1, j), q(i + 1, j + 1))
                faces.append(Face(kind, qubits, (i + 0.5, j + 0.5)))
        # Boundary half-faces sit next to interior faces of the *other* type.
        for j in range(d - 1):
            if j % 2 == 0:
                faces.append(Face("X", (q(0, j), q(0, j + 1)), (-0.5, j + 0.5), "top"))
            if (d - 2 + j) % 2 == 0:
                faces.append(
                    Face("X", (q(d - 1, j), q(d - 1, j + 1)), (d - 0.5, j + 0.5), "bottom")
                )
        for i in range(d - 1):
            if i % 2 == 1:
                faces.append(Face("Z", (q(i, 0), q(i + 1, 0)), (i + 0.5, -0.5), "left"))
            if (i + d - 2) % 2 == 1:
                faces.append(Face("Z", (q(i, d - 1), q(i + 1, d - 1)), (i + 0.5, d - 0.5), "right"))
        return faces

    def _check_matrix(self, face_indices: np.ndarray) -> np.ndarray:
        h = np.zeros((len(face_indices), self.n), dtype=np.uint8)
        for row, fi in enumerate(face_indices):
            h[row, list(self.faces[fi].qubits)] = 1
        return h

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    @cached_property
    def faces_of_qubit(self) -> tuple[tuple[int, ...], ...]:
        """For each qubit, the indices of the faces it belongs to (used for hover preview)."""
        out: list[list[int]] = [[] for _ in range(self.n)]
        for fi, f in enumerate(self.faces):
            for q in f.qubits:
                out[q].append(fi)
        return tuple(tuple(v) for v in out)

    # -- stabilizers ----------------------------------------------------------------------

    def stabilizer(self, face_index: int) -> Pauli:
        f = self.faces[face_index]
        if f.kind == "X":
            return Pauli.from_support(self.n, xs=f.qubits)
        return Pauli.from_support(self.n, zs=f.qubits)

    def stabilizer_faces(self, p: Pauli) -> np.ndarray | None:
        """Face indices whose product equals ``p``, or None if ``p`` is not a stabilizer.

        The generators are independent, so the decomposition is unique when it exists.
        """
        fx = _solve_gf2(self.h_x.T, p.x.astype(np.uint8))
        fz = _solve_gf2(self.h_z.T, p.z.astype(np.uint8))
        if fx is None or fz is None:
            return None
        return np.concatenate([self.x_faces[fx.astype(bool)], self.z_faces[fz.astype(bool)]])

    def syndrome(self, p: Pauli) -> Syndrome:
        """Perfect single-round syndrome, shape (1, n_faces).

        X-type faces detect the Z part of ``p``, Z-type faces detect the X part.
        """
        s = np.zeros((1, self.n_faces), dtype=bool)
        s[0, self.x_faces] = (self.h_x @ p.z.astype(np.uint8)) & 1
        s[0, self.z_faces] = (self.h_z @ p.x.astype(np.uint8)) & 1
        return s


def _solve_gf2(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """Solve a @ x = b over GF(2) by Gaussian elimination; None if inconsistent."""
    m = np.concatenate([a.astype(np.uint8) % 2, (b.astype(np.uint8) % 2)[:, None]], axis=1)
    rows, cols = a.shape
    pivots: list[int] = []
    r = 0
    for c in range(cols):
        hit = np.flatnonzero(m[r:, c])
        if hit.size == 0:
            continue
        m[[r, r + hit[0]]] = m[[r + hit[0], r]]
        others = np.flatnonzero(m[:, c])
        others = others[others != r]
        m[others] ^= m[r]
        pivots.append(c)
        r += 1
        if r == rows:
            break
    if m[r:, -1].any():
        return None
    x = np.zeros(cols, dtype=np.uint8)
    x[pivots] = m[: len(pivots), -1]
    return x
