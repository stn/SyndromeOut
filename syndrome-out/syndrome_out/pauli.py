"""Pauli operators on n qubits in the binary symplectic representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True, slots=True, eq=False)
class Pauli:
    """An n-qubit Pauli (up to phase): qubit i is I/X/Z/Y for (x_i, z_i) = 00/10/01/11."""

    x: np.ndarray
    z: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", np.asarray(self.x, dtype=bool))
        object.__setattr__(self, "z", np.asarray(self.z, dtype=bool))
        if self.x.shape != self.z.shape or self.x.ndim != 1:
            raise ValueError("x and z must be 1-D arrays of equal length")

    @classmethod
    def identity(cls, n: int) -> Pauli:
        return cls(np.zeros(n, dtype=bool), np.zeros(n, dtype=bool))

    @classmethod
    def from_support(cls, n: int, xs: Iterable[int] = (), zs: Iterable[int] = ()) -> Pauli:
        x = np.zeros(n, dtype=bool)
        z = np.zeros(n, dtype=bool)
        x[list(xs)] = True
        z[list(zs)] = True
        return cls(x, z)

    @property
    def n(self) -> int:
        return self.x.shape[0]

    def __mul__(self, other: Pauli) -> Pauli:
        """Product up to phase, i.e. XOR of the symplectic vectors."""
        return Pauli(self.x ^ other.x, self.z ^ other.z)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Pauli):
            return NotImplemented
        return bool(np.array_equal(self.x, other.x) and np.array_equal(self.z, other.z))

    def __hash__(self) -> int:
        return hash((self.x.tobytes(), self.z.tobytes()))

    def symplectic(self, other: Pauli) -> int:
        """Symplectic inner product mod 2; 1 iff the two operators anticommute."""
        return int(np.count_nonzero((self.x & other.z) ^ (self.z & other.x)) & 1)

    def commutes_with(self, other: Pauli) -> bool:
        return self.symplectic(other) == 0

    @property
    def weight(self) -> int:
        return int(np.count_nonzero(self.x | self.z))

    def is_identity(self) -> bool:
        return not (self.x.any() or self.z.any())

    def kind(self, i: int) -> str:
        """Single-qubit factor on qubit i as one of I, X, Y, Z."""
        return "IXZY"[int(self.x[i]) + 2 * int(self.z[i])]

    def __str__(self) -> str:
        return "".join(self.kind(i) for i in range(self.n))
