"""Code-capacity noise: i.i.d. depolarizing errors on data qubits."""

from __future__ import annotations

import numpy as np

from syndrome_out.code import SurfaceCode
from syndrome_out.pauli import Pauli


def sample_depolarizing(code: SurfaceCode, p: float, seed: int) -> Pauli:
    """Each qubit independently suffers X, Y or Z with probability p/3 each."""
    x, z = sample_depolarizing_batch(code, p, seed, 1)
    return Pauli(x[0], z[0])


def sample_depolarizing_batch(
    code: SurfaceCode, p: float, seed: int, shots: int
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised sampler for simulations; returns (x, z) bool arrays of shape (shots, n)."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    rng = np.random.default_rng(seed)
    u = rng.random((shots, code.n))
    # [0, p/3) -> X, [p/3, 2p/3) -> Y, [2p/3, p) -> Z, [p, 1) -> I
    x = u < 2 * p / 3
    z = (u >= p / 3) & (u < p)
    return x, z
