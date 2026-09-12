"""Reference MWPM decoder backed by PyMatching. Needs the ``mwpm`` extra (`uv sync --extra mwpm`)."""

from __future__ import annotations

from functools import cache

import numpy as np
import pymatching

from syndrome_out.code import SurfaceCode, Syndrome
from syndrome_out.pauli import Pauli


@cache
def _matchers(code: SurfaceCode) -> tuple[pymatching.Matching, pymatching.Matching]:
    # Z-type faces see X errors and vice versa; weight-1 columns become boundary edges.
    return pymatching.Matching(code.h_z), pymatching.Matching(code.h_x)


def decode_mwpm(code: SurfaceCode, syndrome: Syndrome) -> Pauli:
    """Decode X and Z errors independently and combine. Uses the last round."""
    s = np.asarray(syndrome, dtype=np.uint8)[-1]
    m_x, m_z = _matchers(code)
    x = m_x.decode(s[code.z_faces]).astype(bool)
    z = m_z.decode(s[code.x_faces]).astype(bool)
    return Pauli(x, z)


def decode_mwpm_batch(code: SurfaceCode, syndromes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batch variant: ``syndromes`` is (shots, n_faces); returns (x, z)."""
    s = np.asarray(syndromes, dtype=np.uint8)
    m_x, m_z = _matchers(code)
    x = m_x.decode_batch(s[:, code.z_faces]).astype(bool)
    z = m_z.decode_batch(s[:, code.x_faces]).astype(bool)
    return x, z
