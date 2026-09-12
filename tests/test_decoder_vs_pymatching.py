"""Cross-check the frontier decoder against PyMatching. Skipped unless the mwpm extra is installed."""

import numpy as np
import pytest

pytest.importorskip("pymatching")

from syndrome_out import Pauli, SurfaceCode, decode, logical_effect  # noqa: E402
from syndrome_out.decoder import decode_classes  # noqa: E402
from syndrome_out.decoder_pymatching import decode_mwpm  # noqa: E402


@pytest.mark.parametrize("d", [3, 5, 7, 9])
def test_min_weight_matches_pymatching(d: int) -> None:
    code = SurfaceCode(d)
    rng = np.random.default_rng(d)
    for _ in range(200):
        error = Pauli(rng.random(code.n) < 0.1, rng.random(code.n) < 0.1)
        s = code.syndrome(error)
        ours = decode(code, s, 0.1).min_weight
        ref = decode_mwpm(code, s)
        assert ours.x.sum() == ref.x.sum()
        assert ours.z.sum() == ref.z.sum()
        (x0, x1), (z0, z1) = decode_classes(code, s, 0.1)
        # Unless both classes tie on weight, the residual class must agree too.
        if x0.weight != x1.weight and z0.weight != z1.weight:
            assert logical_effect(code, error * ours) is logical_effect(code, error * ref)
