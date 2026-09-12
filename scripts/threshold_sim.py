"""M2 acceptance: logical error rate of MWPM at fixed p must decrease with d (below threshold).

Usage: uv run scripts/threshold_sim.py [-p 0.05] [-n 10000] [-d 3 5 7] [--decoder pymatching]

The default frontier decoder is pure Python (a few ms per shot); pass --decoder pymatching
(needs `uv sync --extra mwpm`) for large runs.
"""

from __future__ import annotations

import argparse
import time
from typing import Callable

import numpy as np

from syndrome_out import SurfaceCode
from syndrome_out.noise import sample_depolarizing_batch

BatchDecoder = Callable[[SurfaceCode, np.ndarray], tuple[np.ndarray, np.ndarray]]


def _decoder(name: str) -> BatchDecoder:
    if name == "pymatching":
        from syndrome_out.decoder_pymatching import decode_mwpm_batch

        return decode_mwpm_batch
    from syndrome_out.decoder import decode_min_weight_batch

    return decode_min_weight_batch


def logical_error_rate(
    d: int, p: float, shots: int, seed: int, decode_batch: BatchDecoder
) -> tuple[float, float, float]:
    code = SurfaceCode(d)
    x, z = sample_depolarizing_batch(code, p, seed, shots)
    syndromes = np.zeros((shots, code.n_faces), dtype=bool)
    syndromes[:, code.z_faces] = (x.astype(np.uint8) @ code.h_z.T) & 1
    syndromes[:, code.x_faces] = (z.astype(np.uint8) @ code.h_x.T) & 1
    cx, cz = decode_batch(code, syndromes)
    rx, rz = x ^ cx, z ^ cz
    fail_x = (rx[:, code.logical_z.z].sum(axis=1) & 1).astype(bool)  # residual acts as X-bar
    fail_z = (rz[:, code.logical_x.x].sum(axis=1) & 1).astype(bool)
    return fail_x.mean(), fail_z.mean(), (fail_x | fail_z).mean()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-p", type=float, default=0.05)
    ap.add_argument("-n", "--shots", type=int, default=10_000)
    ap.add_argument("-d", nargs="+", type=int, default=[3, 5, 7])
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--decoder", choices=["frontier", "pymatching"], default="frontier")
    args = ap.parse_args()
    decode_batch = _decoder(args.decoder)

    print(f"p = {args.p}, shots = {args.shots}, decoder = {args.decoder}")
    print(f"{'d':>3} {'P(X-bar)':>10} {'P(Z-bar)':>10} {'P(fail)':>10} {'time':>7}")
    rates = []
    for d in args.d:
        t0 = time.perf_counter()
        fx, fz, f = logical_error_rate(d, args.p, args.shots, args.seed + d, decode_batch)
        rates.append(f)
        print(f"{d:>3} {fx:>10.4f} {fz:>10.4f} {f:>10.4f} {time.perf_counter() - t0:>6.2f}s")
    decreasing = all(a > b for a, b in zip(rates, rates[1:]))
    print("logical error rate decreases with d:", "YES" if decreasing else "NO")


if __name__ == "__main__":
    main()
