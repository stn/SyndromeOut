"""UI-independent game state for the Sweep mode (spec 6.1, 6.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from syndrome_out.code import SurfaceCode, Syndrome
from syndrome_out.decoder import decode
from syndrome_out.judge import LogicalEffect, logical_effect
from syndrome_out.noise import sample_depolarizing
from syndrome_out.pauli import Pauli

# The bot plays a minimum-weight correction. With the `mwpm` extra installed it comes from
# PyMatching and is labelled as such; otherwise the built-in frontier decoder supplies it
# (same weight, possibly a different tie-break).
try:
    from syndrome_out.decoder_pymatching import decode_mwpm
except ImportError:
    decode_mwpm = None
BOT_LABEL = "bot (MWPM)" if decode_mwpm else "bot"

Toggle = Literal["X", "Z", "Y"]

DISTANCES = (3, 5, 7, 9)
ERROR_RATES = (0.05, 0.10, 0.15)

# A game is reproduced from one 24-bit code, shown in hex: d index (2 bits), p index (2 bits),
# then the raw RNG seed (20 bits). The leading hex digit therefore reads as d_index * 4 + p_index.
RAW_BITS = 20
_RAW_MASK = (1 << RAW_BITS) - 1


def pack_seed(d: int, p: float, seed: int) -> int:
    if not 0 <= seed <= _RAW_MASK:
        raise ValueError(f"raw seed must fit in {RAW_BITS} bits")
    return (DISTANCES.index(d) << (RAW_BITS + 2)) | (ERROR_RATES.index(p) << RAW_BITS) | seed


def unpack_seed(code: int) -> tuple[int, float, int]:
    """Inverse of pack_seed: (d, p, raw seed). Raises ValueError on codes no game produces."""
    if not 0 <= code < 1 << (RAW_BITS + 4):
        raise ValueError("seed code must be 6 hex digits")
    p_index = (code >> RAW_BITS) & 3
    if p_index >= len(ERROR_RATES):
        raise ValueError("seed code has no error rate for its p index")
    d_index = code >> (RAW_BITS + 2)
    return DISTANCES[d_index], ERROR_RATES[p_index], code & _RAW_MASK


@dataclass(frozen=True, slots=True)
class Verdict:
    effect: LogicalEffect
    weight: int
    bot_weight: int
    bot_effect: LogicalEffect
    ml_effect: LogicalEffect
    error_weight: int

    @property
    def success(self) -> bool:
        return self.effect is LogicalEffect.NONE

    @property
    def bot_success(self) -> bool:
        return self.bot_effect is LogicalEffect.NONE

    @property
    def not_ml(self) -> bool:
        """Succeeded, but the most likely class (summed over all consistent errors) was another one."""
        return self.success and self.ml_effect is not LogicalEffect.NONE

    @property
    def failed_as_ml(self) -> bool:
        """Failed while playing the most likely class: the board was a losing one."""
        return not self.success and self.ml_effect is self.effect

    @property
    def not_optimal(self) -> bool:
        """A lighter successful correction is known: the hidden error itself, or the bot's (if it succeeded)."""
        bound = self.error_weight
        if self.bot_success:
            bound = min(bound, self.bot_weight)
        return self.success and self.weight > bound


@dataclass(slots=True)
class Board:
    """One puzzle instance: hidden error, the player's correction, undo/redo, judgement."""

    code: SurfaceCode
    p: float
    seed: int
    error: Pauli = field(init=False)
    correction: Pauli = field(init=False)
    bot_correction: Pauli = field(init=False)  # minimum weight (PyMatching if installed)
    ml_correction: Pauli = field(
        init=False
    )  # lightest member of the most likely class under depolarizing noise (ML tags)
    verdict: Verdict | None = field(init=False, default=None)
    # After judging, R = E*C decomposed as (logical strings) * (product of faces); see judge().
    residual_paths: list[Pauli] = field(init=False, default_factory=list)
    residual_faces: np.ndarray = field(init=False, default_factory=lambda: np.zeros(0, dtype=int))
    _undo: list[Pauli] = field(init=False, default_factory=list)
    _redo: list[Pauli] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.error = sample_depolarizing(self.code, self.p, self.seed)
        self.correction = Pauli.identity(self.code.n)
        syndrome = self.code.syndrome(self.error)
        decoding = decode(self.code, syndrome, self.p)
        self.bot_correction = (
            decode_mwpm(self.code, syndrome) if decode_mwpm else decoding.min_weight
        )
        self.ml_correction = decoding.ml

    @classmethod
    def new(cls, d: int, p: float, seed: int) -> Board:
        return cls(SurfaceCode(d), p, seed)

    # -- derived state --------------------------------------------------------------------

    @property
    def d(self) -> int:
        return self.code.d

    @property
    def residual(self) -> Pauli:
        return self.error * self.correction

    @property
    def lit(self) -> Syndrome:
        """Syndrome of E*C, i.e. what the player currently sees."""
        return self.code.syndrome(self.residual)

    @property
    def lit_counts(self) -> tuple[int, int]:
        s = self.lit[-1]
        return int(s[self.code.x_faces].sum()), int(s[self.code.z_faces].sum())

    @property
    def all_clear(self) -> bool:
        return not self.lit.any()

    @property
    def judged(self) -> bool:
        return self.verdict is not None

    def lit_faces(self) -> np.ndarray:
        return np.flatnonzero(self.lit[-1])

    # -- moves ----------------------------------------------------------------------------

    def toggle(self, qubit: int, kind: Toggle) -> None:
        """Multiply the correction by X/Z/Y on ``qubit``. Ignored once judged."""
        if self.judged:
            return
        xs = [qubit] if kind in ("X", "Y") else []
        zs = [qubit] if kind in ("Z", "Y") else []
        self._push(self.correction * Pauli.from_support(self.code.n, xs=xs, zs=zs))

    def _push(self, new: Pauli) -> None:
        self._undo.append(self.correction)
        self._redo.clear()
        self.correction = new

    def undo(self) -> bool:
        if self.judged or not self._undo:
            return False
        self._redo.append(self.correction)
        self.correction = self._undo.pop()
        return True

    def redo(self) -> bool:
        if self.judged or not self._redo:
            return False
        self._undo.append(self.correction)
        self.correction = self._redo.pop()
        return True

    def reset(self) -> None:
        """Same seed, fresh correction (retry)."""
        self.correction = Pauli.identity(self.code.n)
        self.verdict = None
        self.residual_paths = []
        self.residual_faces = np.zeros(0, dtype=int)
        self._undo.clear()
        self._redo.clear()

    # -- judgement ------------------------------------------------------------------------

    def judge(self) -> Verdict | None:
        """Judge the current correction; only allowed when every face is dark."""
        if self.judged:
            return self.verdict
        if not self.all_clear:
            return None
        effect = logical_effect(self.code, self.residual)
        bot_effect = logical_effect(self.code, self.error * self.bot_correction)
        ml_effect = logical_effect(self.code, self.error * self.ml_correction)
        weight = self.correction.weight
        bot_weight = self.bot_correction.weight
        self.verdict = Verdict(effect, weight, bot_weight, bot_effect, ml_effect, self.error.weight)
        self.residual_paths, self.residual_faces = self._decompose_residual(effect)
        return self.verdict

    def _decompose_residual(self, effect: LogicalEffect) -> tuple[list[Pauli], np.ndarray]:
        """Write the syndrome-free residual as logical strings times a product of faces.

        A residual carrying a logical Z contains a Z string across some row (row k's string is
        logical Z times the Z faces between rows 0 and k, so every row is a valid
        representative); likewise a logical X is an X string down some column. The row and
        column that leave the fewest faces are chosen so the overlay stays close to R itself.
        On a success there is no string and the faces alone make up R.
        """
        code = self.code
        d = code.d
        identity = [Pauli.identity(code.n)]
        z_strings = identity
        x_strings = identity
        if effect in (LogicalEffect.Z, LogicalEffect.Y):
            z_strings = [
                Pauli.from_support(code.n, zs=[code.qubit_index(k, c) for c in range(d)])
                for k in range(d)
            ]
        if effect in (LogicalEffect.X, LogicalEffect.Y):
            x_strings = [
                Pauli.from_support(code.n, xs=[code.qubit_index(r, k) for r in range(d)])
                for k in range(d)
            ]
        best: tuple[list[Pauli], np.ndarray] | None = None
        for zs in z_strings:
            for xs in x_strings:
                faces = code.stabilizer_faces(self.residual * zs * xs)
                assert faces is not None  # R times a representative of its class is a stabilizer
                if best is None or len(faces) < len(best[1]):
                    best = ([s for s in (xs, zs) if not s.is_identity()], faces)
        assert best is not None
        return best
