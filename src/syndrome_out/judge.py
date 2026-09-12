"""Logical-error judgement and scoring (spec 3.3 and 6.2)."""

from __future__ import annotations

from enum import Enum

from syndrome_out.code import SurfaceCode
from syndrome_out.pauli import Pauli


class LogicalEffect(Enum):
    """Which logical operator the residual acts as (NONE = in the stabilizer group)."""

    NONE = "I"
    X = "X"
    Z = "Z"
    Y = "Y"


def logical_effect(code: SurfaceCode, residual: Pauli) -> LogicalEffect:
    """Classify a syndrome-free residual E*C into {I, X, Z, Y}.

    Anticommuting with logical Z means the residual carries a logical X, and vice versa.
    Callers must ensure the residual has zero syndrome; this function does not check.
    """
    has_x = residual.symplectic(code.logical_z)
    has_z = residual.symplectic(code.logical_x)
    return LogicalEffect("IXZY"[has_x + 2 * has_z])


def score(d: int, weight: int, bot_weight: int) -> int:
    """Base 100*d, +50 when no heavier than the bot, otherwise -10 per extra unit of weight."""
    base = 100 * d
    if weight <= bot_weight:
        return base + 50
    return base - 10 * (weight - bot_weight)
