"""Board state machine: toggles, undo/redo, judgement gating and classes."""

import pytest

from syndrome_out import LogicalEffect, Pauli
from syndrome_out.game import DISTANCES, ERROR_RATES, Board, pack_seed, unpack_seed


def test_toggle_updates_lit_faces() -> None:
    b = Board.new(5, 0.0, seed=0)
    assert b.error.is_identity()
    assert b.all_clear
    b.toggle(12, "X")  # centre qubit of d=5 lights two Z-type faces
    assert b.lit_counts == (0, 2)
    b.toggle(12, "X")
    assert b.all_clear


def test_undo_redo() -> None:
    b = Board.new(3, 0.0, seed=0)
    b.toggle(0, "X")
    b.toggle(1, "Z")
    assert b.correction.weight == 2
    assert b.undo() and b.correction.weight == 1
    assert b.undo() and b.correction.is_identity()
    assert not b.undo()
    assert b.redo() and b.correction.weight == 1
    b.toggle(4, "Y")  # a new move clears the redo stack
    assert not b.redo()


def test_judge_requires_all_clear() -> None:
    b = Board.new(5, 0.10, seed=42)
    assert not b.all_clear
    assert b.judge() is None
    assert not b.judged


def test_bot_correction_succeeds() -> None:
    b = Board.new(5, 0.10, seed=42)
    # Play the bot's correction ourselves: must clear the board and (for this seed) succeed.
    for q in range(b.code.n):
        k = b.bot_correction.kind(q)
        if k != "I":
            b.toggle(q, k)
    assert b.all_clear
    v = b.judge()
    assert v is not None
    assert v.effect is v.bot_effect
    assert v.weight == v.bot_weight
    if v.success:
        assert v.optimal
        assert v.score == 100 * 5 + 50
    # Judged boards are frozen.
    b.toggle(0, "X")
    assert b.correction == b.bot_correction


def test_clearing_with_a_logical_error_fails() -> None:
    b = Board.new(3, 0.0, seed=0)
    for q in range(b.code.n):
        if b.code.logical_x.x[q]:
            b.toggle(q, "X")
    assert b.all_clear
    v = b.judge()
    assert v is not None
    assert v.effect is LogicalEffect.X
    assert not v.success
    assert v.score == 0
    assert v.bot_success
    assert b.crossing_logicals() == [b.code.logical_z]


def test_reset_keeps_seed_and_error() -> None:
    b = Board.new(5, 0.10, seed=7)
    e = b.error
    b.toggle(3, "Z")
    b.reset()
    assert b.error == e
    assert b.correction == Pauli.identity(25)
    assert not b.judged


def test_demo_seed_minimum_weight_fails() -> None:
    """The seed documented in README: MWPM clears the board but flips the logical qubit."""
    assert unpack_seed(0x500032) == (5, 0.10, 50)
    b = Board.new(5, 0.10, seed=50)
    for q in range(b.code.n):
        k = b.bot_correction.kind(q)
        if k != "I":
            b.toggle(q, k)
    v = b.judge()
    assert v is not None and b.all_clear
    assert v.effect is LogicalEffect.X
    assert v.weight == 4 and b.error.weight == 3


def test_seed_code_round_trip() -> None:
    for d in DISTANCES:
        for p in ERROR_RATES:
            for raw in (0, 1, 0xABCDE, (1 << 20) - 1):
                code = pack_seed(d, p, raw)
                assert 0 <= code < 1 << 24
                assert unpack_seed(code) == (d, p, raw)
    # The leading hex digit is d_index * 4 + p_index.
    assert f"{pack_seed(7, 0.15, 0):06X}" == "A00000"


def test_seed_code_rejects_impossible_values() -> None:
    with pytest.raises(ValueError):
        unpack_seed(0x300000)  # p index 3 has no error rate
    with pytest.raises(ValueError):
        unpack_seed(1 << 24)
    with pytest.raises(ValueError):
        pack_seed(3, 0.05, 1 << 20)
