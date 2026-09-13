"""Board state machine: toggles, undo/redo, judgement gating and classes."""

import numpy as np
import pytest

from syndrome_out import LogicalEffect, Pauli
from syndrome_out.game import DISTANCES, ERROR_RATES, Board, decode_mwpm, pack_seed, unpack_seed


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
        assert v.not_optimal == (v.weight > b.error.weight)
    # Judged boards are frozen.
    b.toggle(0, "X")
    assert b.correction == b.bot_correction


def test_stabilizer_on_clean_board_is_not_optimal() -> None:
    b = Board.new(3, 0.0, seed=0)
    face = b.code.faces[0]
    for q in face.qubits:
        b.toggle(q, face.kind)
    assert b.all_clear
    v = b.judge()
    assert v is not None
    assert v.success and v.bot_success and not v.not_ml
    assert v.weight == len(face.qubits) and v.error_weight == 0
    assert v.not_optimal


def test_clearing_with_a_logical_error_fails() -> None:
    b = Board.new(3, 0.0, seed=0)
    for q in range(b.code.n):
        if b.code.logical_x.x[q]:
            b.toggle(q, "X")
    assert b.all_clear
    v = b.judge()
    assert v is not None
    assert v.effect is LogicalEffect.X
    assert not v.success and not v.failed_as_ml  # a clean board's likeliest class is I
    assert v.bot_success
    # R is exactly the column-0 X string: one path, no faces.
    assert b.residual_paths == [b.code.logical_x]
    assert len(b.residual_faces) == 0


def _product_of_faces(b: Board, faces) -> Pauli:
    out = Pauli.identity(b.code.n)
    for fi in faces:
        out = out * b.code.stabilizer(int(fi))
    return out


def test_residual_decomposition_on_failure() -> None:
    """50009C: the bot's residual carries a logical Z, so R is one Z row string times faces,
    with the row chosen to need the fewest faces."""
    b = Board.new(5, 0.10, seed=156)
    v = _play(b, b.bot_correction)
    assert v.effect is LogicalEffect.Z
    (path,) = b.residual_paths
    assert not path.x.any() and path.z.sum() == 5
    rows = {b.code.qubit_pos(int(q))[0] for q in np.flatnonzero(path.z)}
    assert len(rows) == 1
    assert path * _product_of_faces(b, b.residual_faces) == b.residual
    # No other row does better.
    for k in range(5):
        alt = Pauli.from_support(b.code.n, zs=[b.code.qubit_index(k, c) for c in range(5)])
        faces = b.code.stabilizer_faces(b.residual * alt)
        assert faces is not None and len(faces) >= len(b.residual_faces)


def test_residual_decomposition_on_success() -> None:
    b = Board.new(5, 0.10, seed=156)
    v = _play(b, b.error)
    assert v.success
    assert b.residual_paths == [] and len(b.residual_faces) == 0  # R is the identity
    b = Board.new(3, 0.0, seed=0)
    face = b.code.faces[0]
    for q in face.qubits:
        b.toggle(q, face.kind)
    b.judge()
    assert b.residual_paths == [] and list(b.residual_faces) == [0]
    b.reset()
    assert b.residual_paths == [] and len(b.residual_faces) == 0


def test_reset_keeps_seed_and_error() -> None:
    b = Board.new(5, 0.10, seed=7)
    e = b.error
    b.toggle(3, "Z")
    b.reset()
    assert b.error == e
    assert b.correction == Pauli.identity(25)
    assert not b.judged


def _play(b: Board, pauli: Pauli):
    for q in range(b.code.n):
        k = pauli.kind(q)
        if k != "I":
            b.toggle(q, k)
    v = b.judge()
    assert v is not None and b.all_clear
    return v


def test_demo_seed_minimum_weight_fails() -> None:
    """The seed documented in README: both bots clear the board but flip the logical qubit."""
    assert unpack_seed(0x50009C) == (5, 0.10, 156)
    b = Board.new(5, 0.10, seed=156)
    v = _play(b, b.bot_correction)
    assert v.effect is LogicalEffect.Z
    assert v.ml_effect is LogicalEffect.Z
    assert v.failed_as_ml and not v.not_ml  # shown as FAIL Z error (but ML)
    assert v.weight == 4 and b.error.weight == 5


def test_demo_seed_true_error_is_not_ml() -> None:
    """Playing the hidden error itself succeeds, but the likelier class was the other one."""
    b = Board.new(5, 0.10, seed=156)
    v = _play(b, b.error)
    assert v.success
    assert v.not_ml and not v.not_optimal
    assert v.ml_effect is LogicalEffect.Z  # shown as SUCCESS (but not ML: Z)


def test_ml_tag_counts_a_y_once() -> None:
    """Board 551A6A: the true error is Z, Y, Y (weight 3). Independent X/Z decoding would call
    the class a logical Z away likelier; under depolarizing noise the true class is ML."""
    assert unpack_seed(0x551A6A) == (5, 0.10, 334442)
    b = Board.new(5, 0.10, seed=334442)
    assert str(b.error).count("Y") == 2 and b.error.weight == 3
    v = _play(b, b.error)
    assert v.success and not v.not_ml and not v.not_optimal  # plain SUCCESS
    assert v.bot_effect is LogicalEffect.Z and v.bot_weight == 4
    b = Board.new(5, 0.10, seed=334442)
    v = _play(b, b.bot_correction)
    assert v.effect is LogicalEffect.Z and not v.failed_as_ml  # FAIL Z error, no (but ML)


def test_old_demo_seed_true_class_is_ml() -> None:
    """50FCF8 (the former README example): the min-weight correction still fails, but the true
    class is the likelier one, so neither verdict carries an ML tag."""
    b = Board.new(5, 0.10, seed=64760)
    v = _play(b, b.bot_correction)
    assert v.effect is LogicalEffect.Z and not v.failed_as_ml
    assert v.weight == (6 if decode_mwpm else 5) and b.error.weight == 5
    b = Board.new(5, 0.10, seed=64760)
    v = _play(b, b.error)
    assert v.success and not v.not_ml


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
