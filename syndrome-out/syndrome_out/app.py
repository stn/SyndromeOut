"""Pyxel front-end: Lights Out look and feel for the Sweep mode."""

from __future__ import annotations

import argparse
import random
from typing import NamedTuple

import numpy as np
import pyxel

from syndrome_out.code import Face
from syndrome_out.game import (
    BOT_LABEL,
    DISTANCES,
    ERROR_RATES,
    RAW_BITS,
    Board,
    Toggle,
    pack_seed,
    unpack_seed,
)
from syndrome_out.web import show_code_in_url

WIDTH, HEIGHT = 360, 240
BOARD_W = 236  # board area on the left, panel on the right
PANEL_X = BOARD_W + 6
MAX_CELL = 28
VERDICT_Y = 74  # panel row where the JUDGE button, then the verdict readout, is drawn

# Palette indices (see PALETTE below).
BG, BOARD_BG, TILE_OFF, X_EDGE, Z_EDGE = 0, 1, 2, 3, 4
X_LIT, X_DIM, X_GLOW, Z_LIT, Z_DIM, Z_GLOW = 5, 6, 7, 8, 9, 10
BUTTON, BUTTON_HI, TEXT, GREEN, YELLOW = 11, 12, 13, 14, 15
RED = X_LIT
BLUE = Z_LIT

PALETTE = [
    0x0B0E14,
    0x141923,
    0x1F2733,
    0x4D2A31,
    0x25395A,
    0xFF4D5E,
    0xB33546,
    0x5E1F2A,
    0x4DA6FF,
    0x2F6FB5,
    0x1C3A5E,
    0x3C4657,
    0x5B6A85,
    0xE3E8F0,
    0x4ADE80,
    0xFACC15,
]

HELP_LINES = [
    "SYNDROME OUT - HOW TO PLAY",
    "",
    "Buttons are data qubits, tiles are",
    "stabilizer faces. A hidden error E",
    "lights faces. A RED tile means an X",
    "error is next to it: clear it with",
    "X (red mark). A BLUE tile means a Z",
    "error: clear it with Z (blue mark).",
    "",
    "Place X / Z (or both = Y) on qubits",
    "to flip the parity of the faces",
    "around them. Turn every tile off,",
    "then press Enter to judge.",
    "",
    "DIFFERENCE FROM LIGHTS OUT",
    "All-dark is only the precondition.",
    "Many corrections clear the board;",
    "they fall into 4 classes I/X/Z/Y.",
    "Only class I is safe: if E*C equals",
    "a logical operator (a string of X",
    "across a column or Z across a row)",
    "your correction silently flipped",
    "the logical qubit. This is called",
    "DEGENERACY: you cannot see E, only",
    "its syndrome, so you must guess the",
    "most likely class (summed over every",
    "error). 'but not ML: Z': the class a Z",
    "away was likelier; 'but ML': yours was.",
    "press ? to close",
]


# Operators shown side by side after judging: (key, ring colour, banner). Banners must fit a
# half-width cell (about 27 characters including the weight suffix).
VIEWS = (
    ("C", YELLOW, "C  your correction"),
    ("B", BLUE, f"B  {BOT_LABEL}"),
    ("E", GREEN, "E  true error"),
    ("R", RED, "R  residual E*C"),
)
RESIDUAL_VIEW = 3


class Layout(NamedTuple):
    """Where a board is drawn: origin, cell pitch and button radius."""

    ox: int
    oy: int
    cs: int
    rq: int


def tile_is_red(face: Face) -> bool:
    """Colour a tile by the correction that clears it: Z-type faces detect X errors -> red."""
    return face.kind == "Z"


class App:
    def __init__(self, d: int, p: float, seed: int) -> None:
        self.d_index = DISTANCES.index(d)
        self.p_index = ERROR_RATES.index(p)
        self.board = Board.new(d, p, seed)
        show_code_in_url(pack_seed(d, p, seed))
        self.cursor = (d // 2, d // 2)
        self.show_help = False
        self.hover: int | None = None
        self.pressed: int | None = None  # qubit whose button is drawn "sunk" this frame
        self._layout()

        pyxel.init(WIDTH, HEIGHT, title="Syndrome Out", fps=30, quit_key=pyxel.KEY_ESCAPE)
        pyxel.colors[:] = PALETTE
        pyxel.mouse(True)
        self._init_sounds()
        pyxel.run(self.update, self.draw)

    # -- layout -----------------------------------------------------------------------------

    def _layout(self) -> None:
        d = self.board.d
        cs = min(MAX_CELL, (BOARD_W - 8) // (d + 1))
        size = (d + 1) * cs
        self.play = Layout((BOARD_W - size) // 2, (HEIGHT - size) // 2, cs, max(3, cs // 4 - 1))

        # After judging, the board area splits into a 2x2 grid, one cell per VIEWS entry,
        # each cell keeping a 9px banner strip above its board.
        cell_w, cell_h, banner = BOARD_W // 2, HEIGHT // 2, 10
        cs = min(MAX_CELL, (cell_w - 8) // (d + 1), (cell_h - banner - 4) // (d + 1))
        size = (d + 1) * cs
        self.quad = [
            Layout(
                (i % 2) * cell_w + (cell_w - size) // 2,
                (i // 2) * cell_h + banner + (cell_h - banner - size) // 2,
                cs,
                max(3, cs // 4 - 1),
            )
            for i in range(len(VIEWS))
        ]

    def qubit_xy(self, q: int, lay: Layout) -> tuple[int, int]:
        r, c = self.board.code.qubit_pos(q)
        return lay.ox + (c + 1) * lay.cs, lay.oy + (r + 1) * lay.cs

    def face_xy(self, f: Face, lay: Layout) -> tuple[int, int]:
        fr, fc = f.center
        return int(lay.ox + (fc + 1) * lay.cs), int(lay.oy + (fr + 1) * lay.cs)

    def face_rect(self, f: Face, lay: Layout) -> tuple[int, int, int, int]:
        """(x, y, w, h) of the tile; boundary faces are half tiles hugging the edge."""
        cx, cy = self.face_xy(f, lay)
        hs = lay.cs // 2 - 2
        match f.side:
            case None:
                return cx - hs, cy - hs, 2 * hs + 1, 2 * hs + 1
            case "top":
                return cx - hs, cy, 2 * hs + 1, hs
            case "bottom":
                return cx - hs, cy - hs + 1, 2 * hs + 1, hs
            case "left":
                return cx, cy - hs, hs, 2 * hs + 1
            case "right":
                return cx - hs + 1, cy - hs, hs, 2 * hs + 1
        raise AssertionError(f.side)

    def qubit_at(self, mx: int, my: int) -> int | None:
        hit = self.play.rq + 3
        for q in range(self.board.code.n):
            x, y = self.qubit_xy(q, self.play)
            if abs(mx - x) <= hit and abs(my - y) <= hit:
                return q
        return None

    def judge_button_rect(self) -> tuple[int, int, int, int]:
        """JUDGE sits under the legend; NEW / RETRY moves below the verdict readout."""
        y = VERDICT_Y
        v = self.board.verdict
        if v is not None:
            y += 8 * (3 + v.not_optimal) + 6  # verdict, tag, blank, bot line
        return PANEL_X, y, WIDTH - PANEL_X - 6, 13

    # -- sounds -----------------------------------------------------------------------------

    def _init_sounds(self) -> None:
        pyxel.sounds[0].set("c3", "p", "5", "n", 4)  # click
        pyxel.sounds[1].set("c3e3g3c4", "t", "6", "n", 6)  # success
        pyxel.sounds[2].set("c2c2", "s", "6", "f", 12)  # failure
        pyxel.sounds[3].set("g2", "p", "4", "n", 4)  # denied

    # -- update -----------------------------------------------------------------------------

    def update(self) -> None:
        self.pressed = None
        if pyxel.btnp(pyxel.KEY_SLASH) and pyxel.btn(pyxel.KEY_SHIFT):  # '?'
            self.show_help = not self.show_help
        if self.show_help:
            return

        self.hover = self.qubit_at(pyxel.mouse_x, pyxel.mouse_y)
        if self.hover is not None:
            self.cursor = self.board.code.qubit_pos(self.hover)
            if pyxel.btnp(pyxel.MOUSE_BUTTON_LEFT):
                self.toggle(self.hover, "X")
            if pyxel.btnp(pyxel.MOUSE_BUTTON_RIGHT):
                self.toggle(self.hover, "Z")
        elif pyxel.btnp(pyxel.MOUSE_BUTTON_LEFT) and self.board.all_clear:
            bx, by, bw, bh = self.judge_button_rect()
            if bx <= pyxel.mouse_x < bx + bw and by <= pyxel.mouse_y < by + bh:
                self.press_judge_button()

        self._update_keys()

    def _update_keys(self) -> None:
        d = self.board.d
        r, c = self.cursor
        if pyxel.btnp(pyxel.KEY_LEFT, 10, 3) or pyxel.btnp(pyxel.KEY_H, 10, 3):
            c = (c - 1) % d
        if pyxel.btnp(pyxel.KEY_RIGHT, 10, 3) or pyxel.btnp(pyxel.KEY_L, 10, 3):
            c = (c + 1) % d
        if pyxel.btnp(pyxel.KEY_UP, 10, 3) or pyxel.btnp(pyxel.KEY_K, 10, 3):
            r = (r - 1) % d
        if pyxel.btnp(pyxel.KEY_DOWN, 10, 3) or pyxel.btnp(pyxel.KEY_J, 10, 3):
            r = (r + 1) % d
        self.cursor = (r, c)
        q = self.board.code.qubit_index(r, c)

        ctrl = pyxel.btn(pyxel.KEY_CTRL)
        shift = pyxel.btn(pyxel.KEY_SHIFT)
        if pyxel.btnp(pyxel.KEY_Z):
            self.undo() if ctrl else self.toggle(q, "Z")
        if pyxel.btnp(pyxel.KEY_Y):
            self.redo() if ctrl else self.toggle(q, "Y")
        if pyxel.btnp(pyxel.KEY_X) and not ctrl:
            self.toggle(q, "X")
        if pyxel.btnp(pyxel.KEY_U):
            self.redo() if shift else self.undo()
        if pyxel.btnp(pyxel.KEY_RETURN):
            self.judge()
        if pyxel.btnp(pyxel.KEY_R):
            self.board.reset()
        if pyxel.btnp(pyxel.KEY_N):
            self.new_board(seed=random.getrandbits(RAW_BITS))
        if pyxel.btnp(pyxel.KEY_D):
            self.d_index = (self.d_index + 1) % len(DISTANCES)
            self.new_board(seed=random.getrandbits(RAW_BITS))
        if pyxel.btnp(pyxel.KEY_P):
            self.p_index = (self.p_index + 1) % len(ERROR_RATES)
            self.new_board(seed=random.getrandbits(RAW_BITS))

    def new_board(self, seed: int) -> None:
        d, p = DISTANCES[self.d_index], ERROR_RATES[self.p_index]
        self.board = Board.new(d, p, seed)
        show_code_in_url(pack_seed(d, p, seed))
        self.cursor = (min(self.cursor[0], d - 1), min(self.cursor[1], d - 1))
        # hover was computed against the old board at the top of this frame; drop it so
        # draw() never indexes the new code with a stale qubit number.
        self.hover = None
        self._layout()

    def toggle(self, q: int, kind: Toggle) -> None:
        if self.board.judged:
            pyxel.play(0, 3)
            return
        self.board.toggle(q, kind)
        self.pressed = q
        pyxel.play(0, 0)

    def undo(self) -> None:
        if self.board.undo():
            pyxel.play(0, 0)

    def redo(self) -> None:
        if self.board.redo():
            pyxel.play(0, 0)

    def judge_button_label(self) -> str:
        """JUDGE before the verdict; afterwards NEW on a clean success, RETRY otherwise."""
        v = self.board.verdict
        if v is None:
            return "JUDGE (Enter)"
        return "NEW (n)" if v.success and not v.not_optimal else "RETRY (r)"

    def press_judge_button(self) -> None:
        v = self.board.verdict
        if v is None:
            self.judge()
        elif v.success and not v.not_optimal:
            self.new_board(seed=random.getrandbits(RAW_BITS))
        else:
            self.board.reset()

    def judge(self) -> None:
        if self.board.judged:
            return
        verdict = self.board.judge()
        if verdict is None:
            pyxel.play(0, 3)
            return
        pyxel.play(0, 1 if verdict.success else 2)

    def pauli_for(self, view: int):
        """Operator drawn on the buttons of one board."""
        b = self.board
        return {"C": b.correction, "E": b.error, "R": b.residual, "B": b.bot_correction}[
            VIEWS[view][0]
        ]

    # -- draw -------------------------------------------------------------------------------

    def draw(self) -> None:
        pyxel.cls(BG)
        if self.show_help:
            self.draw_help()
            return
        if self.board.judged:
            for view, lay in enumerate(self.quad):
                self.draw_board(lay, view)
        else:
            self.draw_board(self.play, 0)
        self.draw_panel()

    def draw_board(self, lay: Layout, view: int) -> None:
        """One board showing VIEWS[view]; before judging, view 0 (C) is the live game."""
        board = self.board
        code = board.code
        judged = board.judged
        residual_view = judged and view == RESIDUAL_VIEW
        lit = board.lit[-1]
        if judged and not residual_view:
            # C, E and the bot's correction all share the original syndrome: re-light it so the
            # marks visibly sit between the defects. R is syndrome-free, so it stays dark.
            lit = code.syndrome(board.error)[-1]
        size = (board.d + 1) * lay.cs
        pyxel.rect(lay.ox - 2, lay.oy - 2, size + 4, size + 4, BOARD_BG)

        hover_faces: set[int] = set()
        if not judged and self.hover is not None:
            hover_faces = set(code.faces_of_qubit[self.hover])
        stab_faces = set(int(f) for f in board.residual_faces) if residual_view else set()
        frame = GREEN if board.verdict is not None and board.verdict.success else RED
        pulse = (pyxel.frame_count // 10) % 2 == 0

        for fi, face in enumerate(code.faces):
            x, y, w, h = self.face_rect(face, lay)
            red = tile_is_red(face)
            if fi in stab_faces:
                # R (minus its crossing strings, if any) is a product of these faces: show them
                # as faces, not as an error chain.
                pyxel.rect(x - 1, y - 1, w + 2, h + 2, frame)
                pyxel.rect(x, y, w, h, TILE_OFF)
            elif lit[fi] and board.judged:
                # Calm, dimmed defects so the correction paths drawn over them stand out.
                pyxel.rect(x, y, w, h, X_DIM if red else Z_DIM)
                pyxel.rectb(x, y, w, h, RED if red else BLUE)
            elif lit[fi]:
                glow = X_GLOW if red else Z_GLOW
                g = 2 if pulse else 1
                pyxel.rect(x - g, y - g, w + 2 * g, h + 2 * g, glow)
                pyxel.rect(x, y, w, h, RED if red else BLUE)
                if not pulse:
                    pyxel.rectb(x, y, w, h, X_DIM if red else Z_DIM)
            else:
                pyxel.rect(x, y, w, h, TILE_OFF)
                pyxel.rectb(x, y, w, h, X_EDGE if red else Z_EDGE)
            if fi in hover_faces:
                pyxel.rectb(x - 1, y - 1, w + 2, h + 2, TEXT)

        if residual_view:
            self.draw_residual_paths(lay)

        cursor_q = None if judged else code.qubit_index(*self.cursor)
        pauli = self.pauli_for(view)
        ring = VIEWS[view][1] if judged else None
        for q in range(code.n):
            self.draw_qubit(q, lay, pauli.kind(q), ring, q == cursor_q)

        if judged:
            self.draw_view_banner(lay, view, pauli.weight)

    def draw_qubit(self, q: int, lay: Layout, k: str, ring: int | None, is_cursor: bool) -> None:
        x, y = self.qubit_xy(q, lay)
        r = lay.rq
        sunk = q == self.pressed
        col = BUTTON_HI if (q == self.hover and ring is None and not sunk) else BUTTON
        pyxel.circ(x, y, r - 1 if sunk else r, col)
        if not sunk:
            pyxel.circb(x, y, r, BUTTON_HI)
        m = max(1, r - 2)
        # Each mark is a thick stroke along the diagonal joining the two faces it flips: the
        # checkerboard puts the Z-type (red) faces of qubit (r, c) on the TL-BR diagonal when
        # r + c is even and on the TR-BL diagonal otherwise; the X-type (blue) faces sit on
        # the other one. Y draws both, so it reads as a two-coloured cross.
        r_, c_ = self.board.code.qubit_pos(q)
        red_sign = 1 if (r_ + c_) % 2 == 0 else -1
        if k in ("X", "Y"):
            for dx in (-1, 0, 1):
                pyxel.line(x - m * red_sign + dx, y - m, x + m * red_sign + dx, y + m, RED)
        if k in ("Z", "Y"):
            for dx in (-1, 0, 1):
                pyxel.line(x + m * red_sign + dx, y - m, x - m * red_sign + dx, y + m, BLUE)
        if k != "I" and ring is not None:
            pyxel.circb(x, y, r + 1, ring)
        if is_cursor:
            pyxel.circb(x, y, r + 3, TEXT)

    def draw_residual_paths(self, lay: Layout) -> None:
        """Dashed line along each logical string that R contains (a row or a column)."""
        for path in self.board.residual_paths:
            support = np.flatnonzero(path.x | path.z)
            x0, y0 = self.qubit_xy(int(support[0]), lay)
            x1, y1 = self.qubit_xy(int(support[-1]), lay)
            horizontal = y0 == y1
            length = (x1 - x0) if horizontal else (y1 - y0)
            for t in range(0, length + 1, 6):
                if horizontal:
                    pyxel.rect(x0 + t, y0 - 1, 3, 3, RED)
                else:
                    pyxel.rect(x0 - 1, y0 + t, 3, 3, RED)

    def draw_view_banner(self, lay: Layout, view: int, weight: int) -> None:
        """One-line caption in the strip above a board of the 2x2 grid."""
        key, col, banner = VIEWS[view]
        text = f"{banner} |{key}|={weight}"
        board = self.board
        if view == RESIDUAL_VIEW and board.verdict is not None and not board.residual.is_identity():
            n = len(board.residual_faces)
            faces = f"{n} face{'s' if n != 1 else ''}"
            if board.verdict.success:
                text = f"R = {faces}: harmless"
                col = GREEN
            else:
                kinds = ",".join("X" if p.x.any() else "Z" for p in board.residual_paths)
                paths = f"{kinds} path{'s' if len(board.residual_paths) > 1 else ''}"
                text = f"R = {paths} + {faces}" if n else f"R = {paths}"
                col = RED
        size = (self.board.d + 1) * lay.cs
        pyxel.text(lay.ox + (size - len(text) * 4) // 2, lay.oy - 10, text, col)

    def draw_panel(self) -> None:
        board = self.board
        x = PANEL_X
        line = 6

        def put(s: str, col: int = TEXT, dy: int = 8) -> None:
            nonlocal line
            pyxel.text(x, line, s, col)
            line += dy

        put("SYNDROME OUT", YELLOW, 14)
        put(f"d={board.d}  p={board.p:.2f}  seed {pack_seed(board.d, board.p, board.seed):06X}")
        line += 8
        n_xtype, n_ztype = board.lit_counts
        # Legend: tile colour = the correction that clears it (Z-type faces see X errors).
        for col, count, label in (
            (RED, n_ztype, "X err: put X (L)"),
            (BLUE, n_xtype, "Z err: put Z (R)"),
        ):
            pyxel.rect(x, line, 6, 6, col)
            pyxel.rectb(x - 1, line - 1, 8, 8, col if count else BUTTON)
            pyxel.text(x + 10, line + 1, f"{count:>2} {label}", col if count else BUTTON_HI)
            line += 9

        v = board.verdict
        if v is not None:
            line = VERDICT_Y
            if v.success:
                ml = f" (but not ML: {v.ml_effect.value})" if v.not_ml else ""
                put(f"SUCCESS{ml}", GREEN)
                if v.not_optimal:
                    put("NOT OPTIMAL", YELLOW)
            else:
                ml = " (but ML)" if v.failed_as_ml else ""
                put(f"FAIL {v.effect.value} error{ml}", RED)
            put("")
            bot = "SUCCESS" if v.bot_success else f"FAIL {v.bot_effect.value} error"
            put(f"{BOT_LABEL}: {bot}", GREEN if v.bot_success else RED)

        bx, by, bw, bh = self.judge_button_rect()
        if board.all_clear:
            pyxel.rect(bx, by, bw, bh, BUTTON)
            pyxel.rectb(bx, by, bw, bh, BUTTON_HI)
            label = self.judge_button_label()
            pyxel.text(bx + (bw - len(label) * 4) // 2, by + 4, label, TEXT)
        else:
            pyxel.text(bx, by + 4, "Clear every err", GREEN)

        line = HEIGHT - 7 * 8 - 4
        put("LMB/x: X  RMB/z: Z  y: Y", BUTTON_HI)
        put("arrows/hjkl: cursor", BUTTON_HI)
        put("r: retry   n: new", BUTTON_HI)
        put("u: undo  U: redo", BUTTON_HI)
        put("d: size   p: noise", BUTTON_HI)
        put("?: help  Esc: quit", BUTTON_HI)

    def draw_help(self) -> None:
        y = 6
        for s in HELP_LINES:
            if s is HELP_LINES[-1]:
                col = GREEN
            else:
                col = YELLOW if s.isupper() and s else TEXT
            pyxel.text(8, y, s, col)
            y += 8


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="syndrome-out", description="Syndrome Out")
    ap.add_argument(
        "code",
        nargs="?",
        type=lambda s: int(s, 16),
        metavar="HEX",
        help="the 6-digit seed shown in game; it fixes d, p and the board",
    )
    ap.add_argument("-d", type=int, choices=DISTANCES, help="code distance (default 5)")
    ap.add_argument("-p", type=float, choices=ERROR_RATES, help="error rate (default 0.10)")
    ap.add_argument("--seed", type=int, help="raw RNG seed (random if omitted)")
    args = ap.parse_args(argv)
    if args.code is None:
        d = args.d if args.d is not None else 5
        p = args.p if args.p is not None else 0.10
        seed = args.seed if args.seed is not None else random.getrandbits(RAW_BITS)
        if not 0 <= seed < 1 << RAW_BITS:
            ap.error(f"--seed must be in [0, 2^{RAW_BITS})")
        App(d, p, seed)
        return
    if args.d is not None or args.p is not None or args.seed is not None:
        ap.error("HEX already encodes d, p and the seed; drop -d / -p / --seed")
    try:
        d, p, seed = unpack_seed(args.code)
    except ValueError as e:
        ap.error(str(e))
    App(d, p, seed)


if __name__ == "__main__":
    main()
