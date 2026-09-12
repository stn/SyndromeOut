"""Pyxel front-end: Lights Out look and feel for the Sweep mode."""

from __future__ import annotations

import argparse
import random

import numpy as np
import pyxel

from syndrome_out.code import Face
from syndrome_out.game import DISTANCES, ERROR_RATES, Board, Toggle

WIDTH, HEIGHT = 360, 240
BOARD_W = 236  # board area on the left, panel on the right
PANEL_X = BOARD_W + 6
MAX_CELL = 28

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
    "most likely class. Lighter is",
    "usually (not always) safer.",
    "",
    "press H to close",
]


# Operators shown on the buttons after judging: (key, ring colour, banner).
VIEWS = (
    ("C", YELLOW, "YOUR CORRECTION C"),
    ("B", BLUE, "BOT CORRECTION (MWPM matching)"),
    ("E", GREEN, "TRUE ERROR E (hidden until now)"),
    ("R", RED, "RESIDUAL R = E*C  (what is left)"),
)
RESIDUAL_VIEW = 3


def tile_is_red(face: Face) -> bool:
    """Colour a tile by the correction that clears it: Z-type faces detect X errors -> red."""
    return face.kind == "Z"


class App:
    def __init__(self, d: int, p: float, seed: int) -> None:
        self.d_index = DISTANCES.index(d)
        self.p_index = ERROR_RATES.index(p)
        self.board = Board.new(d, p, seed)
        self.cursor = (d // 2, d // 2)
        self.view = 0  # index into VIEWS, meaningful once judged
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
        self.cs = min(MAX_CELL, (BOARD_W - 8) // (d + 1))
        size = (d + 1) * self.cs
        self.ox = (BOARD_W - size) // 2
        self.oy = (HEIGHT - size) // 2
        self.rq = max(3, self.cs // 4 - 1)

    def qubit_xy(self, q: int) -> tuple[int, int]:
        r, c = self.board.code.qubit_pos(q)
        return self.ox + (c + 1) * self.cs, self.oy + (r + 1) * self.cs

    def face_xy(self, f: Face) -> tuple[int, int]:
        fr, fc = f.center
        return int(self.ox + (fc + 1) * self.cs), int(self.oy + (fr + 1) * self.cs)

    def face_rect(self, f: Face) -> tuple[int, int, int, int]:
        """(x, y, w, h) of the tile; boundary faces are half tiles hugging the edge."""
        cx, cy = self.face_xy(f)
        hs = self.cs // 2 - 2
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
        hit = self.rq + 3
        for q in range(self.board.code.n):
            x, y = self.qubit_xy(q)
            if abs(mx - x) <= hit and abs(my - y) <= hit:
                return q
        return None

    def judge_button_rect(self) -> tuple[int, int, int, int]:
        return PANEL_X, 74, WIDTH - PANEL_X - 6, 13

    # -- sounds -----------------------------------------------------------------------------

    def _init_sounds(self) -> None:
        pyxel.sounds[0].set("c3", "p", "5", "n", 4)  # click
        pyxel.sounds[1].set("c3e3g3c4", "t", "6", "n", 6)  # success
        pyxel.sounds[2].set("c2c2", "s", "6", "f", 12)  # failure
        pyxel.sounds[3].set("g2", "p", "4", "n", 4)  # denied

    # -- update -----------------------------------------------------------------------------

    def update(self) -> None:
        self.pressed = None
        if pyxel.btnp(pyxel.KEY_H):
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
        elif pyxel.btnp(pyxel.MOUSE_BUTTON_LEFT):
            bx, by, bw, bh = self.judge_button_rect()
            if bx <= pyxel.mouse_x < bx + bw and by <= pyxel.mouse_y < by + bh:
                self.judge()

        self._update_keys()

    def _update_keys(self) -> None:
        d = self.board.d
        r, c = self.cursor
        if pyxel.btnp(pyxel.KEY_LEFT, 10, 3):
            c = (c - 1) % d
        if pyxel.btnp(pyxel.KEY_RIGHT, 10, 3):
            c = (c + 1) % d
        if pyxel.btnp(pyxel.KEY_UP, 10, 3):
            r = (r - 1) % d
        if pyxel.btnp(pyxel.KEY_DOWN, 10, 3):
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
            self.view = 0
        if pyxel.btnp(pyxel.KEY_N):
            self.new_board(seed=random.randrange(1_000_000))
        if pyxel.btnp(pyxel.KEY_A) and self.board.judged:
            self.view = (self.view + 1) % len(VIEWS)
        if pyxel.btnp(pyxel.KEY_D):
            self.d_index = (self.d_index + 1) % len(DISTANCES)
            self.new_board(seed=random.randrange(1_000_000))
        if pyxel.btnp(pyxel.KEY_P):
            self.p_index = (self.p_index + 1) % len(ERROR_RATES)
            self.new_board(seed=random.randrange(1_000_000))

    def new_board(self, seed: int) -> None:
        d, p = DISTANCES[self.d_index], ERROR_RATES[self.p_index]
        self.board = Board.new(d, p, seed)
        self.cursor = (min(self.cursor[0], d - 1), min(self.cursor[1], d - 1))
        self.view = 0
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

    def judge(self) -> None:
        if self.board.judged:
            return
        verdict = self.board.judge()
        if verdict is None:
            pyxel.play(0, 3)
            return
        pyxel.play(0, 1 if verdict.success else 2)
        self.view = 0 if verdict.success else RESIDUAL_VIEW

    def harmless_residual_faces(self) -> set[int]:
        """Faces whose product is R, when R is a stabilizer (i.e. the player succeeded)."""
        v = self.board.verdict
        if v is None or not v.success or self.view != RESIDUAL_VIEW:
            return set()
        faces = self.board.code.stabilizer_faces(self.board.residual)
        return set() if faces is None else set(int(f) for f in faces)

    def shown_pauli(self):
        """Operator drawn on the buttons: the correction while playing, the chosen view after."""
        if not self.board.judged:
            return self.board.correction
        b = self.board
        return {"C": b.correction, "E": b.error, "R": b.residual, "B": b.bot_correction}[
            VIEWS[self.view][0]
        ]

    # -- draw -------------------------------------------------------------------------------

    def draw(self) -> None:
        pyxel.cls(BG)
        if self.show_help:
            self.draw_help()
            return
        self.draw_board()
        self.draw_panel()

    def draw_board(self) -> None:
        board = self.board
        code = board.code
        lit = board.lit[-1]
        if board.judged and self.view != RESIDUAL_VIEW:
            # C, E and the bot's correction all share the original syndrome: re-light it so the
            # paths visibly connect the defects. R is syndrome-free, so it stays dark.
            lit = code.syndrome(board.error)[-1]
        size = (board.d + 1) * self.cs
        pyxel.rect(self.ox - 2, self.oy - 2, size + 4, size + 4, BOARD_BG)

        hover_faces = set(code.faces_of_qubit[self.hover]) if self.hover is not None else set()
        stab_faces = self.harmless_residual_faces()
        pulse = (pyxel.frame_count // 10) % 2 == 0

        for fi, face in enumerate(code.faces):
            x, y, w, h = self.face_rect(face)
            red = tile_is_red(face)
            if fi in stab_faces:
                # R is a product of these faces: show them as faces, not as an error chain.
                pyxel.rect(x - 1, y - 1, w + 2, h + 2, GREEN)
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

        if board.judged:
            if not stab_faces:
                self.draw_pauli_paths(self.shown_pauli())
            self.draw_logical_paths()

        cursor_q = code.qubit_index(*self.cursor)
        for q in range(code.n):
            self.draw_qubit(q, q == cursor_q)

        if board.judged:
            self.draw_view_banner()

    def draw_qubit(self, q: int, is_cursor: bool) -> None:
        x, y = self.qubit_xy(q)
        r = self.rq
        sunk = q == self.pressed
        col = BUTTON_HI if (q == self.hover and not sunk) else BUTTON
        pyxel.circ(x, y, r - 1 if sunk else r, col)
        if not sunk:
            pyxel.circb(x, y, r, BUTTON_HI)
        k = self.shown_pauli().kind(q)
        m = max(1, r - 2)
        if k in ("X", "Y"):
            pyxel.line(x - m, y - m, x + m, y + m, RED)
            pyxel.line(x - m, y + m, x + m, y - m, RED)
        if k in ("Z", "Y"):
            pyxel.rect(x - m, y - 1, 2 * m + 1, 3, Z_LIT)
        if k != "I" and self.board.judged:
            pyxel.circb(x, y, r + 1, VIEWS[self.view][1])
        if is_cursor:
            pyxel.circb(x, y, r + 3, TEXT)

    def draw_logical_paths(self) -> None:
        """Dashed line along each logical operator the residual anticommutes with."""
        for logical in self.board.crossing_logicals():
            support = np.flatnonzero(logical.x | logical.z)
            x0, y0 = self.qubit_xy(int(support[0]))
            x1, y1 = self.qubit_xy(int(support[-1]))
            horizontal = y0 == y1
            length = (x1 - x0) if horizontal else (y1 - y0)
            for t in range(0, length + 1, 6):
                if horizontal:
                    pyxel.rect(x0 + t, y0 - 1, 3, 3, TEXT)
                else:
                    pyxel.rect(x0 - 1, y0 + t, 3, 3, TEXT)

    def draw_pauli_paths(self, pauli) -> None:
        """Draw a Pauli as chains: each X (Z) on a qubit is an edge between the two Z-type
        (X-type) faces that contain it, or a stub out to the boundary when there is only one."""
        code = self.board.code
        for q in range(code.n):
            qx, qy = self.qubit_xy(q)
            for part, kind, col in ((pauli.x, "Z", RED), (pauli.z, "X", BLUE)):
                if not part[q]:
                    continue
                ends = [
                    self.face_xy(code.faces[f])
                    for f in code.faces_of_qubit[q]
                    if code.faces[f].kind == kind
                ]
                if len(ends) == 1:
                    ((fx, fy),) = ends
                    ends.append((qx + (qx - fx) // 2, qy + (qy - fy) // 2))
                (x0, y0), (x1, y1) = ends
                # Dark outline first so the path reads on top of a lit tile of the same colour.
                for dx in (-1, 0, 1, 2):
                    for dy in (-1, 0, 1, 2):
                        pyxel.line(x0 + dx, y0 + dy, x1 + dx, y1 + dy, BG)
                for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
                    pyxel.line(x0 + dx, y0 + dy, x1 + dx, y1 + dy, col)

    def draw_view_banner(self) -> None:
        _, col, banner = VIEWS[self.view]
        shown = self.shown_pauli()
        text = f"{banner}  |{VIEWS[self.view][0]}|={shown.weight}"
        faces = self.harmless_residual_faces()
        if faces:
            n = len(faces)
            text = f"R = E*C = product of {n} stabilizer face{'s' if n != 1 else ''}: harmless"
            col = GREEN
        pyxel.rect(0, 0, BOARD_W, 9, BOARD_BG)
        pyxel.text(4, 2, text, col)

    def draw_panel(self) -> None:
        board = self.board
        x = PANEL_X
        line = 6

        def put(s: str, col: int = TEXT, dy: int = 8) -> None:
            nonlocal line
            pyxel.text(x, line, s, col)
            line += dy

        put("SYNDROME OUT", YELLOW, 10)
        put(f"d={board.d}  p={board.p:.2f}")
        put(f"seed {board.seed}")
        line += 4
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
        put(f"|C| = {board.correction.weight}")
        line += 4
        line += 10

        bx, by, bw, bh = self.judge_button_rect()
        enabled = board.all_clear and not board.judged
        pyxel.rect(bx, by, bw, bh, BUTTON if enabled else BOARD_BG)
        pyxel.rectb(bx, by, bw, bh, BUTTON_HI if enabled else BUTTON)
        label = "JUDGE (Enter)" if not board.judged else "JUDGED"
        pyxel.text(bx + (bw - len(label) * 4) // 2, by + 4, label, TEXT if enabled else BUTTON_HI)
        line = by + bh + 8

        v = board.verdict
        if v is not None:
            if v.success:
                put("SUCCESS  class I", GREEN)
                if v.optimal:
                    put("OPTIMAL weight", YELLOW)
            else:
                bar = v.effect.value
                put(f"FAIL  class {bar}-bar", RED)
                put(f"logical {bar} error", RED)
                put("all dark, yet wrong:", BUTTON_HI)
                put("E*C is a logical op", BUTTON_HI)
            put(f"|C|={v.weight}  bot |C|={v.bot_weight}")
            bot = "OK" if v.bot_success else f"FAIL ({v.bot_effect.value}-bar)"
            put(f"bot (MWPM): {bot}", GREEN if v.bot_success else RED)
            put(f"score {v.score}", YELLOW)
            if v.beat_bot:
                put("YOU BEAT THE BOT!", YELLOW)
            key, col, _ = VIEWS[self.view]
            name = {"C": "your C", "E": "error E", "R": "residual R", "B": "bot C"}[key]
            put(f"[A] view: {name}", col)
        elif board.all_clear:
            put("all dark. judge?", GREEN)
        else:
            put("clear every tile", BUTTON_HI)

        line = HEIGHT - 8 * 8 - 4
        put("LMB/x: X  RMB/z: Z  y: Y", BUTTON_HI)
        put("arrows: cursor", BUTTON_HI)
        put("Enter: judge", BUTTON_HI)
        put("r: retry   n: new", BUTTON_HI)
        put("u: undo  U: redo", BUTTON_HI)
        put("d: size   p: noise", BUTTON_HI)
        put("h: help  Esc: quit", BUTTON_HI)

    def draw_help(self) -> None:
        y = 6
        for s in HELP_LINES:
            col = YELLOW if s.isupper() and s else TEXT
            pyxel.text(8, y, s, col)
            y += 8


def main() -> None:
    ap = argparse.ArgumentParser(prog="syndrome-out", description="Syndrome Out")
    ap.add_argument("-d", type=int, default=5, choices=DISTANCES, help="code distance")
    ap.add_argument("-p", type=float, default=0.10, choices=ERROR_RATES, help="error rate")
    ap.add_argument("--seed", type=int, default=None, help="board seed (random if omitted)")
    args = ap.parse_args()
    seed = args.seed if args.seed is not None else random.randrange(1_000_000)
    App(args.d, args.p, seed)


if __name__ == "__main__":
    main()
