# Syndrome Out

Lights Out for the rotated surface code. Clear every lit stabilizer face by placing
X / Z corrections on data qubits, then find out whether your correction caused a
logical error. See `syndrome-out-spec.md` for the design document.

```
uv sync
uv run syndrome-out            # play (d=5, p=0.10 by default)
uv run syndrome-out -d 7 -p 0.05 --seed 42
uv run pytest                  # acceptance tests
uv run scripts/threshold_sim.py
```

## Controls

A red tile means an X error sits next to it: clear it with X (red mark). A blue tile means
a Z error: clear it with Z (blue mark). The panel legend shows how many of each are lit.

| Input | Action |
|---|---|
| left click / `x` | toggle X on the qubit under the mouse / cursor |
| right click / `z` | toggle Z |
| `y` | toggle Y (X and Z) |
| arrows | move the keyboard cursor |
| Enter or JUDGE button | judge (only when every tile is dark) |
| `a` | after judging, cycle what the board shows, drawn as chains between faces: your C (yellow), the MWPM bot's C (blue), true error E (green), residual R = E*C (red) |
| `u` / `U` (or Ctrl+Z / Ctrl+Y) | undo / redo |
| `r` / `n` | retry the same seed / new random seed |
| `d` / `p` | cycle distance 3-5-7-9 / error rate 0.05-0.10-0.15 |
| `h` | help, including the difference from Lights Out |

## "All dark, yet wrong"

The whole point of the game is that clearing the board is not the same as succeeding.
A reproducible example: `uv run syndrome-out -d 5 -p 0.10 --seed 50`. The minimum-weight
correction (what the MWPM bot plays, weight 4) turns every tile off and still produces a
logical X error, because the hidden error has weight 3 and E*C is a logical operator.
`tests/test_game.py::test_demo_seed_minimum_weight_fails` pins this down.

Licensed under MIT OR Apache-2.0.
