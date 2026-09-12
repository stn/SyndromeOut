"""Launcher for `pyxel package` / `pyxel app2exe`.

`pyxel play` puts the startup script's directory on sys.path, so this file must live
next to the `syndrome_out` package (not inside it) for the absolute imports to resolve.
sys.argv still holds the `pyxel play ...` arguments here, so they are not forwarded.
"""

from syndrome_out.app import main

main([])
