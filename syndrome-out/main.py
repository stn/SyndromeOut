"""Launcher for `pyxel package` / `pyxel app2exe` / `pyxel app2html`.

`pyxel play` puts the startup script's directory on sys.path, so this file must live
next to the `syndrome_out` package (not inside it) for the absolute imports to resolve.
sys.argv still holds the `pyxel play ...` arguments here, so they are not forwarded.

In the browser (Pyodide) the page URL plays the role of the command line: a trailing
`/50FCF8` is the positional HEX code and `?d=9&p=0.15&seed=42` are the options.
"""

import re
import sys

from syndrome_out.app import main

HEX_CODE = re.compile(r"[0-9A-Fa-f]{1,6}")


def argv_from_url() -> list[str]:
    from urllib.parse import parse_qs

    import js  # Pyodide runs on the main thread, so the page's window is reachable

    argv: list[str] = []
    segment = str(js.window.location.pathname).rstrip("/").rsplit("/", 1)[-1]
    if HEX_CODE.fullmatch(segment):
        argv.append(segment)
    query = parse_qs(str(js.window.location.search).lstrip("?"))
    for key, flag in (("d", "-d"), ("p", "-p"), ("seed", "--seed")):
        if key in query:
            argv += [flag, query[key][0]]
    return argv


if sys.platform == "emscripten":
    try:
        main(argv_from_url())
    except SystemExit:
        # argparse rejected the URL (message already on stderr); fall back to a fresh board
        main([])
else:
    main([])
