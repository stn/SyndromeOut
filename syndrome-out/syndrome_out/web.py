"""Browser-only glue: in the Pyodide build the page URL doubles as the command line.

A trailing `/50FCF8` is the positional HEX code and `?d=9&p=0.15&seed=42` are the options
(see main.py). Whenever a board is created the URL is rewritten to `/<HEX>` so the address
bar always names the current board and can be shared. Everything is a no-op outside Pyodide.
"""

from __future__ import annotations

import re
import sys
from urllib.parse import parse_qs

IN_BROWSER = sys.platform == "emscripten"
HEX_CODE = re.compile(r"[0-9A-Fa-f]{1,6}")


def argv_from_url() -> list[str]:
    import js  # Pyodide runs on the main thread, so the page's window is reachable

    argv: list[str] = []
    _, last = _split_path(str(js.window.location.pathname))
    if HEX_CODE.fullmatch(last):
        argv.append(last)
    query = parse_qs(str(js.window.location.search).lstrip("?"))
    for key, flag in (("d", "-d"), ("p", "-p"), ("seed", "--seed")):
        if key in query:
            argv += [flag, query[key][0]]
    return argv


def show_code_in_url(code: int) -> None:
    if not IN_BROWSER:
        return
    import js

    base, last = _split_path(str(js.window.location.pathname))
    if not (HEX_CODE.fullmatch(last) or last.endswith(".html")):
        base = f"{base}/{last}" if last else base
    js.window.history.replaceState(None, "", f"{base}/{code:06X}")


def _split_path(pathname: str) -> tuple[str, str]:
    """`/SyndromeOut/50FCF8/` -> ("/SyndromeOut", "50FCF8"); `/` -> ("", "")."""
    base, _, last = pathname.rstrip("/").rpartition("/")
    return base, last
