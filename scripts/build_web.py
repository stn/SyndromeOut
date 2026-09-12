"""Build the browser version: pyxel package -> pyxel app2html -> add numpy to the page.

Usage: uv run scripts/build_web.py [-o dist/web]

`pyxel app2html` embeds the .pyxapp in a page that boots Pyodide, but its template does not
load any extra packages. Pyxel Web still honours a `packages` launch parameter (a list of
packages built into Pyodide), so the page is patched to request numpy, the only dependency
besides pyxel itself.

The page is written twice, as index.html and 404.html: GitHub Pages serves 404.html for any
unknown path, which is what lets `/SyndromeOut/50FCF8` boot the game with that seed (main.py
reads the URL). Nothing in the page is fetched relative to its own URL, so this is safe.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = "syndrome-out"
NEEDLE = 'launchPyxel({ command: "play", '


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "dist" / "web")
    args = ap.parse_args()

    pyxel = [sys.executable, "-m", "pyxel"]
    subprocess.run([*pyxel, "package", APP, f"{APP}/main.py"], cwd=ROOT, check=True)
    subprocess.run([*pyxel, "app2html", f"{APP}.pyxapp"], cwd=ROOT, check=True)

    html_file = ROOT / f"{APP}.html"
    html = html_file.read_text(encoding="utf-8")
    if html.count(NEEDLE) != 1:
        raise SystemExit(f"unexpected app2html output: {NEEDLE!r} not found exactly once")
    html = html.replace(NEEDLE, NEEDLE + 'packages: "numpy", ')
    html_file.unlink()

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "index.html"
    out.write_text(html, encoding="utf-8")
    (args.out / "404.html").write_text(html, encoding="utf-8")
    print(
        f"wrote {out} and 404.html ({out.stat().st_size / 1024:.0f} KB each); "
        f"serve with: python -m http.server -d {args.out}"
    )


if __name__ == "__main__":
    main()
