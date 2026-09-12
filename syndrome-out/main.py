"""Launcher for `pyxel package` / `pyxel app2exe` / `pyxel app2html`.

`pyxel play` puts the startup script's directory on sys.path, so this file must live
next to the `syndrome_out` package (not inside it) for the absolute imports to resolve.
sys.argv still holds the `pyxel play ...` arguments here, so they are not forwarded.

In the browser the page URL plays the role of the command line; see syndrome_out/web.py.
"""

from syndrome_out.app import main
from syndrome_out.web import IN_BROWSER, argv_from_url

if IN_BROWSER:
    try:
        main(argv_from_url())
    except SystemExit:
        # argparse rejected the URL (message already on stderr); fall back to a fresh board
        main([])
else:
    main([])
