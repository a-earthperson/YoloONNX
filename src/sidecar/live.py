from __future__ import annotations

from importlib.resources import files

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

_LIVE_HTML = (
    files("sidecar").joinpath("static", "live.html").read_text(encoding="utf-8")
)


def install_live_sidecar(app: FastAPI) -> None:
    if getattr(app.state, "sidecar_installed", False):
        return

    app.mount(
        "/live/assets",
        StaticFiles(packages=[("sidecar", "static")]),
        name="sidecar-live-assets",
    )

    @app.get("/live", include_in_schema=False)
    @app.get("/live/", include_in_schema=False)
    def live() -> HTMLResponse:
        return HTMLResponse(_LIVE_HTML)

    app.state.sidecar_installed = True
