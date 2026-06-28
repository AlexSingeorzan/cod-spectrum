"""Vercel Python (ASGI) entrypoint for cod-spectrum.

Vercel's filesystem is read-only except for /tmp, so we redirect the SQLite
database and the data directory there before importing the app. Note: /tmp is
ephemeral per invocation, so persisted data does not survive cold starts.
"""

from __future__ import annotations

import os

os.environ.setdefault("COD_SPECTRUM_DATABASE_URL", "sqlite:////tmp/cod_spectrum.db")
os.environ.setdefault("COD_SPECTRUM_DATA_DIR", "/tmp/cod_spectrum_data")

from backend.app.main import app  # noqa: E402

__all__ = ["app"]
