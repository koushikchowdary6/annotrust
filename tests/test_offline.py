"""The library does not import a network client."""

from __future__ import annotations

from pathlib import Path


def test_sources_do_not_import_network_clients() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "annotrust"
    banned = (
        "import requests",
        "import httpx",
        "import urllib",
        "from urllib",
        "import socket",
        "urlopen(",
        "http.client",
    )
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in banned:
            assert token not in text, f"{path.name} contains {token}"
