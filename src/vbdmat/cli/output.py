"""Stable JSON and concise human output for the Phase 1 CLI."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def json_line(document: Mapping[str, Any]) -> str:
    """Serialize one deterministic, newline-terminated JSON document."""
    return json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"


def human_summary(document: Mapping[str, Any]) -> str:
    """Return a compact summary without duplicating the machine document."""
    status = str(document.get("status", "ok"))
    operation = str(document.get("operation", "vbdmat"))
    path = document.get("path") or document.get("output_path")
    suffix = f": {path}" if path is not None else ""
    return f"{operation}: {status}{suffix}\n"
