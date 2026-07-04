"""Common machine-readable capability diagnostics for renderer adapters."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from vbdmat.boundaries import CapabilityStatus


@dataclass(frozen=True, slots=True)
class CapabilityEntry:
    """Disposition and concrete mapping for one canonical or derived property."""

    field: str
    status: CapabilityStatus
    mapping: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        """Return a stable JSON-compatible representation."""
        return {
            "field": self.field,
            "status": self.status.value,
            "mapping": self.mapping,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class CapabilityReport:
    """Complete adapter capability record for one exported canonical asset."""

    consumer: str
    adapter: str
    adapter_version: str
    schema_name: str
    schema_version: str
    entries: tuple[CapabilityEntry, ...]

    def __post_init__(self) -> None:
        fields = tuple(entry.field for entry in self.entries)
        if len(fields) != len(set(fields)):
            raise ValueError("capability report fields must be unique")

    def by_field(self, field: str) -> CapabilityEntry:
        """Return the entry for ``field`` or raise ``KeyError``."""
        for entry in self.entries:
            if entry.field == field:
                return entry
        raise KeyError(field)

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible representation."""
        return {
            "consumer": self.consumer,
            "adapter": self.adapter,
            "adapter_version": self.adapter_version,
            "schema": {"name": self.schema_name, "version": self.schema_version},
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def write_json(self, path: str | Path) -> None:
        """Write deterministic indented JSON with a trailing newline."""
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
