"""Schema identity and reproducibility metadata."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

_SEMANTIC_VERSION = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_SHA256_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True, order=True)
class SchemaVersion:
    """A release-free semantic version for a persisted schema."""

    major: int
    minor: int
    patch: int

    def __post_init__(self) -> None:
        for field in ("major", "minor", "patch"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"schema version {field} must be an integer")
            if value < 0:
                raise ValueError(f"schema version {field} must not be negative")

    @classmethod
    def parse(cls, value: str) -> "SchemaVersion":
        """Parse strict ``MAJOR.MINOR.PATCH`` schema notation."""
        if not isinstance(value, str):
            raise TypeError("schema version must be a string")
        match = _SEMANTIC_VERSION.fullmatch(value)
        if match is None:
            raise ValueError("schema version must use MAJOR.MINOR.PATCH")
        return cls(*(int(item) for item in match.groups()))

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def has_compatible_major(self, other: "SchemaVersion") -> bool:
        """Return whether two versions use the same major schema contract."""
        return self.major == other.major


@dataclass(frozen=True, slots=True)
class SchemaIdentity:
    """Stable schema name paired with a semantic version."""

    name: str
    version: SchemaVersion

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("schema name must be a non-empty string")
        if not isinstance(self.version, SchemaVersion):
            raise TypeError("schema version must be a SchemaVersion")


@dataclass(frozen=True, slots=True)
class Provenance:
    """Minimum reproducibility metadata shared by canonical assets."""

    generator: str
    generator_version: str
    created_utc: datetime | None = None
    configuration_digest: str | None = None
    sources: tuple[str, ...] = ()
    notes: str | None = None

    def __post_init__(self) -> None:
        for field in ("generator", "generator_version"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"provenance {field} must be a non-empty string")

        if self.created_utc is not None:
            if not isinstance(self.created_utc, datetime):
                raise TypeError("created_utc must be a datetime or None")
            offset = self.created_utc.utcoffset()
            if offset is None:
                raise ValueError("created_utc must be timezone-aware")
            if offset.total_seconds() != 0:
                raise ValueError("created_utc must use UTC")

        if (
            self.configuration_digest is not None
            and _SHA256_DIGEST.fullmatch(self.configuration_digest) is None
        ):
            raise ValueError(
                "configuration_digest must be 'sha256:' plus 64 lowercase hex digits"
            )

        sources = _normalize_sources(self.sources)
        object.__setattr__(self, "sources", sources)

        if self.notes is not None and not isinstance(self.notes, str):
            raise TypeError("provenance notes must be a string or None")


VOLUME_SCHEMA = SchemaIdentity(
    name="vbdmat.volume", version=SchemaVersion(major=1, minor=0, patch=0)
)


def _normalize_sources(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        raise TypeError(
            "provenance sources must be a sequence of strings, not a string"
        )
    sources = tuple(value)
    for index, source in enumerate(sources):
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"provenance source {index} must be a non-empty string")
    return sources
