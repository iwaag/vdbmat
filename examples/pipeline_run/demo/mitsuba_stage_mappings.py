"""Optical-mapping catalog for the Mitsuba stage viewer.

The catalog is rooted at a server-local ``--mapping-root`` and exposes only
``*.optical-mapping.json`` files whose resolved targets stay inside that
root.  Catalog enumeration is intentionally cheap: JSON parsing and
``OpticalMappingConfig`` validation are deferred until a selected mapping is
described or loaded, so an external edit is picked up by re-describing the
same catalog entry rather than by re-scanning.

This module imports neither Mitsuba nor viser.  Parsing and validation are
delegated entirely to :func:`vdbmat.optics.load_optical_mapping`, so the
mapping schema, palette rules, and digest are never redefined here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from vdbmat.optics import (
    OpticalMappingConfig,
    OpticalMappingError,
    load_optical_mapping,
)

_MAPPING_SUFFIX = ".optical-mapping.json"
_BUILTIN_MAPPING_DIR = (
    Path(__file__).resolve().parents[2] / "pipeline_run" / "mappings"
)


class MappingCatalogError(Exception):
    """A mapping root, candidate, or document is invalid."""


@dataclass(frozen=True, slots=True)
class MappingCandidate:
    """One contained ``*.optical-mapping.json`` file in the catalog."""

    root_relative: str
    path: Path


@dataclass(frozen=True, slots=True)
class MappingSummary:
    """Display-only summary of a parsed and validated optical mapping."""

    configuration_id: str
    version: str
    calibration_status: str
    digest: str
    materials: tuple[tuple[int, str], ...]


def resolve_mapping_root(cli_root: Path | None) -> Path:
    """Resolve the explicit root, or the checked-in demo mappings directory.

    Unlike the input and preset roots, a mapping root has no "initial
    selection" fallback: mapping selection is optional on every Load/Rebuild,
    so there is no initial mapping to take a parent directory from.
    """
    if cli_root is not None:
        return _require_directory(cli_root, option="--mapping-root")
    return _require_directory(_BUILTIN_MAPPING_DIR, option="builtin mapping root")


def scan_mapping_catalog(root: Path) -> list[MappingCandidate]:
    """Return contained mapping files in deterministic relative-path order.

    A file only needs the expected suffix to appear in the catalog. Invalid
    JSON and invalid mapping documents remain visible so describing or
    applying the selected entry can report the actual error to the user.
    """
    resolved_root = _require_directory(root, option="--mapping-root")
    found: list[MappingCandidate] = []
    for dirpath, dirnames, filenames in os.walk(resolved_root):
        current = Path(dirpath)
        dirnames[:] = sorted(
            name for name in dirnames if _is_contained(resolved_root, current / name)
        )
        for name in sorted(filenames):
            if not name.endswith(_MAPPING_SUFFIX):
                continue
            lexical_path = current / name
            if not _is_contained(resolved_root, lexical_path):
                continue
            resolved_path = lexical_path.resolve()
            if not resolved_path.is_file():
                continue
            found.append(
                MappingCandidate(
                    root_relative=lexical_path.relative_to(resolved_root).as_posix(),
                    path=resolved_path,
                )
            )
    found.sort(key=lambda item: item.root_relative)
    return found


def resolve_mapping_candidate(root: Path, user_path: Path) -> MappingCandidate:
    """Resolve one root-relative mapping under strict lexical containment.

    GUI selections are catalog keys, not arbitrary server paths. Absolute
    paths and parent traversal are rejected even when they would happen to
    resolve inside the root; symlinks are accepted only when their final
    target remains inside the resolved root.
    """
    resolved_root = _require_directory(root, option="--mapping-root")
    if user_path.is_absolute():
        raise MappingCatalogError(f"mapping path must be relative: {user_path}")
    if not user_path.parts or user_path == Path("."):
        raise MappingCatalogError("mapping path must not be empty")
    if ".." in user_path.parts:
        raise MappingCatalogError(
            f"mapping path must not contain parent traversal: {user_path}"
        )
    if not user_path.name.endswith(_MAPPING_SUFFIX):
        raise MappingCatalogError(
            f"mapping path must end with {_MAPPING_SUFFIX!r}: {user_path}"
        )

    lexical_path = resolved_root / user_path
    if not lexical_path.exists():
        raise MappingCatalogError(f"mapping does not exist: {user_path}")
    try:
        resolved_path = lexical_path.resolve()
    except OSError as error:
        raise MappingCatalogError(
            f"cannot resolve mapping {user_path}: {error}"
        ) from error
    if not _is_contained(resolved_root, resolved_path):
        raise MappingCatalogError(
            f"mapping resolves outside --mapping-root: {user_path}"
        )
    if not resolved_path.is_file():
        raise MappingCatalogError(f"mapping is not a file: {user_path}")
    return MappingCandidate(root_relative=user_path.as_posix(), path=resolved_path)


def load_mapping(candidate: MappingCandidate) -> OpticalMappingConfig:
    """Parse and validate one selected optical mapping document."""
    try:
        return load_optical_mapping(candidate.path)
    except OpticalMappingError as error:
        raise MappingCatalogError(
            f"invalid optical mapping {candidate.root_relative}: {error}"
        ) from error


def describe_mapping(candidate: MappingCandidate) -> MappingSummary:
    """Parse a mapping and return the fields needed by a GUI summary."""
    config = load_mapping(candidate)
    return MappingSummary(
        configuration_id=config.configuration_id,
        version=str(config.version),
        calibration_status=config.calibration_status.value,
        digest=config.digest,
        materials=tuple((item.material_id, item.name) for item in config.materials),
    )


def _require_directory(path: Path, *, option: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise MappingCatalogError(f"{option} is not a directory: {path}")
    return resolved


def _is_contained(root: Path, path: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return resolved == root or resolved.is_relative_to(root)


__all__ = [
    "MappingCandidate",
    "MappingCatalogError",
    "MappingSummary",
    "describe_mapping",
    "load_mapping",
    "resolve_mapping_candidate",
    "resolve_mapping_root",
    "scan_mapping_catalog",
]
