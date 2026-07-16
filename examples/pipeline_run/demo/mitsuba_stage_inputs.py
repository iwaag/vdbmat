"""Input catalog for the Mitsuba stage viewer.

Scans a server-local ``--input-root`` directory for canonical run bundles
(directories containing ``run.json`` and ``optical.zarr``) and standalone
``optical.zarr`` stores, and describes a selected candidate for display in
the GUI Input tab.

This module intentionally has no dependency on ``viser`` or ``mitsuba``: it
is pure filesystem/manifest logic so it can be unit tested without either
and reused by any future non-GUI entry point. It never reads voxel array
data — enumeration and description use directory names and Zarr manifest
attributes only. Full validation (``read_volume``) is deferred to the
Load/Rebuild transaction that consumes a resolved candidate.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import zarr

from vdbmat.core.volumes import VolumeAssetType
from vdbmat.io import inspect_volume

_RUN_MANIFEST_NAME = "run.json"
_OPTICAL_ASSET_NAME = "optical.zarr"
_MANIFEST_ATTRIBUTE = "vdbmat"


class InputKind(StrEnum):
    """The two catalog input shapes defined by the Phase 2 plan."""

    RUN_BUNDLE = "run-bundle"
    OPTICAL_ZARR = "optical-zarr"


class InputCatalogError(Exception):
    """A candidate could not be resolved or described."""


@dataclass(frozen=True, slots=True)
class InputCandidate:
    """One catalog entry: a run bundle or a standalone optical volume."""

    kind: InputKind
    root_relative: str
    path: Path
    optical_zarr: Path


@dataclass(frozen=True, slots=True)
class InputSummary:
    """Display-only description of a resolved candidate."""

    kind: InputKind
    schema_name: str
    schema_version: str
    shape_zyx: tuple[int, int, int]
    voxel_size_xyz_m: tuple[float, float, float]
    run_id: str | None
    provenance_sources: tuple[str, ...]
    provenance_notes: str | None


def resolve_input_root(cli_root: Path | None, initial_input: Path) -> Path:
    """Return the catalog root: the explicit ``--input-root``, or a default.

    The default is the parent directory of the initial input (the bundle
    root if the initial input is a bundle, otherwise the parent of the
    ``optical.zarr``), so existing single-input command lines keep working
    while gaining sibling-input browsing for free.
    """
    if cli_root is not None:
        resolved = cli_root.resolve()
        if not resolved.is_dir():
            raise InputCatalogError(f"--input-root is not a directory: {cli_root}")
        return resolved

    return initial_input.resolve().parent


def _is_run_bundle_dir(path: Path) -> bool:
    return (path / _RUN_MANIFEST_NAME).is_file() and (
        path / _OPTICAL_ASSET_NAME
    ).exists()


def _is_optical_zarr_dir(path: Path) -> bool:
    if not path.name.endswith(".zarr"):
        return False
    manifest = _read_manifest_attrs(path)
    if manifest is None:
        return False
    return manifest.get("asset_type") == VolumeAssetType.OPTICAL_PROPERTY.value


def _read_manifest_attrs(path: Path) -> dict[str, Any] | None:
    """Read the raw ``vdbmat`` manifest attrs without validating arrays.

    Enumeration only needs the asset type declared in the manifest, not the
    full ``inspect_volume`` array-declaration check, so a directory that
    merely looks like a Zarr store but is malformed in some other way is
    excluded here rather than raised. Anything wrong beyond asset type
    surfaces later, from ``describe_candidate`` or Load/Rebuild.
    """
    try:
        root = zarr.open_group(path, mode="r")
        manifest = root.attrs.get(_MANIFEST_ATTRIBUTE)
    except Exception:
        return None
    return manifest if isinstance(manifest, dict) else None


def _candidate_for(root: Path, path: Path) -> InputCandidate | None:
    if _is_run_bundle_dir(path):
        return InputCandidate(
            kind=InputKind.RUN_BUNDLE,
            root_relative=_root_relative(root, path),
            path=path,
            optical_zarr=path / _OPTICAL_ASSET_NAME,
        )
    if _is_optical_zarr_dir(path):
        return InputCandidate(
            kind=InputKind.OPTICAL_ZARR,
            root_relative=_root_relative(root, path),
            path=path,
            optical_zarr=path,
        )
    return None


def _root_relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _is_contained(root: Path, path: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return resolved == root or resolved.is_relative_to(root)


def scan_input_catalog(root: Path) -> list[InputCandidate]:
    """Enumerate bundles and standalone optical volumes under ``root``.

    Directories recognized as a bundle or an optical store are not
    descended into (Zarr chunk files and bundle-internal stores are never
    walked). A candidate whose resolved real path escapes ``root`` (a
    root-outer symlink) is silently excluded, matching the containment
    rule enforced for direct candidate resolution.
    """
    root = root.resolve()
    found: list[InputCandidate] = []
    for dirpath, dirnames, _filenames in os.walk(root):
        current = Path(dirpath)
        dirnames.sort()
        keep: list[str] = []
        for name in dirnames:
            child = current / name
            if not _is_contained(root, child):
                continue
            candidate = _candidate_for(root, child)
            if candidate is not None:
                found.append(candidate)
                continue
            keep.append(name)
        dirnames[:] = keep
    found.sort(key=lambda item: item.root_relative)
    return found


def resolve_candidate(root: Path, user_path: Path) -> InputCandidate:
    """Resolve a directly specified path under the same containment rule.

    Raises :class:`InputCatalogError` for a path outside ``root`` (whether
    directly or via symlink resolution), a nonexistent path, or a path that
    is neither a run bundle nor a standalone optical volume.
    """
    root = root.resolve()
    candidate_path = user_path if user_path.is_absolute() else root / user_path
    if not candidate_path.exists():
        raise InputCatalogError(f"input does not exist: {user_path}")
    resolved = candidate_path.resolve()
    if not _is_contained(root, resolved):
        raise InputCatalogError(f"input resolves outside --input-root: {user_path}")
    candidate = _candidate_for(root, resolved)
    if candidate is None:
        raise InputCatalogError(f"not a run bundle or optical.zarr: {user_path}")
    return candidate


def describe_candidate(candidate: InputCandidate) -> InputSummary:
    """Return a display-only summary without reading voxel array data."""
    inspection = inspect_volume(candidate.optical_zarr)
    provenance = _read_provenance_attrs(candidate.optical_zarr)
    run_id = None
    if candidate.kind is InputKind.RUN_BUNDLE:
        run_id = _read_run_id(candidate.path)
    return InputSummary(
        kind=candidate.kind,
        schema_name=inspection.schema_name,
        schema_version=inspection.schema_version,
        shape_zyx=inspection.geometry.shape_zyx,
        voxel_size_xyz_m=inspection.geometry.voxel_size_xyz_m,
        run_id=run_id,
        provenance_sources=provenance[0],
        provenance_notes=provenance[1],
    )


def _read_provenance_attrs(
    optical_zarr: Path,
) -> tuple[tuple[str, ...], str | None]:
    root = zarr.open_group(optical_zarr, mode="r")
    manifest: Any = root.attrs.get(_MANIFEST_ATTRIBUTE, {})
    provenance = manifest.get("provenance", {}) if isinstance(manifest, dict) else {}
    sources = provenance.get("sources", []) if isinstance(provenance, dict) else []
    notes = provenance.get("notes") if isinstance(provenance, dict) else None
    return tuple(str(item) for item in sources), notes


def _read_run_id(bundle_path: Path) -> str | None:
    try:
        manifest = json.loads((bundle_path / _RUN_MANIFEST_NAME).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise InputCatalogError(f"invalid run manifest: {error}") from error
    run_id = manifest.get("run_id")
    return str(run_id) if run_id is not None else None


__all__ = [
    "InputCandidate",
    "InputCatalogError",
    "InputKind",
    "InputSummary",
    "describe_candidate",
    "resolve_candidate",
    "resolve_input_root",
    "scan_input_catalog",
]
