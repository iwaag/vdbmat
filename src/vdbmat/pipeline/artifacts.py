"""Run-bundle artifact schemas, checksums, and diagnostic builders (ADR-007).

This module owns the *content* of a run bundle's non-Zarr artifacts — the stable
``run.json`` / ``summary.json`` / ``validation.json`` schemas — and the deterministic
checksum helpers the manifest links them by. It performs no orchestration and knows
nothing about temporary directories or publication; :mod:`vbdmat.pipeline.runner`
drives those. Everything here is pure given its inputs, so two identical runs produce
byte-identical artifacts (ADR-007 D8), the sole exception being the isolated
``created_utc`` field that the runner injects into ``run.json``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from vbdmat.core import (
    MaterialLabelVolume,
    OpticalPropertyVolume,
    SchemaIdentity,
    SchemaVersion,
)

#: Schema of the run manifest (``run.json``).
RUN_SCHEMA = SchemaIdentity(name="vbdmat.run", version=SchemaVersion(1, 0, 0))
#: Schema of the diagnostics summary (``diagnostics/summary.json``).
SUMMARY_SCHEMA = SchemaIdentity(name="vbdmat.summary", version=SchemaVersion(1, 0, 0))
#: Schema of the per-asset validation report (``diagnostics/validation.json``).
VALIDATION_SCHEMA = SchemaIdentity(
    name="vbdmat.validation", version=SchemaVersion(1, 0, 0)
)

_READ_CHUNK_BYTES = 1 << 20


def canonical_dumps(payload: Mapping[str, Any]) -> str:
    """Return sorted-key, tight-separator JSON matching the project convention."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data: bytes) -> str:
    """Return the ``sha256:`` digest of a byte string."""
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def sha256_file(path: Path) -> str:
    """Return the ``sha256:`` digest of a single file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_READ_CHUNK_BYTES), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def zarr_store_sha256(path: Path) -> str:
    """Return a deterministic digest over a Zarr store directory.

    A Zarr store is a directory of chunk and metadata files, so a single-file hash
    does not apply. The digest folds every regular file below ``path`` in sorted
    POSIX-relative-path order, mixing each file's relative path and byte length in so
    that renaming or truncating a chunk changes the result. File-system iteration order
    is normalized away by sorting, so the digest is reproducible across runs and hosts
    (ADR-007 D5/D8).
    """
    digest = hashlib.sha256()
    root = path.resolve()
    files = sorted(
        (item for item in root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    )
    for item in files:
        relative = item.relative_to(root).as_posix().encode("utf-8")
        data = item.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return f"sha256:{digest.hexdigest()}"


def path_size_bytes(path: Path) -> int:
    """Return the total size in bytes of a file or of every file below a directory."""
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return path.stat().st_size


def build_summary(
    material: MaterialLabelVolume,
    optical: OpticalPropertyVolume,
    *,
    config_digest: str,
    input_payload_sha256: str,
    mapping_digest: str,
) -> dict[str, Any]:
    """Build the implementation-independent ``summary.json`` document (ADR-007)."""
    geometry = material.geometry
    counts: dict[str, int] = {}
    palette: list[str] = []
    label = np.asarray(material.material_id)
    for definition in sorted(material.palette, key=lambda item: item.material_id):
        counts[str(definition.material_id)] = int(
            np.count_nonzero(label == definition.material_id)
        )
        palette.append(definition.name)

    return {
        "schema": {
            "name": SUMMARY_SCHEMA.name,
            "version": str(SUMMARY_SCHEMA.version),
        },
        "geometry": {
            "shape_zyx": list(geometry.shape_zyx),
            "voxel_size_xyz_m": list(geometry.voxel_size_xyz_m),
        },
        "material": {"counts": counts, "palette": palette},
        "optical": {
            "sigma_a_range_per_m": _range(optical.sigma_a),
            "sigma_s_range_per_m": _range(optical.sigma_s),
            "g_range": _range(optical.g),
            "ior_range": _range(optical.ior),
        },
        "digests": {
            "config": config_digest,
            "input_payload": input_payload_sha256,
            "mapping": mapping_digest,
        },
    }


def build_validation(assets: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Build the ``validation.json`` document from per-asset validation results.

    Each entry is ``{path, asset_type, schema, status}`` where ``status`` is ``ok`` for
    an asset that read back and fully re-validated against its persisted bytes.
    """
    return {
        "schema": {
            "name": VALIDATION_SCHEMA.name,
            "version": str(VALIDATION_SCHEMA.version),
        },
        "assets": [dict(entry) for entry in assets],
    }


def _range(array: Any) -> list[float]:
    values = np.asarray(array)
    if values.size == 0:
        return [0.0, 0.0]
    return [float(values.min()), float(values.max())]
