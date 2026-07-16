"""Derived optical-volume regeneration for the Mitsuba stage viewer.

Given a canonical run bundle (the Load/Rebuild input) and a selected optical
mapping, this module builds an in-memory :class:`~vdbmat.pipeline.PipelineConfig`
that re-runs the bundle's own preserved ``source/*.voxels.json`` material input
through the selected mapping, and publishes the result as a fresh canonical run
bundle under an explicit work root via :func:`~vdbmat.pipeline.run_pipeline`.

No mapping application, palette-coverage check, mixing rule, or provenance
logic is reimplemented here: this module only locates the bundle's material
source, resolves the mapping file, and calls the existing pipeline. Content
identity of ``source/*.voxels.json`` is verified against the bundle's own
``run.json`` before anything runs, so a hand-edited or substituted bundle
fails at ``validate`` rather than silently regenerating from the wrong
material input.

Repeated calls with the same source payload digest and mapping digest reuse a
previously published derived bundle instead of re-running the pipeline; the
reuse check reads only the derived bundle's own ``run.json`` declarations,
never re-hashes the derived ``optical.zarr`` content.

This module imports neither Mitsuba nor viser.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from mitsuba_stage_mappings import MappingCandidate

from vdbmat.io import VoxelManifestError, inspect_material_label_manifest
from vdbmat.optics import OpticalMappingError, load_optical_mapping
from vdbmat.pipeline import (
    InputKind,
    PipelineConfig,
    PipelineConfigError,
    PipelineRunError,
    run_pipeline,
)

_RUN_MANIFEST_NAME = "run.json"
_OPTICAL_ZARR_NAME = "optical.zarr"
_SOURCE_DIRNAME = "source"
_VOXELS_SUFFIX = ".voxels.json"
_SLUG_PATTERN = re.compile(r"[^a-zA-Z0-9]+")


class RegenError(Exception):
    """Regeneration failed at a named, user-visible stage."""

    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        self.message = message
        super().__init__(f"regenerate optical failed at {stage}: {message}")


@dataclass(frozen=True, slots=True)
class DerivedBundle:
    """One published (or reused) derived canonical run bundle."""

    bundle_path: Path
    optical_zarr: Path
    mapping_digest: str
    source_payload_sha256: str
    reused: bool


def derived_bundle_key(source_payload_sha256: str, mapping_digest: str) -> str:
    """Return the deterministic cache key of one (source, mapping) pair."""
    seed = f"{source_payload_sha256}\n{mapping_digest}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def locate_bundle_source(bundle_path: Path) -> tuple[Path, str]:
    """Return the bundle's preserved direct-voxel manifest and payload digest.

    Reads ``run.json`` to find the single ``source/*.voxels.json`` asset
    entry, then re-inspects that manifest file's own payload declaration and
    confirms it matches the bundle's recorded ``input_payload_sha256``. This
    catches a hand-edited or substituted ``source/`` directory before any
    pipeline stage runs, rather than silently mapping the wrong material
    input.
    """
    run_manifest_path = bundle_path / _RUN_MANIFEST_NAME
    try:
        text = run_manifest_path.read_text(encoding="utf-8")
    except OSError as error:
        raise RegenError(
            "validate", f"cannot read {run_manifest_path}: {error}"
        ) from error
    try:
        run_manifest = json.loads(text)
    except json.JSONDecodeError as error:
        raise RegenError(
            "validate", f"{run_manifest_path} is not valid JSON: {error}"
        ) from error
    if not isinstance(run_manifest, dict):
        raise RegenError("validate", f"{run_manifest_path} must be a JSON object")

    assets = run_manifest.get("assets")
    if not isinstance(assets, list):
        raise RegenError("validate", f"{run_manifest_path} has no assets list")
    candidates = [
        entry["path"]
        for entry in assets
        if isinstance(entry, dict)
        and isinstance(entry.get("path"), str)
        and entry["path"].startswith(f"{_SOURCE_DIRNAME}/")
        and entry["path"].endswith(_VOXELS_SUFFIX)
    ]
    if len(candidates) != 1:
        raise RegenError(
            "validate",
            "bundle must declare exactly one source voxel manifest asset, "
            f"found {len(candidates)}",
        )
    manifest_path = bundle_path / candidates[0]
    if not manifest_path.is_file():
        raise RegenError(
            "validate", f"declared source manifest is missing: {manifest_path}"
        )

    declared_payload_sha256 = run_manifest.get("input_payload_sha256")
    if not isinstance(declared_payload_sha256, str):
        raise RegenError("validate", f"{run_manifest_path} has no input_payload_sha256")

    try:
        inspection = inspect_material_label_manifest(manifest_path)
    except VoxelManifestError as error:
        raise RegenError(
            "validate", f"bundle source manifest is invalid: {error}"
        ) from error
    actual_payload_sha256 = f"sha256:{inspection.payload_sha256}"
    if actual_payload_sha256 != declared_payload_sha256:
        raise RegenError(
            "validate",
            "bundle source payload digest mismatch: declared "
            f"{declared_payload_sha256}, actual {actual_payload_sha256} "
            "(the bundle's source/ directory may have been modified)",
        )
    return manifest_path, actual_payload_sha256


def regenerate_optical(
    source_bundle: Path,
    mapping_candidate: MappingCandidate,
    work_root: Path,
    *,
    on_stage: Callable[[str], None] = lambda stage: None,
) -> DerivedBundle:
    """Apply a selected mapping to a bundle's source material, with caching.

    Publishes (or reuses) a full canonical run bundle under ``work_root``.
    Never writes into ``source_bundle`` or the mapping file. The derived
    bundle path collides with neither, since it is only ever computed as
    ``work_root / "<slug>-<key>"`` — but a caller-supplied ``work_root`` that
    itself lies inside ``source_bundle`` is rejected defensively, since
    ``run_pipeline`` overwrites ``work_root`` contents at that path.
    """
    on_stage("validate")
    resolved_source = source_bundle.resolve()
    manifest_path, source_payload_sha256 = locate_bundle_source(resolved_source)
    try:
        mapping_config = load_optical_mapping(mapping_candidate.path)
    except OpticalMappingError as error:
        raise RegenError("validate", str(error)) from error
    mapping_digest = mapping_config.digest

    key = derived_bundle_key(source_payload_sha256, mapping_digest)
    resolved_work_root = work_root.resolve()
    output_dir = resolved_work_root / f"{_slug_for(resolved_source)}-{key[:12]}"
    if output_dir == resolved_source or resolved_source in output_dir.parents:
        raise RegenError(
            "validate",
            f"derived bundle path collides with source bundle: {output_dir}",
        )
    if output_dir in resolved_source.parents:
        raise RegenError(
            "validate",
            f"derived bundle work root contains the source bundle: {output_dir}",
        )

    if _cache_matches(output_dir, source_payload_sha256, mapping_digest):
        return DerivedBundle(
            bundle_path=output_dir,
            optical_zarr=output_dir / _OPTICAL_ZARR_NAME,
            mapping_digest=mapping_digest,
            source_payload_sha256=source_payload_sha256,
            reused=True,
        )

    on_stage("map")
    try:
        config = PipelineConfig(
            input_kind=InputKind.DIRECT_VOXEL,
            input_path=str(manifest_path),
            # Keep the persisted pipeline config independent of the caller's
            # absolute work-root location. The output path participates in the
            # config/provenance digest stamped into optical.zarr, so storing an
            # absolute path here would make an otherwise identical headless
            # regeneration fail the session's derived digest verification.
            output_path=output_dir.name,
            mapping_path=str(mapping_candidate.path),
            mapping_digest=mapping_digest,
            validate_material=True,
            validate_optical=True,
            overwrite=True,
        )
        run_pipeline(config, base_dir=str(resolved_work_root))
    except (PipelineConfigError, PipelineRunError) as error:
        raise RegenError("map", str(error)) from error

    return DerivedBundle(
        bundle_path=output_dir,
        optical_zarr=output_dir / _OPTICAL_ZARR_NAME,
        mapping_digest=mapping_digest,
        source_payload_sha256=source_payload_sha256,
        reused=False,
    )


def _cache_matches(
    output_dir: Path, source_payload_sha256: str, mapping_digest: str
) -> bool:
    run_json = output_dir / _RUN_MANIFEST_NAME
    if not run_json.is_file():
        return False
    try:
        manifest = json.loads(run_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(manifest, dict):
        return False
    return (
        manifest.get("input_payload_sha256") == source_payload_sha256
        and manifest.get("mapping_digest") == mapping_digest
    )


def _slug_for(bundle_path: Path) -> str:
    slug = _SLUG_PATTERN.sub("-", bundle_path.name).strip("-").lower()
    return slug or "bundle"


__all__ = [
    "DerivedBundle",
    "RegenError",
    "derived_bundle_key",
    "locate_bundle_source",
    "regenerate_optical",
]
