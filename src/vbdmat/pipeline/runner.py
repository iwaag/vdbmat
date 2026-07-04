"""Deterministic Phase 1 pipeline orchestration and failure-safe run bundles.

Implements ADR-007: a fixed, typed stage sequence

    load -> validate material -> persist material
         -> map optics -> validate optical -> persist optical
         -> summarize -> optional export

driven from a :class:`~vbdmat.pipeline.config.PipelineConfig`. The canonical bundle is
built inside a sibling temporary directory and published by a single atomic directory
rename, so an interrupted run never leaves a valid-looking partial ``run/`` (ADR-007
D7). The ``run_id`` is derived only from scientific inputs — configuration digest,
input payload checksum, and mapping digest — so two equivalent runs share an id and
produce byte-equal canonical artifacts; only the isolated ``created_utc`` field differs
(ADR-007 D3/D8).

Optional renderer exports (Step 8) run *after* the canonical bundle is published and
consume the persisted ``optical.zarr``. A failing export is attributed to the ``export``
stage and cannot corrupt the already-published canonical artifacts (ADR-007 D1).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import vbdmat
from vbdmat.core import (
    MaterialLabelVolume,
    OpticalPropertyVolume,
    Provenance,
)
from vbdmat.io.voxel_manifest import (
    inspect_material_label_manifest,
    read_material_label_manifest,
)
from vbdmat.io.zarr import read_volume, write_volume
from vbdmat.optics import map_material_volume_to_optical

from . import artifacts
from .config import ExportTarget, PipelineConfig
from .errors import PipelineRunError

#: Canonical bundle-relative artifact names (ADR-007 D4).
CONFIG_NAME = "config.json"
RUN_MANIFEST_NAME = "run.json"
SOURCE_DIR = "source"
MATERIAL_ZARR = "material.zarr"
OPTICAL_ZARR = "optical.zarr"
DIAGNOSTICS_DIR = "diagnostics"
VALIDATION_NAME = "diagnostics/validation.json"
SUMMARY_NAME = "diagnostics/summary.json"
EXPORTS_DIR = "exports"

#: Signature of an optional export backend. Given a target, the
#: published ``optical.zarr`` path, and a destination directory, it produces the export
#: artifacts and returns adapter/version metadata recorded in ``run.json``.
ExportRunner = Callable[[ExportTarget, Path, Path], Mapping[str, Any]]


class StageStatus(StrEnum):
    """Content-relevant status of one pipeline stage (ADR-007 D1)."""

    OK = "ok"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class StageRecord:
    """One executed pipeline stage and its status."""

    name: str
    status: StageStatus


@dataclass(frozen=True, slots=True)
class RunResult:
    """The outcome of a published pipeline run."""

    run_id: str
    output_path: Path
    config_digest: str
    input_payload_sha256: str
    mapping_digest: str
    created_utc: str
    stages: tuple[StageRecord, ...]
    manifest: Mapping[str, Any]
    summary: Mapping[str, Any]


def run_pipeline(
    config: PipelineConfig,
    *,
    base_dir: str,
    created_utc: datetime | None = None,
    export_runner: ExportRunner | None = None,
) -> RunResult:
    """Execute the Phase 1 pipeline and publish an ADR-007 run bundle.

    ``base_dir`` is the explicit directory the configuration's portable relative paths
    resolve against (there is no implicit current directory). ``created_utc`` may be
    supplied for tests; it is recorded only in the isolated ``run.json`` timestamp and
    never influences a canonical artifact or the ``run_id``.
    """
    input_path = Path(config.resolve_input_path(base_dir))
    output_path = Path(config.resolve_output_path(base_dir))

    if output_path.exists() and not config.overwrite:
        raise PipelineRunError(
            "publish",
            f"output already exists: {output_path}; set overwrite to replace it",
        )

    config_digest = config.digest
    mapping_digest = config.mapping_digest
    timestamp = (created_utc or datetime.now(UTC)).isoformat()

    stage = "load"
    temporary: Path | None = None
    try:
        # -- load --------------------------------------------------------------
        material, input_payload_sha256, copy_source = _load(input_path)
        material = _restamp(
            material, config_digest, (f"input-payload:{input_payload_sha256}",)
        )
        run_id = _run_id(config_digest, input_payload_sha256, mapping_digest)

        # -- map optics (in memory; equal to the persisted material) ---------------
        stage = "map-optics"
        mapping = config.resolve_mapping()
        optical = map_material_volume_to_optical(material, mapping)
        optical = _restamp(
            optical, config_digest, (f"mapping-digest:{mapping_digest}",)
        )

        # -- build the bundle in a sibling temporary directory (ADR-007 D7) --------
        temporary = output_path.parent / f".{output_path.name}.tmp-{run_id}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)

        # config.json is the exact canonical configuration, so its file checksum is
        # the config_digest by construction.
        (temporary / CONFIG_NAME).write_text(config.canonical_json(), encoding="utf-8")

        stage = "load"
        source_files = copy_source(temporary / SOURCE_DIR)

        stage = "persist-material"
        write_volume(temporary / MATERIAL_ZARR, material)

        stage = "validate-material"
        validate_material = _validate(
            temporary / MATERIAL_ZARR, config.validate_material, "material-label"
        )

        stage = "persist-optical"
        write_volume(temporary / OPTICAL_ZARR, optical)

        stage = "validate-optical"
        validate_optical = _validate(
            temporary / OPTICAL_ZARR, config.validate_optical, "optical-property"
        )

        stage = "summarize"
        summary = artifacts.build_summary(
            material,
            optical,
            config_digest=config_digest,
            input_payload_sha256=input_payload_sha256,
            mapping_digest=mapping_digest,
        )
        (temporary / DIAGNOSTICS_DIR).mkdir()
        (temporary / SUMMARY_NAME).write_text(
            artifacts.canonical_dumps(summary), encoding="utf-8"
        )
        (temporary / VALIDATION_NAME).write_text(
            artifacts.canonical_dumps(
                artifacts.build_validation([validate_material, validate_optical])
            ),
            encoding="utf-8",
        )

        stages = _stage_records(config, StageStatus.SKIPPED)
        manifest = _build_manifest(
            temporary,
            config=config,
            run_id=run_id,
            created_utc=timestamp,
            config_digest=config_digest,
            input_payload_sha256=input_payload_sha256,
            mapping_digest=mapping_digest,
            source_files=source_files,
            stages=stages,
            versions={"vbdmat": vbdmat.__version__},
        )
        (temporary / RUN_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except PipelineRunError:
        _cleanup(temporary)
        raise
    except Exception as error:
        _cleanup(temporary)
        raise PipelineRunError(stage, str(error)) from error

    # -- publish atomically (ADR-007 D7/D8) ---------------------------------------
    _publish(temporary, output_path, run_id)

    # -- optional export, post-publish (ADR-007 D1) -------------------------------
    if config.exports:
        stages, manifest = _run_exports(
            output_path,
            config=config,
            manifest=dict(manifest),
            export_runner=export_runner,
        )

    return RunResult(
        run_id=run_id,
        output_path=output_path,
        config_digest=config_digest,
        input_payload_sha256=input_payload_sha256,
        mapping_digest=mapping_digest,
        created_utc=timestamp,
        stages=stages,
        manifest=manifest,
        summary=summary,
    )


# -- load ---------------------------------------------------------------------------


def _load(
    input_path: Path,
) -> tuple[MaterialLabelVolume, str, Callable[[Path], list[Path]]]:
    """Return the material volume, input payload checksum, and a source copier."""
    inspection = inspect_material_label_manifest(input_path)
    payload_sha256 = f"sha256:{inspection.payload_sha256}"
    volume = read_material_label_manifest(input_path)
    payload_rel = inspection.payload_path

    def copy_source(source_dir: Path) -> list[Path]:
        source_dir.mkdir(parents=True)
        manifest_dest = source_dir / input_path.name
        shutil.copyfile(input_path, manifest_dest)
        payload_src = input_path.parent / payload_rel
        payload_dest = source_dir / payload_rel
        payload_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(payload_src, payload_dest)
        return [manifest_dest, payload_dest]

    return volume, payload_sha256, copy_source


def _restamp(volume: Any, config_digest: str, extra_sources: tuple[str, ...]) -> Any:
    """Chain provenance: stamp the run config digest and add upstream sources (D6)."""
    provenance = volume.provenance
    sources = tuple(provenance.sources)
    for item in extra_sources:
        if item not in sources:
            sources = (*sources, item)
    stamped = Provenance(
        generator=provenance.generator,
        generator_version=provenance.generator_version,
        created_utc=provenance.created_utc,
        configuration_digest=config_digest,
        sources=sources,
        notes=provenance.notes,
    )
    return dataclasses.replace(volume, provenance=stamped)


def _validate(zarr_path: Path, enabled: bool, asset_type: str) -> dict[str, Any]:
    """Validate a persisted asset by fully reading it back (ADR-007 D7)."""
    schema = "vbdmat.volume/1.0.0"
    entry: dict[str, Any] = {
        "path": zarr_path.name,
        "asset_type": asset_type,
        "schema": schema,
    }
    if not enabled:
        entry["status"] = StageStatus.SKIPPED.value
        return entry
    volume = read_volume(zarr_path)
    if isinstance(volume, MaterialLabelVolume) and asset_type != "material-label":
        raise PipelineRunError("validate", "persisted asset type mismatch")
    if isinstance(volume, OpticalPropertyVolume) and asset_type != "optical-property":
        raise PipelineRunError("validate", "persisted asset type mismatch")
    entry["status"] = StageStatus.OK.value
    return entry


# -- run identifier ----------------------------------------------------------------


def _run_id(config_digest: str, input_payload_sha256: str, mapping_digest: str) -> str:
    seed = f"{config_digest}\n{input_payload_sha256}\n{mapping_digest}"
    return "run-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


# -- manifest / stages -------------------------------------------------------------


def _stage_records(
    config: PipelineConfig,
    export_status: StageStatus,
) -> tuple[StageRecord, ...]:
    def flag(enabled: bool) -> StageStatus:
        return StageStatus.OK if enabled else StageStatus.SKIPPED

    return (
        StageRecord("load", StageStatus.OK),
        StageRecord("validate-material", flag(config.validate_material)),
        StageRecord("persist-material", StageStatus.OK),
        StageRecord("map-optics", StageStatus.OK),
        StageRecord("validate-optical", flag(config.validate_optical)),
        StageRecord("persist-optical", StageStatus.OK),
        StageRecord("summarize", StageStatus.OK),
        StageRecord("export", export_status),
    )


def _build_manifest(
    bundle: Path,
    *,
    config: PipelineConfig,
    run_id: str,
    created_utc: str,
    config_digest: str,
    input_payload_sha256: str,
    mapping_digest: str,
    source_files: list[Path],
    stages: tuple[StageRecord, ...],
    versions: Mapping[str, Any],
) -> dict[str, Any]:
    assets = _collect_assets(bundle, source_files)
    provenance: dict[str, Any] = {
        "input": {
            "kind": config.input_kind.value,
            "path": config.input_path,
            "payload_sha256": input_payload_sha256,
        },
        "material": {"configuration_digest": config_digest},
        "mapping": {"name": config.mapping_name, "digest": mapping_digest},
        "optical": {"configuration_digest": config_digest},
        "config_digest": config_digest,
    }

    return {
        "schema": {
            "name": artifacts.RUN_SCHEMA.name,
            "version": str(artifacts.RUN_SCHEMA.version),
        },
        "run_id": run_id,
        "created_utc": created_utc,
        "config_digest": config_digest,
        "input_payload_sha256": input_payload_sha256,
        "mapping_digest": mapping_digest,
        "input": {
            "kind": config.input_kind.value,
            "path": config.input_path,
            "payload_sha256": input_payload_sha256,
        },
        "stages": [{"name": item.name, "status": item.status.value} for item in stages],
        "assets": assets,
        "provenance": provenance,
        "versions": dict(versions),
    }


_ASSET_SCHEMAS = {
    CONFIG_NAME: "vbdmat.pipeline-config/2.0.0",
    MATERIAL_ZARR: "vbdmat.volume/1.0.0",
    OPTICAL_ZARR: "vbdmat.volume/1.0.0",
    SUMMARY_NAME: "vbdmat.summary/1.0.0",
    VALIDATION_NAME: "vbdmat.validation/1.0.0",
}


def _collect_assets(bundle: Path, source_files: list[Path]) -> list[dict[str, Any]]:
    paths: list[Path] = [
        bundle / CONFIG_NAME,
        *source_files,
        bundle / MATERIAL_ZARR,
        bundle / OPTICAL_ZARR,
        bundle / SUMMARY_NAME,
        bundle / VALIDATION_NAME,
    ]
    entries = [_asset_entry(bundle, item) for item in paths]
    return sorted(entries, key=lambda entry: entry["path"])


def _asset_entry(bundle: Path, path: Path) -> dict[str, Any]:
    relative = path.relative_to(bundle).as_posix()
    if path.is_dir():
        sha256 = artifacts.zarr_store_sha256(path)
    else:
        sha256 = artifacts.sha256_file(path)
    return {
        "path": relative,
        "schema": _ASSET_SCHEMAS.get(relative),
        "sha256": sha256,
        "size_bytes": artifacts.path_size_bytes(path),
    }


# -- publication -------------------------------------------------------------------


def _publish(temporary: Path, output_path: Path, run_id: str) -> None:
    if not output_path.exists():
        os.replace(temporary, output_path)
        return
    backup = output_path.parent / f".{output_path.name}.bak-{run_id}"
    if backup.exists():
        shutil.rmtree(backup)
    os.replace(output_path, backup)
    try:
        os.replace(temporary, output_path)
    except BaseException:
        if not output_path.exists():
            os.replace(backup, output_path)
        raise
    shutil.rmtree(backup)


def _cleanup(temporary: Path | None) -> None:
    if temporary is not None and temporary.exists():
        shutil.rmtree(temporary, ignore_errors=True)


# -- optional export (post-publish, failure-isolated) ------------------------------


def _run_exports(
    output_path: Path,
    *,
    config: PipelineConfig,
    manifest: dict[str, Any],
    export_runner: ExportRunner | None,
) -> tuple[tuple[StageRecord, ...], dict[str, Any]]:
    if export_runner is None:
        export_runner = _default_export_runner

    exports_dir = output_path / EXPORTS_DIR
    optical_zarr = output_path / OPTICAL_ZARR
    status = StageStatus.OK
    export_versions: dict[str, Any] = {}
    export_results: dict[str, Any] = {}
    exported_assets: list[dict[str, Any]] = []
    error_message: str | None = None
    for export in config.exports:
        target = export.target
        destination = exports_dir / target.value
        destination.mkdir(parents=True, exist_ok=True)
        try:
            info = export_runner(target, optical_zarr, destination)
        except Exception as error:
            status = StageStatus.FAILED
            error_message = f"{target.value}: {error}"
            shutil.rmtree(destination, ignore_errors=True)
            break
        result = dict(info)
        export_results[target.value] = result
        export_versions[target.value] = {
            "adapter": result.get("adapter"),
            "adapter_version": result.get("adapter_version"),
            "renderer": result.get("renderer"),
        }
        exported_assets.extend(
            _asset_entry(output_path, item)
            for item in sorted(destination.rglob("*"))
            if item.is_file()
        )

    stages = _stage_records(config, status)
    manifest["stages"] = [
        {"name": item.name, "status": item.status.value} for item in stages
    ]
    manifest.setdefault("versions", {})["exporters"] = export_versions
    manifest["assets"] = sorted(
        [*manifest.get("assets", []), *exported_assets],
        key=lambda entry: entry["path"],
    )
    manifest.setdefault("provenance", {})["exports"] = export_results
    export_state: dict[str, Any] = {
        "status": status.value,
        "targets": export_results,
    }
    if error_message is not None:
        export_state["error"] = error_message
    manifest["export"] = export_state
    _rewrite_manifest(output_path, manifest)
    return stages, manifest


def _default_export_runner(
    target: ExportTarget, optical_zarr: Path, destination: Path
) -> Mapping[str, Any]:
    """Dispatch below the pipeline boundary without importing renderer bindings."""
    from vbdmat.exporters import export_restored_optical

    outcome = export_restored_optical(target.value, optical_zarr, destination)
    return outcome.to_dict()


def _rewrite_manifest(output_path: Path, manifest: Mapping[str, Any]) -> None:
    (output_path / RUN_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
