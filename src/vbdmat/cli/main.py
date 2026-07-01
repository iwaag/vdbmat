"""Installed ``vbdmat`` console entry point for the Phase 1 research MVP.

This module owns argument parsing, presentation, and exit-code mapping only. Scientific
work is delegated to the package APIs fixed in Steps 2--6.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn, cast

import numpy as np

from vbdmat.core import (
    MaterialLabelVolume,
    MaterialMixtureVolume,
    OpticalPropertyVolume,
    VolumeValidationError,
)
from vbdmat.io import (
    MeshReadError,
    VolumeIOError,
    VoxelManifestError,
    read_material_label_manifest,
    read_stl,
    read_volume,
    write_volume,
)
from vbdmat.optics import OpticalMappingError, map_material_volume_to_optical
from vbdmat.pipeline import (
    DEFAULT_MAPPING_NAME,
    InputKind,
    PipelineConfig,
    PipelineConfigError,
    PipelineRunError,
    run_pipeline,
    sha256_file,
    zarr_store_sha256,
)
from vbdmat.voxelize import MeshTopologyError, VoxelizationError, voxelize_mesh

from .errors import CliError, ExitCode
from .output import human_summary, json_line

_PROVISIONAL = (
    "Optical coefficients are provisional and uncalibrated; outputs are not physical "
    "print predictions."
)


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise CliError(ExitCode.USAGE, message)


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="vbdmat",
        description="Renderer-independent voxel material preprocessing.",
        epilog=_PROVISIONAL,
    )
    parser.add_argument("--debug", action="store_true", help="show tracebacks")
    commands = parser.add_subparsers(dest="command", required=True)
    command_parsers: list[argparse.ArgumentParser] = []

    import_parser = commands.add_parser(
        "import-voxels", help="import a vbdmat.voxels manifest to material Zarr"
    )
    command_parsers.append(import_parser)
    _paths(import_parser, "MANIFEST")
    _writer_flags(import_parser)

    voxelize_parser = commands.add_parser(
        "voxelize", help="voxelize one watertight STL into material Zarr"
    )
    command_parsers.append(voxelize_parser)
    _paths(voxelize_parser, "MESH")
    voxelize_parser.add_argument(
        "--unit", required=True, choices=("m", "mm"), help="explicit STL unit"
    )
    voxelize_parser.add_argument(
        "--voxel-size",
        required=True,
        metavar="M[,M,M]",
        help="voxel size in metres: one scalar or explicit X,Y,Z values",
    )
    voxelize_parser.add_argument(
        "--material-id", required=True, type=int, help="material ID in [1, 65535]"
    )
    voxelize_parser.add_argument(
        "--material-name", default="material", help="material palette name"
    )
    voxelize_parser.add_argument(
        "--placement", type=Path, help="JSON file containing a rigid 4x4 matrix"
    )
    voxelize_parser.add_argument(
        "--padding-cells", type=int, default=1, help="domain padding (default: 1)"
    )
    _writer_flags(voxelize_parser)

    convert_parser = commands.add_parser(
        "convert", help="map canonical material Zarr to optical Zarr"
    )
    command_parsers.append(convert_parser)
    _paths(convert_parser, "MATERIAL_ZARR")
    convert_parser.add_argument(
        "--mapping",
        default=DEFAULT_MAPPING_NAME,
        help=f"builtin mapping (default: {DEFAULT_MAPPING_NAME}); {_PROVISIONAL}",
    )
    _writer_flags(convert_parser)

    for name, help_text in (
        ("inspect", "inspect a canonical Zarr asset or run bundle"),
        ("validate", "fully validate a canonical Zarr asset or run bundle"),
    ):
        subparser = commands.add_parser(name, help=help_text)
        command_parsers.append(subparser)
        subparser.add_argument("asset", type=Path, metavar="ASSET")
        subparser.add_argument("--json", action="store_true", dest="json_output")

    run_parser = commands.add_parser(
        "run", help="execute a versioned pipeline configuration"
    )
    command_parsers.append(run_parser)
    run_parser.add_argument("config", type=Path, metavar="CONFIG")
    run_parser.add_argument("--json", action="store_true", dest="json_output")

    export_parser = commands.add_parser(
        "export", help="export optical Zarr through an optional renderer adapter"
    )
    command_parsers.append(export_parser)
    export_parser.add_argument("target", choices=("mitsuba", "openvdb"))
    export_parser.add_argument("optical_zarr", type=Path, metavar="OPTICAL_ZARR")
    export_parser.add_argument("output", type=Path, metavar="OUTPUT")
    _writer_flags(export_parser)
    for command_parser in command_parsers:
        command_parser.epilog = _PROVISIONAL
    return parser


def _paths(parser: argparse.ArgumentParser, input_name: str) -> None:
    parser.add_argument("input", type=Path, metavar=input_name)
    parser.add_argument("output", type=Path, metavar="OUTPUT")


def _writer_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return its documented process exit code."""
    parser = _parser()
    raw_arguments = tuple(sys.argv[1:] if argv is None else argv)
    arguments: argparse.Namespace | None = None
    try:
        arguments = parser.parse_args(raw_arguments)
        document = _dispatch(arguments)
        _write_success(document, json_output=bool(arguments.json_output))
        return int(ExitCode.SUCCESS)
    except CliError as error:
        debug = (
            bool(getattr(arguments, "debug", False))
            or "--debug" in raw_arguments
            or os.environ.get("VBDMAT_DEBUG") == "1"
        )
        if debug:
            raise
        json_output = bool(getattr(arguments, "json_output", False)) or (
            "--json" in raw_arguments
        )
        if json_output:
            sys.stdout.write(
                json_line(
                    {
                        "status": "error",
                        "exit_code": int(error.code),
                        "message": error.message,
                    }
                )
            )
        sys.stderr.write(f"vbdmat: {error.message}\n")
        return int(error.code)
    except Exception as error:
        debug = (
            bool(getattr(arguments, "debug", False))
            or "--debug" in raw_arguments
            or os.environ.get("VBDMAT_DEBUG") == "1"
        )
        if debug:
            raise
        sys.stderr.write(f"vbdmat: internal error: {error}\n")
        return int(ExitCode.INTERNAL)


def _dispatch(arguments: argparse.Namespace) -> dict[str, Any]:
    command = cast(str, arguments.command)
    try:
        if command == "import-voxels":
            _refuse_overwrite(arguments.output, arguments.overwrite)
            volume = read_material_label_manifest(arguments.input)
            write_volume(arguments.output, volume, overwrite=arguments.overwrite)
            return _asset_result(command, arguments.output, volume)
        if command == "voxelize":
            _refuse_overwrite(arguments.output, arguments.overwrite)
            placement = _placement(arguments.placement)
            voxelization = voxelize_mesh(
                read_stl(arguments.input),
                source_unit=arguments.unit,
                voxel_size_xyz_m=_voxel_size(arguments.voxel_size),
                material_id=arguments.material_id,
                material_name=arguments.material_name,
                placement=placement,
                padding_cells=arguments.padding_cells,
                identity=str(arguments.input),
            )
            write_volume(
                arguments.output,
                voxelization.volume,
                overwrite=arguments.overwrite,
            )
            document = _asset_result(command, arguments.output, voxelization.volume)
            document["diagnostics"] = {
                "triangle_count": voxelization.diagnostics.triangle_count,
                "occupied_cells": voxelization.diagnostics.occupied_cells,
                "bounds_min_xyz_m": list(voxelization.diagnostics.bounds_min_xyz_m),
                "bounds_max_xyz_m": list(voxelization.diagnostics.bounds_max_xyz_m),
            }
            return document
        if command == "convert":
            _refuse_overwrite(arguments.output, arguments.overwrite)
            material = read_volume(arguments.input)
            if not isinstance(material, (MaterialLabelVolume, MaterialMixtureVolume)):
                raise CliError(
                    ExitCode.CONVERSION, "convert input must be a material volume"
                )
            config = _mapping(arguments.mapping)
            optical = map_material_volume_to_optical(material, config.resolve_mapping())
            write_volume(arguments.output, optical, overwrite=arguments.overwrite)
            document = _asset_result(command, arguments.output, optical)
            document["mapping"] = arguments.mapping
            document["mapping_digest"] = config.mapping_digest
            return document
        if command == "inspect":
            return _inspect(arguments.asset, validate=False)
        if command == "validate":
            return _inspect(arguments.asset, validate=True)
        if command == "run":
            config_path = cast(Path, arguments.config)
            config = PipelineConfig.from_json(_read_text(config_path, "config"))
            run_result = run_pipeline(
                config, base_dir=str(config_path.resolve().parent)
            )
            return {
                "status": "ok",
                "operation": "run",
                "path": str(run_result.output_path),
                "run_id": run_result.run_id,
                "config_digest": run_result.config_digest,
                "input_payload_sha256": run_result.input_payload_sha256,
                "mapping_digest": run_result.mapping_digest,
                "stages": [
                    {"name": item.name, "status": item.status.value}
                    for item in run_result.stages
                ],
            }
        if command == "export":
            raise CliError(
                ExitCode.OPTIONAL_DEPENDENCY,
                f"{arguments.target} export is connected in Phase 1 Step 8; "
                "the canonical optical asset is unchanged",
            )
    except CliError:
        raise
    except FileExistsError as error:
        raise CliError(ExitCode.USAGE, str(error)) from error
    except VoxelManifestError as error:
        code = _manifest_error_code(error)
        raise CliError(code, str(error)) from error
    except MeshReadError as error:
        code = ExitCode.IO if "file not found" in error.message else ExitCode.VALIDATION
        raise CliError(code, str(error)) from error
    except (MeshTopologyError, VoxelizationError, VolumeValidationError) as error:
        raise CliError(ExitCode.VALIDATION, str(error)) from error
    except VolumeIOError as error:
        code = ExitCode.IO if error.field_path == "store" else ExitCode.VALIDATION
        raise CliError(code, str(error)) from error
    except PipelineConfigError as error:
        raise CliError(ExitCode.VALIDATION, str(error)) from error
    except (OpticalMappingError, PipelineRunError) as error:
        if (
            isinstance(error, PipelineRunError)
            and error.stage == "publish"
            and ("already exists" in error.message)
        ):
            raise CliError(ExitCode.USAGE, str(error)) from error
        raise CliError(ExitCode.CONVERSION, str(error)) from error
    except OSError as error:
        raise CliError(ExitCode.IO, str(error)) from error
    raise CliError(ExitCode.USAGE, f"unknown command: {command}")


def _mapping(name: str) -> PipelineConfig:
    try:
        return PipelineConfig(
            input_kind=InputKind.DIRECT_VOXEL,
            input_path="unused",
            output_path="unused",
            mapping_name=name,
        )
    except (KeyError, PipelineConfigError) as error:
        raise CliError(ExitCode.CONVERSION, f"unsupported mapping: {name}") from error


def _refuse_overwrite(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise CliError(
            ExitCode.USAGE,
            f"refusing to overwrite existing path: {path} (use --overwrite)",
        )


def _voxel_size(text: str) -> tuple[float, float, float]:
    try:
        values = tuple(float(item) for item in text.split(","))
    except ValueError as error:
        raise CliError(
            ExitCode.USAGE, "--voxel-size must contain numeric metre values"
        ) from error
    if len(values) == 1:
        return (values[0], values[0], values[0])
    if len(values) == 3:
        return values
    raise CliError(
        ExitCode.USAGE, "--voxel-size must be one scalar or three X,Y,Z values"
    )


def _placement(path: Path | None) -> Any:
    if path is None:
        return (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    try:
        return json.loads(_read_text(path, "placement"))
    except json.JSONDecodeError as error:
        raise CliError(
            ExitCode.VALIDATION, f"placement: invalid JSON: {error}"
        ) from error


def _read_text(path: Path, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise CliError(ExitCode.IO, f"{label}: cannot read {path}: {error}") from error


def _inspect(path: Path, *, validate: bool) -> dict[str, Any]:
    if (path / "run.json").is_file():
        return _inspect_bundle(path, validate=validate)
    volume = read_volume(path)
    document = _volume_document(volume)
    document.update(
        {
            "status": "ok",
            "operation": "validate" if validate else "inspect",
            "path": str(path),
            "validation": {"status": "ok", "mode": "full-read"},
        }
    )
    return document


def _inspect_bundle(path: Path, *, validate: bool) -> dict[str, Any]:
    try:
        manifest = json.loads(_read_text(path / "run.json", "run manifest"))
        summary = json.loads(
            _read_text(path / "diagnostics" / "summary.json", "summary")
        )
    except json.JSONDecodeError as error:
        raise CliError(
            ExitCode.VALIDATION, f"run bundle: invalid JSON: {error}"
        ) from error
    if not isinstance(manifest, dict) or not isinstance(summary, dict):
        raise CliError(
            ExitCode.VALIDATION,
            "run bundle: run.json and diagnostics/summary.json must be objects",
        )
    assets: list[dict[str, Any]] = []
    for name in ("material.zarr", "optical.zarr"):
        volume = read_volume(path / name)
        assets.append(_volume_document(volume))
    checksum_status = "not-checked"
    if validate:
        _validate_bundle_checksums(path, manifest)
        checksum_status = "ok"
    return {
        "status": "ok",
        "operation": "validate" if validate else "inspect",
        "path": str(path),
        "asset_kind": "run-bundle",
        "schema": manifest.get("schema"),
        "run_id": manifest.get("run_id"),
        "stages": manifest.get("stages"),
        "summary": summary,
        "assets": assets,
        "validation": {
            "status": "ok",
            "mode": "full-read",
            "checksums": checksum_status,
        },
    }


def _validate_bundle_checksums(path: Path, manifest: Mapping[str, Any]) -> None:
    declarations = manifest.get("assets")
    if not isinstance(declarations, list):
        raise CliError(ExitCode.VALIDATION, "run.assets: must be an array")
    root = path.resolve()
    for index, declaration in enumerate(declarations):
        if not isinstance(declaration, dict):
            raise CliError(
                ExitCode.VALIDATION, f"run.assets[{index}]: must be an object"
            )
        relative = declaration.get("path")
        expected = declaration.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise CliError(
                ExitCode.VALIDATION,
                f"run.assets[{index}]: path and sha256 must be strings",
            )
        candidate = (root / relative).resolve()
        if candidate != root and root not in candidate.parents:
            raise CliError(
                ExitCode.VALIDATION,
                f"run.assets[{index}].path: escapes the run bundle",
            )
        if not candidate.exists():
            raise CliError(ExitCode.VALIDATION, f"run asset is missing: {relative}")
        actual = (
            zarr_store_sha256(candidate)
            if candidate.is_dir()
            else sha256_file(candidate)
        )
        if actual != expected:
            raise CliError(
                ExitCode.VALIDATION,
                f"run asset checksum mismatch: {relative}; expected {expected}, "
                f"actual {actual}",
            )


def _asset_result(
    operation: str,
    path: Path,
    volume: MaterialLabelVolume | MaterialMixtureVolume | OpticalPropertyVolume,
) -> dict[str, Any]:
    return {
        "status": "ok",
        "operation": operation,
        "path": str(path),
        "asset_type": volume.asset_type.value,
        "schema": {"name": volume.schema.name, "version": str(volume.schema.version)},
        "shape_zyx": list(volume.geometry.shape_zyx),
    }


def _volume_document(
    volume: MaterialLabelVolume | MaterialMixtureVolume | OpticalPropertyVolume,
) -> dict[str, Any]:
    geometry = volume.geometry
    document: dict[str, Any] = {
        "asset_kind": "canonical-volume",
        "asset_type": volume.asset_type.value,
        "schema": {"name": volume.schema.name, "version": str(volume.schema.version)},
        "geometry": {
            "shape_zyx": list(geometry.shape_zyx),
            "voxel_size_xyz_m": list(geometry.voxel_size_xyz_m),
            "local_to_world": [list(row) for row in geometry.local_to_world],
            "coordinate_system": "right-handed-world-xyz",
            "storage_order": "zyx",
            "sampling": "cell-centred",
            "length_unit": "m",
        },
        "provenance": _provenance(volume.provenance),
    }
    if isinstance(volume, MaterialLabelVolume):
        document["materials"] = [
            {
                "material_id": item.material_id,
                "name": item.name,
                "role": item.role.value,
                "count": int(np.count_nonzero(volume.material_id == item.material_id)),
            }
            for item in volume.palette
        ]
    elif isinstance(volume, MaterialMixtureVolume):
        document["materials"] = [
            {
                "material_id": item.material_id,
                "name": item.name,
                "role": item.role.value,
            }
            for item in volume.palette
        ]
        document["field_ranges"] = {"fractions": _range(volume.fractions)}
    else:
        basis = volume.optical_basis
        document["optical_basis"] = {
            "kind": basis.kind.value,
            "identifier": basis.identifier,
            "coordinates": list(basis.coordinates),
            "reference_white": basis.reference_white,
            "observer": basis.observer,
            "transfer": basis.transfer,
        }
        document["field_ranges"] = {
            "sigma_a_per_m": _range(volume.sigma_a),
            "sigma_s_per_m": _range(volume.sigma_s),
            "g": _range(volume.g),
            "ior": _range(volume.ior),
        }
        document["calibration"] = "provisional-uncalibrated"
    return document


def _provenance(value: Any) -> dict[str, Any]:
    return {
        "generator": value.generator,
        "generator_version": value.generator_version,
        "created_utc": value.created_utc.isoformat() if value.created_utc else None,
        "configuration_digest": value.configuration_digest,
        "sources": list(value.sources),
        "notes": value.notes,
    }


def _range(value: Any) -> list[float]:
    array = np.asarray(value)
    return [float(array.min()), float(array.max())]


def _manifest_error_code(error: VoxelManifestError) -> ExitCode:
    if error.field_path in {"payload.path", "payload.sha256"}:
        return ExitCode.IO
    if error.field_path == "manifest" and (
        "file not found" in error.message or "cannot read" in error.message
    ):
        return ExitCode.IO
    return ExitCode.VALIDATION


def _write_success(document: Mapping[str, Any], *, json_output: bool) -> None:
    sys.stdout.write(json_line(document) if json_output else human_summary(document))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
