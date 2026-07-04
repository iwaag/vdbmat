"""Generate the fixed Phase 1 run bundles and Mitsuba visual baselines.

The output root is relocatable: configurations contain only relative paths and the
baseline manifest contains only paths relative to that root. A clean rerun with the
same locked renderer is therefore byte-comparable.

Usage::

    uv run --group mitsuba python examples/phase1/generate_reference_baselines.py \
        .local/phase1/step10
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import struct
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vbdmat.conformance import (
    check_fixture_conformance,
    check_png_sanity,
    png_pixel_sha256,
)
from vbdmat.core import OpticalPropertyVolume
from vbdmat.exporters.mitsuba import (
    MITSUBA_ADAPTER,
    MITSUBA_ADAPTER_VERSION,
    MitsubaExportConfig,
    render_mitsuba,
)
from vbdmat.exporters.openvdb import OPENVDB_ADAPTER, OPENVDB_ADAPTER_VERSION
from vbdmat.fixtures import write_phase1_fixtures
from vbdmat.io import read_volume
from vbdmat.pipeline import (
    InputKind,
    PipelineConfig,
    run_pipeline,
    sha256_file,
    zarr_store_sha256,
)

BASELINE_CREATED_UTC = datetime(2026, 7, 2, tzinfo=UTC)
RENDER_CONFIG = MitsubaExportConfig(
    width=256,
    height=256,
    spp=64,
    seed=20260628,
    max_depth=8,
    fov_degrees=35.0,
    attenuation_diagnostic_gain=128.0,
)


def _configs() -> dict[str, PipelineConfig]:
    return {
        "window_coupon": PipelineConfig(
            input_kind=InputKind.DIRECT_VOXEL,
            input_path="inputs/window_coupon.voxels.json",
            output_path="runs/window_coupon",
        ),
        "stepped_wedge": PipelineConfig(
            input_kind=InputKind.DIRECT_VOXEL,
            input_path="inputs/stepped_wedge.voxels.json",
            output_path="runs/stepped_wedge",
        ),
    }


def _file_artifacts(directory: Path, root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]


def _png_dimensions(path: Path) -> tuple[int, int]:
    data = path.read_bytes()[:24]
    if len(data) != 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG image: {path}")
    return struct.unpack(">II", data[16:24])


def _render_settings() -> dict[str, object]:
    return {
        "variant": RENDER_CONFIG.variant,
        "width": RENDER_CONFIG.width,
        "height": RENDER_CONFIG.height,
        "spp": RENDER_CONFIG.spp,
        "seed": RENDER_CONFIG.seed,
        "max_depth": RENDER_CONFIG.max_depth,
        "fov_degrees": RENDER_CONFIG.fov_degrees,
        "attenuation_diagnostic_gain": (RENDER_CONFIG.attenuation_diagnostic_gain),
        "camera": "geometry-framed perspective from normalized (1.6,-2.2,1.4)",
        "lighting": "opposed rectangular area backlight, unit RGB radiance",
    }


def generate(output: Path, *, overwrite: bool) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    write_phase1_fixtures(output / "inputs")

    objects: dict[str, Any] = {}
    for name, config in _configs().items():
        result = run_pipeline(
            config,
            base_dir=str(output),
            created_utc=BASELINE_CREATED_UTC,
        )
        bundle = result.output_path
        optical = read_volume(bundle / "optical.zarr")
        if not isinstance(optical, OpticalPropertyVolume):
            raise TypeError("pipeline optical output has the wrong canonical type")

        render_directory = bundle / "exports" / "mitsuba"
        rendered = render_mitsuba(
            optical,
            render_directory,
            name="baseline",
            config=RENDER_CONFIG,
        )
        png_ok, png_detail = check_png_sanity(
            rendered.png_path,
            expected_size=(RENDER_CONFIG.width, RENDER_CONFIG.height),
        )
        attenuation_ok, attenuation_detail = check_png_sanity(
            rendered.attenuation_png_path,
            expected_size=(RENDER_CONFIG.width, RENDER_CONFIG.height),
        )
        if not png_ok or not attenuation_ok:
            raise RuntimeError(
                f"{name}: invalid visual baseline: {png_detail}; {attenuation_detail}"
            )

        conformance = check_fixture_conformance(name, optical, optical)
        if not conformance.passed:
            raise RuntimeError(f"{name}: field-level conformance failed")
        summary = json.loads(
            (bundle / "diagnostics" / "summary.json").read_text(encoding="utf-8")
        )
        width, height = _png_dimensions(rendered.png_path)
        objects[name] = {
            "bundle": bundle.relative_to(output).as_posix(),
            "source_payload_sha256": result.input_payload_sha256,
            "config_digest": result.config_digest,
            "scientific_config_digest": config.scientific_digest,
            "mapping_digest": result.mapping_digest,
            "run_id": result.run_id,
            "run_manifest_sha256": sha256_file(bundle / "run.json"),
            "material_zarr_sha256": zarr_store_sha256(bundle / "material.zarr"),
            "optical_zarr_sha256": zarr_store_sha256(bundle / "optical.zarr"),
            "canonical_summary": summary,
            "render": {
                "settings": _render_settings(),
                "image_dimensions": [width, height],
                "minimum_linear": rendered.minimum,
                "maximum_linear": rendered.maximum,
                "mean_linear_rgb": list(rendered.mean_linear_rgb),
                "png_sha256": rendered.png_sha256,
                "attenuation_png_sha256": rendered.attenuation_png_sha256,
                "png_sanity": png_detail,
                "attenuation_png_sanity": attenuation_detail,
            },
            "field_conformance": {
                "passed": conformance.passed,
                "checks": [check.name for check in conformance.checks if check.passed],
            },
            "capabilities": rendered.capability_report.to_dict(),
            "mitsuba_artifacts": _file_artifacts(render_directory, output),
        }

    manifest: dict[str, Any] = {
        "schema": {"name": "vbdmat.phase1-baselines", "version": "1.0.0"},
        "created_utc": BASELINE_CREATED_UTC.isoformat(),
        "renderer": {
            "name": "mitsuba",
            "version": importlib.metadata.version("mitsuba"),
        },
        "adapter": {
            "name": MITSUBA_ADAPTER,
            "version": MITSUBA_ADAPTER_VERSION,
        },
        "objects": objects,
        "known_differences": [
            "Mitsuba uses RGB sigma_t/albedo and derived IOR interface meshes.",
            "Cycles uses scalar coefficient reductions and omits internal IOR "
            "interfaces; its image is an interoperability smoke result only.",
            "Pixels from different renderers are not physical equivalents.",
        ],
        "limitations": (
            "Coefficients are provisional and uncalibrated; these baselines are "
            "software regressions, not predictions of a physical print."
        ),
    }
    manifest_path = output / "baseline-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def record_cycles_smoke(
    output: Path, *, openvdb_version: str, blender_version: str
) -> dict[str, Any]:
    """Add externally generated OpenVDB/Cycles smoke evidence to the manifest."""
    output = output.resolve()
    manifest_path = output / "baseline-manifest.json"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, item in manifest["objects"].items():
        directory = output / item["bundle"] / "exports" / "openvdb"
        image_path = directory / "cycles.png"
        manifest_data = json.loads(
            (directory / "openvdb-manifest.json").read_text(encoding="utf-8")
        )
        cycles = manifest_data["cycles"]
        dimensions = (int(cycles["width"]), int(cycles["height"]))
        image_ok, image_detail = check_png_sanity(image_path, expected_size=dimensions)
        if not image_ok:
            raise RuntimeError(f"{name}: invalid Cycles smoke output: {image_detail}")
        item["cycles_smoke"] = {
            "renderer": {"name": "blender-cycles", "version": blender_version},
            "openvdb": {"version": openvdb_version},
            "adapter": {
                "name": OPENVDB_ADAPTER,
                "version": OPENVDB_ADAPTER_VERSION,
            },
            "settings": manifest_data["cycles"],
            "image_dimensions": list(_png_dimensions(image_path)),
            "png_sha256": sha256_file(image_path),
            "pixel_sha256": png_pixel_sha256(image_path),
            "byte_stable": False,
            "png_sanity": image_detail,
            "artifacts": _file_artifacts(directory, output),
            "role": "interoperability smoke; not a visual or physical baseline",
        }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--record-cycles", action="store_true")
    parser.add_argument("--openvdb-version", default="10.0.1")
    parser.add_argument("--blender-version", default="4.5.11")
    arguments = parser.parse_args()
    if arguments.record_cycles:
        manifest = record_cycles_smoke(
            arguments.output,
            openvdb_version=arguments.openvdb_version,
            blender_version=arguments.blender_version,
        )
    else:
        manifest = generate(arguments.output, overwrite=arguments.overwrite)
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(arguments.output),
                "objects": sorted(manifest["objects"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
