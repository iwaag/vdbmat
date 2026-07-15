"""Render a Mitsuba proof scene with a legible checkerboard stage added.

This is a demo-track helper (see :mod:`blender_glass_demo` for the Cycles
equivalent); it is not part of the canonical pipeline and produces qualitative,
uncalibrated images. ``vdbmat.exporters.mitsuba.prepare_mitsuba_scene()`` /
``render_mitsuba()`` / ``MitsubaExportConfig`` are untouched by this script.

Why this exists: the canonical Mitsuba scene built by ``prepare_mitsuba_scene()``
has exactly one plain white backlight rectangle and nothing else — no floor, no
patterned backdrop. A render of a transparent/nested-material object against that
scene looks like "something cube-shaped on a white background"; there is nothing in
frame whose distortion or occlusion tells a viewer where refraction or an opaque
interior is happening. This script takes the loadable ``scene_dict`` returned by
``prepare_mitsuba_scene()`` unmodified in its medium/exterior/interior entries, and
adds, purely additively, a checkerboard backdrop plane, a checkerboard floor plane,
and one oblique key light, then renders directly with ``mi.render`` (bypassing
``render_mitsuba()``, whose single-PNG contract has no hook for extra scene
entries).

Stage construction lives in the sibling module :mod:`mitsuba_stage`
(``StageConfig`` + ``apply_stage``); this script is the thin CLI around it.
A stage-config JSON preset (``*.stage.json``, see
``examples/pipeline_run/demo/presets/``) can override lights, camera, and
backdrop/floor patterns; running without one reproduces the built-in defaults.
Explicit ``--width/--height/--spp/--checker-scale`` arguments win over the
preset.

Invoke on the host (no Docker needed for Mitsuba):

    uv run --group mitsuba python \
        examples/pipeline_run/demo/mitsuba_stage_demo.py -- \
        OPTICAL_ZARR OUTPUT_PNG [--stage-config PRESET.stage.json] \
        [--width 512] [--height 512] [--spp 128] [--checker-scale 8]

``OPTICAL_ZARR`` is an ``optical.zarr`` bundle written by ``vdbmat run``.
``OUTPUT_PNG`` is where the rendered PNG is written; the exterior/interior PLY
meshes, ``capabilities.json``, and ``scene-summary.json`` that
``prepare_mitsuba_scene()`` writes as a side effect land next to it, under
``<OUTPUT_PNG stem>_scene/``. The script prints ``PIXELSTATS`` (min/max/mean/std of
the rendered pixels), the same cheap headless regression signal used by the
Blender demo scripts in this directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
from mitsuba_stage import StageConfig, apply_stage, stage_config_from_json

from vdbmat.core.volumes import OpticalPropertyVolume
from vdbmat.exporters.mitsuba import MitsubaExportConfig, prepare_mitsuba_scene
from vdbmat.io.zarr import read_volume


def _parse_args() -> argparse.Namespace:
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    parser = argparse.ArgumentParser(prog="mitsuba_stage_demo")
    parser.add_argument("optical_zarr", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument(
        "--stage-config",
        type=Path,
        default=None,
        help="stage-config JSON preset (*.stage.json); omitted fields keep "
        "their defaults, which equal the built-in stage",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="override render width (default: preset value or 512)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="override render height (default: preset value or 512)",
    )
    parser.add_argument(
        "--spp",
        type=int,
        default=None,
        help="override samples per pixel (default: preset value or 128)",
    )
    parser.add_argument(
        "--checker-scale",
        type=int,
        default=None,
        help="override the number of checkerboard tiles across both stage "
        "planes (default: preset value or 8)",
    )
    parser.add_argument(
        "--variant",
        choices=("llvm_ad_rgb", "cuda_ad_rgb"),
        default="llvm_ad_rgb",
        help="Mitsuba execution backend (default: llvm_ad_rgb, CPU)",
    )
    return parser.parse_args(argv)


def _load_mitsuba(variant: str) -> ModuleType:
    import importlib

    mi = importlib.import_module("mitsuba")
    if mi.variant() != variant:
        mi.set_variant(variant)
    return mi


def main() -> None:
    args = _parse_args()
    if args.stage_config is not None:
        stage = stage_config_from_json(args.stage_config)
    else:
        stage = StageConfig()
    stage = stage.with_cli_overrides(
        width=args.width,
        height=args.height,
        spp=args.spp,
        checker_scale=args.checker_scale,
    )

    volume = read_volume(args.optical_zarr)
    if not isinstance(volume, OpticalPropertyVolume):
        raise SystemExit(f"{args.optical_zarr} is not an optical property volume")

    config = MitsubaExportConfig(
        width=stage.render.width,
        height=stage.render.height,
        spp=stage.render.spp,
        variant=args.variant,
    )
    scene_dir = args.output_png.parent / f"{args.output_png.stem}_scene"
    prepared = prepare_mitsuba_scene(volume, scene_dir, config=config)

    mi = _load_mitsuba(config.variant)
    scene_dict = dict(prepared.scene_dict)
    apply_stage(mi, scene_dict, volume.geometry, stage)

    scene = mi.load_dict(scene_dict)
    image = mi.render(scene, seed=config.seed, spp=config.spp)

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    mi.util.write_bitmap(str(args.output_png), image, write_async=False)

    pixels = np.asarray(image, dtype=np.float32)
    print(
        "PIXELSTATS "
        f"min={float(np.min(pixels)):.6g} "
        f"max={float(np.max(pixels)):.6g} "
        f"mean={float(np.mean(pixels)):.6g} "
        f"std={float(np.std(pixels)):.6g}"
    )


if __name__ == "__main__":
    main()
