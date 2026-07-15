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

Invoke on the host (no Docker needed for Mitsuba):

    uv run --group mitsuba python \
        examples/pipeline_run/demo/mitsuba_stage_demo.py -- \
        OPTICAL_ZARR OUTPUT_PNG [--width 512] [--height 512] [--spp 128] \
        [--checker-scale 8]

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

from vdbmat.core.geometry import GridGeometry
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
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--spp", type=int, default=128)
    parser.add_argument(
        "--checker-scale",
        type=int,
        default=8,
        help="number of checkerboard tiles across each stage plane",
    )
    return parser.parse_args(argv)


def _load_mitsuba(variant: str) -> ModuleType:
    import importlib

    mi = importlib.import_module("mitsuba")
    if mi.variant() != variant:
        mi.set_variant(variant)
    return mi


def _scene_bounds(
    geometry: GridGeometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Recompute the same center/radius/camera_direction frame as the exporter.

    Reimplemented here (rather than importing the exporter's private
    ``_scene_frame``) per the plan's boundary: this demo only reads the public
    ``GridGeometry`` API, never private exporter helpers.
    """
    corners = np.asarray(
        [
            geometry.continuous_index_to_world((x, y, z))
            for x in (0, geometry.shape_xyz[0])
            for y in (0, geometry.shape_xyz[1])
            for z in (0, geometry.shape_xyz[2])
        ],
        dtype=np.float64,
    )
    minimum = np.min(corners, axis=0)
    maximum = np.max(corners, axis=0)
    center = (minimum + maximum) * 0.5
    radius = float(np.linalg.norm(maximum - minimum) * 0.5)
    camera_direction = np.asarray((1.6, -2.2, 1.4), dtype=np.float64)
    camera_direction /= np.linalg.norm(camera_direction)
    return minimum, maximum, center, radius, camera_direction


def _checkerboard_bsdf(
    mi: ModuleType,
    checker_scale: int,
    color0: tuple[float, float, float],
    color1: tuple[float, float, float],
) -> dict[str, object]:
    return {
        "type": "diffuse",
        "reflectance": {
            "type": "checkerboard",
            "color0": {"type": "rgb", "value": list(color0)},
            "color1": {"type": "rgb", "value": list(color1)},
            "to_uv": mi.ScalarTransform4f.scale(
                [float(checker_scale), float(checker_scale), 1.0]
            ),
        },
    }


def _add_stage(
    mi: ModuleType,
    scene: dict[str, object],
    geometry: GridGeometry,
    checker_scale: int,
) -> None:
    """Add a checkerboard backdrop, checkerboard floor, and a key light.

    Additive only: the medium, exterior/interior meshes, sensor, and backlight
    entries already in ``scene`` (from ``prepare_mitsuba_scene``) are untouched.
    """
    minimum, _maximum, center, radius, camera_direction = _scene_bounds(geometry)

    # Backdrop: a diffuse checkerboard wall behind the object, nearer to the
    # object than the canonical white backlight rectangle (which sits at
    # radius * 4 and already fills the camera frame at that distance, so
    # anything placed further back would be fully hidden behind it).
    backdrop_distance = radius * 2.2
    backdrop_position = center - camera_direction * backdrop_distance
    scene["stage_backdrop"] = {
        "type": "rectangle",
        "to_world": mi.ScalarTransform4f.look_at(
            origin=backdrop_position.tolist(),
            target=center.tolist(),
            up=[0.0, 0.0, 1.0],
        )
        @ mi.ScalarTransform4f.scale([radius * 2.6, radius * 2.6, 1.0]),
        # Teal/orange: distinct in hue (not just value) from the floor below,
        # so a viewer can tell which surface is being seen through a
        # refracted/distorted patch instead of everything reading as one grey
        # blur.
        "bsdf": _checkerboard_bsdf(
            mi, checker_scale, color0=(0.02, 0.09, 0.11), color1=(0.85, 0.5, 0.12)
        ),
    }

    # Floor: a horizontal diffuse checkerboard plane below the object's lower
    # world-space bound. The default rectangle normal (0, 0, 1) already faces
    # up, matching the up=[0, 0, 1] convention used by the sensor/backlight, so
    # no rotation is needed.
    floor_z = minimum[2] - radius * 0.1
    scene["stage_floor"] = {
        "type": "rectangle",
        "to_world": mi.ScalarTransform4f.translate(
            [float(center[0]), float(center[1]), float(floor_z)]
        )
        @ mi.ScalarTransform4f.scale([radius * 6.0, radius * 6.0, 1.0]),
        "bsdf": _checkerboard_bsdf(
            mi, checker_scale, color0=(0.03, 0.03, 0.13), color1=(0.82, 0.76, 0.14)
        ),
    }

    # Key light: a small area light from an oblique angle (distinct from the
    # camera and from the backlight's straight-behind position) so the stage
    # and object read with visible shading/shadow, not just backlit silhouette.
    key_light_direction = np.asarray((-1.0, -1.5, 2.1), dtype=np.float64)
    key_light_direction /= np.linalg.norm(key_light_direction)
    key_light_position = center + key_light_direction * radius * 3.5
    scene["stage_key_light"] = {
        "type": "rectangle",
        "to_world": mi.ScalarTransform4f.look_at(
            origin=key_light_position.tolist(),
            target=center.tolist(),
            up=[0.0, 0.0, 1.0],
        )
        @ mi.ScalarTransform4f.scale([radius * 1.0, radius * 1.0, 1.0]),
        "emitter": {
            "type": "area",
            # Slightly warm rather than pure white, so it reads as a distinct
            # source from the canonical backlight and casts a faint colour
            # cue on the stage/object instead of everything looking lit by
            # one flat white source.
            "radiance": {"type": "rgb", "value": [6.4, 5.6, 4.2]},
        },
    }


def main() -> None:
    args = _parse_args()
    volume = read_volume(args.optical_zarr)
    if not isinstance(volume, OpticalPropertyVolume):
        raise SystemExit(f"{args.optical_zarr} is not an optical property volume")

    config = MitsubaExportConfig(width=args.width, height=args.height, spp=args.spp)
    scene_dir = args.output_png.parent / f"{args.output_png.stem}_scene"
    prepared = prepare_mitsuba_scene(volume, scene_dir, config=config)

    mi = _load_mitsuba(config.variant)
    scene_dict = dict(prepared.scene_dict)
    _add_stage(mi, scene_dict, volume.geometry, args.checker_scale)

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
