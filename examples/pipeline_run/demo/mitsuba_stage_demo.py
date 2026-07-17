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
Explicit ``--width/--height/--spp/--max-depth/--checker-scale`` arguments win
over the preset.

Invoke on the host (no Docker needed for Mitsuba):

    uv run --group mitsuba python \
        examples/pipeline_run/demo/mitsuba_stage_demo.py -- \
        OPTICAL_ZARR OUTPUT_PNG [--stage-config PRESET.stage.json] \
        [--width 512] [--height 512] [--spp 128] [--max-depth 8] \
        [--checker-scale 8] [--variant llvm_ad_rgb|cuda_ad_rgb] [--seed SEED]

``OPTICAL_ZARR`` is an ``optical.zarr`` bundle written by ``vdbmat run``.
``OUTPUT_PNG`` is where the rendered PNG is written; the exterior/interior PLY
meshes, ``capabilities.json``, and ``scene-summary.json`` that
``prepare_mitsuba_scene()`` writes as a side effect land next to it, under
``<OUTPUT_PNG stem>_scene/``. The script prints ``PIXELSTATS`` (min/max/mean/std of
the rendered pixels), the same cheap headless regression signal used by the
Blender demo scripts in this directory.

This script also replays a ``mitsuba_viewer_session`` manifest (see
:mod:`mitsuba_viewer_session` and :mod:`mitsuba_stage_viewer`) headlessly, using
the same resolver the viewer uses so the two never disagree about a path,
digest, or effective config:

    uv run --group mitsuba python \
        examples/pipeline_run/demo/mitsuba_stage_demo.py -- \
        --session SESSION.json --input-root ROOT [--preset-root ROOT] \
        [--mapping-root ROOT --mapping-work-root WORK_ROOT] \
        --output-png OUTPUT.png

Session replay resolves the input/preset references and verifies every
declared digest before rendering; it cannot be combined with the positional
``OPTICAL_ZARR OUTPUT_PNG`` form, ``--stage-config``, or any of the
width/height/spp/max-depth/checker-scale overrides, since the manifest is
already a fully-resolved effective config. An explicit ``--variant``/``--seed``
is accepted only if it matches the manifest's value, to catch stale command
lines rather than silently overriding what "session replay" means.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
from mitsuba_stage import StageConfig, apply_stage, stage_config_from_json
from mitsuba_stage_inputs import resolve_input_root
from mitsuba_stage_mappings import MappingCatalogError, resolve_mapping_root
from mitsuba_stage_presets import resolve_preset_root
from mitsuba_stage_regen import RegenError, regenerate_optical
from mitsuba_viewer_session import (
    ViewerSessionError,
    resolve_viewer_session,
    verify_derived_optical,
    viewer_session_from_json,
)

from vdbmat.core.volumes import OpticalPropertyVolume
from vdbmat.exporters.mitsuba import MitsubaExportConfig, prepare_mitsuba_scene
from vdbmat.io.zarr import read_volume
from vdbmat.pipeline import sha256_file


def _parse_args() -> argparse.Namespace:
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    parser = argparse.ArgumentParser(prog="mitsuba_stage_demo")
    parser.add_argument("optical_zarr", nargs="?", type=Path, default=None)
    parser.add_argument("output_png", nargs="?", type=Path, default=None)
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
        "--max-depth",
        type=int,
        default=None,
        help="override positive path depth limit (default: preset value or 8)",
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
        default=None,
        help="Mitsuba execution backend (legacy default: llvm_ad_rgb, CPU; "
        "with --session must match the manifest if given)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="legacy default: MitsubaExportConfig().seed; with --session must "
        "match the manifest if given",
    )
    parser.add_argument(
        "--session",
        type=Path,
        default=None,
        help="replay a viewer-session manifest instead of the legacy "
        "positional/override form",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="server-local root the session's input path resolves against "
        "(required with --session)",
    )
    parser.add_argument(
        "--preset-root",
        type=Path,
        default=None,
        help="server-local root the session's optional preset provenance "
        "resolves against (required if the session records one)",
    )
    parser.add_argument(
        "--mapping-root",
        type=Path,
        default=None,
        help="server-local root the session's optional optical mapping path "
        "resolves against (required for a mapping-bearing session)",
    )
    parser.add_argument(
        "--mapping-work-root",
        type=Path,
        default=None,
        help="directory for regenerated mapping-derived run bundles "
        "(required for a mapping-bearing session)",
    )
    parser.add_argument(
        "--output-png",
        dest="session_output_png",
        type=Path,
        default=None,
        help="output PNG path for --session replay",
    )
    args = parser.parse_args(argv)

    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be >= 0")

    if args.session is None:
        if args.optical_zarr is None or args.output_png is None:
            parser.error("OPTICAL_ZARR OUTPUT_PNG are required without --session")
        if (
            args.input_root is not None
            or args.preset_root is not None
            or args.mapping_root is not None
            or args.mapping_work_root is not None
            or args.session_output_png is not None
        ):
            parser.error(
                "--input-root/--preset-root/--mapping-root/--mapping-work-root/"
                "--output-png require --session"
            )
    else:
        if args.optical_zarr is not None or args.output_png is not None:
            parser.error(
                "positional OPTICAL_ZARR/OUTPUT_PNG cannot be used with --session"
            )
        if args.input_root is None:
            parser.error("--input-root is required with --session")
        if args.session_output_png is None:
            parser.error("--output-png is required with --session")
        if (args.mapping_root is None) != (args.mapping_work_root is None):
            parser.error("--mapping-root and --mapping-work-root must be used together")
        if args.stage_config is not None:
            parser.error("--stage-config cannot be used with --session")
        if any(
            value is not None
            for value in (
                args.width,
                args.height,
                args.spp,
                args.max_depth,
                args.checker_scale,
            )
        ):
            parser.error(
                "--width/--height/--spp/--max-depth/--checker-scale cannot be "
                "used with --session"
            )
    return args


def _load_mitsuba(variant: str) -> ModuleType:
    import importlib

    mi = importlib.import_module("mitsuba")
    if mi.variant() != variant:
        mi.set_variant(variant)
    return mi


def render_stage(
    optical_zarr: Path,
    output_png: Path,
    stage: StageConfig,
    variant: str,
    seed: int,
) -> np.ndarray:
    """Render one optical volume through a resolved stage; write and log it.

    Shared by the legacy CLI form and ``--session`` replay so the two never
    diverge in how a resolved (``optical_zarr``, ``stage``, ``variant``,
    ``seed``) tuple becomes pixels.
    """
    volume = read_volume(optical_zarr)
    if not isinstance(volume, OpticalPropertyVolume):
        raise SystemExit(f"{optical_zarr} is not an optical property volume")

    config = MitsubaExportConfig(
        width=stage.render.width,
        height=stage.render.height,
        spp=stage.render.spp,
        max_depth=stage.render.max_depth,
        variant=variant,
        seed=seed,
    )
    scene_dir = output_png.parent / f"{output_png.stem}_scene"
    prepared = prepare_mitsuba_scene(volume, scene_dir, config=config)

    mi = _load_mitsuba(config.variant)
    scene_dict = dict(prepared.scene_dict)
    apply_stage(mi, scene_dict, volume.geometry, stage)

    scene = mi.load_dict(scene_dict)
    image = mi.render(scene, seed=config.seed, spp=config.spp)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    mi.util.write_bitmap(str(output_png), image, write_async=False)

    pixels = np.asarray(image, dtype=np.float32)
    print(
        f"PIXELSTATS variant={config.variant} seed={config.seed} "
        f"max_depth={config.max_depth} "
        f"min={float(np.min(pixels)):.6g} "
        f"max={float(np.max(pixels)):.6g} "
        f"mean={float(np.mean(pixels)):.6g} "
        f"std={float(np.std(pixels)):.6g}"
    )
    return pixels


def _resolve_legacy(
    args: argparse.Namespace,
) -> tuple[Path, Path, StageConfig, str, int]:
    if args.stage_config is not None:
        stage = stage_config_from_json(args.stage_config)
    else:
        stage = StageConfig()
    stage = stage.with_cli_overrides(
        width=args.width,
        height=args.height,
        spp=args.spp,
        max_depth=args.max_depth,
        checker_scale=args.checker_scale,
    )
    variant = args.variant or "llvm_ad_rgb"
    seed = MitsubaExportConfig().seed if args.seed is None else args.seed
    assert args.optical_zarr is not None
    assert args.output_png is not None
    return args.optical_zarr, args.output_png, stage, variant, seed


def _resolve_session(
    args: argparse.Namespace,
) -> tuple[Path, Path, StageConfig, str, int]:
    assert args.session is not None
    assert args.input_root is not None
    assert args.session_output_png is not None
    session = viewer_session_from_json(args.session)
    input_root = resolve_input_root(args.input_root, args.input_root)
    preset_root = resolve_preset_root(args.preset_root, None)
    if session.preset is not None and args.preset_root is None:
        raise ViewerSessionError(
            "resolve", "stage preset reference requires --preset-root"
        )
    mapping_root: Path | None = None
    if session.mapping is not None:
        if args.mapping_root is None or args.mapping_work_root is None:
            raise ViewerSessionError(
                "resolve",
                "mapping-bearing session requires --mapping-root and "
                "--mapping-work-root",
            )
        try:
            mapping_root = resolve_mapping_root(args.mapping_root)
        except MappingCatalogError as error:
            raise ViewerSessionError("resolve", str(error)) from error
        _require_disjoint_roots(args.mapping_work_root, input_root)
    resolved = resolve_viewer_session(session, input_root, preset_root, mapping_root)
    if args.variant is not None and args.variant != resolved.variant:
        raise ViewerSessionError(
            "resolve",
            f"--variant {args.variant} does not match session variant "
            f"{resolved.variant}",
        )
    if args.seed is not None and args.seed != resolved.seed:
        raise ViewerSessionError(
            "resolve", f"--seed {args.seed} does not match session seed {resolved.seed}"
        )
    optical_zarr = resolved.optical_zarr
    if resolved.session.mapping is not None:
        assert resolved.mapping_candidate is not None
        assert args.mapping_work_root is not None
        try:
            derived = regenerate_optical(
                resolved.input_candidate.path,
                resolved.mapping_candidate,
                args.mapping_work_root,
            )
        except RegenError as error:
            raise ViewerSessionError(error.stage, error.message) from error
        verify_derived_optical(resolved, derived.optical_zarr)
        optical_zarr = derived.optical_zarr
        cache_status = "reused" if derived.reused else "generated"
        print(
            f"MAPPING {resolved.session.mapping.path} "
            f"digest={derived.mapping_digest} cache={cache_status}"
        )
    return (
        optical_zarr,
        args.session_output_png,
        resolved.stage_config,
        resolved.variant,
        resolved.seed,
    )


def _require_disjoint_roots(mapping_work_root: Path, input_root: Path) -> None:
    resolved_work = mapping_work_root.resolve()
    resolved_input = input_root.resolve()
    if (
        resolved_work == resolved_input
        or resolved_work.is_relative_to(resolved_input)
        or resolved_input.is_relative_to(resolved_work)
    ):
        raise ViewerSessionError(
            "resolve",
            "--mapping-work-root and --input-root must not overlap",
        )


def main() -> None:
    args = _parse_args()
    if args.session is None:
        optical_zarr, output_png, stage, variant, seed = _resolve_legacy(args)
    else:
        try:
            optical_zarr, output_png, stage, variant, seed = _resolve_session(args)
        except ViewerSessionError as error:
            raise SystemExit(
                f"session replay failed at {error.stage}: {error.message}"
            ) from error
        # The input session itself is the tracking document for this replay
        # (unlike the viewer's final-render sidecar, which needs one written
        # next to the PNG) — just log which one and its digest, so a replay
        # PNG can still be traced back to its exact session file on disk.
        print(f"RENDER session={args.session} digest={sha256_file(args.session)}")
    render_stage(optical_zarr, output_png, stage, variant, seed)


if __name__ == "__main__":
    main()
