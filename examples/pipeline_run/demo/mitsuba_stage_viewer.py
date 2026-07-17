"""Browser GUI for tuning the Mitsuba stage (viser + live re-render).

This is a demo-track helper; it is not part of the canonical pipeline and
produces qualitative, uncalibrated images. ``prepare_mitsuba_scene()`` /
``render_mitsuba()`` / ``MitsubaExportConfig`` are untouched, and this viewer
is a pure consumer of the Phase-1 contract in :mod:`mitsuba_stage`: it edits a
``StageConfig``, previews it, and its durable outputs are a ``*.stage.json``
preset, a digest-pinned ``*.session.json`` manifest, and a final PNG.

Architecture (see .devdocs/vision/mitsuba_gui/p3/plan.md):

- ``prepare_mitsuba_scene()`` runs exactly twice at startup — once with a
  low-resolution preview sensor, once at the final resolution — so the heavy
  boundary-mesh extraction and PLY writing never sit inside the parameter
  loop. The final scene is rebuilt only when its requested resolution or
  max depth changes.
- The preview scene stays loaded. Continuous changes update explicit
  ``mi.traverse()`` keys; graph changes (enabled/pattern/override toggles)
  rebuild through ``apply_stage()`` and then resume traversed updates.
- All renders run on one worker thread. A change produces a low-spp
  interactive image immediately and a settled high-spp image after the input
  goes quiet. Generation guards prevent stale renders from reaching the GUI.
- Preview renders override only width/height/spp before calling
  ``apply_stage``; max depth remains the configured transport setting. A depth
  change is a planned scene rebuild because Mitsuba does not expose it through
  traversal. Final renders use the same width/height/spp/max depth/seed as the
  headless demo, so a saved preset reproduces the final PNG pixel-identically.
- GUI decompositions (radiance = colour picker x intensity slider, key-light
  direction = azimuth/elevation sliders) exist only inside the GUI. The saved
  JSON is the unchanged Phase-1 schema, and fields the user never touched are
  written back exactly as loaded (per-field dirty tracking), so lossy
  widget quantisation cannot creep into an untouched preset field.

Invoke on the host (no Docker needed for Mitsuba):

    uv run --group mitsuba-viewer python \
        examples/pipeline_run/demo/mitsuba_stage_viewer.py -- \
        [OPTICAL_ZARR_OR_BUNDLE] [--input-root DIR] \
        [--stage-config PRESET.stage.json] [--preset-root DIR] [--port 8080] \
        [--session SESSION.json] [--session-root DIR] [--seed SEED] \
        [--work-dir DIR] [--preview-size 256] [--preview-spp 16] \
        [--interactive-spp 4] [--settle-delay 0.35] \
        [--variant llvm_ad_rgb|cuda_ad_rgb] \
        [--preset-out PATH] [--final-out PATH]

``--work-dir`` (default: a fresh temp directory) receives the PLY/scene
side-effect files and the default preset/PNG outputs; point it at
``.local/...`` to keep artifacts with the repo checkout.

The positional argument is the initial input: either a single
``optical.zarr`` store, or a canonical run bundle directory (containing
``run.json``; its ``optical.zarr`` member is read). ``--input-root``
(default: the initial input's parent directory) is the server-local
directory the GUI's Input tab scans for sibling bundles/stores to switch
to via Load/Rebuild; see :mod:`mitsuba_stage_inputs` for the catalog
contract and containment rules.
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType

import numpy as np
from mitsuba_stage import (
    RGB,
    BackdropSettings,
    BacklightOverride,
    CameraOverride,
    FloorSettings,
    KeyLightSettings,
    RenderSettings,
    StageConfig,
    apply_stage,
    stage_config_from_json,
    stage_config_to_dict,
)
from mitsuba_stage_inputs import (
    InputCandidate,
    InputCatalogError,
    InputKind,
    describe_candidate,
    resolve_candidate,
    resolve_input_root,
    scan_input_catalog,
)
from mitsuba_stage_mappings import (
    MappingCandidate,
    MappingCatalogError,
    describe_mapping,
    load_mapping,
    resolve_mapping_candidate,
    resolve_mapping_root,
    scan_mapping_catalog,
)
from mitsuba_stage_presets import (
    PresetCatalogError,
    describe_preset,
    load_preset,
    resolve_preset,
    resolve_preset_root,
    scan_preset_catalog,
    stage_config_digest,
)
from mitsuba_stage_regen import RegenError, regenerate_optical
from mitsuba_viewer_session import (
    SessionMappingRef,
    SessionPresetRef,
    ViewerSessionError,
    create_viewer_session,
    resolve_viewer_session,
    verify_derived_optical,
    viewer_session_from_json,
    write_viewer_session,
)

from vdbmat.core.volumes import OpticalPropertyVolume
from vdbmat.exporters.mitsuba import MitsubaExportConfig, prepare_mitsuba_scene
from vdbmat.io.zarr import read_volume
from vdbmat.pipeline import zarr_store_sha256


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    parser = argparse.ArgumentParser(prog="mitsuba_stage_viewer")
    parser.add_argument(
        "optical_zarr",
        nargs="?",
        type=Path,
        help="initial input: an optical.zarr store, or a run bundle "
        "directory (containing run.json)",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="server-local directory the Input tab scans for switchable "
        "bundles/stores (default: the initial input's parent directory)",
    )
    parser.add_argument(
        "--preset-root",
        type=Path,
        default=None,
        help="server-local directory the Preset tab scans for *.stage.json "
        "files (default: --stage-config parent, or builtin demo presets)",
    )
    parser.add_argument("--stage-config", type=Path, default=None)
    parser.add_argument(
        "--mapping-root",
        type=Path,
        default=None,
        help="server-local directory the Input tab's mapping dropdown scans "
        "for *.optical-mapping.json files (default: checked-in demo mappings)",
    )
    parser.add_argument(
        "--mapping-work-root",
        type=Path,
        default=None,
        help="server-local directory for derived (mapping-applied) run "
        "bundles (default: WORK_DIR/derived); must not overlap --input-root",
    )
    parser.add_argument(
        "--session",
        type=Path,
        default=None,
        help="restore a viewer session; requires --input-root",
    )
    parser.add_argument(
        "--session-root",
        type=Path,
        default=None,
        help="root for GUI session Save/Load paths",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="directory for scene side-effect files and default outputs "
        "(default: a fresh temp directory)",
    )
    parser.add_argument("--preview-size", type=int, default=256)
    parser.add_argument("--preview-spp", type=int, default=16)
    parser.add_argument("--interactive-spp", type=int, default=4)
    parser.add_argument("--settle-delay", type=float, default=0.35)
    parser.add_argument(
        "--variant",
        choices=("llvm_ad_rgb", "cuda_ad_rgb"),
        default=None,
        help="Mitsuba execution backend (legacy default: llvm_ad_rgb, CPU)",
    )
    parser.add_argument(
        "--preset-out",
        type=Path,
        default=None,
        help="default path for 'Save preset' (default: WORK_DIR/viewer.stage.json)",
    )
    parser.add_argument(
        "--final-out",
        type=Path,
        default=None,
        help="default path for 'Render final' (default: WORK_DIR/final.png)",
    )
    args = parser.parse_args(argv)
    if args.preview_size <= 0 or args.preview_spp <= 0:
        parser.error("--preview-size and --preview-spp must be > 0")
    if args.interactive_spp <= 0:
        parser.error("--interactive-spp must be > 0")
    if args.interactive_spp > args.preview_spp:
        parser.error("--interactive-spp must be <= --preview-spp")
    if args.settle_delay < 0.0:
        parser.error("--settle-delay must be >= 0")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be >= 0")
    if args.session is None and args.optical_zarr is None:
        parser.error("OPTICAL_ZARR_OR_BUNDLE is required without --session")
    if args.session is not None:
        if args.optical_zarr is not None:
            parser.error("OPTICAL_ZARR_OR_BUNDLE cannot be used with --session")
        if args.input_root is None:
            parser.error("--input-root is required with --session")
        if args.stage_config is not None:
            parser.error("--stage-config cannot be used with --session")
    return args


def _resolve_session_root(
    cli_root: Path | None, startup_session: Path | None, work_dir: Path
) -> Path:
    candidate = cli_root
    if candidate is None and startup_session is not None:
        candidate = startup_session.resolve().parent
    if candidate is None:
        candidate = work_dir
    root = candidate.resolve()
    if not root.is_dir():
        raise ViewerSessionError("resolve", f"session root is not a directory: {root}")
    return root


def _resolve_session_path(root: Path, user_path: Path, *, must_exist: bool) -> Path:
    """Resolve one root-relative session path without permitting root escape."""
    if user_path.is_absolute() or not user_path.parts or user_path == Path("."):
        raise ViewerSessionError("resolve", "session path must be root-relative")
    if ".." in user_path.parts:
        raise ViewerSessionError("resolve", "session path must not contain '..'")
    candidate = root / user_path
    resolved = candidate.resolve(strict=must_exist)
    if not resolved.is_relative_to(root.resolve()):
        raise ViewerSessionError(
            "resolve", f"session path escapes session root: {user_path}"
        )
    if must_exist and not resolved.is_file():
        raise ViewerSessionError("resolve", f"session file does not exist: {user_path}")
    return resolved


@dataclass(frozen=True, slots=True)
class SessionDerivation:
    """Committed record of a mapping applied to the session's source input.

    ``derivation is None`` on an :class:`InputSession` means the session
    renders its bundle's own ``optical.zarr`` as-is. When present, the Input
    tab's dropdown still names ``source_candidate`` (root-relative, inside
    ``--input-root``) as "the input"; ``derived_bundle`` is the actual
    optical volume being rendered.
    """

    source_candidate: InputCandidate
    mapping_candidate: MappingCandidate
    mapping_digest: str
    derived_bundle: Path


def _require_disjoint_roots(a: Path, b: Path, *, label: str) -> None:
    """Reject a ``--mapping-work-root`` that overlaps ``--input-root``.

    ``regenerate_optical()`` publishes derived bundles with ``overwrite=True``
    at a path under its work root; if that work root were inside (or
    contained) the input root, a derived bundle could collide with or shadow
    a real catalog entry.
    """
    resolved_a = a.resolve()
    resolved_b = b.resolve()
    if (
        resolved_a == resolved_b
        or resolved_a.is_relative_to(resolved_b)
        or resolved_b.is_relative_to(resolved_a)
    ):
        raise ViewerSessionError("resolve", f"{label}: {a} and {b} must not overlap")


@dataclass(frozen=True, slots=True)
class ViewerStartup:
    initial_input: Path
    initial_config: StageConfig
    input_root: Path
    preset_root: Path
    mapping_root: Path
    mapping_work_root: Path
    session_root: Path
    variant: str
    seed: int
    applied_preset: SessionPresetRef | None
    initial_derivation: SessionDerivation | None = None


def _startup_preset_ref(
    stage_config_path: Path | None,
    config: StageConfig,
    preset_root: Path,
) -> SessionPresetRef | None:
    if stage_config_path is None:
        return None
    try:
        relative = stage_config_path.resolve().relative_to(preset_root)
        candidate = resolve_preset(preset_root, relative)
        loaded = load_preset(candidate)
    except (OSError, PresetCatalogError, ValueError):
        return None
    digest = stage_config_digest(config)
    if stage_config_digest(loaded) != digest:
        return None
    return SessionPresetRef(path=candidate.root_relative, digest=digest)


def _resolve_viewer_startup(args: argparse.Namespace, work_dir: Path) -> ViewerStartup:
    """Resolve all startup state before importing Mitsuba or building scenes."""
    mapping_root = resolve_mapping_root(args.mapping_root)
    mapping_work_root = (
        args.mapping_work_root
        if args.mapping_work_root is not None
        else work_dir / "derived"
    ).resolve()

    if args.session is None:
        assert args.optical_zarr is not None
        initial = (
            stage_config_from_json(args.stage_config)
            if args.stage_config is not None
            else StageConfig()
        )
        preset_root = resolve_preset_root(args.preset_root, args.stage_config)
        input_root = resolve_input_root(args.input_root, args.optical_zarr)
        _require_disjoint_roots(
            mapping_work_root, input_root, label="--mapping-work-root/--input-root"
        )
        session_root = _resolve_session_root(
            None if args.session_root is None else args.session_root, None, work_dir
        )
        return ViewerStartup(
            initial_input=args.optical_zarr.resolve(),
            initial_config=initial,
            input_root=input_root,
            preset_root=preset_root,
            mapping_root=mapping_root,
            mapping_work_root=mapping_work_root,
            session_root=session_root,
            variant=args.variant or "llvm_ad_rgb",
            seed=MitsubaExportConfig().seed if args.seed is None else args.seed,
            applied_preset=_startup_preset_ref(args.stage_config, initial, preset_root),
        )

    startup_path = args.session.resolve()
    session_root = _resolve_session_root(args.session_root, startup_path, work_dir)
    session_user_path = (
        Path(startup_path.name) if args.session_root is None else args.session
    )
    session_path = _resolve_session_path(
        session_root, session_user_path, must_exist=True
    )
    session = viewer_session_from_json(session_path)
    preset_root = resolve_preset_root(args.preset_root, None)
    if session.preset is not None and args.preset_root is None:
        raise ViewerSessionError(
            "resolve", "stage preset reference requires --preset-root"
        )
    if session.mapping is not None and args.mapping_root is None:
        raise ViewerSessionError("resolve", "mapping reference requires --mapping-root")
    assert args.input_root is not None
    input_root = resolve_input_root(args.input_root, args.input_root)
    _require_disjoint_roots(
        mapping_work_root, input_root, label="--mapping-work-root/--input-root"
    )
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

    initial_input = resolved.input_candidate.path
    initial_derivation: SessionDerivation | None = None
    if resolved.session.mapping is not None:
        assert resolved.mapping_candidate is not None
        try:
            derived = regenerate_optical(
                resolved.input_candidate.path,
                resolved.mapping_candidate,
                mapping_work_root,
            )
        except RegenError as error:
            raise ViewerSessionError(error.stage, error.message) from error
        verify_derived_optical(resolved, derived.optical_zarr)
        initial_input = derived.bundle_path
        initial_derivation = SessionDerivation(
            source_candidate=resolved.input_candidate,
            mapping_candidate=resolved.mapping_candidate,
            mapping_digest=resolved.session.mapping.digest,
            derived_bundle=derived.bundle_path,
        )

    return ViewerStartup(
        initial_input=initial_input,
        initial_config=resolved.stage_config,
        input_root=input_root,
        preset_root=preset_root,
        mapping_root=mapping_root,
        mapping_work_root=mapping_work_root,
        session_root=session_root,
        variant=resolved.variant,
        seed=resolved.seed,
        applied_preset=resolved.session.preset,
        initial_derivation=initial_derivation,
    )


def _load_mitsuba(variant: str) -> ModuleType:
    import importlib

    mi = importlib.import_module("mitsuba")
    if mi.variant() != variant:
        mi.set_variant(variant)
    return mi


def _pixel_stats(pixels: np.ndarray, max_depth: int) -> str:
    return (
        f"max_depth={max_depth} "
        f"min={float(np.min(pixels)):.6g} "
        f"max={float(np.max(pixels)):.6g} "
        f"mean={float(np.mean(pixels)):.6g} "
        f"std={float(np.std(pixels)):.6g}"
    )


StructureKey = tuple[bool, str, bool, str, bool, bool, bool, int]
FinalRenderKey = tuple[int, int, int]


def _structure_key(config: StageConfig) -> StructureKey:
    return (
        config.backdrop.enabled,
        config.backdrop.pattern,
        config.floor.enabled,
        config.floor.pattern,
        config.key_light.enabled,
        config.camera is not None,
        config.backlight is not None,
        config.render.max_depth,
    )


def _preview_stage_config(
    config: StageConfig, preview_size: int, preview_spp: int
) -> StageConfig:
    """Override preview sampling fields while preserving render transport."""
    render = replace(
        config.render,
        width=preview_size,
        height=preview_size,
        spp=preview_spp,
    )
    return replace(config, render=render)


def _final_render_key(render: RenderSettings) -> FinalRenderKey:
    return render.width, render.height, render.max_depth


_INPUTS_DIRNAME = "inputs"
_OPTICAL_ASSET_NAME = "optical.zarr"
_RUN_MANIFEST_NAME = "run.json"
_SLUG_PATTERN = re.compile(r"[^a-zA-Z0-9]+")


def _resolve_initial_optical_zarr(initial_input: Path) -> Path:
    """Resolve the CLI positional argument to an actual ``optical.zarr`` store.

    Accepts either a bare ``optical.zarr`` store or a canonical run bundle
    directory (containing ``run.json``), matching the bundle/store
    distinction ``mitsuba_stage_inputs`` uses for catalog candidates. Kept
    independent of that module's candidate/containment logic since the CLI's
    initial input is exempt from the ``--input-root`` containment rule.
    """
    if (initial_input / _RUN_MANIFEST_NAME).is_file():
        return initial_input / _OPTICAL_ASSET_NAME
    return initial_input


def _slug_for(optical_zarr: Path) -> str:
    """A short, filesystem-safe identifier derived from an input path."""
    base = (
        optical_zarr.parent.name
        if optical_zarr.name == _OPTICAL_ASSET_NAME
        else optical_zarr.stem
    )
    slug = _SLUG_PATTERN.sub("-", base).strip("-").lower()
    return slug or "input"


def _session_work_dir(work_dir: Path, seq: int, optical_zarr: Path) -> Path:
    """Return the per-input work directory for one session.

    Every session, including the initial one, gets a freshly numbered
    directory under ``work_dir/inputs/``, so PLY/scene-summary artefacts from
    different inputs (or repeated loads of the same input) never collide.
    ``work_dir``'s internal layout is a scratch side effect, not an external
    contract, so this scheme is free to differ from Phase 1's fixed
    ``preview_scene`` / ``final_scene`` paths.
    """
    return work_dir / _INPUTS_DIRNAME / f"{seq:03d}-{_slug_for(optical_zarr)}"


_LOAD_STAGES = ("validate", "map", "prepare", "load", "smoke", "swap")
_NO_PRESETS = "(no presets found)"
_AS_IS_MAPPING = "(bundle optical as-is)"


class _StageTimer:
    """Times the stages of one transaction (Load/Rebuild, session load, ...).

    Wraps an ``on_stage`` callback: each call logs the just-finished stage's
    elapsed seconds as a ``STAGE <transaction> <stage> <elapsed_s>`` stdout
    line (the cancel-adoption and Phase 5 report measurement source) and
    returns a status suffix like ``" (map 3.2s)"`` for the caller to append
    to its next status message. Stage names and transition order are exactly
    what the caller already passes; this only adds timing, so it does not
    change any existing stage-sequence test.
    """

    def __init__(self, transaction: str) -> None:
        self.transaction = transaction
        self._stage: str | None = None
        self._stage_started = time.perf_counter()
        self._total_started = self._stage_started

    def advance(self, stage: str) -> str:
        """Log the just-finished stage (if any) and start timing ``stage``.

        Returns "" for the first stage of the transaction, else a status
        suffix describing the previous stage's elapsed time.
        """
        now = time.perf_counter()
        suffix = ""
        if self._stage is not None:
            elapsed = now - self._stage_started
            print(f"STAGE {self.transaction} {self._stage} {elapsed:.3f}")
            suffix = f" ({self._stage} {elapsed:.1f}s)"
        self._stage = stage
        self._stage_started = now
        return suffix

    def finish(self) -> float:
        """Log the final stage and return the transaction's total elapsed."""
        now = time.perf_counter()
        if self._stage is not None:
            elapsed = now - self._stage_started
            print(f"STAGE {self.transaction} {self._stage} {elapsed:.3f}")
        return now - self._total_started


class InputLoadError(Exception):
    """Load/Rebuild failed at a named stage; the current session is untouched.

    Any failure before the ``swap`` stage discards the half-built session and
    leaves ``StageCore``'s current session, preview image, and GUI settings
    exactly as they were.
    """

    def __init__(self, stage: str, message: str) -> None:
        assert stage in _LOAD_STAGES
        super().__init__(f"input load failed at {stage}: {message}")
        self.stage = stage
        self.message = message


def _discard_session_dir(session_dir: Path) -> None:
    """Best-effort removal of a session directory abandoned by a failed load."""
    try:
        if session_dir.exists():
            shutil.rmtree(session_dir)
    except OSError as error:
        print(
            f"INPUT LOAD CLEANUP WARNING: could not remove {session_dir}: {error}",
            file=sys.stderr,
        )


def _nested(mapping: dict[str, object], path: str) -> object:
    value: object = mapping
    for component in path.split("."):
        if not isinstance(value, dict) or component not in value:
            raise KeyError(path)
        value = value[component]
    return value


class TraversedPreviewScene:
    """A loaded preview scene updated through an explicit traverse mapping."""

    def __init__(self, mi, base, geometry, initial: StageConfig, seed: int) -> None:
        self.mi = mi
        self.base = base
        self.geometry = geometry
        self.seed = seed
        self.scene = None
        self.params = None
        self.config: StageConfig | None = None
        self.rebuild(initial)

    def _stage_dict(self, config: StageConfig) -> dict[str, object]:
        scene_dict = dict(self.base.scene_dict)
        integrator = scene_dict.get("integrator")
        if not isinstance(integrator, dict):
            raise TypeError("base scene has no integrator dict")
        scene_dict["integrator"] = {
            **integrator,
            "max_depth": config.render.max_depth,
        }
        apply_stage(self.mi, scene_dict, self.geometry, config)
        return scene_dict

    def rebuild(self, config: StageConfig) -> None:
        self.scene = self.mi.load_dict(self._stage_dict(config))
        self.params = self.mi.traverse(self.scene)
        self.config = config

    def _set(self, key: str, value: object) -> None:
        assert self.params is not None
        if key not in self.params:
            raise KeyError(f"traverse key unavailable: {key}")
        self.params[key] = value

    def _apply_continuous(self, config: StageConfig) -> None:
        assert self.config is not None
        candidate = self._stage_dict(config)
        old = self.config

        for name, section in (
            ("stage_backdrop", config.backdrop),
            ("stage_floor", config.floor),
        ):
            if not section.enabled:
                continue
            old_section = old.backdrop if name == "stage_backdrop" else old.floor
            if section.pattern == "checker":
                if section.color0 != old_section.color0:
                    self._set(f"{name}.bsdf.reflectance.color0.value", section.color0)
                if section.color1 != old_section.color1:
                    self._set(f"{name}.bsdf.reflectance.color1.value", section.color1)
                if section.checker_scale != old_section.checker_scale:
                    transform = self.mi.ScalarTransform3f.scale(
                        [float(section.checker_scale), float(section.checker_scale)]
                    )
                    self._set(f"{name}.bsdf.reflectance.to_uv", transform)
            elif section.color0 != old_section.color0:
                self._set(f"{name}.bsdf.reflectance.value", section.color0)

        transform_fields = (
            (
                "stage_backdrop",
                config.backdrop.enabled
                and (
                    config.backdrop.distance_factor != old.backdrop.distance_factor
                    or config.backdrop.scale_factor != old.backdrop.scale_factor
                ),
            ),
            (
                "stage_floor",
                config.floor.enabled
                and (
                    config.floor.drop_factor != old.floor.drop_factor
                    or config.floor.scale_factor != old.floor.scale_factor
                ),
            ),
            (
                "stage_key_light",
                config.key_light.enabled
                and (
                    config.key_light.direction != old.key_light.direction
                    or config.key_light.distance_factor != old.key_light.distance_factor
                    or config.key_light.scale_factor != old.key_light.scale_factor
                ),
            ),
        )
        for name, changed in transform_fields:
            if changed:
                self._set(f"{name}.to_world", _nested(candidate, f"{name}.to_world"))

        if (
            config.key_light.enabled
            and config.key_light.radiance != old.key_light.radiance
        ):
            self._set(
                "stage_key_light.emitter.radiance.value", config.key_light.radiance
            )
        if config.backlight is not None and config.backlight != old.backlight:
            self._set("backlight.emitter.radiance.value", config.backlight.radiance)
        if config.camera is not None and config.camera != old.camera:
            self._set("sensor.to_world", _nested(candidate, "sensor.to_world"))
            if config.camera.fov_deg != old.camera.fov_deg:
                self._set("sensor.x_fov", float(config.camera.fov_deg))

        assert self.params is not None
        self.params.update()
        self.config = config

    def render(self, config: StageConfig, spp: int) -> tuple[np.ndarray, str]:
        route = "traverse"
        assert self.config is not None
        if _structure_key(config) != _structure_key(self.config):
            self.rebuild(config)
            route = "rebuild"
        elif config != self.config:
            try:
                self._apply_continuous(config)
            except (KeyError, TypeError, RuntimeError, ValueError) as error:
                print(f"TRAVERSE FALLBACK {error}", file=sys.stderr)
                self.rebuild(config)
                route = "rebuild-fallback"
        assert self.scene is not None
        return self.mi.render(self.scene, seed=self.seed, spp=spp), route


@dataclass
class InputSession:
    """Everything bound to one loaded optical volume: the swap unit.

    Building one is the expensive, input-dependent part of ``StageCore``
    (boundary-mesh extraction, PLY writing, preview scene load). Nothing
    outside :class:`StageCore` holds a reference across a swap, so replacing
    this object is exactly what "switch input" means.
    """

    optical_zarr: Path
    work_dir: Path
    volume: OpticalPropertyVolume
    seed: int
    preview_scene: TraversedPreviewScene
    base_final: object | None = None
    final_key: FinalRenderKey | None = None
    derivation: SessionDerivation | None = None


class StageCore:
    """Viser-free rendering core: prepare twice, then cheap re-renders.

    Kept independent of the GUI so the render/save/reproduce paths can be
    exercised by scripts (verification) as well as by the viser bindings.

    Process-wide state (the Mitsuba module/variant, preview size/spp) lives
    directly on ``StageCore``; everything bound to one input lives on the
    current :class:`InputSession`, reachable through ``self._session``.
    ``session_generation`` increments on every :meth:`swap_session` so a
    render worker can detect and drop a result computed against a session
    that is no longer current — a defense-in-depth guard on top of the
    primary protection, which is that the render worker itself only ever
    runs one job (including a swap) at a time.
    """

    def __init__(
        self,
        optical_zarr: Path,
        work_dir: Path,
        preview_size: int,
        preview_spp: int,
        initial: StageConfig,
        variant: str = "llvm_ad_rgb",
        seed: int = MitsubaExportConfig().seed,
    ) -> None:
        self.work_dir = work_dir
        self.preview_size = preview_size
        self.preview_spp = preview_spp
        self.mi = _load_mitsuba(variant)
        self.session_generation = 0
        self._session_seq = itertools.count()
        self._session = self._build_session(optical_zarr, initial, seed)

    @property
    def current_session(self) -> InputSession:
        return self._session

    def _build_session(
        self,
        optical_zarr: Path,
        initial: StageConfig,
        seed: int = MitsubaExportConfig().seed,
    ) -> InputSession:
        volume = read_volume(optical_zarr)
        if not isinstance(volume, OpticalPropertyVolume):
            raise SystemExit(f"{optical_zarr} is not an optical property volume")
        session_dir = _session_work_dir(
            self.work_dir, next(self._session_seq), optical_zarr
        )
        preview_config = MitsubaExportConfig(
            width=self.preview_size,
            height=self.preview_size,
            spp=self.preview_spp,
            max_depth=initial.render.max_depth,
            variant=self.mi.variant(),
            seed=seed,
        )
        base_preview = prepare_mitsuba_scene(
            volume, session_dir / "preview_scene", config=preview_config
        )
        preview_initial = _preview_stage_config(
            initial, self.preview_size, self.preview_spp
        )
        preview_scene = TraversedPreviewScene(
            self.mi, base_preview, volume.geometry, preview_initial, seed
        )
        session = InputSession(
            optical_zarr=optical_zarr,
            work_dir=session_dir,
            volume=volume,
            seed=seed,
            preview_scene=preview_scene,
        )
        self._ensure_final(session, initial.render)
        return session

    def swap_session(self, session: InputSession) -> None:
        """Replace the current session and advance the session generation."""
        self._session = session
        self.session_generation += 1

    def _ensure_final(self, session: InputSession, render: RenderSettings) -> None:
        final_key = _final_render_key(render)
        if session.final_key == final_key:
            return
        config = MitsubaExportConfig(
            width=render.width,
            height=render.height,
            spp=render.spp,
            max_depth=render.max_depth,
            variant=self.mi.variant(),
            seed=session.seed,
        )
        session.base_final = prepare_mitsuba_scene(
            session.volume, session.work_dir / "final_scene", config=config
        )
        session.final_key = final_key

    def _render(
        self,
        base,
        volume: OpticalPropertyVolume,
        config: StageConfig,
        spp: int,
        seed: int,
    ) -> np.ndarray:
        scene_dict = dict(base.scene_dict)
        apply_stage(self.mi, scene_dict, volume.geometry, config)
        scene = self.mi.load_dict(scene_dict)
        return self.mi.render(scene, seed=seed, spp=spp)

    def render_preview(
        self, config: StageConfig, spp: int | None = None
    ) -> tuple[np.ndarray, str, str]:
        """Render a preview; return (uint8 sRGB image, stats, update route)."""
        preview_config = _preview_stage_config(
            config, self.preview_size, self.preview_spp
        )
        image, route = self._session.preview_scene.render(
            preview_config, self.preview_spp if spp is None else spp
        )
        stats = _pixel_stats(
            np.asarray(image, dtype=np.float32), preview_config.render.max_depth
        )
        bitmap = self.mi.util.convert_to_bitmap(image)
        return np.asarray(bitmap), stats, route

    def render_final(self, config: StageConfig, output_png: Path) -> str:
        """Render at the config's full resolution/spp and write the PNG.

        Reads ``self._session`` once, at call time, so a job already queued
        on the render worker always renders whichever session is current
        when it actually runs (not whichever was current when it was
        submitted).
        """
        session = self._session
        self._ensure_final(session, config.render)
        image = self._render(
            session.base_final, session.volume, config, config.render.spp, session.seed
        )
        output_png.parent.mkdir(parents=True, exist_ok=True)
        self.mi.util.write_bitmap(str(output_png), image, write_async=False)
        return _pixel_stats(
            np.asarray(image, dtype=np.float32), config.render.max_depth
        )

    @staticmethod
    def save_preset(config: StageConfig, path: Path) -> None:
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(stage_config_to_dict(config), handle, indent=2)
            handle.write("\n")

    def load_input(
        self,
        root: Path,
        user_path: Path,
        current_config: StageConfig,
        smoke_spp: int,
        *,
        on_stage: Callable[[str], None] = lambda stage: None,
    ) -> InputSession:
        """Validate, build, and smoke-test a new input, then swap it in.

        Runs the five stages the plan fixes as the Load/Rebuild transaction
        (validate/prepare/load/smoke/swap), reporting each through
        ``on_stage`` before it runs. ``current_config`` (the GUI's live
        stage/render settings) drives prepare/load/smoke so a switch keeps
        the user's current settings and, as a side effect, rejects a new
        input that is incompatible with them. Any failure before ``swap``
        raises :class:`InputLoadError` naming the stage, discards the
        half-built session's work directory, and leaves the current session
        (and therefore the live preview) untouched. Only ``swap`` mutates
        ``self``.
        """
        session = self.prepare_input_session(
            root,
            user_path,
            current_config,
            smoke_spp,
            on_stage=on_stage,
        )
        on_stage("swap")
        self.swap_session(session)
        return session

    def prepare_input_session(
        self,
        root: Path,
        user_path: Path,
        current_config: StageConfig,
        smoke_spp: int,
        *,
        seed: int | None = None,
        on_stage: Callable[[str], None] = lambda stage: None,
    ) -> InputSession:
        """Resolve a catalog reference, then build and smoke-test its session.

        Thin wrapper around :meth:`prepare_candidate_session` that adds the
        ``--input-root`` catalog-resolution step; the live session is never
        changed. The wrapper's own ``on_stage("validate")`` covers both
        catalog resolution and the delegate's content validation, so the
        delegate's inner "validate" callback is suppressed to keep the
        stage sequence a caller sees exactly one "validate" long.
        """
        on_stage("validate")
        try:
            candidate = resolve_candidate(root, user_path)
        except Exception as error:
            raise InputLoadError("validate", str(error)) from error
        return self.prepare_candidate_session(
            candidate,
            current_config,
            smoke_spp,
            seed=seed,
            on_stage=lambda stage: None if stage == "validate" else on_stage(stage),
        )

    def prepare_candidate_session(
        self,
        candidate: InputCandidate,
        current_config: StageConfig,
        smoke_spp: int,
        *,
        seed: int | None = None,
        on_stage: Callable[[str], None] = lambda stage: None,
    ) -> InputSession:
        """Build and smoke-test an already-resolved candidate's session.

        Split out from :meth:`prepare_input_session` so a caller that
        already has an :class:`InputCandidate` outside ``--input-root``
        (e.g. a mapping-derived bundle under ``--mapping-work-root``) can
        drive the same validate/prepare/load/smoke transaction without
        going through catalog path resolution. The live session is never
        changed; ``derivation`` is not set here (the caller attaches it to
        the returned session, since only the caller knows whether this
        candidate came from a mapping or an as-is catalog selection).
        """
        session_seed = self._session.seed if seed is None else seed
        on_stage("validate")
        try:
            volume = read_volume(candidate.optical_zarr)
            if not isinstance(volume, OpticalPropertyVolume):
                raise InputLoadError(
                    "validate",
                    f"{candidate.optical_zarr} is not an optical property volume",
                )
        except InputLoadError:
            raise
        except Exception as error:
            raise InputLoadError("validate", str(error)) from error

        session_dir = _session_work_dir(
            self.work_dir, next(self._session_seq), candidate.optical_zarr
        )
        on_stage("prepare")
        try:
            preview_config = MitsubaExportConfig(
                width=self.preview_size,
                height=self.preview_size,
                spp=self.preview_spp,
                max_depth=current_config.render.max_depth,
                variant=self.mi.variant(),
                seed=session_seed,
            )
            base_preview = prepare_mitsuba_scene(
                volume, session_dir / "preview_scene", config=preview_config
            )
        except Exception as error:
            _discard_session_dir(session_dir)
            raise InputLoadError("prepare", str(error)) from error

        on_stage("load")
        try:
            preview_initial = _preview_stage_config(
                current_config, self.preview_size, self.preview_spp
            )
            preview_scene = TraversedPreviewScene(
                self.mi,
                base_preview,
                volume.geometry,
                preview_initial,
                session_seed,
            )
        except Exception as error:
            _discard_session_dir(session_dir)
            raise InputLoadError("load", str(error)) from error

        on_stage("smoke")
        try:
            preview_scene.render(preview_initial, smoke_spp)
        except Exception as error:
            _discard_session_dir(session_dir)
            raise InputLoadError("smoke", str(error)) from error

        session = InputSession(
            optical_zarr=candidate.optical_zarr,
            work_dir=session_dir,
            volume=volume,
            seed=session_seed,
            preview_scene=preview_scene,
        )
        return session


class RenderWorker(threading.Thread):
    """Single worker: immediate coarse preview, then latest settled preview.

    Preview staleness is guarded on two independent axes. ``_generation`` is
    this worker's own monotonic per-request counter (latest-wins: a newer
    request always supersedes a pending older one). ``session_generation`` is
    supplied by the caller (``StageCore.session_generation``) and identifies
    which input the requested config belongs to; a result is only published
    if the session that was current when the request was made is still
    current when the render finishes. Because this worker only ever runs one
    job (a preview render or a submitted job, e.g. an input swap) at a time,
    the session check can in practice never fail today — swapping and
    rendering never interleave. It exists as a defense-in-depth guard for
    re-entrancy or a future multi-worker design, and is exercised directly
    in tests since the single-thread serialization makes it otherwise
    unreachable.
    """

    def __init__(self, settle_delay: float = 0.35) -> None:
        super().__init__(daemon=True)
        self._condition = threading.Condition()
        self._pending_preview: tuple[int, int, StageConfig] | None = None
        self._generation = 0
        self._preview_fn: Callable[[StageConfig, str], object] | None = None
        self._publish_fn: Callable[[object, str], None] | None = None
        self._current_session_generation: Callable[[], int] = lambda: 0
        self._jobs: list[Callable[[], None]] = []
        self._settle_delay = settle_delay
        self._on_error: Callable[[str], None] = lambda message: print(
            f"RENDER ERROR {message}", file=sys.stderr
        )

    def configure(
        self,
        preview_fn: Callable[[StageConfig, str], object],
        publish_fn: Callable[[object, str], None],
        on_error: Callable[[str], None],
        current_session_generation: Callable[[], int] = lambda: 0,
    ) -> None:
        self._preview_fn = preview_fn
        self._publish_fn = publish_fn
        self._on_error = on_error
        self._current_session_generation = current_session_generation

    def request_preview(self, config: StageConfig, session_generation: int = 0) -> int:
        with self._condition:
            self._generation += 1
            self._pending_preview = (self._generation, session_generation, config)
            self._condition.notify()
            return self._generation

    def submit(self, job: Callable[[], None]) -> None:
        with self._condition:
            self._jobs.append(job)
            self._condition.notify()

    def _publish(
        self, result: object, quality: str, generation: int, session_generation: int
    ) -> None:
        with self._condition:
            current = self._generation
        if (
            generation == current
            and session_generation == self._current_session_generation()
            and self._publish_fn is not None
        ):
            self._publish_fn(result, quality)

    def run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: bool(self._jobs) or self._pending_preview is not None
                )
                if self._jobs:
                    job = self._jobs.pop(0)
                    preview = None
                else:
                    job = None
                    preview = self._pending_preview
                    self._pending_preview = None
            try:
                if job is not None:
                    job()
                    continue
                if preview is None or self._preview_fn is None:
                    continue
                generation, session_generation, config = preview
                result = self._preview_fn(config, "interactive")
                self._publish(result, "interactive", generation, session_generation)
                deadline = time.monotonic() + self._settle_delay
                with self._condition:
                    while (
                        self._pending_preview is None
                        and not self._jobs
                        and self._generation == generation
                    ):
                        remaining = deadline - time.monotonic()
                        if remaining <= 0.0:
                            break
                        self._condition.wait(remaining)
                    settled = (
                        self._pending_preview is None
                        and not self._jobs
                        and self._generation == generation
                    )
                if settled:
                    result = self._preview_fn(config, "settled")
                    self._publish(result, "settled", generation, session_generation)
            except Exception:
                self._on_error(traceback.format_exc(limit=3))


def _decompose_radiance(radiance: RGB) -> tuple[tuple[int, int, int], float]:
    intensity = max(radiance)
    if intensity <= 0.0:
        return (0, 0, 0), 1.0
    colour = tuple(
        round(min(max(component / intensity, 0.0), 1.0) * 255) for component in radiance
    )
    return colour, intensity


def _compose_radiance(colour: tuple[int, int, int], intensity: float) -> RGB:
    return tuple(component / 255.0 * intensity for component in colour)


def _decompose_direction(direction: RGB) -> tuple[float, float]:
    vector = np.asarray(direction, dtype=np.float64)
    vector /= np.linalg.norm(vector)
    azimuth = math.degrees(math.atan2(vector[1], vector[0]))
    elevation = math.degrees(math.asin(np.clip(vector[2], -1.0, 1.0)))
    return azimuth, elevation


def _compose_direction(azimuth_deg: float, elevation_deg: float) -> RGB:
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    return (
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
    )


def _rgb_int(colour: RGB) -> tuple[int, int, int]:
    return tuple(round(min(max(component, 0.0), 1.0) * 255) for component in colour)


def _fit_preview_to_aspect(pixels: np.ndarray, viewport_aspect: float) -> np.ndarray:
    """Pad an image so viser's stretched background preserves its aspect ratio."""
    height, width = pixels.shape[:2]
    image_aspect = width / height
    if viewport_aspect > image_aspect:
        target_width = max(width, math.ceil(height * viewport_aspect))
        before = (target_width - width) // 2
        padding = ((0, 0), (before, target_width - width - before), (0, 0))
    else:
        target_height = max(height, math.ceil(width / viewport_aspect))
        before = (target_height - height) // 2
        padding = ((before, target_height - height - before), (0, 0), (0, 0))
    return np.pad(pixels, padding, mode="constant")


class StageBinder:
    """GUI <-> StageConfig binding with per-field dirty tracking.

    Fields backed by lossy widgets (sliders, 0-255 colour pickers, and the
    GUI-only radiance/direction decompositions) are read from the GUI only
    once the user has touched them; otherwise the value loaded at startup is
    passed through untouched. Exact widgets (checkboxes, dropdowns, number
    inputs) are always read directly.
    """

    def __init__(
        self,
        server,
        base: StageConfig,
        on_change: Callable[[], None],
        *,
        input_tab: Callable[[object], None] | None = None,
        preset_tab: Callable[[object], None] | None = None,
    ) -> None:
        self.base = base
        self.dirty: set[str] = set()
        self._on_change = on_change
        self._suspend_updates = False
        gui = server.gui
        camera_base = base.camera if base.camera is not None else CameraOverride()
        backlight_base = (
            base.backlight if base.backlight is not None else BacklightOverride()
        )
        key_colour, key_intensity = _decompose_radiance(base.key_light.radiance)
        back_colour, back_intensity = _decompose_radiance(backlight_base.radiance)
        key_azimuth, key_elevation = _decompose_direction(base.key_light.direction)

        # Keep the controls in one bounded settings area. A vertical stack of
        # expanded folders is unwieldy on shorter screens.
        self.tabs = gui.add_tab_group()
        if input_tab is not None:
            # Input comes first: choosing what to render precedes tuning how.
            with self.tabs.add_tab("Input"):
                input_tab(gui)
        if preset_tab is not None:
            with self.tabs.add_tab("Preset"):
                preset_tab(gui)
        with self.tabs.add_tab("Render"):
            self.width = gui.add_number(
                "width", base.render.width, min=16, max=4096, step=16
            )
            self.height = gui.add_number(
                "height", base.render.height, min=16, max=4096, step=16
            )
            self.spp = gui.add_number("spp", base.render.spp, min=1, max=4096, step=1)
            self.max_depth = gui.add_number(
                "max depth",
                base.render.max_depth,
                min=1,
                step=1,
                hint="Higher values allow longer light paths and can render slower.",
            )
        with self.tabs.add_tab("Backdrop"):
            self.backdrop_enabled = gui.add_checkbox("enabled", base.backdrop.enabled)
            self.backdrop_pattern = gui.add_dropdown(
                "pattern", ("checker", "solid"), initial_value=base.backdrop.pattern
            )
            self.backdrop_distance = self._slider(
                gui,
                "distance",
                0.2,
                10.0,
                base.backdrop.distance_factor,
                "backdrop.distance_factor",
            )
            self.backdrop_scale = self._slider(
                gui,
                "scale",
                0.2,
                10.0,
                base.backdrop.scale_factor,
                "backdrop.scale_factor",
            )
            self.backdrop_checker = self._int_slider(
                gui,
                "checker tiles",
                base.backdrop.checker_scale,
                "backdrop.checker_scale",
            )
            self.backdrop_color0 = self._rgb(
                gui, "color0", base.backdrop.color0, "backdrop.color0"
            )
            self.backdrop_color1 = self._rgb(
                gui, "color1", base.backdrop.color1, "backdrop.color1"
            )
        with self.tabs.add_tab("Floor"):
            self.floor_enabled = gui.add_checkbox("enabled", base.floor.enabled)
            self.floor_pattern = gui.add_dropdown(
                "pattern", ("checker", "solid"), initial_value=base.floor.pattern
            )
            self.floor_drop = self._slider(
                gui, "drop", 0.0, 2.0, base.floor.drop_factor, "floor.drop_factor"
            )
            self.floor_scale = self._slider(
                gui, "scale", 0.2, 20.0, base.floor.scale_factor, "floor.scale_factor"
            )
            self.floor_checker = self._int_slider(
                gui, "checker tiles", base.floor.checker_scale, "floor.checker_scale"
            )
            self.floor_color0 = self._rgb(
                gui, "color0", base.floor.color0, "floor.color0"
            )
            self.floor_color1 = self._rgb(
                gui, "color1", base.floor.color1, "floor.color1"
            )
        with self.tabs.add_tab("Key light"):
            self.key_enabled = gui.add_checkbox("enabled", base.key_light.enabled)
            self.key_azimuth = self._slider(
                gui, "azimuth °", -180.0, 180.0, key_azimuth, "key_light.direction"
            )
            self.key_elevation = self._slider(
                gui, "elevation °", -89.0, 89.0, key_elevation, "key_light.direction"
            )
            self.key_distance = self._slider(
                gui,
                "distance",
                0.5,
                10.0,
                base.key_light.distance_factor,
                "key_light.distance_factor",
            )
            self.key_scale = self._slider(
                gui,
                "scale",
                0.1,
                5.0,
                base.key_light.scale_factor,
                "key_light.scale_factor",
            )
            self.key_colour = self._rgb_raw(
                gui, "colour", key_colour, "key_light.radiance"
            )
            self.key_intensity = self._slider(
                gui, "intensity", 0.0, 30.0, key_intensity, "key_light.radiance"
            )
        with self.tabs.add_tab("Camera"):
            self.camera_enabled = gui.add_checkbox(
                "override camera", base.camera is not None
            )
            self.camera_azimuth = self._slider(
                gui,
                "azimuth °",
                -180.0,
                180.0,
                camera_base.azimuth_deg,
                "camera.azimuth_deg",
            )
            self.camera_elevation = self._slider(
                gui,
                "elevation °",
                -89.0,
                89.0,
                camera_base.elevation_deg,
                "camera.elevation_deg",
            )
            self.camera_distance = self._slider(
                gui,
                "distance",
                1.0,
                12.0,
                camera_base.distance_factor,
                "camera.distance_factor",
            )
            self.camera_fov = self._slider(
                gui, "fov °", 10.0, 120.0, camera_base.fov_deg, "camera.fov_deg"
            )
        with self.tabs.add_tab("Backlight"):
            self.backlight_enabled = gui.add_checkbox(
                "override backlight", base.backlight is not None
            )
            self.backlight_colour = self._rgb_raw(
                gui, "colour", back_colour, "backlight.radiance"
            )
            self.backlight_intensity = self._slider(
                gui, "intensity", 0.0, 30.0, back_intensity, "backlight.radiance"
            )

        for handle in (
            self.width,
            self.height,
            self.spp,
            self.max_depth,
            self.backdrop_enabled,
            self.backdrop_pattern,
            self.floor_enabled,
            self.floor_pattern,
            self.key_enabled,
            self.camera_enabled,
            self.backlight_enabled,
        ):
            handle.on_update(lambda _event: self._notify_change())

    def _slider(self, gui, label, low, high, initial, key):
        clamped = min(max(float(initial), low), high)
        handle = gui.add_slider(
            label, min=low, max=high, step=0.01, initial_value=clamped
        )
        self._track(handle, key)
        return handle

    def _int_slider(self, gui, label, initial, key):
        handle = gui.add_slider(label, min=1, max=32, step=1, initial_value=initial)
        self._track(handle, key)
        return handle

    def _rgb(self, gui, label, initial: RGB, key):
        handle = gui.add_rgb(label, initial_value=_rgb_int(initial))
        self._track(handle, key)
        return handle

    def _rgb_raw(self, gui, label, initial: tuple[int, int, int], key):
        handle = gui.add_rgb(label, initial_value=initial)
        self._track(handle, key)
        return handle

    def _track(self, handle, key: str) -> None:
        def _mark(_event, key: str = key) -> None:
            if self._suspend_updates:
                return
            self.dirty.add(key)
            self._notify_change()

        handle.on_update(_mark)

    def _notify_change(self) -> None:
        if not self._suspend_updates:
            self._on_change()

    def replace_config(self, config: StageConfig) -> None:
        """Replace every widget from ``config`` without emitting a change.

        Lossy controls are updated for display, then ``base`` becomes the
        exact supplied config and dirty tracking is cleared.  Until the user
        edits one of those controls, :meth:`current` therefore returns the
        unquantized source values rather than values reconstructed from GUI
        sliders or 8-bit colour pickers.
        """
        camera = config.camera if config.camera is not None else CameraOverride()
        backlight = (
            config.backlight if config.backlight is not None else BacklightOverride()
        )
        key_colour, key_intensity = _decompose_radiance(config.key_light.radiance)
        back_colour, back_intensity = _decompose_radiance(backlight.radiance)
        key_azimuth, key_elevation = _decompose_direction(config.key_light.direction)

        self._suspend_updates = True
        try:
            assignments = (
                (self.width, config.render.width),
                (self.height, config.render.height),
                (self.spp, config.render.spp),
                (self.max_depth, config.render.max_depth),
                (self.backdrop_enabled, config.backdrop.enabled),
                (self.backdrop_pattern, config.backdrop.pattern),
                (
                    self.backdrop_distance,
                    min(max(config.backdrop.distance_factor, 0.2), 10.0),
                ),
                (
                    self.backdrop_scale,
                    min(max(config.backdrop.scale_factor, 0.2), 10.0),
                ),
                (
                    self.backdrop_checker,
                    min(max(config.backdrop.checker_scale, 1), 32),
                ),
                (self.backdrop_color0, _rgb_int(config.backdrop.color0)),
                (self.backdrop_color1, _rgb_int(config.backdrop.color1)),
                (self.floor_enabled, config.floor.enabled),
                (self.floor_pattern, config.floor.pattern),
                (self.floor_drop, min(max(config.floor.drop_factor, 0.0), 2.0)),
                (
                    self.floor_scale,
                    min(max(config.floor.scale_factor, 0.2), 20.0),
                ),
                (
                    self.floor_checker,
                    min(max(config.floor.checker_scale, 1), 32),
                ),
                (self.floor_color0, _rgb_int(config.floor.color0)),
                (self.floor_color1, _rgb_int(config.floor.color1)),
                (self.key_enabled, config.key_light.enabled),
                (self.key_azimuth, min(max(key_azimuth, -180.0), 180.0)),
                (self.key_elevation, min(max(key_elevation, -89.0), 89.0)),
                (
                    self.key_distance,
                    min(max(config.key_light.distance_factor, 0.5), 10.0),
                ),
                (
                    self.key_scale,
                    min(max(config.key_light.scale_factor, 0.1), 5.0),
                ),
                (self.key_colour, key_colour),
                (self.key_intensity, min(max(key_intensity, 0.0), 30.0)),
                (self.camera_enabled, config.camera is not None),
                (
                    self.camera_azimuth,
                    min(max(camera.azimuth_deg, -180.0), 180.0),
                ),
                (
                    self.camera_elevation,
                    min(max(camera.elevation_deg, -89.0), 89.0),
                ),
                (
                    self.camera_distance,
                    min(max(camera.distance_factor, 1.0), 12.0),
                ),
                (self.camera_fov, min(max(camera.fov_deg, 10.0), 120.0)),
                (self.backlight_enabled, config.backlight is not None),
                (self.backlight_colour, back_colour),
                (
                    self.backlight_intensity,
                    min(max(back_intensity, 0.0), 30.0),
                ),
            )
            for handle, value in assignments:
                handle.value = value
            self.base = config
            self.dirty.clear()
        finally:
            self._suspend_updates = False

    def _value(self, key: str, handle, fallback):
        return float(handle.value) if key in self.dirty else fallback

    def _colour(self, key: str, handle, fallback: RGB) -> RGB:
        if key in self.dirty:
            return tuple(component / 255.0 for component in handle.value)
        return fallback

    def current(self) -> StageConfig:
        """Assemble the StageConfig currently described by the GUI."""
        base = self.base
        camera_base = base.camera if base.camera is not None else CameraOverride()
        backlight_base = (
            base.backlight if base.backlight is not None else BacklightOverride()
        )
        if "key_light.direction" in self.dirty:
            direction = _compose_direction(
                self.key_azimuth.value, self.key_elevation.value
            )
        else:
            direction = base.key_light.direction
        if "key_light.radiance" in self.dirty:
            key_radiance = _compose_radiance(
                self.key_colour.value, self.key_intensity.value
            )
        else:
            key_radiance = base.key_light.radiance
        if "backlight.radiance" in self.dirty:
            back_radiance = _compose_radiance(
                self.backlight_colour.value, self.backlight_intensity.value
            )
        else:
            back_radiance = backlight_base.radiance
        camera = None
        if self.camera_enabled.value:
            camera = CameraOverride(
                azimuth_deg=self._value(
                    "camera.azimuth_deg",
                    self.camera_azimuth,
                    camera_base.azimuth_deg,
                ),
                elevation_deg=self._value(
                    "camera.elevation_deg",
                    self.camera_elevation,
                    camera_base.elevation_deg,
                ),
                distance_factor=self._value(
                    "camera.distance_factor",
                    self.camera_distance,
                    camera_base.distance_factor,
                ),
                fov_deg=self._value(
                    "camera.fov_deg", self.camera_fov, camera_base.fov_deg
                ),
            )
        return StageConfig(
            render=RenderSettings(
                width=int(self.width.value),
                height=int(self.height.value),
                spp=int(self.spp.value),
                max_depth=int(self.max_depth.value),
            ),
            backdrop=BackdropSettings(
                enabled=self.backdrop_enabled.value,
                pattern=self.backdrop_pattern.value,
                distance_factor=self._value(
                    "backdrop.distance_factor",
                    self.backdrop_distance,
                    base.backdrop.distance_factor,
                ),
                scale_factor=self._value(
                    "backdrop.scale_factor",
                    self.backdrop_scale,
                    base.backdrop.scale_factor,
                ),
                checker_scale=(
                    int(self.backdrop_checker.value)
                    if "backdrop.checker_scale" in self.dirty
                    else base.backdrop.checker_scale
                ),
                color0=self._colour(
                    "backdrop.color0", self.backdrop_color0, base.backdrop.color0
                ),
                color1=self._colour(
                    "backdrop.color1", self.backdrop_color1, base.backdrop.color1
                ),
            ),
            floor=FloorSettings(
                enabled=self.floor_enabled.value,
                pattern=self.floor_pattern.value,
                drop_factor=self._value(
                    "floor.drop_factor", self.floor_drop, base.floor.drop_factor
                ),
                scale_factor=self._value(
                    "floor.scale_factor", self.floor_scale, base.floor.scale_factor
                ),
                checker_scale=(
                    int(self.floor_checker.value)
                    if "floor.checker_scale" in self.dirty
                    else base.floor.checker_scale
                ),
                color0=self._colour(
                    "floor.color0", self.floor_color0, base.floor.color0
                ),
                color1=self._colour(
                    "floor.color1", self.floor_color1, base.floor.color1
                ),
            ),
            key_light=KeyLightSettings(
                enabled=self.key_enabled.value,
                direction=direction,
                distance_factor=self._value(
                    "key_light.distance_factor",
                    self.key_distance,
                    base.key_light.distance_factor,
                ),
                scale_factor=self._value(
                    "key_light.scale_factor",
                    self.key_scale,
                    base.key_light.scale_factor,
                ),
                radiance=key_radiance,
            ),
            camera=camera,
            backlight=(
                BacklightOverride(radiance=back_radiance)
                if self.backlight_enabled.value
                else None
            ),
        )


class ViewerApp:
    """Wires StageCore, RenderWorker, and the viser GUI together."""

    def __init__(self, args: argparse.Namespace) -> None:
        import viser

        work_dir = args.work_dir
        if work_dir is None:
            work_dir = Path(tempfile.mkdtemp(prefix="mitsuba-stage-viewer-"))
        work_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir = work_dir

        try:
            startup = _resolve_viewer_startup(args, work_dir)
        except (
            InputCatalogError,
            PresetCatalogError,
            MappingCatalogError,
            RegenError,
            ViewerSessionError,
        ) as error:
            raise SystemExit(str(error)) from error
        initial = startup.initial_config
        self.preset_root = startup.preset_root
        self.mapping_root = startup.mapping_root
        self.mapping_work_root = startup.mapping_work_root
        self.session_root = startup.session_root
        self.applied_preset = startup.applied_preset
        self._committed_derivation = startup.initial_derivation

        # The Input tab always names the *source* catalog entry as "the
        # input" — even when a mapping derivation is active and the scene
        # actually being rendered lives under --mapping-work-root, outside
        # --input-root (see SessionDerivation).
        self._initial_input_path = (
            startup.initial_derivation.source_candidate.path
            if startup.initial_derivation is not None
            else startup.initial_input
        ).resolve()
        self.input_root = startup.input_root
        if self._initial_input_path.is_relative_to(self.input_root):
            self._initial_sentinel: str | None = None
            self._current_selection = self._initial_input_path.relative_to(
                self.input_root
            ).as_posix()
        else:
            self._initial_sentinel = f"(initial) {self._initial_input_path}"
            self._current_selection = self._initial_sentinel

        self.core = StageCore(
            _resolve_initial_optical_zarr(startup.initial_input),
            work_dir,
            preview_size=args.preview_size,
            preview_spp=args.preview_spp,
            initial=initial,
            variant=startup.variant,
            seed=startup.seed,
        )
        if startup.initial_derivation is not None:
            self.core._session.derivation = startup.initial_derivation
        self.interactive_spp = args.interactive_spp
        self.worker = RenderWorker(settle_delay=args.settle_delay)
        self.server = viser.ViserServer(host="127.0.0.1", port=args.port)
        gui = self.server.gui
        gui.configure_theme(control_layout="fixed", control_width="large")
        gui.set_panel_label("Mitsuba stage viewer")

        placeholder = np.zeros(
            (args.preview_size, args.preview_size, 3), dtype=np.uint8
        )
        # The render belongs in the persistent viewport, not in the scrolling
        # controls panel. This keeps it visible while any settings tab is used.
        self._preview_lock = threading.Lock()
        self._preview_pixels = placeholder
        self._client_viewport_sizes: dict[int, tuple[int, int]] = {}

        @self.server.on_client_connect
        def _connect_preview(client) -> None:
            @client.camera.on_update
            def _resize_preview(_camera) -> None:
                self._update_client_preview(client)

            # Cover the case where the initial camera-size message arrived
            # before this connection callback was scheduled.
            self._update_client_preview(client, force=True)

        @self.server.on_client_disconnect
        def _disconnect_preview(client) -> None:
            with self._preview_lock:
                self._client_viewport_sizes.pop(client.client_id, None)

        self.status = gui.add_markdown("starting…")
        self.binder = StageBinder(
            self.server,
            initial,
            self._on_stage_change,
            input_tab=self._build_input_tab,
            preset_tab=self._build_preset_tab,
        )

        preset_default = args.preset_out or (work_dir / "viewer.stage.json")
        final_default = args.final_out or (work_dir / "final.png")
        with self.binder.tabs.add_tab("Output"):
            self.preset_path = gui.add_text("preset path", str(preset_default))
            save_button = gui.add_button("Save preset")
            self.session_save_path = gui.add_text(
                "session path (session-root relative)", "viewer.session.json"
            )
            self.session_save_button = gui.add_button("Save session")
            self.final_path = gui.add_text("final PNG path", str(final_default))
            render_button = gui.add_button("Render final")

        save_button.on_click(lambda _event: self._save_preset())
        self.session_save_button.on_click(lambda _event: self._queue_save_session())
        render_button.on_click(lambda _event: self._queue_final())

        self.worker.configure(
            self._render_preview,
            self._publish_preview,
            self._show_error,
            self._current_session_generation,
        )
        self.worker.start()
        self._schedule_preview()

    # -- worker-side operations -------------------------------------------

    def _current_session_generation(self) -> int:
        return self.core.session_generation

    def _on_stage_change(self) -> None:
        self.applied_preset = None
        self._schedule_preview()

    def _schedule_preview(self) -> None:
        self.status.content = "rendering preview…"
        self.worker.request_preview(self.binder.current(), self.core.session_generation)

    def _render_preview(
        self, config: StageConfig, quality: str
    ) -> tuple[np.ndarray, str, str, float]:
        started = time.perf_counter()
        spp = self.interactive_spp if quality == "interactive" else None
        pixels, stats, route = self.core.render_preview(config, spp=spp)
        elapsed = time.perf_counter() - started
        return pixels, stats, route, elapsed

    def _publish_preview(self, result: object, quality: str) -> None:
        pixels, stats, route, elapsed = result
        with self._preview_lock:
            self._preview_pixels = pixels
        for client in self.server.get_clients().values():
            self._update_client_preview(client, force=True)
        self.status.content = (
            f"preview {quality}/{route} {elapsed:.2f}s — PIXELSTATS {stats}"
        )

    def _update_client_preview(self, client, *, force: bool = False) -> None:
        try:
            viewport_size = (client.camera.image_width, client.camera.image_height)
        except AssertionError:
            return
        if viewport_size[0] <= 0 or viewport_size[1] <= 0:
            return
        with self._preview_lock:
            if (
                not force
                and self._client_viewport_sizes.get(client.client_id) == viewport_size
            ):
                return
            self._client_viewport_sizes[client.client_id] = viewport_size
            pixels = self._preview_pixels
        client.scene.set_background_image(
            _fit_preview_to_aspect(pixels, viewport_size[0] / viewport_size[1])
        )

    def _save_preset(self) -> None:
        path = Path(self.preset_path.value)
        self.core.save_preset(self.binder.current(), path)
        self.status.content = f"preset saved: {path}"
        print(f"PRESET saved {path}")

    def _capture_session(self, path: Path) -> None:
        selection = self._current_selection
        if selection == self._initial_sentinel:
            raise ViewerSessionError("capture", "current input is outside --input-root")
        candidate = resolve_candidate(self.input_root, Path(selection))
        live_session = self.core.current_session
        derivation = live_session.derivation
        mapping_ref: SessionMappingRef | None = None
        if derivation is None:
            if candidate.optical_zarr.resolve() != live_session.optical_zarr.resolve():
                raise ViewerSessionError(
                    "capture", "current input selection does not match the live scene"
                )
        else:
            if candidate.path.resolve() != derivation.source_candidate.path.resolve():
                raise ViewerSessionError(
                    "capture",
                    "current input selection does not match the live scene's "
                    "mapping source",
                )
            try:
                current_mapping_candidate = resolve_mapping_candidate(
                    self.mapping_root,
                    Path(derivation.mapping_candidate.root_relative),
                )
                current_mapping_digest = load_mapping(current_mapping_candidate).digest
            except MappingCatalogError as error:
                raise ViewerSessionError("capture", f"mapping: {error}") from error
            if current_mapping_digest != derivation.mapping_digest:
                raise ViewerSessionError(
                    "capture",
                    "mapping file has changed since it was applied; "
                    "Load/Rebuild again before saving",
                )
            try:
                derived_digest = zarr_store_sha256(live_session.optical_zarr)
            except OSError as error:
                raise ViewerSessionError(
                    "capture", f"cannot digest derived optical store: {error}"
                ) from error
            mapping_ref = SessionMappingRef(
                path=derivation.mapping_candidate.root_relative,
                digest=current_mapping_digest,
                derived_optical_sha256=derived_digest,
            )
        session = create_viewer_session(
            candidate,
            self.binder.current(),
            self.core.mi.variant(),
            live_session.seed,
            preset=self.applied_preset,
            mapping=mapping_ref,
        )
        write_viewer_session(path, session)

    def _queue_save_session(self) -> None:
        self.session_save_button.disabled = True
        self.status.content = "session save: resolve path…"

        def job() -> None:
            try:
                path = _resolve_session_path(
                    self.session_root,
                    Path(self.session_save_path.value),
                    must_exist=False,
                )
                self.status.content = "session save: digest input…"
                self._capture_session(path)
            except Exception as error:
                self.status.content = f"**session save failed**: {error}"
                print(f"SESSION SAVE ERROR {error}", file=sys.stderr)
            else:
                self.status.content = f"session saved: {path}"
                print(f"SESSION saved {path}")
            finally:
                self.session_save_button.disabled = False

        self.worker.submit(job)

    def _load_session_transaction(
        self, path: Path, on_stage: Callable[[str], None]
    ) -> None:
        """Verify, prepare, and atomically commit one viewer session."""
        on_stage("parse")
        session = viewer_session_from_json(path)
        resolved = resolve_viewer_session(
            session,
            self.input_root,
            self.preset_root,
            self.mapping_root,
            on_stage=on_stage,
        )
        if resolved.variant != self.core.mi.variant():
            raise ViewerSessionError(
                "resolve",
                f"session variant {resolved.variant} differs from running variant "
                f"{self.core.mi.variant()}; restart the viewer with --session",
            )

        derivation: SessionDerivation | None = None
        if resolved.session.mapping is not None:
            assert resolved.mapping_candidate is not None
            try:
                derived = regenerate_optical(
                    resolved.input_candidate.path,
                    resolved.mapping_candidate,
                    self.mapping_work_root,
                    on_stage=lambda stage: (
                        None if stage == "validate" else on_stage(stage)
                    ),
                )
            except RegenError as error:
                raise ViewerSessionError(error.stage, error.message) from error
            if derived.reused:
                on_stage("map: reused cache")
            on_stage("verify")
            verify_derived_optical(resolved, derived.optical_zarr)
            derived_candidate = InputCandidate(
                kind=InputKind.RUN_BUNDLE,
                root_relative=derived.bundle_path.name,
                path=derived.bundle_path,
                optical_zarr=derived.optical_zarr,
            )
            prepared = self.core.prepare_candidate_session(
                derived_candidate,
                resolved.stage_config,
                self.interactive_spp,
                seed=resolved.seed,
                on_stage=lambda stage: None if stage == "validate" else on_stage(stage),
            )
            derivation = SessionDerivation(
                source_candidate=resolved.input_candidate,
                mapping_candidate=resolved.mapping_candidate,
                mapping_digest=resolved.session.mapping.digest,
                derived_bundle=derived.bundle_path,
            )
            prepared.derivation = derivation
        else:
            prepared = self.core.prepare_input_session(
                self.input_root,
                Path(session.input.path),
                resolved.stage_config,
                self.interactive_spp,
                seed=resolved.seed,
                on_stage=lambda stage: None if stage == "validate" else on_stage(stage),
            )

        old_core_session = self.core.current_session
        old_generation = self.core.session_generation
        old_config = self.binder.current()
        old_selection = self._current_selection
        old_mapping_selection = self._committed_mapping_selection()
        old_source = self.applied_preset
        on_stage("commit")
        try:
            self.binder.replace_config(resolved.stage_config)
            self.core.swap_session(prepared)
            self._current_selection = session.input.path
            self.input_dropdown.value = session.input.path
            self._committed_derivation = derivation
            mapping_selection = (
                session.mapping.path if session.mapping is not None else _AS_IS_MAPPING
            )
            if mapping_selection in tuple(self.mapping_dropdown.options):
                self.mapping_dropdown.value = mapping_selection
            self.applied_preset = session.preset
            if session.preset is not None:
                options = tuple(self.preset_dropdown.options)
                if session.preset.path in options:
                    self.preset_dropdown.value = session.preset.path
        except Exception as error:
            self.core._session = old_core_session
            self.core.session_generation = old_generation
            self._current_selection = old_selection
            self._committed_derivation = old_core_session.derivation
            self.applied_preset = old_source
            try:
                self.binder.replace_config(old_config)
                self.input_dropdown.value = old_selection
                if old_mapping_selection in tuple(self.mapping_dropdown.options):
                    self.mapping_dropdown.value = old_mapping_selection
            finally:
                _discard_session_dir(prepared.work_dir)
            raise ViewerSessionError("commit", str(error)) from error

    def _queue_load_session(self) -> None:
        self.session_load_button.disabled = True
        self.status.content = "session load: resolve path…"

        timer = _StageTimer("session load")

        def on_stage(stage: str) -> None:
            suffix = timer.advance(stage)
            self.status.content = f"session load: {stage}…{suffix}"

        def job() -> None:
            try:
                path = _resolve_session_path(
                    self.session_root,
                    Path(self.session_load_path.value),
                    must_exist=True,
                )
                self._load_session_transaction(path, on_stage)
            except (InputLoadError, ViewerSessionError) as error:
                timer.finish()
                self.status.content = f"**session load failed**: {error}"
                print(f"SESSION LOAD ERROR {error}", file=sys.stderr)
            except Exception as error:
                timer.finish()
                self.status.content = f"**session load failed**: {error}"
                print(f"SESSION LOAD ERROR {error}", file=sys.stderr)
            else:
                total = timer.finish()
                self.status.content = f"session loaded ({total:.1f}s)"
                self._update_input_summary()
                self._update_preset_summary()
                self._update_mapping_summary()
                self._schedule_preview()
            finally:
                self.session_load_button.disabled = False

        self.worker.submit(job)

    def _queue_final(self) -> None:
        config = self.binder.current()
        path = Path(self.final_path.value)
        self.status.content = "final render queued…"

        def job() -> None:
            timer = _StageTimer("final render")
            timer.advance("render")
            stats = self.core.render_final(config, path)
            elapsed = timer.finish()
            self.status.content = f"final {elapsed:.1f}s → {path} — PIXELSTATS {stats}"
            print(f"FINAL {path} PIXELSTATS {stats}")

        self.worker.submit(job)

    def _show_error(self, message: str) -> None:
        self.status.content = f"**render error**\n```\n{message}\n```"
        print(f"RENDER ERROR {message}", file=sys.stderr)

    # -- Preset tab --------------------------------------------------------
    #
    # Selecting and describing a preset are read-only.  Only the explicit
    # Apply button replaces the binder config and schedules one preview.

    def _preset_options(self) -> list[str]:
        options = [
            candidate.root_relative
            for candidate in scan_preset_catalog(self.preset_root)
        ]
        return options or [_NO_PRESETS]

    def _describe_preset_selection(self, selection: str) -> str:
        if selection == _NO_PRESETS:
            return "No `*.stage.json` presets found under `--preset-root`."
        try:
            candidate = resolve_preset(self.preset_root, Path(selection))
            summary = describe_preset(candidate)
        except Exception as error:
            return f"**cannot describe stage preset**: {error}"
        return "\n".join(
            (
                f"- format version: {summary.format_version}",
                f"- render: {summary.width}x{summary.height}, "
                f"spp {summary.spp}, max depth {summary.max_depth}",
                f"- camera override: {summary.camera_override}",
                f"- backlight override: {summary.backlight_override}",
                f"- digest: `{summary.digest}`",
            )
        )

    def _build_preset_tab(self, gui) -> None:
        options = self._preset_options()
        applied_path = (
            self.applied_preset.path if self.applied_preset is not None else None
        )
        selection = applied_path if applied_path in options else options[0]
        self.preset_dropdown = gui.add_dropdown(
            "stage preset", tuple(options), initial_value=selection
        )
        self.preset_refresh_button = gui.add_button("Refresh")
        self.preset_summary = gui.add_markdown(
            self._describe_preset_selection(selection)
        )
        self.preset_apply_button = gui.add_button("Apply stage preset")

        self.preset_dropdown.on_update(lambda _event: self._update_preset_summary())
        self.preset_refresh_button.on_click(
            lambda _event: self._refresh_preset_catalog()
        )
        self.preset_apply_button.on_click(lambda _event: self._apply_stage_preset())

    def _update_preset_summary(self) -> None:
        self.preset_summary.content = self._describe_preset_selection(
            self.preset_dropdown.value
        )

    def _refresh_preset_catalog(self) -> None:
        options = self._preset_options()
        previous = self.preset_dropdown.value
        self.preset_dropdown.options = tuple(options)
        if previous not in options:
            applied_path = (
                self.applied_preset.path if self.applied_preset is not None else None
            )
            self.preset_dropdown.value = (
                applied_path if applied_path in options else options[0]
            )
        self._update_preset_summary()

    def _apply_stage_preset(self) -> None:
        selection = self.preset_dropdown.value
        if selection == _NO_PRESETS:
            self.status.content = "**cannot apply stage preset**: no presets found"
            return
        try:
            candidate = resolve_preset(self.preset_root, Path(selection))
            config = load_preset(candidate)
            source = SessionPresetRef(
                path=candidate.root_relative,
                digest=stage_config_digest(config),
            )
        except (PresetCatalogError, TypeError, ValueError) as error:
            self.status.content = f"**cannot apply stage preset**: {error}"
            return

        self.binder.replace_config(config)
        self.applied_preset = source
        self.status.content = f"stage preset applied: {candidate.root_relative}"
        self._schedule_preview()

    # -- Input tab ----------------------------------------------------------
    #
    # Dropdown selection, Refresh, and the summary markdown are read-only I/O
    # (catalog scan, manifest read) and never touch the worker, the core, or
    # Mitsuba: choosing a candidate is deliberately separate from applying it.
    # Only Load/Rebuild submits a job.

    def _root_and_path_for(self, selection: str) -> tuple[Path, Path]:
        if selection == self._initial_sentinel:
            return self._initial_input_path.parent, Path(self._initial_input_path.name)
        return self.input_root, Path(selection)

    def _dropdown_options(self) -> list[str]:
        relatives = [c.root_relative for c in scan_input_catalog(self.input_root)]
        options = (
            [self._initial_sentinel, *relatives]
            if self._initial_sentinel is not None
            else relatives
        )
        if self._current_selection not in options:
            options = [self._current_selection, *options]
        return options

    def _describe_selection(self, selection: str) -> str:
        root, user_path = self._root_and_path_for(selection)
        try:
            candidate = resolve_candidate(root, user_path)
            summary = describe_candidate(candidate)
        except Exception as error:
            return f"**cannot describe input**: {error}"
        lines = [
            f"- kind: {summary.kind.value}",
            f"- schema: {summary.schema_name} {summary.schema_version}",
            f"- shape (z,y,x): {summary.shape_zyx}",
            f"- voxel size (x,y,z) m: {summary.voxel_size_xyz_m}",
        ]
        if summary.run_id is not None:
            lines.append(f"- run id: {summary.run_id}")
        if summary.provenance_sources:
            lines.append(f"- sources: {', '.join(summary.provenance_sources)}")
        if summary.provenance_notes:
            lines.append(f"- notes: {summary.provenance_notes}")
        return "\n".join(lines)

    def _committed_mapping_selection(self) -> str:
        derivation = self._committed_derivation
        if derivation is None:
            return _AS_IS_MAPPING
        return derivation.mapping_candidate.root_relative

    def _mapping_options(self) -> list[str]:
        relatives = [c.root_relative for c in scan_mapping_catalog(self.mapping_root)]
        options = [_AS_IS_MAPPING, *relatives]
        committed = self._committed_mapping_selection()
        if committed not in options:
            options.append(committed)
        return options

    def _describe_mapping_selection(self, selection: str) -> str:
        if selection == _AS_IS_MAPPING:
            return (
                "Render the selected input's bundle `optical.zarr` as-is "
                "(no material re-mapping)."
            )
        try:
            candidate = resolve_mapping_candidate(self.mapping_root, Path(selection))
            summary = describe_mapping(candidate)
        except Exception as error:
            return f"**cannot describe mapping**: {error}"
        materials = ", ".join(
            f"{material_id}:{name}" for material_id, name in summary.materials
        )
        return "\n".join(
            (
                f"- configuration id: {summary.configuration_id}",
                f"- version: {summary.version}",
                f"- calibration status: {summary.calibration_status}",
                f"- materials: {materials}",
                f"- digest: `{summary.digest}`",
            )
        )

    def _build_input_tab(self, gui) -> None:
        options = self._dropdown_options()
        self.input_dropdown = gui.add_dropdown(
            "input", tuple(options), initial_value=self._current_selection
        )
        self.input_refresh_button = gui.add_button("Refresh")
        self.input_summary = gui.add_markdown(
            self._describe_selection(self._current_selection)
        )
        mapping_options = self._mapping_options()
        mapping_selection = self._committed_mapping_selection()
        if mapping_selection not in mapping_options:
            mapping_selection = _AS_IS_MAPPING
        self.mapping_dropdown = gui.add_dropdown(
            "optical mapping", tuple(mapping_options), initial_value=mapping_selection
        )
        self.mapping_refresh_button = gui.add_button("Refresh")
        self.mapping_summary = gui.add_markdown(
            self._describe_mapping_selection(mapping_selection)
        )
        self.input_load_button = gui.add_button("Load / Rebuild")
        self.session_load_path = gui.add_text(
            "session path (session-root relative)", "viewer.session.json"
        )
        self.session_load_button = gui.add_button("Load session")

        self.input_dropdown.on_update(lambda _event: self._update_input_summary())
        self.input_refresh_button.on_click(lambda _event: self._refresh_input_catalog())
        self.mapping_dropdown.on_update(lambda _event: self._update_mapping_summary())
        self.mapping_refresh_button.on_click(
            lambda _event: self._refresh_mapping_catalog()
        )
        self.input_load_button.on_click(lambda _event: self._queue_load_input())
        self.session_load_button.on_click(lambda _event: self._queue_load_session())

    def _update_input_summary(self) -> None:
        self.input_summary.content = self._describe_selection(self.input_dropdown.value)

    def _refresh_input_catalog(self) -> None:
        options = self._dropdown_options()
        previous = self.input_dropdown.value
        self.input_dropdown.options = tuple(options)
        if previous not in options:
            self.input_dropdown.value = self._current_selection
        self._update_input_summary()

    def _update_mapping_summary(self) -> None:
        self.mapping_summary.content = self._describe_mapping_selection(
            self.mapping_dropdown.value
        )

    def _refresh_mapping_catalog(self) -> None:
        options = self._mapping_options()
        previous = self.mapping_dropdown.value
        self.mapping_dropdown.options = tuple(options)
        if previous not in options:
            self.mapping_dropdown.value = self._committed_mapping_selection()
        self._update_mapping_summary()

    def _load_input_transaction(
        self,
        selection: str,
        mapping_selection: str,
        current_config: StageConfig,
        on_stage: Callable[[str], None],
    ) -> SessionDerivation | None:
        """Validate, build, and swap one Input-tab selection.

        ``mapping_selection == _AS_IS_MAPPING`` delegates entirely to
        :meth:`StageCore.load_input` (Phase 2/3 behaviour, unchanged).
        Otherwise the selected input must be a run bundle; its mapping is
        applied via :func:`regenerate_optical` (reusing a cached derived
        bundle when the source payload and mapping digest are unchanged)
        before the resulting candidate is prepared and swapped in. Returns
        the committed :class:`SessionDerivation`, or ``None`` for an as-is
        load — mirrors :meth:`_load_session_transaction`'s split between a
        testable transaction method and its ``_queue_*`` job wrapper.
        """
        root, user_path = self._root_and_path_for(selection)
        if mapping_selection == _AS_IS_MAPPING:
            self.core.load_input(
                root,
                user_path,
                current_config,
                self.interactive_spp,
                on_stage=on_stage,
            )
            return None

        on_stage("validate")
        try:
            candidate = resolve_candidate(root, user_path)
        except Exception as error:
            raise InputLoadError("validate", str(error)) from error
        if candidate.kind is not InputKind.RUN_BUNDLE:
            raise InputLoadError(
                "validate",
                "applying a mapping requires a run bundle input, "
                f"not {candidate.kind.value}",
            )
        try:
            mapping_candidate = resolve_mapping_candidate(
                self.mapping_root, Path(mapping_selection)
            )
        except MappingCatalogError as error:
            raise InputLoadError("validate", str(error)) from error

        try:
            derived = regenerate_optical(
                candidate.path,
                mapping_candidate,
                self.mapping_work_root,
                on_stage=lambda stage: None if stage == "validate" else on_stage(stage),
            )
        except RegenError as error:
            raise InputLoadError(error.stage, error.message) from error
        if derived.reused:
            on_stage("map: reused cache")

        derived_candidate = InputCandidate(
            kind=InputKind.RUN_BUNDLE,
            root_relative=derived.bundle_path.name,
            path=derived.bundle_path,
            optical_zarr=derived.optical_zarr,
        )
        session = self.core.prepare_candidate_session(
            derived_candidate,
            current_config,
            self.interactive_spp,
            on_stage=lambda stage: None if stage == "validate" else on_stage(stage),
        )
        derivation = SessionDerivation(
            source_candidate=candidate,
            mapping_candidate=mapping_candidate,
            mapping_digest=derived.mapping_digest,
            derived_bundle=derived.bundle_path,
        )
        session.derivation = derivation
        on_stage("swap")
        self.core.swap_session(session)
        return derivation

    def _queue_load_input(self) -> None:
        selection = self.input_dropdown.value
        mapping_selection = self.mapping_dropdown.value
        current_config = self.binder.current()

        self.input_load_button.disabled = True
        self.status.content = "input load: validate…"

        timer = _StageTimer("input load")

        def on_stage(stage: str) -> None:
            suffix = timer.advance(stage)
            self.status.content = f"input load: {stage}…{suffix}"

        def job() -> None:
            try:
                derivation = self._load_input_transaction(
                    selection, mapping_selection, current_config, on_stage
                )
            except InputLoadError as error:
                timer.finish()
                self.status.content = f"**{error}**"
                print(f"INPUT LOAD ERROR {error}", file=sys.stderr)
            else:
                total = timer.finish()
                self._current_selection = selection
                self._committed_derivation = derivation
                self.status.content = f"input loaded ({total:.1f}s)"
                self._schedule_preview()
            finally:
                self.input_load_button.disabled = False

        self.worker.submit(job)


def main() -> None:
    args = _parse_args()
    app = ViewerApp(args)
    print(f"viewer ready: http://127.0.0.1:{args.port} (work dir: {app.work_dir})")
    try:
        while True:
            time.sleep(3600.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
