"""Browser GUI for tuning the Mitsuba stage (viser + live re-render).

This is a demo-track helper; it is not part of the canonical pipeline and
produces qualitative, uncalibrated images. ``prepare_mitsuba_scene()`` /
``render_mitsuba()`` / ``MitsubaExportConfig`` are untouched, and this viewer
is a pure consumer of the Phase-1 contract in :mod:`mitsuba_stage`: it edits a
``StageConfig``, previews it, and its durable outputs are a ``*.stage.json``
preset, a digest-pinned ``*.session.json`` manifest, and a final PNG.

Architecture (see .devdocs/vision/mitsuba_gui/p3/plan.md and
.devdocs/function/mitsubav_refactor/plan.md): the scene/session/render
core lives in :mod:`mitsuba_stage_core`, the GUI field bindings in
:mod:`mitsuba_stage_binder`, and this module wires them to viser.

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
import math
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from mitsuba_stage import StageConfig, stage_config_from_json
from mitsuba_stage_binder import StageBinder as StageBinder
from mitsuba_stage_core import _RUN_MANIFEST_NAME, StageCore
from mitsuba_stage_core import InputLoadError as InputLoadError
from mitsuba_stage_core import InputSession as InputSession
from mitsuba_stage_core import RenderWorker as RenderWorker
from mitsuba_stage_core import TraversedPreviewScene as TraversedPreviewScene
from mitsuba_stage_core import _discard_session_dir as _discard_session_dir
from mitsuba_stage_core import _final_render_key as _final_render_key
from mitsuba_stage_core import _preview_stage_config as _preview_stage_config
from mitsuba_stage_core import (
    _resolve_initial_optical_zarr as _resolve_initial_optical_zarr,
)
from mitsuba_stage_core import _session_work_dir as _session_work_dir
from mitsuba_stage_core import _slug_for as _slug_for
from mitsuba_stage_core import _StageTimer as _StageTimer
from mitsuba_stage_core import _structure_key as _structure_key
from mitsuba_stage_core import _sweep_stale_session_dirs as _sweep_stale_session_dirs
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
    ViewerSession,
    ViewerSessionError,
    create_viewer_session,
    resolve_viewer_session,
    verify_derived_optical,
    viewer_session_from_json,
    write_viewer_session,
)

from vdbmat.exporters.mitsuba import MitsubaExportConfig
from vdbmat.pipeline import sha256_file, zarr_store_sha256


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
    reused: bool = False


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
            reused=derived.reused,
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


_NO_PRESETS = "(no presets found)"
_AS_IS_MAPPING = "(bundle optical as-is)"


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


class ViewerApp:
    """Wires StageCore, RenderWorker, and the viser GUI together."""

    def __init__(self, args: argparse.Namespace) -> None:
        import viser

        work_dir = args.work_dir
        if work_dir is None:
            work_dir = Path(tempfile.mkdtemp(prefix="mitsuba-stage-viewer-"))
        work_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir = work_dir
        _sweep_stale_session_dirs(work_dir)

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
            variant=startup.variant,
        )
        self._update_effective_state()

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
        self._update_effective_state()
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

    def _build_current_session_document(self) -> ViewerSession:
        """Verify and capture the live session as a ``vdbmat.viewer-session`` doc.

        Shared by Save session and the final-render sidecar (Step 3) so both
        reject exactly the same conditions and reuse the same digest cache.
        Digests come from the current :class:`InputSession`'s cache,
        computed (and cached) on first use here.
        """
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
                derived_digest = live_session.cached_digest(
                    live_session.optical_zarr, zarr_store_sha256
                )
            except OSError as error:
                raise ViewerSessionError(
                    "capture", f"cannot digest derived optical store: {error}"
                ) from error
            mapping_ref = SessionMappingRef(
                path=derivation.mapping_candidate.root_relative,
                digest=current_mapping_digest,
                derived_optical_sha256=derived_digest,
            )
        try:
            input_digest = live_session.cached_digest(
                candidate.optical_zarr, zarr_store_sha256
            )
            run_digest = (
                live_session.cached_digest(
                    candidate.path / _RUN_MANIFEST_NAME, sha256_file
                )
                if candidate.kind is InputKind.RUN_BUNDLE
                else None
            )
        except OSError as error:
            raise ViewerSessionError(
                "capture", f"cannot digest input: {error}"
            ) from error
        return create_viewer_session(
            candidate,
            self.binder.current(),
            self.core.mi.variant(),
            live_session.seed,
            preset=self.applied_preset,
            mapping=mapping_ref,
            optical_digest=input_digest,
            run_manifest_digest=run_digest,
        )

    def _capture_session(self, path: Path) -> None:
        session = self._build_current_session_document()
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
                reused=derived.reused,
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
                self._update_effective_state()
                self._schedule_preview()
            finally:
                self.session_load_button.disabled = False

        self.worker.submit(job)

    def _final_sidecar_path(self, output_png: Path) -> Path:
        return output_png.with_name(f"{output_png.stem}.session.json")

    def _write_final_sidecar(self, output_png: Path) -> str | None:
        """Write ``<basename>.session.json`` next to a successful final render.

        Reuses :meth:`_build_current_session_document` (the same
        construction/validation Save session uses — no new schema) so the
        sidecar is a plain ``vdbmat.viewer-session`` document that
        ``mitsuba_stage_demo.py --session`` can replay directly. A render
        that cannot currently produce a valid session document (initial
        sentinel input, stale mapping — the same conditions Save session
        already rejects) is not a render failure: the PNG is kept and only
        the sidecar is skipped, with the reason surfaced in the status line.
        """
        try:
            document = self._build_current_session_document()
        except ViewerSessionError as error:
            print(f"FINAL SIDECAR SKIPPED {error}", file=sys.stderr)
            return f"final: session sidecar skipped: {error}"
        sidecar_path = self._final_sidecar_path(output_png)
        write_viewer_session(sidecar_path, document)
        print(f"FINAL SIDECAR {sidecar_path}")
        return None

    def _queue_final(self) -> None:
        config = self.binder.current()
        path = Path(self.final_path.value)
        self.status.content = "final render queued…"

        def job() -> None:
            timer = _StageTimer("final render")
            timer.advance("render")
            stats = self.core.render_final(config, path)
            elapsed = timer.finish()
            sidecar_note = self._write_final_sidecar(path)
            status = f"final {elapsed:.1f}s → {path} — PIXELSTATS {stats}"
            if sidecar_note is not None:
                status = f"{status}\n{sidecar_note}"
            self.status.content = status
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
        self._update_effective_state()
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

    def _describe_effective_state(self) -> str:
        """Render-only markdown for the *committed* session (Phase 5 Step 3).

        Never reads the Input/mapping/preset dropdowns — only what is
        actually loaded (``self.core.current_session``, ``self._current_selection``,
        ``self.applied_preset``, ``self.binder.current()``). Digests are
        shown only if already cached on the current session (``not
        computed`` otherwise); this method never hashes anything itself.
        """
        session = self.core.current_session
        derivation = session.derivation
        config = self.binder.current()

        root, user_path = self._root_and_path_for(self._current_selection)
        try:
            candidate = resolve_candidate(root, user_path)
            summary = describe_candidate(candidate)
            kind = summary.kind.value
            run_id = summary.run_id if summary.run_id is not None else "(none)"
        except Exception as error:
            candidate = None
            kind = "(unresolvable)"
            run_id = f"(error: {error})"

        lines = [
            "**input**",
            f"- kind: {kind}",
            f"- path: {self._current_selection}",
            f"- run id: {run_id}",
            "",
            "**derivation**",
        ]
        if derivation is None:
            lines.append("- as-is (no mapping applied)")
        else:
            lines.extend(
                [
                    f"- mapping: {derivation.mapping_candidate.root_relative}",
                    f"- mapping digest: `{derivation.mapping_digest}`",
                    f"- derived bundle: {derivation.derived_bundle}",
                    f"- cache reused: {'yes' if derivation.reused else 'no'}",
                ]
            )

        lines.extend(["", "**stage**"])
        if self.applied_preset is not None:
            lines.append(
                f"- preset: {self.applied_preset.path} (`{self.applied_preset.digest}`)"
            )
        else:
            lines.append("- inline (no preset provenance)")

        lines.extend(
            [
                "",
                "**render**",
                f"- width x height: {config.render.width} x {config.render.height}",
                f"- spp: {config.render.spp}",
                f"- max_depth: {config.render.max_depth}",
                "",
                "**mitsuba**",
                f"- variant: {self.core.mi.variant()}",
                f"- seed: {session.seed}",
                "",
                "**digests**",
            ]
        )
        if derivation is not None:
            input_optical_path = derivation.source_candidate.optical_zarr
        elif candidate is not None:
            input_optical_path = candidate.optical_zarr
        else:
            input_optical_path = session.optical_zarr
        input_digest = session.peek_digest(input_optical_path)
        lines.append(f"- input optical: {input_digest or 'not computed'}")
        if derivation is not None:
            derived_digest = session.peek_digest(session.optical_zarr)
            lines.append(f"- derived optical: {derived_digest or 'not computed'}")
        return "\n".join(lines)

    def _update_effective_state(self) -> None:
        self.effective_state.content = self._describe_effective_state()

    def _verify_digests(self) -> str:
        """Re-hash the live session's stores/mapping; report drift from cache.

        Runs on the render worker (I/O-bound hashing). Unlike
        :meth:`_describe_effective_state`, this always re-reads the
        filesystem — it is the one explicit, user-triggered way to detect an
        externally modified input bundle or mapping file, since digests are
        otherwise only computed once and cached per session.
        """
        session = self.core.current_session
        derivation = session.derivation
        findings: list[str] = []

        root, user_path = self._root_and_path_for(self._current_selection)
        try:
            candidate = resolve_candidate(root, user_path)
        except Exception as error:
            return f"verify digests: cannot resolve current input: {error}"

        try:
            _fresh, drifted = session.refresh_digest(
                candidate.optical_zarr, zarr_store_sha256
            )
        except OSError as error:
            return f"verify digests: cannot hash input optical store: {error}"
        if drifted:
            findings.append("input optical store changed since last digest")

        if candidate.kind is InputKind.RUN_BUNDLE:
            try:
                _fresh, drifted = session.refresh_digest(
                    candidate.path / _RUN_MANIFEST_NAME, sha256_file
                )
            except OSError as error:
                return f"verify digests: cannot hash run manifest: {error}"
            if drifted:
                findings.append("run manifest changed since last digest")

        if derivation is not None:
            try:
                _fresh, drifted = session.refresh_digest(
                    session.optical_zarr, zarr_store_sha256
                )
            except OSError as error:
                return f"verify digests: cannot hash derived optical store: {error}"
            if drifted:
                findings.append("derived optical store changed since last digest")
            try:
                current_mapping_candidate = resolve_mapping_candidate(
                    self.mapping_root, Path(derivation.mapping_candidate.root_relative)
                )
                current_mapping_digest = load_mapping(current_mapping_candidate).digest
            except MappingCatalogError as error:
                findings.append(f"mapping unreadable: {error}")
            else:
                if current_mapping_digest != derivation.mapping_digest:
                    findings.append("mapping file changed since it was applied")

        if findings:
            return "verify digests: drift — " + "; ".join(findings)
        return "verify digests: ok — matches cached digests"

    def _queue_verify_digests(self) -> None:
        self.verify_digests_button.disabled = True
        self.status.content = "verify digests: hashing…"

        def job() -> None:
            try:
                result = self._verify_digests()
            except Exception as error:
                self.status.content = f"**verify digests failed**: {error}"
                print(f"VERIFY DIGESTS ERROR {error}", file=sys.stderr)
            else:
                self.status.content = result
                self._update_effective_state()
                print(f"VERIFY {result}")
            finally:
                self.verify_digests_button.disabled = False

        self.worker.submit(job)

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

        # Effective state reflects the *committed* session only — never the
        # dropdown selections above, which are free to change without
        # touching the live scene (Phase 2's "selection vs. apply" split).
        # Populated with real content once ``self.binder`` exists (see the
        # end of ``ViewerApp.__init__``): this tab is built *during*
        # ``StageBinder.__init__``, before that attribute is assigned.
        self.effective_state = gui.add_markdown("(pending)")
        self.verify_digests_button = gui.add_button("Verify digests")

        self.input_dropdown.on_update(lambda _event: self._update_input_summary())
        self.input_refresh_button.on_click(lambda _event: self._refresh_input_catalog())
        self.mapping_dropdown.on_update(lambda _event: self._update_mapping_summary())
        self.mapping_refresh_button.on_click(
            lambda _event: self._refresh_mapping_catalog()
        )
        self.input_load_button.on_click(lambda _event: self._queue_load_input())
        self.session_load_button.on_click(lambda _event: self._queue_load_session())
        self.verify_digests_button.on_click(lambda _event: self._queue_verify_digests())

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
            reused=derived.reused,
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
                self._update_effective_state()
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
