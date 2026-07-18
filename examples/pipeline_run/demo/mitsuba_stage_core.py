"""Viser-free scene/session/render core for the Mitsuba stage viewer.

Split out of ``mitsuba_stage_viewer.py`` (see
``.devdocs/function/mitsubav_refactor/plan.md``): everything here is
independent of viser and the GUI, so the render/save/reproduce paths can be
exercised by scripts (verification) as well as by the viser bindings in
``mitsuba_stage_viewer.py``.
"""

from __future__ import annotations

import itertools
import re
import shutil
import sys
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import ModuleType

import numpy as np
from mitsuba_stage import RenderSettings, StageConfig, apply_stage
from mitsuba_stage_inputs import InputCandidate, resolve_candidate

from vdbmat.core.volumes import OpticalPropertyVolume
from vdbmat.exporters.mitsuba import MitsubaExportConfig, prepare_mitsuba_scene
from vdbmat.io.zarr import read_volume


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


_CUDA_VARIANT_PREFIX = "cuda"


class DenoiseVariantError(Exception):
    """``render.denoise`` was requested with a non-CUDA Mitsuba variant.

    ``mi.OptixDenoiser`` is CUDA-only; this repo's "no silent approximation"
    principle means a CPU (``llvm_ad_rgb``) request must fail explicitly
    rather than quietly rendering without denoising.
    """


def require_denoise_variant(variant: str) -> None:
    if not variant.startswith(_CUDA_VARIANT_PREFIX):
        raise DenoiseVariantError(
            "render.denoise requires a cuda_ad_rgb-family Mitsuba variant "
            f"(mi.OptixDenoiser is CUDA-only); got variant={variant!r}"
        )


def denoise_image(
    mi: ModuleType,
    image: object,
    width: int,
    height: int,
    cache: dict[tuple[int, int], object],
) -> object:
    """Apply ``mi.OptixDenoiser`` to a rendered image.

    ``cache`` holds one denoiser per ``(width, height)``, reused across
    calls (denoiser construction is per-resolution, not per-render).
    """
    key = (width, height)
    denoiser = cache.get(key)
    if denoiser is None:
        denoiser = mi.OptixDenoiser(input_size=[width, height])
        cache[key] = denoiser
    return denoiser(image)


def finalize_render_image(
    mi: ModuleType,
    image: object,
    render: RenderSettings,
    output_png: Path,
    stats: str,
    denoiser_cache: dict[tuple[int, int], object],
) -> str:
    """Write ``image`` to ``output_png``, honoring ``render.denoise``.

    When ``render.denoise`` is false, writes ``image`` unchanged and returns
    ``stats`` unchanged (pixel-identical to pre-denoise behavior). When true,
    writes the raw image first (as ``<stem>.raw<suffix>``, so PIXELSTATS —
    computed by the caller from the raw pixels — stays a science-track
    regression signal unaffected by denoising), then writes the denoised
    result to ``output_png`` and appends ``" denoise=optix"`` to ``stats``.
    Raises :class:`DenoiseVariantError` if ``mi.variant()`` is not a CUDA
    variant.
    """
    if not render.denoise:
        mi.util.write_bitmap(str(output_png), image, write_async=False)
        return stats
    require_denoise_variant(mi.variant())
    raw_png = output_png.with_name(f"{output_png.stem}.raw{output_png.suffix}")
    mi.util.write_bitmap(str(raw_png), image, write_async=False)
    denoised = denoise_image(mi, image, render.width, render.height, denoiser_cache)
    mi.util.write_bitmap(str(output_png), denoised, write_async=False)
    return f"{stats} denoise=optix"


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
    """Best-effort removal of one no-longer-referenced session directory.

    Used both for a session abandoned by a failed load and for a session
    replaced by a successful swap.
    """
    try:
        if session_dir.exists():
            shutil.rmtree(session_dir)
    except OSError as error:
        print(
            f"INPUT LOAD CLEANUP WARNING: could not remove {session_dir}: {error}",
            file=sys.stderr,
        )


def _sweep_stale_session_dirs(work_dir: Path) -> None:
    """Remove leftover ``work_dir/inputs/`` entries from a prior process.

    Called once at startup, before the initial :class:`StageCore` session is
    built. Scoped to ``work_dir/inputs/`` only — ``derived/`` and any other
    user files under ``work_dir`` are never touched. Each candidate's
    resolved path is verified to stay within ``work_dir`` before removal
    (defense against a symlink planted under ``inputs/``). A missing
    ``inputs/`` directory (fresh ``--work-dir``, or none given) is a no-op.
    """
    inputs_dir = work_dir / _INPUTS_DIRNAME
    if not inputs_dir.is_dir():
        return
    resolved_work_dir = work_dir.resolve()
    for entry in inputs_dir.iterdir():
        try:
            resolved_entry = entry.resolve()
        except OSError:
            continue
        if resolved_work_dir not in resolved_entry.parents:
            continue
        _discard_session_dir(entry)


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
    derivation: object | None = None
    _digest_cache: dict[Path, str] = field(default_factory=dict)

    def cached_digest(self, path: Path, compute: Callable[[Path], str]) -> str:
        """Return ``compute(path)``, computed once per resolved path and reused.

        Digests (a full store/file hash) are expensive; Save session, the
        final-render sidecar, and Verify digests all want "the digest of
        this store for the currently live session" without re-hashing it
        every time they're invoked. The cache is scoped to this
        :class:`InputSession` and is simply dropped when the session is
        replaced by a swap.
        """
        resolved = Path(path).resolve()
        cached = self._digest_cache.get(resolved)
        if cached is not None:
            return cached
        value = compute(resolved)
        self._digest_cache[resolved] = value
        return value

    def peek_digest(self, path: Path) -> str | None:
        """Return the cached digest for ``path`` without computing it."""
        return self._digest_cache.get(Path(path).resolve())

    def refresh_digest(
        self, path: Path, compute: Callable[[Path], str]
    ) -> tuple[str, bool]:
        """Unconditionally re-hash ``path``; return ``(digest, drifted)``.

        ``drifted`` is true only when a previously cached digest for this
        path exists and disagrees with the freshly computed one. The cache
        is updated to the fresh value either way (used by Verify digests).
        """
        resolved = Path(path).resolve()
        fresh = compute(resolved)
        previous = self._digest_cache.get(resolved)
        self._digest_cache[resolved] = fresh
        return fresh, previous is not None and previous != fresh


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
        self._denoiser_cache: dict[tuple[int, int], object] = {}
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
        """Replace the current session and advance the session generation.

        Discards the replaced session's own work directory (its
        ``inputs/NNN-slug/`` PLY/scene side-effect files). Safe because the
        render worker runs one job at a time and nothing outside
        :class:`StageCore` holds a reference across a swap (see
        :class:`InputSession`): a queued preview/final-render job always
        reads ``self._session`` at execution time, never a captured old
        session, so the replaced directory is never read again once this
        method returns.
        """
        old_session = self._session
        self._session = session
        self.session_generation += 1
        if old_session.work_dir != session.work_dir:
            _discard_session_dir(old_session.work_dir)

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
        """Render a preview; return (uint8 sRGB image, stats, update route).

        Denoising (``config.render.denoise``) applies only to the settled
        preview (``spp is None``, i.e. the caller did not ask for a specific
        interactive sample count) — never to the low-spp interactive preview,
        to keep drag-time responsiveness unaffected. Stats are always
        computed from the raw (pre-denoise) pixels.
        """
        preview_config = _preview_stage_config(
            config, self.preview_size, self.preview_spp
        )
        image, route = self._session.preview_scene.render(
            preview_config, self.preview_spp if spp is None else spp
        )
        stats = _pixel_stats(
            np.asarray(image, dtype=np.float32), preview_config.render.max_depth
        )
        if spp is None and preview_config.render.denoise:
            require_denoise_variant(self.mi.variant())
            image = denoise_image(
                self.mi,
                image,
                preview_config.render.width,
                preview_config.render.height,
                self._denoiser_cache,
            )
            stats = f"{stats} denoise=optix"
        bitmap = self.mi.util.convert_to_bitmap(image)
        return np.asarray(bitmap), stats, route

    def render_final(self, config: StageConfig, output_png: Path) -> str:
        """Render at the config's full resolution/spp and write the PNG.

        Reads ``self._session`` once, at call time, so a job already queued
        on the render worker always renders whichever session is current
        when it actually runs (not whichever was current when it was
        submitted). When ``config.render.denoise`` is set, also writes the
        raw (pre-denoise) image as ``<output_png stem>.raw<suffix>`` and
        computes PIXELSTATS from that raw image (see
        :func:`finalize_render_image`).
        """
        session = self._session
        self._ensure_final(session, config.render)
        image = self._render(
            session.base_final, session.volume, config, config.render.spp, session.seed
        )
        output_png.parent.mkdir(parents=True, exist_ok=True)
        stats = _pixel_stats(
            np.asarray(image, dtype=np.float32), config.render.max_depth
        )
        return finalize_render_image(
            self.mi, image, config.render, output_png, stats, self._denoiser_cache
        )

    @staticmethod
    def save_preset(config: StageConfig, path: Path) -> None:
        import json

        from mitsuba_stage import stage_config_to_dict

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
