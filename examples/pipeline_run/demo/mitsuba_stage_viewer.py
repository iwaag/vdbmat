"""Browser GUI for tuning the Mitsuba stage (viser + live re-render).

This is a demo-track helper; it is not part of the canonical pipeline and
produces qualitative, uncalibrated images. ``prepare_mitsuba_scene()`` /
``render_mitsuba()`` / ``MitsubaExportConfig`` are untouched, and this viewer
is a pure consumer of the Phase-1 contract in :mod:`mitsuba_stage`: it edits a
``StageConfig``, previews it, and its only durable outputs are a
``*.stage.json`` preset (replayable headlessly with
``mitsuba_stage_demo.py --stage-config``) and a final PNG.

Architecture (see .devdocs/vision/mitsuba_gui/p3/plan.md):

- ``prepare_mitsuba_scene()`` runs exactly twice at startup — once with a
  low-resolution preview sensor, once at the final resolution — so the heavy
  boundary-mesh extraction and PLY writing never sit inside the parameter
  loop. The final scene is rebuilt only when its requested resolution changes.
- The preview scene stays loaded. Continuous changes update explicit
  ``mi.traverse()`` keys; graph changes (enabled/pattern/override toggles)
  rebuild through ``apply_stage()`` and then resume traversed updates.
- All renders run on one worker thread. A change produces a low-spp
  interactive image immediately and a settled high-spp image after the input
  goes quiet. Generation guards prevent stale renders from reaching the GUI.
- Preview renders swap the config's ``render`` section for the preview
  resolution before calling ``apply_stage``, so a camera override previews at
  preview resolution while "Render final" and the saved preset keep the real
  ``render`` settings. Final renders use the same width/height/spp/seed as the
  headless demo, so a saved preset reproduces the final PNG pixel-identically.
- GUI decompositions (radiance = colour picker x intensity slider, key-light
  direction = azimuth/elevation sliders) exist only inside the GUI. The saved
  JSON is the unchanged Phase-1 schema, and fields the user never touched are
  written back exactly as loaded (per-field dirty tracking), so lossy
  widget quantisation cannot creep into an untouched preset field.

Invoke on the host (no Docker needed for Mitsuba):

    uv run --group mitsuba-viewer python \
        examples/pipeline_run/demo/mitsuba_stage_viewer.py -- \
        OPTICAL_ZARR [--stage-config PRESET.stage.json] [--port 8080] \
        [--work-dir DIR] [--preview-size 256] [--preview-spp 16] \
        [--interactive-spp 4] [--settle-delay 0.35] \
        [--variant llvm_ad_rgb|cuda_ad_rgb] \
        [--preset-out PATH] [--final-out PATH]

``--work-dir`` (default: a fresh temp directory) receives the PLY/scene
side-effect files and the default preset/PNG outputs; point it at
``.local/...`` to keep artifacts with the repo checkout.
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import replace
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

from vdbmat.core.volumes import OpticalPropertyVolume
from vdbmat.exporters.mitsuba import MitsubaExportConfig, prepare_mitsuba_scene
from vdbmat.io.zarr import read_volume


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    parser = argparse.ArgumentParser(prog="mitsuba_stage_viewer")
    parser.add_argument("optical_zarr", type=Path)
    parser.add_argument("--stage-config", type=Path, default=None)
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
        default="llvm_ad_rgb",
        help="Mitsuba execution backend (default: llvm_ad_rgb, CPU)",
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
    return args


def _load_mitsuba(variant: str) -> ModuleType:
    import importlib

    mi = importlib.import_module("mitsuba")
    if mi.variant() != variant:
        mi.set_variant(variant)
    return mi


def _pixel_stats(pixels: np.ndarray) -> str:
    return (
        f"min={float(np.min(pixels)):.6g} "
        f"max={float(np.max(pixels)):.6g} "
        f"mean={float(np.mean(pixels)):.6g} "
        f"std={float(np.std(pixels)):.6g}"
    )


StructureKey = tuple[bool, str, bool, str, bool, bool, bool]


def _structure_key(config: StageConfig) -> StructureKey:
    return (
        config.backdrop.enabled,
        config.backdrop.pattern,
        config.floor.enabled,
        config.floor.pattern,
        config.key_light.enabled,
        config.camera is not None,
        config.backlight is not None,
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
                    self._set(
                        f"{name}.bsdf.reflectance.color0.value", section.color0
                    )
                if section.color1 != old_section.color1:
                    self._set(
                        f"{name}.bsdf.reflectance.color1.value", section.color1
                    )
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
                    or config.key_light.distance_factor
                    != old.key_light.distance_factor
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


class StageCore:
    """Viser-free rendering core: prepare twice, then cheap re-renders.

    Kept independent of the GUI so the render/save/reproduce paths can be
    exercised by scripts (verification) as well as by the viser bindings.
    """

    def __init__(
        self,
        optical_zarr: Path,
        work_dir: Path,
        preview_size: int,
        preview_spp: int,
        initial: StageConfig,
        variant: str = "llvm_ad_rgb",
    ) -> None:
        volume = read_volume(optical_zarr)
        if not isinstance(volume, OpticalPropertyVolume):
            raise SystemExit(f"{optical_zarr} is not an optical property volume")
        self.volume = volume
        self.work_dir = work_dir
        self.preview_spp = preview_spp
        self._preview_render = RenderSettings(
            width=preview_size, height=preview_size, spp=preview_spp
        )
        self._seed = MitsubaExportConfig().seed
        self.mi = _load_mitsuba(variant)

        preview_config = MitsubaExportConfig(
            width=preview_size,
            height=preview_size,
            spp=preview_spp,
            variant=variant,
        )
        self._base_preview = prepare_mitsuba_scene(
            volume, work_dir / "preview_scene", config=preview_config
        )
        preview_initial = replace(initial, render=self._preview_render)
        self._preview_scene = TraversedPreviewScene(
            self.mi,
            self._base_preview,
            self.volume.geometry,
            preview_initial,
            self._seed,
        )
        self._base_final = None
        self._final_res: tuple[int, int] | None = None
        self._ensure_final(initial.render)

    def _ensure_final(self, render: RenderSettings) -> None:
        if self._final_res == (render.width, render.height):
            return
        config = MitsubaExportConfig(
            width=render.width,
            height=render.height,
            spp=render.spp,
            variant=self.mi.variant(),
        )
        self._base_final = prepare_mitsuba_scene(
            self.volume, self.work_dir / "final_scene", config=config
        )
        self._final_res = (render.width, render.height)

    def _render(self, base, config: StageConfig, spp: int) -> np.ndarray:
        scene_dict = dict(base.scene_dict)
        apply_stage(self.mi, scene_dict, self.volume.geometry, config)
        scene = self.mi.load_dict(scene_dict)
        return self.mi.render(scene, seed=self._seed, spp=spp)

    def render_preview(
        self, config: StageConfig, spp: int | None = None
    ) -> tuple[np.ndarray, str, str]:
        """Render a preview; return (uint8 sRGB image, stats, update route)."""
        preview_config = replace(config, render=self._preview_render)
        image, route = self._preview_scene.render(
            preview_config, self.preview_spp if spp is None else spp
        )
        stats = _pixel_stats(np.asarray(image, dtype=np.float32))
        bitmap = self.mi.util.convert_to_bitmap(image)
        return np.asarray(bitmap), stats, route

    def render_final(self, config: StageConfig, output_png: Path) -> str:
        """Render at the config's full resolution/spp and write the PNG."""
        self._ensure_final(config.render)
        image = self._render(self._base_final, config, config.render.spp)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        self.mi.util.write_bitmap(str(output_png), image, write_async=False)
        return _pixel_stats(np.asarray(image, dtype=np.float32))

    @staticmethod
    def save_preset(config: StageConfig, path: Path) -> None:
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(stage_config_to_dict(config), handle, indent=2)
            handle.write("\n")


class RenderWorker(threading.Thread):
    """Single worker: immediate coarse preview, then latest settled preview."""

    def __init__(self, settle_delay: float = 0.35) -> None:
        super().__init__(daemon=True)
        self._condition = threading.Condition()
        self._pending_preview: tuple[int, StageConfig] | None = None
        self._generation = 0
        self._preview_fn: Callable[[StageConfig, str], object] | None = None
        self._publish_fn: Callable[[object, str], None] | None = None
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
    ) -> None:
        self._preview_fn = preview_fn
        self._publish_fn = publish_fn
        self._on_error = on_error

    def request_preview(self, config: StageConfig) -> int:
        with self._condition:
            self._generation += 1
            self._pending_preview = (self._generation, config)
            self._condition.notify()
            return self._generation

    def submit(self, job: Callable[[], None]) -> None:
        with self._condition:
            self._jobs.append(job)
            self._condition.notify()

    def _publish(self, result: object, quality: str, generation: int) -> None:
        with self._condition:
            current = self._generation
        if generation == current and self._publish_fn is not None:
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
                generation, config = preview
                result = self._preview_fn(config, "interactive")
                self._publish(result, "interactive", generation)
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
                    self._publish(result, "settled", generation)
            except Exception:
                self._on_error(traceback.format_exc(limit=3))


def _decompose_radiance(radiance: RGB) -> tuple[tuple[int, int, int], float]:
    intensity = max(radiance)
    if intensity <= 0.0:
        return (0, 0, 0), 1.0
    colour = tuple(
        round(min(max(component / intensity, 0.0), 1.0) * 255)
        for component in radiance
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
    return tuple(
        round(min(max(component, 0.0), 1.0) * 255) for component in colour
    )


class StageBinder:
    """GUI <-> StageConfig binding with per-field dirty tracking.

    Fields backed by lossy widgets (sliders, 0-255 colour pickers, and the
    GUI-only radiance/direction decompositions) are read from the GUI only
    once the user has touched them; otherwise the value loaded at startup is
    passed through untouched. Exact widgets (checkboxes, dropdowns, number
    inputs) are always read directly.
    """

    def __init__(
        self, server, base: StageConfig, on_change: Callable[[], None]
    ) -> None:
        self.base = base
        self.dirty: set[str] = set()
        self._on_change = on_change
        gui = server.gui
        camera_base = base.camera if base.camera is not None else CameraOverride()
        backlight_base = (
            base.backlight if base.backlight is not None else BacklightOverride()
        )
        key_colour, key_intensity = _decompose_radiance(base.key_light.radiance)
        back_colour, back_intensity = _decompose_radiance(backlight_base.radiance)
        key_azimuth, key_elevation = _decompose_direction(base.key_light.direction)

        with gui.add_folder("Render (final only)", expand_by_default=False):
            self.width = gui.add_number(
                "width", base.render.width, min=16, max=4096, step=16
            )
            self.height = gui.add_number(
                "height", base.render.height, min=16, max=4096, step=16
            )
            self.spp = gui.add_number("spp", base.render.spp, min=1, max=4096, step=1)
        with gui.add_folder("Backdrop"):
            self.backdrop_enabled = gui.add_checkbox(
                "enabled", base.backdrop.enabled
            )
            self.backdrop_pattern = gui.add_dropdown(
                "pattern", ("checker", "solid"), initial_value=base.backdrop.pattern
            )
            self.backdrop_distance = self._slider(
                gui, "distance", 0.2, 10.0, base.backdrop.distance_factor,
                "backdrop.distance_factor",
            )
            self.backdrop_scale = self._slider(
                gui, "scale", 0.2, 10.0, base.backdrop.scale_factor,
                "backdrop.scale_factor",
            )
            self.backdrop_checker = self._int_slider(
                gui, "checker tiles", base.backdrop.checker_scale,
                "backdrop.checker_scale",
            )
            self.backdrop_color0 = self._rgb(
                gui, "color0", base.backdrop.color0, "backdrop.color0"
            )
            self.backdrop_color1 = self._rgb(
                gui, "color1", base.backdrop.color1, "backdrop.color1"
            )
        with gui.add_folder("Floor"):
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
        with gui.add_folder("Key light"):
            self.key_enabled = gui.add_checkbox("enabled", base.key_light.enabled)
            self.key_azimuth = self._slider(
                gui, "azimuth °", -180.0, 180.0, key_azimuth, "key_light.direction"
            )
            self.key_elevation = self._slider(
                gui, "elevation °", -89.0, 89.0, key_elevation, "key_light.direction"
            )
            self.key_distance = self._slider(
                gui, "distance", 0.5, 10.0, base.key_light.distance_factor,
                "key_light.distance_factor",
            )
            self.key_scale = self._slider(
                gui, "scale", 0.1, 5.0, base.key_light.scale_factor,
                "key_light.scale_factor",
            )
            self.key_colour = self._rgb_raw(
                gui, "colour", key_colour, "key_light.radiance"
            )
            self.key_intensity = self._slider(
                gui, "intensity", 0.0, 30.0, key_intensity, "key_light.radiance"
            )
        with gui.add_folder("Camera", expand_by_default=False):
            self.camera_enabled = gui.add_checkbox(
                "override camera", base.camera is not None
            )
            self.camera_azimuth = self._slider(
                gui, "azimuth °", -180.0, 180.0, camera_base.azimuth_deg,
                "camera.azimuth_deg",
            )
            self.camera_elevation = self._slider(
                gui, "elevation °", -89.0, 89.0, camera_base.elevation_deg,
                "camera.elevation_deg",
            )
            self.camera_distance = self._slider(
                gui, "distance", 1.0, 12.0, camera_base.distance_factor,
                "camera.distance_factor",
            )
            self.camera_fov = self._slider(
                gui, "fov °", 10.0, 120.0, camera_base.fov_deg, "camera.fov_deg"
            )
        with gui.add_folder("Backlight", expand_by_default=False):
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
            self.width, self.height, self.spp,
            self.backdrop_enabled, self.backdrop_pattern,
            self.floor_enabled, self.floor_pattern,
            self.key_enabled, self.camera_enabled, self.backlight_enabled,
        ):
            handle.on_update(lambda _event: self._on_change())

    def _slider(self, gui, label, low, high, initial, key):
        clamped = min(max(float(initial), low), high)
        handle = gui.add_slider(label, min=low, max=high, step=0.01,
                                initial_value=clamped)
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
            self.dirty.add(key)
            self._on_change()

        handle.on_update(_mark)

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
                    "camera.azimuth_deg", self.camera_azimuth,
                    camera_base.azimuth_deg,
                ),
                elevation_deg=self._value(
                    "camera.elevation_deg", self.camera_elevation,
                    camera_base.elevation_deg,
                ),
                distance_factor=self._value(
                    "camera.distance_factor", self.camera_distance,
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
            ),
            backdrop=BackdropSettings(
                enabled=self.backdrop_enabled.value,
                pattern=self.backdrop_pattern.value,
                distance_factor=self._value(
                    "backdrop.distance_factor", self.backdrop_distance,
                    base.backdrop.distance_factor,
                ),
                scale_factor=self._value(
                    "backdrop.scale_factor", self.backdrop_scale,
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
                    "key_light.distance_factor", self.key_distance,
                    base.key_light.distance_factor,
                ),
                scale_factor=self._value(
                    "key_light.scale_factor", self.key_scale,
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

        if args.stage_config is not None:
            initial = stage_config_from_json(args.stage_config)
        else:
            initial = StageConfig()

        self.core = StageCore(
            args.optical_zarr,
            work_dir,
            preview_size=args.preview_size,
            preview_spp=args.preview_spp,
            initial=initial,
            variant=args.variant,
        )
        self.interactive_spp = args.interactive_spp
        self.worker = RenderWorker(settle_delay=args.settle_delay)
        self.server = viser.ViserServer(host="127.0.0.1", port=args.port)
        gui = self.server.gui

        placeholder = np.zeros(
            (args.preview_size, args.preview_size, 3), dtype=np.uint8
        )
        self.image = gui.add_image(placeholder, label="preview")
        self.status = gui.add_markdown("starting…")
        self.binder = StageBinder(self.server, initial, self._schedule_preview)

        preset_default = args.preset_out or (work_dir / "viewer.stage.json")
        final_default = args.final_out or (work_dir / "final.png")
        self.preset_path = gui.add_text("preset path", str(preset_default))
        save_button = gui.add_button("Save preset")
        self.final_path = gui.add_text("final PNG path", str(final_default))
        render_button = gui.add_button("Render final")

        save_button.on_click(lambda _event: self._save_preset())
        render_button.on_click(lambda _event: self._queue_final())

        self.worker.configure(
            self._render_preview, self._publish_preview, self._show_error
        )
        self.worker.start()
        self._schedule_preview()

    # -- worker-side operations -------------------------------------------

    def _schedule_preview(self) -> None:
        self.status.content = "rendering preview…"
        self.worker.request_preview(self.binder.current())

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
        self.image.image = pixels
        self.status.content = (
            f"preview {quality}/{route} {elapsed:.2f}s — PIXELSTATS {stats}"
        )

    def _save_preset(self) -> None:
        path = Path(self.preset_path.value)
        self.core.save_preset(self.binder.current(), path)
        self.status.content = f"preset saved: {path}"
        print(f"PRESET saved {path}")

    def _queue_final(self) -> None:
        config = self.binder.current()
        path = Path(self.final_path.value)
        self.status.content = "final render queued…"

        def job() -> None:
            started = time.perf_counter()
            stats = self.core.render_final(config, path)
            elapsed = time.perf_counter() - started
            self.status.content = (
                f"final {elapsed:.1f}s → {path} — PIXELSTATS {stats}"
            )
            print(f"FINAL {path} PIXELSTATS {stats}")

        self.worker.submit(job)

    def _show_error(self, message: str) -> None:
        self.status.content = f"**render error**\n```\n{message}\n```"
        print(f"RENDER ERROR {message}", file=sys.stderr)


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
