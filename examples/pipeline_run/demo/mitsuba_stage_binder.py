"""GUI <-> StageConfig field binding for the Mitsuba stage viewer.

Split out of ``mitsuba_stage_viewer.py`` (see
``.devdocs/function/mitsubav_refactor/plan.md``). ``StageBinder`` depends
only on a viser ``server`` and a ``StageConfig``; the Input/Preset tab
contents are injected as callables by ``ViewerApp``, so this module has no
dependency on ``mitsuba_stage_viewer``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

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
)


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
        variant: str = "llvm_ad_rgb",
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
            self.denoise = gui.add_checkbox(
                "denoise (OptiX)",
                base.render.denoise,
                disabled=not variant.startswith("cuda"),
                hint="Applies mi.OptixDenoiser to final render and settled "
                "preview. Requires a cuda_ad_rgb-family variant.",
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
            self.denoise,
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
                (self.denoise, config.render.denoise),
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
                denoise=bool(self.denoise.value),
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
