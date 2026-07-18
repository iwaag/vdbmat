"""Stage configuration and scene assembly for the Mitsuba stage demo.

This is a demo-track module (imported by :mod:`mitsuba_stage_demo`); it is not
part of the canonical pipeline and produces qualitative, uncalibrated scenes.
``vdbmat.exporters.mitsuba.prepare_mitsuba_scene()`` / ``render_mitsuba()`` /
``MitsubaExportConfig`` are untouched by this module.

It holds two things:

1. :class:`StageConfig` — a JSON round-trippable description of the legible
   stage added around the canonical scene: checkerboard backdrop and floor
   planes, an oblique key light, render resolution/spp/max depth, and optional
   overrides for the canonical sensor and backlight. Every default equals the
   value that was previously hardcoded in ``mitsuba_stage_demo.py``, so a
   default ``StageConfig()`` reproduces the pre-refactor image exactly.
2. :func:`apply_stage` — a function that adds those stage entries to the
   ``scene_dict`` returned by ``prepare_mitsuba_scene()``. The canonical
   entries (medium, exterior/interior meshes, sensor, backlight) are left
   untouched unless the config *explicitly* provides a ``camera`` or
   ``backlight`` override; the defaults (``None``) mean "pass through", which
   is what structurally guarantees pixel-identity of the default output.

Stage-config JSON files (``*.stage.json``) carry a format header::

    {"format": "vdbmat.stage-config", "format_version": "1.2.0", ...}

Sections and fields may be given partially; anything omitted keeps its
default. Unknown keys, wrong types, and out-of-range values are rejected
explicitly rather than ignored. Readers also accept legacy versions ``1.0.0``
and ``1.1.0``; version ``1.0.0`` does not allow ``render.max_depth`` (default
8 supplied) and neither ``1.0.0`` nor ``1.1.0`` allow ``render.denoise``
(default ``False`` supplied).

Camera-override convention (only used when ``camera`` is non-null): the
direction from the object to the camera is built from ``azimuth_deg``
(measured in the XY plane from +X toward +Y) and ``elevation_deg`` (from the
XY plane toward +Z), with the Z-up convention shared by the canonical sensor
and backlight. The camera sits at ``center + direction * radius *
distance_factor``. The canonical sensor's direction ``(1.6, -2.2, 1.4)``
corresponds to ``azimuth_deg=-54.0``, ``elevation_deg=27.2`` and its distance
to ``distance_factor≈5.07`` at the canonical 35° FOV — but exact agreement
with the canonical sensor is not a contract of the override path.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from vdbmat.core.geometry import GridGeometry

STAGE_CONFIG_FORMAT = "vdbmat.stage-config"
STAGE_CONFIG_FORMAT_VERSION = "1.2.0"
STAGE_CONFIG_ACCEPTED_VERSIONS = frozenset(
    {"1.0.0", "1.1.0", STAGE_CONFIG_FORMAT_VERSION}
)

_PATTERNS = ("checker", "solid")

RGB = tuple[float, float, float]


class StageConfigError(ValueError):
    """Raised when a stage-config value or JSON document is invalid."""


def _check_rgb(name: str, value: object) -> RGB:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 3
        or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value)
    ):
        raise StageConfigError(f"{name} must be a sequence of 3 numbers")
    rgb = (float(value[0]), float(value[1]), float(value[2]))
    if any(v < 0.0 for v in rgb):
        raise StageConfigError(f"{name} components must be >= 0")
    return rgb


def _check_positive(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StageConfigError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise StageConfigError(f"{name} must be a positive finite number")
    return number


def _check_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StageConfigError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise StageConfigError(f"{name} must be finite")
    return number


def _check_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StageConfigError(f"{name} must be an integer")
    if value <= 0:
        raise StageConfigError(f"{name} must be > 0")
    return value


def _check_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise StageConfigError(f"{name} must be a boolean")
    return value


def _check_pattern(name: str, value: object) -> str:
    if value not in _PATTERNS:
        raise StageConfigError(f"{name} must be one of {_PATTERNS}, got {value!r}")
    return str(value)


@dataclass(frozen=True)
class RenderSettings:
    """Demo render resolution, sample count, and positive path-depth limit."""

    width: int = 512
    height: int = 512
    spp: int = 128
    max_depth: int = 8
    denoise: bool = False

    def __post_init__(self) -> None:
        for name in ("width", "height", "spp", "max_depth"):
            _check_positive_int(f"render.{name}", getattr(self, name))
        _check_bool("render.denoise", self.denoise)


@dataclass(frozen=True)
class BackdropSettings:
    """Vertical plane behind the object, facing the camera direction."""

    enabled: bool = True
    pattern: str = "checker"
    distance_factor: float = 2.2
    scale_factor: float = 2.6
    checker_scale: int = 8
    color0: RGB = (0.02, 0.09, 0.11)
    color1: RGB = (0.85, 0.5, 0.12)

    def __post_init__(self) -> None:
        _check_bool("backdrop.enabled", self.enabled)
        _check_pattern("backdrop.pattern", self.pattern)
        _check_positive("backdrop.distance_factor", self.distance_factor)
        _check_positive("backdrop.scale_factor", self.scale_factor)
        _check_positive_int("backdrop.checker_scale", self.checker_scale)
        object.__setattr__(self, "color0", _check_rgb("backdrop.color0", self.color0))
        object.__setattr__(self, "color1", _check_rgb("backdrop.color1", self.color1))


@dataclass(frozen=True)
class FloorSettings:
    """Horizontal plane below the object's lower world-space bound."""

    enabled: bool = True
    pattern: str = "checker"
    drop_factor: float = 0.1
    scale_factor: float = 6.0
    checker_scale: int = 8
    color0: RGB = (0.03, 0.03, 0.13)
    color1: RGB = (0.82, 0.76, 0.14)

    def __post_init__(self) -> None:
        _check_bool("floor.enabled", self.enabled)
        _check_pattern("floor.pattern", self.pattern)
        if _check_finite("floor.drop_factor", self.drop_factor) < 0.0:
            raise StageConfigError("floor.drop_factor must be >= 0")
        _check_positive("floor.scale_factor", self.scale_factor)
        _check_positive_int("floor.checker_scale", self.checker_scale)
        object.__setattr__(self, "color0", _check_rgb("floor.color0", self.color0))
        object.__setattr__(self, "color1", _check_rgb("floor.color1", self.color1))


@dataclass(frozen=True)
class KeyLightSettings:
    """Oblique area light, distinct from the canonical straight-behind backlight."""

    enabled: bool = True
    direction: RGB = (-1.0, -1.5, 2.1)
    distance_factor: float = 3.5
    scale_factor: float = 1.0
    radiance: RGB = (6.4, 5.6, 4.2)

    def __post_init__(self) -> None:
        _check_bool("key_light.enabled", self.enabled)
        if (
            not isinstance(self.direction, (tuple, list))
            or len(self.direction) != 3
            or any(
                isinstance(v, bool) or not isinstance(v, (int, float))
                for v in self.direction
            )
        ):
            raise StageConfigError(
                "key_light.direction must be a sequence of 3 numbers"
            )
        direction = tuple(float(v) for v in self.direction)
        if not all(math.isfinite(v) for v in direction) or not any(direction):
            raise StageConfigError("key_light.direction must be finite and non-zero")
        object.__setattr__(self, "direction", direction)
        _check_positive("key_light.distance_factor", self.distance_factor)
        _check_positive("key_light.scale_factor", self.scale_factor)
        object.__setattr__(
            self, "radiance", _check_rgb("key_light.radiance", self.radiance)
        )


@dataclass(frozen=True)
class CameraOverride:
    """Replacement for the canonical sensor. Only built when explicitly given."""

    azimuth_deg: float = -54.0
    elevation_deg: float = 27.2
    distance_factor: float = 5.07
    fov_deg: float = 35.0

    def __post_init__(self) -> None:
        _check_finite("camera.azimuth_deg", self.azimuth_deg)
        elevation = _check_finite("camera.elevation_deg", self.elevation_deg)
        if not -90.0 < elevation < 90.0:
            raise StageConfigError("camera.elevation_deg must be in (-90, 90)")
        _check_positive("camera.distance_factor", self.distance_factor)
        fov = _check_positive("camera.fov_deg", self.fov_deg)
        if fov >= 180.0:
            raise StageConfigError("camera.fov_deg must be < 180")


@dataclass(frozen=True)
class BacklightOverride:
    """Replacement radiance for the canonical backlight rectangle."""

    radiance: RGB = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "radiance", _check_rgb("backlight.radiance", self.radiance)
        )


@dataclass(frozen=True)
class StageConfig:
    """Full stage description; every default equals the pre-refactor hardcode."""

    render: RenderSettings = field(default_factory=RenderSettings)
    backdrop: BackdropSettings = field(default_factory=BackdropSettings)
    floor: FloorSettings = field(default_factory=FloorSettings)
    key_light: KeyLightSettings = field(default_factory=KeyLightSettings)
    camera: CameraOverride | None = None
    backlight: BacklightOverride | None = None

    def with_cli_overrides(
        self,
        width: int | None = None,
        height: int | None = None,
        spp: int | None = None,
        max_depth: int | None = None,
        checker_scale: int | None = None,
    ) -> StageConfig:
        """Apply explicit CLI arguments on top of this config (CLI wins)."""
        config = self
        render_updates = {
            name: value
            for name, value in (
                ("width", width),
                ("height", height),
                ("spp", spp),
                ("max_depth", max_depth),
            )
            if value is not None
        }
        if render_updates:
            config = replace(config, render=replace(config.render, **render_updates))
        if checker_scale is not None:
            config = replace(
                config,
                backdrop=replace(config.backdrop, checker_scale=checker_scale),
                floor=replace(config.floor, checker_scale=checker_scale),
            )
        return config


_SECTION_TYPES: dict[str, type] = {
    "render": RenderSettings,
    "backdrop": BackdropSettings,
    "floor": FloorSettings,
    "key_light": KeyLightSettings,
}
_RENDER_ALLOWED_FIELDS_BY_VERSION: dict[str, set[str]] = {
    "1.0.0": {"width", "height", "spp"},
    "1.1.0": {"width", "height", "spp", "max_depth"},
}
_OVERRIDE_TYPES: dict[str, type] = {
    "camera": CameraOverride,
    "backlight": BacklightOverride,
}


def _section_from_dict(
    cls: type,
    data: object,
    section: str,
    *,
    allowed_fields: set[str] | None = None,
) -> Any:
    if not isinstance(data, dict):
        raise StageConfigError(f"section {section!r} must be an object")
    known = (
        allowed_fields if allowed_fields is not None else {f.name for f in fields(cls)}
    )
    unknown = set(data) - known
    if unknown:
        raise StageConfigError(
            f"section {section!r} has unknown keys: {sorted(unknown)}"
        )
    kwargs = {
        name: tuple(value) if isinstance(value, list) else value
        for name, value in data.items()
    }
    return cls(**kwargs)


def stage_config_from_dict(document: object) -> StageConfig:
    """Parse and validate a stage-config JSON document (partial spec allowed)."""
    if not isinstance(document, dict):
        raise StageConfigError("stage config must be a JSON object")
    if document.get("format") != STAGE_CONFIG_FORMAT:
        raise StageConfigError(
            f"format must be {STAGE_CONFIG_FORMAT!r}, got {document.get('format')!r}"
        )
    version = document.get("format_version")
    if version not in STAGE_CONFIG_ACCEPTED_VERSIONS:
        raise StageConfigError(
            "format_version must be one of "
            f"{sorted(STAGE_CONFIG_ACCEPTED_VERSIONS)!r}, got {version!r}"
        )
    known = {"format", "format_version", *_SECTION_TYPES, *_OVERRIDE_TYPES}
    unknown = set(document) - known
    if unknown:
        raise StageConfigError(f"unknown top-level keys: {sorted(unknown)}")

    kwargs: dict[str, Any] = {}
    for section, cls in _SECTION_TYPES.items():
        if section in document:
            allowed_fields = None
            if section == "render":
                allowed_fields = _RENDER_ALLOWED_FIELDS_BY_VERSION.get(version)
            kwargs[section] = _section_from_dict(
                cls,
                document[section],
                section,
                allowed_fields=allowed_fields,
            )
    for section, cls in _OVERRIDE_TYPES.items():
        if section in document and document[section] is not None:
            kwargs[section] = _section_from_dict(cls, document[section], section)
    return StageConfig(**kwargs)


def stage_config_from_json(path: Path) -> StageConfig:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise StageConfigError(f"{path} is not valid JSON: {error}") from error
    return stage_config_from_dict(document)


def stage_config_to_dict(config: StageConfig) -> dict[str, Any]:
    """Serialize a StageConfig to a JSON-ready document (all fields explicit)."""

    def section(value: Any) -> dict[str, Any]:
        return {
            f.name: list(v) if isinstance(v := getattr(value, f.name), tuple) else v
            for f in fields(value)
        }

    return {
        "format": STAGE_CONFIG_FORMAT,
        "format_version": STAGE_CONFIG_FORMAT_VERSION,
        "render": section(config.render),
        "backdrop": section(config.backdrop),
        "floor": section(config.floor),
        "key_light": section(config.key_light),
        "camera": None if config.camera is None else section(config.camera),
        "backlight": None if config.backlight is None else section(config.backlight),
    }


@dataclass(frozen=True)
class StageBounds:
    """World-space frame of the volume, matching the exporter's private frame."""

    minimum: np.ndarray
    maximum: np.ndarray
    center: np.ndarray
    radius: float
    camera_direction: np.ndarray


def scene_bounds(geometry: GridGeometry) -> StageBounds:
    """Recompute the same center/radius/camera_direction frame as the exporter.

    Reimplemented here (rather than importing the exporter's private
    ``_scene_frame``) per the demo-track boundary: this module only reads the
    public ``GridGeometry`` API, never private exporter helpers.
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
    return StageBounds(minimum, maximum, center, radius, camera_direction)


def _plane_bsdf(
    mi: ModuleType,
    pattern: str,
    checker_scale: int,
    color0: RGB,
    color1: RGB,
) -> dict[str, object]:
    if pattern == "solid":
        return {
            "type": "diffuse",
            "reflectance": {"type": "rgb", "value": list(color0)},
        }
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


def _sensor_override_dict(
    mi: ModuleType, bounds: StageBounds, config: StageConfig
) -> dict[str, object]:
    camera = config.camera
    assert camera is not None
    azimuth = math.radians(camera.azimuth_deg)
    elevation = math.radians(camera.elevation_deg)
    direction = np.asarray(
        (
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ),
        dtype=np.float64,
    )
    origin = bounds.center + direction * bounds.radius * camera.distance_factor
    return {
        "type": "perspective",
        "fov": camera.fov_deg,
        "near_clip": max(bounds.radius * 0.001, 1e-9),
        "far_clip": max(bounds.radius * 20.0, 1e-6),
        "to_world": mi.ScalarTransform4f.look_at(
            origin=origin.tolist(), target=bounds.center.tolist(), up=[0.0, 0.0, 1.0]
        ),
        "sampler": {"type": "independent", "sample_count": config.render.spp},
        "film": {
            "type": "hdrfilm",
            "width": config.render.width,
            "height": config.render.height,
            "pixel_format": "rgb",
            "component_format": "float32",
            "rfilter": {"type": "box"},
        },
    }


def apply_stage(
    mi: ModuleType,
    scene: dict[str, object],
    geometry: GridGeometry,
    config: StageConfig,
) -> None:
    """Add the configured stage to ``scene`` (a scene_dict copy), in place.

    Additive by default: the medium, exterior/interior meshes, sensor, and
    backlight entries already in ``scene`` (from ``prepare_mitsuba_scene``) are
    untouched. Only an explicit non-null ``camera`` / ``backlight`` section
    replaces the corresponding canonical entry — and only on this copy, never
    in the exporter.
    """
    bounds = scene_bounds(geometry)
    center, radius = bounds.center, bounds.radius

    if config.backdrop.enabled:
        # Backdrop: a diffuse wall behind the object, nearer to the object
        # than the canonical white backlight rectangle (which sits at
        # radius * 4 and already fills the camera frame at that distance, so
        # anything placed further back would be fully hidden behind it).
        backdrop = config.backdrop
        backdrop_position = (
            center - bounds.camera_direction * radius * backdrop.distance_factor
        )
        scene["stage_backdrop"] = {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.look_at(
                origin=backdrop_position.tolist(),
                target=center.tolist(),
                up=[0.0, 0.0, 1.0],
            )
            @ mi.ScalarTransform4f.scale(
                [radius * backdrop.scale_factor, radius * backdrop.scale_factor, 1.0]
            ),
            # Default teal/orange: distinct in hue (not just value) from the
            # floor below, so a viewer can tell which surface is being seen
            # through a refracted/distorted patch.
            "bsdf": _plane_bsdf(
                mi,
                backdrop.pattern,
                backdrop.checker_scale,
                backdrop.color0,
                backdrop.color1,
            ),
        }

    if config.floor.enabled:
        # Floor: the default rectangle normal (0, 0, 1) already faces up,
        # matching the up=[0, 0, 1] convention used by the sensor/backlight.
        floor = config.floor
        floor_z = bounds.minimum[2] - radius * floor.drop_factor
        scene["stage_floor"] = {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.translate(
                [float(center[0]), float(center[1]), float(floor_z)]
            )
            @ mi.ScalarTransform4f.scale(
                [radius * floor.scale_factor, radius * floor.scale_factor, 1.0]
            ),
            "bsdf": _plane_bsdf(
                mi, floor.pattern, floor.checker_scale, floor.color0, floor.color1
            ),
        }

    if config.key_light.enabled:
        # Key light: an area light from an oblique angle (distinct from the
        # camera and from the backlight's straight-behind position) so the
        # stage and object read with visible shading/shadow, not just a
        # backlit silhouette. Default is slightly warm rather than pure white.
        key = config.key_light
        key_direction = np.asarray(key.direction, dtype=np.float64)
        key_direction /= np.linalg.norm(key_direction)
        key_position = center + key_direction * radius * key.distance_factor
        scene["stage_key_light"] = {
            "type": "rectangle",
            "to_world": mi.ScalarTransform4f.look_at(
                origin=key_position.tolist(),
                target=center.tolist(),
                up=[0.0, 0.0, 1.0],
            )
            @ mi.ScalarTransform4f.scale(
                [radius * key.scale_factor, radius * key.scale_factor, 1.0]
            ),
            "emitter": {
                "type": "area",
                "radiance": {"type": "rgb", "value": list(key.radiance)},
            },
        }

    if config.camera is not None:
        scene["sensor"] = _sensor_override_dict(mi, bounds, config)

    if config.backlight is not None:
        canonical = scene["backlight"]
        if not isinstance(canonical, dict):
            raise StageConfigError("scene has no canonical backlight dict to override")
        # Replace via a copy so the prepared scene_dict's nested entries are
        # never mutated through the shallow copy the caller passed in.
        scene["backlight"] = {
            **canonical,
            "emitter": {
                "type": "area",
                "radiance": {"type": "rgb", "value": list(config.backlight.radiance)},
            },
        }
