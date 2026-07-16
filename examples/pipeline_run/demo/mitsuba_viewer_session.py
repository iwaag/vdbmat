"""Versioned, portable session manifests for the Mitsuba stage viewer.

A viewer session records the exact optical input, effective non-render stage,
render settings, Mitsuba variant, and seed needed to replay one viewer state.
All filesystem references are POSIX paths relative to explicit server-local
roots and are paired with deterministic digests.

This module has no dependency on Mitsuba or viser.  It owns strict JSON
parsing, serialization, atomic publication, path containment, and digest
verification so the browser viewer and headless renderer can share one replay
contract.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

from mitsuba_stage import (
    RenderSettings,
    StageConfig,
    StageConfigError,
    stage_config_from_dict,
    stage_config_to_dict,
)
from mitsuba_stage_inputs import (
    InputCandidate,
    InputCatalogError,
    InputKind,
    resolve_candidate,
)
from mitsuba_stage_presets import (
    PresetCandidate,
    PresetCatalogError,
    load_preset,
    resolve_preset,
    stage_config_digest,
)

from vdbmat.pipeline import sha256_file, zarr_store_sha256

VIEWER_SESSION_FORMAT = "vdbmat.viewer-session"
VIEWER_SESSION_FORMAT_VERSION = "1.0.0"

_RUN_MANIFEST_NAME = "run.json"
_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_VARIANTS = frozenset({"llvm_ad_rgb", "cuda_ad_rgb"})
_TOP_LEVEL_KEYS = frozenset(
    {"format", "format_version", "input", "stage", "render", "mitsuba"}
)
_EFFECTIVE_STAGE_KEYS = frozenset(
    {
        "format",
        "format_version",
        "backdrop",
        "floor",
        "key_light",
        "camera",
        "backlight",
    }
)
_RENDER_KEYS = frozenset({"width", "height", "spp", "max_depth"})
_EFFECTIVE_SECTION_KEYS = {
    "backdrop": frozenset(
        {
            "enabled",
            "pattern",
            "distance_factor",
            "scale_factor",
            "checker_scale",
            "color0",
            "color1",
        }
    ),
    "floor": frozenset(
        {
            "enabled",
            "pattern",
            "drop_factor",
            "scale_factor",
            "checker_scale",
            "color0",
            "color1",
        }
    ),
    "key_light": frozenset(
        {
            "enabled",
            "direction",
            "distance_factor",
            "scale_factor",
            "radiance",
        }
    ),
}
_EFFECTIVE_OVERRIDE_KEYS = {
    "camera": frozenset({"azimuth_deg", "elevation_deg", "distance_factor", "fov_deg"}),
    "backlight": frozenset({"radiance"}),
}


class ViewerSessionError(Exception):
    """Session parsing or resolution failed at a named, user-visible stage."""

    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        self.message = message
        super().__init__(f"viewer session failed at {stage}: {message}")


@dataclass(frozen=True, slots=True)
class SessionInputRef:
    """Portable reference to a catalog input and its rendered bytes."""

    kind: InputKind
    path: str
    optical_sha256: str
    run_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        _portable_path(self.path, field="input.path")
        _digest(self.optical_sha256, field="input.optical_sha256")
        if self.kind is InputKind.RUN_BUNDLE:
            if self.run_manifest_sha256 is None:
                raise ValueError("input.run_manifest_sha256 is required for run-bundle")
            _digest(
                self.run_manifest_sha256,
                field="input.run_manifest_sha256",
            )
        elif self.kind is InputKind.OPTICAL_ZARR:
            if self.run_manifest_sha256 is not None:
                raise ValueError(
                    "input.run_manifest_sha256 is not allowed for optical-zarr"
                )
        else:
            raise ValueError(f"input.kind is unsupported: {self.kind!r}")


@dataclass(frozen=True, slots=True)
class SessionPresetRef:
    """Optional provenance link to an applied stage preset."""

    path: str
    digest: str

    def __post_init__(self) -> None:
        _portable_path(self.path, field="stage.preset.path")
        _digest(self.digest, field="stage.preset.digest")


@dataclass(frozen=True, slots=True)
class ViewerSession:
    """Fully validated in-memory representation of a session manifest."""

    input: SessionInputRef
    stage_config: StageConfig
    effective_digest: str
    variant: str
    seed: int
    preset: SessionPresetRef | None = None

    def __post_init__(self) -> None:
        _digest(self.effective_digest, field="stage.effective_digest")
        actual_digest = stage_config_digest(self.stage_config)
        if self.effective_digest != actual_digest:
            raise ValueError(
                "stage.effective_digest mismatch: "
                f"expected {self.effective_digest}, actual {actual_digest}"
            )
        if self.preset is not None and self.preset.digest != self.effective_digest:
            raise ValueError("stage.preset.digest must match stage.effective_digest")
        if self.variant not in _VARIANTS:
            raise ValueError(f"mitsuba.variant must be one of {sorted(_VARIANTS)!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("mitsuba.seed must be an integer")
        if self.seed < 0:
            raise ValueError("mitsuba.seed must be >= 0")


@dataclass(frozen=True, slots=True)
class ResolvedViewerSession:
    """Verified session with concrete paths for renderer consumption."""

    session: ViewerSession
    input_candidate: InputCandidate
    optical_zarr: Path
    stage_config: StageConfig
    variant: str
    seed: int
    preset_candidate: PresetCandidate | None


def create_viewer_session(
    candidate: InputCandidate,
    stage_config: StageConfig,
    variant: str,
    seed: int,
    *,
    preset: SessionPresetRef | None = None,
) -> ViewerSession:
    """Capture one verified catalog candidate as a digest-pinned session."""
    try:
        optical_digest = zarr_store_sha256(candidate.optical_zarr)
        run_digest = (
            sha256_file(candidate.path / _RUN_MANIFEST_NAME)
            if candidate.kind is InputKind.RUN_BUNDLE
            else None
        )
        input_ref = SessionInputRef(
            kind=candidate.kind,
            path=candidate.root_relative,
            optical_sha256=optical_digest,
            run_manifest_sha256=run_digest,
        )
        return ViewerSession(
            input=input_ref,
            stage_config=stage_config,
            effective_digest=stage_config_digest(stage_config),
            variant=variant,
            seed=seed,
            preset=preset,
        )
    except (OSError, TypeError, ValueError) as error:
        raise ViewerSessionError("capture", str(error)) from error


def viewer_session_from_dict(document: object) -> ViewerSession:
    """Strictly parse and validate a viewer-session JSON document."""
    try:
        root = _object(document, field="session")
        _exact_keys(root, _TOP_LEVEL_KEYS, field="session")
        if root["format"] != VIEWER_SESSION_FORMAT:
            raise ValueError(
                f"format must be {VIEWER_SESSION_FORMAT!r}, got {root['format']!r}"
            )
        if root["format_version"] != VIEWER_SESSION_FORMAT_VERSION:
            raise ValueError(
                "format_version must be "
                f"{VIEWER_SESSION_FORMAT_VERSION!r}, "
                f"got {root['format_version']!r}"
            )

        input_ref = _parse_input(root["input"])
        stage_config, effective_digest, preset = _parse_stage_and_render(
            root["stage"], root["render"]
        )
        variant, seed = _parse_mitsuba(root["mitsuba"])
        return ViewerSession(
            input=input_ref,
            stage_config=stage_config,
            effective_digest=effective_digest,
            variant=variant,
            seed=seed,
            preset=preset,
        )
    except ViewerSessionError:
        raise
    except (KeyError, StageConfigError, TypeError, ValueError) as error:
        raise ViewerSessionError("parse", str(error)) from error


def viewer_session_from_json(path: Path) -> ViewerSession:
    """Read a UTF-8 JSON file and parse it as a viewer session."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ViewerSessionError("parse", f"cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ViewerSessionError(
            "parse", f"{path} is not valid JSON: {error}"
        ) from error
    return viewer_session_from_dict(document)


def viewer_session_to_dict(session: ViewerSession) -> dict[str, Any]:
    """Serialize a validated session with all v1 fields explicit."""
    stage_document = stage_config_to_dict(session.stage_config)
    render = stage_document.pop("render")
    input_document: dict[str, Any] = {
        "kind": session.input.kind.value,
        "path": session.input.path,
        "optical_sha256": session.input.optical_sha256,
    }
    if session.input.run_manifest_sha256 is not None:
        input_document["run_manifest_sha256"] = session.input.run_manifest_sha256
    stage: dict[str, Any] = {
        "effective": stage_document,
        "effective_digest": session.effective_digest,
    }
    if session.preset is not None:
        stage["preset"] = {
            "path": session.preset.path,
            "digest": session.preset.digest,
        }
    return {
        "format": VIEWER_SESSION_FORMAT,
        "format_version": VIEWER_SESSION_FORMAT_VERSION,
        "input": input_document,
        "stage": stage,
        "render": render,
        "mitsuba": {"variant": session.variant, "seed": session.seed},
    }


def write_viewer_session(path: Path, session: ViewerSession) -> None:
    """Atomically publish a round-trip-validated viewer-session document."""
    document = viewer_session_to_dict(session)
    viewer_session_from_dict(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def resolve_viewer_session(
    session: ViewerSession,
    input_root: Path,
    preset_root: Path | None = None,
    *,
    on_stage: Callable[[str], None] = lambda _stage: None,
) -> ResolvedViewerSession:
    """Resolve root-relative references and verify every declared digest."""
    on_stage("resolve")
    try:
        candidate = resolve_candidate(input_root, Path(session.input.path))
    except (InputCatalogError, OSError) as error:
        raise ViewerSessionError("resolve", f"input: {error}") from error
    if candidate.kind is not session.input.kind:
        raise ViewerSessionError(
            "resolve",
            "input kind mismatch: "
            f"expected {session.input.kind.value}, actual {candidate.kind.value}",
        )

    on_stage("verify")
    try:
        actual_optical_digest = zarr_store_sha256(candidate.optical_zarr)
    except OSError as error:
        raise ViewerSessionError(
            "verify", f"cannot digest input optical store: {error}"
        ) from error
    _verify_digest(
        "input optical",
        expected=session.input.optical_sha256,
        actual=actual_optical_digest,
    )

    if candidate.kind is InputKind.RUN_BUNDLE:
        try:
            actual_run_digest = sha256_file(candidate.path / _RUN_MANIFEST_NAME)
        except OSError as error:
            raise ViewerSessionError(
                "verify", f"cannot digest input run manifest: {error}"
            ) from error
        assert session.input.run_manifest_sha256 is not None
        _verify_digest(
            "input run manifest",
            expected=session.input.run_manifest_sha256,
            actual=actual_run_digest,
        )

    preset_candidate = _resolve_session_preset(session.preset, preset_root)
    return ResolvedViewerSession(
        session=session,
        input_candidate=candidate,
        optical_zarr=candidate.optical_zarr,
        stage_config=session.stage_config,
        variant=session.variant,
        seed=session.seed,
        preset_candidate=preset_candidate,
    )


def _parse_input(value: object) -> SessionInputRef:
    document = _object(value, field="input")
    kind_value = document.get("kind")
    try:
        kind = InputKind(kind_value)
    except (TypeError, ValueError) as error:
        raise ValueError("input.kind must be 'run-bundle' or 'optical-zarr'") from error
    required = {"kind", "path", "optical_sha256"}
    if kind is InputKind.RUN_BUNDLE:
        required.add("run_manifest_sha256")
    _exact_keys(document, frozenset(required), field="input")
    return SessionInputRef(
        kind=kind,
        path=_string(document["path"], field="input.path"),
        optical_sha256=_string(
            document["optical_sha256"], field="input.optical_sha256"
        ),
        run_manifest_sha256=(
            _string(
                document["run_manifest_sha256"],
                field="input.run_manifest_sha256",
            )
            if kind is InputKind.RUN_BUNDLE
            else None
        ),
    )


def _parse_stage_and_render(
    stage_value: object, render_value: object
) -> tuple[StageConfig, str, SessionPresetRef | None]:
    stage = _object(stage_value, field="stage")
    allowed_stage_keys = frozenset({"effective", "effective_digest", "preset"})
    required_stage_keys = frozenset({"effective", "effective_digest"})
    _required_and_allowed_keys(
        stage,
        required=required_stage_keys,
        allowed=allowed_stage_keys,
        field="stage",
    )
    effective = _object(stage["effective"], field="stage.effective")
    _exact_keys(effective, _EFFECTIVE_STAGE_KEYS, field="stage.effective")
    for section, expected_keys in _EFFECTIVE_SECTION_KEYS.items():
        section_document = _object(
            effective[section], field=f"stage.effective.{section}"
        )
        _exact_keys(
            section_document,
            expected_keys,
            field=f"stage.effective.{section}",
        )
    for section, expected_keys in _EFFECTIVE_OVERRIDE_KEYS.items():
        section_value = effective[section]
        if section_value is not None:
            section_document = _object(
                section_value, field=f"stage.effective.{section}"
            )
            _exact_keys(
                section_document,
                expected_keys,
                field=f"stage.effective.{section}",
            )

    render = _object(render_value, field="render")
    _exact_keys(render, _RENDER_KEYS, field="render")
    render_settings = RenderSettings(
        width=render["width"],
        height=render["height"],
        spp=render["spp"],
        max_depth=render["max_depth"],
    )
    config_without_render = stage_config_from_dict(effective)
    config = replace(config_without_render, render=render_settings)
    effective_digest = _string(
        stage["effective_digest"], field="stage.effective_digest"
    )
    preset = _parse_preset(stage["preset"]) if "preset" in stage else None
    return config, effective_digest, preset


def _parse_preset(value: object) -> SessionPresetRef:
    document = _object(value, field="stage.preset")
    _exact_keys(document, frozenset({"path", "digest"}), field="stage.preset")
    return SessionPresetRef(
        path=_string(document["path"], field="stage.preset.path"),
        digest=_string(document["digest"], field="stage.preset.digest"),
    )


def _parse_mitsuba(value: object) -> tuple[str, int]:
    document = _object(value, field="mitsuba")
    _exact_keys(document, frozenset({"variant", "seed"}), field="mitsuba")
    variant = _string(document["variant"], field="mitsuba.variant")
    seed = document["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("mitsuba.seed must be an integer")
    return variant, seed


def _resolve_session_preset(
    reference: SessionPresetRef | None, preset_root: Path | None
) -> PresetCandidate | None:
    if reference is None:
        return None
    if preset_root is None:
        raise ViewerSessionError(
            "resolve", "stage preset reference requires --preset-root"
        )
    try:
        candidate = resolve_preset(preset_root, Path(reference.path))
        preset_config = load_preset(candidate)
    except (PresetCatalogError, OSError) as error:
        raise ViewerSessionError("resolve", f"stage preset: {error}") from error
    _verify_digest(
        "stage preset",
        expected=reference.digest,
        actual=stage_config_digest(preset_config),
    )
    return candidate


def _verify_digest(label: str, *, expected: str, actual: str) -> None:
    if expected != actual:
        raise ViewerSessionError(
            "verify",
            f"{label} digest mismatch: expected {expected}, actual {actual}",
        )


def _portable_path(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if not value or value == ".":
        raise ValueError(f"{field} must not be empty")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain NUL")
    if "\\" in value:
        raise ValueError(f"{field} must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute():
        raise ValueError(f"{field} must be relative")
    if ".." in path.parts:
        raise ValueError(f"{field} must not contain parent traversal")
    if path.as_posix() != value:
        raise ValueError(f"{field} must be a normalized POSIX path")
    return value


def _digest(value: str, *, field: str) -> str:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be sha256:<64 lowercase hex>")
    return value


def _object(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be an object")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    return value


def _exact_keys(
    document: dict[str, Any], expected: frozenset[str], *, field: str
) -> None:
    _required_and_allowed_keys(
        document, required=expected, allowed=expected, field=field
    )


def _required_and_allowed_keys(
    document: dict[str, Any],
    *,
    required: frozenset[str],
    allowed: frozenset[str],
    field: str,
) -> None:
    missing = required - set(document)
    if missing:
        raise ValueError(f"{field} is missing keys: {sorted(missing)}")
    unknown = set(document) - allowed
    if unknown:
        raise ValueError(f"{field} has unknown keys: {sorted(unknown)}")


__all__ = [
    "VIEWER_SESSION_FORMAT",
    "VIEWER_SESSION_FORMAT_VERSION",
    "ResolvedViewerSession",
    "SessionInputRef",
    "SessionPresetRef",
    "ViewerSession",
    "ViewerSessionError",
    "create_viewer_session",
    "resolve_viewer_session",
    "viewer_session_from_dict",
    "viewer_session_from_json",
    "viewer_session_to_dict",
    "write_viewer_session",
]
