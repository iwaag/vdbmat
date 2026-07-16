"""Stage-preset catalog for the Mitsuba stage viewer.

The catalog is rooted at a server-local ``--preset-root`` and exposes only
``*.stage.json`` files whose resolved targets stay inside that root.  Catalog
enumeration is intentionally cheap: JSON parsing and StageConfig validation
are deferred until a selected preset is described or loaded.

This module imports neither Mitsuba nor viser.  It can therefore define and
test the preset-selection, containment, and digest contracts independently of
the GUI and renderer that consume them.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mitsuba_stage import (
    StageConfig,
    StageConfigError,
    stage_config_from_dict,
    stage_config_to_dict,
)

_PRESET_SUFFIX = ".stage.json"
_BUILTIN_PRESET_DIR = Path(__file__).resolve().parent / "presets"


class PresetCatalogError(Exception):
    """A preset root, candidate, or document is invalid."""


@dataclass(frozen=True, slots=True)
class PresetCandidate:
    """One contained ``*.stage.json`` file in the preset catalog."""

    root_relative: str
    path: Path


@dataclass(frozen=True, slots=True)
class PresetSummary:
    """Display-only summary of a parsed and validated stage preset."""

    format_version: str
    width: int
    height: int
    spp: int
    max_depth: int
    camera_override: bool
    backlight_override: bool
    digest: str


def resolve_preset_root(
    cli_root: Path | None, initial_stage_config: Path | None
) -> Path:
    """Resolve the explicit root, initial preset parent, or builtin presets.

    An explicit ``--preset-root`` wins.  Otherwise an initial
    ``--stage-config`` makes its parent the browsing root.  A viewer started
    without either option browses the checked-in demo preset directory.
    """
    if cli_root is not None:
        return _require_directory(cli_root, option="--preset-root")
    if initial_stage_config is not None:
        return _require_directory(
            initial_stage_config.resolve().parent,
            option="initial stage-config parent",
        )
    return _require_directory(_BUILTIN_PRESET_DIR, option="builtin preset root")


def scan_preset_catalog(root: Path) -> list[PresetCandidate]:
    """Return contained preset files in deterministic relative-path order.

    A file only needs the expected suffix to appear in the catalog.  Invalid
    JSON and invalid stage-config documents remain visible so describing or
    applying the selected entry can report the actual error to the user.
    """
    resolved_root = _require_directory(root, option="--preset-root")
    found: list[PresetCandidate] = []
    for dirpath, dirnames, filenames in os.walk(resolved_root):
        current = Path(dirpath)
        dirnames[:] = sorted(
            name
            for name in dirnames
            if _is_contained(resolved_root, current / name)
        )
        for name in sorted(filenames):
            if not name.endswith(_PRESET_SUFFIX):
                continue
            lexical_path = current / name
            if not _is_contained(resolved_root, lexical_path):
                continue
            resolved_path = lexical_path.resolve()
            if not resolved_path.is_file():
                continue
            found.append(
                PresetCandidate(
                    root_relative=lexical_path.relative_to(resolved_root).as_posix(),
                    path=resolved_path,
                )
            )
    found.sort(key=lambda item: item.root_relative)
    return found


def resolve_preset(root: Path, user_path: Path) -> PresetCandidate:
    """Resolve one root-relative preset under strict lexical containment.

    GUI selections are catalog keys, not arbitrary server paths.  Absolute
    paths and parent traversal are rejected even when they would happen to
    resolve inside the root; symlinks are accepted only when their final
    target remains inside the resolved root.
    """
    resolved_root = _require_directory(root, option="--preset-root")
    if user_path.is_absolute():
        raise PresetCatalogError(f"preset path must be relative: {user_path}")
    if not user_path.parts or user_path == Path("."):
        raise PresetCatalogError("preset path must not be empty")
    if ".." in user_path.parts:
        raise PresetCatalogError(
            f"preset path must not contain parent traversal: {user_path}"
        )
    if not user_path.name.endswith(_PRESET_SUFFIX):
        raise PresetCatalogError(
            f"preset path must end with {_PRESET_SUFFIX!r}: {user_path}"
        )

    lexical_path = resolved_root / user_path
    if not lexical_path.exists():
        raise PresetCatalogError(f"preset does not exist: {user_path}")
    try:
        resolved_path = lexical_path.resolve()
    except OSError as error:
        raise PresetCatalogError(
            f"cannot resolve preset {user_path}: {error}"
        ) from error
    if not _is_contained(resolved_root, resolved_path):
        raise PresetCatalogError(
            f"preset resolves outside --preset-root: {user_path}"
        )
    if not resolved_path.is_file():
        raise PresetCatalogError(f"preset is not a file: {user_path}")
    return PresetCandidate(root_relative=user_path.as_posix(), path=resolved_path)


def load_preset(candidate: PresetCandidate) -> StageConfig:
    """Parse and validate one selected stage preset."""
    document = _read_document(candidate)
    try:
        return stage_config_from_dict(document)
    except (StageConfigError, TypeError, ValueError) as error:
        raise PresetCatalogError(
            f"invalid stage preset {candidate.root_relative}: {error}"
        ) from error


def stage_config_digest(config: StageConfig) -> str:
    """Return a semantic digest of a fully materialized StageConfig.

    Serialization normalizes legacy and partial documents to the current
    all-fields-explicit stage-config form.  Whitespace, source key order, and
    omitted default fields therefore do not affect the digest.
    """
    normalized = json.dumps(
        stage_config_to_dict(config),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(normalized).hexdigest()}"


def describe_preset(candidate: PresetCandidate) -> PresetSummary:
    """Parse a preset and return the fields needed by a GUI summary."""
    document = _read_document(candidate)
    try:
        config = stage_config_from_dict(document)
    except (StageConfigError, TypeError, ValueError) as error:
        raise PresetCatalogError(
            f"invalid stage preset {candidate.root_relative}: {error}"
        ) from error
    version = document.get("format_version")
    assert isinstance(version, str)  # guaranteed by stage_config_from_dict()
    return PresetSummary(
        format_version=version,
        width=config.render.width,
        height=config.render.height,
        spp=config.render.spp,
        max_depth=config.render.max_depth,
        camera_override=config.camera is not None,
        backlight_override=config.backlight is not None,
        digest=stage_config_digest(config),
    )


def _read_document(candidate: PresetCandidate) -> dict[str, Any]:
    try:
        document = json.loads(candidate.path.read_text(encoding="utf-8"))
    except OSError as error:
        raise PresetCatalogError(
            f"cannot read stage preset {candidate.root_relative}: {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise PresetCatalogError(
            f"invalid stage preset {candidate.root_relative}: not valid JSON: {error}"
        ) from error
    if not isinstance(document, dict):
        raise PresetCatalogError(
            f"invalid stage preset {candidate.root_relative}: must be a JSON object"
        )
    return document


def _require_directory(path: Path, *, option: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise PresetCatalogError(f"{option} is not a directory: {path}")
    return resolved


def _is_contained(root: Path, path: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return resolved == root or resolved.is_relative_to(root)


__all__ = [
    "PresetCandidate",
    "PresetCatalogError",
    "PresetSummary",
    "describe_preset",
    "load_preset",
    "resolve_preset",
    "resolve_preset_root",
    "scan_preset_catalog",
    "stage_config_digest",
]
