"""Immutable, digestible Phase 1 pipeline configuration (ADR-007 / ADR-008 / ADR-009).

The configuration is the single, portable declaration of *what* a pipeline run does:
which voxel manifest to read, which optical mapping to apply, which validation and
export stages to run, and where to publish the run bundle. It is a pure data object —
building it performs no I/O and creates no output — so an invalid combination fails
before anything is written (plan Step 5).

Per ADR-009 D1, the only supported input is the ``vdbmat.voxels`` direct-voxel
manifest; the core owns no geometry-to-voxel conversion.

Two canonicalizations are exposed:

* :meth:`PipelineConfig.canonical_json` / :attr:`PipelineConfig.digest` identify the
  *whole* run configuration and back the ADR-007 D2 ``config_digest``.
* :meth:`PipelineConfig.scientific_canonical_json` /
  :attr:`PipelineConfig.scientific_digest` identify only the portion that determines
  the canonical ``material.zarr`` / ``optical.zarr`` volumes. Renderer and export
  settings are excluded, so they provably cannot alter canonical results.
"""

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from numbers import Integral
from pathlib import Path
from typing import Any

from vdbmat.core import SchemaIdentity, SchemaVersion
from vdbmat.optics import (
    OpticalMappingConfig,
    OpticalMappingError,
    load_optical_mapping,
    phase0_provisional_mapping,
)

from .errors import PipelineConfigError

PIPELINE_CONFIG_SCHEMA = SchemaIdentity(
    name="vdbmat.pipeline-config", version=SchemaVersion(2, 0, 0)
)

#: Builtin optical mappings referenced by name from a configuration.
_BUILTIN_MAPPINGS: Mapping[str, Callable[[], OpticalMappingConfig]] = {
    "phase0-provisional-materials-v1": phase0_provisional_mapping,
}
#: Default optical mapping (ADR-008 D1): the Phase 0 provisional coefficients.
DEFAULT_MAPPING_NAME = "phase0-provisional-materials-v1"


class InputKind(StrEnum):
    """The single supported input path (ADR-009 D1).

    The enum is retained as the explicit extension point should a second stable
    input contract ever be adopted; external generators emit voxel manifests
    rather than adding members here.
    """

    DIRECT_VOXEL = "direct-voxel"


class ExportTarget(StrEnum):
    """Optional renderer export targets (ADR-007/ADR-008)."""

    MITSUBA = "mitsuba"
    OPENVDB = "openvdb"


@dataclass(frozen=True, slots=True)
class ExportSettings:
    """One requested optional renderer export stage.

    Export settings never enter a canonical stage and never influence
    :attr:`PipelineConfig.scientific_digest`; a renderer export consumes the restored
    ``optical.zarr`` only (ADR-007 D1).
    """

    target: ExportTarget

    def __post_init__(self) -> None:
        try:
            target = ExportTarget(self.target)
        except ValueError as error:
            raise PipelineConfigError(
                "stages.exports[].target",
                f"unsupported export target: {self.target!r}",
            ) from error
        object.__setattr__(self, "target", target)

    def to_json_dict(self) -> dict[str, Any]:
        """Return the portable JSON form of this export request."""
        return {"target": self.target.value}

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> "ExportSettings":
        """Reconstruct one export request from its portable JSON form."""
        _require_mapping("stages.exports[]", data)
        _reject_unknown_keys("stages.exports[]", data, {"target"})
        _require_key("stages.exports[]", data, "target")
        return cls(target=data["target"])


@dataclass(frozen=True, slots=True)
class RendererConfig:
    """Opaque references to external renderer scene material.

    These are recorded for provenance and consumed only by the optional export/render
    stages. They are deliberately kept outside the scientific digest so renderer
    configuration cannot alter canonical material or optical results (plan Step 5).
    """

    references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.references, str):
            raise PipelineConfigError(
                "renderer.references", "must be a sequence of strings, not a string"
            )
        references = tuple(self.references)
        for index, reference in enumerate(references):
            if not isinstance(reference, str) or not reference.strip():
                raise PipelineConfigError(
                    f"renderer.references[{index}]", "must be a non-empty string"
                )
        object.__setattr__(self, "references", references)

    def to_json_dict(self) -> dict[str, Any]:
        """Return the portable JSON form of this renderer reference set."""
        return {"references": list(self.references)}

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> "RendererConfig":
        """Reconstruct a renderer reference set from its portable JSON form."""
        _require_mapping("renderer", data)
        _reject_unknown_keys("renderer", data, {"references"})
        return cls(references=tuple(data.get("references", ())))


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """A versioned, immutable Phase 1 pipeline configuration.

    Paths are retained exactly as declared (portable, typically relative);
    resolution to an absolute execution path is an explicit, separate step
    (:meth:`resolve_input_path` / :meth:`resolve_output_path`), so the configuration
    carries no implicit current-directory behaviour.
    """

    input_kind: InputKind
    input_path: str
    output_path: str
    mapping_name: str | None = None
    mapping_path: str | None = None
    mapping_digest: str | None = None
    validate_material: bool = True
    validate_optical: bool = True
    exports: tuple[ExportSettings, ...] = ()
    overwrite: bool = False
    random_seed: int = 0
    renderer: RendererConfig | None = None

    def __post_init__(self) -> None:
        try:
            input_kind = InputKind(self.input_kind)
        except ValueError as error:
            raise PipelineConfigError(
                "input.kind", f"unsupported input kind: {self.input_kind!r}"
            ) from error
        object.__setattr__(self, "input_kind", input_kind)

        object.__setattr__(
            self, "input_path", _normalize_path("input.path", self.input_path)
        )
        object.__setattr__(
            self, "output_path", _normalize_path("output.path", self.output_path)
        )

        # Mapping reference (ADR-009 D3): a builtin name or an external document path,
        # never both. A file-based mapping carries its declared digest so canonical
        # results stay a pure function of the configuration; a name-based mapping's
        # digest is computed here (and checked when one was recorded).
        if self.mapping_name is not None and self.mapping_path is not None:
            raise PipelineConfigError(
                "mapping",
                "declare exactly one of mapping.name or mapping.path, not both",
            )
        if self.mapping_name is None and self.mapping_path is None:
            object.__setattr__(self, "mapping_name", DEFAULT_MAPPING_NAME)

        if self.mapping_name is not None:
            if self.mapping_name not in _BUILTIN_MAPPINGS:
                raise PipelineConfigError(
                    "mapping.name",
                    f"unknown mapping; must be one of {sorted(_BUILTIN_MAPPINGS)}, "
                    f"got {self.mapping_name!r}",
                )
            builtin_digest = _BUILTIN_MAPPINGS[self.mapping_name]().digest
            if self.mapping_digest is not None and (
                self.mapping_digest != builtin_digest
            ):
                raise PipelineConfigError(
                    "mapping.digest",
                    f"recorded digest {self.mapping_digest!r} does not match "
                    f"mapping {self.mapping_name!r} ({builtin_digest})",
                )
            object.__setattr__(self, "mapping_digest", builtin_digest)
        else:
            object.__setattr__(
                self,
                "mapping_path",
                _normalize_path("mapping.path", self.mapping_path),
            )
            _require_digest_format("mapping.digest", self.mapping_digest)

        for name in ("validate_material", "validate_optical", "overwrite"):
            if not isinstance(getattr(self, name), bool):
                raise PipelineConfigError(name, "must be a boolean")

        exports = tuple(self.exports)
        for item in exports:
            if not isinstance(item, ExportSettings):
                raise PipelineConfigError(
                    "stages.exports", "must contain ExportSettings objects"
                )
        seen: set[ExportTarget] = set()
        for item in exports:
            if item.target in seen:
                raise PipelineConfigError(
                    "stages.exports",
                    f"duplicate export target: {item.target.value}",
                )
            seen.add(item.target)
        object.__setattr__(self, "exports", exports)

        if isinstance(self.random_seed, bool) or not isinstance(
            self.random_seed, Integral
        ):
            raise PipelineConfigError("execution.random_seed", "must be an integer")
        object.__setattr__(self, "random_seed", int(self.random_seed))

        if self.renderer is not None and not isinstance(self.renderer, RendererConfig):
            raise PipelineConfigError("renderer", "must be a RendererConfig or None")

    # -- Mapping resolution ------------------------------------------------------

    def resolve_mapping(self, base_dir: str | None = None) -> OpticalMappingConfig:
        """Return the concrete optical mapping this configuration references.

        A builtin mapping resolves without I/O. A file-based mapping requires an
        explicit ``base_dir``, is loaded from disk, and must hash to the declared
        ``mapping.digest`` — a swapped or edited mapping file fails here rather
        than silently changing canonical results (ADR-009 D3).
        """
        if self.mapping_name is not None:
            return _BUILTIN_MAPPINGS[self.mapping_name]()
        if base_dir is None:
            raise PipelineConfigError(
                "mapping.path",
                "resolving a file-based mapping requires an explicit base_dir",
            )
        assert self.mapping_path is not None  # guaranteed by __post_init__
        resolved = _resolve_against(base_dir, self.mapping_path)
        try:
            mapping = load_optical_mapping(resolved)
        except OpticalMappingError as error:
            raise PipelineConfigError("mapping.path", str(error)) from error
        if mapping.digest != self.mapping_digest:
            raise PipelineConfigError(
                "mapping.digest",
                f"mapping document at {self.mapping_path!r} hashes to "
                f"{mapping.digest}, but the configuration declares "
                f"{self.mapping_digest}",
            )
        return mapping

    # -- Path resolution ---------------------------------------------------------

    def resolve_input_path(self, base_dir: str) -> str:
        """Resolve ``input_path`` against an explicit base directory."""
        return _resolve_against(base_dir, self.input_path)

    def resolve_output_path(self, base_dir: str) -> str:
        """Resolve ``output_path`` against an explicit base directory."""
        return _resolve_against(base_dir, self.output_path)

    # -- Serialization -----------------------------------------------------------

    def to_json_dict(self) -> dict[str, Any]:
        """Return the exact, portable JSON object recorded as ``config.json``."""
        mapping_section: dict[str, Any] = {"digest": self.mapping_digest}
        if self.mapping_name is not None:
            mapping_section["name"] = self.mapping_name
        else:
            mapping_section["path"] = self.mapping_path
        document: dict[str, Any] = {
            "schema": {
                "name": PIPELINE_CONFIG_SCHEMA.name,
                "version": str(PIPELINE_CONFIG_SCHEMA.version),
            },
            "input": {
                "kind": self.input_kind.value,
                "path": self.input_path,
            },
            "mapping": mapping_section,
            "stages": {
                "validate_material": self.validate_material,
                "validate_optical": self.validate_optical,
                "exports": [item.to_json_dict() for item in self.exports],
            },
            "output": {"path": self.output_path, "overwrite": self.overwrite},
            "execution": {"random_seed": self.random_seed},
            "renderer": (
                None if self.renderer is None else self.renderer.to_json_dict()
            ),
        }
        return document

    def canonical_json(self) -> str:
        """Return stable JSON identifying this exact configuration (ADR-007 D2)."""
        return _canonical_dumps(self.to_json_dict())

    @property
    def digest(self) -> str:
        """Return the SHA-256 identity of the whole canonical configuration."""
        return _sha256(self.canonical_json())

    def scientific_canonical_json(self) -> str:
        """Return stable JSON of only the canonical-result-determining settings.

        This excludes ``output``, ``overwrite``, ``exports`` and ``renderer`` so that
        those cannot change the canonical material/optical volumes. It also excludes
        the input *path* and the mapping *path/name*: canonical results depend on the
        input payload content and the mapping content digest (ADR-007 D3, ADR-009
        D3), not on where either lives or how it was supplied.
        """
        payload = {
            "input": {"kind": self.input_kind.value},
            "mapping": {"digest": self.mapping_digest},
            "stages": {
                "validate_material": self.validate_material,
                "validate_optical": self.validate_optical,
            },
            "execution": {"random_seed": self.random_seed},
        }
        return _canonical_dumps(payload)

    @property
    def scientific_digest(self) -> str:
        """Return the SHA-256 identity of the canonical-result-determining settings."""
        return _sha256(self.scientific_canonical_json())

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> "PipelineConfig":
        """Reconstruct a configuration from its portable JSON form.

        Rejects an incompatible major schema version and any unknown top-level or
        nested keys so unrecognized required semantics fail loudly (plan Step 5).
        """
        _require_mapping("config", data)
        _reject_unknown_keys(
            "config",
            data,
            {"schema", "input", "mapping", "stages", "output", "execution", "renderer"},
        )
        _check_schema(data)

        input_section = _require_mapping("input", data.get("input"))
        _reject_unknown_keys("input", input_section, {"kind", "path"})
        _require_key("input", input_section, "kind")
        _require_key("input", input_section, "path")

        mapping_section = _require_mapping("mapping", data.get("mapping"))
        _reject_unknown_keys("mapping", mapping_section, {"name", "path", "digest"})
        if ("name" in mapping_section) == ("path" in mapping_section):
            raise PipelineConfigError(
                "mapping", "declare exactly one of mapping.name or mapping.path"
            )
        if "path" in mapping_section:
            _require_key("mapping", mapping_section, "digest")

        stages_section = _require_mapping("stages", data.get("stages"))
        _reject_unknown_keys(
            "stages",
            stages_section,
            {"validate_material", "validate_optical", "exports"},
        )
        exports = tuple(
            ExportSettings.from_json_dict(item)
            for item in stages_section.get("exports", ())
        )

        output_section = _require_mapping("output", data.get("output"))
        _reject_unknown_keys("output", output_section, {"path", "overwrite"})
        _require_key("output", output_section, "path")

        execution_section = _require_mapping("execution", data.get("execution", {}))
        _reject_unknown_keys("execution", execution_section, {"random_seed"})

        renderer_section = data.get("renderer")
        renderer = (
            None
            if renderer_section is None
            else RendererConfig.from_json_dict(renderer_section)
        )

        return cls(
            input_kind=input_section["kind"],
            input_path=input_section["path"],
            output_path=output_section["path"],
            mapping_name=mapping_section.get("name"),
            mapping_path=mapping_section.get("path"),
            mapping_digest=mapping_section.get("digest"),
            validate_material=stages_section.get("validate_material", True),
            validate_optical=stages_section.get("validate_optical", True),
            exports=exports,
            overwrite=output_section.get("overwrite", False),
            random_seed=execution_section.get("random_seed", 0),
            renderer=renderer,
        )

    @classmethod
    def from_json(cls, text: str) -> "PipelineConfig":
        """Parse a configuration from a JSON document string."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as error:
            raise PipelineConfigError("config", f"invalid JSON: {error}") from error
        return cls.from_json_dict(data)


# -- Helpers -----------------------------------------------------------------------


def _canonical_dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _sha256(text: str) -> str:
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _normalize_path(field_path: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PipelineConfigError(field_path, "must be a non-empty string")
    return value


def _resolve_against(base_dir: str, path: str) -> str:
    if not isinstance(base_dir, str) or not base_dir.strip():
        raise PipelineConfigError("base_dir", "must be a non-empty string")
    return str((Path(base_dir) / path).resolve())


def _require_mapping(field_path: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PipelineConfigError(field_path, "must be a JSON object")
    return value


def _reject_unknown_keys(
    field_path: str, data: Mapping[str, Any], allowed: set[str]
) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise PipelineConfigError(field_path, f"unknown keys: {unknown}")


def _require_key(field_path: str, data: Mapping[str, Any], key: str) -> None:
    if key not in data:
        raise PipelineConfigError(f"{field_path}.{key}", "is required")


def _check_schema(data: Mapping[str, Any]) -> None:
    schema = _require_mapping("schema", data.get("schema"))
    _reject_unknown_keys("schema", schema, {"name", "version"})
    name = schema.get("name")
    if name != PIPELINE_CONFIG_SCHEMA.name:
        raise PipelineConfigError(
            "schema.name",
            f"must be {PIPELINE_CONFIG_SCHEMA.name!r}, got {name!r}",
        )
    version_text = schema.get("version")
    if not isinstance(version_text, str):
        raise PipelineConfigError("schema.version", "must be a string")
    try:
        version = SchemaVersion.parse(version_text)
    except (TypeError, ValueError) as error:
        raise PipelineConfigError("schema.version", str(error)) from error
    if not PIPELINE_CONFIG_SCHEMA.version.has_compatible_major(version):
        raise PipelineConfigError(
            "schema.version",
            f"incompatible major version {version}; this build supports "
            f"{PIPELINE_CONFIG_SCHEMA.version.major}.x",
        )


def _require_digest_format(field_path: str, value: Any) -> None:
    if not isinstance(value, str):
        raise PipelineConfigError(
            field_path,
            "a file-based mapping requires its declared sha256 digest (ADR-009 D3)",
        )
    prefix, _, digest = value.partition(":")
    if prefix != "sha256" or len(digest) != 64 or any(
        char not in "0123456789abcdef" for char in digest
    ):
        raise PipelineConfigError(
            field_path, "must be 'sha256:' followed by 64 lowercase hex digits"
        )
