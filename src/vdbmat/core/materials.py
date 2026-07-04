"""Immutable material identities and ordered palettes."""

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Integral
from types import MappingProxyType


class MaterialRole(StrEnum):
    """Schema 1.0 material roles."""

    BACKGROUND = "background"
    MATERIAL = "material"


@dataclass(frozen=True, slots=True)
class MaterialDefinition:
    """A material identity; optical values live outside this object."""

    material_id: int
    name: str
    role: MaterialRole
    external_id: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.material_id, bool) or not isinstance(
            self.material_id, Integral
        ):
            raise TypeError("material_id must be an integer")
        material_id = int(self.material_id)
        if not 0 <= material_id <= 65535:
            raise ValueError("material_id must be in [0, 65535]")
        object.__setattr__(self, "material_id", material_id)

        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("material name must be a non-empty string")
        try:
            role = MaterialRole(self.role)
        except ValueError as error:
            raise ValueError(f"unsupported material role: {self.role!r}") from error
        object.__setattr__(self, "role", role)

        if self.external_id is not None and (
            not isinstance(self.external_id, str) or not self.external_id.strip()
        ):
            raise ValueError("external_id must be None or a non-empty string")

        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class MaterialPalette:
    """An immutable ordered collection of unique material identities."""

    materials: tuple[MaterialDefinition, ...]
    _by_id: Mapping[int, MaterialDefinition] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        materials = tuple(self.materials)
        if not materials:
            raise ValueError("material palette must not be empty")
        if any(not isinstance(item, MaterialDefinition) for item in materials):
            raise TypeError(
                "material palette entries must be MaterialDefinition objects"
            )

        by_id: dict[int, MaterialDefinition] = {}
        for item in materials:
            if item.material_id in by_id:
                raise ValueError(f"duplicate material_id: {item.material_id}")
            by_id[item.material_id] = item

        background = by_id.get(0)
        if background is None:
            raise ValueError("material palette must define background material_id 0")
        if background.role is not MaterialRole.BACKGROUND:
            raise ValueError("material_id 0 must have role 'background'")
        if any(
            item.material_id != 0 and item.role is MaterialRole.BACKGROUND
            for item in materials
        ):
            raise ValueError("only material_id 0 may have role 'background'")

        object.__setattr__(self, "materials", materials)
        object.__setattr__(self, "_by_id", MappingProxyType(by_id))

    @classmethod
    def from_sequence(
        cls, materials: Sequence[MaterialDefinition]
    ) -> "MaterialPalette":
        """Normalize a material sequence into an immutable palette."""
        return cls(tuple(materials))

    @property
    def material_ids(self) -> tuple[int, ...]:
        """Return IDs in normative palette order."""
        return tuple(item.material_id for item in self.materials)

    def by_id(self, material_id: int) -> MaterialDefinition:
        """Return a material by ID, raising ``KeyError`` if it is undeclared."""
        return self._by_id[material_id]

    def __len__(self) -> int:
        return len(self.materials)

    def __iter__(self) -> Iterator[MaterialDefinition]:
        return iter(self.materials)


def _freeze_metadata(metadata: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(metadata, Mapping):
        raise TypeError("material metadata must be a mapping")
    frozen: dict[str, object] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key:
            raise ValueError("material metadata keys must be non-empty strings")
        frozen[key] = _freeze_metadata_value(value, field=f"metadata.{key}")
    return MappingProxyType(frozen)


def _freeze_metadata_value(value: object, *, field: str) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} must be finite")
        return value
    if isinstance(value, Mapping):
        nested: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{field} keys must be non-empty strings")
            nested[key] = _freeze_metadata_value(item, field=f"{field}.{key}")
        return MappingProxyType(nested)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_metadata_value(item, field=f"{field}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{field} must contain only JSON-compatible values")
