"""Tests for immutable material definitions and palettes."""

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from vdbmat.core import MaterialDefinition, MaterialPalette, MaterialRole


def background() -> MaterialDefinition:
    return MaterialDefinition(0, "air", MaterialRole.BACKGROUND)


def resin(material_id: int = 7) -> MaterialDefinition:
    return MaterialDefinition(material_id, "example-resin", MaterialRole.MATERIAL)


def test_palette_preserves_normative_order_and_supports_lookup() -> None:
    palette = MaterialPalette.from_sequence([background(), resin()])

    assert palette.material_ids == (0, 7)
    assert [item.name for item in palette] == ["air", "example-resin"]
    assert palette.by_id(7).name == "example-resin"
    assert len(palette) == 2
    with pytest.raises(KeyError):
        palette.by_id(99)


def test_material_metadata_is_deeply_frozen() -> None:
    source = {"batch": "A", "measurements": [1, {"valid": True}]}
    material = MaterialDefinition(7, "resin", MaterialRole.MATERIAL, metadata=source)

    assert isinstance(material.metadata, MappingProxyType)
    assert material.metadata["measurements"] == (1, {"valid": True})
    source["batch"] = "changed"
    assert material.metadata["batch"] == "A"
    with pytest.raises(TypeError):
        material.metadata["batch"] = "B"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        material.name = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "material_id",
    [-1, 65536, True, 1.5],
)
def test_invalid_material_ids_are_rejected(material_id: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        MaterialDefinition(  # type: ignore[arg-type]
            material_id, "bad", MaterialRole.MATERIAL
        )


def test_non_json_material_metadata_is_rejected() -> None:
    with pytest.raises(TypeError, match="JSON-compatible"):
        MaterialDefinition(7, "bad", MaterialRole.MATERIAL, metadata={"bad": object()})


@pytest.mark.parametrize(
    "materials",
    [
        (),
        (resin(),),
        (
            MaterialDefinition(0, "not-background", MaterialRole.MATERIAL),
            resin(),
        ),
        (background(), MaterialDefinition(1, "second-bg", MaterialRole.BACKGROUND)),
        (background(), resin(), resin()),
    ],
)
def test_invalid_palettes_are_rejected(
    materials: tuple[MaterialDefinition, ...],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        MaterialPalette(materials)
