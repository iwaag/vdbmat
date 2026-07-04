from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from vdbmat.core import GridGeometry
from vdbmat.core.volumes import (
    MaterialLabelVolume,
    MaterialMixtureVolume,
    OpticalPropertyVolume,
    VolumeAssetType,
)
from vdbmat.fixtures import (
    all_synthetic_fixtures,
    transparent_opaque_interface,
    two_material_mixture_ramp,
)
from vdbmat.io import (
    VolumeIOError,
    inspect_volume,
    read_optical_region,
    read_volume,
    write_volume,
)
from vdbmat.optics import (
    map_material_volume_to_optical,
    phase0_provisional_mapping,
)


def _optical() -> OpticalPropertyVolume:
    return map_material_volume_to_optical(
        transparent_opaque_interface().volume, phase0_provisional_mapping()
    )


def _assert_volume_equal(left: object, right: object) -> None:
    assert type(left) is type(right)
    assert left.geometry == right.geometry  # type: ignore[attr-defined]
    assert left.provenance == right.provenance  # type: ignore[attr-defined]
    assert left.schema == right.schema  # type: ignore[attr-defined]
    if isinstance(left, MaterialLabelVolume) and isinstance(right, MaterialLabelVolume):
        assert left.palette == right.palette
        np.testing.assert_array_equal(left.material_id, right.material_id)
    elif isinstance(left, MaterialMixtureVolume) and isinstance(
        right, MaterialMixtureVolume
    ):
        assert left.palette == right.palette
        np.testing.assert_array_equal(left.fractions, right.fractions)
        np.testing.assert_array_equal(left.material_ids, right.material_ids)
    elif isinstance(left, OpticalPropertyVolume) and isinstance(
        right, OpticalPropertyVolume
    ):
        assert left.optical_basis == right.optical_basis
        for name in ("sigma_a", "sigma_s", "g", "ior"):
            np.testing.assert_array_equal(getattr(left, name), getattr(right, name))
    else:
        raise AssertionError("unexpected volume pair")


@pytest.mark.parametrize(
    "fixture", all_synthetic_fixtures(), ids=lambda item: item.manifest.name
)
def test_material_fixture_exact_round_trip(tmp_path: Path, fixture: Any) -> None:
    path = tmp_path / "asset.zarr"
    write_volume(path, fixture.volume)
    _assert_volume_equal(read_volume(path), fixture.volume)


def test_optical_exact_round_trip(tmp_path: Path) -> None:
    volume = _optical()
    path = tmp_path / "asset.zarr"
    write_volume(path, volume)
    _assert_volume_equal(read_volume(path), volume)


def test_inspection_reports_fields_chunks_units_and_schema(tmp_path: Path) -> None:
    path = tmp_path / "asset.zarr"
    write_volume(path, _optical())
    result = inspect_volume(path)
    assert result.asset_type is VolumeAssetType.OPTICAL_PROPERTY
    assert result.schema_name == "vdbmat.volume"
    assert result.schema_version == "1.0.0"
    assert [item.name for item in result.arrays] == ["sigma_a", "sigma_s", "g", "ior"]
    assert result.arrays[0].dimensions == ("z", "y", "x", "basis")
    assert result.arrays[0].unit == "m^-1"
    assert result.arrays[0].chunks == (2, 2, 2, 3)


def test_inspection_does_not_read_array_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "asset.zarr"
    write_volume(path, _optical())

    def fail_payload_read(*args: object, **kwargs: object) -> object:
        raise AssertionError("inspection attempted an array payload read")

    monkeypatch.setattr(zarr.Array, "__getitem__", fail_payload_read)
    assert inspect_volume(path).asset_type is VolumeAssetType.OPTICAL_PROPERTY


def test_partial_read_returns_exact_values_and_shifted_world_geometry(
    tmp_path: Path,
) -> None:
    volume = _optical()
    path = tmp_path / "asset.zarr"
    write_volume(path, volume)
    region = (slice(1, 2), slice(1, 4), slice(2, 5))
    partial = read_optical_region(path, region)
    for name in ("sigma_a", "sigma_s", "g", "ior"):
        np.testing.assert_array_equal(
            getattr(partial, name), getattr(volume, name)[region]
        )
    assert partial.geometry.shape_zyx == (1, 3, 3)
    assert partial.geometry.cell_center_world((0, 0, 0)) == pytest.approx(
        volume.geometry.cell_center_world((1, 1, 2))
    )


def test_partial_read_preserves_world_location_under_rotation(tmp_path: Path) -> None:
    volume = _optical()
    geometry = GridGeometry(
        shape_zyx=volume.geometry.shape_zyx,
        voxel_size_xyz_m=volume.geometry.voxel_size_xyz_m,
        local_to_world=(
            (0.0, -1.0, 0.0, 1.0),
            (1.0, 0.0, 0.0, 2.0),
            (0.0, 0.0, 1.0, 3.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )
    rotated = replace(volume, geometry=geometry)
    path = tmp_path / "asset.zarr"
    write_volume(path, rotated)
    partial = read_optical_region(path, (slice(1, 2), slice(2, 4), slice(3, 6)))
    assert partial.geometry.cell_center_world((0, 0, 0)) == pytest.approx(
        rotated.geometry.cell_center_world((1, 2, 3))
    )


@pytest.mark.parametrize(
    "region",
    [
        (slice(0, 0), slice(None), slice(None)),
        (slice(None, None, 2), slice(None), slice(None)),
    ],
)
def test_partial_read_rejects_empty_or_strided_regions(
    tmp_path: Path, region: Any
) -> None:
    path = tmp_path / "asset.zarr"
    write_volume(path, _optical())
    with pytest.raises(ValueError, match="region_zyx"):
        read_optical_region(path, region)


def test_partial_read_rejects_material_asset(tmp_path: Path) -> None:
    path = tmp_path / "asset.zarr"
    write_volume(path, transparent_opaque_interface().volume)
    with pytest.raises(VolumeIOError, match="must be 'optical-property'"):
        read_optical_region(path, (slice(None), slice(None), slice(None)))


def _mutate_manifest(path: Path, mutation: Any) -> None:
    root = zarr.open_group(path, mode="r+")
    manifest = dict(root.attrs["vdbmat"])
    mutation(manifest)
    root.attrs["vdbmat"] = manifest


def test_missing_array_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "asset.zarr"
    write_volume(path, _optical())
    shutil.rmtree(path / "arrays" / "g")
    with pytest.raises(VolumeIOError, match=r"arrays\.g: required array is missing"):
        read_volume(path)


def test_wrong_stored_shape_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "asset.zarr"
    write_volume(path, _optical())
    root = zarr.open_group(path, mode="r+")
    arrays = root["arrays"]
    del arrays["ior"]
    arrays.create_array(
        "ior",
        data=np.ones((1, 1, 1), dtype=np.float32),
        attributes={"dimensions": ["z", "y", "x"], "unit": "1"},
    )
    with pytest.raises(VolumeIOError, match=r"arrays\.ior\.shape"):
        read_volume(path)


def test_invalid_geometry_unit_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "asset.zarr"
    write_volume(path, _optical())

    def mutate(manifest: dict[str, Any]) -> None:
        manifest["geometry"]["length_unit"] = "mm"

    _mutate_manifest(path, mutate)
    with pytest.raises(VolumeIOError, match="length_unit: must be 'm'"):
        read_volume(path)


def test_incompatible_major_schema_is_rejected_clearly(tmp_path: Path) -> None:
    path = tmp_path / "asset.zarr"
    write_volume(path, _optical())

    def mutate(manifest: dict[str, Any]) -> None:
        manifest["schema"]["version"] = "2.0.0"

    _mutate_manifest(path, mutate)
    with pytest.raises(VolumeIOError, match="incompatible major version 2"):
        read_volume(path)


def test_missing_required_manifest_attribute_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "asset.zarr"
    write_volume(path, _optical())
    root = zarr.open_group(path, mode="r+")
    del root.attrs["vdbmat"]
    with pytest.raises(VolumeIOError, match="vdbmat: must be an object"):
        read_volume(path)


def test_unknown_optional_array_and_attributes_are_ignored(tmp_path: Path) -> None:
    volume = two_material_mixture_ramp().volume
    path = tmp_path / "asset.zarr"
    write_volume(path, volume)
    root = zarr.open_group(path, mode="r+")
    root.attrs["extension"] = {"producer": "test"}
    arrays = root["arrays"]
    arrays.create_array("preview", data=np.zeros((1,), dtype=np.uint8))
    _assert_volume_equal(read_volume(path), volume)


def test_existing_target_requires_explicit_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "asset.zarr"
    first = transparent_opaque_interface().volume
    second = two_material_mixture_ramp().volume
    write_volume(path, first)
    with pytest.raises(FileExistsError):
        write_volume(path, second)
    _assert_volume_equal(read_volume(path), first)
    write_volume(path, second, overwrite=True)
    _assert_volume_equal(read_volume(path), second)
    assert not list(tmp_path.glob(".asset.zarr.*"))
