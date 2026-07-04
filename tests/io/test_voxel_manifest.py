"""Tests for the ``vdbmat.voxels`` direct material-voxel reader (ADR-006, Step 2)."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vdbmat.core import MaterialRole
from vdbmat.io import (
    VoxelManifestError,
    inspect_material_label_manifest,
    read_material_label_manifest,
    write_material_label_manifest,
)


def _coupon_label() -> np.ndarray[Any, np.dtype[np.uint16]]:
    label = np.zeros((2, 3, 4), dtype=np.uint16)
    label[0, 0, :] = [1, 1, 1, 1]
    label[0, 1, :] = [1, 2, 2, 1]
    label[0, 2, :] = [1, 1, 1, 3]
    label[1, :, :] = 1
    return label


def _payload_bytes(array: np.ndarray[Any, np.dtype[np.uint16]]) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def _base_manifest(sha: str) -> dict[str, Any]:
    return {
        "format": "vdbmat.voxels",
        "format_version": "1.0.0",
        "asset_type": "material-label",
        "payload": {
            "path": "coupon.material_id.npy",
            "sha256": sha,
            "dtype": "uint16",
            "dimensions": ["z", "y", "x"],
        },
        "shape_zyx": [2, 3, 4],
        "voxel_size_xyz_m": [0.00004, 0.00005, 0.00003],
        "local_to_world": [
            [1, 0, 0, 0.010],
            [0, 1, 0, 0.020],
            [0, 0, 1, 0.030],
            [0, 0, 0, 1],
        ],
        "materials": [
            {"material_id": 0, "name": "background", "role": "background"},
            {"material_id": 1, "name": "transparent", "role": "material"},
            {"material_id": 2, "name": "white", "role": "material"},
            {"material_id": 3, "name": "black", "role": "material"},
        ],
        "source": {
            "generator": "vdbmat.voxels",
            "generator_version": "1.0.0",
            "identity": "window-coupon",
        },
    }


def _write(tmp_path: Path, manifest: dict[str, Any], payload: bytes) -> Path:
    (tmp_path / "coupon.material_id.npy").write_bytes(payload)
    manifest_path = tmp_path / "coupon.voxels.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


@pytest.fixture
def coupon(tmp_path: Path) -> Path:
    payload = _payload_bytes(_coupon_label())
    sha = hashlib.sha256(payload).hexdigest()
    return _write(tmp_path, _base_manifest(sha), payload)


def test_reads_exact_cells_geometry_palette(coupon: Path) -> None:
    volume = read_material_label_manifest(coupon)
    label = _coupon_label()
    assert np.array_equal(np.asarray(volume.material_id), label)
    # The asymmetric black marker survives at exactly material_id[0, 2, 3].
    assert int(volume.material_id[0, 2, 3]) == 3
    assert volume.geometry.shape_zyx == (2, 3, 4)
    assert volume.geometry.voxel_size_xyz_m == (0.00004, 0.00005, 0.00003)
    assert volume.palette.material_ids == (0, 1, 2, 3)
    assert volume.palette.by_id(0).role is MaterialRole.BACKGROUND


def test_provenance_records_format_and_checksum(coupon: Path) -> None:
    payload = _payload_bytes(_coupon_label())
    sha = hashlib.sha256(payload).hexdigest()
    volume = read_material_label_manifest(coupon)
    assert "vdbmat.voxels/1.0.0" in volume.provenance.sources
    assert f"sha256:{sha}" in volume.provenance.sources
    assert "identity:window-coupon" in volume.provenance.sources


def test_repeated_reads_are_structurally_equal(coupon: Path) -> None:
    first = read_material_label_manifest(coupon)
    second = read_material_label_manifest(coupon)
    assert np.array_equal(np.asarray(first.material_id), np.asarray(second.material_id))
    assert first.geometry == second.geometry
    assert first.palette.material_ids == second.palette.material_ids
    assert first.provenance.sources == second.provenance.sources


def test_metadata_only_inspection(coupon: Path) -> None:
    inspection = inspect_material_label_manifest(coupon)
    assert inspection.shape_zyx == (2, 3, 4)
    assert inspection.material_ids == (0, 1, 2, 3)
    assert inspection.source_identity == "window-coupon"
    assert len(inspection.payload_sha256) == 64


def test_convenience_millimetre_unit(tmp_path: Path) -> None:
    payload = _payload_bytes(_coupon_label())
    sha = hashlib.sha256(payload).hexdigest()
    manifest = _base_manifest(sha)
    del manifest["voxel_size_xyz_m"]
    manifest["voxel_size"] = {"value": [0.04, 0.05, 0.03], "unit": "mm"}
    volume = read_material_label_manifest(_write(tmp_path, manifest, payload))
    assert volume.geometry.voxel_size_xyz_m == pytest.approx(
        (0.00004, 0.00005, 0.00003)
    )


def test_malformed_json_fails(tmp_path: Path) -> None:
    (tmp_path / "bad.voxels.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(VoxelManifestError) as info:
        read_material_label_manifest(tmp_path / "bad.voxels.json")
    assert info.value.field_path == "manifest"


def test_incompatible_major_version_fails(tmp_path: Path) -> None:
    payload = _payload_bytes(_coupon_label())
    sha = hashlib.sha256(payload).hexdigest()
    manifest = _base_manifest(sha)
    manifest["format_version"] = "2.0.0"
    with pytest.raises(VoxelManifestError) as info:
        read_material_label_manifest(_write(tmp_path, manifest, payload))
    assert info.value.field_path == "format_version"


def test_wrong_dtype_fails(tmp_path: Path) -> None:
    # Ship an int32 payload while the manifest still declares uint16.
    buffer = io.BytesIO()
    np.save(buffer, _coupon_label().astype(np.int32))
    payload = buffer.getvalue()
    manifest = _base_manifest(hashlib.sha256(payload).hexdigest())
    with pytest.raises(VoxelManifestError) as info:
        read_material_label_manifest(_write(tmp_path, manifest, payload))
    assert info.value.field_path == "payload.dtype"


def test_wrong_shape_fails(tmp_path: Path) -> None:
    array = np.ones((2, 3, 5), dtype=np.uint16)
    payload = _payload_bytes(array)
    manifest = _base_manifest(hashlib.sha256(payload).hexdigest())
    with pytest.raises(VoxelManifestError) as info:
        read_material_label_manifest(_write(tmp_path, manifest, payload))
    assert info.value.field_path == "payload.shape"


def test_undeclared_material_id_fails(tmp_path: Path) -> None:
    label = _coupon_label()
    label[0, 0, 0] = 99
    payload = _payload_bytes(label)
    manifest = _base_manifest(hashlib.sha256(payload).hexdigest())
    with pytest.raises(ValueError) as info:
        read_material_label_manifest(_write(tmp_path, manifest, payload))
    assert "material" in str(info.value).lower()


def test_missing_payload_fails(tmp_path: Path) -> None:
    payload = _payload_bytes(_coupon_label())
    manifest = _base_manifest(hashlib.sha256(payload).hexdigest())
    manifest_path = tmp_path / "coupon.voxels.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(VoxelManifestError) as info:
        read_material_label_manifest(manifest_path)
    assert info.value.field_path == "payload.path"


def test_checksum_mismatch_fails(tmp_path: Path) -> None:
    payload = _payload_bytes(_coupon_label())
    manifest = _base_manifest("0" * 64)
    with pytest.raises(VoxelManifestError) as info:
        read_material_label_manifest(_write(tmp_path, manifest, payload))
    assert info.value.field_path == "payload.sha256"


def test_path_traversal_fails(tmp_path: Path) -> None:
    payload = _payload_bytes(_coupon_label())
    manifest = _base_manifest(hashlib.sha256(payload).hexdigest())
    manifest["payload"]["path"] = "../escape.npy"
    with pytest.raises(VoxelManifestError) as info:
        read_material_label_manifest(_write(tmp_path, manifest, payload))
    assert info.value.field_path == "payload.path"


def test_absolute_path_fails(tmp_path: Path) -> None:
    payload = _payload_bytes(_coupon_label())
    manifest = _base_manifest(hashlib.sha256(payload).hexdigest())
    manifest["payload"]["path"] = "/etc/passwd"
    with pytest.raises(VoxelManifestError) as info:
        read_material_label_manifest(_write(tmp_path, manifest, payload))
    assert info.value.field_path == "payload.path"


def test_non_finite_geometry_fails(tmp_path: Path) -> None:
    payload = _payload_bytes(_coupon_label())
    manifest = _base_manifest(hashlib.sha256(payload).hexdigest())
    manifest["voxel_size_xyz_m"] = [0.0, 0.00005, 0.00003]
    with pytest.raises(VoxelManifestError) as info:
        read_material_label_manifest(_write(tmp_path, manifest, payload))
    assert info.value.field_path.startswith("voxel_size_xyz_m")


def test_invalid_transform_fails(tmp_path: Path) -> None:
    payload = _payload_bytes(_coupon_label())
    manifest = _base_manifest(hashlib.sha256(payload).hexdigest())
    manifest["local_to_world"] = [
        [2, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    with pytest.raises(VoxelManifestError) as info:
        read_material_label_manifest(_write(tmp_path, manifest, payload))
    assert info.value.field_path == "local_to_world"


def test_unknown_top_level_field_fails(tmp_path: Path) -> None:
    payload = _payload_bytes(_coupon_label())
    manifest = _base_manifest(hashlib.sha256(payload).hexdigest())
    manifest["surprise"] = True
    with pytest.raises(VoxelManifestError) as info:
        read_material_label_manifest(_write(tmp_path, manifest, payload))
    assert info.value.field_path == "manifest"


def test_writer_output_round_trips_through_the_reader(tmp_path: Path) -> None:
    payload = _payload_bytes(_coupon_label())
    manifest = _base_manifest(hashlib.sha256(payload).hexdigest())
    volume = read_material_label_manifest(_write(tmp_path, manifest, payload))

    out_dir = tmp_path / "emitted"
    manifest_path = write_material_label_manifest(
        out_dir, "roundtrip", volume, identity="writer-roundtrip"
    )
    assert manifest_path == out_dir / "roundtrip.voxels.json"

    restored = read_material_label_manifest(manifest_path)
    assert restored.geometry == volume.geometry
    assert restored.palette == volume.palette
    np.testing.assert_array_equal(restored.material_id, volume.material_id)
    assert restored.provenance.generator == volume.provenance.generator
    assert "identity:writer-roundtrip" in restored.provenance.sources


def test_writer_rejects_invalid_name(tmp_path: Path) -> None:
    payload = _payload_bytes(_coupon_label())
    manifest = _base_manifest(hashlib.sha256(payload).hexdigest())
    volume = read_material_label_manifest(_write(tmp_path, manifest, payload))
    with pytest.raises(VoxelManifestError) as info:
        write_material_label_manifest(tmp_path, "a/b", volume)
    assert info.value.field_path == "name"


def test_import_does_not_require_zarr_or_renderers() -> None:
    import vdbmat.io.voxel_manifest as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "zarr" not in source
    assert "mitsuba" not in source
    assert "openvdb" not in source
