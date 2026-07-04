"""Tests for the committed Phase 1 representative input fixtures (plan Step 4).

These assert the committed fixtures regenerate byte-identically, that both
fixture objects produce valid canonical material volumes through the direct-voxel
contract, and that reader output matches the *committed analytic summaries* —
which are computed from the fixture definition, independent of the reader under
test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vbdmat.core import MaterialLabelVolume, MaterialRole
from vbdmat.fixtures import (
    COUPON_MANIFEST_NAME,
    COUPON_PAYLOAD_NAME,
    WEDGE_MANIFEST_NAME,
    WEDGE_PAYLOAD_NAME,
    window_coupon_label,
    write_phase1_fixtures,
)
from vbdmat.io import read_material_label_manifest
from vbdmat.io.errors import VoxelManifestError

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INPUTS = _REPO_ROOT / "examples" / "phase1" / "inputs"


def _committed_summary(name: str) -> dict:
    return json.loads((_INPUTS / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Deterministic, byte-identical regeneration
# --------------------------------------------------------------------------- #


def test_committed_inputs_exist() -> None:
    for name in (
        COUPON_PAYLOAD_NAME,
        COUPON_MANIFEST_NAME,
        "window_coupon.expected.json",
        WEDGE_MANIFEST_NAME,
        WEDGE_PAYLOAD_NAME,
        "stepped_wedge.expected.json",
    ):
        assert (_INPUTS / name).is_file(), name


def test_regeneration_is_byte_identical(tmp_path: Path) -> None:
    written = write_phase1_fixtures(tmp_path)
    for path in written.values():
        committed = _INPUTS / path.name
        assert committed.read_bytes() == path.read_bytes(), path.name


# --------------------------------------------------------------------------- #
# Direct-voxel coupon: valid canonical volume matching the analytic summary
# --------------------------------------------------------------------------- #


def test_coupon_imports_to_valid_canonical_volume() -> None:
    volume = read_material_label_manifest(_INPUTS / COUPON_MANIFEST_NAME)
    assert isinstance(volume, MaterialLabelVolume)  # validated on construction
    assert volume.palette.by_id(0).role is MaterialRole.BACKGROUND


def test_coupon_matches_committed_summary() -> None:
    expected = _committed_summary("window_coupon.expected.json")
    volume = read_material_label_manifest(_INPUTS / COUPON_MANIFEST_NAME)
    label = np.asarray(volume.material_id)

    assert list(volume.geometry.shape_zyx) == expected["shape_zyx"]
    assert list(volume.geometry.voxel_size_xyz_m) == expected["voxel_size_xyz_m"]

    ids, counts = np.unique(label, return_counts=True)
    actual_counts = {int(i): int(c) for i, c in zip(ids, counts, strict=True)}
    expected_counts = {
        int(k): v for k, v in expected["material_counts"].items() if v > 0
    }
    assert actual_counts == expected_counts

    # The asymmetric black marker sits where the summary says it does.
    marker = expected["black_marker"]
    z0, y0, x0 = marker["z"][0], marker["y"][0], marker["x"][0]
    assert int(label[z0, y0, x0]) == marker["material_id"]
    assert f"sha256:{expected['payload_sha256']}" in volume.provenance.sources


def test_coupon_features_are_axis_asymmetric() -> None:
    # A transpose of any two axes relocates the unique black marker, so an accidental
    # axis swap in a consumer is detectable.
    label = window_coupon_label()
    (z, y, x) = np.argwhere(label == 3)[0]
    assert int(label[z, y, x]) == 3
    for a, b in ((0, 1), (0, 2), (1, 2)):
        swapped = np.swapaxes(label, a, b)
        if swapped.shape != label.shape:
            continue  # differing extents already make a swap invalid
        assert not np.array_equal(swapped, label)


# --------------------------------------------------------------------------- #
# Stepped wedge: valid canonical volume matching the analytic summary
# --------------------------------------------------------------------------- #


def test_wedge_imports_to_valid_canonical_volume() -> None:
    volume = read_material_label_manifest(_INPUTS / WEDGE_MANIFEST_NAME)
    assert isinstance(volume, MaterialLabelVolume)  # validated on construction
    assert volume.palette.by_id(0).role is MaterialRole.BACKGROUND


def test_wedge_matches_committed_summary() -> None:
    expected = _committed_summary("stepped_wedge.expected.json")
    volume = read_material_label_manifest(_INPUTS / WEDGE_MANIFEST_NAME)
    label = np.asarray(volume.material_id)

    assert list(volume.geometry.shape_zyx) == expected["shape_zyx"]
    assert list(volume.geometry.voxel_size_xyz_m) == expected["voxel_size_xyz_m"]
    translation = [row[3] for row in volume.geometry.local_to_world[:3]]
    assert translation == pytest.approx(
        expected["local_to_world_translation_m"]
    )
    assert int(np.count_nonzero(label)) == expected["occupied_cells"]
    assert f"sha256:{expected['payload_sha256']}" in volume.provenance.sources

    # Per-step occupancy along X (padded interior index 1..16, 4 cells per step).
    for step, count in expected["per_step_occupied"].items():
        k = int(step)
        xs = slice(1 + (k - 1) * 4, 1 + k * 4)
        assert int(label[:, :, xs].sum()) == count


# --------------------------------------------------------------------------- #
# Intentionally invalid samples for error-path tests
# --------------------------------------------------------------------------- #


def test_invalid_coupon_checksum_is_rejected() -> None:
    bad = _INPUTS / "invalid" / "window_coupon.bad_checksum.voxels.json"
    with pytest.raises(VoxelManifestError) as info:
        read_material_label_manifest(bad)
    assert info.value.field_path == "payload.sha256"
