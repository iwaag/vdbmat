from __future__ import annotations

import json
import struct
import subprocess
import sys
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vdbmat.conformance import (
    ConformanceLayer,
    check_fixture_conformance,
    check_png_sanity,
    png_pixel_sha256,
)
from vdbmat.fixtures import all_synthetic_fixtures, layered_material_slab
from vdbmat.optics import map_material_volume_to_optical, phase0_provisional_mapping


def _mapped(fixture):  # type: ignore[no-untyped-def]
    return map_material_volume_to_optical(fixture.volume, phase0_provisional_mapping())


@pytest.mark.parametrize(
    "fixture", all_synthetic_fixtures(), ids=lambda item: item.manifest.name
)
def test_every_fixture_conforms_before_native_consumers(fixture) -> None:  # type: ignore[no-untyped-def]
    volume = _mapped(fixture)
    result = check_fixture_conformance(fixture.manifest.name, volume, volume)
    assert result.passed
    assert {check.name for check in result.checks} == {
        "canonical-metadata",
        "zarr-round-trip",
        "openvdb-pre-consumer-fields",
        "mitsuba-pre-consumer-fields",
        "volume-bounds-and-transform",
        "material-regions-and-boundaries",
        "background-treatment",
        "coefficient-units-and-scale",
        "global-phase-reduction",
        "capability-reports",
    }


def test_serialization_mismatch_is_attributed_to_serialization() -> None:
    volume = _mapped(layered_material_slab())
    changed = volume.sigma_a.copy()
    changed[0, 0, 0, 0] += 1.0
    restored = replace(volume, sigma_a=changed)
    result = check_fixture_conformance("changed", volume, restored)
    failure = next(check for check in result.checks if not check.passed)
    assert failure.layer is ConformanceLayer.SERIALIZATION
    assert failure.name == "zarr-round-trip"


def test_png_sanity_rejects_flat_image_and_accepts_spatial_signal(
    tmp_path: Path,
) -> None:
    flat = tmp_path / "flat.png"
    varied = tmp_path / "varied.png"
    _write_rgb_png(flat, np.zeros((2, 3, 3), dtype=np.uint8))
    pixels = np.zeros((2, 3, 3), dtype=np.uint8)
    pixels[1, 2] = 255
    _write_rgb_png(varied, pixels)
    assert not check_png_sanity(flat, expected_size=(3, 2))[0]
    assert check_png_sanity(varied, expected_size=(3, 2))[0]
    assert not check_png_sanity(varied, expected_size=(2, 3))[0]


def test_png_pixel_hash_ignores_compression_bytes(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    _write_rgb_png(first, pixels, compression_level=1)
    _write_rgb_png(second, pixels, compression_level=9)

    assert first.read_bytes() != second.read_bytes()
    assert png_pixel_sha256(first) == png_pixel_sha256(second)


def test_conformance_command_processes_every_fixture(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    subprocess.run(
        [
            sys.executable,
            "examples/native_fixtures/check_cross_consumer_conformance.py",
            str(report),
        ],
        check=True,
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["fixture_count"] == len(all_synthetic_fixtures())
    assert payload["failure_count"] == 0


def _write_rgb_png(
    path: Path, pixels: np.ndarray, *, compression_level: int = -1
) -> None:  # type: ignore[type-arg]
    height, width, _ = pixels.shape
    raw = b"".join(b"\x00" + pixels[row].tobytes() for row in range(height))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return (
            struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))
        )

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=compression_level))
        + chunk(b"IEND", b"")
    )
