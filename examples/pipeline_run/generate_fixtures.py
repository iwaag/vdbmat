"""Regenerate the committed Phase 1 representative input fixtures.

Deterministic: rerunning this script reproduces byte-identical payloads and expected
summaries under ``examples/phase1/inputs/``. Intentionally invalid samples for
error-path tests are written under ``examples/phase1/inputs/invalid/``.

Usage::

    uv run python examples/phase1/generate_fixtures.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from vdbmat.fixtures import (
    COUPON_PAYLOAD_NAME,
    window_coupon_manifest,
    window_coupon_payload_bytes,
    write_phase1_fixtures,
)
from vdbmat.optics import phase0_provisional_mapping, write_optical_mapping

INPUTS = Path(__file__).parent / "inputs"
INVALID = INPUTS / "invalid"
MAPPINGS = Path(__file__).parent / "mappings"


def _write_invalid_samples() -> None:
    INVALID.mkdir(parents=True, exist_ok=True)

    # 1. Direct-voxel manifest whose declared checksum does not match the payload.
    payload = window_coupon_payload_bytes()
    correct_sha = hashlib.sha256(payload).hexdigest()
    bad_manifest = window_coupon_manifest("0" * 64)
    bad_manifest["source"]["identity"] = "window-coupon-bad-checksum"
    (INVALID / COUPON_PAYLOAD_NAME).write_bytes(payload)
    (INVALID / "window_coupon.bad_checksum.voxels.json").write_text(
        json.dumps(bad_manifest, indent=2) + "\n", encoding="utf-8"
    )
    assert correct_sha != "0" * 64  # sanity: the sample really is inconsistent


def main() -> None:
    written = write_phase1_fixtures(INPUTS)
    _write_invalid_samples()
    # The builtin mapping as an external document (ADR-009 D3): its digest must
    # equal the builtin's, which tests/optics/test_mapping_document.py asserts.
    mapping = phase0_provisional_mapping()
    mapping_path = write_optical_mapping(
        MAPPINGS / "phase0-provisional-materials-v1.optical-mapping.json", mapping
    )
    for name, path in written.items():
        print(f"{name}: {path.relative_to(INPUTS.parent)}")
    print("invalid samples written under", INVALID.relative_to(INPUTS.parent))
    print(f"mapping: {mapping_path.relative_to(INPUTS.parent)} ({mapping.digest})")


if __name__ == "__main__":
    main()
