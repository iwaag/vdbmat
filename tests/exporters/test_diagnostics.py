import json
from pathlib import Path

import pytest

from vdbmat.boundaries import CapabilityStatus
from vdbmat.exporters.diagnostics import CapabilityEntry, CapabilityReport


def _report() -> CapabilityReport:
    return CapabilityReport(
        consumer="test-consumer",
        adapter="test-adapter",
        adapter_version="1.0.0",
        schema_name="vdbmat.volume",
        schema_version="1.0.0",
        entries=(
            CapabilityEntry(
                field="ior",
                status=CapabilityStatus.UNSUPPORTED,
                mapping="none",
                detail="test",
            ),
        ),
    )


def test_report_lookup_and_stable_json(tmp_path: Path) -> None:
    report = _report()
    assert report.by_field("ior").status is CapabilityStatus.UNSUPPORTED
    with pytest.raises(KeyError, match="missing"):
        report.by_field("missing")
    path = tmp_path / "report.json"
    report.write_json(path)
    assert json.loads(path.read_text()) == report.to_dict()
    assert path.read_bytes().endswith(b"\n")


def test_report_rejects_duplicate_fields() -> None:
    entry = CapabilityEntry("ior", CapabilityStatus.UNSUPPORTED, "none", "test")
    with pytest.raises(ValueError, match="unique"):
        CapabilityReport("test", "test", "1", "schema", "1.0.0", (entry, entry))
