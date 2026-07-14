"""Check shared contracts across all canonical Phase 0 consumer fixtures."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from vdbmat.conformance import (
    EXPECTED_ADAPTER_DIFFERENCES,
    ConformanceCheck,
    ConformanceLayer,
    CrossConsumerConformanceReport,
    FixtureConformance,
    check_fixture_conformance,
    image_sanity_check,
)
from vdbmat.core.volumes import OpticalPropertyVolume
from vdbmat.fixtures import all_synthetic_fixtures
from vdbmat.io import read_volume, write_volume
from vdbmat.optics import map_material_volume_to_optical, phase0_provisional_mapping


def _render_records(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    report = json.loads((path / "render-report.json").read_text(encoding="utf-8"))
    records = {str(item["fixture"]): item for item in report["fixtures"]}
    return report, records


def _append_image_checks(
    results: list[FixtureConformance],
    directory: Path,
    *,
    consumer: str,
    expected_size: tuple[int, int],
) -> None:
    try:
        _, records = _render_records(directory)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        for index, result in enumerate(results):
            failure = ConformanceCheck(
                f"{consumer}-image-{result.fixture}",
                ConformanceLayer.IMAGE_SANITY,
                False,
                f"render report could not be read: {error}",
            )
            results[index] = FixtureConformance(
                result.fixture, (*result.checks, failure)
            )
        return

    for index, result in enumerate(results):
        record = records.get(result.fixture)
        if record is None:
            check = ConformanceCheck(
                f"{consumer}-image-{result.fixture}",
                ConformanceLayer.IMAGE_SANITY,
                False,
                "fixture is missing from render report",
            )
        else:
            try:
                image_path = directory / str(record["png"])
            except KeyError:
                check = ConformanceCheck(
                    f"{consumer}-image-{result.fixture}",
                    ConformanceLayer.IMAGE_SANITY,
                    False,
                    "render record has no PNG path",
                )
            else:
                check = image_sanity_check(
                    result.fixture,
                    image_path,
                    expected_size=expected_size,
                    consumer=consumer,
                )
        results[index] = FixtureConformance(result.fixture, (*result.checks, check))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--mitsuba-renders", type=Path)
    parser.add_argument("--mitsuba-width", type=int, default=64)
    parser.add_argument("--mitsuba-height", type=int, default=64)
    parser.add_argument("--cycles-renders", type=Path)
    parser.add_argument("--cycles-width", type=int, default=64)
    parser.add_argument("--cycles-height", type=int, default=64)
    args = parser.parse_args()

    mapping = phase0_provisional_mapping()
    results: list[FixtureConformance] = []
    with tempfile.TemporaryDirectory(prefix="vdbmat-conformance-") as temporary:
        root = Path(temporary)
        for fixture in all_synthetic_fixtures():
            name = fixture.manifest.name
            try:
                canonical = map_material_volume_to_optical(fixture.volume, mapping)
            except Exception as error:
                results.append(
                    FixtureConformance(
                        name,
                        (
                            ConformanceCheck(
                                "canonical-mapping",
                                ConformanceLayer.CANONICAL,
                                False,
                                f"{type(error).__name__}: {error}",
                            ),
                        ),
                    )
                )
                continue
            try:
                store = root / f"{name}.zarr"
                write_volume(store, canonical)
                restored = read_volume(store)
                if not isinstance(restored, OpticalPropertyVolume):
                    raise TypeError("round-trip did not return an optical volume")
            except Exception as error:
                results.append(
                    FixtureConformance(
                        name,
                        (
                            ConformanceCheck(
                                "zarr-round-trip",
                                ConformanceLayer.SERIALIZATION,
                                False,
                                f"{type(error).__name__}: {error}",
                            ),
                        ),
                    )
                )
                continue
            try:
                results.append(check_fixture_conformance(name, canonical, restored))
            except Exception as error:
                results.append(
                    FixtureConformance(
                        name,
                        (
                            ConformanceCheck(
                                "consumer-conversion",
                                ConformanceLayer.ADAPTER_CONVERSION,
                                False,
                                f"{type(error).__name__}: {error}",
                            ),
                        ),
                    )
                )

    if args.mitsuba_renders is not None:
        _append_image_checks(
            results,
            args.mitsuba_renders,
            consumer="mitsuba",
            expected_size=(args.mitsuba_width, args.mitsuba_height),
        )
    if args.cycles_renders is not None:
        _append_image_checks(
            results,
            args.cycles_renders,
            consumer="cycles",
            expected_size=(args.cycles_width, args.cycles_height),
        )

    conformance = CrossConsumerConformanceReport(
        tuple(results), EXPECTED_ADAPTER_DIFFERENCES
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    conformance.write_json(args.output)
    print(
        f"{len(results)} fixtures: "
        f"{'PASS' if conformance.passed else 'FAIL'}; report={args.output}"
    )
    if not conformance.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
