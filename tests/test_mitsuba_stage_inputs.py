from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

DEMO_DIR = Path(__file__).parents[1] / "examples" / "pipeline_run" / "demo"
sys.path.insert(0, str(DEMO_DIR))

from mitsuba_stage_inputs import (  # noqa: E402
    InputCandidate,
    InputCatalogError,
    InputKind,
    describe_candidate,
    resolve_candidate,
    resolve_input_root,
    scan_input_catalog,
)

from vdbmat.fixtures import transparent_opaque_interface  # noqa: E402
from vdbmat.io import VolumeIOError, write_volume  # noqa: E402
from vdbmat.optics import (  # noqa: E402
    map_material_volume_to_optical,
    phase0_provisional_mapping,
)


def _optical_volume():
    return map_material_volume_to_optical(
        transparent_opaque_interface().volume, phase0_provisional_mapping()
    )


def _write_optical_zarr(path: Path) -> None:
    write_volume(path, _optical_volume())


def _write_material_zarr(path: Path) -> None:
    write_volume(path, transparent_opaque_interface().volume)


def _write_bundle(path: Path, *, run_id: str = "run-test0000000001") -> None:
    path.mkdir(parents=True)
    _write_material_zarr(path / "material.zarr")
    _write_optical_zarr(path / "optical.zarr")
    manifest = {
        "schema": {"name": "vdbmat.run", "version": "1.0.0"},
        "run_id": run_id,
        "stages": [{"name": "load", "status": "ok"}],
    }
    (path / "run.json").write_text(json.dumps(manifest))


def test_scan_finds_bundle_and_standalone_optical(tmp_path: Path) -> None:
    root = tmp_path
    _write_bundle(root / "bundle_a")
    _write_optical_zarr(root / "standalone.zarr")

    catalog = scan_input_catalog(root)

    assert [item.root_relative for item in catalog] == [
        "bundle_a",
        "standalone.zarr",
    ]
    assert catalog[0].kind is InputKind.RUN_BUNDLE
    assert catalog[0].optical_zarr == root / "bundle_a" / "optical.zarr"
    assert catalog[1].kind is InputKind.OPTICAL_ZARR
    assert catalog[1].optical_zarr == root / "standalone.zarr"


def test_scan_excludes_material_only_and_unrelated_entries(tmp_path: Path) -> None:
    root = tmp_path
    _write_material_zarr(root / "material_only.zarr")
    (root / "run_json_without_optical").mkdir()
    (root / "run_json_without_optical" / "run.json").write_text("{}")
    (root / "unrelated_dir").mkdir()
    (root / "unrelated_file.txt").write_text("hello")

    catalog = scan_input_catalog(root)

    assert catalog == []


def test_scan_does_not_descend_into_bundle_or_zarr_store(tmp_path: Path) -> None:
    root = tmp_path
    _write_bundle(root / "bundle_a")
    # A decoy directory named like a zarr store, planted inside the bundle.
    (root / "bundle_a" / "nested.zarr").mkdir()
    _write_optical_zarr(root / "standalone.zarr")
    (root / "standalone.zarr" / "decoy_bundle").mkdir()
    (root / "standalone.zarr" / "decoy_bundle" / "run.json").write_text("{}")

    catalog = scan_input_catalog(root)

    assert [item.root_relative for item in catalog] == [
        "bundle_a",
        "standalone.zarr",
    ]


def test_scan_orders_candidates_by_relative_path(tmp_path: Path) -> None:
    root = tmp_path
    _write_optical_zarr(root / "zeta.zarr")
    _write_bundle(root / "alpha_bundle")
    _write_optical_zarr(root / "middle" / "nested.zarr")

    catalog = scan_input_catalog(root)

    assert [item.root_relative for item in catalog] == [
        "alpha_bundle",
        "middle/nested.zarr",
        "zeta.zarr",
    ]


def test_scan_excludes_symlink_escaping_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.zarr"
    _write_optical_zarr(outside)
    (root / "escape.zarr").symlink_to(outside, target_is_directory=True)

    catalog = scan_input_catalog(root)

    assert catalog == []


def test_resolve_candidate_rejects_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.zarr"
    _write_optical_zarr(outside)

    with pytest.raises(InputCatalogError):
        resolve_candidate(root, outside)


def test_resolve_candidate_rejects_symlink_escaping_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.zarr"
    _write_optical_zarr(outside)
    link = root / "escape.zarr"
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(InputCatalogError):
        resolve_candidate(root, link)


def test_resolve_candidate_rejects_missing_path(tmp_path: Path) -> None:
    root = tmp_path
    with pytest.raises(InputCatalogError):
        resolve_candidate(root, root / "does_not_exist.zarr")


def test_resolve_candidate_accepts_relative_bundle_path(tmp_path: Path) -> None:
    root = tmp_path
    _write_bundle(root / "bundle_a")

    candidate = resolve_candidate(root, Path("bundle_a"))

    assert candidate == InputCandidate(
        kind=InputKind.RUN_BUNDLE,
        root_relative="bundle_a",
        path=root.resolve() / "bundle_a",
        optical_zarr=root.resolve() / "bundle_a" / "optical.zarr",
    )


def test_resolve_input_root_uses_explicit_cli_root(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    initial = tmp_path / "somewhere" / "optical.zarr"

    assert resolve_input_root(explicit, initial) == explicit.resolve()


def test_resolve_input_root_rejects_non_directory_cli_root(tmp_path: Path) -> None:
    not_a_dir = tmp_path / "file.txt"
    not_a_dir.write_text("x")

    with pytest.raises(InputCatalogError):
        resolve_input_root(not_a_dir, tmp_path / "initial.zarr")


def test_resolve_input_root_defaults_to_initial_parent_for_zarr(
    tmp_path: Path,
) -> None:
    initial = tmp_path / "runs" / "standalone.zarr"
    _write_optical_zarr(initial)

    assert resolve_input_root(None, initial) == (tmp_path / "runs").resolve()


def test_resolve_input_root_defaults_to_bundle_root_parent(tmp_path: Path) -> None:
    bundle = tmp_path / "runs" / "bundle_a"
    _write_bundle(bundle)

    assert resolve_input_root(None, bundle) == (tmp_path / "runs").resolve()


def test_describe_candidate_for_standalone_optical(tmp_path: Path) -> None:
    root = tmp_path
    _write_optical_zarr(root / "standalone.zarr")
    candidate = resolve_candidate(root, Path("standalone.zarr"))

    summary = describe_candidate(candidate)

    assert summary.kind is InputKind.OPTICAL_ZARR
    assert summary.schema_name == "vdbmat.volume"
    assert summary.shape_zyx == (2, 4, 6)
    assert summary.run_id is None


def test_describe_candidate_for_bundle_includes_run_id(tmp_path: Path) -> None:
    root = tmp_path
    _write_bundle(root / "bundle_a", run_id="run-abc123")
    candidate = resolve_candidate(root, Path("bundle_a"))

    summary = describe_candidate(candidate)

    assert summary.kind is InputKind.RUN_BUNDLE
    assert summary.run_id == "run-abc123"
    assert summary.shape_zyx == (2, 4, 6)


def test_scan_excludes_broken_zarr_named_store(tmp_path: Path) -> None:
    root = tmp_path
    zarr_path = root / "broken.zarr"
    _write_optical_zarr(zarr_path)
    import zarr as zarr_module

    group = zarr_module.open_group(zarr_path, mode="r+")
    group.attrs["vdbmat"] = {"broken": True}

    catalog = scan_input_catalog(root)

    assert catalog == []


def test_describe_candidate_raises_for_scanned_but_corrupted_optical_store(
    tmp_path: Path,
) -> None:
    """Scan only checks the declared asset type; full validation is deferred.

    A store whose manifest still declares ``optical-property`` but has an
    inconsistent array declaration is enumerated by scan (cheap check) and
    only fails once ``describe_candidate`` runs ``inspect_volume`` against
    it, matching the plan's "validate at Load/Rebuild time" boundary.
    """
    root = tmp_path
    zarr_path = root / "corrupted.zarr"
    _write_optical_zarr(zarr_path)
    import zarr as zarr_module

    group = zarr_module.open_group(zarr_path, mode="r+")
    manifest = dict(group.attrs["vdbmat"])
    manifest["arrays"] = dict(manifest["arrays"])
    manifest["arrays"]["g"] = dict(manifest["arrays"]["g"])
    manifest["arrays"]["g"]["unit"] = "not-a-real-unit"
    group.attrs["vdbmat"] = manifest

    catalog = scan_input_catalog(root)
    assert [item.root_relative for item in catalog] == ["corrupted.zarr"]

    candidate = resolve_candidate(root, Path("corrupted.zarr"))
    with pytest.raises(VolumeIOError):
        describe_candidate(candidate)
