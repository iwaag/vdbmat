from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

DEMO_DIR = Path(__file__).parents[1] / "examples" / "pipeline_run" / "demo"
sys.path.insert(0, str(DEMO_DIR))

from mitsuba_stage import (  # noqa: E402
    BackdropSettings,
    RenderSettings,
    StageConfig,
    stage_config_to_dict,
)
from mitsuba_stage_presets import (  # noqa: E402
    PresetCandidate,
    PresetCatalogError,
    describe_preset,
    load_preset,
    resolve_preset,
    resolve_preset_root,
    scan_preset_catalog,
    stage_config_digest,
)


def _write_document(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _stage_document(**updates: object) -> dict[str, object]:
    document = stage_config_to_dict(StageConfig())
    document.update(updates)
    return document


def test_scan_lists_only_stage_json_in_relative_path_order(tmp_path: Path) -> None:
    _write_document(tmp_path / "zeta.stage.json", _stage_document())
    _write_document(tmp_path / "alpha" / "nested.stage.json", _stage_document())
    _write_document(tmp_path / "ignored.json", _stage_document())
    (tmp_path / "not-a-file.stage.json").mkdir()

    catalog = scan_preset_catalog(tmp_path)

    assert [candidate.root_relative for candidate in catalog] == [
        "alpha/nested.stage.json",
        "zeta.stage.json",
    ]


def test_scan_keeps_invalid_document_for_deferred_diagnostics(tmp_path: Path) -> None:
    path = tmp_path / "broken.stage.json"
    path.write_text("{", encoding="utf-8")

    catalog = scan_preset_catalog(tmp_path)

    assert [candidate.root_relative for candidate in catalog] == [
        "broken.stage.json"
    ]
    with pytest.raises(PresetCatalogError, match="not valid JSON"):
        describe_preset(catalog[0])


def test_scan_excludes_file_symlink_escaping_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.stage.json"
    _write_document(outside, _stage_document())
    (root / "escape.stage.json").symlink_to(outside)

    assert scan_preset_catalog(root) == []


def test_scan_and_resolve_accept_file_symlink_staying_inside_root(
    tmp_path: Path,
) -> None:
    target = tmp_path / "actual.stage.json"
    _write_document(target, _stage_document())
    link = tmp_path / "alias.stage.json"
    link.symlink_to(target)

    catalog = scan_preset_catalog(tmp_path)
    candidate = resolve_preset(tmp_path, Path("alias.stage.json"))

    assert [item.root_relative for item in catalog] == [
        "actual.stage.json",
        "alias.stage.json",
    ]
    assert candidate == PresetCandidate(
        root_relative="alias.stage.json", path=target.resolve()
    )


@pytest.mark.parametrize(
    ("user_path", "message"),
    [
        (Path("/tmp/absolute.stage.json"), "must be relative"),
        (Path("../parent.stage.json"), "parent traversal"),
        (Path("wrong.json"), "must end with"),
        (Path("."), "must not be empty"),
    ],
)
def test_resolve_rejects_invalid_lexical_paths(
    tmp_path: Path, user_path: Path, message: str
) -> None:
    with pytest.raises(PresetCatalogError, match=message):
        resolve_preset(tmp_path, user_path)


def test_resolve_rejects_missing_preset(tmp_path: Path) -> None:
    with pytest.raises(PresetCatalogError, match="does not exist"):
        resolve_preset(tmp_path, Path("missing.stage.json"))


def test_resolve_rejects_symlink_escaping_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.stage.json"
    _write_document(outside, _stage_document())
    (root / "escape.stage.json").symlink_to(outside)

    with pytest.raises(PresetCatalogError, match="outside --preset-root"):
        resolve_preset(root, Path("escape.stage.json"))


def test_load_rejects_wrong_format_and_version(tmp_path: Path) -> None:
    wrong_format = tmp_path / "wrong-format.stage.json"
    wrong_version = tmp_path / "wrong-version.stage.json"
    _write_document(
        wrong_format,
        {"format": "not-stage", "format_version": "1.1.0"},
    )
    _write_document(
        wrong_version,
        {"format": "vdbmat.stage-config", "format_version": "9.0.0"},
    )

    with pytest.raises(PresetCatalogError, match="format must be"):
        load_preset(resolve_preset(tmp_path, Path(wrong_format.name)))
    with pytest.raises(PresetCatalogError, match="format_version"):
        load_preset(resolve_preset(tmp_path, Path(wrong_version.name)))


def test_semantic_digest_normalizes_legacy_partial_and_current_full(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy.stage.json"
    partial = tmp_path / "partial.stage.json"
    current = tmp_path / "current.stage.json"
    _write_document(
        legacy,
        {
            "format": "vdbmat.stage-config",
            "format_version": "1.0.0",
            "render": {"width": 512, "height": 512, "spp": 128},
        },
    )
    _write_document(
        partial,
        {"format": "vdbmat.stage-config", "format_version": "1.1.0"},
    )
    _write_document(current, stage_config_to_dict(StageConfig()))

    configs = [
        load_preset(resolve_preset(tmp_path, Path(path.name)))
        for path in (legacy, partial, current)
    ]

    assert configs == [StageConfig(), StageConfig(), StageConfig()]
    assert len({stage_config_digest(config) for config in configs}) == 1


def test_semantic_digest_ignores_json_formatting_and_key_order(
    tmp_path: Path,
) -> None:
    compact = tmp_path / "compact.stage.json"
    pretty = tmp_path / "pretty.stage.json"
    document = stage_config_to_dict(StageConfig())
    compact.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")
    pretty.write_text(
        json.dumps(dict(reversed(list(document.items()))), indent=4),
        encoding="utf-8",
    )

    first = load_preset(resolve_preset(tmp_path, Path(compact.name)))
    second = load_preset(resolve_preset(tmp_path, Path(pretty.name)))

    assert stage_config_digest(first) == stage_config_digest(second)


def test_semantic_digest_changes_with_effective_value() -> None:
    original = StageConfig()
    changed = StageConfig(
        backdrop=BackdropSettings(checker_scale=original.backdrop.checker_scale + 1)
    )

    assert stage_config_digest(original) != stage_config_digest(changed)


def test_describe_reports_effective_defaults_and_overrides(tmp_path: Path) -> None:
    path = tmp_path / "partial.stage.json"
    _write_document(
        path,
        {
            "format": "vdbmat.stage-config",
            "format_version": "1.1.0",
            "render": {"width": 320, "max_depth": 12},
            "camera": {
                "azimuth_deg": -30.0,
                "elevation_deg": 20.0,
                "distance_factor": 4.0,
                "fov_deg": 40.0,
            },
        },
    )

    summary = describe_preset(resolve_preset(tmp_path, Path(path.name)))

    assert summary.format_version == "1.1.0"
    assert (summary.width, summary.height, summary.spp, summary.max_depth) == (
        320,
        512,
        128,
        12,
    )
    assert summary.camera_override is True
    assert summary.backlight_override is False
    assert summary.digest.startswith("sha256:")


def test_resolve_root_prefers_explicit_directory(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    initial = tmp_path / "initial" / "preset.stage.json"

    assert resolve_preset_root(explicit, initial) == explicit.resolve()


def test_resolve_root_defaults_to_initial_preset_parent(tmp_path: Path) -> None:
    initial = tmp_path / "presets" / "preset.stage.json"
    _write_document(initial, _stage_document())

    assert resolve_preset_root(None, initial) == initial.parent.resolve()


def test_resolve_root_defaults_to_builtin_presets() -> None:
    assert resolve_preset_root(None, None) == (DEMO_DIR / "presets").resolve()


def test_resolve_root_rejects_non_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "file"
    file_path.write_text("x", encoding="utf-8")

    with pytest.raises(PresetCatalogError, match="not a directory"):
        resolve_preset_root(file_path, None)


def test_render_change_alters_digest() -> None:
    assert stage_config_digest(StageConfig()) != stage_config_digest(
        StageConfig(render=RenderSettings(max_depth=16))
    )
