from __future__ import annotations

import sys
from pathlib import Path

import pytest

DEMO_DIR = Path(__file__).parents[1] / "examples" / "pipeline_run" / "demo"
sys.path.insert(0, str(DEMO_DIR))

from mitsuba_stage import (  # noqa: E402
    RenderSettings,
    StageConfig,
    StageConfigError,
    stage_config_from_dict,
    stage_config_to_dict,
)


def _document(
    version: str, render: dict[str, object] | None = None
) -> dict[str, object]:
    document: dict[str, object] = {
        "format": "vdbmat.stage-config",
        "format_version": version,
    }
    if render is not None:
        document["render"] = render
    return document


def test_render_settings_default_max_depth_is_eight() -> None:
    assert RenderSettings().max_depth == 8


@pytest.mark.parametrize("max_depth", [1, 8, 64])
def test_render_settings_accepts_positive_integer_max_depth(max_depth: int) -> None:
    assert RenderSettings(max_depth=max_depth).max_depth == max_depth


@pytest.mark.parametrize("max_depth", [0, -1, 1.5, True])
def test_render_settings_rejects_invalid_max_depth(max_depth: object) -> None:
    with pytest.raises(StageConfigError, match=r"render\.max_depth"):
        RenderSettings(max_depth=max_depth)  # type: ignore[arg-type]


def test_legacy_stage_config_supplies_default_max_depth() -> None:
    config = stage_config_from_dict(_document("1.0.0", {"spp": 32}))

    assert config.render.spp == 32
    assert config.render.max_depth == 8


def test_legacy_stage_config_rejects_max_depth_as_unknown() -> None:
    with pytest.raises(StageConfigError, match=r"unknown keys.*max_depth"):
        stage_config_from_dict(_document("1.0.0", {"max_depth": 16}))


@pytest.mark.parametrize("render", [{"max_depth": 16}, {}])
def test_current_stage_config_reads_explicit_or_default_max_depth(
    render: dict[str, object],
) -> None:
    config = stage_config_from_dict(_document("1.2.0", render))

    assert config.render.max_depth == render.get("max_depth", 8)


def test_render_settings_default_denoise_is_false() -> None:
    assert RenderSettings().denoise is False


@pytest.mark.parametrize("denoise", [True, False])
def test_render_settings_accepts_bool_denoise(denoise: bool) -> None:
    assert RenderSettings(denoise=denoise).denoise is denoise


@pytest.mark.parametrize("denoise", [0, 1, "true", None])
def test_render_settings_rejects_invalid_denoise(denoise: object) -> None:
    with pytest.raises(StageConfigError, match=r"render\.denoise"):
        RenderSettings(denoise=denoise)  # type: ignore[arg-type]


def test_legacy_stage_config_supplies_default_denoise() -> None:
    config = stage_config_from_dict(_document("1.0.0", {"spp": 32}))

    assert config.render.denoise is False


def test_legacy_stage_config_rejects_denoise_as_unknown() -> None:
    with pytest.raises(StageConfigError, match=r"unknown keys.*denoise"):
        stage_config_from_dict(_document("1.0.0", {"denoise": True}))


def test_v1_1_0_stage_config_supplies_default_denoise() -> None:
    config = stage_config_from_dict(_document("1.1.0", {"max_depth": 16}))

    assert config.render.denoise is False


def test_v1_1_0_stage_config_rejects_denoise_as_unknown() -> None:
    with pytest.raises(StageConfigError, match=r"unknown keys.*denoise"):
        stage_config_from_dict(_document("1.1.0", {"denoise": True}))


@pytest.mark.parametrize("render", [{"denoise": True}, {}])
def test_current_stage_config_reads_explicit_or_default_denoise(
    render: dict[str, object],
) -> None:
    config = stage_config_from_dict(_document("1.2.0", render))

    assert config.render.denoise == render.get("denoise", False)


def test_serializer_writes_current_version_and_all_render_fields() -> None:
    config = StageConfig(
        render=RenderSettings(width=320, height=240, spp=16, max_depth=12, denoise=True)
    )

    document = stage_config_to_dict(config)

    assert document["format_version"] == "1.2.0"
    assert document["render"] == {
        "width": 320,
        "height": 240,
        "spp": 16,
        "max_depth": 12,
        "denoise": True,
    }
    assert stage_config_from_dict(document) == config


@pytest.mark.parametrize("version", ["0.9.0", "1.3.0", "2.0.0"])
def test_stage_config_rejects_unknown_version(version: str) -> None:
    with pytest.raises(StageConfigError, match="format_version must be one of"):
        stage_config_from_dict(_document(version))


def test_cli_max_depth_override_wins_and_none_preserves_preset() -> None:
    preset = StageConfig(render=RenderSettings(max_depth=20))

    assert preset.with_cli_overrides().render.max_depth == 20
    assert preset.with_cli_overrides(max_depth=6).render.max_depth == 6
