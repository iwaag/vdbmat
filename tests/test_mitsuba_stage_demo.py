from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from vdbmat.fixtures import homogeneous_transparent
from vdbmat.optics import map_material_volume_to_optical, phase0_provisional_mapping

DEMO_DIR = Path(__file__).parents[1] / "examples" / "pipeline_run" / "demo"
sys.path.insert(0, str(DEMO_DIR))

import mitsuba_stage_demo  # noqa: E402


def test_parse_args_accepts_max_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["mitsuba_stage_demo", "input.zarr", "output.png", "--max-depth", "14"],
    )

    args = mitsuba_stage_demo._parse_args()

    assert args.max_depth == 14


@pytest.mark.parametrize(
    ("version", "render", "cli_max_depth", "expected"),
    [
        ("1.0.0", {"spp": 128}, None, 8),
        ("1.1.0", {"max_depth": 20}, None, 20),
        ("1.1.0", {"max_depth": 20}, 14, 14),
    ],
)
def test_main_propagates_effective_max_depth_to_export_and_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    version: str,
    render: dict[str, object],
    cli_max_depth: int | None,
    expected: int,
) -> None:
    preset_path = tmp_path / "preset.stage.json"
    preset_path.write_text(
        json.dumps(
            {
                "format": "vdbmat.stage-config",
                "format_version": version,
                "render": render,
            }
        ),
        encoding="utf-8",
    )
    output_png = tmp_path / "render.png"
    args = argparse.Namespace(
        optical_zarr=tmp_path / "optical.zarr",
        output_png=output_png,
        stage_config=preset_path,
        width=None,
        height=None,
        spp=None,
        max_depth=cli_max_depth,
        checker_scale=None,
        variant="llvm_ad_rgb",
    )
    volume = map_material_volume_to_optical(
        homogeneous_transparent().volume, phase0_provisional_mapping()
    )
    captured: dict[str, Any] = {}

    def prepare(_volume: object, output: Path, config: object) -> SimpleNamespace:
        captured["output"] = output
        captured["config"] = config
        return SimpleNamespace(scene_dict={})

    class FakeUtil:
        @staticmethod
        def write_bitmap(_path: str, _image: object, *, write_async: bool) -> None:
            assert write_async is False

    class FakeMitsuba:
        util = FakeUtil()

        @staticmethod
        def load_dict(scene_dict: dict[str, object]) -> dict[str, object]:
            return scene_dict

        @staticmethod
        def render(_scene: object, *, seed: int, spp: int) -> np.ndarray:
            assert seed >= 0
            assert spp == 128
            return np.zeros((2, 2, 3), dtype=np.float32)

    monkeypatch.setattr(mitsuba_stage_demo, "_parse_args", lambda: args)
    monkeypatch.setattr(mitsuba_stage_demo, "read_volume", lambda _path: volume)
    monkeypatch.setattr(mitsuba_stage_demo, "prepare_mitsuba_scene", prepare)
    monkeypatch.setattr(
        mitsuba_stage_demo, "_load_mitsuba", lambda _variant: FakeMitsuba()
    )
    monkeypatch.setattr(mitsuba_stage_demo, "apply_stage", lambda *_args: None)

    mitsuba_stage_demo.main()

    config = captured["config"]
    assert config.max_depth == expected
    assert captured["output"] == tmp_path / "render_scene"
    assert f"PIXELSTATS max_depth={expected}" in capsys.readouterr().out
