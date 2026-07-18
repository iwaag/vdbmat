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


def test_parse_args_denoise_defaults_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["mitsuba_stage_demo", "input.zarr", "output.png"])

    args = mitsuba_stage_demo._parse_args()

    assert args.denoise is None


def test_parse_args_accepts_denoise_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["mitsuba_stage_demo", "input.zarr", "output.png", "--denoise"],
    )

    args = mitsuba_stage_demo._parse_args()

    assert args.denoise is True


def test_parse_args_accepts_no_denoise_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["mitsuba_stage_demo", "input.zarr", "output.png", "--no-denoise"],
    )

    args = mitsuba_stage_demo._parse_args()

    assert args.denoise is False


def test_parse_args_accepts_seed_in_legacy_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["mitsuba_stage_demo", "input.zarr", "output.png", "--seed", "7"],
    )

    args = mitsuba_stage_demo._parse_args()

    assert args.seed == 7


def test_parse_args_rejects_negative_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["mitsuba_stage_demo", "input.zarr", "output.png", "--seed", "-1"],
    )

    with pytest.raises(SystemExit):
        mitsuba_stage_demo._parse_args()


def test_parse_args_session_mode_requires_input_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mitsuba_stage_demo",
            "--session",
            "s.json",
            "--output-png",
            "out.png",
        ],
    )

    with pytest.raises(SystemExit):
        mitsuba_stage_demo._parse_args()


def test_parse_args_session_mode_requires_output_png(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mitsuba_stage_demo",
            "--session",
            "s.json",
            "--input-root",
            "root",
        ],
    )

    with pytest.raises(SystemExit):
        mitsuba_stage_demo._parse_args()


def test_parse_args_session_mode_rejects_positional_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mitsuba_stage_demo",
            "input.zarr",
            "output.png",
            "--session",
            "s.json",
            "--input-root",
            "root",
            "--output-png",
            "out.png",
        ],
    )

    with pytest.raises(SystemExit):
        mitsuba_stage_demo._parse_args()


def test_parse_args_session_mode_rejects_stage_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mitsuba_stage_demo",
            "--session",
            "s.json",
            "--input-root",
            "root",
            "--output-png",
            "out.png",
            "--stage-config",
            "preset.stage.json",
        ],
    )

    with pytest.raises(SystemExit):
        mitsuba_stage_demo._parse_args()


@pytest.mark.parametrize(
    "flag", ["--width", "--height", "--spp", "--max-depth", "--checker-scale"]
)
def test_parse_args_session_mode_rejects_render_overrides(
    monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mitsuba_stage_demo",
            "--session",
            "s.json",
            "--input-root",
            "root",
            "--output-png",
            "out.png",
            flag,
            "4",
        ],
    )

    with pytest.raises(SystemExit):
        mitsuba_stage_demo._parse_args()


@pytest.mark.parametrize("flag", ["--denoise", "--no-denoise"])
def test_parse_args_session_mode_rejects_denoise_overrides(
    monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mitsuba_stage_demo",
            "--session",
            "s.json",
            "--input-root",
            "root",
            "--output-png",
            "out.png",
            flag,
        ],
    )

    with pytest.raises(SystemExit):
        mitsuba_stage_demo._parse_args()


def test_parse_args_legacy_mode_rejects_session_only_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mitsuba_stage_demo",
            "input.zarr",
            "output.png",
            "--input-root",
            "root",
        ],
    )

    with pytest.raises(SystemExit):
        mitsuba_stage_demo._parse_args()


def test_parse_args_session_mode_accepts_mapping_roots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mitsuba_stage_demo",
            "--session",
            "s.json",
            "--input-root",
            "inputs",
            "--mapping-root",
            "mappings",
            "--mapping-work-root",
            "derived",
            "--output-png",
            "out.png",
        ],
    )

    args = mitsuba_stage_demo._parse_args()

    assert args.mapping_root == Path("mappings")
    assert args.mapping_work_root == Path("derived")


@pytest.mark.parametrize("flag", ["--mapping-root", "--mapping-work-root"])
def test_parse_args_session_mode_requires_mapping_roots_as_pair(
    monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mitsuba_stage_demo",
            "--session",
            "s.json",
            "--input-root",
            "inputs",
            flag,
            "value",
            "--output-png",
            "out.png",
        ],
    )

    with pytest.raises(SystemExit):
        mitsuba_stage_demo._parse_args()


def test_resolve_mapping_session_regenerates_and_verifies_derived_optical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mitsuba_stage import StageConfig

    mapping_ref = SimpleNamespace(path="tinted.optical-mapping.json")
    session = SimpleNamespace(preset=None, mapping=mapping_ref)
    stage = StageConfig()
    resolved = SimpleNamespace(
        session=session,
        optical_zarr=tmp_path / "inputs/bundle/optical.zarr",
        input_candidate=SimpleNamespace(path=tmp_path / "inputs/bundle"),
        mapping_candidate=SimpleNamespace(path=tmp_path / "mappings/tinted.json"),
        stage_config=stage,
        variant="llvm_ad_rgb",
        seed=42,
    )
    derived = SimpleNamespace(
        optical_zarr=tmp_path / "derived/bundle/optical.zarr",
        mapping_digest="sha256:" + "1" * 64,
        reused=False,
    )
    args = argparse.Namespace(
        session=tmp_path / "viewer.session.json",
        input_root=tmp_path / "inputs",
        preset_root=None,
        mapping_root=tmp_path / "mappings",
        mapping_work_root=tmp_path / "derived",
        session_output_png=tmp_path / "out.png",
        variant=None,
        seed=None,
    )
    verified: list[Path] = []

    monkeypatch.setattr(
        mitsuba_stage_demo, "viewer_session_from_json", lambda _path: session
    )
    monkeypatch.setattr(
        mitsuba_stage_demo,
        "resolve_input_root",
        lambda _cli, _initial: tmp_path / "inputs",
    )
    monkeypatch.setattr(
        mitsuba_stage_demo,
        "resolve_preset_root",
        lambda _cli, _initial: tmp_path / "presets",
    )
    monkeypatch.setattr(
        mitsuba_stage_demo,
        "resolve_mapping_root",
        lambda _cli: tmp_path / "mappings",
    )
    monkeypatch.setattr(
        mitsuba_stage_demo,
        "resolve_viewer_session",
        lambda *_args, **_kwargs: resolved,
    )
    monkeypatch.setattr(
        mitsuba_stage_demo,
        "regenerate_optical",
        lambda *_args, **_kwargs: derived,
    )
    monkeypatch.setattr(
        mitsuba_stage_demo,
        "verify_derived_optical",
        lambda _resolved, path: verified.append(path),
    )

    optical_zarr, output_png, actual_stage, variant, seed = (
        mitsuba_stage_demo._resolve_session(args)
    )

    assert optical_zarr == derived.optical_zarr
    assert output_png == args.session_output_png
    assert actual_stage is stage
    assert variant == "llvm_ad_rgb"
    assert seed == 42
    assert verified == [derived.optical_zarr]


def test_resolve_mapping_session_requires_mapping_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = SimpleNamespace(preset=None, mapping=SimpleNamespace(path="mapping.json"))
    args = argparse.Namespace(
        session=tmp_path / "viewer.session.json",
        input_root=tmp_path / "inputs",
        preset_root=None,
        mapping_root=None,
        mapping_work_root=None,
        session_output_png=tmp_path / "out.png",
        variant=None,
        seed=None,
    )
    monkeypatch.setattr(
        mitsuba_stage_demo, "viewer_session_from_json", lambda _path: session
    )
    monkeypatch.setattr(
        mitsuba_stage_demo,
        "resolve_input_root",
        lambda _cli, _initial: tmp_path / "inputs",
    )
    monkeypatch.setattr(
        mitsuba_stage_demo,
        "resolve_preset_root",
        lambda _cli, _initial: tmp_path / "presets",
    )

    with pytest.raises(
        mitsuba_stage_demo.ViewerSessionError,
        match="requires --mapping-root and --mapping-work-root",
    ):
        mitsuba_stage_demo._resolve_session(args)


def test_headless_mapping_work_root_must_not_overlap_input_root(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "inputs"

    with pytest.raises(mitsuba_stage_demo.ViewerSessionError, match="must not overlap"):
        mitsuba_stage_demo._require_disjoint_roots(input_root / "derived", input_root)


def test_main_session_mode_replays_resolved_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from mitsuba_stage import StageConfig

    args = argparse.Namespace(
        optical_zarr=None,
        output_png=None,
        stage_config=None,
        width=None,
        height=None,
        spp=None,
        max_depth=None,
        checker_scale=None,
        variant=None,
        seed=None,
        session=tmp_path / "viewer.session.json",
        input_root=tmp_path / "inputs",
        preset_root=None,
        mapping_root=None,
        mapping_work_root=None,
        session_output_png=tmp_path / "replay.png",
    )
    stage = StageConfig()
    resolved = SimpleNamespace(
        session=SimpleNamespace(mapping=None),
        optical_zarr=tmp_path / "inputs" / "case" / "optical.zarr",
        stage_config=stage,
        variant="llvm_ad_rgb",
        seed=42,
    )
    args.session.write_text("{}", encoding="utf-8")
    captured: dict[str, Any] = {}

    def fake_render_stage(
        optical_zarr: Path,
        output_png: Path,
        stage_config: object,
        variant: str,
        seed: int,
    ) -> np.ndarray:
        captured["optical_zarr"] = optical_zarr
        captured["output_png"] = output_png
        captured["stage_config"] = stage_config
        captured["variant"] = variant
        captured["seed"] = seed
        return np.zeros((1, 1, 3), dtype=np.float32)

    monkeypatch.setattr(mitsuba_stage_demo, "_parse_args", lambda: args)
    monkeypatch.setattr(
        mitsuba_stage_demo,
        "viewer_session_from_json",
        lambda _path: SimpleNamespace(preset=None, mapping=None),
    )
    monkeypatch.setattr(
        mitsuba_stage_demo,
        "resolve_input_root",
        lambda _cli, _initial: tmp_path / "inputs",
    )
    monkeypatch.setattr(
        mitsuba_stage_demo,
        "resolve_preset_root",
        lambda _cli, _initial: tmp_path / "presets",
    )
    monkeypatch.setattr(
        mitsuba_stage_demo, "resolve_viewer_session", lambda *_a, **_k: resolved
    )
    monkeypatch.setattr(mitsuba_stage_demo, "render_stage", fake_render_stage)

    mitsuba_stage_demo.main()

    assert captured["optical_zarr"] == resolved.optical_zarr
    assert captured["output_png"] == args.session_output_png
    assert captured["stage_config"] is stage
    assert captured["variant"] == "llvm_ad_rgb"
    assert captured["seed"] == 42

    from vdbmat.pipeline import sha256_file

    out = capsys.readouterr().out
    assert f"RENDER session={args.session} digest={sha256_file(args.session)}" in out


def test_main_session_mode_rejects_variant_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mitsuba_stage import StageConfig

    args = argparse.Namespace(
        optical_zarr=None,
        output_png=None,
        stage_config=None,
        width=None,
        height=None,
        spp=None,
        max_depth=None,
        checker_scale=None,
        variant="cuda_ad_rgb",
        seed=None,
        session=tmp_path / "viewer.session.json",
        input_root=tmp_path / "inputs",
        preset_root=None,
        mapping_root=None,
        mapping_work_root=None,
        session_output_png=tmp_path / "replay.png",
    )
    resolved = SimpleNamespace(
        session=SimpleNamespace(mapping=None),
        optical_zarr=tmp_path / "inputs" / "case" / "optical.zarr",
        stage_config=StageConfig(),
        variant="llvm_ad_rgb",
        seed=42,
    )

    monkeypatch.setattr(mitsuba_stage_demo, "_parse_args", lambda: args)
    monkeypatch.setattr(
        mitsuba_stage_demo,
        "viewer_session_from_json",
        lambda _path: SimpleNamespace(preset=None, mapping=None),
    )
    monkeypatch.setattr(
        mitsuba_stage_demo,
        "resolve_input_root",
        lambda _cli, _initial: tmp_path / "inputs",
    )
    monkeypatch.setattr(
        mitsuba_stage_demo,
        "resolve_preset_root",
        lambda _cli, _initial: tmp_path / "presets",
    )
    monkeypatch.setattr(
        mitsuba_stage_demo, "resolve_viewer_session", lambda *_a, **_k: resolved
    )

    with pytest.raises(SystemExit, match="does not match session variant"):
        mitsuba_stage_demo.main()


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
        denoise=None,
        variant="llvm_ad_rgb",
        seed=None,
        session=None,
        input_root=None,
        preset_root=None,
        mapping_root=None,
        mapping_work_root=None,
        session_output_png=None,
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
    assert f"max_depth={expected}" in capsys.readouterr().out
