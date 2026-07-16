from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from vdbmat.exporters.mitsuba import MitsubaExportConfig, prepare_mitsuba_scene
from vdbmat.fixtures import homogeneous_transparent, transparent_opaque_interface
from vdbmat.io import read_material_label_manifest, write_volume
from vdbmat.optics import map_material_volume_to_optical, phase0_provisional_mapping

DEMO_DIR = Path(__file__).parents[2] / "examples" / "pipeline_run" / "demo"
sys.path.insert(0, str(DEMO_DIR))

import mitsuba_stage_demo  # noqa: E402
import mitsuba_stage_viewer  # noqa: E402
from mitsuba_stage import (  # noqa: E402
    BacklightOverride,
    CameraOverride,
    RenderSettings,
    StageConfig,
    apply_stage,
)
from mitsuba_stage_viewer import (  # noqa: E402
    InputLoadError,
    StageCore,
    TraversedPreviewScene,
    _session_work_dir,
)

mi = pytest.importorskip("mitsuba")

pytestmark = pytest.mark.mitsuba


def _fresh_render(prepared, volume, config: StageConfig, spp: int) -> np.ndarray:  # type: ignore[no-untyped-def]
    scene_dict = dict(prepared.scene_dict)
    scene_dict["integrator"] = {
        **scene_dict["integrator"],
        "max_depth": config.render.max_depth,
    }
    apply_stage(mi, scene_dict, volume.geometry, config)
    return np.asarray(mi.render(mi.load_dict(scene_dict), seed=17, spp=spp))


def test_traverse_updates_match_fresh_rebuild_and_structure_falls_back(
    tmp_path: Path,
) -> None:
    volume = map_material_volume_to_optical(
        homogeneous_transparent().volume, phase0_provisional_mapping()
    )
    render = MitsubaExportConfig(width=12, height=12, spp=2, seed=17)
    prepared = prepare_mitsuba_scene(volume, tmp_path / "scene", render)
    config = StageConfig(
        camera=CameraOverride(), backlight=BacklightOverride()
    ).with_cli_overrides(width=12, height=12, spp=2)
    preview = TraversedPreviewScene(mi, prepared, volume.geometry, config, seed=17)

    configs = [
        replace(
            config,
            backdrop=replace(
                config.backdrop,
                color0=(0.2, 0.1, 0.1),
                checker_scale=5,
                distance_factor=2.5,
            ),
        ),
        replace(
            config,
            floor=replace(
                config.floor,
                color1=(0.4, 0.5, 0.1),
                drop_factor=0.2,
                scale_factor=5.0,
            ),
        ),
        replace(
            config,
            key_light=replace(
                config.key_light,
                direction=(-1.0, -1.0, 3.0),
                distance_factor=4.0,
                scale_factor=0.8,
                radiance=(8.0, 4.0, 2.0),
            ),
        ),
        replace(config, backlight=BacklightOverride((2.0, 1.0, 1.0))),
        replace(config, camera=CameraOverride(-40.0, 35.0, 4.5, 42.0)),
    ]
    for changed in configs:
        traversed, route = preview.render(changed, spp=2)
        fresh = _fresh_render(prepared, volume, changed, spp=2)
        assert route == "traverse"
        np.testing.assert_allclose(np.asarray(traversed), fresh, rtol=0.0, atol=1e-5)
        config = changed

    structural = replace(config, floor=replace(config.floor, pattern="solid"))
    _, route = preview.render(structural, spp=2)
    assert route == "rebuild"
    continuous_again = replace(
        structural, floor=replace(structural.floor, color0=(0.1, 0.2, 0.3))
    )
    traversed, route = preview.render(continuous_again, spp=2)
    fresh = _fresh_render(prepared, volume, continuous_again, spp=2)
    assert route == "traverse"
    np.testing.assert_allclose(np.asarray(traversed), fresh, rtol=0.0, atol=1e-5)


def test_max_depth_rebuilds_preview_then_continuous_updates_traverse(
    tmp_path: Path,
) -> None:
    volume = map_material_volume_to_optical(
        homogeneous_transparent().volume, phase0_provisional_mapping()
    )
    render = MitsubaExportConfig(width=12, height=12, spp=2, seed=17, max_depth=8)
    prepared = prepare_mitsuba_scene(volume, tmp_path / "scene", render)
    config = StageConfig().with_cli_overrides(width=12, height=12, spp=2)
    preview = TraversedPreviewScene(mi, prepared, volume.geometry, config, seed=17)

    depth_changed = replace(config, render=replace(config.render, max_depth=16))
    rebuilt, route = preview.render(depth_changed, spp=2)
    fresh = _fresh_render(prepared, volume, depth_changed, spp=2)

    assert route == "rebuild"
    assert preview._stage_dict(depth_changed)["integrator"]["max_depth"] == 16
    assert prepared.scene_dict["integrator"]["max_depth"] == 8
    np.testing.assert_allclose(np.asarray(rebuilt), fresh, rtol=0.0, atol=1e-5)

    continuous = replace(
        depth_changed,
        key_light=replace(depth_changed.key_light, radiance=(7.0, 6.0, 5.0)),
    )
    traversed, route = preview.render(continuous, spp=2)
    fresh = _fresh_render(prepared, volume, continuous, spp=2)

    assert route == "traverse"
    np.testing.assert_allclose(np.asarray(traversed), fresh, rtol=0.0, atol=1e-5)


def test_stage_core_final_reprepare_uses_max_depth(tmp_path: Path) -> None:
    volume = map_material_volume_to_optical(
        homogeneous_transparent().volume, phase0_provisional_mapping()
    )
    optical_zarr = tmp_path / "optical.zarr"
    write_volume(optical_zarr, volume)
    initial = StageConfig(
        render=RenderSettings(width=12, height=12, spp=1, max_depth=8)
    )
    core = StageCore(
        optical_zarr,
        tmp_path / "work",
        preview_size=12,
        preview_spp=1,
        initial=initial,
    )
    changed = replace(initial, render=replace(initial.render, max_depth=14))

    _pixels, preview_stats, route = core.render_preview(changed)
    stats = core.render_final(changed, tmp_path / "final.png")
    session_dir = _session_work_dir(tmp_path / "work", 0, optical_zarr)
    summary = json.loads(
        (session_dir / "final_scene" / "scene-summary.json").read_text(
            encoding="utf-8"
        )
    )

    assert route == "rebuild"
    assert "max_depth=14" in preview_stats
    assert summary["render"]["max_depth"] == 14
    assert "max_depth=14" in stats


def test_swap_session_uses_fresh_work_dir_and_advances_generation(
    tmp_path: Path,
) -> None:
    volume = map_material_volume_to_optical(
        homogeneous_transparent().volume, phase0_provisional_mapping()
    )
    optical_zarr = tmp_path / "optical.zarr"
    write_volume(optical_zarr, volume)
    initial = StageConfig(render=RenderSettings(width=8, height=8, spp=1))
    core = StageCore(
        optical_zarr,
        tmp_path / "work",
        preview_size=8,
        preview_spp=1,
        initial=initial,
    )
    first_session_dir = core._session.work_dir
    assert core.session_generation == 0
    assert first_session_dir == _session_work_dir(tmp_path / "work", 0, optical_zarr)

    second_session = core._build_session(optical_zarr, initial)
    core.swap_session(second_session)

    assert core.session_generation == 1
    assert core._session is second_session
    assert second_session.work_dir == _session_work_dir(
        tmp_path / "work", 1, optical_zarr
    )
    assert second_session.work_dir != first_session_dir
    # Reloading the same input never reuses or overwrites the old artefacts.
    assert (first_session_dir / "preview_scene").exists()
    assert (second_session.work_dir / "preview_scene").exists()

    # The swapped-in session renders through the normal StageCore API.
    pixels, _stats, _route = core.render_preview(initial)
    assert pixels.shape == (8, 8, 3)


def _write_two_inputs(root: Path) -> tuple[Path, Path]:
    root.mkdir()
    volume_a = map_material_volume_to_optical(
        homogeneous_transparent().volume, phase0_provisional_mapping()
    )
    optical_a = root / "a.zarr"
    write_volume(optical_a, volume_a)
    volume_b = map_material_volume_to_optical(
        transparent_opaque_interface().volume, phase0_provisional_mapping()
    )
    optical_b = root / "b.zarr"
    write_volume(optical_b, volume_b)
    return optical_a, optical_b


def test_load_input_swaps_to_new_session_using_current_stage_settings(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    optical_a, optical_b = _write_two_inputs(root)
    initial = StageConfig(
        render=RenderSettings(width=8, height=8, spp=1, max_depth=11)
    )
    core = StageCore(
        optical_a, tmp_path / "work", preview_size=8, preview_spp=1, initial=initial
    )
    first_session = core._session

    stages: list[str] = []
    session = core.load_input(
        root, Path("b.zarr"), initial, smoke_spp=1, on_stage=stages.append
    )

    assert stages == ["validate", "prepare", "load", "smoke", "swap"]
    assert core.session_generation == 1
    assert core._session is session
    assert session is not first_session
    assert session.optical_zarr == optical_b.resolve()
    assert session.work_dir == _session_work_dir(tmp_path / "work", 1, optical_b)

    # Current stage/render settings (including max_depth) carried over.
    pixels, stats, _route = core.render_preview(initial)
    assert pixels.shape == (8, 8, 3)
    assert "max_depth=11" in stats

    # A later non-structural change to the new session still traverses.
    tweaked = replace(
        initial, key_light=replace(initial.key_light, radiance=(3.0, 2.0, 1.0))
    )
    _pixels2, _stats2, route2 = core.render_preview(tweaked)
    assert route2 == "traverse"


def test_load_input_validate_failure_preserves_current_session(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    optical_a, _optical_b = _write_two_inputs(root)
    outside = tmp_path / "outside.zarr"
    write_volume(
        outside,
        map_material_volume_to_optical(
            transparent_opaque_interface().volume, phase0_provisional_mapping()
        ),
    )
    initial = StageConfig(render=RenderSettings(width=8, height=8, spp=1))
    core = StageCore(
        optical_a, tmp_path / "work", preview_size=8, preview_spp=1, initial=initial
    )
    original_session = core._session

    stages: list[str] = []
    with pytest.raises(InputLoadError) as excinfo:
        core.load_input(root, outside, initial, smoke_spp=1, on_stage=stages.append)

    assert excinfo.value.stage == "validate"
    assert stages == ["validate"]
    assert core.session_generation == 0
    assert core._session is original_session

    pixels, _stats, _route = core.render_preview(initial)
    assert pixels.shape == (8, 8, 3)


def test_load_input_prepare_failure_discards_new_session_and_preserves_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    optical_a, optical_b = _write_two_inputs(root)
    initial = StageConfig(render=RenderSettings(width=8, height=8, spp=1))
    core = StageCore(
        optical_a, tmp_path / "work", preview_size=8, preview_spp=1, initial=initial
    )
    original_session = core._session

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("forced prepare failure")

    monkeypatch.setattr(mitsuba_stage_viewer, "prepare_mitsuba_scene", _boom)

    stages: list[str] = []
    with pytest.raises(InputLoadError) as excinfo:
        core.load_input(
            root, Path("b.zarr"), initial, smoke_spp=1, on_stage=stages.append
        )

    assert excinfo.value.stage == "prepare"
    assert stages == ["validate", "prepare"]
    assert core.session_generation == 0
    assert core._session is original_session
    assert not _session_work_dir(tmp_path / "work", 1, optical_b).exists()


def test_load_input_load_failure_removes_prepared_work_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    optical_a, optical_b = _write_two_inputs(root)
    initial = StageConfig(render=RenderSettings(width=8, height=8, spp=1))
    core = StageCore(
        optical_a, tmp_path / "work", preview_size=8, preview_spp=1, initial=initial
    )
    original_session = core._session

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("forced load failure")

    monkeypatch.setattr(mitsuba_stage_viewer, "TraversedPreviewScene", _boom)

    with pytest.raises(InputLoadError) as excinfo:
        core.load_input(root, Path("b.zarr"), initial, smoke_spp=1)

    assert excinfo.value.stage == "load"
    assert core.session_generation == 0
    assert core._session is original_session
    failed_dir = _session_work_dir(tmp_path / "work", 1, optical_b)
    assert not failed_dir.exists()


def test_load_input_smoke_failure_removes_prepared_work_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    optical_a, optical_b = _write_two_inputs(root)
    initial = StageConfig(render=RenderSettings(width=8, height=8, spp=1))
    core = StageCore(
        optical_a, tmp_path / "work", preview_size=8, preview_spp=1, initial=initial
    )
    original_session = core._session

    def _boom(self: object, _config: StageConfig, _spp: int) -> None:
        raise RuntimeError("forced smoke failure")

    monkeypatch.setattr(TraversedPreviewScene, "render", _boom)

    with pytest.raises(InputLoadError) as excinfo:
        core.load_input(root, Path("b.zarr"), initial, smoke_spp=1)

    assert excinfo.value.stage == "smoke"
    assert core.session_generation == 0
    assert core._session is original_session
    failed_dir = _session_work_dir(tmp_path / "work", 1, optical_b)
    assert not failed_dir.exists()


@pytest.mark.parametrize("max_depth", [8, 16])
def test_saved_preset_viewer_final_matches_headless_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    max_depth: int,
) -> None:
    repository = Path(__file__).parents[2]
    material = read_material_label_manifest(
        repository
        / "examples/pipeline_run/inputs/nested_material_cube.voxels.json"
    )
    volume = map_material_volume_to_optical(material, phase0_provisional_mapping())
    optical_zarr = tmp_path / "optical.zarr"
    write_volume(optical_zarr, volume)
    stage = StageConfig(
        render=RenderSettings(
            width=12,
            height=12,
            spp=2,
            max_depth=max_depth,
        )
    )
    preset = tmp_path / "saved.stage.json"
    viewer_png = tmp_path / "viewer.png"
    headless_png = tmp_path / "headless.png"
    StageCore.save_preset(stage, preset)
    core = StageCore(
        optical_zarr,
        tmp_path / "viewer-work",
        preview_size=12,
        preview_spp=1,
        initial=stage,
    )

    core.render_final(stage, viewer_png)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mitsuba_stage_demo",
            str(optical_zarr),
            str(headless_png),
            "--stage-config",
            str(preset),
            "--variant",
            "llvm_ad_rgb",
        ],
    )
    mitsuba_stage_demo.main()

    viewer_pixels = np.asarray(mi.Bitmap(str(viewer_png)))
    headless_pixels = np.asarray(mi.Bitmap(str(headless_png)))
    headless_summary = json.loads(
        (tmp_path / "headless_scene/scene-summary.json").read_text(encoding="utf-8")
    )

    assert np.array_equal(viewer_pixels, headless_pixels)
    assert headless_summary["render"]["max_depth"] == max_depth
    assert f"PIXELSTATS max_depth={max_depth}" in capsys.readouterr().out
