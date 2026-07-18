from __future__ import annotations

import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vdbmat.core.volumes import OpticalPropertyVolume
from vdbmat.exporters.mitsuba import MitsubaExportConfig, prepare_mitsuba_scene
from vdbmat.fixtures import (
    homogeneous_transparent,
    transparent_opaque_interface,
    write_phase1_fixtures,
)
from vdbmat.io import read_material_label_manifest, write_volume
from vdbmat.io.zarr import read_volume
from vdbmat.optics import (
    load_optical_mapping,
    map_material_volume_to_optical,
    optical_mapping_to_json_dict,
    phase0_provisional_mapping,
)
from vdbmat.pipeline import PipelineConfig, run_pipeline, zarr_store_sha256

DEMO_DIR = Path(__file__).parents[2] / "examples" / "pipeline_run" / "demo"
sys.path.insert(0, str(DEMO_DIR))

import mitsuba_stage_core  # noqa: E402
import mitsuba_stage_demo  # noqa: E402
import mitsuba_stage_viewer  # noqa: E402
from mitsuba_stage import (  # noqa: E402
    BacklightOverride,
    CameraOverride,
    RenderSettings,
    StageConfig,
    apply_stage,
    stage_config_to_dict,
)
from mitsuba_stage_core import (  # noqa: E402
    DenoiseVariantError,
    InputLoadError,
    InputSession,
    StageCore,
    TraversedPreviewScene,
    _pixel_stats,
    _session_work_dir,
)
from mitsuba_stage_inputs import (  # noqa: E402
    InputCandidate,
    InputKind,
    resolve_candidate,
)
from mitsuba_stage_mappings import resolve_mapping_candidate  # noqa: E402
from mitsuba_stage_presets import load_preset, resolve_preset  # noqa: E402
from mitsuba_stage_regen import RegenError, regenerate_optical  # noqa: E402
from mitsuba_viewer_session import (  # noqa: E402
    SessionMappingRef,
    ViewerSessionError,
    create_viewer_session,
    resolve_viewer_session,
    viewer_session_from_json,
    write_viewer_session,
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
        (session_dir / "final_scene" / "scene-summary.json").read_text(encoding="utf-8")
    )

    assert route == "rebuild"
    assert "max_depth=14" in preview_stats
    assert summary["render"]["max_depth"] == 14
    assert "max_depth=14" in stats


def test_denoise_off_final_render_writes_no_raw_sidecar(tmp_path: Path) -> None:
    volume = map_material_volume_to_optical(
        homogeneous_transparent().volume, phase0_provisional_mapping()
    )
    optical_zarr = tmp_path / "optical.zarr"
    write_volume(optical_zarr, volume)
    stage = StageConfig(render=RenderSettings(width=8, height=8, spp=1, denoise=False))
    core = StageCore(
        optical_zarr,
        tmp_path / "work",
        preview_size=8,
        preview_spp=1,
        initial=stage,
    )
    output_png = tmp_path / "final.png"

    stats = core.render_final(stage, output_png)

    assert output_png.exists()
    assert not output_png.with_name("final.raw.png").exists()
    assert "denoise=optix" not in stats


def test_denoise_requires_cuda_variant_for_final_and_settled_preview(
    tmp_path: Path,
) -> None:
    volume = map_material_volume_to_optical(
        homogeneous_transparent().volume, phase0_provisional_mapping()
    )
    optical_zarr = tmp_path / "optical.zarr"
    write_volume(optical_zarr, volume)
    stage = StageConfig(render=RenderSettings(width=8, height=8, spp=1, denoise=True))
    core = StageCore(
        optical_zarr,
        tmp_path / "work",
        preview_size=8,
        preview_spp=1,
        initial=replace(stage, render=replace(stage.render, denoise=False)),
        variant="llvm_ad_rgb",
    )

    with pytest.raises(DenoiseVariantError):
        core.render_final(stage, tmp_path / "final.png")
    with pytest.raises(DenoiseVariantError):
        core.render_preview(stage, spp=None)

    # Interactive preview (explicit spp) is never denoised, so it is exempt
    # from the variant guard even when the config asks for denoise.
    pixels, stats, _route = core.render_preview(stage, spp=1)
    assert pixels.shape == (8, 8, 3)
    assert "denoise=optix" not in stats


@pytest.mark.skipif(
    "cuda_ad_rgb" not in mi.variants(), reason="requires a CUDA-capable host"
)
def test_denoise_cuda_final_render_writes_raw_and_denoised_matching_stats(
    tmp_path: Path,
) -> None:
    volume = map_material_volume_to_optical(
        homogeneous_transparent().volume, phase0_provisional_mapping()
    )
    optical_zarr = tmp_path / "optical.zarr"
    write_volume(optical_zarr, volume)
    stage = StageConfig(render=RenderSettings(width=8, height=8, spp=4, denoise=True))
    core = StageCore(
        optical_zarr,
        tmp_path / "work",
        preview_size=8,
        preview_spp=4,
        initial=replace(stage, render=replace(stage.render, denoise=False)),
        variant="cuda_ad_rgb",
    )
    output_png = tmp_path / "final.png"
    raw_png = tmp_path / "final.raw.png"

    # Deterministic (same session/config/seed) render of the raw pixels,
    # independent of render_final's internal bitmap writing, so the
    # comparison below isn't distorted by the lossy float->8-bit PNG
    # round-trip write_bitmap performs.
    session = core._session
    core._ensure_final(session, stage.render)
    direct_image = core._render(
        session.base_final, session.volume, stage, stage.render.spp, session.seed
    )
    expected_raw_stats = _pixel_stats(
        np.asarray(direct_image, dtype=np.float32), stage.render.max_depth
    )

    stats = core.render_final(stage, output_png)

    assert output_png.exists()
    assert raw_png.exists()
    assert stats == f"{expected_raw_stats} denoise=optix"
    raw_pixels = np.asarray(mi.Bitmap(str(raw_png)), dtype=np.float32)
    denoised_pixels = np.asarray(mi.Bitmap(str(output_png)), dtype=np.float32)
    assert not np.array_equal(raw_pixels, denoised_pixels)


@pytest.mark.skipif(
    "cuda_ad_rgb" not in mi.variants(), reason="requires a CUDA-capable host"
)
def test_denoise_cuda_applies_only_to_settled_preview(tmp_path: Path) -> None:
    volume = map_material_volume_to_optical(
        homogeneous_transparent().volume, phase0_provisional_mapping()
    )
    optical_zarr = tmp_path / "optical.zarr"
    write_volume(optical_zarr, volume)
    stage = StageConfig(render=RenderSettings(width=8, height=8, spp=4, denoise=True))
    core = StageCore(
        optical_zarr,
        tmp_path / "work",
        preview_size=8,
        preview_spp=4,
        initial=replace(stage, render=replace(stage.render, denoise=False)),
        variant="cuda_ad_rgb",
    )

    _pixels, interactive_stats, _route = core.render_preview(stage, spp=1)
    assert "denoise=optix" not in interactive_stats

    _pixels, settled_stats, _route = core.render_preview(stage, spp=None)
    assert "denoise=optix" in settled_stats


def test_loaded_stage_preset_drives_preview_and_final_render(tmp_path: Path) -> None:
    volume = map_material_volume_to_optical(
        homogeneous_transparent().volume, phase0_provisional_mapping()
    )
    optical_zarr = tmp_path / "optical.zarr"
    write_volume(optical_zarr, volume)
    initial = StageConfig(render=RenderSettings(width=8, height=8, spp=1, max_depth=8))
    applied = replace(
        initial,
        render=replace(initial.render, max_depth=15),
        camera=CameraOverride(-35.0, 22.0, 4.2, 40.0),
    )
    preset_root = tmp_path / "presets"
    preset_root.mkdir()
    preset = preset_root / "applied.stage.json"
    preset.write_text(json.dumps(stage_config_to_dict(applied)))
    loaded = load_preset(resolve_preset(preset_root, Path(preset.name)))
    core = StageCore(
        optical_zarr,
        tmp_path / "work",
        preview_size=8,
        preview_spp=1,
        initial=initial,
    )

    pixels, preview_stats, route = core.render_preview(loaded)
    final_stats = core.render_final(loaded, tmp_path / "preset-final.png")
    summary = json.loads(
        (core._session.work_dir / "final_scene" / "scene-summary.json").read_text(
            encoding="utf-8"
        )
    )

    assert pixels.shape == (8, 8, 3)
    assert route == "rebuild"
    assert "max_depth=15" in preview_stats
    assert "max_depth=15" in final_stats
    assert summary["render"]["max_depth"] == 15


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
    # Reloading the same input never reuses or overwrites the old artefacts,
    # and a successful swap discards the replaced session's own directory
    # (Phase 5 Step 2 cleanup rule) rather than letting it accumulate.
    assert not first_session_dir.exists()
    assert (second_session.work_dir / "preview_scene").exists()

    # The swapped-in session renders through the normal StageCore API.
    pixels, _stats, _route = core.render_preview(initial)
    assert pixels.shape == (8, 8, 3)


def test_preview_and_final_render_unaffected_by_discarded_old_session_dir(
    tmp_path: Path,
) -> None:
    """Neither preview nor final render holds a lazy reference to an old dir.

    Preview scenes are ``mi.load_dict()``-ed (geometry read into memory)
    at session-build time, and a final render always re-``load_dict()``s
    from the *current* session's own ``final_scene`` directory (never a
    stale one) — see :meth:`StageCore.swap_session`'s docstring. This test
    exercises both through two real ``load_input`` swaps (each of which
    discards the just-replaced directory) and confirms preview/final still
    render correctly off the surviving, current session.
    """
    root = tmp_path / "root"
    optical_a, _optical_b = _write_two_inputs(root)
    stage = StageConfig(render=RenderSettings(width=8, height=8, spp=1))
    core = StageCore(
        optical_a, tmp_path / "work", preview_size=8, preview_spp=1, initial=stage
    )
    first_dir = core._session.work_dir

    second_session = core.load_input(root, Path("b.zarr"), stage, smoke_spp=1)
    second_dir = second_session.work_dir
    assert not first_dir.exists()

    third_session = core.load_input(root, Path("a.zarr"), stage, smoke_spp=1)
    assert not second_dir.exists()
    assert core._session is third_session

    pixels, stats, _route = core.render_preview(stage)
    assert pixels.shape == (8, 8, 3)
    assert "max_depth" in stats

    final_png = tmp_path / "final.png"
    core.render_final(stage, final_png)
    assert final_png.exists()
    summary = json.loads(
        (third_session.work_dir / "final_scene" / "scene-summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["render"]["max_depth"] == stage.render.max_depth


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


def _write_bundle_input(root: Path, name: str, volume: object) -> Path:
    bundle = root / name
    bundle.mkdir()
    write_volume(bundle / "optical.zarr", volume)
    (bundle / "run.json").write_text(
        json.dumps(
            {
                "schema": {"name": "vdbmat.run", "version": "1.0.0"},
                "run_id": f"run-{name}",
                "stages": [],
            }
        ),
        encoding="utf-8",
    )
    return bundle


def _write_pipeline_bundle(root: Path, name: str) -> Path:
    """Publish a real canonical run bundle at ``root / name``.

    Unlike ``_write_bundle_input``, this goes through the actual
    ``run_pipeline()`` orchestration so the bundle carries a
    ``source/*.voxels.json`` and a ``run.json`` whose declared digests
    ``mitsuba_stage_regen.regenerate_optical()`` can verify and re-run.
    Deterministic Phase 1 fixture inputs are written to a sibling directory
    (outside ``root``) so the input catalog never sees them as candidates.
    """
    fixtures_dir = root.parent / f"{root.name}-pipeline-fixtures"
    if not fixtures_dir.exists():
        write_phase1_fixtures(fixtures_dir)
    config = PipelineConfig(
        input_kind="direct-voxel",
        input_path="window_coupon.voxels.json",
        output_path=str(root / name),
        mapping_name="phase0-provisional-materials-v1",
    )
    result = run_pipeline(config, base_dir=str(fixtures_dir))
    return result.output_path


def _write_nested_material_cube_bundle(root: Path, name: str) -> Path:
    repository = Path(__file__).parents[2]
    inputs = repository / "examples/pipeline_run/inputs"
    config = PipelineConfig(
        input_kind="direct-voxel",
        input_path="nested_material_cube.voxels.json",
        output_path=str(root / name),
        mapping_name="phase0-provisional-materials-v1",
    )
    return run_pipeline(config, base_dir=str(inputs)).output_path


def _write_mapping_document(path: Path, *, tint: bool = False) -> None:
    document = optical_mapping_to_json_dict(phase0_provisional_mapping())
    if tint:
        for material in document["materials"]:
            if material["material_id"] == 1:
                material["sigma_a_rgb_per_m"] = [9.0, 9.0, 9.0]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def test_load_input_round_trip_renders_and_separates_final_artifacts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    volume_a = map_material_volume_to_optical(
        homogeneous_transparent().volume, phase0_provisional_mapping()
    )
    bundle_a = _write_bundle_input(root, "bundle-a", volume_a)
    volume_b = map_material_volume_to_optical(
        transparent_opaque_interface().volume, phase0_provisional_mapping()
    )
    optical_b = root / "standalone-b.zarr"
    write_volume(optical_b, volume_b)
    stage = StageConfig(render=RenderSettings(width=8, height=8, spp=1, max_depth=13))
    work = tmp_path / "work"
    core = StageCore(
        bundle_a / "optical.zarr",
        work,
        preview_size=8,
        preview_spp=1,
        initial=stage,
    )

    def _assert_final_scene_max_depth(session: InputSession) -> None:
        summary = json.loads(
            (session.work_dir / "final_scene" / "scene-summary.json").read_text(
                encoding="utf-8"
            )
        )
        assert summary["render"]["max_depth"] == 13

    sessions = [core._session]
    work_dirs = {core._session.work_dir}
    # The initial session's own final_scene is prepared during StageCore
    # construction; check it before the first swap discards it.
    _assert_final_scene_max_depth(core._session)

    for selection in (Path("standalone-b.zarr"), Path("bundle-a")):
        session = core.load_input(root, selection, stage, smoke_spp=1)
        sessions.append(session)
        work_dirs.add(session.work_dir)
        pixels, stats, _route = core.render_preview(stage)
        assert pixels.shape == (8, 8, 3)
        assert "max_depth=13" in stats
        core.render_final(stage, tmp_path / f"final-{len(sessions)}.png")
        _assert_final_scene_max_depth(session)
        # A successful swap discards the replaced session's own work
        # directory (Phase 5 Step 2 cleanup rule) instead of letting it
        # accumulate; every earlier session's directory is gone by now.
        for previous in sessions[:-1]:
            assert not previous.work_dir.exists()

    assert core.session_generation == 2
    assert [session.optical_zarr for session in sessions] == [
        bundle_a / "optical.zarr",
        optical_b.resolve(),
        (bundle_a / "optical.zarr").resolve(),
    ]
    assert len(work_dirs) == 3


def test_prepare_input_session_is_transactional_and_preserves_live_seed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    optical_a, optical_b = _write_two_inputs(root)
    stage = StageConfig(render=RenderSettings(width=8, height=8, spp=1))
    core = StageCore(
        optical_a,
        tmp_path / "work",
        preview_size=8,
        preview_spp=1,
        initial=stage,
        seed=37,
    )
    original = core.current_session

    prepared = core.prepare_input_session(root, Path("b.zarr"), stage, smoke_spp=1)

    assert core.current_session is original
    assert core.session_generation == 0
    assert prepared.optical_zarr == optical_b.resolve()
    assert prepared.seed == 37

    core.swap_session(prepared)
    reloaded = core.load_input(root, Path("a.zarr"), stage, smoke_spp=1)

    assert core.session_generation == 2
    assert reloaded.seed == 37


def test_viewer_session_load_verifies_then_commits_input_config_and_seed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    optical_a, optical_b = _write_two_inputs(root)
    initial = StageConfig(render=RenderSettings(width=8, height=8, spp=1))
    restored = StageConfig(
        render=RenderSettings(width=8, height=8, spp=1, max_depth=13),
        camera=CameraOverride(azimuth_deg=-25.0),
    )
    session = create_viewer_session(
        resolve_candidate(root, Path("a.zarr")),
        restored,
        "llvm_ad_rgb",
        91,
    )
    session_path = tmp_path / "saved.session.json"
    write_viewer_session(session_path, session)

    startup_args = mitsuba_stage_viewer._parse_args(
        [
            "--session",
            str(session_path),
            "--input-root",
            str(root),
            "--variant",
            "llvm_ad_rgb",
            "--seed",
            "91",
        ]
    )
    startup = mitsuba_stage_viewer._resolve_viewer_startup(startup_args, tmp_path)
    assert startup.initial_input == optical_a.resolve()
    assert startup.initial_config == restored
    assert startup.variant == "llvm_ad_rgb"
    assert startup.seed == 91
    assert startup.session_root == tmp_path.resolve()

    core = StageCore(
        optical_b,
        tmp_path / "work",
        preview_size=8,
        preview_spp=1,
        initial=initial,
    )

    class Binder:
        def __init__(self) -> None:
            self.config = initial

        def current(self) -> StageConfig:
            return self.config

        def replace_config(self, config: StageConfig) -> None:
            self.config = config

    app = mitsuba_stage_viewer.ViewerApp.__new__(mitsuba_stage_viewer.ViewerApp)
    app.core = core
    app.input_root = root
    app.preset_root = tmp_path
    app.mapping_root = tmp_path
    app.mapping_work_root = tmp_path / "derived"
    app.interactive_spp = 1
    app.binder = Binder()
    app._current_selection = "b.zarr"
    app._committed_derivation = None
    app.applied_preset = None
    app.input_dropdown = SimpleNamespace(value="b.zarr")
    app.preset_dropdown = SimpleNamespace(options=(), value="")
    app.mapping_dropdown = SimpleNamespace(
        options=(mitsuba_stage_viewer._AS_IS_MAPPING,),
        value=mitsuba_stage_viewer._AS_IS_MAPPING,
    )

    corrupted_path = tmp_path / "corrupted.session.json"
    document = json.loads(session_path.read_text(encoding="utf-8"))
    document["input"]["optical_sha256"] = "sha256:" + "0" * 64
    corrupted_path.write_text(json.dumps(document), encoding="utf-8")
    original_session = core.current_session
    with pytest.raises(Exception, match="input optical digest mismatch"):
        app._load_session_transaction(corrupted_path, lambda _stage: None)
    assert core.current_session is original_session
    assert core.session_generation == 0
    assert app.binder.current() == initial
    assert app._current_selection == "b.zarr"

    stages: list[str] = []
    app._load_session_transaction(session_path, stages.append)

    assert stages == [
        "parse",
        "resolve",
        "verify",
        "prepare",
        "load",
        "smoke",
        "commit",
    ]
    assert core.session_generation == 1
    assert core.current_session.optical_zarr == optical_a.resolve()
    assert core.current_session.seed == 91
    assert app.binder.current() == restored
    assert app._current_selection == "a.zarr"
    core.render_final(restored, tmp_path / "restored.png")
    summary = json.loads(
        (
            core.current_session.work_dir / "final_scene" / "scene-summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["render"]["seed"] == 91


def test_viewer_app_session_startup_builds_core_from_resolved_input(
    tmp_path: Path,
) -> None:
    """``ViewerApp(--session ...)`` must load the session's input, not None.

    Regression test: the constructor used to pass the CLI's raw
    ``args.optical_zarr`` (always ``None`` in session mode, since the
    positional argument is forbidden alongside ``--session``) to
    ``StageCore`` instead of the already-resolved ``startup.initial_input``,
    crashing with a ``TypeError`` before any GUI wiring happened.
    """
    root = tmp_path / "root"
    optical_a, _optical_b = _write_two_inputs(root)
    stage = StageConfig(render=RenderSettings(width=8, height=8, spp=1))
    session = create_viewer_session(
        resolve_candidate(root, Path("a.zarr")), stage, "llvm_ad_rgb", 5
    )
    session_path = tmp_path / "startup.session.json"
    write_viewer_session(session_path, session)

    args = mitsuba_stage_viewer._parse_args(
        [
            "--session",
            str(session_path),
            "--input-root",
            str(root),
            "--port",
            "0",
            "--work-dir",
            str(tmp_path / "work"),
        ]
    )
    app = mitsuba_stage_viewer.ViewerApp(args)
    try:
        assert app.core.current_session.optical_zarr == optical_a.resolve()
        assert app.core.current_session.seed == 5
        assert app._current_selection == "a.zarr"
    finally:
        app.server.stop()


def test_viewer_app_startup_sweeps_stale_inputs_but_keeps_derived(
    tmp_path: Path,
) -> None:
    """A restart with the same ``--work-dir`` cleans up the prior process's
    ``inputs/`` leftovers (Phase 5 Step 2 cleanup rule (b)) before building
    the initial session, without touching ``derived/`` or other work-dir
    files.
    """
    root = tmp_path / "root"
    optical_a, _optical_b = _write_two_inputs(root)
    work_dir = tmp_path / "work"

    stale_dir = work_dir / "inputs" / "999-leftover"
    stale_dir.mkdir(parents=True)
    (stale_dir / "marker.txt").write_text("stale", encoding="utf-8")
    derived_marker = work_dir / "derived" / "keepme.txt"
    derived_marker.parent.mkdir(parents=True)
    derived_marker.write_text("keep", encoding="utf-8")

    args = mitsuba_stage_viewer._parse_args(
        [
            str(optical_a),
            "--input-root",
            str(root),
            "--port",
            "0",
            "--work-dir",
            str(work_dir),
        ]
    )
    app = mitsuba_stage_viewer.ViewerApp(args)
    try:
        assert not stale_dir.exists()
        assert derived_marker.exists()
        # The initial session's own directory (built after the sweep) is
        # untouched by it.
        assert app.core.current_session.work_dir.exists()
    finally:
        app.server.stop()


def test_load_input_rejects_invalid_candidates_without_changing_live_preview(
    tmp_path: Path,
) -> None:
    import zarr as zarr_module

    root = tmp_path / "root"
    optical_a, _optical_b = _write_two_inputs(root)
    stage = StageConfig(render=RenderSettings(width=8, height=8, spp=1))
    core = StageCore(
        optical_a,
        tmp_path / "work",
        preview_size=8,
        preview_spp=1,
        initial=stage,
    )
    original_session = core._session
    original_pixels, _stats, _route = core.render_preview(stage)

    corrupted = root / "corrupted.zarr"
    write_volume(
        corrupted,
        map_material_volume_to_optical(
            transparent_opaque_interface().volume, phase0_provisional_mapping()
        ),
    )
    group = zarr_module.open_group(corrupted, mode="r+")
    manifest = dict(group.attrs["vdbmat"])
    manifest["arrays"] = dict(manifest["arrays"])
    manifest["arrays"]["g"] = dict(manifest["arrays"]["g"])
    manifest["arrays"]["g"]["unit"] = "not-a-real-unit"
    group.attrs["vdbmat"] = manifest

    material = root / "material.zarr"
    write_volume(material, transparent_opaque_interface().volume)
    outside = tmp_path / "outside.zarr"
    write_volume(outside, transparent_opaque_interface().volume)
    escaping_link = root / "escape.zarr"
    escaping_link.symlink_to(outside, target_is_directory=True)

    for selection in (Path("corrupted.zarr"), Path("material.zarr"), escaping_link):
        stages: list[str] = []
        with pytest.raises(InputLoadError) as excinfo:
            core.load_input(root, selection, stage, smoke_spp=1, on_stage=stages.append)
        assert excinfo.value.stage == "validate"
        assert stages == ["validate"]
        assert core._session is original_session
        assert core.session_generation == 0

        current_pixels, _stats, _route = core.render_preview(stage)
        np.testing.assert_array_equal(current_pixels, original_pixels)


def test_load_input_swaps_to_new_session_using_current_stage_settings(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    optical_a, optical_b = _write_two_inputs(root)
    initial = StageConfig(render=RenderSettings(width=8, height=8, spp=1, max_depth=11))
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

    monkeypatch.setattr(mitsuba_stage_core, "prepare_mitsuba_scene", _boom)

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

    monkeypatch.setattr(mitsuba_stage_core, "TraversedPreviewScene", _boom)

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
        repository / "examples/pipeline_run/inputs/nested_material_cube.voxels.json"
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
    assert f"max_depth={max_depth}" in capsys.readouterr().out


def test_saved_session_viewer_final_matches_headless_session_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "root"
    optical_a, _optical_b = _write_two_inputs(root)
    stage = StageConfig(
        render=RenderSettings(width=12, height=12, spp=2, max_depth=17),
        camera=CameraOverride(azimuth_deg=-12.0),
    )
    seed = 4242
    session = create_viewer_session(
        resolve_candidate(root, Path("a.zarr")), stage, "llvm_ad_rgb", seed
    )
    session_path = tmp_path / "saved.session.json"
    write_viewer_session(session_path, session)

    viewer_png = tmp_path / "viewer.png"
    core = StageCore(
        optical_a,
        tmp_path / "viewer-work",
        preview_size=12,
        preview_spp=1,
        initial=stage,
        seed=seed,
    )
    core.render_final(stage, viewer_png)

    headless_png = tmp_path / "headless.png"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mitsuba_stage_demo",
            "--session",
            str(session_path),
            "--input-root",
            str(root),
            "--output-png",
            str(headless_png),
        ],
    )
    mitsuba_stage_demo.main()

    viewer_pixels = np.asarray(mi.Bitmap(str(viewer_png)))
    headless_pixels = np.asarray(mi.Bitmap(str(headless_png)))
    headless_summary = json.loads(
        (tmp_path / "headless_scene/scene-summary.json").read_text(encoding="utf-8")
    )

    assert np.array_equal(viewer_pixels, headless_pixels)
    assert headless_summary["render"]["max_depth"] == 17
    assert headless_summary["render"]["seed"] == seed
    out = capsys.readouterr().out
    assert f"seed={seed}" in out
    assert "max_depth=17" in out


def test_denoise_recorded_in_saved_session_and_survives_headless_resolve(
    tmp_path: Path,
) -> None:
    """``render.denoise`` flows session save -> resolve unchanged (denoise plan Step 3).

    Full pixel-identical viewer/headless replay of a *denoised* final render
    is exercised once ``mitsuba_stage_demo.py``'s own denoise application
    lands (plan Step 4); here we only confirm the session mechanism itself
    (save, digest, resolve) carries ``render.denoise`` through unchanged,
    matching ``test_saved_session_viewer_final_matches_headless_session_replay``
    above but for the ``render.denoise`` field specifically.
    """
    root = tmp_path / "root"
    _write_two_inputs(root)
    stage = StageConfig(render=RenderSettings(width=12, height=12, spp=2, denoise=True))
    seed = 4242
    session = create_viewer_session(
        resolve_candidate(root, Path("a.zarr")), stage, "cuda_ad_rgb", seed
    )
    session_path = tmp_path / "saved.session.json"
    write_viewer_session(session_path, session)

    reloaded = viewer_session_from_json(session_path)
    resolved = resolve_viewer_session(reloaded, root)

    assert reloaded.stage_config.render.denoise is True
    assert resolved.stage_config.render.denoise is True
    assert resolved.variant == "cuda_ad_rgb"


# -- Phase 4: optical mapping selection and canonical re-generation -----------------


def test_prepare_candidate_session_builds_from_arbitrary_candidate(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    optical_a, optical_b = _write_two_inputs(root)
    stage = StageConfig(render=RenderSettings(width=8, height=8, spp=1))
    core = StageCore(
        optical_a, tmp_path / "work", preview_size=8, preview_spp=1, initial=stage
    )
    original = core.current_session

    outside_candidate = InputCandidate(
        kind=InputKind.OPTICAL_ZARR,
        root_relative="outside.zarr",
        path=optical_b,
        optical_zarr=optical_b,
    )
    prepared = core.prepare_candidate_session(outside_candidate, stage, smoke_spp=1)

    assert core.current_session is original
    assert core.session_generation == 0
    assert prepared.optical_zarr == optical_b.resolve()
    assert prepared.derivation is None


def test_regenerate_optical_and_prepare_candidate_session_render_derived_bundle(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    bundle = _write_pipeline_bundle(root, "bundle")
    mapping_root = tmp_path / "mappings"
    mapping_path = mapping_root / "tinted.optical-mapping.json"
    _write_mapping_document(mapping_path, tint=True)
    mapping_candidate = resolve_mapping_candidate(
        mapping_root, Path("tinted.optical-mapping.json")
    )

    derived = regenerate_optical(bundle, mapping_candidate, tmp_path / "derived")

    stage = StageConfig(render=RenderSettings(width=8, height=8, spp=1))
    core = StageCore(
        bundle / "optical.zarr",
        tmp_path / "work",
        preview_size=8,
        preview_spp=1,
        initial=stage,
    )
    derived_candidate = InputCandidate(
        kind=InputKind.RUN_BUNDLE,
        root_relative=derived.bundle_path.name,
        path=derived.bundle_path,
        optical_zarr=derived.optical_zarr,
    )
    session = core.prepare_candidate_session(derived_candidate, stage, smoke_spp=1)
    core.swap_session(session)

    assert core.current_session.optical_zarr == derived.optical_zarr
    pixels, _stats, _route = core.render_preview(stage)
    assert pixels.shape == (8, 8, 3)


def test_load_input_transaction_applies_mapping_and_records_derivation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    bundle = _write_pipeline_bundle(root, "bundle")
    mapping_root = tmp_path / "mappings"
    _write_mapping_document(mapping_root / "tinted.optical-mapping.json", tint=True)

    stage = StageConfig(render=RenderSettings(width=8, height=8, spp=1))
    core = StageCore(
        bundle / "optical.zarr",
        tmp_path / "work",
        preview_size=8,
        preview_spp=1,
        initial=stage,
    )
    original_session = core.current_session

    app = mitsuba_stage_viewer.ViewerApp.__new__(mitsuba_stage_viewer.ViewerApp)
    app.core = core
    app.input_root = root
    app.mapping_root = mapping_root
    app.mapping_work_root = tmp_path / "derived"
    app.interactive_spp = 1
    app._initial_input_path = (bundle / "optical.zarr").resolve()
    app._initial_sentinel = None

    stages: list[str] = []
    derivation = app._load_input_transaction(
        "bundle", "tinted.optical-mapping.json", stage, stages.append
    )

    assert stages == ["validate", "map", "prepare", "load", "smoke", "swap"]
    assert derivation is not None
    assert derivation.source_candidate.path == bundle.resolve()
    assert derivation.mapping_candidate.root_relative == "tinted.optical-mapping.json"
    assert core.current_session is not original_session
    assert core.current_session.derivation is derivation
    assert (
        core.current_session.optical_zarr == derivation.derived_bundle / "optical.zarr"
    )

    pixels, _stats, _route = core.render_preview(stage)
    assert pixels.shape == (8, 8, 3)

    reused_stages: list[str] = []
    second_derivation = app._load_input_transaction(
        "bundle", "tinted.optical-mapping.json", stage, reused_stages.append
    )
    assert reused_stages == [
        "validate",
        "map: reused cache",
        "prepare",
        "load",
        "smoke",
        "swap",
    ]
    assert second_derivation is not None
    assert second_derivation.derived_bundle == derivation.derived_bundle


def test_load_input_transaction_regenerates_after_derived_cache_wiped(
    tmp_path: Path,
) -> None:
    """A fully deleted ``--mapping-work-root`` recovers on the next Load/Rebuild.

    Phase 5 Step 2 cleanup rule (c): the derived cache is content-addressed
    and safe to delete wholesale between runs — a subsequent Load/Rebuild
    regenerates it rather than failing or silently reusing something stale.
    """
    root = tmp_path / "root"
    bundle = _write_pipeline_bundle(root, "bundle")
    mapping_root = tmp_path / "mappings"
    _write_mapping_document(mapping_root / "tinted.optical-mapping.json", tint=True)
    mapping_work_root = tmp_path / "derived"

    stage = StageConfig(render=RenderSettings(width=8, height=8, spp=1))
    core = StageCore(
        bundle / "optical.zarr",
        tmp_path / "work",
        preview_size=8,
        preview_spp=1,
        initial=stage,
    )

    app = mitsuba_stage_viewer.ViewerApp.__new__(mitsuba_stage_viewer.ViewerApp)
    app.core = core
    app.input_root = root
    app.mapping_root = mapping_root
    app.mapping_work_root = mapping_work_root
    app.interactive_spp = 1
    app._initial_input_path = (bundle / "optical.zarr").resolve()
    app._initial_sentinel = None

    first_derivation = app._load_input_transaction(
        "bundle", "tinted.optical-mapping.json", stage, lambda _stage: None
    )
    assert first_derivation is not None
    assert mapping_work_root.exists()

    shutil.rmtree(mapping_work_root)
    assert not mapping_work_root.exists()

    stages: list[str] = []
    second_derivation = app._load_input_transaction(
        "bundle", "tinted.optical-mapping.json", stage, stages.append
    )

    # No cache to reuse: "map" runs the full pipeline again, not
    # "map: reused cache".
    assert stages == ["validate", "map", "prepare", "load", "smoke", "swap"]
    assert second_derivation is not None
    assert second_derivation.mapping_digest == first_derivation.mapping_digest
    assert core.current_session.derivation is second_derivation

    pixels, _stats, _route = core.render_preview(stage)
    assert pixels.shape == (8, 8, 3)


def test_load_input_transaction_preserves_map_stage_error_and_live_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    bundle = _write_pipeline_bundle(root, "bundle")
    mapping_root = tmp_path / "mappings"
    _write_mapping_document(mapping_root / "tinted.optical-mapping.json", tint=True)
    stage = StageConfig(render=RenderSettings(width=8, height=8, spp=1))
    core = StageCore(
        bundle / "optical.zarr",
        tmp_path / "work",
        preview_size=8,
        preview_spp=1,
        initial=stage,
    )
    original_session = core.current_session

    app = mitsuba_stage_viewer.ViewerApp.__new__(mitsuba_stage_viewer.ViewerApp)
    app.core = core
    app.input_root = root
    app.mapping_root = mapping_root
    app.mapping_work_root = tmp_path / "derived"
    app.interactive_spp = 1
    app._initial_input_path = (bundle / "optical.zarr").resolve()
    app._initial_sentinel = None

    def fail_map(*_args: object, **_kwargs: object) -> None:
        raise RegenError("map", "mapping pipeline failed")

    monkeypatch.setattr(mitsuba_stage_viewer, "regenerate_optical", fail_map)
    with pytest.raises(InputLoadError, match="mapping pipeline failed") as excinfo:
        app._load_input_transaction(
            "bundle", "tinted.optical-mapping.json", stage, lambda _stage: None
        )

    assert excinfo.value.stage == "map"
    assert core.current_session is original_session
    assert core.session_generation == 0


def test_load_input_transaction_rejects_mapping_on_standalone_zarr(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    optical_a, _optical_b = _write_two_inputs(root)
    mapping_root = tmp_path / "mappings"
    _write_mapping_document(mapping_root / "tinted.optical-mapping.json")

    stage = StageConfig(render=RenderSettings(width=8, height=8, spp=1))
    core = StageCore(
        optical_a, tmp_path / "work", preview_size=8, preview_spp=1, initial=stage
    )
    original_session = core.current_session

    app = mitsuba_stage_viewer.ViewerApp.__new__(mitsuba_stage_viewer.ViewerApp)
    app.core = core
    app.input_root = root
    app.mapping_root = mapping_root
    app.mapping_work_root = tmp_path / "derived"
    app.interactive_spp = 1
    app._initial_input_path = optical_a.resolve()
    app._initial_sentinel = None

    with pytest.raises(InputLoadError) as excinfo:
        app._load_input_transaction(
            "a.zarr", "tinted.optical-mapping.json", stage, lambda _stage: None
        )

    assert excinfo.value.stage == "validate"
    assert core.current_session is original_session
    assert core.session_generation == 0


def test_capture_session_records_mapping_and_rejects_stale_mapping_file(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    bundle = _write_pipeline_bundle(root, "bundle")
    mapping_root = tmp_path / "mappings"
    mapping_path = mapping_root / "tinted.optical-mapping.json"
    _write_mapping_document(mapping_path, tint=True)

    stage = StageConfig(render=RenderSettings(width=8, height=8, spp=1))
    core = StageCore(
        bundle / "optical.zarr",
        tmp_path / "work",
        preview_size=8,
        preview_spp=1,
        initial=stage,
    )

    class Binder:
        def __init__(self) -> None:
            self.config = stage

        def current(self) -> StageConfig:
            return self.config

    app = mitsuba_stage_viewer.ViewerApp.__new__(mitsuba_stage_viewer.ViewerApp)
    app.core = core
    app.input_root = root
    app.mapping_root = mapping_root
    app.mapping_work_root = tmp_path / "derived"
    app.interactive_spp = 1
    app._initial_input_path = (bundle / "optical.zarr").resolve()
    app._initial_sentinel = None
    app.binder = Binder()
    app.applied_preset = None

    derivation = app._load_input_transaction(
        "bundle", "tinted.optical-mapping.json", stage, lambda _stage: None
    )
    app._current_selection = "bundle"
    app._committed_derivation = derivation

    session_path = tmp_path / "session.json"
    app._capture_session(session_path)

    document = json.loads(session_path.read_text(encoding="utf-8"))
    assert document["mapping"]["path"] == "tinted.optical-mapping.json"
    assert document["mapping"]["digest"] == derivation.mapping_digest
    assert document["mapping"]["derived_optical_sha256"] == zarr_store_sha256(
        derivation.derived_bundle / "optical.zarr"
    )

    edited = json.loads(mapping_path.read_text(encoding="utf-8"))
    edited["materials"][0]["sigma_a_rgb_per_m"] = [1.0, 2.0, 3.0]
    mapping_path.write_text(json.dumps(edited), encoding="utf-8")

    with pytest.raises(ViewerSessionError, match="mapping file has changed"):
        app._capture_session(tmp_path / "second.json")


def _build_mapping_session_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, str, str]:
    """Build a bundle, a tinted mapping, and its derived optical volume.

    Returns (root, mapping_root, mapping_work_root, mapping_digest,
    derived_optical_sha256) so callers can assemble a
    ``SessionMappingRef``-bearing session against a real, cache-consistent
    derived bundle.
    """
    root = tmp_path / "root"
    _write_pipeline_bundle(root, "bundle-a")
    mapping_root = tmp_path / "mappings"
    mapping_path = mapping_root / "tinted.optical-mapping.json"
    _write_mapping_document(mapping_path, tint=True)
    mapping_candidate = resolve_mapping_candidate(
        mapping_root, Path("tinted.optical-mapping.json")
    )
    mapping_digest = load_optical_mapping(mapping_path).digest
    mapping_work_root = tmp_path / "derived"
    derived = regenerate_optical(
        root / "bundle-a", mapping_candidate, mapping_work_root
    )
    derived_digest = zarr_store_sha256(derived.optical_zarr)
    return root, mapping_root, mapping_work_root, mapping_digest, derived_digest


def test_load_session_transaction_applies_mapping_and_commits_derivation(
    tmp_path: Path,
) -> None:
    root, mapping_root, mapping_work_root, mapping_digest, derived_digest = (
        _build_mapping_session_fixture(tmp_path)
    )
    optical_b = root / "b.zarr"
    write_volume(
        optical_b,
        map_material_volume_to_optical(
            transparent_opaque_interface().volume, phase0_provisional_mapping()
        ),
    )
    stage = StageConfig(render=RenderSettings(width=8, height=8, spp=1))
    session = create_viewer_session(
        resolve_candidate(root, Path("bundle-a")),
        stage,
        "llvm_ad_rgb",
        5,
        mapping=SessionMappingRef(
            path="tinted.optical-mapping.json",
            digest=mapping_digest,
            derived_optical_sha256=derived_digest,
        ),
    )
    session_path = tmp_path / "session.json"
    write_viewer_session(session_path, session)

    core = StageCore(
        optical_b,
        tmp_path / "work",
        preview_size=8,
        preview_spp=1,
        initial=stage,
    )

    class Binder:
        def __init__(self) -> None:
            self.config = stage

        def current(self) -> StageConfig:
            return self.config

        def replace_config(self, config: StageConfig) -> None:
            self.config = config

    app = mitsuba_stage_viewer.ViewerApp.__new__(mitsuba_stage_viewer.ViewerApp)
    app.core = core
    app.input_root = root
    app.preset_root = tmp_path
    app.mapping_root = mapping_root
    app.mapping_work_root = mapping_work_root
    app.interactive_spp = 1
    app.binder = Binder()
    app._current_selection = "b.zarr"
    app._committed_derivation = None
    app.applied_preset = None
    app.input_dropdown = SimpleNamespace(value="b.zarr")
    app.preset_dropdown = SimpleNamespace(options=(), value="")
    app.mapping_dropdown = SimpleNamespace(
        options=(mitsuba_stage_viewer._AS_IS_MAPPING, "tinted.optical-mapping.json"),
        value=mitsuba_stage_viewer._AS_IS_MAPPING,
    )

    stages: list[str] = []
    app._load_session_transaction(session_path, stages.append)

    assert "map: reused cache" in stages
    assert "verify" in stages
    assert core.current_session.derivation is not None
    assert core.current_session.derivation.mapping_digest == mapping_digest
    assert zarr_store_sha256(core.current_session.optical_zarr) == derived_digest
    assert app._committed_derivation is not None
    assert app._current_selection == "bundle-a"
    assert app.mapping_dropdown.value == "tinted.optical-mapping.json"

    pixels, _stats, _route = core.render_preview(stage)
    assert pixels.shape == (8, 8, 3)


def test_load_session_transaction_rejects_derived_digest_mismatch(
    tmp_path: Path,
) -> None:
    root, mapping_root, mapping_work_root, mapping_digest, _derived_digest = (
        _build_mapping_session_fixture(tmp_path)
    )
    optical_b = root / "b.zarr"
    write_volume(
        optical_b,
        map_material_volume_to_optical(
            transparent_opaque_interface().volume, phase0_provisional_mapping()
        ),
    )
    stage = StageConfig(render=RenderSettings(width=8, height=8, spp=1))
    bogus_digest = "sha256:" + "0" * 64
    session = create_viewer_session(
        resolve_candidate(root, Path("bundle-a")),
        stage,
        "llvm_ad_rgb",
        5,
        mapping=SessionMappingRef(
            path="tinted.optical-mapping.json",
            digest=mapping_digest,
            derived_optical_sha256=bogus_digest,
        ),
    )
    session_path = tmp_path / "session.json"
    write_viewer_session(session_path, session)

    core = StageCore(
        optical_b,
        tmp_path / "work",
        preview_size=8,
        preview_spp=1,
        initial=stage,
    )
    original_session = core.current_session

    class Binder:
        def __init__(self) -> None:
            self.config = stage

        def current(self) -> StageConfig:
            return self.config

        def replace_config(self, config: StageConfig) -> None:
            self.config = config

    app = mitsuba_stage_viewer.ViewerApp.__new__(mitsuba_stage_viewer.ViewerApp)
    app.core = core
    app.input_root = root
    app.preset_root = tmp_path
    app.mapping_root = mapping_root
    app.mapping_work_root = mapping_work_root
    app.interactive_spp = 1
    app.binder = Binder()
    app._current_selection = "b.zarr"
    app._committed_derivation = None
    app.applied_preset = None
    app.input_dropdown = SimpleNamespace(value="b.zarr")
    app.preset_dropdown = SimpleNamespace(options=(), value="")
    app.mapping_dropdown = SimpleNamespace(
        options=(mitsuba_stage_viewer._AS_IS_MAPPING, "tinted.optical-mapping.json"),
        value=mitsuba_stage_viewer._AS_IS_MAPPING,
    )

    with pytest.raises(ViewerSessionError, match="derived optical digest mismatch"):
        app._load_session_transaction(session_path, lambda _stage: None)

    assert core.current_session is original_session
    assert core.session_generation == 0
    assert app._committed_derivation is None
    assert app._current_selection == "b.zarr"
    assert app.mapping_dropdown.value == mitsuba_stage_viewer._AS_IS_MAPPING


def test_resolve_viewer_startup_with_mapping_session_builds_derived_input(
    tmp_path: Path,
) -> None:
    root, mapping_root, mapping_work_root, mapping_digest, derived_digest = (
        _build_mapping_session_fixture(tmp_path)
    )
    stage = StageConfig(render=RenderSettings(width=8, height=8, spp=1))
    bundle_candidate = resolve_candidate(root, Path("bundle-a"))
    session = create_viewer_session(
        bundle_candidate,
        stage,
        "llvm_ad_rgb",
        3,
        mapping=SessionMappingRef(
            path="tinted.optical-mapping.json",
            digest=mapping_digest,
            derived_optical_sha256=derived_digest,
        ),
    )
    session_path = tmp_path / "session.json"
    write_viewer_session(session_path, session)

    args = mitsuba_stage_viewer._parse_args(
        [
            "--session",
            str(session_path),
            "--input-root",
            str(root),
            "--mapping-root",
            str(mapping_root),
            "--mapping-work-root",
            str(mapping_work_root),
            "--variant",
            "llvm_ad_rgb",
            "--seed",
            "3",
        ]
    )
    startup = mitsuba_stage_viewer._resolve_viewer_startup(args, tmp_path / "work")

    assert startup.initial_derivation is not None
    assert startup.initial_input == startup.initial_derivation.derived_bundle
    assert startup.initial_derivation.source_candidate.path == bundle_candidate.path
    assert startup.initial_derivation.mapping_digest == mapping_digest
    assert startup.mapping_root == mapping_root.resolve()
    assert (
        zarr_store_sha256(startup.initial_derivation.derived_bundle / "optical.zarr")
        == derived_digest
    )


def test_resolve_viewer_startup_rejects_overlapping_mapping_work_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    optical_a, _optical_b = _write_two_inputs(root)
    args = mitsuba_stage_viewer._parse_args(
        [
            str(optical_a),
            "--input-root",
            str(root),
            "--mapping-work-root",
            str(root / "derived"),
        ]
    )

    with pytest.raises(ViewerSessionError, match="overlap"):
        mitsuba_stage_viewer._resolve_viewer_startup(args, tmp_path / "work")


def test_checked_in_mappings_switch_nested_cube_as_is_and_back(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).parents[2]
    mapping_root = repository / "examples/pipeline_run/mappings"
    root = tmp_path / "inputs"
    bundle = _write_nested_material_cube_bundle(root, "nested-cube")
    stage = StageConfig(render=RenderSettings(width=8, height=8, spp=1))
    core = StageCore(
        bundle / "optical.zarr",
        tmp_path / "viewer-work",
        preview_size=8,
        preview_spp=1,
        initial=stage,
    )
    app = mitsuba_stage_viewer.ViewerApp.__new__(mitsuba_stage_viewer.ViewerApp)
    app.core = core
    app.input_root = root
    app.mapping_root = mapping_root
    app.mapping_work_root = tmp_path / "derived"
    app.interactive_spp = 1
    app._initial_input_path = (bundle / "optical.zarr").resolve()
    app._initial_sentinel = None

    as_is_digest = zarr_store_sha256(bundle / "optical.zarr")
    tinted = app._load_input_transaction(
        "nested-cube",
        "phase0-provisional-materials-v1-tinted.optical-mapping.json",
        stage,
        lambda _stage: None,
    )
    assert tinted is not None
    tinted_digest = zarr_store_sha256(core.current_session.optical_zarr)
    assert tinted_digest != as_is_digest

    derivation = app._load_input_transaction(
        "nested-cube",
        mitsuba_stage_viewer._AS_IS_MAPPING,
        stage,
        lambda _stage: None,
    )
    assert derivation is None
    assert zarr_store_sha256(core.current_session.optical_zarr) == as_is_digest

    provisional = app._load_input_transaction(
        "nested-cube",
        "phase0-provisional-materials-v1.optical-mapping.json",
        stage,
        lambda _stage: None,
    )
    assert provisional is not None
    original_volume = read_volume(bundle / "optical.zarr")
    provisional_volume = read_volume(core.current_session.optical_zarr)
    assert isinstance(original_volume, OpticalPropertyVolume)
    assert isinstance(provisional_volume, OpticalPropertyVolume)
    assert np.array_equal(original_volume.sigma_a, provisional_volume.sigma_a)
    assert np.array_equal(original_volume.sigma_s, provisional_volume.sigma_s)


def test_mapping_session_viewer_final_matches_headless_with_and_without_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = Path(__file__).parents[2]
    mapping_root = repository / "examples/pipeline_run/mappings"
    root = tmp_path / "inputs"
    bundle = _write_nested_material_cube_bundle(root, "nested-cube")
    mapping_candidate = resolve_mapping_candidate(
        mapping_root,
        Path("phase0-provisional-materials-v1-tinted.optical-mapping.json"),
    )
    viewer_derived = regenerate_optical(
        bundle, mapping_candidate, tmp_path / "viewer-derived"
    )
    stage = StageConfig(
        render=RenderSettings(width=12, height=12, spp=2, max_depth=11),
        camera=CameraOverride(azimuth_deg=-12.0),
    )
    seed = 4242
    session = create_viewer_session(
        resolve_candidate(root, Path("nested-cube")),
        stage,
        "llvm_ad_rgb",
        seed,
        mapping=SessionMappingRef(
            path=mapping_candidate.root_relative,
            digest=viewer_derived.mapping_digest,
            derived_optical_sha256=zarr_store_sha256(viewer_derived.optical_zarr),
        ),
    )
    session_path = tmp_path / "mapping.session.json"
    write_viewer_session(session_path, session)

    viewer_png = tmp_path / "viewer.png"
    core = StageCore(
        viewer_derived.optical_zarr,
        tmp_path / "viewer-work",
        preview_size=12,
        preview_spp=1,
        initial=stage,
        seed=seed,
    )
    core.render_final(stage, viewer_png)
    viewer_pixels = np.asarray(mi.Bitmap(str(viewer_png)))

    headless_work = tmp_path / "headless-derived"
    rendered: list[np.ndarray] = []
    for index, expected_cache in enumerate(("generated", "reused"), start=1):
        output_png = tmp_path / f"headless-{index}.png"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "mitsuba_stage_demo",
                "--session",
                str(session_path),
                "--input-root",
                str(root),
                "--mapping-root",
                str(mapping_root),
                "--mapping-work-root",
                str(headless_work),
                "--output-png",
                str(output_png),
            ],
        )
        mitsuba_stage_demo.main()
        output = capsys.readouterr().out
        assert f"cache={expected_cache}" in output
        rendered.append(np.asarray(mi.Bitmap(str(output_png))))

    assert np.array_equal(viewer_pixels, rendered[0])
    assert np.array_equal(viewer_pixels, rendered[1])


# -- Phase 5 Step 3: digest cache, Effective state, final sidecar ----------


class _FakeBinder:
    def __init__(self, config: StageConfig) -> None:
        self._config = config

    def current(self) -> StageConfig:
        return self._config


def _make_bare_app(
    core: StageCore,
    root: Path,
    selection: str,
    stage: StageConfig,
    *,
    mapping_root: Path | None = None,
    mapping_work_root: Path | None = None,
) -> mitsuba_stage_viewer.ViewerApp:
    """A GUI-free ``ViewerApp`` exposing only what Step 3's methods need.

    Save session / Verify digests / the final sidecar / Effective state
    don't touch viser widgets directly (status text and dropdowns are set
    by their ``_queue_*`` callers, not by these building blocks), so this
    mirrors the ``ViewerApp.__new__`` pattern already used elsewhere in this
    file for testing transaction methods without a browser.
    """
    app = mitsuba_stage_viewer.ViewerApp.__new__(mitsuba_stage_viewer.ViewerApp)
    app.core = core
    app.input_root = root
    app.mapping_root = mapping_root if mapping_root is not None else root
    app.mapping_work_root = (
        mapping_work_root if mapping_work_root is not None else root / "derived"
    )
    app.interactive_spp = 1
    app._initial_input_path = (root / selection).resolve()
    app._initial_sentinel = None
    app._current_selection = selection
    app.applied_preset = None
    app.binder = _FakeBinder(stage)
    return app


def test_capture_session_reuses_cached_digest_across_saves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    volume = map_material_volume_to_optical(
        homogeneous_transparent().volume, phase0_provisional_mapping()
    )
    bundle = _write_bundle_input(root, "bundle", volume)
    stage = StageConfig(render=RenderSettings(width=8, height=8, spp=1))
    core = StageCore(
        bundle / "optical.zarr",
        tmp_path / "work",
        preview_size=8,
        preview_spp=1,
        initial=stage,
    )
    app = _make_bare_app(core, root, "bundle", stage)

    calls: list[Path] = []
    real_hash = mitsuba_stage_viewer.zarr_store_sha256

    def counting_hash(path: Path) -> str:
        calls.append(path)
        return real_hash(path)

    monkeypatch.setattr(mitsuba_stage_viewer, "zarr_store_sha256", counting_hash)

    app._capture_session(tmp_path / "first.session.json")
    app._capture_session(tmp_path / "second.session.json")

    # optical.zarr is hashed once and reused for the second Save session,
    # not re-hashed from scratch every time (Phase 5 Step 3 digest cache).
    assert len(calls) == 1


def test_effective_state_reflects_only_committed_input_not_dropdown(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
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

    stage = StageConfig(render=RenderSettings(width=8, height=8, spp=1))
    core = StageCore(
        optical_a, tmp_path / "work", preview_size=8, preview_spp=1, initial=stage
    )
    app = _make_bare_app(core, root, "a.zarr", stage)

    before = app._describe_effective_state()
    assert "a.zarr" in before

    # A dropdown change with no Load/Rebuild commit must not move the panel:
    # Effective state reads only ``self._current_selection`` (committed),
    # never a dropdown's live ``.value``.
    app.input_dropdown = SimpleNamespace(value="b.zarr")
    assert app._describe_effective_state() == before

    # A real swap (what Load/Rebuild does on success) does update it.
    session = core.load_input(root, Path("b.zarr"), stage, smoke_spp=1)
    core.swap_session(session)
    app._current_selection = "b.zarr"
    after = app._describe_effective_state()
    assert "b.zarr" in after
    assert after != before


def test_render_final_sidecar_headless_replay_matches_viewer_pixels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    volume = map_material_volume_to_optical(
        homogeneous_transparent().volume, phase0_provisional_mapping()
    )
    bundle = _write_bundle_input(root, "bundle", volume)
    stage = StageConfig(render=RenderSettings(width=8, height=8, spp=1))
    core = StageCore(
        bundle / "optical.zarr",
        tmp_path / "work",
        preview_size=8,
        preview_spp=1,
        initial=stage,
        seed=17,
    )
    app = _make_bare_app(core, root, "bundle", stage)

    final_png = tmp_path / "final.png"
    core.render_final(stage, final_png)
    note = app._write_final_sidecar(final_png)
    assert note is None

    sidecar_path = app._final_sidecar_path(final_png)
    assert sidecar_path == tmp_path / "final.session.json"
    assert sidecar_path.exists()

    viewer_pixels = np.asarray(mi.Bitmap(str(final_png)))

    output_png = tmp_path / "headless.png"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mitsuba_stage_demo",
            "--session",
            str(sidecar_path),
            "--input-root",
            str(root),
            "--output-png",
            str(output_png),
        ],
    )
    mitsuba_stage_demo.main()
    out = capsys.readouterr().out
    assert f"RENDER session={sidecar_path}" in out

    headless_pixels = np.asarray(mi.Bitmap(str(output_png)))
    assert np.array_equal(viewer_pixels, headless_pixels)


def test_render_final_skips_sidecar_for_initial_sentinel_input(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    volume = map_material_volume_to_optical(
        homogeneous_transparent().volume, phase0_provisional_mapping()
    )
    optical_zarr = outside / "model.zarr"
    write_volume(optical_zarr, volume)

    stage = StageConfig(render=RenderSettings(width=8, height=8, spp=1))
    core = StageCore(
        optical_zarr, tmp_path / "work", preview_size=8, preview_spp=1, initial=stage
    )
    app = _make_bare_app(core, root, "model.zarr", stage)
    app._initial_sentinel = f"(initial) {optical_zarr.resolve()}"
    app._current_selection = app._initial_sentinel

    final_png = tmp_path / "final.png"
    core.render_final(stage, final_png)
    note = app._write_final_sidecar(final_png)

    assert note is not None
    assert "session sidecar skipped" in note
    assert final_png.exists()
    assert not app._final_sidecar_path(final_png).exists()


def test_verify_digests_reports_ok_then_drift_after_external_edit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    volume = map_material_volume_to_optical(
        homogeneous_transparent().volume, phase0_provisional_mapping()
    )
    bundle = _write_bundle_input(root, "bundle", volume)
    stage = StageConfig(render=RenderSettings(width=8, height=8, spp=1))
    core = StageCore(
        bundle / "optical.zarr",
        tmp_path / "work",
        preview_size=8,
        preview_spp=1,
        initial=stage,
    )
    app = _make_bare_app(core, root, "bundle", stage)

    first = app._verify_digests()
    assert first == "verify digests: ok — matches cached digests"

    (bundle / "run.json").write_text(
        json.dumps(
            {
                "schema": {"name": "vdbmat.run", "version": "1.0.0"},
                "run_id": "changed",
                "stages": [],
            }
        ),
        encoding="utf-8",
    )
    second = app._verify_digests()
    assert "drift" in second
    assert "run manifest" in second
