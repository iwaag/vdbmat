from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

DEMO_DIR = Path(__file__).parents[1] / "examples" / "pipeline_run" / "demo"
sys.path.insert(0, str(DEMO_DIR))

import mitsuba_stage_viewer  # noqa: E402
from mitsuba_stage import (  # noqa: E402
    BackdropSettings,
    BacklightOverride,
    CameraOverride,
    FloorSettings,
    KeyLightSettings,
    RenderSettings,
    StageConfig,
    stage_config_from_json,
    stage_config_to_dict,
)
from mitsuba_stage_viewer import (  # noqa: E402
    RenderWorker,
    StageBinder,
    StageCore,
    ViewerApp,
    _discard_session_dir,
    _final_render_key,
    _fit_preview_to_aspect,
    _parse_args,
    _preview_stage_config,
    _resolve_initial_optical_zarr,
    _resolve_session_path,
    _resolve_session_root,
    _session_work_dir,
    _slug_for,
    _StageTimer,
    _structure_key,
    _sweep_stale_session_dirs,
)
from mitsuba_viewer_session import SessionPresetRef, ViewerSessionError  # noqa: E402


class _FakeHandle:
    def __init__(self, value: object) -> None:
        self._value = value
        self.callbacks = []

    @property
    def value(self) -> object:
        return self._value

    @value.setter
    def value(self, value: object) -> None:
        self._value = value
        for callback in self.callbacks:
            callback(None)

    def on_update(self, callback):  # type: ignore[no-untyped-def]
        self.callbacks.append(callback)

    def set_value(self, value: object) -> None:
        self.value = value


class _FakeTab:
    def __enter__(self) -> _FakeTab:
        return self

    def __exit__(self, *_args: object) -> None:
        pass


class _FakeTabs:
    def __init__(self) -> None:
        self.labels: list[str] = []

    def add_tab(self, label: str) -> _FakeTab:
        self.labels.append(label)
        return _FakeTab()


class _FakeGui:
    def __init__(self) -> None:
        self.number_options: dict[str, dict[str, object]] = {}
        self.tabs = _FakeTabs()

    def add_tab_group(self) -> _FakeTabs:
        return self.tabs

    def add_number(
        self, label: str, initial_value: object, **options: object
    ) -> _FakeHandle:
        self.number_options[label] = options
        return _FakeHandle(initial_value)

    def add_checkbox(self, _label: str, initial_value: object) -> _FakeHandle:
        return _FakeHandle(initial_value)

    def add_dropdown(
        self, _label: str, _options: object, *, initial_value: object
    ) -> _FakeHandle:
        return _FakeHandle(initial_value)

    def add_slider(
        self, _label: str, *, initial_value: object, **_options: object
    ) -> _FakeHandle:
        return _FakeHandle(initial_value)

    def add_rgb(self, _label: str, *, initial_value: object) -> _FakeHandle:
        return _FakeHandle(initial_value)


def _wait_until(predicate, timeout: float = 2.0) -> None:  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for render worker")
        time.sleep(0.005)


def test_render_worker_discards_stale_and_settles_latest() -> None:
    rendered: list[tuple[int, str]] = []
    published: list[tuple[int, str]] = []
    first_started = threading.Event()
    release_first = threading.Event()

    def render(config: StageConfig, quality: str) -> int:
        rendered.append((config.render.width, quality))
        if len(rendered) == 1:
            first_started.set()
            assert release_first.wait(1.0)
        return config.render.width

    worker = RenderWorker(settle_delay=0.02)
    worker.configure(
        render,
        lambda result, quality: published.append((int(result), quality)),
        lambda message: (_ for _ in ()).throw(AssertionError(message)),
    )
    worker.start()
    worker.request_preview(StageConfig())
    assert first_started.wait(1.0)
    latest = StageConfig().with_cli_overrides(width=640)
    worker.request_preview(latest)
    release_first.set()

    _wait_until(lambda: (640, "settled") in published)
    assert (512, "interactive") not in published
    assert published[-2:] == [(640, "interactive"), (640, "settled")]


def test_render_worker_serializes_final_job_and_latest_preview() -> None:
    events: list[str] = []
    final_started = threading.Event()
    release_final = threading.Event()

    def render(_config: StageConfig, quality: str) -> str:
        events.append(f"render-{quality}")
        return quality

    def final_job() -> None:
        events.append("final-start")
        final_started.set()
        assert release_final.wait(1.0)
        events.append("final-end")

    worker = RenderWorker(settle_delay=0.01)
    worker.configure(render, lambda _result, _quality: None, pytest_fail)
    worker.start()
    worker.submit(final_job)
    assert final_started.wait(1.0)
    worker.request_preview(StageConfig())
    release_final.set()

    _wait_until(lambda: "render-settled" in events)
    assert events.index("final-end") < events.index("render-interactive")


def pytest_fail(message: str) -> None:
    raise AssertionError(message)


def test_render_worker_recovers_after_preview_error() -> None:
    calls = 0
    errors: list[str] = []
    published: list[str] = []

    def render(_config: StageConfig, quality: str) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("expected failure")
        return quality

    worker = RenderWorker(settle_delay=0.01)
    worker.configure(
        render,
        lambda result, _quality: published.append(str(result)),
        errors.append,
    )
    worker.start()
    worker.request_preview(StageConfig())
    _wait_until(lambda: bool(errors))
    worker.request_preview(StageConfig().with_cli_overrides(width=640))
    _wait_until(lambda: "settled" in published)
    assert "expected failure" in errors[0]


def test_viewer_cli_rejects_interactive_spp_above_preview_spp() -> None:
    try:
        _parse_args(["input.zarr", "--preview-spp", "4", "--interactive-spp", "8"])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("invalid spp combination was accepted")


def test_viewer_cli_accepts_input_root() -> None:
    args = _parse_args(["input.zarr", "--input-root", "/some/root"])
    assert args.input_root == Path("/some/root")


def test_viewer_cli_input_root_defaults_to_none() -> None:
    args = _parse_args(["input.zarr"])
    assert args.input_root is None


def test_viewer_cli_accepts_session_root_seed_and_startup_session() -> None:
    args = _parse_args(
        [
            "--session",
            "saved.session.json",
            "--input-root",
            "/inputs",
            "--session-root",
            "/sessions",
            "--seed",
            "37",
        ]
    )

    assert args.optical_zarr is None
    assert args.session == Path("saved.session.json")
    assert args.session_root == Path("/sessions")
    assert args.seed == 37
    assert args.variant is None


def test_viewer_cli_accepts_mapping_root_and_mapping_work_root() -> None:
    args = _parse_args(
        [
            "input.zarr",
            "--mapping-root",
            "/some/mappings",
            "--mapping-work-root",
            "/some/derived",
        ]
    )
    assert args.mapping_root == Path("/some/mappings")
    assert args.mapping_work_root == Path("/some/derived")


def test_viewer_cli_mapping_roots_default_to_none() -> None:
    args = _parse_args(["input.zarr"])
    assert args.mapping_root is None
    assert args.mapping_work_root is None


def test_require_disjoint_roots_rejects_overlap(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    nested = input_root / "derived"

    for a, b in ((nested, input_root), (input_root, input_root)):
        try:
            mitsuba_stage_viewer._require_disjoint_roots(a, b, label="test")
        except ViewerSessionError as error:
            assert error.stage == "resolve"
        else:
            raise AssertionError(f"overlapping roots were accepted: {a}, {b}")


def test_require_disjoint_roots_accepts_sibling_directories(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    mapping_work_root = tmp_path / "derived"
    input_root.mkdir()
    mapping_work_root.mkdir()

    mitsuba_stage_viewer._require_disjoint_roots(
        mapping_work_root, input_root, label="test"
    )


def test_mapping_refresh_preserves_deleted_committed_selection(tmp_path: Path) -> None:
    app = ViewerApp.__new__(ViewerApp)
    app.mapping_root = tmp_path
    app._committed_derivation = SimpleNamespace(
        mapping_candidate=SimpleNamespace(root_relative="deleted.optical-mapping.json")
    )
    app.mapping_dropdown = SimpleNamespace(
        options=(), value="uncommitted.optical-mapping.json"
    )
    app.mapping_summary = SimpleNamespace(content="")

    app._refresh_mapping_catalog()

    assert app.mapping_dropdown.options == (
        mitsuba_stage_viewer._AS_IS_MAPPING,
        "deleted.optical-mapping.json",
    )
    assert app.mapping_dropdown.value == "deleted.optical-mapping.json"
    assert "cannot describe mapping" in app.mapping_summary.content


def test_viewer_cli_rejects_missing_or_conflicting_session_inputs() -> None:
    invalid = (
        [],
        ["--session", "saved.json"],
        ["input.zarr", "--session", "saved.json", "--input-root", "/inputs"],
        [
            "--session",
            "saved.json",
            "--input-root",
            "/inputs",
            "--stage-config",
            "stage.json",
        ],
        ["input.zarr", "--seed", "-1"],
    )
    for argv in invalid:
        try:
            _parse_args(argv)
        except SystemExit as error:
            assert error.code == 2
        else:
            raise AssertionError(f"invalid CLI was accepted: {argv}")


def test_session_paths_are_root_relative_and_contained(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    root.mkdir()
    existing = root / "saved.json"
    existing.write_text("{}", encoding="utf-8")

    assert _resolve_session_root(root, None, tmp_path) == root.resolve()
    assert _resolve_session_path(root, Path("saved.json"), must_exist=True) == (
        existing.resolve()
    )
    assert (
        _resolve_session_path(root, Path("nested/new.json"), must_exist=False)
        == (root / "nested/new.json").resolve()
    )

    for invalid in (Path("/absolute.json"), Path("../escape.json"), Path(".")):
        try:
            _resolve_session_path(root, invalid, must_exist=False)
        except Exception as error:
            assert "session" in str(error)
        else:
            raise AssertionError(f"escaping session path was accepted: {invalid}")


def test_resolve_initial_optical_zarr_passes_through_bare_store(
    tmp_path: Path,
) -> None:
    optical_zarr = tmp_path / "standalone.zarr"
    optical_zarr.mkdir()

    assert _resolve_initial_optical_zarr(optical_zarr) == optical_zarr


def test_resolve_initial_optical_zarr_resolves_bundle_directory(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle_a"
    bundle.mkdir()
    (bundle / "run.json").write_text("{}")

    assert _resolve_initial_optical_zarr(bundle) == bundle / "optical.zarr"


def test_preview_is_padded_to_wide_and_tall_viewports_without_distortion() -> None:
    pixels = np.full((4, 4, 3), 255, dtype=np.uint8)

    wide = _fit_preview_to_aspect(pixels, 2.0)
    assert wide.shape == (4, 8, 3)
    assert np.array_equal(wide[:, 2:6], pixels)
    assert not np.any(wide[:, :2])
    assert not np.any(wide[:, 6:])

    tall = _fit_preview_to_aspect(pixels, 0.5)
    assert tall.shape == (8, 4, 3)
    assert np.array_equal(tall[2:6], pixels)
    assert not np.any(tall[:2])
    assert not np.any(tall[6:])


def test_preview_override_preserves_max_depth() -> None:
    config = StageConfig(
        render=RenderSettings(width=640, height=480, spp=64, max_depth=18)
    )

    preview = _preview_stage_config(config, preview_size=192, preview_spp=4)

    assert preview.render == RenderSettings(width=192, height=192, spp=4, max_depth=18)
    assert config.render == RenderSettings(width=640, height=480, spp=64, max_depth=18)


def test_max_depth_changes_structure_and_final_cache_keys() -> None:
    depth8 = StageConfig(render=RenderSettings(max_depth=8))
    depth16 = StageConfig(render=RenderSettings(max_depth=16))

    assert _structure_key(depth8) != _structure_key(depth16)
    assert _final_render_key(depth8.render) != _final_render_key(depth16.render)


def test_stage_binder_max_depth_widget_and_preset_round_trip(
    tmp_path: Path,
) -> None:
    gui = _FakeGui()
    changes: list[str] = []
    base = StageConfig(
        render=RenderSettings(width=320, height=240, spp=32, max_depth=13)
    )
    binder = StageBinder(
        SimpleNamespace(gui=gui), base, lambda: changes.append("changed")
    )

    assert binder.max_depth.value == 13
    assert gui.number_options["max depth"]["min"] == 1
    assert gui.number_options["max depth"]["step"] == 1
    assert "max" not in gui.number_options["max depth"]
    assert binder.current() == base

    binder.max_depth.set_value(21)
    current = binder.current()

    assert changes == ["changed"]
    assert current.render == RenderSettings(width=320, height=240, spp=32, max_depth=21)

    preset_path = tmp_path / "viewer.stage.json"
    StageCore.save_preset(current, preset_path)

    assert stage_config_from_json(preset_path) == current


def test_stage_binder_replace_config_is_exact_and_suppresses_callbacks() -> None:
    gui = _FakeGui()
    changes: list[str] = []
    binder = StageBinder(
        SimpleNamespace(gui=gui),
        StageConfig(),
        lambda: changes.append("changed"),
    )
    binder.backdrop_distance.set_value(3.25)
    assert binder.dirty == {"backdrop.distance_factor"}
    changes.clear()

    replacement = StageConfig(
        render=RenderSettings(width=640, height=384, spp=24, max_depth=17),
        backdrop=BackdropSettings(
            pattern="solid",
            distance_factor=13.141592,
            checker_scale=13,
            color0=(0.12345, 0.23456, 0.34567),
            color1=(0.76543, 0.65432, 0.54321),
        ),
        floor=FloorSettings(
            enabled=False,
            drop_factor=0.23456,
            checker_scale=9,
            color0=(0.11111, 0.22222, 0.33333),
            color1=(0.88888, 0.77777, 0.66666),
        ),
        key_light=KeyLightSettings(
            direction=(0.37, -1.19, 2.73),
            distance_factor=14.12345,
            scale_factor=1.23456,
            radiance=(4.12345, 3.23456, 2.34567),
        ),
        camera=CameraOverride(
            azimuth_deg=-331.2345,
            elevation_deg=18.7654,
            distance_factor=4.54321,
            fov_deg=142.3456,
        ),
        backlight=BacklightOverride(radiance=(1.2345, 0.9876, 0.5432)),
    )

    binder.replace_config(replacement)

    assert changes == []
    assert binder.base is replacement
    assert binder.dirty == set()
    assert binder.current() == replacement
    assert binder.camera_enabled.value is True
    assert binder.backlight_enabled.value is True
    assert binder.backdrop_distance.value == 10.0
    assert binder.key_distance.value == 10.0
    assert binder.camera_azimuth.value == -180.0
    assert binder.camera_fov.value == 120.0


def test_stage_binder_places_preset_tab_between_input_and_render() -> None:
    gui = _FakeGui()
    built: list[str] = []

    StageBinder(
        SimpleNamespace(gui=gui),
        StageConfig(),
        lambda: None,
        input_tab=lambda _gui: built.append("input"),
        preset_tab=lambda _gui: built.append("preset"),
    )

    assert built == ["input", "preset"]
    assert gui.tabs.labels[:3] == ["Input", "Preset", "Render"]


def test_viewer_cli_accepts_preset_root() -> None:
    args = _parse_args(["input.zarr", "--preset-root", "/preset/root"])

    assert args.preset_root == Path("/preset/root")


def test_viewer_cli_preset_root_defaults_to_none() -> None:
    assert _parse_args(["input.zarr"]).preset_root is None


def test_stage_edit_clears_applied_preset_and_schedules_once() -> None:
    app = ViewerApp.__new__(ViewerApp)
    app.applied_preset = SessionPresetRef(
        path="default.stage.json", digest="sha256:" + "a" * 64
    )
    scheduled: list[str] = []
    app._schedule_preview = lambda: scheduled.append("preview")

    app._on_stage_change()

    assert app.applied_preset is None
    assert scheduled == ["preview"]


def test_apply_stage_preset_replaces_config_once_and_tracks_source(
    tmp_path: Path,
) -> None:
    preset_root = tmp_path / "presets"
    preset_root.mkdir()
    applied = StageConfig(
        render=RenderSettings(width=320, height=240, spp=8, max_depth=15),
        backdrop=BackdropSettings(color0=(0.12345, 0.23456, 0.34567)),
        camera=CameraOverride(),
    )
    preset_path = preset_root / "applied.stage.json"
    preset_path.write_text(json.dumps(stage_config_to_dict(applied)))

    gui = _FakeGui()
    binder = StageBinder(SimpleNamespace(gui=gui), StageConfig(), lambda: None)
    app = ViewerApp.__new__(ViewerApp)
    app.preset_root = preset_root
    app.preset_dropdown = SimpleNamespace(value="applied.stage.json")
    app.status = SimpleNamespace(content="")
    app.binder = binder
    app.applied_preset = None
    scheduled: list[str] = []
    app._schedule_preview = lambda: scheduled.append("preview")

    summary = app._describe_preset_selection("applied.stage.json")
    assert "max depth 15" in summary
    assert binder.current() == StageConfig()
    assert scheduled == []

    app._apply_stage_preset()

    assert binder.current() == applied
    assert app.applied_preset is not None
    assert app.applied_preset.path == "applied.stage.json"
    assert scheduled == ["preview"]
    assert app.status.content == "stage preset applied: applied.stage.json"


def test_apply_broken_preset_preserves_config_source_and_preview(
    tmp_path: Path,
) -> None:
    preset_root = tmp_path / "presets"
    preset_root.mkdir()
    (preset_root / "broken.stage.json").write_text("{")
    original = StageConfig(render=RenderSettings(max_depth=12))
    source = SessionPresetRef(
        path="original.stage.json",
        digest="sha256:" + "b" * 64,
    )
    binder = StageBinder(SimpleNamespace(gui=_FakeGui()), original, lambda: None)
    app = ViewerApp.__new__(ViewerApp)
    app.preset_root = preset_root
    app.preset_dropdown = SimpleNamespace(value="broken.stage.json")
    app.status = SimpleNamespace(content="")
    app.binder = binder
    app.applied_preset = source
    scheduled: list[str] = []
    app._schedule_preview = lambda: scheduled.append("preview")

    app._apply_stage_preset()

    assert binder.current() == original
    assert app.applied_preset is source
    assert scheduled == []
    assert "cannot apply stage preset" in app.status.content


def test_render_worker_discards_preview_from_stale_session_generation() -> None:
    """A result computed against a superseded session is never published.

    This scenario cannot actually arise today (the worker only ever runs one
    job at a time, so a swap can never interleave with an in-flight render),
    but the guard exists for defense in depth. The fake ``render`` callback
    below stands in for "the swap happened while this render was running".
    """
    published: list[tuple[str, str]] = []
    current_generation = {"value": 1}

    def render(_config: StageConfig, quality: str) -> str:
        current_generation["value"] = 2
        return quality

    worker = RenderWorker(settle_delay=0.01)
    worker.configure(
        render,
        lambda result, quality: published.append((str(result), quality)),
        pytest_fail,
        lambda: current_generation["value"],
    )
    worker.start()
    worker.request_preview(StageConfig(), session_generation=1)

    time.sleep(0.2)
    assert published == []


def test_render_worker_publishes_when_session_generation_matches() -> None:
    published: list[tuple[str, str]] = []

    worker = RenderWorker(settle_delay=0.01)
    worker.configure(
        lambda _config, quality: quality,
        lambda result, quality: published.append((str(result), quality)),
        pytest_fail,
        lambda: 7,
    )
    worker.start()
    worker.request_preview(StageConfig(), session_generation=7)

    _wait_until(lambda: ("settled", "settled") in published)
    assert ("interactive", "interactive") in published


def test_render_worker_defaults_preserve_prior_behaviour_without_sessions() -> None:
    """configure()/request_preview() with no session args behave as before."""
    published: list[tuple[str, str]] = []

    worker = RenderWorker(settle_delay=0.01)
    worker.configure(
        lambda _config, quality: quality,
        lambda result, quality: published.append((str(result), quality)),
        pytest_fail,
    )
    worker.start()
    worker.request_preview(StageConfig())

    _wait_until(lambda: ("settled", "settled") in published)


def test_slug_for_distinguishes_bundle_and_standalone_optical_paths() -> None:
    assert _slug_for(Path("/root/bundle_a/optical.zarr")) == "bundle-a"
    assert _slug_for(Path("/root/standalone.zarr")) == "standalone"


def test_session_work_dir_sequence_is_unique_for_repeated_input(
    tmp_path: Path,
) -> None:
    optical_zarr = Path("bundle_a/optical.zarr")

    first = _session_work_dir(tmp_path, 0, optical_zarr)
    second = _session_work_dir(tmp_path, 1, optical_zarr)

    assert first == tmp_path / "inputs" / "000-bundle-a"
    assert second == tmp_path / "inputs" / "001-bundle-a"
    assert first != second


def test_stage_timer_first_advance_has_no_suffix_and_logs_nothing(
    capsys: object,
) -> None:
    timer = _StageTimer("input load")

    suffix = timer.advance("validate")

    assert suffix == ""
    assert capsys.readouterr().out == ""  # type: ignore[attr-defined]


def test_stage_timer_advance_logs_previous_stage_and_returns_status_suffix(
    capsys: object,
) -> None:
    timer = _StageTimer("input load")
    timer.advance("validate")

    suffix = timer.advance("map")

    out = capsys.readouterr().out  # type: ignore[attr-defined]
    lines = out.strip().splitlines()
    assert len(lines) == 1
    prefix, elapsed_text = lines[0].rsplit(" ", 1)
    assert prefix == "STAGE input load validate"
    assert float(elapsed_text) >= 0.0
    assert suffix.startswith(" (validate ")
    assert suffix.endswith("s)")


def test_stage_timer_finish_logs_last_stage_and_returns_total_elapsed(
    capsys: object,
) -> None:
    timer = _StageTimer("input load")
    timer.advance("validate")
    timer.advance("map")

    total = timer.finish()

    out = capsys.readouterr().out  # type: ignore[attr-defined]
    lines = out.strip().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("STAGE input load validate ")
    assert lines[1].startswith("STAGE input load map ")
    assert total >= 0.0


def test_stage_timer_finish_without_any_advance_logs_nothing(
    capsys: object,
) -> None:
    timer = _StageTimer("session load")

    total = timer.finish()

    assert capsys.readouterr().out == ""  # type: ignore[attr-defined]
    assert total >= 0.0


def test_stage_timer_full_transaction_logs_one_stage_line_per_transition() -> None:
    """The full validate/map/prepare/load/smoke/swap sequence stays intact.

    Timing is purely additive: every stage that fires through ``advance()``
    produces exactly one STAGE line (via the following ``advance()`` or the
    closing ``finish()``), and the stage names/order the caller passed
    through are echoed back verbatim.
    """
    import io
    from contextlib import redirect_stdout

    stages = ("validate", "map", "prepare", "load", "smoke", "swap")
    timer = _StageTimer("input load")
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        for stage in stages:
            timer.advance(stage)
        timer.finish()

    lines = buffer.getvalue().strip().splitlines()
    logged_stages = [line.rsplit(" ", 2)[1] for line in lines]
    assert logged_stages == list(stages)


def test_sweep_stale_session_dirs_removes_only_inputs_entries(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "work"
    stale_a = work_dir / "inputs" / "000-bundle-a"
    stale_b = work_dir / "inputs" / "001-bundle-b"
    stale_a.mkdir(parents=True)
    stale_b.mkdir(parents=True)
    (stale_a / "preview_scene").mkdir()
    derived = work_dir / "derived" / "some-digest"
    derived.mkdir(parents=True)
    other_file = work_dir / "viewer.stage.json"
    other_file.write_text("{}", encoding="utf-8")

    _sweep_stale_session_dirs(work_dir)

    assert not stale_a.exists()
    assert not stale_b.exists()
    assert (work_dir / "inputs").exists()  # the inputs/ dir itself is kept
    assert derived.exists()
    assert other_file.exists()


def test_sweep_stale_session_dirs_is_noop_without_inputs_dir(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    (work_dir / "derived").mkdir()

    _sweep_stale_session_dirs(work_dir)  # must not raise

    assert (work_dir / "derived").exists()


def test_sweep_stale_session_dirs_skips_entry_escaping_work_dir_via_symlink(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "work"
    inputs_dir = work_dir / "inputs"
    inputs_dir.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keepme").write_text("data", encoding="utf-8")
    escaping_link = inputs_dir / "000-escapes"
    escaping_link.symlink_to(outside, target_is_directory=True)

    _sweep_stale_session_dirs(work_dir)

    assert (outside / "keepme").exists()


def test_discard_session_dir_removes_directory_and_tolerates_missing(
    tmp_path: Path,
) -> None:
    present = tmp_path / "present"
    present.mkdir()
    (present / "file.txt").write_text("x", encoding="utf-8")
    missing = tmp_path / "missing"

    _discard_session_dir(present)
    _discard_session_dir(missing)  # already-gone directory: no error

    assert not present.exists()
