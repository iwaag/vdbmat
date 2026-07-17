"""Representative-input session-replay regression (Phase 5 Step 4).

Exercises the checked-in fixtures (``nested_material_cube``, ``stepped_wedge``,
``window_coupon``) and checked-in mappings (``phase0-provisional-materials-v1``
and its ``-tinted`` variant) end to end, through the same viewer-core
transaction API the GUI uses (:class:`StageCore` + ``ViewerApp`` transaction
methods), rather than the synthetic in-memory volumes most of
``test_mitsuba_stage_viewer.py`` uses. This is the parameterized regression
the Phase 5 plan calls for: representative pairs, not the full 3x3 product
(see ``_REPRESENTATIVE_PAIRS`` below), each producing a viewer final render
and its sidecar session, replayed headlessly, and checked pixel-for-pixel
against the viewer's own output plus provenance (mapping digest).

Also covers the plan's Step 4 robustness cases: a failure interleaved between
successful loads must leave the committed session untouched, back-to-back
Load/Rebuild requests must leave only the latest one published, and a newly
connecting client must receive the currently-published preview (the unit
level substitute for the real-browser reconnect check, which this
non-interactive environment cannot drive; see report4.md).
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from vdbmat.optics import load_optical_mapping
from vdbmat.pipeline import PipelineConfig, run_pipeline

DEMO_DIR = Path(__file__).parents[2] / "examples" / "pipeline_run" / "demo"
sys.path.insert(0, str(DEMO_DIR))

import mitsuba_stage_demo  # noqa: E402
import mitsuba_stage_viewer  # noqa: E402
from mitsuba_stage import RenderSettings, StageConfig  # noqa: E402
from mitsuba_stage_viewer import InputLoadError, StageCore  # noqa: E402

mi = pytest.importorskip("mitsuba")

pytestmark = pytest.mark.mitsuba

REPO_ROOT = Path(__file__).parents[2]
FIXTURES_ROOT = REPO_ROOT / "examples" / "pipeline_run" / "inputs"
MAPPINGS_ROOT = REPO_ROOT / "examples" / "pipeline_run" / "mappings"

_AS_IS = mitsuba_stage_viewer._AS_IS_MAPPING
_PROVISIONAL_MAPPING = "phase0-provisional-materials-v1.optical-mapping.json"
_TINTED_MAPPING = "phase0-provisional-materials-v1-tinted.optical-mapping.json"

# Full 3 fixtures x 3 mappings would be nine Mitsuba renders per test session;
# the plan calls for representative pairs instead: every fixture at least
# once as-is, the full mapping spread on one fixture (nested_material_cube),
# and one more mapped fixture (window_coupon x tinted) to catch a
# fixture-specific palette-coverage regression that a single fixture's mapping
# sweep wouldn't.
_REPRESENTATIVE_PAIRS = [
    ("nested_material_cube", _AS_IS),
    ("nested_material_cube", _PROVISIONAL_MAPPING),
    ("nested_material_cube", _TINTED_MAPPING),
    ("stepped_wedge", _AS_IS),
    ("window_coupon", _AS_IS),
    ("window_coupon", _TINTED_MAPPING),
]


def _write_fixture_bundle(fixture: str, root: Path, name: str) -> Path:
    """Publish a real canonical run bundle for one checked-in fixture.

    Goes through the actual ``run_pipeline()`` orchestration (not a
    synthetic in-memory volume) so the bundle carries a preserved
    ``source/*.voxels.json`` and a ``run.json`` with declared digests, the
    same shape ``mitsuba_stage_regen.regenerate_optical()`` re-derives from.
    """
    config = PipelineConfig(
        input_kind="direct-voxel",
        input_path=f"{fixture}.voxels.json",
        output_path=str(root / name),
        mapping_name="phase0-provisional-materials-v1",
    )
    return run_pipeline(config, base_dir=str(FIXTURES_ROOT)).output_path


class _FakeBinder:
    def __init__(self, config: StageConfig) -> None:
        self._config = config

    def current(self) -> StageConfig:
        return self._config

    def replace_config(self, config: StageConfig) -> None:
        self._config = config


def _make_bare_app(
    core: StageCore,
    root: Path,
    selection: str,
    stage: StageConfig,
    *,
    mapping_root: Path,
    mapping_work_root: Path,
) -> mitsuba_stage_viewer.ViewerApp:
    """A GUI-free ``ViewerApp`` exposing the transaction/session methods.

    Mirrors the ``ViewerApp.__new__`` pattern used throughout
    ``test_mitsuba_stage_viewer.py`` for exercising transaction methods
    without a browser/viser server.
    """
    app = mitsuba_stage_viewer.ViewerApp.__new__(mitsuba_stage_viewer.ViewerApp)
    app.core = core
    app.input_root = root
    app.mapping_root = mapping_root
    app.mapping_work_root = mapping_work_root
    app.interactive_spp = 1
    app._initial_input_path = (root / selection).resolve()
    app._initial_sentinel = None
    app._current_selection = selection
    app._committed_derivation = None
    app.applied_preset = None
    app.binder = _FakeBinder(stage)
    return app


@pytest.mark.parametrize("fixture,mapping_selection", _REPRESENTATIVE_PAIRS)
def test_representative_pair_session_replay_matches_headless_and_records_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fixture: str,
    mapping_selection: str,
) -> None:
    root = tmp_path / "root"
    bundle = _write_fixture_bundle(fixture, root, "bundle")
    stage = StageConfig(render=RenderSettings(width=8, height=8, spp=1))
    mapping_work_root = tmp_path / "derived"
    core = StageCore(
        bundle / "optical.zarr",
        tmp_path / "work",
        preview_size=8,
        preview_spp=1,
        initial=stage,
        seed=11,
    )
    app = _make_bare_app(
        core,
        root,
        "bundle",
        stage,
        mapping_root=MAPPINGS_ROOT,
        mapping_work_root=mapping_work_root,
    )

    derivation = app._load_input_transaction(
        "bundle", mapping_selection, stage, lambda _stage: None
    )
    app._committed_derivation = derivation

    if mapping_selection == _AS_IS:
        assert derivation is None
    else:
        assert derivation is not None
        expected_digest = load_optical_mapping(MAPPINGS_ROOT / mapping_selection).digest
        assert derivation.mapping_digest == expected_digest

    final_png = tmp_path / "final.png"
    core.render_final(stage, final_png)
    note = app._write_final_sidecar(final_png)
    assert note is None
    sidecar_path = app._final_sidecar_path(final_png)
    assert sidecar_path.exists()

    viewer_pixels = np.asarray(mi.Bitmap(str(final_png)))

    output_png = tmp_path / "headless.png"
    argv = [
        "mitsuba_stage_demo",
        "--session",
        str(sidecar_path),
        "--input-root",
        str(root),
        "--output-png",
        str(output_png),
    ]
    if mapping_selection != _AS_IS:
        argv += [
            "--mapping-root",
            str(MAPPINGS_ROOT),
            "--mapping-work-root",
            str(mapping_work_root),
        ]
    monkeypatch.setattr(sys, "argv", argv)
    mitsuba_stage_demo.main()
    out = capsys.readouterr().out
    assert f"RENDER session={sidecar_path}" in out
    if mapping_selection != _AS_IS:
        assert f"MAPPING {mapping_selection} digest={derivation.mapping_digest}" in out

    headless_pixels = np.asarray(mi.Bitmap(str(output_png)))
    assert np.array_equal(viewer_pixels, headless_pixels)


def test_failed_interim_loads_leave_committed_session_and_effective_state_unchanged(
    tmp_path: Path,
) -> None:
    """A validate-stage, a map-stage, and a coverage failure, each in turn,
    must leave the previously committed session (and therefore the live
    preview) exactly as it was — the Load/Rebuild transaction guarantee this
    module's plan fixes, now checked against real fixtures/mappings instead
    of the synthetic volumes most of ``test_mitsuba_stage_viewer.py`` uses.
    """
    root = tmp_path / "root"
    good_bundle = _write_fixture_bundle("nested_material_cube", root, "good-bundle")
    stage = StageConfig(render=RenderSettings(width=8, height=8, spp=1))
    mapping_work_root = tmp_path / "derived"

    local_mappings = tmp_path / "mappings"
    local_mappings.mkdir()
    provisional_doc = json.loads(
        (MAPPINGS_ROOT / _PROVISIONAL_MAPPING).read_text(encoding="utf-8")
    )
    (local_mappings / _PROVISIONAL_MAPPING).write_text(
        json.dumps(provisional_doc), encoding="utf-8"
    )
    (local_mappings / _TINTED_MAPPING).write_text(
        (MAPPINGS_ROOT / _TINTED_MAPPING).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    # nested_material_cube uses material ids 0/1/3 (air/transparent/opaque);
    # dropping material_id 3 makes the mapping fail palette coverage during
    # regeneration's validate_material/validate_optical stages.
    incomplete_doc = json.loads(json.dumps(provisional_doc))
    incomplete_doc["materials"] = [
        material
        for material in incomplete_doc["materials"]
        if material["material_id"] != 3
    ]
    (local_mappings / "incomplete.optical-mapping.json").write_text(
        json.dumps(incomplete_doc), encoding="utf-8"
    )

    core = StageCore(
        good_bundle / "optical.zarr",
        tmp_path / "work",
        preview_size=8,
        preview_spp=1,
        initial=stage,
        seed=3,
    )
    app = _make_bare_app(
        core,
        root,
        "good-bundle",
        stage,
        mapping_root=local_mappings,
        mapping_work_root=mapping_work_root,
    )

    # Establish a real committed baseline via the provisional mapping.
    baseline_derivation = app._load_input_transaction(
        "good-bundle", _PROVISIONAL_MAPPING, stage, lambda _stage: None
    )
    app._committed_derivation = baseline_derivation
    baseline_session = core.current_session
    baseline_generation = core.session_generation
    assert baseline_derivation is not None

    def assert_unchanged() -> None:
        assert core.current_session is baseline_session
        assert core.session_generation == baseline_generation
        assert app._current_selection == "good-bundle"
        assert app._committed_derivation is baseline_derivation

    # (a) validate-stage failure: input selection does not exist.
    with pytest.raises(InputLoadError) as excinfo:
        app._load_input_transaction(
            "no-such-bundle", _PROVISIONAL_MAPPING, stage, lambda _stage: None
        )
    assert excinfo.value.stage == "validate"
    assert_unchanged()

    # (b) validate-stage failure: mapping file does not exist.
    with pytest.raises(InputLoadError) as excinfo:
        app._load_input_transaction(
            "good-bundle", "no-such-mapping.json", stage, lambda _stage: None
        )
    assert excinfo.value.stage == "validate"
    assert_unchanged()

    # (c) map-stage failure: mapping's palette does not cover the fixture.
    with pytest.raises(InputLoadError) as excinfo:
        app._load_input_transaction(
            "good-bundle",
            "incomplete.optical-mapping.json",
            stage,
            lambda _stage: None,
        )
    assert excinfo.value.stage == "map"
    assert_unchanged()

    # A subsequent good load still succeeds — the failures above did not
    # corrupt any shared state (derived cache, session sequence numbers).
    recovered = app._load_input_transaction(
        "good-bundle", _TINTED_MAPPING, stage, lambda _stage: None
    )
    assert recovered is not None
    assert core.current_session is not baseline_session
    assert core.session_generation == baseline_generation + 1


def test_consecutive_load_requests_leave_only_latest_generation_published(
    tmp_path: Path,
) -> None:
    """Two different representative inputs, loaded back to back, must leave
    the session generation, current input, and preview reflecting only the
    second — never a mix of the two (the generation guard from Phase 2-4,
    now exercised across distinct real fixtures/mappings rather than the
    same synthetic input reloaded).
    """
    root = tmp_path / "root"
    bundle_a = _write_fixture_bundle("nested_material_cube", root, "bundle-a")
    _write_fixture_bundle("window_coupon", root, "bundle-b")
    stage = StageConfig(render=RenderSettings(width=8, height=8, spp=1))
    mapping_work_root = tmp_path / "derived"

    core = StageCore(
        bundle_a / "optical.zarr",
        tmp_path / "work",
        preview_size=8,
        preview_spp=1,
        initial=stage,
        seed=5,
    )
    app = _make_bare_app(
        core,
        root,
        "bundle-a",
        stage,
        mapping_root=MAPPINGS_ROOT,
        mapping_work_root=mapping_work_root,
    )
    initial_generation = core.session_generation

    app._load_input_transaction("bundle-a", _AS_IS, stage, lambda _stage: None)
    app._current_selection = "bundle-a"
    first_session = core.current_session
    first_work_dir = first_session.work_dir

    second_derivation = app._load_input_transaction(
        "bundle-b", _TINTED_MAPPING, stage, lambda _stage: None
    )
    app._current_selection = "bundle-b"
    app._committed_derivation = second_derivation

    assert core.session_generation == initial_generation + 2
    assert core.current_session is not first_session
    assert core.current_session.derivation is second_derivation
    # Cleanup rule (a) from Step 2: the superseded session's own work
    # directory (never the current one) is discarded on a successful swap.
    assert not first_work_dir.exists()
    assert core.current_session.work_dir.exists()

    pixels, _stats, _route = core.render_preview(stage)
    assert pixels.shape == (8, 8, 3)


class _FakeCamera:
    def __init__(self, width: int, height: int) -> None:
        self.image_width = width
        self.image_height = height


class _FakeScene:
    def __init__(self) -> None:
        self.background_images: list[np.ndarray] = []

    def set_background_image(self, image: np.ndarray) -> None:
        self.background_images.append(image)


class _FakeClient:
    def __init__(self, client_id: int, width: int, height: int) -> None:
        self.client_id = client_id
        self.camera = _FakeCamera(width, height)
        self.scene = _FakeScene()


def test_client_connect_callback_resends_current_preview_pixels() -> None:
    """A newly connecting client must receive the currently-published
    preview immediately, independent of whether a Load/Rebuild is mid-flight.

    ``_update_client_preview`` — the body of ``on_client_connect`` — only
    reads ``self._preview_pixels`` under ``self._preview_lock``; Load/Rebuild
    never touches either, so this exercises the reconnect-resend guarantee
    at the unit level without needing a real viser server or browser. The
    real-browser confirmation (closing/reopening a tab mid Load/Rebuild)
    remains a manual check this non-interactive environment cannot drive;
    see report4.md.
    """
    app = mitsuba_stage_viewer.ViewerApp.__new__(mitsuba_stage_viewer.ViewerApp)
    pixels = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    app._preview_pixels = pixels
    app._preview_lock = threading.Lock()
    app._client_viewport_sizes = {}

    client = _FakeClient(client_id=1, width=8, height=8)
    mitsuba_stage_viewer.ViewerApp._update_client_preview(app, client, force=True)

    assert len(client.scene.background_images) == 1
    assert np.array_equal(client.scene.background_images[0], pixels)
    assert app._client_viewport_sizes == {1: (8, 8)}
