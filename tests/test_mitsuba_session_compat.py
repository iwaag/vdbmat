from __future__ import annotations

import sys
from pathlib import Path

import pytest

DEMO_DIR = Path(__file__).parents[1] / "examples" / "pipeline_run" / "demo"
sys.path.insert(0, str(DEMO_DIR))

import mitsuba_session_compat as compat  # noqa: E402
import mitsuba_viewer_session as session_module  # noqa: E402
from mitsuba_stage import RenderSettings, StageConfig  # noqa: E402
from mitsuba_stage_presets import stage_config_digest  # noqa: E402
from mitsuba_viewer_session import (  # noqa: E402
    SessionInputRef,
    SessionMappingRef,
    ViewerSession,
    ViewerSessionError,
    write_viewer_session,
)

_DIGEST_A = "sha256:" + "a" * 64
_DIGEST_B = "sha256:" + "b" * 64
_DIGEST_C = "sha256:" + "c" * 64


def _bundle_session(
    *,
    variant: str = "llvm_ad_rgb",
    seed: int = 7,
    config: StageConfig | None = None,
    optical_sha256: str = _DIGEST_A,
    run_manifest_sha256: str = _DIGEST_B,
    mapping: SessionMappingRef | None = None,
) -> ViewerSession:
    stage = StageConfig() if config is None else config
    return ViewerSession(
        input=SessionInputRef(
            kind=session_module.InputKind.RUN_BUNDLE,
            path="catalog/bundle",
            optical_sha256=optical_sha256,
            run_manifest_sha256=run_manifest_sha256,
        ),
        stage_config=stage,
        effective_digest=stage_config_digest(stage),
        variant=variant,
        seed=seed,
        mapping=mapping,
    )


def _write(path: Path, session: ViewerSession) -> Path:
    write_viewer_session(path, session)
    return path


def test_variant_only_difference_is_scientifically_equal(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.session.json", _bundle_session(variant="llvm_ad_rgb"))
    b = _write(tmp_path / "b.session.json", _bundle_session(variant="cuda_ad_rgb"))

    report = compat.compare_sessions(a, b)

    assert report.scientifically_equal is True
    assert report.differences == ()


def test_identical_sessions_are_scientifically_equal(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.session.json", _bundle_session())
    b = _write(tmp_path / "b.session.json", _bundle_session())

    report = compat.compare_sessions(a, b)

    assert report.scientifically_equal is True
    assert report.differences == ()


def test_input_optical_digest_difference_is_detected(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.session.json", _bundle_session(optical_sha256=_DIGEST_A))
    b = _write(tmp_path / "b.session.json", _bundle_session(optical_sha256=_DIGEST_C))

    report = compat.compare_sessions(a, b)

    assert report.scientifically_equal is False
    assert ("input.optical_sha256", _DIGEST_A, _DIGEST_C) in report.differences


def test_run_manifest_digest_difference_is_detected(tmp_path: Path) -> None:
    a = _write(
        tmp_path / "a.session.json", _bundle_session(run_manifest_sha256=_DIGEST_B)
    )
    b = _write(
        tmp_path / "b.session.json", _bundle_session(run_manifest_sha256=_DIGEST_C)
    )

    report = compat.compare_sessions(a, b)

    assert report.scientifically_equal is False
    assert ("input.run_manifest_sha256", _DIGEST_B, _DIGEST_C) in report.differences


def test_mapping_digest_and_derived_digest_differences_are_detected(
    tmp_path: Path,
) -> None:
    mapping_a = SessionMappingRef(
        path="tinted.optical-mapping.json",
        digest=_DIGEST_A,
        derived_optical_sha256=_DIGEST_B,
    )
    mapping_b = SessionMappingRef(
        path="tinted.optical-mapping.json",
        digest=_DIGEST_C,
        derived_optical_sha256=_DIGEST_A,
    )
    a = _write(tmp_path / "a.session.json", _bundle_session(mapping=mapping_a))
    b = _write(tmp_path / "b.session.json", _bundle_session(mapping=mapping_b))

    report = compat.compare_sessions(a, b)

    assert report.scientifically_equal is False
    assert ("mapping.digest", _DIGEST_A, _DIGEST_C) in report.differences
    assert (
        "mapping.derived_optical_sha256",
        _DIGEST_B,
        _DIGEST_A,
    ) in report.differences


def test_mapping_presence_mismatch_is_detected(tmp_path: Path) -> None:
    mapping = SessionMappingRef(
        path="tinted.optical-mapping.json",
        digest=_DIGEST_A,
        derived_optical_sha256=_DIGEST_B,
    )
    a = _write(tmp_path / "a.session.json", _bundle_session(mapping=None))
    b = _write(tmp_path / "b.session.json", _bundle_session(mapping=mapping))

    report = compat.compare_sessions(a, b)

    assert report.scientifically_equal is False
    assert (
        "mapping.path",
        "(none)",
        "tinted.optical-mapping.json",
    ) in report.differences


def test_effective_stage_render_difference_is_detected(tmp_path: Path) -> None:
    config_a = StageConfig(render=RenderSettings(width=64, height=64, spp=32))
    config_b = StageConfig(render=RenderSettings(width=128, height=128, spp=32))
    a = _write(tmp_path / "a.session.json", _bundle_session(config=config_a))
    b = _write(tmp_path / "b.session.json", _bundle_session(config=config_b))

    report = compat.compare_sessions(a, b)

    assert report.scientifically_equal is False
    fields = {field for field, _a, _b in report.differences}
    assert fields == {"stage.effective_digest"}


def test_seed_difference_is_detected(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.session.json", _bundle_session(seed=7))
    b = _write(tmp_path / "b.session.json", _bundle_session(seed=8))

    report = compat.compare_sessions(a, b)

    assert report.scientifically_equal is False
    assert ("mitsuba.seed", "7", "8") in report.differences


def test_malformed_session_document_surfaces_existing_reader_diagnostic(
    tmp_path: Path,
) -> None:
    a = _write(tmp_path / "a.session.json", _bundle_session())
    corrupted = tmp_path / "corrupted.session.json"
    document = a.read_text(encoding="utf-8")
    corrupted.write_text(
        document.replace('"sha256:' + "a" * 64, '"not-a-digest'), encoding="utf-8"
    )

    with pytest.raises(ViewerSessionError):
        compat.compare_sessions(a, corrupted)


def test_cli_exits_zero_for_variant_only_difference(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a = _write(tmp_path / "a.session.json", _bundle_session(variant="llvm_ad_rgb"))
    b = _write(tmp_path / "b.session.json", _bundle_session(variant="cuda_ad_rgb"))

    old_argv = sys.argv
    sys.argv = ["mitsuba_session_compat", str(a), str(b)]
    try:
        compat.main()
    finally:
        sys.argv = old_argv

    out = capsys.readouterr().out
    assert "SCIENTIFICALLY_EQUAL true" in out


def test_cli_exits_nonzero_and_lists_differences(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a = _write(tmp_path / "a.session.json", _bundle_session(seed=7))
    b = _write(tmp_path / "b.session.json", _bundle_session(seed=8))

    old_argv = sys.argv
    sys.argv = ["mitsuba_session_compat", str(a), str(b)]
    try:
        with pytest.raises(SystemExit) as excinfo:
            compat.main()
    finally:
        sys.argv = old_argv

    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "SCIENTIFICALLY_EQUAL false" in out
    assert "DIFF mitsuba.seed: a=7 b=8" in out
