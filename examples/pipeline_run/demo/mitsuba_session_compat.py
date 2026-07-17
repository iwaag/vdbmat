"""Compare two viewer sessions for scientific (variant-independent) equality.

A CPU (``llvm_ad_rgb``) and a GPU (``cuda_ad_rgb``) session for "the same"
input/mapping/stage/render/seed should agree on everything except the Mitsuba
variant itself — the viewer only pins one variant per process (see
``mitsuba_stage_viewer.py``'s module docstring), so the standard way to check
"did I actually compare like with like" is to save a session under each
variant and diff the two manifests. This module is that diff, decoupled from
Mitsuba and viser so it can run anywhere the two ``*.session.json`` files are
available (including a machine without a GPU).

This module has no dependency on Mitsuba or viser; it only reads
``vdbmat.viewer-session`` documents through the existing
:mod:`mitsuba_viewer_session` reader, so a malformed session file surfaces
that reader's own diagnostic unchanged.

Invoke on the host (no ``--group mitsuba`` needed):

    uv run python examples/pipeline_run/demo/mitsuba_session_compat.py \\
        A.session.json B.session.json

Exit code 0 means the two sessions are scientifically equal (variant is the
only difference, if any); a non-zero exit lists every differing field.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from mitsuba_viewer_session import (
    ViewerSession,
    ViewerSessionError,
    viewer_session_from_json,
)

_NONE_MARKER = "(none)"


@dataclass(frozen=True, slots=True)
class SessionCompatReport:
    """Result of comparing two sessions on everything except ``variant``."""

    scientifically_equal: bool
    differences: tuple[tuple[str, str, str], ...]


def _comparable_fields(session: ViewerSession) -> dict[str, str]:
    """Return the variant-independent fields that define "the same input".

    ``stage.effective_digest`` is ``stage_config_digest(session.stage_config)``
    (enforced by ``ViewerSession.__post_init__``), so comparing it alone is
    sufficient to catch any stage/render setting difference without
    re-deriving a field-by-field diff of the stage config here.
    """
    mapping = session.mapping
    return {
        "input.kind": session.input.kind.value,
        "input.path": session.input.path,
        "input.optical_sha256": session.input.optical_sha256,
        "input.run_manifest_sha256": session.input.run_manifest_sha256 or _NONE_MARKER,
        "mapping.path": mapping.path if mapping is not None else _NONE_MARKER,
        "mapping.digest": mapping.digest if mapping is not None else _NONE_MARKER,
        "mapping.derived_optical_sha256": (
            mapping.derived_optical_sha256 if mapping is not None else _NONE_MARKER
        ),
        "stage.effective_digest": session.effective_digest,
        "mitsuba.seed": str(session.seed),
    }


def compare_sessions(a: Path, b: Path) -> SessionCompatReport:
    """Diff two session manifests on everything except ``mitsuba.variant``.

    Raises :class:`~mitsuba_viewer_session.ViewerSessionError` unchanged if
    either file is not a valid ``vdbmat.viewer-session`` document — this
    module adds no diagnostic of its own for a malformed input, it passes
    the existing reader's through.
    """
    session_a = viewer_session_from_json(a)
    session_b = viewer_session_from_json(b)
    fields_a = _comparable_fields(session_a)
    fields_b = _comparable_fields(session_b)
    differences = tuple(
        (field, fields_a[field], fields_b[field])
        for field in fields_a
        if fields_a[field] != fields_b[field]
    )
    return SessionCompatReport(
        scientifically_equal=not differences, differences=differences
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="mitsuba_session_compat")
    parser.add_argument("session_a", type=Path)
    parser.add_argument("session_b", type=Path)
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args(sys.argv[1:])
    try:
        report = compare_sessions(args.session_a, args.session_b)
    except ViewerSessionError as error:
        raise SystemExit(
            f"session compat failed at {error.stage}: {error.message}"
        ) from error

    session_a = viewer_session_from_json(args.session_a)
    session_b = viewer_session_from_json(args.session_b)
    print(f"A variant={session_a.variant} ({args.session_a})")
    print(f"B variant={session_b.variant} ({args.session_b})")

    if report.scientifically_equal:
        print("SCIENTIFICALLY_EQUAL true — variant is the only possible difference")
        return

    print("SCIENTIFICALLY_EQUAL false")
    for field, value_a, value_b in report.differences:
        print(f"DIFF {field}: a={value_a} b={value_b}")
    raise SystemExit(1)


__all__ = ["SessionCompatReport", "compare_sessions"]

if __name__ == "__main__":
    main()
