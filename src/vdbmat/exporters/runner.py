"""Restored-Zarr entry point shared by the CLI and pipeline export stage."""

from __future__ import annotations

import importlib.metadata
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vbdmat.core import OpticalPropertyVolume
from vbdmat.io import read_volume

from .diagnostics import CapabilityReport
from .mitsuba import (
    MITSUBA_ADAPTER,
    MITSUBA_ADAPTER_VERSION,
    MitsubaDependencyError,
    MitsubaExportError,
    prepare_mitsuba_scene,
    render_mitsuba,
)
from .openvdb import (
    OPENVDB_ADAPTER,
    OPENVDB_ADAPTER_VERSION,
    OpenVDBDependencyError,
    OpenVDBExportError,
    export_openvdb,
)


class ExportInputError(RuntimeError):
    """The export source is not a canonical optical volume."""


@dataclass(frozen=True, slots=True)
class ExportOutcome:
    """Stable export result suitable for CLI output and ``run.json``."""

    target: str
    output_path: Path
    adapter: str
    adapter_version: str
    renderer: str
    renderer_version: str
    capability_report: CapabilityReport
    artifacts: tuple[Path, ...]
    rendered: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a machine-readable result without host-specific absolute artifacts."""
        return {
            "target": self.target,
            "adapter": self.adapter,
            "adapter_version": self.adapter_version,
            "renderer": {
                "name": self.renderer,
                "version": self.renderer_version,
            },
            "rendered": self.rendered,
            "artifacts": [
                path.relative_to(self.output_path).as_posix() for path in self.artifacts
            ],
            "capabilities": self.capability_report.to_dict(),
        }


def export_restored_optical(
    target: str,
    optical_zarr: str | Path,
    output_directory: str | Path,
    *,
    render: bool = False,
) -> ExportOutcome:
    """Restore ``optical_zarr`` and run exactly one optional adapter.

    This function is intentionally below the canonical pipeline boundary. Optional
    renderer modules are imported lazily by the adapters, and the restored Zarr asset
    is the sole scientific input.
    """
    source = Path(optical_zarr)
    output = Path(output_directory)
    volume = read_volume(source)
    if not isinstance(volume, OpticalPropertyVolume):
        raise ExportInputError("export input must be an optical-property volume")

    if target == "mitsuba":
        if render:
            report = render_mitsuba(volume, output).capability_report
        else:
            report = prepare_mitsuba_scene(volume, output).capability_report
        return ExportOutcome(
            target=target,
            output_path=output,
            adapter=MITSUBA_ADAPTER,
            adapter_version=MITSUBA_ADAPTER_VERSION,
            renderer="mitsuba",
            renderer_version=_module_version("mitsuba", "mitsuba"),
            capability_report=report,
            artifacts=_artifacts(output),
            rendered=render,
        )
    if target == "openvdb":
        if render:
            raise ExportInputError(
                "OpenVDB rendering is an external Blender/Cycles follow-up; "
                "--render is supported only for Mitsuba"
            )
        report = export_openvdb(volume, output).capability_report
        return ExportOutcome(
            target=target,
            output_path=output,
            adapter=OPENVDB_ADAPTER,
            adapter_version=OPENVDB_ADAPTER_VERSION,
            renderer="openvdb",
            renderer_version=_module_version("openvdb", "pyopenvdb"),
            capability_report=report,
            artifacts=_artifacts(output),
            rendered=False,
        )
    raise ExportInputError(f"unsupported export target: {target}")


def _artifacts(output: Path) -> tuple[Path, ...]:
    return tuple(sorted(item for item in output.rglob("*") if item.is_file()))


def _module_version(*names: str) -> str:
    for name in names:
        module = sys.modules.get(name)
        value = getattr(module, "__version__", None) if module is not None else None
        if value is None and module is not None:
            value = getattr(module, "LIBRARY_VERSION", None)
        if value is not None:
            if isinstance(value, tuple):
                return ".".join(str(part) for part in value)
            return str(value)
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


__all__ = [
    "ExportInputError",
    "ExportOutcome",
    "MitsubaDependencyError",
    "MitsubaExportError",
    "OpenVDBDependencyError",
    "OpenVDBExportError",
    "export_restored_optical",
]
