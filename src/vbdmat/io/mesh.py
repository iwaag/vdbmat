"""Narrow, repository-owned STL reader (ADR-006).

Parses ASCII or binary STL into a raw triangle soup. Topology inspection and
voxelization semantics live in :mod:`vbdmat.voxelize`; this module only turns bytes
into triangle coordinates and rejects malformed payloads. No third-party mesh
dependency is used.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .errors import MeshReadError

_BINARY_HEADER_BYTES = 80
_BINARY_COUNT_BYTES = 4
_BINARY_TRIANGLE_BYTES = 50  # 12 float32 + 1 uint16 attribute byte count


@dataclass(frozen=True, slots=True)
class RawMesh:
    """An unwelded triangle soup in the STL's own source units."""

    triangles: npt.NDArray[np.float64]  # shape (m, 3, 3): [triangle, vertex, xyz]

    def __post_init__(self) -> None:
        array = np.asarray(self.triangles, dtype=np.float64)
        if array.ndim != 3 or array.shape[1:] != (3, 3):
            raise MeshReadError("triangles", "must have shape (m, 3, 3)")
        if not np.isfinite(array).all():
            raise MeshReadError("triangles", "vertex coordinates must be finite")
        object.__setattr__(self, "triangles", array)

    @property
    def triangle_count(self) -> int:
        """Return the number of triangles."""
        return int(self.triangles.shape[0])


def read_stl(path: str | Path) -> RawMesh:
    """Read an ASCII or binary STL file into a :class:`RawMesh`."""
    file_path = Path(path)
    try:
        data = file_path.read_bytes()
    except FileNotFoundError as error:
        raise MeshReadError("mesh", f"file not found: {file_path}") from error
    except OSError as error:
        raise MeshReadError("mesh", f"cannot read {file_path}: {error}") from error
    return read_stl_bytes(data)


def read_stl_bytes(data: bytes) -> RawMesh:
    """Parse STL bytes, auto-detecting the binary or ASCII encoding."""
    if _looks_like_binary(data):
        return _read_binary_stl(data)
    return _read_ascii_stl(data)


def _looks_like_binary(data: bytes) -> bool:
    if len(data) < _BINARY_HEADER_BYTES + _BINARY_COUNT_BYTES:
        return False
    count = struct.unpack_from("<I", data, _BINARY_HEADER_BYTES)[0]
    expected = (
        _BINARY_HEADER_BYTES
        + _BINARY_COUNT_BYTES
        + count * _BINARY_TRIANGLE_BYTES
    )
    if len(data) == expected:
        return True
    # A leading "solid" token is the ASCII marker only when the size does not match
    # the exact binary layout above.
    return not data[:5].lstrip().lower().startswith(b"solid")


_BINARY_RECORD = np.dtype(
    [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
)


def _read_binary_stl(data: bytes) -> RawMesh:
    count = struct.unpack_from("<I", data, _BINARY_HEADER_BYTES)[0]
    expected = (
        _BINARY_HEADER_BYTES
        + _BINARY_COUNT_BYTES
        + count * _BINARY_TRIANGLE_BYTES
    )
    if len(data) != expected:
        raise MeshReadError(
            "mesh",
            f"binary STL length {len(data)} does not match {count} triangles",
        )
    if count == 0:
        return RawMesh(np.empty((0, 3, 3), dtype=np.float64))
    records = np.frombuffer(
        data,
        dtype=_BINARY_RECORD,
        count=count,
        offset=_BINARY_HEADER_BYTES + _BINARY_COUNT_BYTES,
    )
    vertices = records["vertices"].astype(np.float64)
    return RawMesh(vertices)


def _read_ascii_stl(data: bytes) -> RawMesh:
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as error:
        raise MeshReadError("mesh", "ASCII STL must be ASCII text") from error

    vertices: list[tuple[float, float, float]] = []
    tokens = text.split()
    index = 0
    length = len(tokens)
    while index < length:
        if tokens[index] == "vertex":
            if index + 3 >= length:
                raise MeshReadError("mesh", "truncated vertex in ASCII STL")
            try:
                vertex = (
                    float(tokens[index + 1]),
                    float(tokens[index + 2]),
                    float(tokens[index + 3]),
                )
            except ValueError as error:
                raise MeshReadError(
                    "mesh", "non-numeric vertex coordinate in ASCII STL"
                ) from error
            vertices.append(vertex)
            index += 4
        else:
            index += 1

    if len(vertices) % 3 != 0:
        raise MeshReadError(
            "mesh", f"ASCII STL vertex count {len(vertices)} is not a multiple of 3"
        )
    if not vertices:
        return RawMesh(np.empty((0, 3, 3), dtype=np.float64))
    array = np.asarray(vertices, dtype=np.float64).reshape(-1, 3, 3)
    return RawMesh(array)
