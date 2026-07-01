"""Errors raised during mesh topology inspection and voxelization (ADR-006)."""


class MeshTopologyError(ValueError):
    """A mesh violates the watertight single-solid contract of ADR-006."""

    def __init__(
        self, field_path: str, message: str, *, count: int | None = None
    ) -> None:
        self.field_path = field_path
        self.message = message
        self.count = count
        suffix = f" (count={count})" if count is not None else ""
        super().__init__(f"{field_path}: {message}{suffix}")


class VoxelizationError(ValueError):
    """A voxelization argument or domain violates the ADR-006 contract."""

    def __init__(self, field_path: str, message: str) -> None:
        self.field_path = field_path
        self.message = message
        super().__init__(f"{field_path}: {message}")
