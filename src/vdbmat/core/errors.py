"""Structured validation errors for canonical volume fields."""

from collections.abc import Sequence


class VolumeValidationError(ValueError):
    """A volume invariant failure with machine-readable location details."""

    def __init__(
        self,
        field_path: str,
        message: str,
        *,
        invalid_count: int | None = None,
        first_index: Sequence[int] | None = None,
        first_value: object | None = None,
    ) -> None:
        self.field_path = field_path
        self.message = message
        self.invalid_count = invalid_count
        self.first_index = (
            tuple(int(item) for item in first_index)
            if first_index is not None
            else None
        )
        self.first_value = first_value
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        details: list[str] = []
        if self.invalid_count is not None:
            details.append(f"invalid_count={self.invalid_count}")
        if self.first_index is not None:
            details.append(f"first_index={self.first_index}")
        if self.first_value is not None:
            details.append(f"first_value={self.first_value!r}")
        suffix = f" ({', '.join(details)})" if details else ""
        return f"{self.field_path}: {self.message}{suffix}"
