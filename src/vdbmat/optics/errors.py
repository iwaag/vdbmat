"""Errors raised by material-to-optical conversion."""


class OpticalMappingError(ValueError):
    """A mapping configuration or input compatibility failure."""

    def __init__(self, field_path: str, message: str) -> None:
        self.field_path = field_path
        self.message = message
        super().__init__(f"{field_path}: {message}")
