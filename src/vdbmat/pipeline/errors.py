"""Errors raised while building, interpreting, or executing a pipeline."""


class PipelineConfigError(ValueError):
    """A pipeline configuration violates the Phase 1 contract (ADR-007/ADR-008)."""

    def __init__(self, field_path: str, message: str) -> None:
        self.field_path = field_path
        self.message = message
        super().__init__(f"{field_path}: {message}")


class PipelineRunError(RuntimeError):
    """A pipeline run could not be executed or published (ADR-007).

    ``stage`` names the stage that failed so a caller can attribute an error (for
    example, distinguishing an ``export`` failure from a canonical-stage failure).
    """

    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        self.message = message
        super().__init__(f"{stage}: {message}")
