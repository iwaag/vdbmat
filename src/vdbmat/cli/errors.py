"""CLI error categories and documented process exit codes (ADR-008 D3)."""

from dataclasses import dataclass
from enum import IntEnum


class ExitCode(IntEnum):
    """Stable Phase 1 process exit categories."""

    SUCCESS = 0
    INTERNAL = 1
    USAGE = 2
    VALIDATION = 3
    IO = 4
    CONVERSION = 5
    OPTIONAL_DEPENDENCY = 6


@dataclass(frozen=True, slots=True)
class CliError(Exception):
    """An expected CLI failure with a stable exit category."""

    code: ExitCode
    message: str

    def __str__(self) -> str:
        return self.message
