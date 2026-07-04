"""Shared NumPy validation helpers for canonical volume containers."""

from collections.abc import Sequence
from typing import TypeVar, cast

import numpy as np
import numpy.typing as npt

from .errors import VolumeValidationError

_Scalar = TypeVar("_Scalar", bound=np.generic)
UInt16Array = npt.NDArray[np.uint16]
Float32Array = npt.NDArray[np.float32]


def require_uint16_array(
    value: object, *, field_path: str, shape: Sequence[int]
) -> UInt16Array:
    """Require an exact canonical uint16 ndarray and shape."""
    array = _require_array(value, field_path=field_path)
    _require_dtype(array, field_path=field_path, expected=np.dtype(np.uint16))
    _require_shape(array, field_path=field_path, expected=shape)
    return cast(UInt16Array, array)


def require_float32_array(
    value: object, *, field_path: str, shape: Sequence[int]
) -> Float32Array:
    """Require an exact canonical float32 ndarray, shape, and finite values."""
    array = _require_array(value, field_path=field_path)
    _require_dtype(array, field_path=field_path, expected=np.dtype(np.float32))
    _require_shape(array, field_path=field_path, expected=shape)
    float_array = cast(Float32Array, array)
    raise_for_mask(
        field_path,
        float_array,
        ~np.isfinite(float_array),
        "values must be finite",
    )
    return float_array


def readonly_copy(array: npt.NDArray[_Scalar]) -> npt.NDArray[_Scalar]:
    """Copy an already validated array into read-only C-contiguous storage."""
    copied = np.array(array, copy=True, order="C", subok=False)
    copied.setflags(write=False)
    return cast(npt.NDArray[_Scalar], copied)


def raise_for_mask(
    field_path: str,
    values: npt.NDArray[np.generic],
    invalid_mask: npt.NDArray[np.bool_],
    message: str,
) -> None:
    """Raise a structured error summarizing a Boolean invalid-value mask."""
    invalid_count = int(np.count_nonzero(invalid_mask))
    if invalid_count == 0:
        return
    flat_index = int(np.argmax(invalid_mask))
    first_index = tuple(
        int(item) for item in np.unravel_index(flat_index, invalid_mask.shape)
    )
    first_value = values[first_index]
    if isinstance(first_value, np.generic):
        first_value = first_value.item()
    raise VolumeValidationError(
        field_path,
        message,
        invalid_count=invalid_count,
        first_index=first_index,
        first_value=first_value,
    )


def _require_array(value: object, *, field_path: str) -> npt.NDArray[np.generic]:
    if not isinstance(value, np.ndarray):
        raise VolumeValidationError(field_path, "must be a NumPy ndarray")
    return cast(npt.NDArray[np.generic], value)


def _require_dtype(
    array: npt.NDArray[np.generic], *, field_path: str, expected: np.dtype[np.generic]
) -> None:
    if array.dtype != expected:
        raise VolumeValidationError(
            field_path,
            f"dtype must be {expected.name}; got {array.dtype.name}",
        )


def _require_shape(
    array: npt.NDArray[np.generic],
    *,
    field_path: str,
    expected: Sequence[int],
) -> None:
    expected_shape = tuple(int(item) for item in expected)
    if array.shape != expected_shape:
        raise VolumeValidationError(
            field_path,
            f"shape must be {expected_shape}; got {array.shape}",
        )
