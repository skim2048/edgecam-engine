import numpy as np

from src.inference.common.error import SizeError, ShapeError


def validate_image_size(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise SizeError(f"Image size must be positive, but got width={width}, height={height}.")


def validate_image_type(array: np.ndarray) -> None:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Expected type to be a np.ndarray, but got {type(array).__name__}.")


def validate_image_shape(array: np.ndarray) -> None:
    if array.ndim != 3:
        raise ShapeError(f"Expected image array to have 3 dimensions, but got {array.ndim}.")
    height, width, channels = array.shape
    validate_image_size(width, height)
    if array.shape[-1] != 3:
        raise ShapeError(f"Expected 3 color channels, but got {channels}.")


def validate_image(array: np.ndarray) -> None:
    validate_image_type(array)
    validate_image_shape(array)
