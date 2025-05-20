import numpy as np

from src.inference.common.error import ShapeError, GeometryError
from src.inference.common.image import validate_image_size


def validate_points_type(array: np.ndarray) -> None:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Expected type to be a np.ndarray, but got {type(array).__name__}.")
    if array.dtype != np.float32:
        raise TypeError(f"Expected dtype is np.float32, but got {array.dtype}.")


def validate_points_shape(array: np.ndarray) -> None:
    if array.ndim != 2 or array.shape[1] != 2:
        raise ShapeError(f"Expected shape is (n, 2), but got {array.shape}.")


def validate_points_geometry(array: np.ndarray, image_size: tuple[int, int]) -> None:
    x, y = array[:, 0], array[:, 1]
    width, height = image_size
    validate_image_size(width, height)
    if not np.all((array >= 0) & (x <= width) & (y <= height)):
        raise GeometryError("Some points are outside the image bounds.")


def validate_points(array: np.ndarray, image_size: tuple[int, int]) -> None:
    validate_points_type(array)
    validate_points_shape(array)
    validate_points_geometry(array, image_size)


def to_relative(image_size: tuple[int, int], points: np.ndarray, ensure_valid_inputs: bool=False) -> None:
    width, height = image_size

    if ensure_valid_inputs:
        validate_image_size(width, height)
        validate_points(points, image_size)

    factor = np.array([1 / width, 1 / height], dtype=np.float32)
    points *= factor


def to_absolute(image_size: tuple[int, int], points: np.ndarray, ensure_valid_inputs: bool=False) -> None:
    width, height = image_size

    if ensure_valid_inputs:
        validate_image_size(width, height)
        validate_points_type(points)
        validate_points_shape(points)

    factor = np.array([width, height], dtype=np.float32)
    points *= factor

    if ensure_valid_inputs:
        validate_points_geometry(points, image_size)
