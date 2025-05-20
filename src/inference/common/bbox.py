import cv2
import numpy as np

from src.inference.common.error import ShapeError, GeometryError
from src.inference.common.image import validate_image_size, validate_image


def validate_bboxes_type(array: np.ndarray) -> None:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Expected type to be a np.ndarray, but got {type(array).__name__}.")
    if array.dtype != np.float32:
        raise TypeError(f"Expected dtype is np.float32, but got {array.dtype}.")


def validate_bboxes_shape(array: np.ndarray) -> None:
    if array.ndim != 2 or array.shape[1] != 4:
        raise ShapeError(f"Expected shape is (n, 4), but got {array.shape}.")


def validate_bboxes_geometry(array: np.ndarray, image_size: tuple[int, int]=None) -> None:
    x1, y1, x2, y2 = array[:, 0], array[:, 1], array[:, 2], array[:, 3]
    if not np.all(x1 < x2) or not np.all(y1 < y2):
        raise GeometryError("Some bbox have x1 >= x2 or y1 >= y2.")
    if image_size is not None:
        width, height = image_size
        validate_image_size(width, height)
        if not np.all((array >= 0) & (array[:, [0, 2]] <= width) & (array[:, [1, 3]] <= height)):
            raise GeometryError("Some bbox coordinates are outside the image bounds.")


def validate_bboxes(array: np.ndarray, image_size: tuple[int, int]=None) -> None:
    validate_bboxes_type(array)
    validate_bboxes_shape(array)
    validate_bboxes_geometry(array, image_size)


def clip_bboxes_to_image(image_size: tuple[int, int], bboxes: np.ndarray, ensure_valid_inputs: bool=False) -> None:
    width, height = image_size

    if ensure_valid_inputs:
        validate_image_size(width, height)
        validate_bboxes(bboxes)

    np.clip(bboxes[:, 0], 0, width, out=bboxes[:, 0])
    np.clip(bboxes[:, 1], 0, height, out=bboxes[:, 1])
    np.clip(bboxes[:, 2], 0, width, out=bboxes[:, 2])
    np.clip(bboxes[:, 3], 0, height, out=bboxes[:, 3])


def map_bboxes_to_target_image(source_size: tuple[int, int], target_size: tuple[int, int], bboxes: np.ndarray, ensure_valid_inputs: bool=False) -> None:
    source_width, source_height = source_size
    target_width, target_height = target_size

    if ensure_valid_inputs:
        validate_image_size(source_width, source_height)
        validate_image_size(target_width, target_height)
        validate_bboxes(bboxes, image_size=[source_width, source_height])

    ratio_width = target_width / source_width
    ratio_height = target_height / source_height

    factor = np.array([ratio_width, ratio_height, ratio_width, ratio_height], dtype=np.float32)
    bboxes *= factor


def scale_bboxes(bboxes: np.ndarray, multiplier: float=1.0, ensure_valid_inputs: bool=False) -> None:
    if ensure_valid_inputs:
        if multiplier <= 0.0:
            raise ValueError(f"Multiplier must be positive, but got {multiplier}.")
        if multiplier == 1.0:
            return
        validate_bboxes(bboxes)

    widths = bboxes[:, 2] - bboxes[:, 0]
    heights = bboxes[:, 3] - bboxes[:, 1]

    delta_widths = widths * (multiplier - 1.0)
    delta_heights = heights * (multiplier - 1.0)

    bboxes[:, 0] -= delta_widths * 0.5
    bboxes[:, 1] -= delta_heights * 0.5
    bboxes[:, 2] += delta_widths * 0.5
    bboxes[:, 3] += delta_heights * 0.5


def blur_bbox_regions(image: np.ndarray, bboxes: np.ndarray, kernel_size: tuple[int, int]=(11, 11), ensure_valid_inputs: bool=False) -> None:
    if ensure_valid_inputs:
        for size in kernel_size:
            if size < 3 or size % 2 != 1:
                raise ValueError(f"Each element of the kernel size must be an odd number greater than 1, but got {kernel_size}.")
        validate_image(image)
        validate_bboxes(bboxes, image.shape[:2][-1])

    for (x1, y1, x2, y2) in bboxes.astype(np.int32, copy=False):
        bbox_region = image[y1:y2, x1:x2]
        image[y1:y2, x1:x2] = cv2.GaussianBlur(bbox_region, kernel_size, 0)


def cover_bbox_regions(image: np.ndarray, bboxes: np.ndarray, ensure_valid_inputs: bool=False) -> None:
    if ensure_valid_inputs:
        validate_image(image)
        validate_bboxes(bboxes, image.shape[:2][-1])

    for (x1, y1, x2, y2) in bboxes.astype(np.int32, copy=False):
        image[y1:y2, x1:x2] = 0


def pixelate_bbox_regions(image: np.ndarray, bboxes: np.ndarray, pixel_size: int=5, ensure_valid_inputs: bool=False) -> None:
    if pixel_size <= 1:
        raise ValueError(f"Pixel size must be greater than 1, but got {pixel_size}.")

    if ensure_valid_inputs:
        validate_image(image)
        validate_bboxes(bboxes, image.shape[:2][-1])

    for (x1, y1, x2, y2) in bboxes.astype(np.int32, copy=False):
        bbox_region = image[y1:y2, x1:x2]
        width, height = x2 - x1, y2 - y1
        num_patches = (max(1, width // pixel_size), max(1, height // pixel_size))

        downsampled = cv2.resize(bbox_region, num_patches, interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(downsampled, (width, height), interpolation=cv2.INTER_NEAREST)
        image[y1:y2, x1:x2] = pixelated


# def exclude_boxes(boxes:np.ndarray, roi_mask: np.ndarray, ensure_valid_inputs: bool=False) -> np.ndarray:
#     if ensure_valid_inputs:
#         validate_boxes_format()