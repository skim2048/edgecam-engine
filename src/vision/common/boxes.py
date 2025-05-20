import cv2
import numpy as np


def validate_image_size(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError(f"Image width and height must be positive, but got width={width}, height={height}.")


def validate_image_format(array: np.ndarray) -> None:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Expected array to be a np.ndarray, but got {type(array).__name__}.")
    if array.ndim != 3:
        raise ValueError(f"Expected image array to have 3 dimensions, but got {array.ndim}.")

    height, width, channels = array.shape
    if channels != 3:
        raise ValueError(f"Expected 3 color channels, but got {channels}. Please check the image shape (height, width, channels).")
    validate_image_size(width, height)


def validate_boxes_format(array: np.ndarray) -> None:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Expected array to be a np.ndarray, but got {type(array).__name__}.")
    if array.dtype != np.float32:
        raise TypeError(f"Expected dtype is np.float32, but got {array.dtype}.")
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(f"Expected shape is (n, 4), but got {array.shape}.")

    x1, y1, x2, y2 = array[:, 0], array[:, 1], array[:, 2], array[:, 3]
    if not np.all(x1 < x2) or not np.all(y1 < y2):
        raise ValueError("Some boxes have x1 >= x2 or y1 >= y2.")


def validate_vertexes_format(array: np.ndarray) -> None:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"Expected array to be a np.ndarray, but got {type(array).__name__}.")
    if array.dtype != np.float32:
        raise TypeError(f"Expected dtype is np.float32, but got {array.dtype}.")
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"Expected shape is (n, 2), but got {array.shape}.")


def validate_boxes_within_image(image_size: tuple[int, int], boxes: np.ndarray, ensure_valid_boxes: bool=False) -> None:
    width, height = image_size

    if ensure_valid_boxes:
        validate_image_size(width, height)
        validate_boxes_format(boxes)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    if not (
        np.all((0 <= x1) & (x1 <= width)) and np.all((0 <= x2) & (x2 <= width)) and
        np.all((0 <= y1) & (y1 <= height)) and np.all((0 <= y2) & (y2 <= height))
    ):
        raise ValueError("Some box coordinates are outside the image bounds.")



def clip_boxes_to_image(image_size: tuple[int, int], boxes: np.ndarray, ensure_valid_inputs: bool=False) -> None:
    width, height = image_size

    if ensure_valid_inputs:
        validate_image_size(width, height)
        validate_boxes_format(boxes)

    np.clip(boxes[:, 0], 0, width, out=boxes[:, 0])
    np.clip(boxes[:, 1], 0, height, out=boxes[:, 1])
    np.clip(boxes[:, 2], 0, width, out=boxes[:, 2])
    np.clip(boxes[:, 3], 0, height, out=boxes[:, 3])


def map_boxes_to_image(source_size: tuple[int, int], target_size: tuple[int, int], boxes: np.ndarray, ensure_valid_inputs: bool=False) -> None:
    source_width, source_height = source_size
    target_width, target_height = target_size

    if ensure_valid_inputs:
        validate_image_size(source_width, source_height)
        validate_image_size(target_width, target_height)
        validate_boxes_format(boxes)

    ratio_width = target_width / source_width
    ratio_height = target_height / source_height

    factor = np.array([ratio_width, ratio_height, ratio_width, ratio_height], dtype=np.float32)
    boxes *= factor


def scale_boxes(boxes: np.ndarray, multiplier: float=1.0, ensure_valid_boxes: bool=False) -> None:
    if ensure_valid_boxes:
        if multiplier <= 0.0:
            raise ValueError(f"Multiplier must be positive, but got {multiplier}.")
        if multiplier == 1.0:
            return
        validate_boxes_format(boxes)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]

    delta_widths = widths * (multiplier - 1.0)
    delta_heights = heights * (multiplier - 1.0)

    boxes[:, 0] -= delta_widths * 0.5
    boxes[:, 1] -= delta_heights * 0.5
    boxes[:, 2] += delta_widths * 0.5
    boxes[:, 3] += delta_heights * 0.5


def blur_box_regions(image: np.ndarray, boxes: np.ndarray, kernel_size: tuple[int, int]=(11, 11), ensure_valid_inputs: bool=False) -> None:
    if ensure_valid_inputs:
        for size in kernel_size:
            if size < 3 or size % 2 != 1:
                raise ValueError(f"Each element of the kernel size must be an odd number greater than 1, but got {kernel_size}")
        validate_image_format(image)
        validate_boxes_format(boxes)

    boxes = boxes.astype(np.int32)
    for (x1, y1, x2, y2) in boxes:
        box_region = image[y1:y2, x1:x2]
        image[y1:y2, x1:x2] = cv2.GaussianBlur(box_region, kernel_size, 0)


def cover_box_regions(image: np.ndarray, boxes: np.ndarray, ensure_valid_inputs: bool=False) -> None:
    if ensure_valid_inputs:
        validate_image_format(image)
        validate_boxes_format(boxes)

    boxes = boxes.astype(np.int32)
    for (x1, y1, x2, y2) in boxes:
        box_region = image[y1:y2, x1:x2]
        image[y1:y2, x1:x2] = np.zeros_like(box_region)


def pixelate_box_regions(image: np.ndarray, boxes: np.ndarray, pixel_size: int=5, ensure_valid_inputs: bool=False) -> None:
    if ensure_valid_inputs:
        if pixel_size <= 1:
            raise ValueError(f"The pixel size must be greater than 1, but got {pixel_size}")
        validate_image_format(image)
        validate_boxes_format(boxes)

    boxes = boxes.astype(np.int32)
    for (x1, y1, x2, y2) in boxes:
        box_region = image[y1:y2, x1:x2]
        h, w = box_region.shape[:2]
        box_region = cv2.resize(box_region, (max(1, w // pixel_size), max(1, h // pixel_size)), interpolation=cv2.INTER_LINEAR)
        image[y1:y2, x1:x2] = cv2.resize(box_region, (w, h), interpolation=cv2.INTER_NEAREST)


def create_polygon_roi_mask(image_size: tuple[int, int], rel_vertexes: np.ndarray, ensure_valid_inputs: bool=False) -> np.ndarray:
    width, height = image_size

    if ensure_valid_inputs:
        validate_image_size(width, height)
        validate_vertexes_format(rel_vertexes)

    roi_mask = np.zeros(shape=(height, width), dtype=np.uint8)
    if len(rel_vertexes) >= 3:  # polygon
        abs_vertexes = (rel_vertexes * np.array([width, height])).astype(np.int32)
        roi_mask = cv2.fillPoly(roi_mask, [abs_vertexes.reshape(-1, 1, 2)], 255)

    return roi_mask


def exclude_boxes(boxes:np.ndarray, roi_mask: np.ndarray, ensure_valid_inputs: bool=False) -> np.ndarray:
    if ensure_valid_inputs:
        validate_boxes_format()