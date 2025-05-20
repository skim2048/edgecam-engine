import cv2
import numpy as np

from src.inference.common.image import validate_image_size
from src.inference.common.point import validate_points


def create_polygon_mask_image(target_size: tuple[int, int], points: np.ndarray, ensure_valid_inputs: bool=False) -> np.ndarray:
    width, height = target_size

    if ensure_valid_inputs:
        validate_image_size(width, height)
        validate_points(points, target_size)

    mask_image = np.zeros(shape=(height, width), dtype=np.uint8)
    if len(points) >= 3:
        mask_image = cv2.fillPoly(mask_image, [points.astype(np.int32).reshape(-1, 1, 2)], 255)

    return mask_image
