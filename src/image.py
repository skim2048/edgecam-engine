import cv2
import numpy as np


""" NOTE: skim2048
Throughout this module, 'xyxy' refers to a bounding box format represented as [x1, y1, x2, y2], where:

    - (x1, y1) is the top-left point of the box,
    - (x2, y2) is the bottom-right point of the box.

All functions assume 'xyxy' to be a NumPy array of shape (n, 4), where each row is a bounding box.
"""


def clamp_xyxy(src: tuple[int, int], xyxy: np.ndarray) -> np.ndarray:
    if xyxy.ndim != 2 or xyxy.shape[1] != 4:
        raise ValueError(f"xyxy must be a 2D array with shape (n, 4), got shape {xyxy.shape}.")

    w_src, h_src = src

    xyxy[:, 0] = np.clip(xyxy[:, 0], 0, w_src - 1)  # x1
    xyxy[:, 1] = np.clip(xyxy[:, 1], 0, h_src - 1)  # y1
    xyxy[:, 2] = np.clip(xyxy[:, 2], 0, w_src - 1)  # x2
    xyxy[:, 3] = np.clip(xyxy[:, 3], 0, h_src - 1)  # y2

    return xyxy


def map_xyxy(src: tuple[int, int], dst: tuple[int, int], xyxy: np.ndarray, clamp: bool=False) -> np.ndarray:
    if len(src) != len(dst):
        raise ValueError(f"src and dst must have the same number of dimensions, got {len(src)} and {len(dst)}.")
    if len(src) != 2:
        raise ValueError(f"src and dst must be 2D coordinate spaces, got tuple of length {len(src)}.")
    if xyxy.ndim != 2 or xyxy.shape[1] != 4:
        raise ValueError(f"xyxy must be a 2D array with shape (n, 4), got shape {xyxy.shape}.")
    
    (w_src, h_src), (w_dst, h_dst) = src, dst
    w_ratio = w_dst / w_src
    h_ratio = h_dst / h_src

    factor = (w_ratio, h_ratio, w_ratio, h_ratio)

    mapped_xyxy = (xyxy * factor).astype(int)
    if clamp:
        mapped_xyxy = clamp_xyxy(dst, mapped_xyxy)

    return mapped_xyxy


def blur_xyxy(image: np.ndarray, xyxy: np.ndarray, kernel_size: tuple[int, int]=(51, 51), clamp: bool=False) -> np.ndarray:
    if clamp:
        src = image.shape[:2][::-1]
        xyxy = clamp_xyxy(src, xyxy)

    for (x1, y1, x2, y2) in xyxy:
        roi = image[y1:y2, x1:x2]  # RoI: Region of Interest; 관심영역.
        blurred_roi = cv2.GaussianBlur(roi, kernel_size, 0)
        image[y1:y2, x1:x2] = blurred_roi

    return image


def pixelate_xyxy(image: np.ndarray, xyxy: np.ndarray, pixel_size: int=5, clamp: bool=False) -> np.ndarray:
    if clamp:
        src = image.shape[:2][::-1]
        xyxy = clamp_xyxy(src, xyxy)

    for (x1, y1, x2, y2) in xyxy:
        roi = image[y1:y2, x1:x2]
        h, w = roi.shape[:2]

        # Resize down and then up
        temp = cv2.resize(roi, (max(1, w // pixel_size), max(1, h // pixel_size)), interpolation=cv2.INTER_LINEAR)
        pixelated_roi = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)

        image[y1:y2, x1:x2] = pixelated_roi

    return image


def cover_xyxy(image: np.ndarray, xyxy: np.ndarray, clamp: bool=False) -> np.ndarray:
    if clamp:
        src = image.shape[:2][::-1]
        xyxy = clamp_xyxy(src, xyxy)

    for (x1, y1, x2, y2) in xyxy:
        roi = image[y1:y2, x1:x2]
        image[y1:y2, x1:x2] = np.zeros_like(roi)

    return image


def expand_xyxy(xyxy: np.ndarray, scale_factor: float=0.2) -> np.ndarray:
    w_box = xyxy[:, 2] - xyxy[:, 0]
    h_box = xyxy[:, 3] - xyxy[:, 1]

    scaled_w_box = (w_box * scale_factor * 0.5).astype(int)
    scaled_h_box = (h_box * scale_factor * 0.5).astype(int)

    xyxy[:, 0] -= scaled_w_box
    xyxy[:, 1] -= scaled_h_box
    xyxy[:, 2] += scaled_w_box
    xyxy[:, 3] += scaled_h_box

    return xyxy
