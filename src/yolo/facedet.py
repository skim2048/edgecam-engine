import gc
import os
import shutil
from pathlib import Path

import cv2
import torch
import numpy as np
from loguru import logger
import ultralytics
from ultralytics.engine.results import Results


MODELS = Path(__file__).parents[2]/"models"
MODEL = "facedet.pt"
if not MODEL in os.listdir(MODELS):
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        "arnabdhar/YOLOv8-Face-Detection", "model.pt", local_dir=MODELS)
    os.rename(f"{MODELS}/model.pt", f"{MODELS}/{MODEL}")
    shutil.rmtree(f"{MODELS}/.cache")


class YOLOWrapper:

    def __init__(self, model_pt: str):
        self._model = ultralytics.YOLO(model_pt)

    def predict(self, image: np.ndarray, tracking: bool=False) -> Results:
        if tracking:
            results = self._model.track(image, persist=True, verbose=False)
        else:
            results = self._model.predict(image, verbose=False)
        return results

    def release(self):
        if next(self._model.parameters()).device.type == "cuda":
            self._model.to("cpu")
            torch.cuda.empty_cache()
        del self._model
        gc.collect()
        self._model = None


class Facedetector(YOLOWrapper):

    def __init__(self):
        super().__init__(model_pt=f"{MODELS}/{MODEL}")

    def predict(self, image: np.ndarray, tracking: bool=False) -> np.ndarray:
        results = super().predict(image, tracking)
        boxes = results[0].boxes.data.cpu().numpy()
        return boxes


def remap_points2d(src: tuple, dst: tuple, pts: np.ndarray) -> np.ndarray:
    if src == dst:
        return pts
    factor = np.array(dst) / np.array(src)
    factor = np.tile(factor[::-1], 2)
    return (pts * factor[::-1]).astype(int)


# def stretch_boxes2d(xyxys: np.ndarray, scale: float=0.05) -> np.ndarray:
#     w = xyxys[:, 2] - xyxys[:, 0]
#     h = xyxys[:, 3] - xyxys[:, 1]
#     scaled_w = (w * scale * 0.5).astype(int)
#     scaled_h = (h * scale * 0.5).astype(int)
#     xyxys[:, 0] -= scaled_w
#     xyxys[:, 1] -= scaled_h
#     xyxys[:, 2] += scaled_w
#     xyxys[:, 3] += scaled_h
#     return xyxys

def stretch_boxes2d(xyxys: np.ndarray, img_wh: tuple=None, scale: float=0.2) -> np.ndarray:
    w = xyxys[:, 2] - xyxys[:, 0]
    h = xyxys[:, 3] - xyxys[:, 1]
    scaled_w = (w * scale * 0.5).astype(int)
    scaled_h = (h * scale * 0.5).astype(int)
    xyxys[:, 0] -= scaled_w
    xyxys[:, 1] -= scaled_h
    xyxys[:, 2] += scaled_w
    xyxys[:, 3] += scaled_h

    if img_wh is not None:
        img_w, img_h = img_wh
        xyxys[:, 0] = np.clip(xyxys[:, 0], 0, img_w - 1)
        xyxys[:, 1] = np.clip(xyxys[:, 1], 0, img_h - 1)
        xyxys[:, 2] = np.clip(xyxys[:, 2], 0, img_w - 1)
        xyxys[:, 3] = np.clip(xyxys[:, 3], 0, img_h - 1)

    return xyxys


def blur_boxes2d(frame: np.ndarray, xyxys: np.ndarray):
    for xyxy in xyxys:
        x1, y1, x2, y2 = xyxy
        roi = frame[y1:y2, x1:x2]
        blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
        frame[y1:y2, x1:x2] = blurred_roi
    return frame


def deidentify_face(frame: np.ndarray, facedet: Facedetector):
    shape = frame.shape[:2][::-1]
    frame_720p = cv2.resize(frame, dsize=(1280, 720))
    xyxys = facedet.predict(frame_720p)[:, :4].astype(int)
    xyxys = remap_points2d((1280, 720), shape, xyxys)
    xyxys = stretch_boxes2d(xyxys, shape)
    frame = blur_boxes2d(frame, xyxys)
    return frame