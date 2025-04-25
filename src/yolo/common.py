import gc

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results


class YOLOWrapper:

    def __init__(self, model_pt: str):
        self._model = YOLO(model_pt)

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