import os
import shutil
from pathlib import Path

import numpy as np
from loguru import logger
from ultralytics.engine.results import Results

from src.vision.yolo.wrapper import YOLOWrapper


PROJECT_ROOT = Path(__file__).parents[2]
MODEL_DIR = PROJECT_ROOT/"models"
MODEL_FILE = MODEL_DIR/"facedet.pt"


def download_model():

    if not os.path.isdir(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)
        logger.info(f"Created model directory: {MODEL_DIR}")

    if not os.path.basename(MODEL_FILE) in os.listdir(MODEL_DIR):
        from huggingface_hub import hf_hub_download

        repo_id = "arnabdhar/YOLOv8-Face-Detection"
        pt_name = "model.pt"

        hf_hub_download(repo_id, pt_name, local_dir=MODEL_DIR)
        logger.info(f"Downloaded model from hugging face hub.")

        os.rename(f"{MODEL_DIR}/model.pt", MODEL_FILE)
        shutil.rmtree(f"{MODEL_DIR}/.cache")
        logger.info(f"Model file renamed and cache removed.")


class Facedetector(YOLOWrapper):

    def __init__(self):
        download_model()
        super().__init__(model_pt=MODEL_FILE)

    def predict(self, image: np.ndarray, conf: float=0.25, tracking: bool=False) -> Results:
        """ NOTE: skim2048
        If tracking is true:
          return shape is [n, 7]
          : x1, y1, x2, y2, box id, confidence, and label.
        Else
          return shape is [n, 6]
          : x1, y1, x2, y2, confidence, and label.
        """
        results = super().predict(image, conf=conf, tracking=tracking)
        # boxes = results[0].boxes.data.cpu().numpy()

        return results