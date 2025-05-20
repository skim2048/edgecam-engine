import os
import time
import psutil

import cv2
import numpy as np


class VideoMetrics:
    def __init__(self):
        self._fps = 0.0
        self._mem = 0.0  # GigaBytes
        self._tot_mem = psutil.virtual_memory().total / (1024 ** 3)

        self._position = (10, 30)
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._font_color = (0, 255, 0)
        self._font_scale = 0.7
        self._font_thickness = 2

        self._pid = os.getpid()
        self._process = psutil.Process(self._pid)
        self._prev_time = time.time()

    @property
    def fps(self):
        return self._fps

    @property
    def mem(self):
        return self._mem

    def update(self) -> None:
        curr_time = time.time()
        elapsed_time = curr_time - self._prev_time
        if elapsed_time > 0:
            self._fps = 1.0 / elapsed_time
        else:
            self._fps = 0.0
        self._mem = self._process.memory_info().rss / (1024 ** 3)
        self._prev_time = curr_time

    def draw_metrics(self, frame: np.ndarray, fps: bool=True, mem: bool=True):
        text = ""
        if fps:
            text += f"[FPS: {self._fps:.2f}]"
        if mem:
            ratio = self._mem / self._tot_mem * 100
            text += f" [MEM: {self._mem:.1f}/{self._tot_mem:.1f} GB ({ratio:.2f}%)]"
        cv2.putText(frame, text, self._position, self._font, self._font_scale, self._font_color, self._font_thickness)