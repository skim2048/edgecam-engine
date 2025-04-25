import json
import asyncio
from contextlib import asynccontextmanager

import cv2
import numpy as np
from loguru import logger
from fastapi import FastAPI, Body
from fastapi.websockets import WebSocket
from fastapi.websockets import WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.rtsp import RTSPStreamer
from src.yolo.facedet import Facedetector
from src.image import map_xyxy, expand_xyxy, pixelate_xyxy


FACEDET: Facedetector | None = None
STREAMER: RTSPStreamer | None = None

RESOLUTION = (1280, 720)
EXPAND_SCALE_FACTOR = 0.5
BLUR_KERNEL_SIZE = (101, 101)
PIXEL_SIZE = 25

ROI: np.ndarray | None = None
IS_ROI_UPDATED: bool = False
ROI_MASK: np.ndarray | None = None


def exclude_xyxy(xyxy: np.ndarray) -> np.ndarray:
    if not ROI_MASK is None:
        center = np.zeros((len(xyxy), 2), dtype=int)
        center[:, 0] = ((xyxy[:, 0] + xyxy[:, 2]) / 2).astype(int)
        center[:, 1] = ((xyxy[:, 1] + xyxy[:, 3]) / 2).astype(int)
        inside_mask = ROI_MASK[center[:, 1], center[:, 0]] == 255
        xyxy = xyxy[~inside_mask]
    return xyxy


def update_mask(frame_shape: tuple[int, int]):
    global ROI_MASK
    if not ROI is None and ROI.shape[0] >= 3:
        frame_w, frame_h = frame_shape
        roi_pts = (ROI * (frame_w, frame_h)).astype(int)
        mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        ROI_MASK = cv2.fillPoly(mask, [roi_pts.reshape(-1, 1, 2)], 255)
    else:
        ROI_MASK = None


def deidentify_face(frame: np.ndarray):
    global IS_ROI_UPDATED

    frame_shape = frame.shape[:2][::-1]
    if IS_ROI_UPDATED:
        update_mask(frame_shape)
        IS_ROI_UPDATED = False

    is_720p = frame_shape == RESOLUTION
    if not is_720p:
        res = FACEDET.predict(cv2.resize(frame, dsize=RESOLUTION))
    else:
        res = FACEDET.predict(frame)
    xyxy = res[:, :4].astype(int)
    if not is_720p:
        xyxy = map_xyxy(RESOLUTION, frame_shape, xyxy)
    xyxy = expand_xyxy(xyxy, EXPAND_SCALE_FACTOR)
    xyxy = exclude_xyxy(xyxy)
    # frame = blur_xyxy(frame, xyxy, BLUR_KERNEL_SIZE, clamp=True)
    # frame = cover_xyxy(frame, xyxy, clamp=True)
    frame = pixelate_xyxy(frame, xyxy, pixel_size=PIXEL_SIZE, clamp=True)
    return frame


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ----- START -----
    global FACEDET, STREAMER
    FACEDET = Facedetector()
    STREAMER = RTSPStreamer(deidentify_face)

    with open("tmp.json", "r") as f:
        cam = json.load(f)["cam"]
    STREAMER.start_streaming(cam["104"])
    # -----------------
    yield
    # ------ STOP ------
    STREAMER.stop_streaming()
    # ------------------


app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],)


@app.post("/update/roi")
async def update_roi(roi_data: dict = Body(...)):
    global ROI, IS_ROI_UPDATED
    roi = np.array(roi_data.get("rois", []))
    if roi.ndim != 2:
        roi = np.zeros(shape=(0, 2))
    ROI = roi
    IS_ROI_UPDATED = True
    return {"status": "success", "message": "RoI updated." }


@app.websocket('/stream1')
async def websocket_endpoint(ws: WebSocket):

    async def _send():
        while True:
            frame = await asyncio.to_thread(STREAMER.websocket_queue.get)
            await ws.send_bytes(frame)
            await asyncio.sleep(0)

    await ws.accept()
    host = ws.client.host
    port = ws.client.port
    logger.info(f'{host}:{port} has been accepted!')

    try:
        await _send()
    except WebSocketDisconnect:
        logger.info('Connection has closed.')
    except asyncio.CancelledError:
        logger.info('Connection has cancelled.')
    except Exception as e:
        logger.exception(f'Unexpected error: {str(e)}')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        app='main:app',
        host='0.0.0.0',
        port=12921,
    )