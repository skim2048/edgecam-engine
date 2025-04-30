import json
import asyncio
from contextlib import asynccontextmanager

import cv2
import numpy as np
from loguru import logger
from fastapi import FastAPI, HTTPException, Body
from fastapi.websockets import WebSocket
from fastapi.websockets import WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.rtsp import RTSPStreamer, FailedToExtract
from src.yolo.facedet import Facedetector
from src.image import map_xyxy, expand_xyxy, pixelate_xyxy


FACEDET: Facedetector | None = None
STREAMER: RTSPStreamer | None = None

IS_ROI_UPDATED: bool = False
ROI_MASK: np.ndarray | None = None


with open("configs/streaming.json", "r") as f:
    CFG = json.load(f)

LOCATION = CFG["location"]
INFER_SIZE = CFG["infer_size"]
CONF_THRES = CFG["conf_thres"]
SCALE_FACTOR = CFG["scale_factor"]
PIXEL_SIZE = CFG["pixel_size"]
ROI: np.ndarray = CFG["roi"] if CFG["roi"] else np.zeros(shape=(0, 2))


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

    is_720p = frame_shape == INFER_SIZE
    if not is_720p:
        res = FACEDET.predict(cv2.resize(frame, dsize=INFER_SIZE), conf=CONF_THRES)
    else:
        res = FACEDET.predict(frame, conf=CONF_THRES)
    xyxy = res[:, :4].astype(int)
    if not is_720p:
        xyxy = map_xyxy(INFER_SIZE, frame_shape, xyxy)
    xyxy = expand_xyxy(xyxy, SCALE_FACTOR)
    xyxy = exclude_xyxy(xyxy)
    frame = pixelate_xyxy(frame, xyxy, pixel_size=PIXEL_SIZE, clamp=True)
    return frame


def stop_streamer():
    global STREAMER
    if STREAMER is not None:
        try:
            STREAMER.stop_streaming()
        except:
            logger.warning("Unexpected error occurred while stopping stream.")
        finally:
            STREAMER = None


def start_streamer(location: str):
    global STREAMER
    stop_streamer()
    STREAMER = RTSPStreamer(deidentify_face)
    STREAMER.start_streaming(location)


def store_configuration():
    new_cfg = {
        "location": LOCATION,
        "infer_size": INFER_SIZE,
        "conf_thres": CONF_THRES,
        "scale_factor" : SCALE_FACTOR,
        "pixel_size": PIXEL_SIZE,
        "roi": ROI.tolist()
    }
    with open("configs/streaming.json", "w") as f:
        json.dump(new_cfg, f, indent=4)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global FACEDET
    FACEDET = Facedetector()
    yield
    stop_streamer()
    FACEDET.release()


app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],)


@app.post("/streaming/start")
async def start_streaming(data: dict=Body(...)):
    global LOCATION
    location = data["location"]
    try:
        loc = LOCATION if not location else location
        start_streamer(loc)
    except FailedToExtract as e:
        logger.exception(e)
        stop_streamer()
        raise HTTPException(
            status_code=400,
            detail=f"FastAPI - {str(e)}"
        )
    except Exception as e:
        logger.exception(e)
        stop_streamer()
        raise HTTPException(
            status_code=500,
            detail=f"FastAPI - {str(e)}"
        )
    else:
        LOCATION = loc
        store_configuration()
    return {"status": "success", "message": "streaming started." }


@app.post("/streaming/stop")
async def stop_streaming():
    global STREAMER
    if STREAMER is not None:
        try:
            STREAMER.stop_streaming()
        except Exception:
            logger.exception("Unexpected error occurred while stopping stream.")
        finally:
            STREAMER = None
    return {"status": "success", "message": "streaming stopped." }


@app.post("/configs/store")
async def store_configs():
    store_configuration()
    return {"status": "success", "message": "configs stored." }


@app.get("/get/roi")
async def get_roi():
    global ROI
    if type(ROI) == type([]):
        ROI = np.array(ROI)
    return {"status": "success", "rois": ROI.tolist()}


@app.post("/update/roi")
async def update_roi(roi_data: dict=Body(...)):
    global ROI, IS_ROI_UPDATED
    roi = np.array(roi_data.get("rois", []))
    if roi.ndim != 2:
        roi = np.zeros(shape=(0, 2))
    ROI = roi
    IS_ROI_UPDATED = True
    return {"status": "success", "message": "roi updated." }


# @app.post("/update/conf-thres")
# async def update_conf_thres(data: dict=Body(...)):
#     global CONF_THRES  # range: 0.0 ~ 1.0 (default: 0.25)
#     conf_thres = data["conf_thres"]
#     if conf_thres < 0.0 or conf_thres > 1.0:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Failed to update: confidence threshold {conf_thres} is out of range (0.0 ~ 1.0)."
#         )
#     else:
#         CONF_THRES = conf_thres
#         return {"status": "success", "message": "Confidence threshold updated."}


# @app.post("/update/scale-factor")
# async def update_scale_factor(data: dict=Body(...)):
#     global SCALE_FACTOR  # range: 0.0 ~ 1.0 (default: 0.5)
#     scale_factor = data["scale_factor"]
#     if scale_factor < 0.0 or scale_factor > 1.0:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Failed to update: scale factor {scale_factor} is out of range (0.0 ~ 1.0)."
#         )
#     else:
#         SCALE_FACTOR = scale_factor
#         return {"status": "success", "message": "Scale factor updated."}


# @app.post("/update/pixel-size")
# async def update_pixel_size(data: dict=Body(...)):
#     global PIXEL_SIZE  # range: 10 ~ 100 (default: 25)
#     pixel_size = data["pixel_size"]
#     if pixel_size < 10 or pixel_size > 100:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Failed to update: pixel size {pixel_size} is out of range (10 ~ 100)."
#         )
#     else:
#         PIXEL_SIZE = pixel_size
#         return {"status": "success", "message": "Pixel size updated."}


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
    finally:
        if not ws.client_state.name == "DISCONNECTED":
            await ws.close()


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        app='main:app',
        host='0.0.0.0',
        port=12921,
    )