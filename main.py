import json
import time
import asyncio
import threading
from contextlib import asynccontextmanager

import cv2
import numpy as np
from loguru import logger
from fastapi import FastAPI, HTTPException, Body
from fastapi.websockets import WebSocket
from fastapi.websockets import WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState

from src.rtsp import RTSPStreamer
from src.yolo.facedet import Facedetector
from src.image import map_xyxy, expand_xyxy, pixelate_xyxy


# ============================================================================
""" NOTE 이 코드는 클라이언트로부터 서비스 요청이 매우 적은 특수한 환경을 가정하고 작성
되었습니다. 경합 조건(race condition)을 거의 핸들링하지 않으므로 클라이언트 요청이 많은
환경에서는 문제를 일으킬 여지가 많으니 반드시 코드를 수정하셔야 합니다.

아직 확인되지 않은 버그와 최적화되지 않은 라인이 존재할 가능성이 높습니다.
추후 반드시 충분히 테스트 되어야 합니다. 추가적으로 스트리밍을 Python 대신 Rust로 대체할
필요성이 있는지 검토 중 입니다.
"""

FACEDET: Facedetector = None    # 얼굴 탐지기
STREAMER: RTSPStreamer = None   # 스트림 송출기

IS_ROI_UPDATED: bool = False        # 식별 영역 업데이트 플래그
ROI_MASK: np.ndarray | None = None  # 식별 영역 마스크

IS_RESTARTING = False            # 서버 재시작 플래그
RESTART_LOCK = threading.Lock()  # 서버 재시작 요청의 경합 조건 제어용 락


# 저장된 스트림 설정 불러오기
with open("configs/stream.json", "r") as f:
    CFG = json.load(f)

LOCATION = CFG["location"]           # 카메라 RTSP URL
INFER_SIZE = CFG["infer_size"]       # 추론 시 적용되는 해상도 - 기본: 720p
CONF_THRES = CFG["conf_thres"]       # 추론 확신도의 임계값 - 기본: 25%
SCALE_FACTOR = CFG["scale_factor"]   # 비식별화 박스 사이즈 확대율 - 기본: 50%
PIXEL_SIZE = CFG["pixel_size"]       # 비식별화 픽셀 사이즈 - 기본: 25px
ROI = CFG["roi"]                     # 비식별화 영역 꼭지점들의 상대 좌표


# ============================================================================
""" NOTE 비식별화 작업과 관련된 함수들 입니다. """

def exclude_xyxy(xyxy: np.ndarray) -> np.ndarray:
    """ 식별 영역 안의 박스들을 전체 박스 목록에서 제거합니다. 이들은 비식별 대상이
    아닙니다. """
    if not ROI_MASK is None:
        center = np.zeros((len(xyxy), 2), dtype=int)
        center[:, 0] = ((xyxy[:, 0] + xyxy[:, 2]) / 2).astype(int)
        center[:, 1] = ((xyxy[:, 1] + xyxy[:, 3]) / 2).astype(int)
        inside_mask = ROI_MASK[center[:, 1], center[:, 0]] == 255
        xyxy = xyxy[~inside_mask]
    return xyxy


def update_mask(frame_shape: tuple[int, int]):
    """ 식별 영역을 갱신합니다. 식별 영역의 꼭지점들은 상대 좌표 값을 갖기 때문에 이를
    frame_shape에 대응하는 절대 좌표 값으로 변환해 주어야 합니다. """
    global ROI_MASK
    roi = np.array(ROI)
    if roi.shape[0] >= 3:
        frame_w, frame_h = frame_shape
        roi_pts = (roi * (frame_w, frame_h)).astype(int)
        mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        ROI_MASK = cv2.fillPoly(mask, [roi_pts.reshape(-1, 1, 2)], 255)
    else:
        ROI_MASK = None


def deidentify_face(frame: np.ndarray):
    """ 얼굴을 비식별화 합니다. """
    global IS_ROI_UPDATED

    frame_shape = frame.shape[:2][::-1]
    if IS_ROI_UPDATED:
        update_mask(frame_shape)
        IS_ROI_UPDATED = False

    is_infer_size = frame_shape == INFER_SIZE
    if not is_infer_size:
        res = FACEDET.predict(cv2.resize(frame, dsize=INFER_SIZE), conf=CONF_THRES)
    else:
        res = FACEDET.predict(frame, conf=CONF_THRES)
    xyxy = res[:, :4].astype(int)

    if not is_infer_size:
        xyxy = map_xyxy(INFER_SIZE, frame_shape, xyxy)
    xyxy = expand_xyxy(xyxy, SCALE_FACTOR)
    xyxy = exclude_xyxy(xyxy)

    frame = pixelate_xyxy(frame, xyxy, pixel_size=PIXEL_SIZE, clamp=True)
    return frame


# ============================================================================
""" NOTE 스트림 송출기 제어와 관련된 함수들 입니다. """

def start_streamer(location: str):
    """ 스트림 송출기를 가동합니다. """
    global STREAMER, IS_RESTARTING

    if STREAMER is not None:
        STREAMER.stop_streaming()

    def _error_handler(e: Exception):
        global IS_RESTARTING

        with RESTART_LOCK:
            if IS_RESTARTING:
                return
            IS_RESTARTING = True

        def _restart():
            global IS_RESTARTING
            logger.warning("Stream lost. Reconnecting in 5s ...")
            while True:
                time.sleep(5)
                try:
                    start_streamer(location)
                    break
                except Exception:
                    logger.warning(f"... failed. Retrying in 5s ...")
            with RESTART_LOCK:
                IS_RESTARTING = False

        threading.Thread(target=_restart, daemon=True).start()

    STREAMER = RTSPStreamer(deidentify_face, _error_handler)
    STREAMER.start_streaming(location)


def stop_streamer():
    """ 스트림 송출기를 멈추고 제거합니다. """
    global STREAMER

    if STREAMER is not None:
        try:
            STREAMER.stop_streaming()
        except Exception as e:
            logger.exception(f"main: stop_streamer: {e}")
        finally:
            STREAMER = None


# ============================================================================
""" NOTE FastAPI 서비스와 관련된 함수들 입니다. """

@asynccontextmanager
async def lifespan(app: FastAPI):
    global FACEDET
    FACEDET = Facedetector()
    if LOCATION:
        start_streamer(LOCATION)
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
    loc: str = data.get("location", "").strip()
    try:
        if loc:
            if not loc.startswith("rtsp://"):
                invalid_loc = f"Invalid stream location: '{loc}'. Must start with 'rtsp://'."
                raise HTTPException(status_code=400, detail=invalid_loc)
        else:
            if LOCATION:
                loc = LOCATION
            else:
                empty_loc = "Missing RTSP stream location. Please write a valid location field."
                raise HTTPException(status_code=400, detail=empty_loc)
        start_streamer(loc)
    except Exception as e:
        logger.exception("Failed to start streaming.")
        stop_streamer()
        internal_err = f"Internal server error while starting stream: {str(e)}"
        raise HTTPException(status_code=500, detail=internal_err)
    else:
        LOCATION = loc
    return { "status": "success", "message": "streaming started." }


@app.post("/streaming/stop")
async def stop_streaming():
    global STREAMER
    if STREAMER is not None:
        try:
            STREAMER.stop_streaming()
        except Exception as e:
            logger.exception("Failed to stop streaming.")
            internal_err = f"Internal server error while stopping stream: {str(e)}"
            raise HTTPException(status_code=500, detail=internal_err)
        finally:
            STREAMER = None
    return {"status": "success", "message": "streaming stopped." }


@app.post("/config/save")
async def save_config():
    new_cfg = {
        "location": LOCATION,
        "infer_size": INFER_SIZE,
        "conf_thres": CONF_THRES,
        "scale_factor" : SCALE_FACTOR,
        "pixel_size": PIXEL_SIZE,
        "roi": ROI
    }
    try:
        with open("configs/stream.json", "w") as f:
            json.dump(new_cfg, f, indent=4)
    except Exception as e:
        logger.exception("Failed to store configuration.")
        internal_err = f"Internal server error while store configuration: {str(e)}"
        raise HTTPException(status_code=500, detail=internal_err)
    return {"status": "success", "message": "configuration saved." }


@app.get("/get/roi")
async def get_roi():
    return {"status": "success", "rois": ROI}


@app.post("/update/roi")
async def update_roi(roi_data: dict=Body(...)):
    global ROI, IS_ROI_UPDATED
    ROI = roi_data.get("rois", [])
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


@app.websocket('/live')
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
        logger.info("Connection has closed.")
    except asyncio.CancelledError:
        logger.info("Connection has cancelled.")
    except AttributeError:
        pass
    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")
        if ws.application_state == WebSocketState.CONNECTED:
            try:
                await ws.close(code=1011)
            except RuntimeError:
                pass
    finally:
        if ws.application_state == WebSocketState.CONNECTED:
            try:
                await ws.close()
            except RuntimeError:
                pass


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        app='main:app',
        host='0.0.0.0',
        port=12921,
    )