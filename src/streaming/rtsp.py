from dataclasses import dataclass, fields
import json
import threading
import time
from typing import Callable, Any

import cv2
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
from loguru import logger
import numpy as np

from src.utils.buffer import EvictingQueue
from src.utils.task import SingleThreadTask


Gst.init(None)


with open("configs/chain.json", "r") as f:
    CHAIN = json.load(f)
AUDIO = CHAIN["audio"]
VIDEO = CHAIN["video"]


@dataclass
class AVPair:
    audio: Any
    video: Any

    @property
    def media_type(self) -> list[str]:
        return [f.name for f in fields(self)]
    
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        return setattr(self, key, value)


def get_chain(elements: list[str]) -> str:
    chain = ""
    for i, ele in enumerate(elements):
        name = f"name={ele}" if not i else ""
        chain += f"{ele} {name} ! "
    return chain


def valid_codec(codec: AVPair):
    if not codec.video or not codec.video in VIDEO:
        raise ValueError(f"Invalid or unsupported video codec: {codec.video}")
    if codec.audio and not codec.audio in AUDIO:
        raise ValueError(f"Unsupported audio codec: {codec.audio}")
    

def draw_circle(img: np.ndarray):
    img = img.copy()
    if img.ndim != 3:
        logger.warning("img.ndim != 3")
        return
    cy, cx = (np.array(img.shape[:2])/2).astype(int)
    cv2.circle(img, (cx, cy), 30, (0, 255, 0), 5)
    return img


def extract_codec(location: str, timeout: int=5) -> AVPair:
    # e.g. location = rtsp://admin:****@192.***.***.***:554/stream
    codec = AVPair(audio=None, video=None)
    lock = threading.Lock()
    cond = threading.Condition(lock)
    loop = GLib.MainLoop()

    def on_pad_added(ele, new_pad):
        struct = new_pad.query_caps(None).get_structure(0)
        media = struct.get_string("media")
        if media in codec.media_type:
            name = struct.get_string("encoding-name")
            with cond:
                codec[media] = name
                cond.notify_all()

    def wait_for_codec_extraction():
        with cond:
            start_t = time.time()
            while codec.video is None or codec.audio is None:
                elapsed = time.time() - start_t
                remaining = timeout - elapsed
                if remaining <= 0:
                    logger.warning("Timeout occurred during codec extraction.")
                    break
                cond.wait(timeout=remaining)
        loop.quit()

    pipe = Gst.parse_launch(f"rtspsrc name=src location={location} latency=200 ! fakesink")
    src = pipe.get_by_name('src')
    handler_id = src.connect('pad-added', on_pad_added)

    logger.info(f'Attempting to extract codecs within {timeout}s ...')
    pipe.set_state(Gst.State.PLAYING)

    wait_th = threading.Thread(target=wait_for_codec_extraction)
    wait_th.start()
    loop.run()
    wait_th.join()

    src.disconnect(handler_id)
    pipe.set_state(Gst.State.NULL)
    logger.info(f'Extracting complete: (video: {codec.video}, audio: {codec.audio})')
    return codec


class AVDecoder:
    def __init__(self):
        self._pipe: Gst.Pipeline = None
        self._vqueue: EvictingQueue = None
        self._aqueue: EvictingQueue = None
        self._handlers: list = []

    @property
    def video_queue(self) -> EvictingQueue | None:
        return self._vqueue

    @property
    def audio_queue(self) -> EvictingQueue | None:
        return self._aqueue

    def _handle_bus_message(self, bus, msg):
        if msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            logger.error(f"GStreamer: {err}: {dbg}")
            self.stop()
        elif msg.type == Gst.MessageType.EOS:
            logger.info("GStreamer: end of stream.")
            self.stop()

    def _handle_pad_added(self, src, new_pad, sink: AVPair):
        struct = new_pad.query_caps(None).get_structure(0)
        media = struct.get_string("media")
        if media in sink.media_type:
            if not sink[media]:
                return
            sink_pad = sink[media].get_static_pad("sink")
            if not sink_pad.is_linked():
                new_pad.link(sink_pad)

    def _handle_new_sample(self, appsink):
        sample = appsink.emit("pull-sample")
        buf = sample.get_buffer()
        ret, map_info = buf.map(Gst.MapFlags.READ)
        if not ret:
            return Gst.FlowReturn.ERROR
        struct = sample.get_caps().get_structure(0)
        caps_name = struct.get_name()
        try:
            if caps_name == "video/x-raw":
                w = struct.get_value("width")
                h = struct.get_value("height")
                shape = (h * 3 // 2, w)
                frame = np.frombuffer(map_info.data, dtype=np.uint8)
                frame = frame.reshape(shape)
                blob = (frame, buf.pts, buf.dts, buf.duration)
                self._vqueue.put(blob)
            elif caps_name == "audio/x-raw":
                audio = map_info.data
                blob = (audio, buf.pts, buf.dts, buf.duration)
                self._aqueue.put(blob)
        except:
            logger.exception("Exception occurred while processing the sample:")
        finally:
            buf.unmap(map_info)
        return Gst.FlowReturn.OK

    def _create_pipeline(self, location: str, codec: AVPair):
        sink = AVPair(audio=None, video=None)
        desc = f"rtspsrc name=rtspsrc location={location} latency=200 ! \n"
        if codec.video and codec.video in VIDEO:
            decode_chain = VIDEO[codec.video]["decode_chain"]
            desc += get_chain(decode_chain)
            # desc += "videoconvert ! "
            desc += "appsink name=vappsink emit-signals=true sync=false \n"
            self._vqueue = EvictingQueue()
            sink.video = decode_chain[0]
        if codec.audio and codec.audio in AUDIO:
            decode_chain = AUDIO[codec.audio]["decode_chain"]
            desc += get_chain(decode_chain)
            # desc += "audioconvert ! "
            desc += "audioresample ! "
            desc += "appsink name=aappsink emit-signals=true sync=false \n"
            self._aqueue = EvictingQueue()
            sink.audio = decode_chain[0]

        self._pipe = Gst.parse_launch(desc)
        if sink.video:
            sink.video = self._pipe.get_by_name(sink.video)
        if sink.audio:
            sink.audio = self._pipe.get_by_name(sink.audio)
        src = self._pipe.get_by_name("rtspsrc")
        handler_id = src.connect("pad-added", self._handle_pad_added, sink)
        self._handlers.append((src, handler_id))
        for name in ["vappsink", "aappsink"]:
            appsink = self._pipe.get_by_name(name)
            if appsink:
                handler_id = appsink.connect("new-sample", self._handle_new_sample)
                self._handlers.append((appsink, handler_id))

        bus = self._pipe.get_bus()
        bus.add_signal_watch()
        handler_id = bus.connect("message", self._handle_bus_message)        
        self._handlers.append((bus, handler_id))

    def start(self, location: str, codec: AVPair):
        self.stop()
        valid_codec(codec)
        self._create_pipeline(location, codec)
        self._pipe.set_state(Gst.State.PLAYING)

    def stop(self):
        for ele, handler_id in self._handlers:
            try:
                if isinstance(ele, Gst.Bus):
                    ele.disconnect(handler_id)
                    ele.remove_signal_watch()
                else:
                    ele.disconnect(handler_id)
            except Exception as e:
                logger.warning(f"Failed to release: {ele} ({handler_id}) → {e}")
        self._handlers.clear()

        if self._pipe:
            self._pipe.set_state(Gst.State.NULL)
            self._pipe = None
        self._vqueue = None
        self._aqueue = None


class AVEncoder():
    def __init__(self):
        self._pipe: Gst.Pipeline = None
        self._vappsrc: Gst.Element = None
        self._aappsrc: Gst.Element = None
        self._handlers: list = []

    @property
    def video_appsrc(self) -> Gst.Element | None:
        return self._vappsrc

    @property
    def audio_appsrc(self) -> Gst.Element | None:
        return self._aappsrc

    def _handle_bus_message(self, bus, msg):
        if msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            logger.error(f"[Gst Error] {err}: {dbg}")
            self.stop()
        elif msg.type == Gst.MessageType.EOS:
            logger.info("[Gst EOS] End of stream")
            self.stop()

    def _create_pipeline(self, codec: AVPair, caps_str: AVPair):
        desc_lines: list[str] = []
        # --- 비디오 ---
        if codec.video in VIDEO:
            encode_chain = VIDEO[codec.video]["encode_chain"]
            # 루트
            desc_lines.append(
                "appsrc name=vsrc is-live=true block=true format=time ! "
                "queue ! "
                "videoconvert ! "
                "tee name=vtee"
            )
            # 루트 -> 메인 스트림
            main_chain = " ! ".join(encode_chain)
            desc_lines.append(
                "vtee. ! "
                "queue ! "
                "videoconvert ! "
                f"{main_chain} ! "
                "mux_main."
            )
            # 루트 -> 서브 스트림
            desc_lines.append(
                "vtee. ! "
                "queue ! "
                "videoscale ! "
                "video/x-raw,width=640,height=360 ! "
                "videoconvert ! "
                f"{main_chain} ! "
                "mux_sub."
            )
        # --- 오디오 ---
        if codec.audio in AUDIO:
            encode_chain = AUDIO[codec.audio]["encode_chain"]
            # 루트
            desc_lines.append(
                "appsrc name=asrc is-live=true block=true format=time ! "
                "queue ! "
                "audioconvert ! "
                "tee name=atee"
            )
            audio_chain = " ! ".join(encode_chain)
            # 루트 -> 메인 스트림
            desc_lines.append(
                "atee. ! "
                "queue ! "
                "audioconvert ! "
                f"{audio_chain} ! "
                "mux_main."
            )
            # 루트 -> 서브 스트림
            desc_lines.append(
                "atee. ! "
                "queue ! "
                "audioconvert ! "
                f"{audio_chain} ! "
                "mux_sub."
            )

        # 최종: 비디오+오디오 메인 스트림 (live/main)
        desc_lines.append(
            "flvmux name=mux_main streamable=true latency=1000 ! "
            "queue ! "
            "rtmpsink location=rtmp://localhost:1935/live/main"
        )

        # 최종: 비디오+오디오 서브 스트림 (live/sub)
        desc_lines.append(
            "flvmux name=mux_sub  streamable=true latency=1000 ! "
            "queue ! "
            "rtmpsink location=rtmp://localhost:1935/live/sub"
        )

        desc = "\n".join(desc_lines)

        self._pipe = Gst.parse_launch(desc)
        self._vappsrc = self._pipe.get_by_name("vsrc")
        if self._vappsrc:
            self._vappsrc.set_property("caps", Gst.Caps.from_string(caps_str.video))
        self._aappsrc = self._pipe.get_by_name("asrc")
        if self._aappsrc:
            self._aappsrc.set_property("caps", Gst.Caps.from_string(caps_str.audio))

        bus = self._pipe.get_bus()
        bus.add_signal_watch()
        handler_id = bus.connect("message", self._handle_bus_message)
        self._handlers.append((bus, handler_id))

    def start(self, codec: AVPair, caps_str: AVPair):
        self.stop()
        valid_codec(codec)
        self._create_pipeline(codec, caps_str)
        self._pipe.set_state(Gst.State.PLAYING)

    def stop(self):
        for ele, handler_id in self._handlers:
            try:
                if isinstance(ele, Gst.Bus):
                    ele.disconnect(handler_id)
                    ele.remove_signal_watch()
                else:
                    ele.disconnect(handler_id)
            except Exception as e:
                logger.warning(f"Failed to release: {ele} ({handler_id}) → {e}")
        self._handlers.clear()

        if self._pipe:
            self._pipe.set_state(Gst.State.NULL)
            self._pipe = None
        self._vappsrc = None
        self._aappsrc = None


class RTSPStreamer():

    def __init__(self, image_processor: Callable, error_handler: Callable):
        self._loop: GLib.MainLoop | None = None
        self._loop_th: threading.Thread | None = None

        self._decoder: AVDecoder = None
        self._encoder: AVEncoder = None

        self._caps_str = AVPair(
            audio="audio/x-raw,format=S16LE,channels=1,rate=8000,layout=interleaved",
            video=None
        )
        self._caps_ready = threading.Event()

        # NOTE 1
        # 비디오 스트림은 비디오 큐(vqueue)에서 이미지 처리기(image processor) 콜백을 거친
        # 다음, 이미지 큐(iqueue)와 웹소켓 큐(websocket queue)에 각각 저장된다.
        # 전자는 인코더의 비디오 앱소스(vappsrc)에 입력될 데이터로, 후자는 웹 기반 콘솔에서
        # 화면을 표시할 입력 데이터로 사용된다.
        # 
        # 오디오 스트림은 웹 기반 콘솔에서 재생되지 않으므로 인코더의 오디오 앱소스(aappsrc)
        # 입력으로만 사용된다. 아래의 도식은 비디오 및 오디오 스트림의 흐름을 나타낸다.
        #
        # (1) ... -> vqueue -> [image_processor] -> iqueue -> vappsrc -> ...
        # (2) ... -> vqueue -> [image_processor] -> wqueue -> ...
        # (3) ... -> aqueue --------------------------------> aappsrc -> ...
        self._image_processor = image_processor
        self._iqueue: EvictingQueue = None  # image queue
        self._wqueue: EvictingQueue = None  # websocket queue

        # NOTE 2
        # 이미지 공급자(image_feeder)는 vqueue에서 비디오 프레임 하나를 가져와 이미지 처리기를
        # 이용해 처리한 다음 iqueue와 wqueue에 공급하는 작업을 반복하는 스레드이다.
        # 
        # 비디오 공급자(video_feeder)는 iqueue에서 이미지 하나를 가져와 vappsrc에 공급하는
        # 작업을 반복하는 스레드이고, 오디오 공급자는(audio_feeder)는 aqueue에서 오디오 샘플
        # 하나를 가져와 aappsrc에 공급하는 작업을 반복하는 스레드이다.
        self._image_feeder: SingleThreadTask = None
        self._video_feeder: SingleThreadTask = None
        self._audio_feeder: SingleThreadTask = None

        # NOTE 3
        # 에러 핸들러(error_handler)는 각 공급자가 큐에서 데이터를 가져올 때 예외가 발생할
        # 경우 이를 제어하기 위한 콜백이다. 덧붙이자면, 발생 가능한 예외는 EvictingQueue의
        # 함수 get()의 'Timeout' 또는 어떤 이유에 의해 EvictingQueue가 None이 되었을 때
        # None 타입은 어트리뷰트 get()을 갖지 않는다는 'NoneType'이다. 
        self._error_handler = error_handler

    @property
    def websocket_queue(self) -> EvictingQueue | None:
        return self._wqueue

    def _job_image_feeding(self):
        try:
            blob = self._decoder.video_queue.get()
        except Exception as e:
            # Timeout or NoneType Error
            if self._error_handler:
                self._error_handler(e)
            return
        frame_yuv, pts, dts, duration = blob
        frame_bgr = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2BGR_I420)
        if not self._caps_str.video:
            h, w = frame_bgr.shape[:2]
            self._caps_str.video = f"video/x-raw,format=I420,width={w},height={h}"
            self._caps_ready.set()

        # Image processing
        try:
            res = self._image_processor(frame_bgr)
        except:
            logger.exception("INFERENCE EEROR:")
            res = draw_circle(frame_bgr)
        frame_bgr = res

        # Feed an image to the websocket queue
        _, jpg_buffer = cv2.imencode(".jpg", frame_bgr)
        jpg_bytes = jpg_buffer.tobytes()
        self._wqueue.put(jpg_bytes)

        # Feed an image to the image queue
        frame_yuv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV_I420)
        frame_bytes = frame_yuv.tobytes()
        blob = (frame_bytes, pts, dts, duration)
        self._iqueue.put(blob)

    def _feed_video_to_appsrc(self, appsrc):
        try:
            blob = self._iqueue.get()
        except Exception as e:
            # Timeout or NoneType Error
            if self._error_handler:
                self._error_handler(e)
            return
        frame_bytes, pts, dts, duration = blob
        buf = Gst.Buffer.new_allocate(None, len(frame_bytes), None)
        buf.fill(0, frame_bytes)
        buf.pts = pts
        buf.dts = dts
        buf.duration = duration
        appsrc.emit("push-buffer", buf)

    def _feed_audio_to_appsrc(self, appsrc):
        try:
            blob = self._decoder.audio_queue.get()
        except Exception as e:
            # Timeout or NoneType Error
            if self._error_handler:
                self._error_handler(e)
            return
        audio_bytes, pts, dts, duration = blob
        buf = Gst.Buffer.new_allocate(None, len(audio_bytes), None)
        buf.fill(0, audio_bytes)
        buf.pts = pts
        buf.dts = dts
        buf.duration = duration
        appsrc.emit("push-buffer", buf)

    def start_streaming(self, location: str):
        if self._loop is None:
            self._loop = GLib.MainLoop()
            self._loop_th = threading.Thread(target=self._loop.run)
            self._loop_th.start()

        codec = extract_codec(location)

        self._decoder = AVDecoder()
        self._decoder.start(location, codec)

        if self._decoder.video_queue:
            self._wqueue = EvictingQueue()
            self._iqueue = EvictingQueue()
            self._image_feeder = SingleThreadTask("image_feeder")
            self._image_feeder.start(self._job_image_feeding)
        
        self._caps_ready.wait()

        self._encoder = AVEncoder()
        self._encoder.start(codec, self._caps_str)

        if self._encoder.video_appsrc:
            self._video_feeder = SingleThreadTask("video_feeder")
            self._video_feeder.start(
                self._feed_video_to_appsrc, [self._encoder.video_appsrc])
        if self._encoder.audio_appsrc:
            self._audio_feeder = SingleThreadTask("audio_feeder")
            self._audio_feeder.start(
                self._feed_audio_to_appsrc, [self._encoder.audio_appsrc])

    def stop_streaming(self):
        if self._video_feeder:
            self._video_feeder.stop()
            self._video_feeder = None
        if self._audio_feeder:
            self._audio_feeder.stop()
            self._audio_feeder = None
        if self._image_feeder:
            self._image_feeder.stop()
            self._image_feeder = None

        if self._loop:
            self._loop.quit()
            if self._loop_th:
                self._loop_th.join()
            self._loop = None
            self._loop_th = None

        if self._encoder:
            self._encoder.stop()
            self._encoder = None
        if self._decoder:
            self._decoder.stop()
            self._decoder = None
