from loguru import logger
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

from src.buffer import EvictingQueue


Gst.init(None)


class AVDecoder():

    def __init__(self):
        self._pipeline = None

        self._audio_queue: None | EvictingQueue = None
        self._video_queue: None | EvictingQueue = None

        self._handlers: list = []

    @property
    def video_queue(self) -> EvictingQueue | None:
        return self._vq

    @property
    def audio_queue(self) -> EvictingQueue | None:
        return self._aq

    def _handle_bus_message(self, bus, msg):
        if msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            logger.error(f"GStreamer: {err}: {dbg}")
            self.stop()
        elif msg.type == Gst.MessageType.EOS:
            logger.info("GStreamer: End of stream.")
            self.stop()

    def _handle_pad_added(self, src, new_pad, sink: Pair):
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
                self._vq.put(blob)

            elif caps_name == "audio/x-raw":
                audio = map_info.data
                blob = (audio, buf.pts, buf.dts, buf.duration)
                self._aq.put(blob)
        except:
            logger.exception("Exception occurred while processing the sample:")
        finally:
            buf.unmap(map_info)

        return Gst.FlowReturn.OK

    def _create_pipeline(self, location: str, codec: Pair):
        sink = Pair(None, None)
        desc = f"rtspsrc name=rtspsrc location={location} latency=200 ! \n"

        if codec.video and codec.video in VIDEO:
            decode_set = VIDEO[codec.video]["decode_set"]
            desc += make_chain_desc(decode_set)
            desc += "videoconvert ! "
            desc += "appsink name=v_appsink emit-signals=true sync=false \n"
            self._vq = EvictingQueue()
            sink.video = decode_set[0]

        if codec.audio and codec.audio in AUDIO:
            decode_set = AUDIO[codec.audio]["decode_set"]
            desc += make_chain_desc(decode_set)
            desc += "audioconvert ! "
            desc += "audioresample ! "
            desc += "appsink name=a_appsink emit-signals=true sync=false \n"
            self._aq = EvictingQueue()
            sink.audio = decode_set[0]

        self._pipe = Gst.parse_launch(desc)
        if sink.video:
            sink.video = self._pipe.get_by_name(sink.video)
        if sink.audio:
            sink.audio = self._pipe.get_by_name(sink.audio)
        src = self._pipe.get_by_name("rtspsrc")
        handler_id = src.connect("pad-added", self._handle_pad_added, sink)
        self._handlers.append((src, handler_id))
        for name in ["v_appsink", "a_appsink"]:
            appsink = self._pipe.get_by_name(name)
            if appsink:
                handler_id = appsink.connect("new-sample", self._handle_new_sample)
                self._handlers.append((appsink, handler_id))

        bus = self._pipe.get_bus()
        bus.add_signal_watch()
        handler_id = bus.connect("message", self._handle_bus_message)        
        self._handlers.append((bus, handler_id))

    def start(self, location: str, codec: Pair):
        self.stop()
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
        self._vq = None
        self._aq = None