import time
import threading

from loguru import logger
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib


Gst.init(None)


def extract_av_codec(location: str, timeout: int=10) -> dict[str, str]:

    def on_pad_added(element, pad):
        structure = pad.query_caps(None).get_structure(0)
        media = structure.get_string("media")
        if media in codec:
            encoding_name = structure.get_string("encoding-name")
            with waiting_condition:
                codec[media] = encoding_name
                waiting_condition.notify_all()

    def wait_for_codec_extraction():
        with waiting_condition:
            start_time = time.time()
            while not codec["video"] or not codec["audio"]:
                elapsed_time = time.time() - start_time
                remaining_time = timeout - elapsed_time
                if remaining_time <= 0:
                    logger.warning("Timeout occurred during codec extraction")
                    break
                waiting_condition.wait(timeout=remaining_time)
        loop.quit()

    codec = {"video": "", "audio": ""}

    loop = GLib.MainLoop()    
    waiting_lock = threading.Lock()
    waiting_condition = threading.Condition(waiting_lock)

    description = f"rtspsrc name=s location={location} latency=200 ! fakesink"
    pipeline = Gst.parse_launch(description)
    element_rtspsrc = pipeline.get_by_name("s")
    handler_id = element_rtspsrc.connect("pad-added", on_pad_added)

    pipeline.set_state(Gst.State.PLAYING)
    logger.info(f"Attempting to extract codecs within {timeout} seconds ...")

    waiting_thread = threading.Thread(target=wait_for_codec_extraction)
    waiting_thread.start()
    loop.run()
    waiting_thread.join()

    element_rtspsrc.disconnect(handler_id)
    pipeline.set_state(Gst.State.NULL)
    logger.info(f"Done: (Video: {codec["video"]}, Audio: {codec["audio"]})")

    return codec
