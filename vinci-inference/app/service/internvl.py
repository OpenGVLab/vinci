from sse_starlette import ServerSentEvent
from typing import List

from data import IntervlInferenceResponse
from models import internvl_chat, internvl_stream_chat
from util import url_to_ndarray, base64_to_ndarray

def parse_image_from_urls(image_urls: List[str]):
    """
    从URL列表中解析图像。

    该函数接收一个图像URL列表，将每个URL解析为图像，并返回一个图像列表。
    如果URL无效，将抛出异常。
    """
    images = []
    for image in image_urls:
        if isinstance(image, str):
            image = url_to_ndarray(image)
            if image is None:
                raise Exception("Invalid image url")

        images.append(image)   

    return images

def parse_image_from_base64(image_base64: List[str]):
    """
    从Base64列表中解析图像。

    该函数接收一个Base64图像列表，将每个Base64图像解析为图像，并返回一个图像列表。
    如果Base64图像无效，将抛出异常。
    """
    images = []
    for image in image_base64:
        if isinstance(image, str):
            image = base64_to_ndarray(image)
            if image is None:
                raise Exception("Invalid image url")

        images.append(image)   

    return images


def inference(question: str, history: list, base64_frames: list, frames: list, session_id: str="default", 
              timestamp: int=0, silent: bool=False, model_index: int=0):
    if base64_frames:
        frames = parse_image_from_base64(base64_frames)
    else:
        frames = parse_image_from_urls(frames)

    answer, history = internvl_chat(question, frames, history, session_id, timestamp, silent, 
                                    model_index=model_index)
    return IntervlInferenceResponse(answer=answer, history=[[1, "test"]], session_id=session_id)

def stream_inference(question: str, history: list, base64_frames: list, frames: list, session_id: str="default", 
                     timestamp: int=0, silent: bool=False, model_index: int=0):
    if base64_frames:
        frames = parse_image_from_base64(base64_frames)
    else:
        frames = parse_image_from_urls(frames)

    for answer, history in internvl_stream_chat(question, frames, history, session_id, timestamp, silent, 
                                                model_index=model_index):
        yield ServerSentEvent(event="message", id=0,
                              data=IntervlInferenceResponse(answer=answer, history=[[1, "test"]], session_id=session_id).model_dump_json())

    yield ServerSentEvent(event="end", data="Connection closed")
