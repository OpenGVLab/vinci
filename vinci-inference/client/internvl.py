import cv2
import uuid
import requests
import os
import json
import subprocess
import sseclient

import numpy as np

from PIL import Image
from decord import VideoReader, cpu

video_name = 'demo1.mp4'
video_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'demo', video_name))

def resize_frame(frame, max_size=320):
    """
    Resize a single video frame, ensuring that its dimensions do not exceed max_size,
    while maintaining the aspect ratio.
    """
    image = Image.fromarray(frame)
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate new dimensions
    if orig_width > orig_height:
        new_width = min(orig_width, max_size)
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = min(orig_height, max_size)
        new_width = int(new_height * aspect_ratio)

    # Ensure dimensions do not exceed max_size
    if new_width > max_size:
        new_width = max_size
        new_height = int(new_width / aspect_ratio)
    if new_height > max_size:
        new_height = max_size
        new_width = int(new_height * aspect_ratio)

    # Resize the frame
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    resized_frame = np.array(resized_image)

    return resized_frame

def load_video_slice(video_path, step=10, num=10):
    vr = VideoReader(video_path, ctx=cpu(0))
    num_frames = len(vr)

    frames = []
    for i in range(1, num_frames, step):
        if len(frames) >= num:
            break
 
        frames.append(vr[i].asnumpy())

    return vr.get_avg_fps(), len(vr), frames

def send_inference(stream: bool=False):
    fps, num_frames, frames = load_video_slice(video_path)
    print(f"video fps: {fps}, num frames: {num_frames}")
    
    frames = [resize_frame(frame).tolist() for frame in frames]
    
    input = {
        "question": "视频内容是什么？",
        "history": [],
        "frames": frames,
    }

    print(f'request body size: {len(json.dumps(input).encode("utf-8"))}')
    url = 'http://10.140.0.243:18080/api/v1/inference/internvl'
    if stream:
        url += "/stream"

    return requests.post(url, json=input, stream=stream)

def internvl_inference():
    response = send_inference()

    # 检查响应
    if response.status_code == 200:
        print(f'Inference successfully.')
        result = response.json()
        print(result)
    else:
        print('Failed to upload image:', response.status_code)

def internvl_stream_inference():
    response = send_inference(stream=True)

    # 检查响应
    if response.status_code == 200:
        print(f'Inference successfully.')
        sse_client = sseclient.SSEClient(response)
        for event in sse_client.events():
            print(event.data)
    else:
        print('Failed to upload image:', response.status_code)

if __name__ == '__main__':
    internvl_stream_inference()