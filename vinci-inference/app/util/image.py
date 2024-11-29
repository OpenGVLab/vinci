import base64
import requests
import os
import cv2
import uuid
import subprocess

import numpy as np

from PIL import Image
from io import BytesIO

def url_to_ndarray(url):
    """Convert an image URL to an ndarray.
    """
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to download image from {url}, status: {response.status_code}")
        return None
    
    image = Image.open(BytesIO(response.content))
    return np.array(image)

def base64_to_ndarray(base64_str):
    """Convert a base64-encoded image string to an ndarray."""
    try:
        # 解码 base64 字符串
        image_data = base64.b64decode(base64_str)
        
        # 使用 BytesIO 处理解码后的二进制数据
        image = Image.open(BytesIO(image_data))
        
        # 将 PIL Image 转换为 NumPy 数组
        return np.array(image)
    
    except Exception as e:
        print(f"Failed to convert base64 string to image ndarray: {e}")
        return None

def convert_h256(src_path, dst_path):
    cmd = ["ffmpeg", "-y", "-i", src_path, "-vcodec", "h264", dst_path]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as ex:
        print(f'returncode:{ex.returncode}, \ncmd: {ex.cmd}, \noutput: {ex.output}, \nstderr: {ex.stderr}, \nstdout: {ex.stdout}'.format())
        raise ex 
    finally:
        os.remove(src_path)

def save_video(video, 
               local_path,
               size=None,
               fps: int=30,
               h_256: bool=True,
               rgb2bgr: bool=False):
    dir_path = os.path.dirname(local_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)

    save_path = local_path

    if h_256:
        save_path = "tmp-" + str(uuid.uuid4()) + ".mp4" 
    
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame1 = video[0]
        size = (frame1.shape[1], frame1.shape[0])
        video_writer = cv2.VideoWriter(save_path, fourcc, fps, size)
        
        for frame in video:
            if rgb2bgr:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)
    finally:
        if video_writer:
            video_writer.release()

    if h_256:
        convert_h256(save_path, local_path)