import cv2
import uuid
import requests
import os
import subprocess

import numpy as np

from PIL import Image

image_name = '4.png'
image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'demo', image_name))

# 读取图片并转换为 ndarray
def image_to_array(image_path):
    img = Image.open(image_path)
    img_array = np.array(img)
    return img_array

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


def seine_inference():
    image_array = image_to_array(image_path)
    input = {
        "prompt": "跳舞",
        "image": image_array.tolist()
    }

    fps = 8
    url = 'http://10.140.0.243:18080/api/v1/inference/seine'
    response = requests.post(url, json=input)

    # 检查响应
    if response.status_code == 200:
        print(f'Inference successfully.')
        result = response.json()
        output = result['output']
        output = np.array(output, dtype=np.uint8)
        save_video(output, 'vid.mp4', fps=fps)
    else:
        print('Failed to upload image:', response.status_code)

if __name__ == '__main__':
    seine_inference()