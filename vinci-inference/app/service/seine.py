import cv2
import sys
import os
import uuid
import base64
import os.path as osp
import numpy as np

from PIL import Image

from util import url_to_ndarray, base64_to_ndarray, save_video, OssClient
from data import SeineInferenceResponse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'generation', 'seine-v2')))

print(sys.path)
from seine import gen, model_seine
from omegaconf import OmegaConf

omega_conf = OmegaConf.load('generation/seine-v2/configs/demo.yaml')
omega_conf.run_time = 13
omega_conf.input_path = ''
omega_conf.text_prompt = []
omega_conf.save_img_path = '.'

from util import OssClient

oss_client = None

access_key = os.getenv('access_key')
access_key_secret = os.getenv('access_key_secret')
endpoint = os.getenv('endpoint')
bucket = os.getenv('bucket')
cdn = os.getenv('cdn')
if access_key:
    oss_client = OssClient(access_key=access_key, 
                           access_key_secret=access_key_secret, 
                           endpoint=endpoint,
                           bucket=bucket,
                           cdn=cdn)

# 创建视频存储目录
video_save_dir = osp.join('/tmp', 'vinci')
if not osp.exists(video_save_dir):
    os.makedirs(video_save_dir)

def upload_video(video_path: str='vid.mp4', object_name = 'vinci/vid.mp4'):
    # save_video(frames, video_path, fps=fps, h_256=h_256)
    oss_client.put_object_from_file(object_name, video_path)

    return oss_client.sign_url(object_name, cdn=True)

def inference(prompt: str, base64_image, image):
    if base64_image:
        image = base64_to_ndarray(base64_image)
    else:
        image = url_to_ndarray(image)
    
    image = Image.fromarray(image)

    input_img_path = f'{str(uuid.uuid1())}.png'
    image.save(input_img_path)

    omega_conf.input_path = input_img_path
    omega_conf.text_prompt = [prompt]
    
    video_name = str(uuid.uuid1()) + '.mp4'
    video_path = osp.join(video_save_dir, video_name)
    gen(omega_conf, model_seine, save_path=video_path)
    
    # frames_array = None
    # try:
    #     # 视频文件路径
    #     video_path = 'vid.mp4'
    #     cap = cv2.VideoCapture(video_path)

    #     frames = []
    #     while True:
    #         ret, frame = cap.read()

    #         if not ret:
    #             break  

    #         frames.append(frame)

    #     frames_array = np.array(frames)
    # finally:
    #     cap.release()

    url = ""
    try:
        object_name = f'vinci/{video_name}'
        url = upload_video(video_path=video_path, object_name=object_name)
        print(f"generate video url: {url}")
    except Exception as e:
        print(e)

    # 清理临时文件
    if osp.exists(input_img_path):
        os.remove(input_img_path)
    if osp.exists(video_path):
        os.remove(video_path)

    return SeineInferenceResponse(video=url, video_base64="")