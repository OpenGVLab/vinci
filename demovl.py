import torch, torchvision
torch.backends.cudnn.enabled = False
import gradio as gr
from gradio.themes.utils import colors, fonts, sizes
import os, subprocess
import threading

#generation module
import sys
sys.path.append('generation/seine-v2/')
# torchvision.set_video_backend('video_reader')
from seine import gen, model_seine
from omegaconf import OmegaConf
omega_conf = OmegaConf.load('generation/seine-v2/configs/demo.yaml')
omega_conf.run_time = 13
omega_conf.input_path = ''
omega_conf.text_prompt = []
omega_conf.save_img_path = '.'

import argparse

# Create ArgumentParser object
parser = argparse.ArgumentParser(description='Argument Parser Example')
parser.add_argument('--version', type=str, help='v0 or v1', default='v1')
args = parser.parse_args()
version = args.version


get_gr_video_current_time = """async (video, grtime, one, two, three) => {
  const videoEl = document.querySelector("#up_video video");
  return [video, videoEl.currentTime, one, two, three];
}"""

get_time = """async (video, grtime, one, two, three, four, five) => {
  const videoEl = document.querySelector("#up_video video");
  return [video, videoEl.currentTime, one, two, three, four, five];
}"""

import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from random import randint
from transformers import TextIteratorStreamer
from threading import Thread
import os

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


class Chat:
    def __init__(self, path='Vinci-8B-base', stream=True, device='cuda:0', use_chat_history=False, language='chn', version='v1', max_history=10):
        self.device = device
        self.vr = None
        self.video_fps = None
        self.prev_timestamp = 0
        self.history = []
        self.chat_history = []
        self.stream = stream
        self.use_chat_history = use_chat_history
        self.transform = build_transform(input_size=448)
        self.language = language
        self.version = version
        self.max_history = max_history
        self.model = AutoModel.from_pretrained(
                path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True)
        self.model_lock = threading.Lock()

        if version == 'v0':
            from safetensors.torch import load_file
            def merge_dicts(dict1, dict2, dict3, dict4):
                result = {**dict1, **dict2, **dict3, **dict4}
                return result
            path2 = 'Vinci-8B-ckpt'
            model_weights1 = load_file(os.path.join(path2,"model-00001-of-00004.safetensors"))
            model_weights2 = load_file(os.path.join(path2,"model-00002-of-00004.safetensors"))
            model_weights3 = load_file(os.path.join(path2,"model-00003-of-00004.safetensors"))
            model_weights4 = load_file(os.path.join(path2,"model-00004-of-00004.safetensors"))
            merged_weight = merge_dicts(model_weights1,model_weights2,model_weights3,model_weights4)
            self.model.wrap_llm_lora(r=16, lora_alpha=2 * 16)
            msg = self.model.load_state_dict(merged_weight,strict=False)
            print(msg)
        self.model = self.model.eval().cuda()
        state1 = self.model.state_dict()
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if self.stream:
            self.streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=10)
            self.generation_config = dict(
            num_beams=1,
            max_new_tokens=1024,
            do_sample=False,
            streamer=self.streamer
            )
        else:
            self.generation_config = dict(
            num_beams=1,
            max_new_tokens=1024,
            do_sample=False,
            )

    def load_video_timestamp(self, timestamp, num_segments=4):
        pixel_values_list, num_patches_list = [], []
        offset = np.linspace(-2, 0, num_segments)
        rand_offset = randint(-4, 4)
        offset = offset + rand_offset
        frame_indices = (timestamp+offset) * self.video_fps
        frame_indices = frame_indices.astype(np.int64)
        if frame_indices[0] < 0:
            frame_indices -= frame_indices[0]
        print('***** using video timestamps at:', frame_indices)
        for i, frame_index in enumerate(frame_indices):
            img = Image.fromarray(self.vr[frame_index].asnumpy()).convert('RGB')
            if i == len(frame_indices) - 1:
                img.save('./lastim.jpg')
            img = dynamic_preprocess(img, image_size=448, use_thumbnail=True, max_num=1)
            pixel_values = [self.transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list

    
    def ask(self,text,conv):
        conv['questions'].append(text + '\n')
        return conv

    def answer(self, conv, timestamp=0, add_to_history=False):
        with self.model_lock:
            pixel_values, num_patches_list = self.load_video_timestamp(timestamp)
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
            if add_to_history: # silent ask
                if self.language == 'chn':
                    question = video_prefix + '现在视频到了 %.1f 秒处. 简单的用一句话描述视频中我的动作.' % timestamp
                else:
                    question = video_prefix + 'Now the video is at %.1f second. Briefly describe my actions in the video in one sentence.' % timestamp #conv['questions'][-1]
                
                response, history = self.model.chat(self.tokenizer, pixel_values, question, self.generation_config,
                                   num_patches_list=num_patches_list,
                                   history=None, return_history=True)
                self.history.append((timestamp, response))
                print('VL_HISTORY:', self.history)
            else:
                self.chat_history.append([conv['questions'][-1]])
                question = self.add_history(conv['questions'][-1])
                question = video_prefix + question
                if self.stream:
                    thread = Thread(target=self.model.chat, kwargs=dict(tokenizer=self.tokenizer, pixel_values=pixel_values, question=question, generation_config=self.generation_config,
                                    num_patches_list=num_patches_list,
                                    history=None, return_history=False))
                    thread.start()
                    response = ''
                else:
                    response = self.model.chat(self.tokenizer, pixel_values, question, self.generation_config,
                                num_patches_list=num_patches_list,
                                history=None, return_history=False)
                    self.chat_history[-1].append(timestamp)
                    self.chat_history[-1].append(response)
                    # self.history.append((timestamp, response))
            conv['answers'].append(response + '\n')
            # print('Real question at %.1f is |||' % timestamp, question)
            # print('Answer at %.1f is ||| '%timestamp, response)
            # print('the history is:', self.history)
            return response, conv, './lastim.jpg'

    def add_history(self, question):
        if not self.history:
            print('history not added because self.history is empty')
            return question
        if len(self.history) > 0:
            if self.language == 'chn':
                system = "你是一个视频智能助手。仔细观察我拍摄的视频并重点关注物体的运动和人的动作。由于你看不到发生在当前帧之前的部分，现在以文字形式提供给你这个视频的之前的历史供参考："
            else:
                system = 'You are an intelligent assistant. You receive video frames from my egocentric viewpoint. Carefully watch the video and pay attention to the movement of objects, and the action of human. Since you cannot see the previous part of the video, I provide you the history of this video for reference. The history is: '
            res = system
            for hist in self.history:
                ts = hist[0]
                a = hist[1]
                if self.language == 'chn':
                    res += '当视频在%.1f秒时, 视频的内容是 "%s". ' % (ts, a.strip())
                else:
                    res += 'When the video is at %.1f seconds, the video contect is "%s". ' % (ts, a.strip())
            if self.language == 'chn':
                res += '以上是所有的视频历史, 表明了之前发生了什么.\n'
            else:
                res += 'This is the end of the video history that indicates what happened before.\n'
            if self.use_chat_history and len(self.chat_history)>1:
                if self.language == 'chn':
                    res += '另外我提供根据之前的视频,我们的对话历史如下: '
                else:
                    res += 'Also I provide you with our chat history based on the previous video content: '
                for hist in self.chat_history[:-1]:
                    q = hist[0]
                    ts = hist[1]
                    a = hist[2]
                    if self.language == 'chn':
                        res += '当视频在%.1f秒时, 问题是: "%s", 回答是"%s". '  % (ts, q.strip(), a.strip())
                    else:
                        res += 'When the video is at %.1f seconds, the question was: "%s", and its answer was: "%s". ' % (ts, q.strip(), a.strip())
                if self.language == 'chn':
                    res += '以上是所有的对话历史, 表明了之前我们交流了什么,但是不表明现在的任何信息.\n'
                else:
                    res += 'This is the end of the chat history. The chat history indicate what our previous chat was, but does not necessarily contain the current information.\n'

            # res += 'Now the video is at %0.1fs, the action is "%s". ' % (ts, a)
            # res += 'Given the video information, please answer my question: '
            if self.language == 'chn':
                res += '请根据当前视频, 同时参照视频历史, 用中文回答我的问题. 注意如果问题与之前发生的事情有关, 请参考视频历史, 否则请只关注当前图像信息. 我的问题是: "%s". 用三句话以内回答.' % question
            else:
                res += 'Given the current video and using the previous video as reference, answer my question in English: "%s". Note that if the question is about what has been previously done, please only focus on the history. Otherwise, please only focus on the question and the current video input. Do not repeat.' % question
            # question = res + '\n' + question
            question = res
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        return question

    def upload_video(self, video_path):
        self.vr = VideoReader(video_path, ctx=cpu(0))
        self.num_frames = len(self.vr)
        self.video_fps = self.vr.get_avg_fps()
        return 'succeed'


# ========================================
#             Model Initialization
# ========================================
def init_model():
    print('Initializing VLChat')
    chat = Chat(stream=False, version=args.version)
    print('Initialization Finished')
    return chat


# ========================================
#             Gradio Setting
# ========================================
def gradio_reset(chat_state):
    if chat_state is not None:
        chat_state.messages = []
    # if img_list is not None:
    #     img_list = []
    return gr.update(value=None, interactive=True), gr.update(value=None, interactive=True), gr.update(placeholder='Please upload your video first', interactive=False), gr.update(value="Upload & Start Chat", interactive=True), chat_state


def upload_img(gr_video, chat_state):
    print('gr_video:', gr_video)
    num_segments=4
    chat_state = {
        "questions": [],
        "answers": [],
    }
    # img_list = []
    if gr_video is None:
        return None, gr.update(interactive=True),gr.update(interactive=True, placeholder='Please upload video/image first!'), chat_state, 0.0
    else: 
        llm_message = chat.upload_video(gr_video)
        return gr.update(interactive=True), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, 0.0


def gradio_ask(up_video, gr_video_time, user_message, chatbot, chat_state):
    print('gr_video_time:', gr_video_time)
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    # time_prompt = 'Now the video is at %.1f second. ' % gr_video_time 
    time_prompt = '现在视频到了%.1f秒处. ' % gr_video_time 
    chat_state =  chat.ask(time_prompt+user_message, chat_state)
    chatbot = chatbot + [[f'User@{gr_video_time}s: '+user_message, None]]
    return '', chatbot, chat_state, gr_video_time


def gradio_answer(chatbot, chat_state, gr_video_time):
    llm_message, chat_state, last_img_list = chat.answer(chat_state, timestamp=gr_video_time, add_to_history=False)
    # llm_message = llm_message.replace("<s>", "") # handle <s>
    chatbot[-1][1] = llm_message
    # print(chat_state)
    print(f"Answer: {llm_message}")
    return chatbot, chat_state, last_img_list

def silent_ask(user_message, chat_state, gr_video_time, memory_size):
    chat.max_history = memory_size
    # user_message = 'Now the video is at %.1f second. What am I doing?' % gr_video_time
    user_message = '现在视频到了%.1f秒处. 描述当前视频中我的动作.' % gr_video_time
    chat_state =  chat.ask(user_message, chat_state)
    # chatbot = chatbot + [[f'User@{gr_video_time}s: '+user_message, None]]
    return '', chat_state

def silent_answer(chat_state, gr_video_time):
    llm_message, chat_state, last_img_list = chat.answer(chat_state, timestamp=gr_video_time, add_to_history=True)
    llm_message = llm_message.replace("<s>", "") # handle <s>
    return chat_state


class OpenGVLab(gr.themes.base.Base):
    def __init__(
        self,
        *,
        primary_hue=colors.blue,
        secondary_hue=colors.sky,
        neutral_hue=colors.gray,
        spacing_size=sizes.spacing_md,
        radius_size=sizes.radius_sm,
        text_size=sizes.text_md,
        font=(
            fonts.GoogleFont("Noto Sans"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono=(
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            body_background_fill="*neutral_50",
        )


gvlabtheme = OpenGVLab(primary_hue=colors.blue,
        secondary_hue=colors.sky,
        neutral_hue=colors.gray,
        spacing_size=sizes.spacing_md,
        radius_size=sizes.radius_sm,
        text_size=sizes.text_md,
        )

title = """<h1 align="center">Vinci</h1>"""
description ="""
        An Egocentric Video Foundation Model based Online Intelligent Assistant
        """

with gr.Blocks(title="EgoCentric Skill Assistant Demo",theme=gvlabtheme,css="#chatbot {overflow:auto; height:500px;} #InputVideo {overflow:visible; height:320px;} footer {visibility: none}") as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    gr_timer = gr.Timer(10, active=False)
    silent_time = gr.Number(0.0, visible=False)
    with gr.Row():
        with gr.Column(scale=0.5, visible=True) as video_upload:
            with gr.Column(elem_id="image", scale=0.5) as img_part:
                up_video = gr.Video(interactive=True, elem_id="up_video", height=360,)
            upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
            clear = gr.Button("Restart")
            
            memory_size = gr.Slider(
                minimum=5,
                maximum=25,
                value=10,
                step=1,
                interactive=True,
                label="size of memory)",
            )
            
            memory_stride = gr.Slider(
                minimum=5,
                maximum=100,
                value=10,
                step=1,
                interactive=True,
                label="stride of memory",
            )

        
        with gr.Column(visible=True)  as input_raws:
            chat_state = gr.State()
            img_list = gr.State()
            last_img_list = gr.State()
            chatbot = gr.Chatbot(elem_id="chatbot",label='ChatBot')
            with gr.Row():
                with gr.Column(scale=0.7):
                    text_input = gr.Textbox(show_label=False, placeholder='Please upload your video first', interactive=False, container=False)
                with gr.Column(scale=0.15, min_width=0):
                    run = gr.Button("💭Send")
                with gr.Column(scale=0.15, min_width=0):
                    clear = gr.Button("🔄Clear️")     
            with gr.Row():
                with gr.Column(scale=0.3):
                    inimage_interface = gr.Image(label="input image", elem_id="gr_inimage", visible=True, height=360) 
                with gr.Column(scale=0.7):
                    outvideo_interface = gr.Video(label="output video", elem_id="gr_outvideo", visible=True, height=360) 
            with gr.Row():
                with gr.Column(scale=0.5):
                    generate_button = gr.Button(value="Video how-to demo", interactive=True, variant="primary")
                with gr.Column(scale=0.5):
                    generate_clear_button = gr.Button(value="Clear", interactive=True, variant="primary")
    gr_video_time = gr.Number(value=-1, visible=False)
    def gr_video_time_change(_, video_time):
        return video_time
    def video_change_init_time():
        return 0, gr.update(active=True) 
    
    def timertick(up_video, gr_video_time, silent_time, text_input, chat_state, memory_stride, memory_size):
        if gr_video_time - silent_time < memory_stride:
            return silent_time, chat_state, gr_video_time
        silent_time = gr_video_time
        _,  chat_state = silent_ask(text_input, chat_state, gr_video_time, memory_size)
        chat_state = silent_answer(chat_state, gr_video_time)
        return silent_time, chat_state, gr_video_time

    gr_timer.tick(timertick, [up_video, gr_video_time, silent_time, text_input, chat_state, memory_stride, memory_size], [silent_time, chat_state, gr_video_time], js=get_time)
    up_video.play(video_change_init_time, [], [gr_video_time, gr_timer])

    def generate_video(img, conv, gr_video_time):
        text = conv["answers"][-1]
        omega_conf.input_path = './lastim.jpg'
        omega_conf.text_prompt = [text]
        gen(omega_conf, model_seine)
        return img, './result.mp4'
    generate_button.click(generate_video, [last_img_list, chat_state], [inimage_interface, outvideo_interface])
    chat = init_model()

    def generate_clear():
        return gr.update(value=None), gr.update(value=None)
    generate_clear_button.click(generate_clear, [], [inimage_interface, outvideo_interface])

    upload_button.click(upload_img, [up_video, chat_state], [up_video, text_input, upload_button, chat_state, gr_video_time])
    
    text_input.submit(gradio_ask, [up_video, gr_video_time, text_input, chatbot, chat_state], [text_input, chatbot, chat_state, gr_video_time], js=get_gr_video_current_time).then(
        gradio_answer, [chatbot, chat_state, gr_video_time], [chatbot, chat_state, last_img_list]
    )
    run.click(gradio_ask, [up_video, gr_video_time, text_input, chatbot, chat_state], [text_input, chatbot, chat_state, gr_video_time], js=get_gr_video_current_time).then(
        gradio_answer, [chatbot, chat_state, gr_video_time], [chatbot, chat_state, last_img_list]
    )
    run.click(lambda: "", None, text_input)  
    clear.click(gradio_reset, [chat_state], [chatbot, up_video, text_input, upload_button, chat_state], queue=False)

# demo.launch(share=True, enable_queue=True)
demo.queue(default_concurrency_limit=10)
demo.launch(server_name="0.0.0.0", server_port=10049, debug=True)
