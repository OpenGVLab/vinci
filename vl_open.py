import numpy as np
import torch
import torchvision.transforms as T
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


def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class Chat():
    def __init__(self, path='Vinci-8B-base', path2='Vinci-8B-ckpt', sep_chat=False, stream=True, device='cuda:0', use_chat_history=False, language='chn'):
        super().__init__()
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

        from safetensors.torch import load_file
        
        def merge_dicts(dict1, dict2, dict3, dict4):
            result = {**dict1, **dict2, **dict3, **dict4}
            return result

        model_weights1 = load_file(os.path.join(path2,"model-00001-of-00004.safetensors"))
        model_weights2 = load_file(os.path.join(path2,"model-00002-of-00004.safetensors"))
        model_weights3 = load_file(os.path.join(path2,"model-00003-of-00004.safetensors"))
        model_weights4 = load_file(os.path.join(path2,"model-00004-of-00004.safetensors"))
        merged_weight = merge_dicts(model_weights1,model_weights2,model_weights3,model_weights4)
        self.model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True)
        self.model.wrap_llm_lora(r=16, lora_alpha=2 * 16)
        msg = self.model.load_state_dict(merged_weight,strict=False)

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


    def answer(self, conv, timestamp=0, add_to_history=False):
        pixel_values, num_patches_list = self.load_video_timestamp(timestamp)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        if add_to_history: # silent ask
            if self.language == 'chn':
                question = video_prefix + '现在视频到了 %.1f 秒处. 简单的描述视频中我的动作.' % timestamp
            else:
                question = video_prefix + 'Now the video is at %.1f second. Briefly describe my actions in the video.' % timestamp #conv['questions'][-1]
            
            response, history = self.model.chat(self.tokenizer, pixel_values, question, self.generation_config,
                               num_patches_list=num_patches_list,
                               history=None, return_history=True)
            self.history.append((timestamp, response))
            print('VL_HISTORY:', self.history)
        else:
            if True:
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

        return response, conv, './lastim.jpg'

    def add_history(self, question):
        if not self.history:
            print('history not added because self.history is empty')
            return question
        if len(self.history) > 0:
            if self.language == 'chn':
                system = "你是一个在增强现实(AR)眼镜上的智能助手。看到的图像是来自我第一人称视角的视频帧。仔细观察视频并重点关注物体的运动和我的动作。由于你看不到发生在当前帧之前的部分，现在以文字形式提供给你这个视频的之前的历史供参考。视频历史是："
            else:
                system = 'You are an intelligent assistant on AR glasses. The AR glasses receive video frames from my egocentric viewpoint. Carefully watch the video and pay attention to the movement of objects, and the action of human. Since you cannot see the previous part of the video, I provide you the history of this video for reference. The history is: '
            res = system
            if self.sep_chat:
                for hist in self.history[:-1]:
                    ts = hist[0]
                    a = hist[1]
                    if self.language == 'chn':
                        res += '当视频在%.1f秒时, 视频的内容是 "%s"' % (ts, a)
                    else:
                        res += 'When the video is at %.1f seconds, the video content is "%s". ' % (ts, a)
                ts = self.history[-1][0]
                a = self.history[-1][1]
                if self.language == 'chn':
                    res += '以上是所有的视频历史, 表明了之前发生了什么.\n现在视频到了 %.1f秒, 视频的内容是 "%s". ' % (ts, a)
                else:
                    res += 'This is the end of the history which indicate what have previously happened.\n Now the video is at %.1f seconds, the video content is: "%s". ' % (ts, a)
            else:
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

            if self.language == 'chn':
                res += '请根据当前视频, 同时参照视频历史, 用中文回答我的问题: "%s". 注意如果问题与之前发生的事情有关, 请参考视频历史, 否则请只关注图像信息.' % question
            else:
                res += 'Given the current video and using the previous video as reference, answer my question in English: "%s". Note that if the question is about what has been previously done, please only focus on the history. Otherwise, please only focus on the question and the current video input. If the question is about future planning, provide at most 3 steps.' % question

            question = res
        return question
