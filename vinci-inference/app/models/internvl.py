import torch
import sys
import os
import time

from PIL import Image
from threading import Thread

# 设置模型加载相对路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', "..")))
from vl_open import Chat, dynamic_preprocess

from global_var import FIFOSafeCache

history_cache = FIFOSafeCache(capacity=1000)
first_chat_timestamp_cache = FIFOSafeCache(capacity=1000)

def init_model(sep_chat: False, stream: bool=False, device: str='cuda:0', language: str='chn'):
    print(f'Initializing VLChat, sep_chat: {sep_chat}, stream: {stream}, device: {device}')
    with torch.device(device):
        chat = Chat(sep_chat=sep_chat, stream=stream, language=language)
        print('Initialization Finished')
        return chat


def add_history(question, history, sep_chat: bool=False, language: str='chn'):
    if not history:
        print('history not added because self.history is empty')
        return question
    if len(history) > 0:
        if language == 'chn':
            system = "你是一个在增强现实(AR)眼镜上的智能助手。看到的图像是来自我第一人称视角的视频帧。仔细观察视频并重点关注物体的运动和我的动作。由于你看不到发生在当前帧之前的部分，现在以文字形式提供给你这个视频的之前的历史供参考。视频历史是："
        else:
            system = 'You are an intelligent assistant on AR glasses. The AR glasses receive video frames from my egocentric viewpoint. Carefully watch the video and pay attention to the movement of objects, and the action of human. Since you cannot see the previous part of the video, I provide you the history of this video for reference. The history is: '
        res = system
        if sep_chat:
            for hist in history[:-1]:
                ts = hist[0]
                a = hist[1]
                if language == 'chn':
                    res += '当视频在%.1f秒时, 视频的内容是 "%s"' % (ts, a)
                else:
                    res += 'When the video is at %.1f seconds, the video content is "%s". ' % (ts, a)
            ts = history[-1][0]
            a = history[-1][1]
            if language == 'chn':
                res += '以上是所有的视频历史, 表明了之前发生了什么.\n现在视频到了 %.1f秒, 视频的内容是 "%s". ' % (ts, a)
            else:
                res += 'This is the end of the history which indicate what have previously happened.\n Now the video is at %.1f seconds, the video content is: "%s". ' % (ts, a)
        else:
            for hist in history:
                ts = hist[0]
                a = hist[1]
                if language == 'chn':
                    res += '当视频在%.1f秒时, 视频的内容是 "%s". ' % (ts, a.strip())
                else:
                    res += 'When the video is at %.1f seconds, the video contect is "%s". ' % (ts, a.strip())
            if language == 'chn':
                res += '以上是所有的视频历史, 表明了之前发生了什么.\n'
            else:
                res += 'This is the end of the video history that indicates what happened before.\n'

        if language == 'chn':
            res += '请根据当前视频, 同时参照视频历史, 用中文回答我的问题: "%s". 注意如果问题与之前发生的事情有关, 请参考视频历史, 否则请只关注图像信息.' % question
        else:
            res += 'Given the current video and using the previous video as reference, answer my question in English: "%s". Note that if the question is about what has been previously done, please only focus on the history. Otherwise, please only focus on the question and the current video input. If the question is about future planning, provide at most 3 steps.' % question

        question = res
    return question


class IntervlChat():

    def __init__(self, sep_chat: bool=False, stream: bool=False, 
                 device: str='cuda:0', language: str='chn'):
        self.sep_chat = sep_chat
        self.language = language
        self.device = device
        self.stream = stream
        self.intervl_chat = init_model(sep_chat, stream=self.stream, device=device, language=language)
        self.origin_intervl_model = self.intervl_chat.model

    # frame是否可以resize
    def load_video_frames(self, frames: list):
        pixel_values_list, num_patches_list = [], []
        for i, frame in enumerate(frames):
            # frame = np.array(frame, dtype=np.uint8)
            img = Image.fromarray(frame).convert('RGB')
            if i == len(frames) - 1:
                img.save('./lastim.jpg')

            img = dynamic_preprocess(img, image_size=448, use_thumbnail=True, max_num=1)
            pixel_values = [self.intervl_chat.transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list)

        return pixel_values, num_patches_list
    
    def _chat(self, model, tokenizer, pixel_values, question: str, num_patches_list: list):
        return model.chat(tokenizer, pixel_values, question, self.intervl_chat.generation_config, 
                          num_patches_list=num_patches_list, history=None, return_history=True)
    
    def _chat_stream(self, model, tokenizer, pixel_values, question: str, generation_config, num_patches_list: list):
        thread = Thread(target=model.chat, kwargs=dict(tokenizer=tokenizer, pixel_values=pixel_values, question=question, generation_config=generation_config, 
                                                       num_patches_list=num_patches_list, history=None, return_history=False))
        thread.start()

        return self.intervl_chat.streamer

    def chat_unsafe_silent(self, question: str, frames: list, history: list=[], timestamp: int=0,):
        """
        在静默模式下聊天交互。
        

        参数:
        - question: str
          输入的问题字符串，目前此参数在函数内部被重新构造而不是使用。
        - frames: list
          视频帧的列表，每个帧代表视频的一个时间点。
        - history: list
          之前的聊天历史记录，默认为空列表。
        - session_id: str
          会话的唯一标识符，默认为"default"。
        - timestamp: int
          当前视频的时间戳，用于构造问题和记录历史。

        返回:
        - response: str
          生成的回答。
        - history: list
          更新后的聊天历史记录，包括新增的回答。
        """
        print(f"silent chat history: {history}")
        print(f"silent chat timestamp: {timestamp}")
        
        with torch.device(self.device):
            pixel_values, num_patches_list = self.load_video_frames(frames)
            pixel_values = pixel_values.to(torch.bfloat16).cuda()

            video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
            question = video_prefix + '现在视频到了 %.1f 秒处. 简单的描述视频中我的动作.' % timestamp
            response, _history = self._chat(self.origin_intervl_model, self.intervl_chat.tokenizer, pixel_values, question, num_patches_list)
            history.append((timestamp, response))
            print('VL_HISTORY:', _history)

            print('Real question at %2.1f is |||' % (timestamp), question)
            print('Answer at %2.1f is ||| '% (timestamp), response)
            
            return response, history

    def chat_unsafe(self, question: str, frames: list, history: list=[], timestamp: int=0, silent: bool=False):
        """
        实现与用户视频内容相关的聊天功能，支持实时流式输出。

        参数:
        - question: str, 用户提出的问题。
        - frames: list, 视频的帧数据列表。
        - history: list, 聊天历史记录，默认为空列表。
        - session_id: str, 会话ID，默认为"default"。
        - timestamp: int, 视频的当前时间戳，默认为0。
        - silent: bool, 是否静默模式，默认为False。

        该函数首先从缓存中获取特定会话的历史记录（如果存在），然后将视频帧数据加载到设备上进行处理。
        根据配置决定是使用流式输出还是非流式输出来生成回答。聊天历史记录会被更新并缓存。
        """
        print(f"chat history: {history}")
        print(f"chat timestamp: {timestamp}")
        
        with torch.device(self.device):
            pixel_values, num_patches_list = self.load_video_frames(frames)
            pixel_values = pixel_values.to(torch.bfloat16).cuda()

            video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])

            if self.sep_chat:
                quest = video_prefix + '现在视频到了 %.1f 秒处. 描述视频中我的动作.' % timestamp
                response, _ = self._chat(self.origin_intervl_model, self.intervl_chat.tokenizer, pixel_values, quest, num_patches_list)
                
                history.append((timestamp, response))
                question = add_history(question, history, self.sep_chat, self.language)
#                 question = add_history(question, history)
                
                print(f"chat question: {question}")
                if self.stream:
                    streamer = self._chat_stream(self.intervl_chat.lmmodel, self.intervl_chat.tokenizer, None, question, 
                                                    self.intervl_chat.lmgeneration_config, num_patches_list)
                    
                    for new_text in streamer:
                        if new_text == self.intervl_chat.lmmodel.conv_template.sep:
                            return new_text, history
                        
                        yield new_text, history
                else:
                    response, _ = self._chat(self.intervl_chat.model, self.intervl_chat.tokenizer, None, question, 
                                                self.intervl_chat.generation_config, num_patches_list)
                                    
                    print('Real question at %2.1f is |||' % timestamp, question)
                    print('Answer at %2.1f is ||| '%timestamp, response)

                    yield response, history
            else:
                question = '现在视频到了%.1f秒处. ' % timestamp + question 
                question = add_history(question, history, self.sep_chat)
                question = video_prefix + question

                print(f"chat question: {question}")

                if self.stream:
                    streamer = self._chat_stream(self.intervl_chat.model, self.intervl_chat.tokenizer, pixel_values, question, 
                                                 self.intervl_chat.generation_config, num_patches_list)

                    for new_text in self.intervl_chat.streamer:
                        if new_text == self.intervl_chat.model.conv_template.sep:
                            print(f"set history: {history}")
                            return new_text, history
                        
                        yield new_text, history
                    # response, _ = self._chat(self.intervl_chat.model, self.intervl_chat.tokenizer, pixel_values, question, num_patches_list)
            
                    # print('Real question at %2.1f is |||' % timestamp, question)
                    # print('Answer at %2.1f is ||| '%timestamp, response)

                    # yield response, history
                    
                else:
                    response, _ = self._chat(self.intervl_chat.model, self.intervl_chat.tokenizer, pixel_values, question, num_patches_list)

                    print('Real question at %2.1f is |||' % timestamp, question)
                    print('Answer at %2.1f is ||| '%timestamp, response)

                    yield response, history

# 获取所有可用的CUDA设备
available_devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
print(f"availabel devices: {available_devices}")

sep_chat = False
stream = True
running_language = os.environ.get('RUNNING_LANGUAGE')
if running_language is None:
    running_language = 'chn'
chats = [IntervlChat(sep_chat, stream, device, running_language) for device in available_devices]

def get_timestamp(session_id: str):
    current_timestamp = time.time()

    first_timestamp = first_chat_timestamp_cache.get(session_id)
    if first_timestamp:
        return int(current_timestamp) - int(first_timestamp)

    first_chat_timestamp_cache.put(session_id, current_timestamp)

    return 0

def get_history(session_id: str):
    history = history_cache.get(session_id)
    if not history:
        return []
    
    return history[-10:]

def chat(question: str, frames: list, history: list=[], session_id: str="default", 
         timestamp: int=0, silent: bool=False, model_index: int=0):
    print(f"session id: {session_id}")

    chat = chats[model_index]
    timestamp = get_timestamp(session_id)
    history = get_history(session_id)

    if silent:
        response, history = chat.chat_unsafe_silent(question, frames, history, timestamp=timestamp)
        
        # 记录历史
        print(f"set silent history: {history}")
        history_cache.put(session_id, history)

        return response, history
                    
    answers = chat.chat_unsafe(question, frames, history, timestamp=timestamp, silent=silent)
    
    generate_txt = ''
    history = []
    for answer, history in answers:
        generate_txt += answer
        history = history
    
    # 记录历史
    print(f"set history: {history}")
    history_cache.put(session_id, history)

    return generate_txt, history

def stream_chat(question: str, frames: list, history: list, session_id: str="default", 
                timestamp: int=0, silent: bool=False, model_index: int=0):
    print(f"session id: {session_id}")

    chat = chats[model_index]
    timestamp = get_timestamp(session_id)
    history = get_history(session_id)

    if silent:
        response, history = chat.chat_unsafe_silent(question, frames, history, timestamp=timestamp)

        # 记录历史
        print(f"set silent history: {history}")
        history_cache.put(session_id, history)

        yield response, history
        return
    
    answers = chat.chat_unsafe(question, frames, history, timestamp=timestamp, silent=silent)
    
    generate_txt = ''
    for answer, history in answers:
        generate_txt += answer
        yield generate_txt, history

    # 记录历史
    print(f"set history: {history}")
    history_cache.put(session_id, history)

    print(f"chat answer: {generate_txt}")