import torch, torchvision
torch.backends.cudnn.enabled = False
import gradio as gr
from gradio.themes.utils import colors, fonts, sizes
import os, subprocess
from vl_open import Chat

#generation module
import sys
sys.path.append('generation/seine-v2/')
# torchvision.set_video_backend('video_reader')
from generation.seine import gen, model_seine
from omegaconf import OmegaConf
omega_conf = OmegaConf.load('seine-v2/configs/demo.yaml')
omega_conf.run_time = 13
omega_conf.input_path = ''
omega_conf.text_prompt = []
omega_conf.save_img_path = '.'


get_gr_video_current_time = """async (video, grtime, one, two, three) => {
  const videoEl = document.querySelector("#up_video video");
  return [video, videoEl.currentTime, one, two, three];
}"""

get_time = """async (video, grtime, one, two, three) => {
  const videoEl = document.querySelector("#up_video video");
  return [video, videoEl.currentTime, one, two, three];
}"""
# ========================================
#             Model Initialization
# ========================================
def init_model():
    print('Initializing VLChat')
    chat = Chat(stream=False)
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
    return None, gr.update(value=None, interactive=True), gr.update(value=None, interactive=True), gr.update(placeholder='Please upload your video first', interactive=False),gr.update(value="Upload & Start Chat", interactive=True), chat_state


def upload_img(gr_img, gr_video, chat_state, num_segments):
    print('gr_img:', gr_img, '. gr_video:', gr_video)
    chat_state = {
        "questions": [],
        "answers": [],
    }
    # img_list = []
    if gr_img is None and gr_video is None:
        return None, None, gr.update(interactive=True),gr.update(interactive=True, placeholder='Please upload video/image first!'), chat_state, 0.0
    if gr_video: 
        llm_message = chat.upload_video(gr_video)
        return gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, 0.0
    if gr_img:
        llm_message = chat.upload_img(gr_img)
        return gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, 0.0


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

def silent_ask(user_message, chat_state, gr_video_time):
    # user_message = 'Now the video is at %.1f second. What am I doing?' % gr_video_time
    user_message = '现在视频到了%.1f秒处. 描述当前视频中我的动作.' % gr_video_time
    print('silent gr_video_time:', gr_video_time)
    chat_state =  chat.ask(user_message, chat_state)
    # chatbot = chatbot + [[f'User@{gr_video_time}s: '+user_message, None]]
    return '', chat_state

def silent_answer(chat_state, gr_video_time):
    llm_message, chat_state, last_img_list = chat.answer(chat_state, timestamp=gr_video_time, add_to_history=True)
    llm_message = llm_message.replace("<s>", "") # handle <s>
    # print(chat_state)
    print(f"Silent Answer: {llm_message}")
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

title = """<h1 align="center">Demo </h1>"""
description ="""
        Work?
        """


with gr.Blocks(title="EgoCentric Skill Assistant Demo",theme=gvlabtheme,css="#chatbot {overflow:auto; height:500px;} #InputVideo {overflow:visible; height:320px;} footer {visibility: none}") as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    gr_timer = gr.Timer(10, active=False)
    silent_time = gr.Number(0.0, visible=False)
    with gr.Row():
        with gr.Column(scale=0.5, visible=True) as video_upload:
            with gr.Column(elem_id="image", scale=0.5) as img_part:
                with gr.Tab("Video", elem_id='video_tab'):
                    up_video = gr.Video(interactive=True, elem_id="up_video", height=360)
                with gr.Tab("Image", elem_id='image_tab'):
                    up_image = gr.Image(type="pil", interactive=True, elem_id="image_upload", height=360)
            upload_button = gr.Button(value="Upload & Start Chat", interactive=True, variant="primary")
            clear = gr.Button("Restart")
            
            num_beams = gr.Slider(
                minimum=1,
                maximum=10,
                value=1,
                step=1,
                interactive=True,
                label="beam search numbers)",
            )
            
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="Temperature",
            )
            
            num_segments = gr.Slider(
                minimum=8,
                maximum=64,
                value=8,
                step=1,
                interactive=True,
                label="Video Segments",
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
    
    def timertick(up_video, gr_video_time, silent_time, text_input, chat_state):
        print('timer tick', gr_video_time, 'silent time', silent_time)
        if gr_video_time - silent_time < 10:
            return silent_time, chat_state, gr_video_time
        silent_time = gr_video_time
        _,  chat_state = silent_ask(text_input, chat_state, gr_video_time)
        chat_state = silent_answer(chat_state, gr_video_time)
        return silent_time, chat_state, gr_video_time
    gr_timer.tick(timertick, [up_video, gr_video_time, silent_time, text_input, chat_state], [silent_time, chat_state, gr_video_time], js=get_time)
    up_video.play(video_change_init_time, [], [gr_video_time, gr_timer])

    def generate_video(img, conv, gr_video_time):
        print('current time:', gr_video_time)
        print(conv)
        text = conv["answers"][-1]
        print('generate using', text)
        omega_conf.input_path = './lastim.jpg'
        omega_conf.text_prompt = [text]
        gen(omega_conf, model_seine)
        return img, './result.mp4'
    generate_button.click(generate_video, [last_img_list, chat_state], [inimage_interface, outvideo_interface])
    chat = init_model()

    def generate_clear():
        return gr.update(value=None), gr.update(value=None)
    generate_clear_button.click(generate_clear, [], [inimage_interface, outvideo_interface])

    upload_button.click(upload_img, [up_image, up_video, chat_state, num_segments], [up_image, up_video, text_input, upload_button, chat_state, gr_video_time])
    
    text_input.submit(gradio_ask, [up_video, gr_video_time, text_input, chatbot, chat_state], [text_input, chatbot, chat_state, gr_video_time], js=get_gr_video_current_time).then(
        gradio_answer, [chatbot, chat_state, gr_video_time], [chatbot, chat_state, last_img_list]
    )
    run.click(gradio_ask, [up_video, gr_video_time, text_input, chatbot, chat_state], [text_input, chatbot, chat_state, gr_video_time], js=get_gr_video_current_time).then(
        gradio_answer, [chatbot, chat_state, gr_video_time], [chatbot, chat_state, last_img_list]
    )
    run.click(lambda: "", None, text_input)  
    clear.click(gradio_reset, [chat_state], [chatbot, up_image, up_video, text_input, upload_button, chat_state], queue=False)

# demo.launch(share=True, enable_queue=True)
demo.queue(default_concurrency_limit=10)
demo.launch(server_name="0.0.0.0", server_port=10049, debug=True)
