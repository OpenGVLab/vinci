# Vinci - An Online Egocentric Video-Language Assistant
<a src="https://img.shields.io/badge/cs.CV-2412.21080-b31b1b?logo=arxiv&logoColor=red" href="https://arxiv.org/abs/2412.21080"> <img src="https://img.shields.io/badge/cs.CV-2412.21080-b31b1b?logo=arxiv&logoColor=red">
</a>  <a href="https://huggingface.co/hyf015/Vinci-8B-ckpt"><img src="https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm-dark.svg" alt="Open in Spaces"> </a> 

> **Vinci: A Real-time Embodied Smart Assistant based on Egocentric Vision-Language Model**<br>
Arxiv, 2024<br>

## 💬 TL,DR

- **Overview**: A real-time, embodied smart assistant based on an egocentric vision-language model.
-  **Portable Device Compatibility**: Designed for smartphones and wearable cameras, operating in an "always on" mode.
-  **Hands-Free Interaction**: Users engage in natural conversations to ask questions and get responses delivered via audio.
-  **Real-Time Video Processing**: Processes long video streams to answer queries about current and historical observations.
-  **Task Planning and Guidance**: Provides task planning based on past interactions and generates visual task demonstrations.

## 📣 Demo video

[![Watch the video](https://img.youtube.com/vi/R0cz616OiPs/0.jpg)](https://www.youtube.com/watch?v=R0cz616OiPs)
[![Watch the video](https://img.youtube.com/vi/TOGwhn-vp1s/0.jpg)](https://www.youtube.com/watch?v=TOGwhn-vp1s)


## 🔨 Installation
```
git clone https://github.com/OpenGVLab/vinci.git
conda env create -f environment.yml
```
### Downloading Checkpoints
```
bash download.sh
```
Running download.sh will take up 10GB disk space.

## 🎓 Getting Started
We offer two ways to run our Vinci model

### 🎬  Online Streaming Demo
1. start the frontend, backend and model services: 
```bash
sudo ./boot.sh {start|stop|restart} [--cuda <CUDA_VISIBLE_DEVICES>] [--language chn/eng] [--version v0/v1]
```

- --cuda <CUDA_VISIBLE_DEVICES>: Specify the GPU devices to run the model
- --language <chn|eng>: Choose the language for the demo (default: chn).
  - chn: Chinese 
  - eng: English

- --version <v0|v1>: Select the model version (default: v1).
  - v0: Optimized for first-person perspective videos.
  - v1: Generalized model for both first-person and third-person perspective videos.

Then use the browser to access the frontend page：http://YOUR_IP_ADDRESS:19333 （E.g., http://102.2.52.16:19333）

2. Push live stream
With an smartphone app or GoPro/DJI cameras, push the stream to: `rtmp://YOUR_IP_ADDRESS/vinci/livestream`

With a webcam, use the following command: `ffmpeg -f video4linux2 -framerate 30 -video_size 1280x720 -i /dev/video1 -f alsa -i default  -vcodec libx264 -preset ultrafast -pix_fmt yuv420p -video_size 1280x720   -c:a aac -threads 0 -f flv rtmp://YOUR_IP_ADDRESS:1935/vinci/livestream`

#### Interact with Online Video Streaming Demo
1. Activate Model Service: To wake up the model and begin using it, simply say the wake-up phrase: "你好望舒 (Ni hao wang shu)" (Currently, only Chinese wakeup command is supported)
2. Chat with Vinci: Once activated, you can start chatting with Vinci with speech. The model will respond in text and speech. 
Tip: For the best experience, speak clearly and at a moderate pace. 
3. Generate Predictive Visualizations: If you want to generate a predictive visualization of actions, include the keyword "可视化" in your command. 

### 🎬 Gradio Demo for uploaded videos
```bash
python demovl.py  [--cuda <CUDA_VISIBLE_DEVICES>] [--language chn/eng] [--version v0/v1]
```
- --cuda <CUDA_VISIBLE_DEVICES>: Specify the GPU devices to run the model
- --language <chn|eng>: Choose the language for the demo (default: chn).
  - Options: 
    - chn: Chinese 
    - eng: English
  
- --version <v0|v1>: Select the model version (default: v1).
  - Options: 
    - v0: Optimized for first-person perspective videos.
    - v1: Generalized model for both first-person and third-person perspective videos.

#### Interact with Gradio Demo
1. Upload local video file
<div align="center">
<img src="assets/1-dl.PNG" width="35%">
</div>

2. Click Upload & Start Chat button to initiate the chat session
<div align="center">
<img src="assets/2-start.PNG" width="35%">
</div>

3. Click the play button to start playing the video
<div align="center">
<img src="assets/3-play.PNG" width="55%">
</div>

4. Adjusting the Stride of Memory. This allows you to control the granularity of the model's memory. 
<div align="center">
<img src="assets/4-adjust.PNG" width="55%">
</div>

5. Real-Time Interaction：Type your questions in the chat box. The model will respond based on the current frame and historical context.


<div align="center">
<img src="assets/5-chat.PNG" width="75%">
<figcaption>Describe current action</figcaption>
</div>
<br>
<div align="center">
<img src="assets/6-chat.PNG" width="75%">
<figcaption>Retreive object from the history</figcaption>
</div>
<br>

<div align="center">
<img src="assets/7-chat.PNG" width="75%">
<figcaption>Summarize previous actions</figcaption>
</div>
<br>

<div align="center">
<img src="assets/8-chat.PNG" width="75%">
<figcaption>Scene understanding</figcaption>
</div>
<br>

<div align="center">
<img src="assets/9-chat.PNG" width="75%">
<figcaption>Temporal grounding</figcaption>
</div>
<br>

<div align="center">
<img src="assets/10-chat.PNG" width="75%">
<figcaption>Predict future actions</figcaption>
</div>
<br>


6. Generate future videos: based on the current frame and the historical context, the model can generate a short future video.
<div align="center">
<img src="assets/11-gen.png" width="75%">
<figcaption>Generate future actions</figcaption>
</div>

## ✒️ Citation
If this work is helpful for your research, please consider citing us.
```
@article{pei2024egovideo,
  title={Vinci: A Real-time Embodied Smart Assistant based on Egocentric Vision-Language Model},
  author={Huang, Yifei and Xu, Jilan and Pei, Baoqi and He, Yuping and Chen, Guo and Yang, Lijin and Chen, Xinyuan and Wang, Yaohui and Nie, Zheng and Liu, Jinyao and Fan, Guoshun and Lin, Dechen and Fang, Fang and Li, Kunpeng and Yuan, Chang and Wang, Yali and Qiao, Yu and Wang, Limin},
  journal={arXiv preprint arXiv:2412.21080},
  year={2024}
}
```

```
@article{pei2024egovideo,
  title={EgoVideo: Exploring Egocentric Foundation Model and Downstream Adaptation},
  author={Pei, Baoqi and Chen, Guo and Xu, Jilan and He, Yuping and Liu, Yicheng and Pan, Kanghua and Huang, Yifei and Wang, Yali and Lu, Tong and Wang, Limin and Qiao, Yu},
  journal={arXiv preprint arXiv:2406.18070 },
  year={2024}
}
```
```
@inproceedings{xu2024retrieval,
  title={Retrieval-augmented egocentric video captioning},
  author={Xu, Jilan and Huang, Yifei and Hou, Junlin and Chen, Guo and Zhang, Yuejie and Feng, Rui and Xie, Weidi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13525--13536},
  year={2024}
}
```
```
 @InProceedings{huang2024egoexolearn,
     title={EgoExoLearn: A Dataset for Bridging Asynchronous Ego- and Exo-centric View of Procedural Activities in Real World},
     author={Huang, Yifei and Chen, Guo and Xu, Jilan and Zhang, Mingfang and Yang, Lijin and Pei, Baoqi and Zhang, Hongjie and Lu, Dong and Wang, Yali and Wang, Limin and Qiao, Yu},
     booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
     year={2024}
 }
```
```
@inproceedings{chen2023seine,
  title={Seine: Short-to-long video diffusion model for generative transition and prediction},
  author={Chen, Xinyuan and Wang, Yaohui and Zhang, Lingjun and Zhuang, Shaobin and Ma, Xin and Yu, Jiashuo and Wang, Yali and Lin, Dahua and Qiao, Yu and Liu, Ziwei},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
```
```
@article{wang2024internvideo2,
  title={Internvideo2: Scaling video foundation models for multimodal video understanding},
  author={Wang, Yi and Li, Kunchang and Li, Xinhao and Yu, Jiashuo and He, Yinan and Wang, Chenting and Chen, Guo and Pei, Baoqi and Zheng, Rongkun and Xu, Jilan and Wang, Zun and others},
  journal={arXiv preprint arXiv:2403.15377},
  year={2024}
}
```