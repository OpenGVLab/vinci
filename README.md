# Vinci - An Online Egocentric Video-Language Assistant

### Installation
```
git clone https://github.com/OpenGVLab/vinci.git
conda env create -f environment.yml
bash download.sh
```
Running download.sh will take up 10GB disk space.

### Downloading Checkpoints

### Getting Started
We offer two ways to run our Vinci model
#### Online Streaming Demo
1. start the frontend, backend and model services:
`sudo ./boot.sh {start|stop|restart} [--cuda <CUDA_VISIBLE_DEVICES>] [--language chn/eng] [--version v0/v1]`

- --cuda <CUDA_VISIBLE_DEVICES>: Specify the GPU devices to run the model
- --language <chn|eng>: Choose the language for the demo. 
  - Options: 
    - chn: Chinese 
    - eng: English
  - Default: chn.
- --version <v0|v1>: Select the model version
  - Options: 
    - v0: Optimized for first-person perspective videos.
    - v1: Generalized model for both first-person and third-person perspective videos.
  - Default: v1

Then use the browser to access the frontend page：http://YOUR_IP_ADDRESS:19333 （E.g., http://102.2.52.16:19333）

2. Push live stream
With an smartphone app or GoPro/DJI cameras, push the stream to: `rtmp://YOUR_IP_ADDRESS/vinci/livestream`

With a webcam, use the following command: `ffmpeg -f video4linux2 -framerate 30 -video_size 1280x720 -i /dev/video1 -f alsa -i default  -vcodec libx264 -preset ultrafast -pix_fmt yuv420p -video_size 1280x720   -c:a aac -threads 0 -f flv rtmp://YOUR_IP_ADDRESS:1935/vinci/livestream`

#### Interact with Online Video Streaming Demo
1. Activate Model Service: To wake up the model and begin using it, simply say the wake-up phrase: "你好望舒" (Currently, only Chinese wakeup command is supported)
2. Chat with Vinci: Once activated, you can start chatting with Vinci with speech. The model will respond in text and speech. 
Tip: For the best experience, speak clearly and at a moderate pace. 
3. Generate Predictive Visualizations: If you want to generate a predictive visualization of actions, include the keyword "可视化" in your command. 

#### Gradio Demo for uploaded videos
`python demovl.py  [--cuda <CUDA_VISIBLE_DEVICES>] [--language chn/eng] [--version v0/v1]`
- --cuda <CUDA_VISIBLE_DEVICES>: Specify the GPU devices to run the model
- --language <chn|eng>: Choose the language for the demo. 
  - Options: 
    - chn: Chinese 
    - eng: English
  - Default: chn.
- --version <v0|v1>: Select the model version
  - Options: 
    - v0: Optimized for first-person perspective videos.
    - v1: Generalized model for both first-person and third-person perspective videos.
  - Default: v1
