# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import io
import os
import sys
# sys.path.append('seine-v2/')
import math
# import seine_utils as utils
from diffusion import create_diffusion

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import argparse
import torchvision

from einops import rearrange
# from models import get_models
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from models_new.clip import TextEmbedder
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
from torchvision import transforms
from functions import video_transforms
from decord import VideoReader
# from seine_utils import mask_generation_before
from diffusers.utils.import_utils import is_xformers_available

from models_new.unet import UNet3DConditionModel
current_directory = os.getcwd()
print("Current Working Directory:", current_directory)
# try:
#     model = UNet3DConditionModel.from_pretrained_2d("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="unet", use_concat=True, finetuned_image_sd_path=None)
# except:
    ### modify the absolute path if sd1.5 is available locally
model = UNet3DConditionModel.from_pretrained_2d('seine_weights', subfolder="unet", use_concat=True, finetuned_image_sd_path=None)
vae = AutoencoderKL.from_pretrained("seine_weights", subfolder="vae")
text_encoder = TextEmbedder("seine_weights")
# Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
# https://huggingface.co/hyf015/model_weights/blob/main/finetune_seine_256p_15ep_60K.pt
ckpt_path = 'seine_weights/finetune_seine_256p_15ep_60K.pt'

state_dict = torch.load(ckpt_path, map_location='cpu')['ema']
res = model.load_state_dict(state_dict)
print('loading succeed')
print(res)
model_seine = model

def mask_generation_before(mask_type, shape, dtype, device):
    b, f, c, h, w = shape
    if mask_type.startswith('first'):
        num = int(mask_type.split('first')[-1])
        mask_f = torch.cat([torch.zeros(1, num, 1, 1, 1, dtype=dtype, device=device),
                           torch.ones(1, f-num, 1, 1, 1, dtype=dtype, device=device)], dim=1)
        mask = mask_f.expand(b, -1, c, h, w)
    else:
        raise ValueError(f"Invalid mask type: {mask_type}")
    return mask
    
def get_input(args):
    input_path = args.input_path
        
    transform_video = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.ResizeVideo((args.image_h, args.image_w)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    temporal_sample_func = video_transforms.TemporalRandomCrop(args.num_frames * args.frame_interval)
    
    assert os.path.isfile(input_path), 'Please make sure the given input_path is a JPG/PNG file'

    print(f'loading video from {input_path}')
    _, full_file_name = os.path.split(input_path)
    file_name, extention = os.path.splitext(full_file_name)
    assert extention == '.jpg' or extention == '.png', "Not a PNG or JPG file"
        
    video_frames = []
    num = int(args.mask_type.split('first')[-1])
    first_frame = torch.as_tensor(np.array(Image.open(input_path), dtype=np.uint8, copy=True)).unsqueeze(0)
    for i in range(num):
        video_frames.append(first_frame)
    num_zeros = args.num_frames-num
    for i in range(num_zeros):
        zeros = torch.zeros_like(first_frame)
        video_frames.append(zeros)
    n = 0
    video_frames = torch.cat(video_frames, dim=0).permute(0, 3, 1, 2) # f,c,h,w
    video_frames = transform_video(video_frames)
    return video_frames, n


def auto_inpainting(args, video_input, masked_video, mask, prompt, vae, text_encoder, diffusion, model, device,):
    b, f, c, h, w = video_input.shape
    latent_h = args.image_size[0] // 8
    latent_w = args.image_size[1] // 8

    if args.use_fp16:
        z = torch.randn(1, 4, args.num_frames, args.latent_h, args.latent_w, dtype=torch.float16, device=device) # b,c,f,h,w
        masked_video = masked_video.to(dtype=torch.float16)
        mask = mask.to(dtype=torch.float16)
    else:
        z = torch.randn(1, 4, args.num_frames, args.latent_h, args.latent_w, device=device) # b,c,f,h,w

    masked_video = rearrange(masked_video, 'b f c h w -> (b f) c h w').contiguous()
    masked_video = vae.encode(masked_video).latent_dist.sample().mul_(0.18215)
    masked_video = rearrange(masked_video, '(b f) c h w -> b c f h w', b=b).contiguous()
    mask = torch.nn.functional.interpolate(mask[:,:,0,:], size=(latent_h, latent_w)).unsqueeze(1)

    # classifier_free_guidance
    if args.do_classifier_free_guidance:
        masked_video = torch.cat([masked_video] * 2)
        mask = torch.cat([mask] * 2)
        z = torch.cat([z] * 2)
        prompt_all = [prompt] + [args.negative_prompt]
    else:
        masked_video = masked_video
        mask = mask
        z = z
        prompt_all = [prompt]

    text_prompt = text_encoder(text_prompts=prompt_all, train=False)
    model_kwargs = dict(encoder_hidden_states=text_prompt, 
                        class_labels=None, 
                        cfg_scale=args.cfg_scale,
                        use_fp16=args.use_fp16,
    ) # tav unet
    
    # Sample images:
    if args.sample_method == 'ddim':
        samples = diffusion.ddim_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device, \
            mask=mask, x_start=masked_video, use_concat=args.use_mask
        )
    elif args.sample_method == 'ddpm':
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device, \
            mask=mask, x_start=masked_video, use_concat=args.use_mask
        )
    samples, _ = samples.chunk(2, dim=0) # [1, 4, 16, 32, 32]
    if args.use_fp16:
        samples = samples.to(dtype=torch.float16)

    video_clip = samples[0].permute(1, 0, 2, 3).contiguous() # [16, 4, 32, 32]
    video_clip = vae.decode(video_clip / 0.18215).sample # [16, 3, 256, 256]
    return video_clip


def gen(args, model, save_path='result.mp4'):
    model.cuda()
    print('entered gen')
    # Setup PyTorch:
    if args.seed:
        torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args.latent_h = latent_h = args.image_size[0] // 8
    args.latent_w = latent_w = args.image_size[1] // 8
    args.image_h = args.image_size[0]
    args.image_w = args.image_size[1]

    print('loading model')
    if args.use_compile:
        model = torch.compile(model)

    if args.enable_xformers_memory_efficient_attention:
        print('Using xformers memory efficient attention')
        if is_xformers_available():
            model.enable_xformers_memory_efficient_attention()
        else:
            print('Xformers not available')
            pass
    else:
        print('Not using xformers memory efficient attention')

    model.eval()  # important!
    pretrained_model_path = "seine_weights"
    
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae.to(device)
    text_encoder.to(device)
    
    if args.use_fp16:
        print('Warnning: using half percision for inferencing!')
        vae.to(device, dtype=torch.float16)
        model.to(device, dtype=torch.float16)
        text_encoder.to(device, dtype=torch.float16)

    assert args.use_autoregressive is True
    
    video_input, researve_frames = get_input(args) # f,c,h,w
    video_input = video_input.to(device).unsqueeze(0) # b,f,c,h,w
    mask = mask_generation_before(args.mask_type, video_input.shape, video_input.dtype, device) # b,f,c,h,w
    # TODO: change the first3 to last3
    if args.mask_type.startswith('first') and researve_frames != 0:
        masked_video = torch.cat([video_input[:,-researve_frames:], video_input[:,:-researve_frames]], dim=1) * (mask == 0)
    else:
        masked_video = video_input * (mask == 0)
  
    print(args.text_prompt, 'current text prompt')
    video_clip = auto_inpainting(args, video_input, masked_video, mask, args.text_prompt[0], vae, text_encoder, diffusion, model, device,)
    video_clip_ = video_clip.unsqueeze(0)
    video_ = ((video_clip * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1)
    print(f'Saving result video to {save_path}')
    torchvision.io.write_video(save_path, video_, fps=8)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/sample_mask.yaml")
    parser.add_argument("--run-time", type=int, default=0)
    parser.add_argument("--demoimage", type=str, default=None, required=True)
    parser.add_argument('--demotext', type=str, default=None, required=True)
    parser.add_argument('--demosavepath', type=str, default=None, required=True)
    parser.add_argument('--checkpoint', type=str, default=None, required=True)
    args = parser.parse_args()
    omega_conf = OmegaConf.load(args.config)
    omega_conf.run_time = args.run_time
    
    omega_conf.input_path = args.demoimage
    omega_conf.text_prompt = args.demotext
    omega_conf.save_img_path = args.demosavepath
    omega_conf.ckpt = args.checkpoint
    gen(omega_conf, model, args.demosavepath)
