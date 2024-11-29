import os
import sys
from .unet import UNet3DConditionModel
    
def get_models(args):
    pretrained_model_path = args.pretrained_model_path
    return UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet", use_concat=args.use_mask, finetuned_image_sd_path=args.finetuned_image_sd_path)