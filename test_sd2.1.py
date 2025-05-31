from diffusers import StableDiffusionPipeline

# %%
import torch
from diffusers import DiffusionPipeline
import os

from diffusers import AutoPipelineForText2Image
import torch
import openai 
import re

import json
import os
from typing import Any

import hydra
import torch
from hydra.utils import instantiate
from accelerate.logging import get_logger
from omegaconf import DictConfig, OmegaConf
from torch import nn

from trainer.accelerators.base_accelerator import BaseAccelerator
from trainer.configs.configs import TrainerConfig, instantiate_with_cfg

from diffusers import StableDiffusionXLPipeline
# %%
cfg = {}
cfg['manual_steps'] = True
cfg['gpu_number'] = '0'
cfg['device'] = f'cuda:{cfg["gpu_number"]}'
cfg['save_dir'] = './quick_inference'



from diffusers import DiffusionPipeline


from trainer.configs.configs import TrainerConfig, instantiate_with_cfg
from trainer.models.base_model import BaseModelConfig
# model = instantiate_with_cfg(BaseModelConfig)
def check_contain_chinese(check_str):
    
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)

pipe = pipe.to("cuda")

pipe.to("cuda")
tokenizer = pipe.tokenizer
generator = torch.Generator(device=cfg['device']).manual_seed(0)
path = ""
for test_dir in os.listdir(path):
    if test_dir[0]=='.':
        continue
    
    for dish_dir in os.listdir(os.path.join(path,test_dir)):
        if dish_dir[0]=='.':
            continue
        dish_file = os.listdir(os.path.join(path,test_dir,dish_dir))
        
        step_texts=[]
      
        with open(os.path.join(path,test_dir,dish_dir,j),"r",encoding="utf-8") as f:
            data= f.read()
            data = data.split("\n")
            for step in data:
                if len(step)<4:
                    continue
                step_texts.append(step)
            
       

      
        



        for sample_num in range(len(step_texts)):
            with torch.autocast(device_type='cuda', 
                                dtype=torch.bfloat16, 
                                enabled= True):
                prompts = tokenizer(
                        step_texts[sample_num],
                        max_length=77,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    ).input_ids
              
                step_texts_tokenized = tokenizer.batch_decode(prompts, skip_special_tokens=True)
                output_images = pipe(
                    step_texts_tokenized,
                    width=512, height=512
                    
                ).images
                

            
                output_image_fname = f"{sample_num}-sd21.png"
                output_image_fname = output_image_fname.replace('/', '_')
                save_dir = os.path.join(path,test_dir,dish_dir)
               
                os.makedirs(save_dir, exist_ok=True)
                output_image_fname = os.path.join(save_dir, output_image_fname)
              
                output_images[0].save(output_image_fname)