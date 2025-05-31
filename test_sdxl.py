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



from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")





pipe = pipe.to("cuda")  # 如果您有 GPU
tokenizer = pipe.tokenizer
generator = torch.Generator(device=cfg['device']).manual_seed(0)# DiffusionPipeline.from_pretrained("stabilityai/stable-
path = ""
for test_dir in os.listdir(path):
    if test_dir[0]=='.':
        continue
    
    for dish_dir in os.listdir(os.path.join(path,test_dir)):
        if dish_dir[0]=='.':
            continue
        dish_file = os.listdir(os.path.join(path,test_dir,dish_dir))
    
        step_texts=[]
       
        with open(os.path.join(path,test_dir,dish_dir,"steps.txt"),"r",encoding="utf-8") as f:
            data= f.read()
            print(data)
            data = data.split("\n")
            for step in data:
               
            
                if len(step)<5:
                    continue
                
                step_texts.append(step)
            
        for i, step in enumerate(step_texts):
            step_texts[i] = f'{i}. {step}'


      
        



        for sample_num in range(len(step_texts)):
            with torch.autocast(device_type='cuda', 
                                dtype=torch.bfloat16, 
                                enabled= True):
              
              
                output_images = pipe(
                    step_texts[sample_num],
                    num_inference_steps=40,
    denoising_end=0.8,
                    output_type="latent",
                    
                ).images
                output_images = refiner(
    prompt=step_texts[sample_num],
                    num_inference_steps=40,
    denoising_end=0.8,
    
    image=output_images,
).images[0]
                

            
                # curr_goal_method_id = input_texts[0].replace(' ', '_').replace('/', '_')
                # pattern is samplenum-originalfname.png
                output_image_fname = f"{sample_num}-sdxl.png"
                output_image_fname = output_image_fname.replace('/', '_')
                save_dir = os.path.join(path,test_dir,dish_dir)
                # make the save_dir
                os.makedirs(save_dir, exist_ok=True)
                output_image_fname = os.path.join(save_dir, output_image_fname)
                print(output_image_fname)
                output_images.save(output_image_fname)