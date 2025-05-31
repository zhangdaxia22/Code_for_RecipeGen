from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusion3Pipeline
# %%
import torch
from diffusers import DiffusionPipeline
import os
from diffusers import FluxPipeline
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
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

# from trainer.accelerators.base_accelerator import BaseAccelerator
# from trainer.configs.configs import TrainerConfig, instantiate_with_cfg


# %%
cfg = {}
cfg['manual_steps'] = True
cfg['gpu_number'] = '0'
cfg['device'] = f'cuda:{cfg["gpu_number"]}'
cfg['save_dir'] = './quick_inference'


# create_pipeline_cfg = dict(
#             # pretrained_model_name_or_path=self.cfg.pretrained_model_name_or_path,
#             # cfg=model.cfg,
#             model=model
#         )
# pipeline = ShowNotTellPipeline.from_pretrained(**create_pipeline_cfg)
           
# pipeline = DiffusionPipeline.from_pretrained("sachit-menon/illustrated_instructions", custom_pipeline="snt_pipeline", trust_remote_code=True)
# pipeline.save_pretrained("autodl-tmp")

from diffusers import DiffusionPipeline


from huggingface_hub import login

def check_contain_chinese(check_str):
    
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False
# 加载模型

from diffusers import DiffusionPipeline
from diffusers import StableDiffusion3Pipeline
from peft import PeftModel, PeftConfig




pipe = StableDiffusion3Pipeline.from_pretrained("/root/autodl-tmp/sd3.5",torch_dtype=torch.float16)
pipe = pipe.to("cuda")

pipe.to("cuda")  # Use 'cpu' if you don't have a GPU

# pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
# pipe = pipe.to("cuda")  # 如果您有 GPU
tokenizer = pipe.tokenizer
generator = torch.Generator(device=cfg['device']).manual_seed(0)
path = "/root/autodl-tmp/recipe/"

    
for dish_dir in os.listdir(path):
    if dish_dir[0]=='.':
        continue
    dish_file = os.listdir(os.path.join(path,dish_dir))


    j = "gpt_steps.txt"


    with open(os.path.join(path,dish_dir,j),"r",encoding="utf-8") as f:
        data= f.read()
        print(dish_dir)
        print(data)
        data = data.split("\n")
        data = [line for line in data if line]  # 过滤空字符串
        data = data[1:]
        
        for sample_num in range(len(data)):
            with torch.autocast(device_type='cuda', 
                                dtype=torch.bfloat16, 
                                enabled= True):
                prompts = tokenizer(
                        data[sample_num],
                        max_length=77,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    ).input_ids
                


                step_texts_tokenized = tokenizer.batch_decode(prompts, skip_special_tokens=True)
                output_images = pipe(
                    step_texts_tokenized,
                    height=512,
                    width=512,
                    num_inference_steps=50,
                ).images
                
                output_image_fname = f"{sample_num}-sd3_5.png"
                output_image_fname = output_image_fname.replace('/', '_')
                save_dir = os.path.join(path,dish_dir)
                # make the save_dir
                os.makedirs(save_dir, exist_ok=True)
                output_image_fname = os.path.join(save_dir, output_image_fname)
                print(output_image_fname)
                output_images[0].save(output_image_fname)