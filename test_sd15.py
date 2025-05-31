from diffusers import StableDiffusionPipeline

# %%
import torch
from diffusers import DiffusionPipeline
import os

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

from trainer.accelerators.base_accelerator import BaseAccelerator
from trainer.configs.configs import TrainerConfig, instantiate_with_cfg


# %%
cfg = {}
cfg['manual_steps'] = True
cfg['gpu_number'] = '0'
cfg['device'] = f'cuda:{cfg["gpu_number"]}'
cfg['save_dir'] = './quick_inference'




from diffusers import DiffusionPipeline


from huggingface_hub import login
# login()
from trainer.configs.configs import TrainerConfig, instantiate_with_cfg
from trainer.models.base_model import BaseModelConfig
# model = instantiate_with_cfg(BaseModelConfig)
def check_contain_chinese(check_str):
    
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

from diffusers import DiffusionPipeline
from diffusers import StableDiffusion3Pipeline
from peft import PeftModel, PeftConfig



pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5",torch_dtype=torch.float16)
pipe = pipe.to("cuda")
lora_model_path = "autodl-tmp/recipe1-5.safetensors"
pipe.load_lora_weights(lora_model_path)

pipe.to("cuda")  # Use 'cpu' if you don't have a GPU

tokenizer = pipe.tokenizer
generator = torch.Generator(device=cfg['device']).manual_seed(0)
path = "autodl-tmp/视频/"
for test_dir in os.listdir(path):
    if test_dir[0]=='.':
        continue
    
    for dish_dir in os.listdir(os.path.join(path,test_dir)):
        if dish_dir[0]=='.':
            continue
        dish_file = os.listdir(os.path.join(path,test_dir,dish_dir))
       
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

        # input_texts = [
           
        # ]

        # input_texts.extend(step_texts)

      
        



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
                  
                    num_inference_steps=50,
           
                ).images
                

            
                # curr_goal_method_id = input_texts[0].replace(' ', '_').replace('/', '_')
                # pattern is samplenum-originalfname.png
                output_image_fname = f"{sample_num}-sd1.5.png"
                output_image_fname = output_image_fname.replace('/', '_')
                save_dir = os.path.join(path,test_dir,dish_dir)
                # make the save_dir
                os.makedirs(save_dir, exist_ok=True)
                output_image_fname = os.path.join(save_dir, output_image_fname)
                # print(output_image_fname)
                output_images[0].save(output_image_fname)