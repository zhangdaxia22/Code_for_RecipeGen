from diffusers import StableDiffusionPipeline

# %%
import torch
from diffusers import DiffusionPipeline
import os
from diffusers import FluxPipeline
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import openai 
import re
from PIL import Image
import json
import os
from typing import Any

import hydra
import torch
from hydra.utils import instantiate
from accelerate.logging import get_logger
from omegaconf import DictConfig, OmegaConf
from torch import nn



# %%
cfg = {}
cfg['manual_steps'] = True
cfg['gpu_number'] = '0'
cfg['device'] = f'cuda:{cfg["gpu_number"]}'
cfg['save_dir'] = './quick_inference'



from diffusers import DiffusionPipeline

from huggingface_hub import login

def check_contain_chinese(check_str):
    
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

from diffusers import DiffusionPipeline
from diffusers import StableDiffusion3Pipeline
from peft import PeftModel, PeftConfig



pipe = FluxPipeline.from_pretrained("/root/autodl-tmp/FLUX",torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# load lora
lora_model_path = "/root/autodl-tmp/recipe_all_shu.safetensors"
pipe.load_lora_weights(lora_model_path)

pipe.to("cuda")  # Use 'cpu' if you don't have a GPU

tokenizer = pipe.tokenizer
generator = torch.Generator(device=cfg['device']).manual_seed(0)
path = "/root/autodl-tmp/test/"
for test_dir in os.listdir(path):
    if test_dir[0]=='.':
        continue
    
    for dish_dir in os.listdir(os.path.join(path,test_dir)):
        if dish_dir[0]=='.':
            continue
        
        if not os.path.exists(os.path.join(path,test_dir,dish_dir,"gpt_steps.txt")):
            continue

        # 如果存在"flux_plus_lora.png"，则跳过
        if os.path.exists(os.path.join(path,test_dir,dish_dir,"flux_plus_lora.png")):
            continue
        with open(os.path.join(path,test_dir,dish_dir,"gpt_steps.txt"),"r",encoding="utf-8") as f:
            data= f.read()
            #用t统计data的行数
        # 使用 splitlines() 将数据按行分割，然后过滤掉仅包含空白字符的行（空行）
        non_empty_lines = [line for line in data.splitlines() if line.strip()]

        # 用 t 统计去掉空行后的行数
        t = len(non_empty_lines)-1
        print(data)

        with torch.autocast(device_type='cuda', 
                            dtype=torch.bfloat16, 
                            enabled= True):
            prompts = tokenizer(
                    data,
                    max_length=77,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).input_ids
                

                
            step_texts_tokenized = tokenizer.batch_decode(prompts, skip_special_tokens=True)
            output_images = pipe(
                step_texts_tokenized,
                height=256*t,
                width=256,
                num_inference_steps=50,
            ).images

            output_image_fname = f"flux_plus_lora.png"
            output_image_fname = output_image_fname.replace('/', '_')
            save_dir = os.path.join(path,test_dir,dish_dir)
            # make the save_dir
            os.makedirs(save_dir, exist_ok=True)
            output_image_fname = os.path.join(save_dir, output_image_fname)
            print(output_image_fname)
            output_images[0].save(output_image_fname)

            image = Image.open(output_image_fname)
            for i in range(t):
                box = (0,i*256,256,(i+1)*256)
                cropped_image = image.crop(box)
                output_path = os.path.join(save_dir,f"cropped_{i}.png")
                cropped_image.save(output_path)
                print(f"已保存：{output_path}")
            print("图像切分完成！")