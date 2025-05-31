from tqdm import tqdm
from PIL import Image
import torch
import os
import numpy as np
from transformers import CLIPProcessor, CLIPModel
# 设置 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 加载模型和处理器
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")


def get_all_images(path):
    images=[]
    
            
    return  sorted([os.path.join(path, file) 
                  for file in os.listdir(path) 
                  if file.endswith(".png")])
    


def get_clip_score(images_path, text_prompts):
    """计算所有图像的 CLIP Score"""
    if not images_path:
        return 0  # 若没有图片，返回 0 避免异常

    scores = []
    for image_path, text in zip(images_path, text_prompts):
        try:
            image = Image.open(image_path).convert("RGB").resize((256,256))  # 确保图像是 RGB 格式
            # inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
            inputs = processor(
                text=[text], 
                images=image, 
                return_tensors="pt",
                padding="max_length",      
                truncation=True           
                )
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
            outputs = model(**inputs)
            scores.append(outputs.logits_per_image.item())  # 提取数值
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    if scores:
        avg_score = sum(scores) / len(scores)
        print(f"Processed {len(images_path)} images, avg score: {avg_score:.4f}")
        return avg_score
    return 0


def read_prompts_from_file(path):
    """读取文本提示，并返回后半部分的内容"""
    with open(path,"r") as f:
        data= f.read()
        data = data.split("\n")
        # data = data[1:]
        step_cap=[]
        for step in data:
            if len(step)<3:
                continue
        
            step_cap.append(step)
    # print(step_cap)
    return step_cap
                
        
                    


test_dir = ""
recipe_cnt = 0
recipe_sum = 0
for test in os.listdir(test_dir):
    for recipe in os.listdir(os.path.join(test_dir,test)):
        recipe_dir = os.path.join(test_dir,test, recipe)
        if recipe.startswith(".") or not os.path.isdir(recipe_dir):  # 跳过隐藏文件和非文件夹
            continue
       
   
    
        text_prompts = read_prompts_from_file(os.path.join(test_dir,recipe, "steps.txt"))
       
        image_paths = get_all_images(recipe_dir)

    
        if text_prompts and image_paths:
            score = get_clip_score(image_paths, text_prompts)
            recipe_sum += score
            recipe_cnt += 1

if recipe_cnt > 0:
    print( recipe_cnt)
    print(f"Finish! Avg Score: {recipe_sum / recipe_cnt:.4f}")
else:
    print("No valid recipes processed.")
