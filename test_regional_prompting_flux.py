import torch,os
from pipeline_flux_regional import RegionalFluxPipeline, RegionalFluxAttnProcessor2_0
from pipeline_flux_controlnet_regional import RegionalFluxControlNetPipeline
from diffusers import FluxControlNetModel, FluxMultiControlNetModel
model_path = "black-forest-labs/FLUX.1-dev"
torch.cuda.empty_cache()
pipeline = RegionalFluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")
    
attn_procs = {}
for name in pipeline.transformer.attn_processors.keys():
    if 'transformer_blocks' in name and name.endswith("attn.processor"):
        attn_procs[name] = RegionalFluxAttnProcessor2_0()
    else:
        attn_procs[name] = pipeline.transformer.attn_processors[name]
pipeline.transformer.set_attn_processor(attn_procs)

for num in range(1,15):
    
    
    
    use_lora = False
    use_controlnet = False
   
    num_samples = 1
    num_inference_steps = 24
    guidance_scale = 3.5
    seed = 124
    base_prompt = ""
    background_prompt = "a photo"
    regional_prompt_mask_pairs ={}

    mask_inject_steps = 10
    double_inject_blocks_interval = 1
    single_inject_blocks_interval = 1
    base_ratio = 0.3
    path = ""
    for test_dir in os.listdir(path):
        tdir(os.path.join(path,test_dir)):
            if dish_dir[0]=='.':
                continue
            dish_file = os.listdir(os.path.join(path,test_dir,dish_dir))
    
         
            
            step_texts=[]
        
            with open(os.path.join(path,test_dir,dish_dir,"steps.txt"),"r",encoding="utf-8") as f:
                data= f.read()
                
                data = data.split("\n")
                k=0
                for step in data:
                    
                   
                    if len(step)<1:
                        continue
                    print(step)
                    # if key not in regional_prompt_mask_pairs:
                    regional_prompt_mask_pairs[str(k)] = {}
                    regional_prompt_mask_pairs[str(k)]["description"]=step      
                    regional_prompt_mask_pairs[str(k)]["mask"]=[0,k*256,256,(k+1)*256]   
                    k+=1
                    # step_texts.append(step)
            if k != num:
                continue
            
            image_width = 256
            image_height = k*256

            image_width = (image_width // pipeline.vae_scale_factor) * pipeline.vae_scale_factor
            image_height = (image_height // pipeline.vae_scale_factor) * pipeline.vae_scale_factor
        
            regional_prompts = []
            regional_masks = []
            background_mask = torch.ones((image_height, image_width))
        
            for region_idx, region in regional_prompt_mask_pairs.items():
                description = region['description']
                mask = region['mask']
                x1, y1, x2, y2 = mask
        
                mask = torch.zeros((image_height, image_width))
                mask[y1:y2, x1:x2] = 1.0
        
                background_mask -= mask
        
                regional_prompts.append(description)
                regional_masks.append(mask)
                    
            # if regional masks don't cover the whole image, append background prompt and mask
            if background_mask.sum() > 0:
                regional_prompts.append(background_prompt)
                regional_masks.append(background_mask)
        
            # setup regional kwargs that pass to the pipeline
            joint_attention_kwargs = {
                'regional_prompts': regional_prompts,
                'regional_masks': regional_masks,
                'double_inject_blocks_interval': double_inject_blocks_interval,
                'single_inject_blocks_interval': single_inject_blocks_interval,
                'base_ratio': base_ratio,
            }
            # generate images
            if use_controlnet:
                images = pipeline(
                    prompt=base_prompt,
                    num_samples=num_samples,
                    width=image_width, height=image_height,
                    mask_inject_steps=mask_inject_steps,
                    control_image=control_image,
                    control_mode=control_mode,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=torch.Generator("cuda").manual_seed(seed),
                    joint_attention_kwargs=joint_attention_kwargs,
                ).images
            else:
                images = pipeline(
                    prompt=base_prompt,
                    num_samples=num_samples,
                    width=image_width, height=image_height,
                    mask_inject_steps=mask_inject_steps,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=torch.Generator("cuda").manual_seed(seed),
                    joint_attention_kwargs=joint_attention_kwargs,
                ).images
        
            for idx, image in enumerate(images):
                image.save(os.path.join(path,test_dir,dish_dir,"regional.png"))
            del pipeline
            torch.cuda.empty_cache()
            pipeline = RegionalFluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16,cache_dir="/root/autodl-tmp/flux").to("cuda")
    
            attn_procs = {}
            for name in pipeline.transformer.attn_processors.keys():
                if 'transformer_blocks' in name and name.endswith("attn.processor"):
                    attn_procs[name] = RegionalFluxAttnProcessor2_0()
                else:
                    attn_procs[name] = pipeline.transformer.attn_processors[name]
            pipeline.transformer.set_attn_processor(attn_procs)
