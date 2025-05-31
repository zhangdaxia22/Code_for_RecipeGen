import os
import shutil
import subprocess
import glob
import time
import logging
from pathlib import Path
from tqdm import tqdm
try:
    import pynvml
    pynvml_initialized = False
except ImportError:
    pynvml = None
    pynvml_initialized = True
try:
    import torch
except ImportError:
    torch = None

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('i2v_generation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

def initialize_pynvml():
    global pynvml_initialized
    if pynvml is None:
        logger.warning("pynvml 未安装，无法监控显存。请安装：pip install pynvml")
        pynvml_initialized = True
        return
    try:
        pynvml.nvmlInit()
        pynvml_initialized = True
        logger.info("pynvml 初始化成功")
    except pynvml.NVMLError as e:
        logger.warning(f"pynvml 初始化失败: {e}. 无法监控显存")
        pynvml_initialized = True

def get_gpu_memory():
    if pynvml is None or not pynvml_initialized:
        return "显存监控不可用"
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        memory_info = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used = mem_info.used / 1024**2
            total = mem_info.total / 1024**2
            memory_info.append(f"GPU {i}: {used:.2f}/{total:.2f} MB ({used/total*100:.1f}%)")
        return "; ".join(memory_info)
    except pynvml.NVMLError as e:
        return f"获取显存失败: {e}"

def clear_temp_dir(temp_dir):
    for file in glob.glob(os.path.join(temp_dir, '*.mp4')) + glob.glob(os.path.join(temp_dir, '*.txt')):
        try:
            os.remove(file)
            logger.info(f"已删除临时文件: {file}")
        except Exception as e:
            logger.error(f"删除临时文件 {file} 失败: {e}")

def clear_cuda_cache():
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            logger.info("已清理 CUDA 缓存")
        except Exception as e:
            logger.warning(f"清理 CUDA 缓存失败: {e}")
    else:
        logger.warning("无法清理 CUDA 缓存：torch 未安装或无 CUDA 设备")

def make_safe_prompt(prompt):
    return prompt.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')

def run_with_retry(command, prompt, dish_id, retries=3, delay=10):
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"尝试 {attempt}/{retries} 次执行命令")
            clear_cuda_cache()
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            while process.poll() is None:
                stdout_line = process.stdout.readline().strip()
                if stdout_line:
                    logger.info(stdout_line)
                stderr_line = process.stderr.readline().strip()
                if stderr_line:
                    logger.error(stderr_line)
            
            stdout, stderr = process.communicate()
            if stdout:
                logger.info(stdout.strip())
            if stderr:
                logger.error(stderr.strip())
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, command, stdout, stderr)
            
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"尝试 {attempt} 失败 {prompt} (菜品: {dish_id}): {e.stderr}")
            if attempt < retries:
                logger.info(f"等待 {delay} 秒后重试...")
                clear_cuda_cache()
                time.sleep(delay)
        except Exception as e:
            logger.error(f"尝试 {attempt} 失败 {prompt} (菜品: {dish_id}): {e}")
            if attempt < retries:
                logger.info(f"等待 {delay} 秒后重试...")
                clear_cuda_cache()
                time.sleep(delay)
    return False

def generate_i2v_videos_for_dishes(dish_base_dir, output_base_dir, temp_dir='./results'):
    os.makedirs(temp_dir, exist_ok=True)
    dish_ids = [d for d in os.listdir(dish_base_dir) if os.path.isdir(os.path.join(dish_base_dir, d)) and not d.startswith('.')]
    logger.info(f"找到 {len(dish_ids)} 个菜品文件夹")

    for dish_idx, dish_id in enumerate(tqdm(dish_ids, desc="处理菜品文件夹", position=0), 1):
        dish_dir = os.path.join(dish_base_dir, dish_id)
        logger.info(f"当前处理第 {dish_idx}/{len(dish_ids)} 个菜品: {dish_id}")
        
        steps_file = os.path.join(dish_dir, 'steps.txt')
        if not os.path.exists(steps_file):
            logger.warning(f"未找到 steps.txt，跳过菜品: {dish_id}")
            continue
        
        videos_dir = os.path.join(dish_dir, 'videos')
        os.makedirs(videos_dir, exist_ok=True)
        
        try:
            with open(steps_file, 'r', encoding='utf-8') as f:
                steps = [line.strip() for line in f if line.strip()]
            logger.info(f"菜品 {dish_id} 有 {len(steps)} 个步骤")
        except Exception as e:
            logger.error(f"读取 steps.txt 失败 ({dish_id}): {e}")
            continue
        
        for step in tqdm(steps, desc=f"处理 {dish_id} 的步骤", leave=False, position=1):
            try:
                step_num, prompt = step.split('. ', 1)
                step_num = step_num.strip()
                prompt = prompt.strip()
            except ValueError:
                logger.warning(f"步骤格式错误，跳过: {step} (菜品: {dish_id})")
                continue
                
            image_path = os.path.join(dish_dir, f"{step_num}.jpg")
            if not os.path.exists(image_path):
                logger.warning(f"未找到图片 {image_path}，跳过步骤: {step_num} (菜品: {dish_id})")
                continue
                
            safe_prompt = make_safe_prompt(prompt)
            target_name = f"{step_num}-{safe_prompt}.mp4"
            target_path = os.path.join(videos_dir, target_name)
            
            if os.path.exists(target_path):
                logger.info(f"目标视频已存在，跳过: {target_path}")
                continue
                
            logger.info(f"生成视频前显存使用情况: {get_gpu_memory()}")
            
            command = [
                'python3', 'sample_image2video.py',
                '--model', 'HYVideo-T/2',
                '--prompt', prompt,
                '--i2v-mode',
                '--i2v-image-path', image_path,
                '--i2v-resolution', '360p',
                '--infer-steps', '30',
                '--video-length', '129',
                '--flow-reverse',
                '--flow-shift', '17.0',
                '--embedded-cfg-scale', '6.0',
                '--seed', '0',
                '--use-cpu-offload',
                '--gradient-checkpoint',  # 启用梯度检查点
                '--save-path', temp_dir
            ]
            
            try:
                if not run_with_retry(command, prompt, dish_id, retries=3, delay=10):
                    logger.error(f"生成视频失败 {prompt} (菜品: {dish_id}): 所有重试尝试均失败")
                    continue
                
                generated_files = glob.glob(os.path.join(temp_dir, '*seed0*.mp4'))
                if not generated_files:
                    logger.warning(f"未找到生成的视频文件: {prompt} (菜品: {dish_id})")
                    all_mp4_files = glob.glob(os.path.join(temp_dir, '*.mp4'))
                    if all_mp4_files:
                        generated_files = [max(all_mp4_files, key=os.path.getmtime)]
                        logger.info(f"找到备用视频文件: {generated_files[0]}")
                    else:
                        continue
                
                generated_file = max(generated_files, key=os.path.getmtime)
                shutil.move(generated_file, target_path)
                logger.info(f"成功生成并重命名视频: {target_path}")
                
                logger.info(f"生成视频后显存使用情况: {get_gpu_memory()}")
                clear_temp_dir(temp_dir)
                
            except MemoryError:
                logger.error(f"生成视频失败 {prompt} (菜品: {dish_id}): 显存不足")
            except Exception as e:
                logger.error(f"处理视频文件失败 {prompt} (菜品: {dish_id}): {e}")
        
        try:
            shutil.copy2(steps_file, os.path.join(videos_dir, 'steps.txt'))
            logger.info(f"已复制 steps.txt 到 {videos_dir}/steps.txt")
        except Exception as e:
            logger.error(f"复制 steps.txt 失败 ({dish_id}): {e}")

def main():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    if not pynvml_initialized:
        initialize_pynvml()
    
    dish_base_dir = '/root/autodl-tmp/HunyuanVideo-I2V/sampled-yooucook2-cleaned'
    output_base_dir = dish_base_dir
    temp_dir = './results'
    """
    # 输入和输出路径
    sample_file = '/root/autodl-tmp/douguo/sampled_dishes.txt'
    output_base_dir = '/root/autodl-tmp/douguo/i2v-hunyuan-douguo'
    temp_dir = '/root/autodl-tmp/douguo/results_tmp'
    """
    
    generate_i2v_videos_for_dishes(dish_base_dir, output_base_dir, temp_dir)
    logger.info("所有视频生成和文件复制完成！")

if __name__ == "__main__":
    main()



