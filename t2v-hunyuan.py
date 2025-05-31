# import os
# import shutil
# import subprocess
# import glob
# import time
# try:
#     import pynvml
#     pynvml_initialized = False
# except ImportError:
#     pynvml = None
#     pynvml_initialized = True  # Skip initialization if pynvml is not installed

# def initialize_pynvml():
#     """初始化 pynvml，检测 GPU 可用性"""
#     global pynvml_initialized
#     if pynvml is None:
#         print("pynvml 未安装，无法监控显存。请安装：pip install pynvml")
#         pynvml_initialized = True
#         return
#     try:
#         pynvml.nvmlInit()
#         pynvml_initialized = True
#         print("pynvml 初始化成功")
#     except pynvml.NVMLError as e:
#         print(f"pynvml 初始化失败: {e}. 无法监控显存")
#         pynvml_initialized = True

# def get_gpu_memory():
#     """获取当前 GPU 显存使用情况"""
#     if pynvml is None or not pynvml_initialized:
#         return "显存监控不可用"
#     try:
#         device_count = pynvml.nvmlDeviceGetCount()
#         memory_info = []
#         for i in range(device_count):
#             handle = pynvml.nvmlDeviceGetHandleByIndex(i)
#             mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
#             used = mem_info.used / 1024**2  # 转换为 MB
#             total = mem_info.total / 1024**2  # 转换为 MB
#             memory_info.append(f"GPU {i}: {used:.2f}/{total:.2f} MB ({used/total*100:.1f}%)")
#         return "; ".join(memory_info)
#     except pynvml.NVMLError as e:
#         return f"获取显存失败: {e}"

# def clear_temp_dir(temp_dir):
#     """清空临时文件夹中的 .mp4 文件"""
#     for file in glob.glob(os.path.join(temp_dir, '*.mp4')):
#         try:
#             os.remove(file)
#             print(f"已删除临时文件: {file}")
#         except Exception as e:
#             print(f"删除临时文件 {file} 失败: {e}")

# def generate_videos_for_annotations(annotations_dir, output_base_dir, temp_dir='./results'):
#     """
#     为 annotations 文件夹中的每个菜品标注文件生成视频，重命名、复制标注文件并清空临时文件夹。
#     如果目标视频已存在，则跳过该步骤的生成。
#     :param annotations_dir: 标注文件所在的文件夹（如 'sampled_videos/annotations'）
#     :param output_base_dir: 生成视频的根文件夹（如 'generated_videos'）
#     :param temp_dir: sample_video.py 生成视频的临时文件夹（如 './results'）
#     """
#     # 初始化 pynvml
#     if not pynvml_initialized:
#         initialize_pynvml()
    
#     # 确保输出根文件夹和临时文件夹存在
#     os.makedirs(output_base_dir, exist_ok=True)
#     os.makedirs(temp_dir, exist_ok=True)
    
#     # 遍历 annotations 文件夹中的 .txt 文件
#     for txt_file in os.listdir(annotations_dir):
#         if not txt_file.endswith('.txt'):
#             continue
            
#         # 获取菜品 ID（去掉 .txt 扩展名）
#         dish_id = os.path.splitext(txt_file)[0]
#         txt_path = os.path.join(annotations_dir, txt_file)
        
#         # 创建菜品文件夹
#         dish_dir = os.path.join(output_base_dir, dish_id)
#         os.makedirs(dish_dir, exist_ok=True)
        
#         # 读取步骤
#         with open(txt_path, 'r', encoding='utf-8') as f:
#             steps = [line.strip() for line in f if line.strip()]
        
#         # 为每个步骤生成视频
#         for step in steps:
#             # 解析步骤编号和描述（格式：1. description）
#             try:
#                 step_num, prompt = step.split('. ', 1)
#                 step_num = step_num.strip()
#                 prompt = prompt.strip()
#             except ValueError:
#                 print(f"步骤格式错误，跳过: {step} (菜品: {dish_id})")
#                 continue
                
#             # 构造目标文件名
#             safe_prompt = prompt.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
#             target_name = f"{step_num}-{safe_prompt}.mp4"
#             target_path = os.path.join(dish_dir, target_name)
            
#             # 检查目标视频是否已存在
#             if os.path.exists(target_path):
#                 print(f"目标视频已存在，跳过: {target_path}")
#                 continue
                
#             # 构造调用指令
#             command = [
#                 'python3', 'sample_video.py',
#                 '--video-size', '300', '300',
#                 '--video-length', '129',
#                 '--infer-steps', '30',
#                 '--prompt', prompt,
#                 '--flow-reverse',
#                 '--use-cpu-offload',
#                 '--save-path', temp_dir
#             ]
            
#             # 执行指令
#             try:
#                 print(f"正在为 {dish_id} 的步骤 {step_num} 生成视频: {prompt}")
#                 result = subprocess.run(command, check=True, capture_output=True, text=True)
#                 if result.stdout:
#                     print(result.stdout.strip())
#                 if result.stderr:
#                     print(result.stderr.strip())
                
#                 # 查找生成的视频文件
#                 generated_files = glob.glob(os.path.join(temp_dir, f"*_{prompt}.mp4"))
#                 if not generated_files:
#                     print(f"未找到生成的视频文件: {prompt} (菜品: {dish_id})")
#                     continue
                
#                 # 取最新文件
#                 generated_file = max(generated_files, key=os.path.getmtime)
                
#                 # 移动并重命名视频
#                 shutil.move(generated_file, target_path)
#                 print(f"成功生成并重命名视频: {target_path}")
                
#                 # 输出当前显存
#                 print(f"当前显存使用情况: {get_gpu_memory()}")
                
#                 # 清空临时文件夹
#                 clear_temp_dir(temp_dir)
                
#             except subprocess.CalledProcessError as e:
#                 print(f"生成视频失败 {prompt} (菜品: {dish_id}): {e.stderr}")
#             except Exception as e:
#                 print(f"处理视频文件失败 {prompt} (菜品: {dish_id}): {e}")
        
#         # 复制原始 .txt 文件到菜品文件夹
#         shutil.copy2(txt_path, os.path.join(dish_dir, txt_file))
#         print(f"已复制标注文件到 {dish_dir}/{txt_file}")

# def main():
#     # 输入和输出路径
#     annotations_dir = 'sampled_videos/annotations'  # 标注文件夹
#     output_base_dir = 'generated_videos'  # 生成视频的根文件夹
#     temp_dir = './results_tmp'  # 临时文件夹
    
#     # 执行生成
#     generate_videos_for_annotations(annotations_dir, output_base_dir, temp_dir)
#     print("所有视频生成和文件复制完成！")

# if __name__ == "__main__":
#     main()








#处理prompt被截断的情况
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
        logging.FileHandler('video_generation.log', encoding='utf-8'),
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

def generate_videos_for_annotations(annotations_dir, output_base_dir, temp_dir='./results_tmp'):
    os.makedirs(output_base_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    txt_files = [f for f in os.listdir(annotations_dir) if f.endswith('.txt')]
    logger.info(f"找到 {len(txt_files)} 个标注文件")
    
    for txt_idx, txt_file in enumerate(tqdm(txt_files, desc="处理标注文件", position=0), 1):
        dish_id = os.path.splitext(txt_file)[0]
        txt_path = os.path.join(annotations_dir, txt_file)
        dish_dir = os.path.join(output_base_dir, dish_id)
        os.makedirs(dish_dir, exist_ok=True)
        
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                steps = [line.strip() for line in f if line.strip()]
            logger.info(f"菜品 {dish_id} 有 {len(steps)} 个步骤")
        except Exception as e:
            logger.error(f"读取 {txt_file} 失败: {e}")
            continue
        
        for step in tqdm(steps, desc=f"处理 {dish_id} 的步骤", leave=False, position=1):
            try:
                step_num, prompt = step.split('. ', 1)
                step_num = step_num.strip()
                prompt = prompt.strip()
            except ValueError:
                logger.warning(f"步骤格式错误，跳过: {step} (菜品: {dish_id})")
                continue
                
            safe_prompt = make_safe_prompt(prompt)
            target_name = f"{step_num}-{safe_prompt}.mp4"
            target_path = os.path.join(dish_dir, target_name)
            
            if os.path.exists(target_path):
                logger.info(f"目标视频已存在，跳过: {target_path}")
                continue
                
            logger.info(f"生成视频前显存使用情况: {get_gpu_memory()}")
            
            command = [
                'python3', 'sample_video.py',
                '--model', 'HYVideo-T/2-cfgdistill',
                '--video-size', '300', '300',
                '--video-length', '129',
                '--infer-steps', '30',
                '--prompt', prompt,
                '--flow-reverse',
                '--flow-shift', '7.0',
                '--embedded-cfg-scale', '6.0',
                '--use-cpu-offload',
                '--save-path', temp_dir
            ]
            
            try:
                if not run_with_retry(command, prompt, dish_id):
                    logger.error(f"生成视频失败 {prompt} (菜品: {dish_id}): 所有重试尝试均失败")
                    continue
                
                # 查找生成的视频文件（宽松匹配）
                generated_files = glob.glob(os.path.join(temp_dir, '*.mp4'))
                if not generated_files:
                    logger.warning(f"未找到任何生成的视频文件: {prompt} (菜品: {dish_id})")
                    continue
                
                # 取最新文件
                generated_file = max(generated_files, key=os.path.getmtime)
                
                # 验证文件是否匹配提示词（部分匹配）
                short_prompt = make_safe_prompt(prompt)[:50]  # 截断以匹配可能的文件名
                if short_prompt not in generated_file:
                    logger.warning(f"生成的视频文件 {generated_file} 与提示词 {prompt} 不完全匹配")
                
                shutil.move(generated_file, target_path)
                logger.info(f"成功生成并重命名视频: {target_path}")
                
                logger.info(f"生成视频后显存使用情况: {get_gpu_memory()}")
                clear_temp_dir(temp_dir)
                
            except MemoryError:
                logger.error(f"生成视频失败 {prompt} (菜品: {dish_id}): 显存不足")
            except Exception as e:
                logger.error(f"处理视频文件失败 {prompt} (菜品: {dish_id}): {e}")
        
        try:
            shutil.copy2(txt_path, os.path.join(dish_dir, txt_file))
            logger.info(f"已复制标注文件到 {dish_dir}/{txt_file}")
        except Exception as e:
            logger.error(f"复制 {txt_file} 失败: {e}")

def main():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    if not pynvml_initialized:
        initialize_pynvml()
    
    annotations_dir = 'sampled_videos/annotations'
    output_base_dir = 'generated_videos'
    temp_dir = './results_tmp'
    """
    sample_file = "/root/autodl-tmp/douguo/sampled_dishes.txt"
    dataset_base_dir = "/root/autodl-tmp/douguo/test"
    output_base_dir = "/root/autodl-tmp/douguo/t2v-hunyuan-douguo"
    temp_dir = "/root/autodl-tmp/douguo/results_tmp"
    """
    generate_videos_for_annotations(annotations_dir, output_base_dir, temp_dir)
    logger.info("所有视频生成和文件复制完成！")

if __name__ == "__main__":
    main()