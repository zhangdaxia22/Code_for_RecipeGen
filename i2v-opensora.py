import os
import shutil
import subprocess
import glob
import time
from pathlib import Path
from tqdm import tqdm
try:
    import pynvml
    pynvml_initialized = False
except ImportError:
    pynvml = None
    pynvml_initialized = True  # Skip initialization if pynvml is not installed

def initialize_pynvml():
    """初始化 pynvml，检测 GPU 可用性"""
    global pynvml_initialized
    if pynvml is None:
        print("pynvml 未安装，无法监控显存。请安装：pip install pynvml")
        pynvml_initialized = True
        return
    try:
        pynvml.nvmlInit()
        pynvml_initialized = True
        print("pynvml 初始化成功")
    except pynvml.NVMLError as e:
        print(f"pynvml 初始化失败: {e}. 无法监控显存")
        pynvml_initialized = True

def get_gpu_memory():
    """获取当前 GPU 显存使用情况"""
    if pynvml is None or not pynvml_initialized:
        return "显存监控不可用"
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        memory_info = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used = mem_info.used / 1024**2  # 转换为 MB
            total = mem_info.total / 1024**2  # 转换为 MB
            memory_info.append(f"GPU {i}: {used:.2f}/{total:.2f} MB ({used/total*100:.1f}%)")
        return "; ".join(memory_info)
    except pynvml.NVMLError as e:
        return f"获取显存失败: {e}"

def clear_temp_dir(temp_dir):
    """清空临时文件夹中的 .mp4 和 .txt 文件"""
    for file in glob.glob(os.path.join(temp_dir, '*.mp4')) + glob.glob(os.path.join(temp_dir, '*.txt')):
        try:
            os.remove(file)
            print(f"已删除临时文件: {file}")
        except Exception as e:
            print(f"删除临时文件 {file} 失败: {e}")

def make_safe_prompt(prompt):
    """将提示词中的非法字符替换为下划线"""
    return prompt.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')

def generate_i2v_videos_for_dishes(dish_base_dir, temp_dir='./results'):
    """
    为每个菜品文件夹中的 steps.txt 和对应图片生成视频，保存到 video_256px 目录，完成后重命名为 videos。
    如果目标视频已存在，则跳过该步骤的生成。
    :param dish_base_dir: 菜品文件夹根目录（如 '/root/autodl-tmp/Open-Sora/sampled-yooucook2-cleaned'）
    :param temp_dir: inference.py 生成视频的临时文件夹（如 './results'）
    """
    # 初始化 pynvml
    if not pynvml_initialized:
        initialize_pynvml()
    
    # 确保临时文件夹存在
    os.makedirs(temp_dir, exist_ok=True)
    
    # 获取所有菜品文件夹
    dish_ids = [d for d in os.listdir(dish_base_dir) if os.path.isdir(os.path.join(dish_base_dir, d))]
    
    # 外层进度：菜品文件夹
    for dish_idx, dish_id in enumerate(tqdm(dish_ids, desc="处理菜品文件夹", position=0), 1):
        dish_dir = os.path.join(dish_base_dir, dish_id)
        print(f"\n当前处理第 {dish_idx}/{len(dish_ids)} 个菜品: {dish_id}")
        
        # 检查 steps.txt 是否存在
        steps_file = os.path.join(dish_dir, 'steps.txt')
        if not os.path.exists(steps_file):
            print(f"未找到 steps.txt，跳过菜品: {dish_id}")
            continue
        
        # 读取步骤
        with open(steps_file, 'r', encoding='utf-8') as f:
            steps = [line.strip() for line in f if line.strip()]
        
        # 创建 video_256px 目录
        video_dir = os.path.join(dish_dir, 'video_256px')
        os.makedirs(video_dir, exist_ok=True)
        
        # 内层进度条：步骤
        for step in tqdm(steps, desc=f"处理 {dish_id} 的步骤", leave=False, position=1):
            # 解析步骤编号和描述（格式：1. description）
            try:
                step_num, prompt = step.split('. ', 1)
                step_num = step_num.strip()
                prompt = prompt.strip()
            except ValueError:
                print(f"步骤格式错误，跳过: {step} (菜品: {dish_id})")
                continue
                
            # 检查对应图片是否存在
            image_path = os.path.join(dish_dir, f"{step_num}.jpg")
            if not os.path.exists(image_path):
                print(f"未找到图片 {image_path}，跳过步骤: {step_num} (菜品: {dish_id})")
                continue
                
            # 构造目标文件名
            safe_prompt = make_safe_prompt(prompt)
            target_name = f"{step_num}-{safe_prompt}.mp4"
            target_path = os.path.join(video_dir, target_name)
            
            # 检查目标视频是否已存在
            if os.path.exists(target_path):
                print(f"目标视频已存在，跳过: {target_path}")
                continue
                
            # 构造调用指令
            command = [
                'torchrun', '--nproc_per_node', '1', '--standalone',
                'scripts/diffusion/inference.py',
                'configs/diffusion/inference/256px.py',
                '--cond_type', 'i2v_head',
                '--prompt', prompt,
                '--ref', image_path,
                '--save_dir', temp_dir
            ]
            
            # 执行指令并实时输出
            try:
                print(f"正在为 {dish_id} 的步骤 {step_num} 生成视频: {prompt}")
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # 实时打印 stdout 和 stderr
                while process.poll() is None:
                    stdout_line = process.stdout.readline().strip()
                    if stdout_line:
                        print(stdout_line)
                    stderr_line = process.stderr.readline().strip()
                    if stderr_line:
                        print(stderr_line)
                
                # 确保读取剩余输出
                stdout, stderr = process.communicate()
                if stdout:
                    print(stdout.strip())
                if stderr:
                    print(stderr.strip())
                
                # 检查命令是否成功
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, command, stdout, stderr)
                
                # 查找生成的视频文件
                generated_video = os.path.join(temp_dir, 'video_256px', 'prompt_0000.mp4')
                if not os.path.exists(generated_video):
                    print(f"未找到生成的视频文件: {generated_video} (菜品: {dish_id})")
                    continue
                
                # 移动并重命名视频
                shutil.move(generated_video, target_path)
                print(f"成功生成并重命名视频: {target_path}")
                
                # 输出当前显存
                print(f"当前显存使用情况: {get_gpu_memory()}")
                
                # 清空临时文件夹
                clear_temp_dir(os.path.join(temp_dir, 'video_256px'))
                
            except subprocess.CalledProcessError as e:
                print(f"生成视频失败 {prompt} (菜品: {dish_id}): {e.stderr}")
            except Exception as e:
                print(f"处理视频文件失败 {prompt} (菜品: {dish_id}): {e}")
        
        # 检查是否所有步骤都生成了视频
        videos_generated = True
        for step in steps:
            if '. ' not in step:
                continue
            step_num, prompt = step.split('. ', 1)
            step_num = step_num.strip()
            safe_prompt = make_safe_prompt(prompt.strip())
            video_path = os.path.join(video_dir, f"{step_num}-{safe_prompt}.mp4")
            if not os.path.exists(video_path):
                videos_generated = False
                break
        
        if videos_generated:
            # 重命名 video_256px 为 videos
            videos_dir = os.path.join(dish_dir, 'videos')
            if os.path.exists(videos_dir):
                shutil.rmtree(videos_dir)
            os.rename(video_dir, videos_dir)
            print(f"已将 {video_dir} 重命名为 {videos_dir}")
            
            # 删除 videos 目录下的所有 .txt 文件
            for txt_file in glob.glob(os.path.join(videos_dir, '*.txt')):
                try:
                    os.remove(txt_file)
                    print(f"已删除文本文件: {txt_file}")
                except Exception as e:
                    print(f"删除文本文件 {txt_file} 失败: {e}")
            
            # 复制 steps.txt 到 videos 目录
            shutil.copy2(steps_file, os.path.join(videos_dir, 'steps.txt'))
            print(f"已复制 steps.txt 到 {videos_dir}/steps.txt")

def main():
    # 输入和输出路径
    dish_base_dir = '/root/autodl-tmp/Open-Sora/sampled-yooucook2-cleaned'
    temp_dir = './results'  # 临时文件夹，与指令中的 --save_dir 一致
    
    """
    sample_file = '/root/autodl-tmp/douguo/sampled_dishes.txt'
    dataset_base_dir = '/root/autodl-tmp/douguo/test'
    output_base_dir = '/root/autodl-tmp/douguo/i2v-opensora-douguo'
    temp_dir = '/root/autodl-tmp/douguo/results_tmp'
    """
    # 执行生成
    generate_i2v_videos_for_dishes(dish_base_dir, temp_dir)
    print("所有视频生成和文件复制完成！")

if __name__ == "__main__":
    main()