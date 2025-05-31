import os
import json
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from PIL import Image
import time
import matplotlib.pyplot as plt
from collections import Counter

# 禁用 CuDNN（绕过 CUDNN_STATUS_NOT_SUPPORTED）
torch.backends.cudnn.enabled = False

# 检查 GPU 可用性
print("PyTorch 版本:", torch.__version__)
print("CUDA 可用:", torch.cuda.is_available())
print("CUDA 版本:", torch.version.cuda)
print("GPU 数量:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU 名称:", torch.cuda.get_device_name(0))
    print("总显存:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
else:
    print("错误：未检测到 GPU！请检查 vGPU 驱动、CUDA 和 PyTorch 安装。")
    exit(1)

# 设置设备
device = torch.device("cuda")
print(f"使用设备: {device}")

# 模型和处理器路径
model_path = "/root/models/Qwen2.5-VL-7B-Instruct/"
lora_path = "/root/LLaMA-Factory/saves/Qwen2.5-VL-7B-Instruct/lora/train_2025-05-28-00-02-03/checkpoint-3810"
assert os.path.exists(model_path)

# 加载基础模型
start_time = time.time()
print("开始加载基础模型...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    # ignore_mismatched_sizes=True    #不要开这个，这个会导致一直卡在加载模型
)
model.to(device)
print(f"基础模型加载完成，耗时: {time.time() - start_time:.2f} 秒")

# 加载 LoRA 微调
# print("开始加载 LoRA 微调...")
start_time = time.time()
model = PeftModel.from_pretrained(model, lora_path)
model.to(device)
print(f"LoRA 加载完成，耗时: {time.time() - start_time:.2f} 秒")
print(f"模型加载到设备: {model.device}")
print(f"模型显存占用: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")

# 加载处理器
print("开始加载处理器...")
start_time = time.time()
processor = AutoProcessor.from_pretrained(model_path, max_pixels=720*28*28)
print(f"处理器加载完成，耗时: {time.time() - start_time:.2f} 秒")

print('模型加载完毕！')

def is_valid_image(img_path):
    try:
        with Image.open(img_path) as img:
            img.verify()
            img = Image.open(img_path).convert("RGB")
            width, height = img.size
            if width > 2000 or height > 2000:
                print(f"图片 {img_path} 过大：{width}x{height}，跳过")
                return False
        return True
    except Exception as e:
        print(f"图片 {img_path} 损坏或无法打开：{e}")
        return False

def is_valid_tensor(tensor):
    if tensor is None:
        return False
    return not (torch.isnan(tensor).any() or torch.isinf(tensor).any())

def process_image(img_path, ingredients, query_template, model, processor, debug_log_file, error_log, memory_usage, processing_times):
    try:
        start_time = time.time()
        numbered_ingredients = [f"{i+1}. {ing}" for i, ing in enumerate(ingredients)]
        query = query_template.format(ingredients=', '.join(numbered_ingredients))

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": query}
                ]
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        for key, tensor in inputs.items():
            if not is_valid_tensor(tensor):
                error_log.append({
                    "image_path": img_path,
                    "ingredients": ingredients,
                    "error": f"Invalid tensor {key}",
                    "tensor_shape": str(tensor.shape) if tensor is not None else "None"
                })
                print(f"无效输入张量 {key} for {img_path}")
                return None, None
            print(f"输入张量 {key} 形状: {tensor.shape}")

        inputs = inputs.to(device)
        memory_usage.append(torch.cuda.memory_allocated(device) / 1024**3)
        print(f"处理图片 {img_path}")
        print(f"模型设备: {model.device}")
        print(f"输入设备: {inputs['input_ids'].device}")
        print(f"当前显存占用: {memory_usage[-1]:.2f} GB")

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        debug_info = f"图片 {img_path}，原料: {ingredients}，回答: {output_text}"
        print(debug_info)
        with open(debug_log_file, 'a', encoding='utf-8') as f:
            f.write(debug_info + '\n')

        response = [r.strip().lower() for r in output_text.split(',')]

        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        print(f"处理耗时: {processing_time:.2f} 秒")

        del inputs
        torch.cuda.empty_cache()
        memory_usage.append(torch.cuda.memory_allocated(device) / 1024**3)
        print(f"清理后显存占用: {memory_usage[-1]:.2f} GB")

        return query, response
    except Exception as e:
        error_log.append({
            "image_path": img_path,
            "ingredients": ingredients,
            "error": str(e)
        })
        print(f"处理图片 {img_path} 失败：{e}")
        return None, None

def read_dataset(file_path, max_samples=None):
    results = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if max_samples and i > max_samples:
                    break
                try:
                    example = json.loads(line.strip())
                    conversations = example.get("conversations", [])
                    images = example.get("images", [])
                    if len(conversations) < 2 or not images:
                        print(f"跳过第 {i} 行：无效的 conversations 或 images")
                        continue
                    if conversations[0]["from"] != "human" or conversations[1]["from"] != "assistant":
                        print(f"跳过第 {i} 行：无效的角色")
                        continue

                    ingredients = [ing.strip() for ing in conversations[1]["value"].split(",") if ing.strip()]
                    if not ingredients:
                        print(f"跳过第 {i} 行：空的原料列表")
                        continue

                    results.append({
                        "image_path": images[0],
                        "ingredients": ingredients,
                        "example": example
                    })
                except json.JSONDecodeError:
                    print(f"跳过无效 JSON 第 {i} 行：{line.strip()}")
    except Exception as e:
        print(f"读取 {file_path} 失败：{e}")
    return results

def plot_visualizations(memory_usage, processing_times, error_log, output_dir="visualizations"):
    os.makedirs(output_dir, exist_ok=True)

    # 显存使用曲线
    plt.figure(figsize=(10, 6))
    plt.plot(memory_usage, label="显存占用 (GB)", color='#1f77b4')
    plt.xlabel("处理步骤")
    plt.ylabel("显存 (GB)")
    plt.title("显存使用情况")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "memory_usage.png"))
    plt.close()

    # 处理时间直方图
    plt.figure(figsize=(10, 6))
    plt.hist(processing_times, bins=20, edgecolor='black', color='#ff7f0e')
    plt.xlabel("处理时间 (秒)")
    plt.ylabel("图片数量")
    plt.title("每张图片处理时间分布")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "processing_time_histogram.png"))
    plt.close()

    # 错误统计
    if error_log:
        error_counts = Counter(e["error"] for e in error_log)
        error_images = Counter(e["image_path"] for e in error_log)
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        labels, counts = zip(*error_counts.items())
        plt.bar(labels, counts, color='#2ca02c')
        plt.xlabel("错误类型")
        plt.ylabel("发生次数")
        plt.title("错误类型统计")
        plt.xticks(rotation=45, ha='right')
        plt.subplot(2, 1, 2)
        labels, counts = zip(*error_images.items())
        plt.bar(labels, counts, color='#d62728')
        plt.xlabel("图片路径")
        plt.ylabel("错误次数")
        plt.title("错误图片统计")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "error_stats.png"))
        plt.close()

def main(test_data_path, output_json, debug_log_file="debug_log.txt", error_log_file="error_log.json", max_samples=100):
    results = []
    total_yes = 0
    total_no = 0
    error_log = []
    memory_usage = []
    processing_times = []

    query_template = "Does the image contain the following ingredients: {ingredients}? For each ingredient, answer 'yes' or 'no' in the same order, separated by commas, without any additional text. For example, if the ingredients are mushrooms, garlic, onion, and only mushrooms and onion are present, answer 'yes, no, yes'."

    # 读取数据集
    print("开始读取数据集...")
    start_time = time.time()
    test_data = read_dataset(test_data_path, max_samples)
    total_images = len(test_data)
    print(f"共找到 {total_images} 个测试样本，耗时: {time.time() - start_time:.2f} 秒")
    if total_images == 0:
        print("没有有效的测试样本，退出。")
        return

    # 使用进度条
    for idx, item in enumerate(tqdm(test_data, desc="处理样本", unit="样本"), start=1):
        img_path = item["image_path"]
        ingredients = item["ingredients"]
        tqdm.write(f"[{idx}/{total_images}] 处理图片: {img_path}，原料数: {len(ingredients)}")

        if not os.path.exists(img_path):
            print(f"图片 {img_path} 不存在，跳过")
            error_log.append({
                "image_path": img_path,
                "ingredients": ingredients,
                "error": "Image file not found"
            })
            continue

        if not is_valid_image(img_path):
            continue

        query, response = process_image(img_path, ingredients, query_template, model, processor, debug_log_file, error_log, memory_usage, processing_times)
        if query is None or response is None:
            continue

        expected_len = len(ingredients)
        if len(response) != expected_len:
            # print(f"图片 {img_path} 的回答格式错误：{response}，预期 {expected_len} 个回答，补齐为 no")
            error_log.append({
                "image_path": img_path,
                "ingredients": ingredients,
                "response": response,
                "expected_length": expected_len,
                 "error": "Response length mismatch"
            })
            continue
            # response = response + ['no'] * (expected_len - len(response))

        yes_count = response.count("yes")
        total_count = len(response)
        yes_proportion = yes_count / total_count if total_count > 0 else 0

        results.append({
            "query": query,
            "response": response,
            "image_path": img_path,
            "yes_proportion": yes_proportion
        })

        total_yes += yes_count
        total_no += total_count - yes_count

        # 增量保存结果
        with open(output_json, 'a', encoding='utf-8') as f:
            f.write(json.dumps(results[-1], ensure_ascii=False) + '\n')

        # 绘制可视化（每 10 张图片更新一次）
        if len(results) % 10 == 0 or len(error_log) % 10 == 0:
            plot_visualizations(memory_usage, processing_times, error_log)

    # 最终可视化
    plot_visualizations(memory_usage, processing_times, error_log)

    # 保存错误日志
    if error_log:
        with open(error_log_file, 'w', encoding='utf-8') as f:
            json.dump(error_log, f, indent=2, ensure_ascii=False)
        print(f"错误日志已保存至 {error_log_file}")

    # 打印最终结果
    print(f"\n处理完成！")
    print(f"有效样本: {len(results)}/{total_images}")
    print(f"结果已保存至 {output_json}")
    print(f"总 yes 数量: {total_yes}")
    print(f"总 no 数量: {total_no}")
    print(f"整体 yes 比例: {total_yes / (total_yes + total_no) if total_yes + total_no > 0 else 0}")

if __name__ == "__main__":
    test_data_path = "/root/autodl-tmp/vqa-test.jsonl"
    output_json = "微调后-vqa-test-results.jsonl"
    main(test_data_path, output_json, max_samples=30000)