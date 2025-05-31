import os
import cv2
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

def get_keyframes(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [0, total // 2, total - 1] if total >= 3 else [0] * 3
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb).resize((224, 224))
            frames.append(image)
    cap.release()
    return frames

def get_last_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
    ret, frame = cap.read()
    cap.release()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb).resize((224, 224))
    return None

def compute_clip_score(image, text):
    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        clip_score = outputs.logits_per_image.item()
    return clip_score

def main():
    base_dir = "./i2v-opensora-youcook2"
    output_score_file = os.path.join(os.path.dirname(base_dir), "clip_scores.txt")
    output_goal_file = os.path.join(os.path.dirname(base_dir), "goal_clip_scores.txt")

    all_step_avg_scores = []
    all_goal_scores = []

    with open(output_score_file, "w") as score_out, open(output_goal_file, "w") as goal_out:
        for dish in tqdm(os.listdir(base_dir)):
            dish_path = os.path.join(base_dir, dish)
            if not os.path.isdir(dish_path):
                continue

            steps_file = os.path.join(dish_path, "steps.txt")
            if not os.path.exists(steps_file):
                continue

            with open(steps_file, "r") as f:
                steps = [line.strip() for line in f if line.strip()]
            if not steps:
                continue

            dish_scores = []
            for idx, step_text in enumerate(steps):
                video_path = os.path.join(dish_path, f"{idx+1}.mp4")
                if not os.path.exists(video_path):
                    continue
                frames = get_keyframes(video_path)
                scores = [compute_clip_score(img, step_text) for img in frames]
                avg = sum(scores) / len(scores)
                dish_scores.append(avg)
                score_out.write(f"{dish} Step {idx+1}: {scores}, Avg: {avg:.4f}\n")

            if dish_scores:
                avg_dish_score = sum(dish_scores) / len(dish_scores)
                all_step_avg_scores.append(avg_dish_score)
                score_out.write(f"{dish} Overall Avg: {avg_dish_score:.4f}\n\n")

            # goal score：使用最后一个step作为文本 + 最后一个视频的最后一帧作为图像
            video_files = [f for f in os.listdir(dish_path) if f.endswith(".mp4")]
            if not video_files:
                continue
            last_video = sorted(video_files, key=lambda x: int(x.split('.')[0]))[-1]
            last_video_path = os.path.join(dish_path, last_video)
            last_frame = get_last_frame(last_video_path)
            if last_frame is None:
                continue

            goal_text = steps[-1]  # 使用最后一步作为goal文本
            goal_score = compute_clip_score(last_frame, goal_text)
            all_goal_scores.append(goal_score)
            goal_out.write(f"{dish} Goal Score: {goal_score:.4f}\n")

        # 汇总平均分
        if all_step_avg_scores:
            score_out.write(f"\nTotal Avg Clip Score: {sum(all_step_avg_scores)/len(all_step_avg_scores):.4f}\n")
        if all_goal_scores:
            goal_out.write(f"\nTotal Avg Goal Score: {sum(all_goal_scores)/len(all_goal_scores):.4f}\n")

if __name__ == "__main__":
    main()


"""
遍历 ./i2v-opensora-youcook2/ 下的所有菜品文件夹。(切换不同目录)

每个步骤视频（如 1.mp4、2.mp4 等）：

提取首帧、中帧、尾帧。

分别计算与该步骤文本的 CLIP Score（未归一化，归一化版本已注释，可启用）。

每个菜品写入一个 .txt 文件汇总所有步骤的各帧得分、平均分。

使用最后一个步骤的文本（steps.txt 中最后一行） 作为目标描述，结合 最后一个视频的最后一帧图像，
计算 goal faithfulness 的 CLIP 分数
"""