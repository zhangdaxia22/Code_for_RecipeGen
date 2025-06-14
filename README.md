# Code_for_RecipeGen

Homepage : [ [RecipeGen: A Step-Aligned Multimodal Benchmark for Real-World Recipe Generation](https://wenbin08.github.io/RecipeGen/)]

## Baseline for T2I models
- Stable Diffusion 1.5, 2.1, xl, 3.5 [ [paper](https://arxiv.org/abs/2112.10752)]
- Flux.1-dev [ [website](https://blackforestlabs.ai)]
- In-Context LoRA [ [paper](https://arxiv.org/abs/2410.23775)]
- Regional-Prompting FLUX [ [paper](https://arxiv.org/abs/2411.02395)]
## Baseline for Video Generation models.
- Opensora [ [paper](https://arxiv.org/abs/2412.20404)]
- HunyuanVideo [ [github](https://github.com/Tencent-Hunyuan/HunyuanVideo)]
## Metric 
- Ingredient Accuracy (Based on [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct))
- Interaction Faithfulness  (Based on [GPT-4o](https://openai.com/index/hello-gpt-4o/))
- Cross-Step Consistency (Based on [DINOv2](https://arxiv.org/abs/2304.07193))
- Goal and Step Faithfulness (Based on [CLIP](https://arxiv.org/abs/2103.00020))
