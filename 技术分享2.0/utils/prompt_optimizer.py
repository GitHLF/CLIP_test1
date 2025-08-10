import json

import numpy as np
import torch
import torch.nn.functional as F

from .image_processor import SimpleImageProcessor


def create_optimized_prompts():
    """创建优化的prompt集合"""
    prompt_groups = {
        "两只狗躺在垫子上晒太阳": [
            "两只狗躺在垫子上晒太阳",
            "两只狗在户外的垫子上休息",
            "two dogs lying on a cushion in the sun",
            "two dogs resting on a mat outdoors"
        ],
        "白狗和棕狗在绿垫子上": [
            "一只白色的小狗坐在绿色垫子上，旁边是一只棕白相间的狗",
            "白色小狗和棕色狗在垫子上",
            "a white dog and a brown dog on a green mat",
            "white puppy sitting next to a brown and white dog"
        ],
        "两只狗在露台休息": [
            "两只狗在室外的露台上休息，背景有红色小树",
            "狗狗们在阳台上放松",
            "two dogs relaxing on an outdoor patio",
            "dogs resting on a terrace with plants in background"
        ],
        "猫在沙发上": [
            "一只猫趴在沙发上",
            "猫咪在室内沙发休息",
            "a cat lying on a sofa",
            "cat resting on indoor furniture"
        ],
        "寿司食物": [
            "一盘寿司在桌上",
            "日式寿司料理",
            "sushi on a plate",
            "Japanese food on table"
        ]
    }
    return prompt_groups

def load_prompt_groups(prompt_file):
    """
    从JSON文件加载自定义prompt配置

    Args:
        prompt_file (str): JSON文件路径

    Returns:
        dict: 概念组和对应的prompt列表
    """
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_groups = json.load(f)

        # 验证格式是否正确
        if not isinstance(prompt_groups, dict):
            print(f"⚠️ 格式错误: prompt文件应该是一个字典")
            return create_optimized_prompts()

        # 验证每个组是否包含prompt列表
        for group_name, prompts in prompt_groups.items():
            if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
                print(f"⚠️ 格式错误: 组 '{group_name}' 的prompts应该是字符串列表")
                return create_optimized_prompts()

        print(f"✓ 成功加载自定义prompt配置，包含 {len(prompt_groups)} 个概念组")
        return prompt_groups
    except Exception as e:
        print(f"⚠️ 加载prompt文件失败: {e}，使用默认配置")
        return create_optimized_prompts()

def analyze_with_optimized_prompts(model, tokenizer, image, device="cpu", prompts_per_group=5):
    """
    使用优化的prompt进行分析 - 支持可配置的prompt数量

    Args:
        model: CLIP模型
        tokenizer: 分词器
        image: 输入图像
        device: 设备
        prompts_per_group (int): 每个概念组使用的prompt数量
                                1 = 单prompt模式（只选择每组第一个）
                                >1 = 多prompt模式（选择每组前N个）
    """
    print("步骤3: 使用优化的多prompt分析")
    print(f"配置: 每个概念组使用 {prompts_per_group} 个prompt")

    # 图像预处理
    processor = SimpleImageProcessor()
    image_tensor = processor(image).unsqueeze(0).to(device)

    # 获取优化的prompt组
    prompt_groups = create_optimized_prompts()

    # 根据配置参数选择每组的prompt数量
    filtered_prompt_groups = {}
    for group_name, prompts in prompt_groups.items():
        # 选择每组的前N个prompt
        selected_prompts = prompts[:prompts_per_group]
        filtered_prompt_groups[group_name] = selected_prompts

        if prompts_per_group == 1:
            print(f"概念组 '{group_name}': 使用单个prompt - '{selected_prompts[0]}'")
        else:
            print(f"概念组 '{group_name}': 使用 {len(selected_prompts)} 个prompts")

    # 收集所有prompts进行批量处理
    all_prompts = []
    prompt_to_group = {}

    for group_name, prompts in filtered_prompt_groups.items():
        for prompt in prompts:
            all_prompts.append(prompt)
            prompt_to_group[prompt] = group_name

    print(f"总共分析 {len(all_prompts)} 个prompt")

    # 批量处理所有prompts
    try:
        # 文本tokenization - 一次性处理所有文本
        text_inputs = tokenizer(
            all_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        with torch.no_grad():
            # 使用完整的CLIP模型进行批量推理
            outputs = model(
                pixel_values=image_tensor,
                input_ids=text_inputs['input_ids'],
                attention_mask=text_inputs['attention_mask']
            )

            # 获取相似度分数
            logits_per_image = outputs.logits_per_image  # [1, num_texts]
            probs = F.softmax(logits_per_image, dim=-1)[0]  # [num_texts]

            print(f"获得 {len(probs)} 个概率分数")

    except Exception as e:
        print(f"批量处理失败: {e}")
        return {}, []

    # 整理结果
    group_results = {}
    all_individual_results = []

    for group_name, prompts in filtered_prompt_groups.items():
        group_scores = []

        print(f"\n分析概念组: {group_name}")
        if prompts_per_group == 1:
            print(f"单prompt模式")
        else:
            print(f"包含 {len(prompts)} 个描述变体")

        for prompt in prompts:
            # 找到这个prompt在all_prompts中的索引
            prompt_idx = all_prompts.index(prompt)
            prob = probs[prompt_idx].item()

            group_scores.append(prob)
            all_individual_results.append((prompt, prob))

            print(f"  '{prompt}': {prob:.4f}")

        # 计算组平均分数
        if group_scores:
            if prompts_per_group == 1:
                # 单prompt模式，平均分数就是唯一的分数
                avg_score = group_scores[0]
                print(f"  分数: {avg_score:.4f}")
            else:
                # 多prompt模式，计算平均分数
                avg_score = np.mean(group_scores)
                std_score = np.std(group_scores)
                print(f"  组平均分数: {avg_score:.4f}")
                print(f"  标准差: {std_score:.4f}")

            group_results[group_name] = {
                'average_score': avg_score,
                'individual_scores': group_scores,
                'prompts': prompts
            }

    return group_results, all_individual_results, prompt_to_group

