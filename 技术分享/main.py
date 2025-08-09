import gc
from pathlib import Path

import numpy as np
import psutil
import torch
from PIL import Image

from utils.attention_extractor import extract_attention_weights
from utils.image_processor import SimpleImageProcessor
from utils.model_loader import load_local_clip_model
from utils.prompt_optimizer import analyze_with_optimized_prompts
from visualizations.cross_modal_attention import create_cross_modal_attention_visualization
from visualizations.image_preprocessing import create_image_preprocessing_visualization
from visualizations.prompt_analysis import create_prompt_analysis_visualization


def print_memory_usage():
    """打印内存使用情况"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        print(f"内存使用: {memory_info.rss / 1024 / 1024:.1f} MB")

        if torch.cuda.is_available():
            print(f"GPU内存: {torch.cuda.memory_allocated() / 1024 / 1024:.1f} MB")
    except:
        pass


def force_cleanup(device):
    """强制清理内存"""
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main(prompts_per_group=5):
    """
    主函数

    Args:
        prompts_per_group (int): 每个概念组使用的prompt数量
                                1 = 单prompt模式（只选择每组第一个）
                                >1 = 多prompt模式（选择每组前N个）
    """
    print("CLIP多Prompt优化分析演示 - 包含详细注意力机制")
    print("=" * 60)
    print(f"配置: 每个概念组使用 {prompts_per_group} 个prompt")

    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 创建输出目录（根据配置命名）
    if prompts_per_group == 1:
        output_dir = "outputs_single_prompt"
        mode_name = "单Prompt模式"
    else:
        output_dir = f"outputs_{prompts_per_group}prompts"
        mode_name = f"多Prompt模式({prompts_per_group}个/组)"

    Path(output_dir).mkdir(exist_ok=True)
    print(f"输出目录: {output_dir}")
    print(f"分析模式: {mode_name}")

    # 1. 加载本地模型
    model, tokenizer, model_loaded = load_local_clip_model()

    if not model_loaded:
        print("无法加载本地模型，请确保模型文件存在")
        return

    model = model.to(device).eval()

    # 2. 准备图像
    img_path = "dogs_sun_patio.jpeg"

    try:
        if Path(img_path).exists():
            print(f"加载图像: {img_path}")
            image = Image.open(img_path).convert('RGB')
        else:
            print(f"图像文件 {img_path} 不存在")
            return
    except Exception as e:
        print(f"加载图像失败: {e}")
        return

    # 3. 图像预处理
    processor = SimpleImageProcessor()
    processed_image_tensor = processor(image)

    # 4. 使用优化的多prompt进行分析（传入配置参数）
    group_results, all_individual_results = analyze_with_optimized_prompts(
        model, tokenizer, image, device, prompts_per_group=prompts_per_group
    )

    print(f"\n总共分析了 {len(all_individual_results)} 个prompts")
    print(f"分为 {len(group_results)} 个概念组")

    print_memory_usage()
    force_cleanup(device)

    # 5. 提取注意力权重（现在可以处理更多样本）
    print("\n提取注意力权重用于详细分析...")
    # 增加样本数量，选择每个组的最佳prompt
    sample_texts = []

    # 从每个组选择最佳的prompt
    for group_name, group_data in group_results.items():
        best_idx = np.argmax(group_data['individual_scores'])
        best_prompt = group_data['prompts'][best_idx]
        sample_texts.append(best_prompt)
        print(f"从组 '{group_name}' 选择: {best_prompt}")

    # 限制在前5个组，避免内存问题
    sample_texts = sample_texts[:5]

    attention_data = None
    try:
        print(f"一次性处理 {len(sample_texts)} 个文本...")
        attention_data = extract_attention_weights(model, tokenizer, image, sample_texts, device)

        if attention_data:
            print("✓ 注意力数据提取成功")
            print(f"  - vision_attentions: {len(attention_data.get('vision_attentions', [])) if attention_data.get('vision_attentions') else 0} 层")
            print(f"  - text_attentions: {len(attention_data.get('text_attentions', [])) if attention_data.get('text_attentions') else 0} 层")

            similarities = attention_data.get('similarities')
            if similarities is not None:
                print(f"  - similarities: {similarities.shape}")
            else:
                print(f"  - similarities: None")

            print(f"  - texts: {len(attention_data.get('texts', []))} 个")

            # 打印文本内容以确认数据完整性
            if attention_data.get('texts'):
                print("  - 选中的文本内容:")
                for i, text in enumerate(attention_data['texts']):
                    print(f"    {i+1}. {text}")
        else:
            print("⚠️ 注意力数据为空")

        print_memory_usage()
        force_cleanup(device)

    except Exception as e:
        print(f"✗ 提取注意力权重失败: {e}")
        import traceback
        traceback.print_exc()
        attention_data = None

    # 6. 创建详细可视化
    print("\n生成详细可视化...")

    # 步骤1: 图像预处理可视化
    try:
        print("步骤1: 生成图像预处理可视化...")
        print_memory_usage()

        create_image_preprocessing_visualization(image, processed_image_tensor, attention_data, output_dir)
        print("✓ 图像预处理可视化完成")

        force_cleanup(device)
        print_memory_usage()

    except Exception as e:
        print(f"✗ 图像预处理可视化失败: {e}")
        import traceback
        traceback.print_exc()

    # 步骤2: 文本分析可视化
    try:
        print("步骤2: 生成文本分析可视化...")
        print_memory_usage()

        # 验证数据完整性
        if attention_data:
            print("验证注意力数据完整性:")
            print(f"  - texts: {attention_data.get('texts', 'None')}")
            print(f"  - text_inputs keys: {list(attention_data.get('text_inputs', {}).keys())}")
            if attention_data.get('text_inputs'):
                print(f"  - input_ids shape: {attention_data['text_inputs']['input_ids'].shape}")
                print(f"  - attention_mask shape: {attention_data['text_inputs']['attention_mask'].shape}")

        create_prompt_analysis_visualization(group_results, all_individual_results, attention_data, output_dir)
        print("✓ 文本分析可视化完成")

        force_cleanup(device)
        print_memory_usage()

    except Exception as e:
        print(f"✗ 文本分析可视化失败: {e}")
        import traceback
        traceback.print_exc()

    # 步骤3: 跨模态注意力可视化
    try:
        print("步骤3: 生成跨模态注意力可视化...")
        print_memory_usage()

        # 详细的数据验证和调试信息
        if not attention_data:
            print("⚠️ 跳过跨模态注意力可视化 - attention_data 为空")
        else:
            print("✓ attention_data 存在，检查各个组件...")

            # 检查各个必需的数据组件
            vision_attentions = attention_data.get('vision_attentions')
            text_attentions = attention_data.get('text_attentions')
            similarities = attention_data.get('similarities')
            texts = attention_data.get('texts')

            print(f"  - vision_attentions: {len(vision_attentions) if vision_attentions else 0} 层")
            print(f"  - text_attentions: {len(text_attentions) if text_attentions else 0} 层")
            print(f"  - similarities: {similarities.shape if similarities is not None else 'None'}")
            print(f"  - texts: {len(texts) if texts else 0} 个")

            # 更宽松的验证条件
            missing_components = []
            if not vision_attentions:
                missing_components.append('vision_attentions')
            if not text_attentions:
                missing_components.append('text_attentions')
            if similarities is None:
                missing_components.append('similarities')
            if not texts:
                missing_components.append('texts')

            if missing_components:
                print(f"⚠️ 跳过跨模态注意力可视化 - 缺少组件: {', '.join(missing_components)}")
            else:
                print("✓ 所有数据组件完整，开始生成跨模态注意力可视化...")
                create_cross_modal_attention_visualization(image, attention_data, output_dir)
                print("✓ 跨模态注意力可视化完成")

        force_cleanup(device)
        print_memory_usage()

    except Exception as e:
        print(f"✗ 跨模态注意力可视化失败: {e}")
        import traceback
        traceback.print_exc()

    # 检查生成的文件
    print("\n检查生成的文件:")
    output_files = [
        ("01_image_processing_detailed.png", "图像处理和Vision Transformer详细过程"),
        ("02_text_attention_analysis.png", "文本处理和注意力机制分析"),
        ("03_cross_modal_attention.png", "跨模态注意力对应关系")
    ]

    generated_files = []
    for filename, description in output_files:
        filepath = Path(output_dir) / filename
        if filepath.exists():
            file_size = filepath.stat().st_size / 1024  # KB
            print(f"✓ {filename} - {file_size:.1f} KB")
            generated_files.append((filename, description))
        else:
            print(f"✗ {filename} - 文件不存在")

    # 7. 打印最终结果
    print(f"\n最终分析结果 ({mode_name}):")
    print("-" * 60)

    sorted_groups = sorted(group_results.items(), key=lambda x: x[1]['average_score'], reverse=True)

    for i, (group_name, group_data) in enumerate(sorted_groups):
        rank = i + 1
        avg_score = group_data['average_score']
        std_score = np.std(group_data['individual_scores']) if len(group_data['individual_scores']) > 1 else 0.0

        print(f"{rank}. {group_name}")
        print(f"   平均分数: {avg_score:.4f}")
        if prompts_per_group > 1:
            print(f"   标准差: {std_score:.4f}")
        print(f"   使用变体: {len(group_data['prompts'])}个")

        # 显示该组的所有prompts
        print("   变体详情:")
        for j, (prompt, score) in enumerate(zip(group_data['prompts'], group_data['individual_scores'])):
            print(f"     {j+1}. {prompt}: {score:.4f}")
        print()

    print(f"分析完成！生成的详细可视化文件 ({mode_name}):")
    for filename, description in generated_files:
        print(f"- {output_dir}/{filename}: {description}")

    if not generated_files:
        print("⚠️ 没有成功生成任何可视化文件")

    # 最终清理
    force_cleanup(device)
    print_memory_usage()


if __name__ == "__main__":
    # 可以通过修改这个参数来切换模式
    PROMPTS_PER_GROUP = 4  # 改为1表示单prompt模式，改为5表示多prompt模式

    print(f"启动分析，配置: 每组使用 {PROMPTS_PER_GROUP} 个prompt")
    main(prompts_per_group=PROMPTS_PER_GROUP)

