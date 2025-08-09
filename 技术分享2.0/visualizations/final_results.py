from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def visualize_final_image_with_result(original_image, group_results, output_dir="outputs"):
    """
    可视化最终图像和结果

    Args:
        original_image: 原始图像
        group_results: 概念组分析结果
        output_dir: 输出目录
    """
    print("生成最终图像和结果可视化...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    plt.figure(figsize=(12, 10))
    plt.title('CLIP多Prompt分析结果', fontsize=16, fontweight='bold')

    # 显示原始图像
    plt.imshow(original_image)
    plt.axis('off')

    # 获取排序后的组
    sorted_groups = sorted(group_results.items(), key=lambda x: x[1]['average_score'], reverse=True)

    # 添加结果文本
    result_text = "分析结果 (按匹配分数排序):\n\n"
    for i, (group_name, group_data) in enumerate(sorted_groups[:3]):  # 只显示前3个结果
        avg_score = group_data['average_score']
        result_text += f"{i+1}. {group_name}\n"
        result_text += f"   平均分: {avg_score:.4f}\n"
        result_text += f"   变体数: {len(group_data['prompts'])}\n\n"

    # 在图像右侧添加文本框
    plt.figtext(0.7, 0.5, result_text, ha='left', va='center', fontsize=12,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

    # 保存图像
    output_path = Path(output_dir) / "28_final_image_with_result.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"最终图像和结果可视化已保存: {output_path}")

def visualize_group_ranking(group_results, output_dir="outputs"):
    """
    可视化概念组排序

    Args:
        group_results: 概念组分析结果
        output_dir: 输出目录
    """
    print("生成概念组排序可视化...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    plt.figure(figsize=(12, 8))
    plt.title('概念组匹配排序', fontsize=16, fontweight='bold')

    # 获取排序后的组
    sorted_groups = sorted(group_results.items(), key=lambda x: x[1]['average_score'], reverse=True)
    group_names = [item[0] for item in sorted_groups]
    avg_scores = [item[1]['average_score'] for item in sorted_groups]

    # 使用颜色渐变
    colors = plt.cm.RdYlGn(np.linspace(0.3, 1, len(group_names)))

    # 绘制水平条形图
    bars = plt.barh(range(len(group_names)), avg_scores, color=colors)
    plt.yticks(range(len(group_names)),
              [f"{i+1}. {name}" for i, name in enumerate(group_names)])
    plt.xlabel('平均匹配分数', fontsize=12)
    plt.grid(axis='x', alpha=0.3)

    # 添加数值标签
    for i, (bar, score) in enumerate(zip(bars, avg_scores)):
        plt.text(score + 0.001, i, f'{score:.4f}',
                va='center', fontsize=10, fontweight='bold')

    # 保存图像
    output_path = Path(output_dir) / "29_group_ranking.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"概念组排序可视化已保存: {output_path}")

def visualize_top_groups_comparison(group_results, output_dir="outputs"):
    """
    可视化Top-3概念组详细对比

    Args:
        group_results: 概念组分析结果
        output_dir: 输出目录
    """
    print("生成Top-3概念组详细对比可视化...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    plt.figure(figsize=(14, 8))
    plt.title('Top-3概念组变体对比', fontsize=16, fontweight='bold')

    # 获取排序后的前3个组
    sorted_groups = sorted(group_results.items(), key=lambda x: x[1]['average_score'], reverse=True)[:3]
    group_names = [item[0] for item in sorted_groups]

    # 计算每组最多有多少个变体
    max_variants = max([len(item[1]['prompts']) for item in sorted_groups])

    # 设置x轴位置
    x = np.arange(len(group_names))
    width = 0.8 / max_variants

    # 为每个变体绘制条形图
    for i in range(max_variants):
        variant_scores = []
        for group_name, group_data in sorted_groups:
            if i < len(group_data['individual_scores']):
                variant_scores.append(group_data['individual_scores'][i])
            else:
                variant_scores.append(0)

        offset = -0.4 + (i + 0.5) * width
        bars = plt.bar(x + offset, variant_scores, width, label=f'变体{i+1}', alpha=0.8)

        # 添加数值标签
        for j, score in enumerate(variant_scores):
            if score > 0:  # 只为有效分数添加标签
                plt.text(j + offset, score + 0.005, f'{score:.3f}',
                        ha='center', va='bottom', fontsize=8, rotation=90)

    plt.xlabel('概念组', fontsize=12)
    plt.ylabel('匹配分数', fontsize=12)
    plt.xticks(x, [f"{i+1}. {name[:15]}..." if len(name) > 15 else f"{i+1}. {name}"
                  for i, name in enumerate(group_names)])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    # 添加变体详情说明
    variant_text = "变体详情:\n\n"
    for i, (group_name, group_data) in enumerate(sorted_groups):
        variant_text += f"组{i+1}: {group_name}\n"
        for j, prompt in enumerate(group_data['prompts']):
            variant_text += f"  变体{j+1}: {prompt}\n"
        variant_text += "\n"

    plt.figtext(0.5, 0.01, variant_text, ha='center', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    # 保存图像
    output_path = Path(output_dir) / "30_top_groups_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Top-3概念组详细对比可视化已保存: {output_path}")

def visualize_best_match_details(group_results, output_dir="outputs"):
    """
    可视化最佳匹配详情

    Args:
        group_results: 概念组分析结果
        output_dir: 输出目录
    """
    print("生成最佳匹配详情可视化...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.title('最佳匹配详情', fontsize=16, fontweight='bold')

    # 获取最佳匹配组
    best_group = sorted(group_results.items(), key=lambda x: x[1]['average_score'], reverse=True)[0]
    group_name, group_data = best_group

    # 绘制最佳组的所有变体得分
    prompts = group_data['prompts']
    scores = group_data['individual_scores']

    # 使用颜色渐变
    colors = plt.cm.viridis(np.linspace(0, 1, len(prompts)))

    # 绘制条形图
    bars = plt.bar(range(len(prompts)), scores, color=colors)
    plt.xticks(range(len(prompts)), [f"变体{i+1}" for i in range(len(prompts))], rotation=45)
    plt.xlabel('Prompt变体', fontsize=12)
    plt.ylabel('匹配分数', fontsize=12)
    plt.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{score:.4f}', ha='center', va='bottom', fontsize=10)

    # 添加组名和变体详情
    plt.figtext(0.5, 0.95, f"最佳匹配组: {group_name}", ha='center', fontsize=14,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))

    variant_text = "变体详情:\n\n"
    for i, prompt in enumerate(prompts):
        variant_text += f"变体{i+1}: {prompt}\n"

    plt.figtext(0.5, 0.01, variant_text, ha='center', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    # 保存图像
    output_path = Path(output_dir) / "31_best_match_details.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"最佳匹配详情可视化已保存: {output_path}")

def create_final_results_visualizations(original_image, group_results, output_dir="outputs"):
    """
    创建所有最终结果相关的可视化

    Args:
        original_image: 原始图像
        group_results: 概念组分析结果
        output_dir: 输出目录
    """
    print("生成最终结果可视化...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    # 1. 最终图像和结果
    visualize_final_image_with_result(original_image, group_results, output_dir)

    # 2. 概念组排序
    visualize_group_ranking(group_results, output_dir)

    # 3. Top-3概念组详细对比
    visualize_top_groups_comparison(group_results, output_dir)

    # 4. 最佳匹配详情
    visualize_best_match_details(group_results, output_dir)

    print("最终结果可视化完成")

