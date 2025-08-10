from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def visualize_concept_groups_overview(group_results, output_dir="outputs"):
    """
    可视化概念组分析概览

    Args:
        group_results: 概念组分析结果
        output_dir: 输出目录
    """
    print("生成概念组分析概览...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.title('概念组分析概览', fontsize=16, fontweight='bold')

    # 显示所有概念组的信息
    text_display = "概念组分析结果:\n\n"
    sorted_groups = sorted(group_results.items(), key=lambda x: x[1]['average_score'], reverse=True)

    for i, (group_name, group_data) in enumerate(sorted_groups):
        avg_score = group_data['average_score']
        text_display += f"{i+1}. {group_name}\n"
        text_display += f"   平均分: {avg_score:.3f}\n"
        text_display += f"   变体数: {len(group_data['prompts'])}\n\n"

    plt.text(0.05, 0.95, text_display, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    plt.axis('off')

    # 保存图像
    output_path = Path(output_dir) / "01_concept_groups_overview.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"概念组分析概览已保存: {output_path}")

def visualize_group_scores_comparison(group_results, output_dir="outputs"):
    """
    可视化概念组得分比较

    Args:
        group_results: 概念组分析结果
        output_dir: 输出目录
    """
    print("生成概念组得分比较...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    plt.figure(figsize=(12, 8))
    plt.title('概念组匹配得分比较', fontsize=16, fontweight='bold')

    # 准备数据
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
    output_path = Path(output_dir) / "02_group_scores_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"概念组得分比较已保存: {output_path}")

def visualize_prompt_variants_per_group(group_results, output_dir="outputs"):
    """
    可视化每个概念组内的prompt变体得分

    Args:
        group_results: 概念组分析结果
        output_dir: 输出目录
    """
    print("生成每个概念组的prompt变体得分...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    # 获取排序后的组
    sorted_groups = sorted(group_results.items(), key=lambda x: x[1]['average_score'], reverse=True)

    # 为每个组创建单独的图表
    for i, (group_name, group_data) in enumerate(sorted_groups):
        plt.figure(figsize=(10, 6))
        plt.title(f'概念组: {group_name}', fontsize=16, fontweight='bold')

        prompts = group_data['prompts']
        scores = group_data['individual_scores']

        # 使用颜色渐变
        colors = plt.cm.viridis(np.linspace(0, 1, len(prompts)))

        # 绘制条形图
        bars = plt.bar(range(len(prompts)), scores, color=colors)
        plt.xticks(range(len(prompts)),
                  [f"变体{j+1}" for j in range(len(prompts))],
                  rotation=0)
        plt.xlabel('Prompt变体', fontsize=12)
        plt.ylabel('匹配分数', fontsize=12)
        plt.grid(axis='y', alpha=0.3)

        # 添加数值标签
        for bar, score, prompt in zip(bars, scores, prompts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{score:.4f}', ha='center', va='bottom', fontsize=10)

        # 添加prompt文本说明
        prompt_text = "变体详情:\n\n"
        for j, prompt in enumerate(prompts):
            prompt_text += f"变体{j+1}: {prompt}\n"

        plt.figtext(0.5, 0.01, prompt_text, ha='center', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

        # 保存图像
        output_path = Path(output_dir) / f"03_{i+1}_group_{group_name[:10]}_variants.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"概念组 '{group_name}' 的变体分析已保存: {output_path}")

def visualize_all_scores_distribution(all_individual_results, output_dir="outputs"):
    """
    可视化所有prompt的分数分布

    Args:
        all_individual_results: 所有prompt的分析结果
        output_dir: 输出目录
    """
    print("生成所有prompt的分数分布...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.title('所有Prompts分数分布', fontsize=16, fontweight='bold')

    # 提取分数
    all_scores = [result[1] for result in all_individual_results]

    # 绘制直方图
    plt.hist(all_scores, bins=15, color='orange', alpha=0.7, edgecolor='black')
    plt.xlabel('匹配分数', fontsize=12)
    plt.ylabel('频次', fontsize=12)
    plt.axvline(np.mean(all_scores), color='red', linestyle='--', linewidth=2,
               label=f'平均值: {np.mean(all_scores):.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 保存图像
    output_path = Path(output_dir) / "04_all_scores_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"所有prompt的分数分布已保存: {output_path}")

def visualize_top_matches(all_individual_results, prompt_to_group, group_results, output_dir="outputs", top_n=8):
    """
    可视化Top-N最佳匹配

    Args:
        all_individual_results: 所有prompt的分析结果
        prompt_to_group: prompt到组名的映射
        group_results: 概念组分析结果
        output_dir: 输出目录
        top_n: 显示前N个结果
    """
    print(f"生成Top-{top_n}最佳匹配...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    plt.figure(figsize=(12, 8))
    plt.title(f'Top-{top_n}最佳匹配', fontsize=16, fontweight='bold')

    # 排序并选择前N个结果
    sorted_results = sorted(all_individual_results, key=lambda x: x[1], reverse=True)
    top_results = sorted_results[:top_n]

    prompts_top = [r[0] for r in top_results]
    scores_top = [r[1] for r in top_results]

    # 获取组名和变体索引
    groups_top = []
    variants_top = []

    for prompt in prompts_top:
        group_name = prompt_to_group.get(prompt, "未知组")
        groups_top.append(group_name)

        # 查找该prompt在其组内的索引
        variant_idx = 0  # 默认为第一个变体
        for group, data in group_results.items():
            if prompt in data['prompts']:
                variant_idx = data['prompts'].index(prompt)
                break
        variants_top.append(variant_idx)

    # 使用颜色渐变
    colors = plt.cm.viridis(np.linspace(0, 1, len(prompts_top)))

    # 绘制水平条形图
    bars = plt.barh(range(len(prompts_top)), scores_top, color=colors)
    plt.yticks(range(len(prompts_top)),
              [f"{i+1}. 组: {g[:10]}..., 变体{v+1}" for i, (g, v) in enumerate(zip(groups_top, variants_top))])
    plt.xlabel('匹配分数', fontsize=12)
    plt.grid(axis='x', alpha=0.3)

    # 添加数值标签
    for i, (bar, score) in enumerate(zip(bars, scores_top)):
        plt.text(score + 0.001, i, f'{score:.4f}',
                va='center', fontsize=10, fontweight='bold')

    # 添加prompt文本说明
    prompt_text = "Top匹配详情:\n\n"
    for i, (prompt, score, group, variant) in enumerate(zip(prompts_top, scores_top, groups_top, variants_top)):
        prompt_text += f"{i+1}. [{group}] 变体{variant+1}: {prompt} ({score:.4f})\n"

    plt.figtext(0.5, 0.01, prompt_text, ha='center', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    # 保存图像
    output_path = Path(output_dir) / f"05_top_{top_n}_matches.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Top-{top_n}最佳匹配已保存: {output_path}")

def visualize_group_performance_comparison(group_results, output_dir="outputs"):
    """
    可视化概念组性能比较（平均分和标准差）

    Args:
        group_results: 概念组分析结果
        output_dir: 输出目录
    """
    print("生成概念组性能比较...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    # 只有当每组有多个prompt时才计算标准差
    has_multiple_prompts = any(len(data['individual_scores']) > 1 for data in group_results.values())

    if not has_multiple_prompts:
        print("每个组只有一个prompt，跳过性能比较图")
        return

    plt.figure(figsize=(12, 8))
    plt.title('概念组性能比较 (平均分和标准差)', fontsize=16, fontweight='bold')

    # 准备数据
    sorted_groups = sorted(group_results.items(), key=lambda x: x[1]['average_score'], reverse=True)
    group_names = [item[0] for item in sorted_groups]
    avg_scores = [item[1]['average_score'] for item in sorted_groups]
    std_scores = [np.std(item[1]['individual_scores']) for item in sorted_groups]

    x = np.arange(len(group_names))
    width = 0.35

    # 绘制平均分和标准差
    plt.bar(x - width/2, avg_scores, width, label='平均分', alpha=0.7)
    plt.bar(x + width/2, std_scores, width, label='标准差', alpha=0.7)

    plt.xlabel('概念组', fontsize=12)
    plt.ylabel('分数', fontsize=12)
    plt.xticks(x, [name[:10] + '...' if len(name) > 10 else name for name in group_names],
              rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 添加数值标签
    for i, (avg, std) in enumerate(zip(avg_scores, std_scores)):
        plt.text(i - width/2, avg + 0.01, f'{avg:.3f}', ha='center', va='bottom', fontsize=9)
        plt.text(i + width/2, std + 0.01, f'{std:.3f}', ha='center', va='bottom', fontsize=9)

    # 保存图像
    output_path = Path(output_dir) / "06_group_performance_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"概念组性能比较已保存: {output_path}")

def create_concept_group_visualizations(group_results, all_individual_results, prompt_to_group, output_dir="outputs"):
    """
    创建所有概念组相关的可视化

    Args:
        group_results: 概念组分析结果
        all_individual_results: 所有prompt的分析结果
        prompt_to_group: prompt到组名的映射
        output_dir: 输出目录
    """
    print("生成概念组分析可视化...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    # 1. 概念组分析概览
    visualize_concept_groups_overview(group_results, output_dir)

    # 2. 概念组得分比较
    visualize_group_scores_comparison(group_results, output_dir)

    # 3. 每个概念组内的prompt变体得分
    visualize_prompt_variants_per_group(group_results, output_dir)

    # 4. 所有prompt的分数分布
    visualize_all_scores_distribution(all_individual_results, output_dir)

    # 5. Top-N最佳匹配
    visualize_top_matches(all_individual_results, prompt_to_group, group_results, output_dir, top_n=8)

    # 6. 概念组性能比较（平均分和标准差）
    visualize_group_performance_comparison(group_results, output_dir)

    print("概念组分析可视化完成")

