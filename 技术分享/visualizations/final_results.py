import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_final_results_visualization(original_image, group_results, output_dir="outputs"):
    """创建最终结果可视化"""
    print("生成最终结果可视化...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('CLIP多Prompt分析最终结果', fontsize=16, fontweight='bold')

    # 1. 原始图像
    ax1 = axes[0, 0]
    ax1.imshow(original_image)
    ax1.set_title('输入图像', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # 2. 概念组排序结果
    ax2 = axes[0, 1]
    sorted_groups = sorted(group_results.items(), key=lambda x: x[1]['average_score'], reverse=True)
    group_names = [item[0] for item in sorted_groups]
    avg_scores = [item[1]['average_score'] for item in sorted_groups]

    colors = plt.cm.RdYlGn(np.linspace(0.3, 1, len(group_names)))

    bars = ax2.barh(range(len(group_names)), avg_scores, color=colors)
    ax2.set_yticks(range(len(group_names)))
    ax2.set_yticklabels([f"{i + 1}. {name[:20]}..." if len(name) > 20 else f"{i + 1}. {name}"
                         for i, name in enumerate(group_names)], fontsize=10)
    ax2.set_xlabel('平均匹配分数', fontsize=12)
    ax2.set_title('概念组匹配排序', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    for i, (bar, score) in enumerate(zip(bars, avg_scores)):
        ax2.text(score + 0.001, i, f'{score:.4f}',
                 va='center', fontsize=9, fontweight='bold')

    # 3. Top-3概念组详细对比
    ax3 = axes[1, 0]
    top3_groups = sorted_groups[:3]

    x = np.arange(len(top3_groups))
    width = 0.15

    for i in range(4):  # 每组最多4个变体
        variant_scores = []
        for group_name, group_data in top3_groups:
            if i < len(group_data['individual_scores']):
                variant_scores.append(group_data['individual_scores'][i])
            else:
                variant_scores.append(0)

        ax3.bar(x + i * width, variant_scores, width,
                label=f'变体{i + 1}', alpha=0.8)

    ax3.set_title('Top-3概念组变体对比', fontsize=14, fontweight='bold')
    ax3.set_xlabel('概念组')
    ax3.set_ylabel('匹配分数')
    ax3.set_xticks(x + width * 1.5)
    ax3.set_xticklabels([f"{name[:15]}..." if len(name) > 15 else name
                         for name, _ in top3_groups])
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # 4. 结果总结表格
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')

    table_data = []
    for i, (group_name, group_data) in enumerate(sorted_groups):
        rank = i + 1
        avg_score = group_data['average_score']
        std_score = np.std(group_data['individual_scores'])
        max_score = max(group_data['individual_scores'])

        short_name = group_name[:25] + "..." if len(group_name) > 25 else group_name
        table_data.append([f"{rank}", short_name, f"{avg_score:.4f}",
                           f"{std_score:.4f}", f"{max_score:.4f}"])

    table = ax4.table(cellText=table_data,
                      colLabels=['排名', '概念描述', '平均分', '标准差', '最高分'],
                      cellLoc='left',
                      loc='center',
                      colWidths=[0.08, 0.52, 0.12, 0.12, 0.12])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax4.set_title('详细结果总结', fontsize=14, fontweight='bold')

    # 设置表格样式
    for i in range(len(table_data) + 1):
        for j in range(5):
            cell = table[(i, j)]
            if i == 0:  # 表头
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            elif i == 1:  # 第一名
                cell.set_facecolor('#FFD700')
            elif i == 2:  # 第二名
                cell.set_facecolor('#C0C0C0')
            elif i == 3:  # 第三名
                cell.set_facecolor('#CD7F32')

    plt.tight_layout()
    output_path = Path(output_dir) / "03_final_results.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"最终结果可视化已保存: {output_path}")
    plt.close()

