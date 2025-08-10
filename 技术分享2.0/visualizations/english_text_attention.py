"""
专门针对英文文本"two dogs lying on a cushion in the sun"的注意力机制可视化
使用真实的CLIP模型数据，不做过多容错处理
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def visualize_english_text_attention(attention_data, output_dir="outputs"):
    """
    专门可视化英文文本"two dogs lying on a cushion in the sun"的注意力机制

    Args:
        attention_data: 注意力数据，必须包含真实的text_attentions
        output_dir: 输出目录
    """
    print("生成英文文本注意力机制可视化...")

    Path(output_dir).mkdir(exist_ok=True)

    # 目标文本
    target_text = "two dogs lying on a cushion in the sun"

    # 寻找目标文本
    target_idx = 0
    for i, text in enumerate(attention_data['texts']):
        if target_text.lower() in text.lower():
            target_idx = i
            break

    print(f"找到目标文本: {attention_data['texts'][target_idx]}")

    # 获取真实数据
    text_attentions = attention_data['text_attentions']
    text_inputs = attention_data['text_inputs']

    input_ids = text_inputs['input_ids'][target_idx].numpy()
    attention_mask = text_inputs['attention_mask'][target_idx].numpy()
    valid_length = int(attention_mask.sum())

    print(f"Token序列长度: {valid_length}")
    print(f"Input IDs: {input_ids[:valid_length]}")

    # 过滤掉CLS和SEP token，只保留内容token
    content_start = 1  # 跳过CLS
    content_end = valid_length - 1  # 跳过SEP
    display_length = min(content_end - content_start, 9)  # 最多显示9个内容token

    # 英文内容tokens（预期的分词结果）
    expected_tokens = ['two', 'dogs', 'lying', 'on', 'a', 'cushion', 'in', 'the', 'sun']

    # 创建token标签
    token_labels = expected_tokens[:display_length]

    # 获取最后一层的注意力并过滤
    final_attention = text_attentions[-1][target_idx]  # [num_heads, seq_len, seq_len]
    avg_attention = final_attention.mean(dim=0).numpy()  # 平均所有头
    attention_matrix = avg_attention[content_start:content_start+display_length,
                                   content_start:content_start+display_length]

    print(f"注意力矩阵形状: {attention_matrix.shape}")
    print(f"显示tokens: {token_labels}")

    # 创建大图，包含多个子图
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'English Text Attention Analysis\n"{target_text}"',
                 fontsize=18, fontweight='bold')

    # 1. Token序列展示 (顶部)
    ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=4)
    ax1.set_title('Content Token Sequence (CLS/SEP Filtered)', fontsize=14, fontweight='bold')

    # 绘制token序列
    colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink',
              'lightgray', 'lightcyan', 'wheat', 'lavender']

    for i, (token, color) in enumerate(zip(token_labels, colors)):
        rect = plt.Rectangle((i, 0), 1, 1, facecolor=color, edgecolor='black', linewidth=2)
        ax1.add_patch(rect)
        ax1.text(i+0.5, 0.5, token, ha='center', va='center', fontweight='bold', fontsize=12)
        ax1.text(i+0.5, -0.3, f'ID:{input_ids[content_start + i]}', ha='center', va='center', fontsize=8)

    ax1.set_xlim(0, display_length)
    ax1.set_ylim(-0.5, 1.2)
    ax1.set_aspect('equal')
    ax1.axis('off')

    # 2. 注意力热力图 (左上)
    ax2 = plt.subplot2grid((4, 4), (1, 0), colspan=2, rowspan=2)

    im = ax2.imshow(attention_matrix, cmap='Blues', interpolation='nearest')
    ax2.set_title('Token-to-Token Attention Matrix', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Key Token', fontsize=12)
    ax2.set_ylabel('Query Token', fontsize=12)
    ax2.set_xticks(range(display_length))
    ax2.set_yticks(range(display_length))
    ax2.set_xticklabels(token_labels, rotation=45, ha='right')
    ax2.set_yticklabels(token_labels)

    # 添加数值标签
    for i in range(display_length):
        for j in range(display_length):
            value = attention_matrix[i, j]
            color = 'white' if value > 0.5 else 'black'
            ax2.text(j, i, f'{value:.2f}', ha='center', va='center',
                    color=color, fontsize=9, fontweight='bold')

    plt.colorbar(im, ax=ax2, shrink=0.8)

    # 3. 关键词"dogs"的注意力分布 (右上)
    ax3 = plt.subplot2grid((4, 4), (1, 2), colspan=2)

    dogs_idx = 1  # "dogs"在内容token中的位置
    key_attention = attention_matrix[dogs_idx, :]

    bars = ax3.bar(range(display_length), key_attention,
                  color=['red' if i == dogs_idx else 'lightblue' for i in range(display_length)],
                  alpha=0.7)
    ax3.set_title('Key Token "dogs" - Attention Distribution', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Target Token')
    ax3.set_ylabel('Attention Weight')
    ax3.set_xticks(range(display_length))
    ax3.set_xticklabels(token_labels, rotation=45)

    # 添加数值标签
    for bar, value in zip(bars, key_attention):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.2f}', ha='center', va='bottom', fontsize=9)

    # 4. 自注意力强度 (左下)
    ax4 = plt.subplot2grid((4, 4), (2, 2))

    self_attention = np.diag(attention_matrix)
    bars = ax4.bar(range(display_length), self_attention, color='orange', alpha=0.7)
    ax4.set_title('Self-Attention Strength', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Token')
    ax4.set_ylabel('Self-Attention')
    ax4.set_xticks(range(display_length))
    ax4.set_xticklabels(token_labels, rotation=45)

    # 5. 注意力网络图 (右下)
    ax5 = plt.subplot2grid((4, 4), (2, 3))

    # 创建圆形布局的注意力网络
    threshold = 0.15  # 只显示注意力权重大于阈值的连接

    positions = {}
    for i in range(display_length):
        angle = 2 * np.pi * i / display_length
        x = np.cos(angle)
        y = np.sin(angle)
        positions[i] = (x, y)

        # 节点大小根据总注意力权重
        node_size = attention_matrix[i, :].sum() * 800
        color = 'red' if i == dogs_idx else 'lightblue'
        ax5.scatter(x, y, s=node_size, alpha=0.7, c=color)
        ax5.text(x*1.3, y*1.3, token_labels[i], ha='center', va='center', fontsize=10, fontweight='bold')

    # 绘制注意力连接
    for i in range(display_length):
        for j in range(display_length):
            if i != j and attention_matrix[i, j] > threshold:
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                alpha = attention_matrix[i, j]
                ax5.plot([x1, x2], [y1, y2], 'gray', alpha=alpha, linewidth=alpha*4)

    ax5.set_title('Attention Network', fontsize=12, fontweight='bold')
    ax5.set_xlim(-1.8, 1.8)
    ax5.set_ylim(-1.8, 1.8)
    ax5.axis('off')

    # 6. 多头注意力分析 (底部左)
    ax6 = plt.subplot2grid((4, 4), (3, 0), colspan=2)

    num_heads = min(4, final_attention.shape[0])
    head_patterns = []

    for head in range(num_heads):
        head_att = final_attention[head, content_start:content_start+display_length,
                                  content_start:content_start+display_length].numpy()
        diag_strength = np.diag(head_att).mean()
        off_diag_strength = (head_att.sum() - np.diag(head_att).sum()) / (display_length*display_length - display_length)
        head_patterns.append([diag_strength, off_diag_strength])

    head_patterns = np.array(head_patterns)
    x = np.arange(num_heads)
    width = 0.35

    ax6.bar(x - width/2, head_patterns[:, 0], width, label='Self-Attention', alpha=0.8, color='orange')
    ax6.bar(x + width/2, head_patterns[:, 1], width, label='Cross-Attention', alpha=0.8, color='lightblue')
    ax6.set_title('Multi-Head Attention Patterns', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Attention Head')
    ax6.set_ylabel('Attention Strength')
    ax6.set_xticks(x)
    ax6.set_xticklabels([f'Head{i+1}' for i in range(num_heads)])
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 7. 详细分析报告 (底部右)
    ax7 = plt.subplot2grid((4, 4), (3, 2), colspan=2)
    ax7.axis('off')

    # 生成分析报告
    most_attended_by_dogs = np.argmax(attention_matrix[dogs_idx, :])
    strongest_self_attention = np.argmax(self_attention)
    max_cross_attention = attention_matrix.max()
    avg_attention = attention_matrix.mean()

    report_text = f"""ATTENTION ANALYSIS REPORT

Text: "{target_text}"
Tokens Analyzed: {display_length} content tokens

KEY FINDINGS:
• "dogs" pays most attention to: "{token_labels[most_attended_by_dogs]}"
• Strongest self-attention: "{token_labels[strongest_self_attention]}"
• Maximum cross-attention: {max_cross_attention:.3f}
• Average attention weight: {avg_attention:.3f}

TOKEN BREAKDOWN:
"""

    for i, (token, token_id) in enumerate(zip(token_labels, input_ids[content_start:content_start+display_length])):
        report_text += f"[{i}] {token} (ID: {token_id})\n"

    ax7.text(0.05, 0.95, report_text, transform=ax7.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()

    # 保存图像
    output_path = Path(output_dir) / "english_text_attention_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ 英文文本注意力分析完成")
    print(f"✓ 分析了 {display_length} 个内容tokens（已过滤CLS/SEP）")
    print(f"✓ 可视化已保存: {output_path}")

    return output_path

def run_english_text_attention_demo():
    """
    自闭环运行英文文本注意力分析演示
    直接加载模型，处理"two dogs lying on a cushion in the sun"这句话
    """
    print("=" * 60)
    print("英文文本注意力机制可视化演示")
    print("目标文本: 'two dogs lying on a cushion in the sun'")
    print("=" * 60)

    # 导入必要的模块
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))

    from utils.model_loader import load_local_clip_model
    from utils.attention_extractor import extract_attention_weights
    from PIL import Image
    import torch

    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 1. 加载模型
    print("加载CLIP模型...")
    model, tokenizer, model_loaded = load_local_clip_model()
    if not model_loaded:
        print("❌ 无法加载模型")
        return

    model = model.to(device).eval()
    print("✓ 模型加载成功")

    # 2. 加载图像
    img_path = Path(__file__).parent.parent / "dogs_sun_patio.jpeg"
    if not img_path.exists():
        print(f"❌ 图像文件不存在: {img_path}")
        return

    image = Image.open(img_path).convert('RGB')
    print("✓ 图像加载成功")

    # 3. 准备文本
    target_text = "two dogs lying on a cushion in the sun"
    texts = [target_text]  # 只分析这一句话
    print(f"✓ 目标文本: {target_text}")

    # 4. 提取注意力数据
    print("提取注意力权重...")
    attention_data = extract_attention_weights(model, tokenizer, image, texts, device)

    if not attention_data:
        print("❌ 注意力数据提取失败")
        return

    print("✓ 注意力数据提取成功")
    print(f"  - 文本注意力层数: {len(attention_data.get('text_attentions', []))}")
    print(f"  - 文本长度: {len(attention_data.get('texts', []))}")

    # 5. 生成可视化
    output_dir = Path(__file__).parent.parent / "outputs" / "english_text_attention"
    print(f"生成可视化到: {output_dir}")

    result_path = visualize_english_text_attention(attention_data, str(output_dir))

    print("=" * 60)
    print("✅ 英文文本注意力分析完成！")
    print(f"📁 结果保存在: {result_path}")
    print("=" * 60)

    return result_path

if __name__ == "__main__":
    # 直接运行演示
    run_english_text_attention_demo()

