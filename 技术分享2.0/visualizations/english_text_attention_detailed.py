"""
详细的英文文本注意力分析 - 将每个subplot分别保存为单独图片
专门解释"two dogs lying on a cushion in the sun"的注意力机制
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_individual_attention_plots(attention_data, output_dir="outputs"):
    """
    创建单独的注意力分析图片，每个subplot一个文件

    Args:
        attention_data: 注意力数据
        output_dir: 输出目录
    """
    print("生成详细的英文文本注意力分析（单独图片）...")

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

    # 过滤掉CLS和SEP token，只保留内容token
    content_start = 1  # 跳过CLS
    content_end = valid_length - 1  # 跳过SEP
    display_length = min(content_end - content_start, 9)  # 最多显示9个内容token

    # 英文内容tokens
    expected_tokens = ['two', 'dogs', 'lying', 'on', 'a', 'cushion', 'in', 'the', 'sun']
    token_labels = expected_tokens[:display_length]

    # 获取最后一层的注意力并过滤
    final_attention = text_attentions[-1][target_idx]  # [num_heads, seq_len, seq_len]
    avg_attention = final_attention.mean(dim=0).numpy()  # 平均所有头
    attention_matrix = avg_attention[content_start:content_start+display_length,
                                   content_start:content_start+display_length]

    print(f"注意力矩阵形状: {attention_matrix.shape}")
    print(f"显示tokens: {token_labels}")

    # 1. Token序列展示
    create_token_sequence_plot(token_labels, input_ids, content_start, display_length, target_text, output_dir)

    # 2. 注意力热力图
    create_attention_heatmap(attention_matrix, token_labels, target_text, output_dir)

    # 3. 关键词"dogs"的注意力分布
    create_dogs_attention_distribution(attention_matrix, token_labels, target_text, output_dir)

    # 4. 自注意力强度
    create_self_attention_plot(attention_matrix, token_labels, target_text, output_dir)

    # 5. 注意力网络图
    create_attention_network(attention_matrix, token_labels, target_text, output_dir)

    # 6. 多头注意力分析
    create_multi_head_analysis(final_attention, content_start, display_length, target_text, output_dir)

    # 7. 详细分析报告
    create_analysis_report(attention_matrix, token_labels, input_ids, content_start, target_text, output_dir)

    print("✓ 所有单独的注意力分析图片已生成完成")

def create_token_sequence_plot(token_labels, input_ids, content_start, display_length, target_text, output_dir):
    """创建Token序列展示图"""
    plt.figure(figsize=(16, 4))
    plt.suptitle(f'Token Sequence Analysis\n"{target_text}"', fontsize=16, fontweight='bold')

    colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink',
              'lightgray', 'lightcyan', 'wheat', 'lavender']

    for i, (token, color) in enumerate(zip(token_labels, colors)):
        rect = plt.Rectangle((i, 0), 1, 1, facecolor=color, edgecolor='black', linewidth=2)
        plt.gca().add_patch(rect)
        plt.text(i+0.5, 0.5, token, ha='center', va='center', fontweight='bold', fontsize=14)
        plt.text(i+0.5, -0.3, f'ID:{input_ids[content_start + i]}', ha='center', va='center', fontsize=10)

    plt.xlim(0, display_length)
    plt.ylim(-0.5, 1.2)
    plt.axis('off')

    # 添加解释
    explanation = """Token序列说明：
• 每个彩色方块代表一个内容token（已过滤CLS/SEP）
• 上方显示实际单词，下方显示token ID
• 颜色用于区分不同的token，便于在其他图中识别"""

    plt.figtext(0.5, 0.02, explanation, ha='center', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "01_token_sequence.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Token序列图已保存")

def create_attention_heatmap(attention_matrix, token_labels, target_text, output_dir):
    """创建注意力热力图"""
    plt.figure(figsize=(12, 10))
    plt.suptitle(f'Token-to-Token Attention Matrix\n"{target_text}"', fontsize=16, fontweight='bold')

    im = plt.imshow(attention_matrix, cmap='Blues', interpolation='nearest')
    plt.xlabel('Key Token (被关注的词)', fontsize=14)
    plt.ylabel('Query Token (关注其他词的词)', fontsize=14)
    plt.xticks(range(len(token_labels)), token_labels, rotation=45, ha='right')
    plt.yticks(range(len(token_labels)), token_labels)

    # 添加数值标签
    for i in range(len(token_labels)):
        for j in range(len(token_labels)):
            value = attention_matrix[i, j]
            color = 'white' if value > 0.5 else 'black'
            plt.text(j, i, f'{value:.2f}', ha='center', va='center',
                    color=color, fontsize=10, fontweight='bold')

    plt.colorbar(im, shrink=0.8, label='Attention Weight')

    # 添加解释
    explanation = """注意力热力图说明：
• 行(Y轴)：Query token - 发起注意力的词
• 列(X轴)：Key token - 被关注的词
• 颜色深浅：注意力权重大小（越蓝越强）
• 对角线：自注意力（词对自己的关注）
• 非对角线：跨词注意力（词对其他词的关注）

解读示例：
• "two"对"dogs"的注意力 = 矩阵[0,1]位置的值
• "dogs"对"cushion"的注意力 = 矩阵[1,5]位置的值"""

    plt.figtext(0.5, 0.02, explanation, ha='center', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "02_attention_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 注意力热力图已保存")

def create_dogs_attention_distribution(attention_matrix, token_labels, target_text, output_dir):
    """创建"dogs"的注意力分布图"""
    plt.figure(figsize=(14, 8))
    plt.suptitle(f'Key Token "dogs" - Attention Distribution\n"{target_text}"', fontsize=16, fontweight='bold')

    dogs_idx = 1  # "dogs"在内容token中的位置
    key_attention = attention_matrix[dogs_idx, :]

    bars = plt.bar(range(len(token_labels)), key_attention,
                  color=['red' if i == dogs_idx else 'lightblue' for i in range(len(token_labels))],
                  alpha=0.7)
    plt.xlabel('Target Token (被"dogs"关注的词)', fontsize=14)
    plt.ylabel('Attention Weight (注意力权重)', fontsize=14)
    plt.xticks(range(len(token_labels)), token_labels, rotation=45)

    # 添加数值标签
    for bar, value in zip(bars, key_attention):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.grid(True, alpha=0.3)

    # 找出dogs最关注的词
    most_attended_idx = np.argmax(key_attention)
    most_attended_word = token_labels[most_attended_idx]

    # 添加解释
    explanation = f"""Dogs注意力分布说明：
• 红色柱子：dogs对自己的注意力 = {key_attention[dogs_idx]:.3f}
• 蓝色柱子：dogs对其他词的注意力
• 最高关注：dogs最关注"{most_attended_word}" (权重={key_attention[most_attended_idx]:.3f})

为什么其他词的注意力很低？
• 注意力权重经过softmax归一化，总和=1.0
• dogs主要关注自己({key_attention[dogs_idx]:.3f})，剩余权重分配给其他词
• 这表明"dogs"是一个语义上相对独立的核心概念"""

    plt.figtext(0.5, 0.02, explanation, ha='center', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "03_dogs_attention_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Dogs注意力分布图已保存")

def create_self_attention_plot(attention_matrix, token_labels, target_text, output_dir):
    """创建自注意力强度图"""
    plt.figure(figsize=(14, 8))
    plt.suptitle(f'Self-Attention Strength\n"{target_text}"', fontsize=16, fontweight='bold')

    self_attention = np.diag(attention_matrix)
    bars = plt.bar(range(len(token_labels)), self_attention, color='orange', alpha=0.7)
    plt.xlabel('Token', fontsize=14)
    plt.ylabel('Self-Attention Weight (自注意力权重)', fontsize=14)
    plt.xticks(range(len(token_labels)), token_labels, rotation=45)

    # 添加数值标签
    for bar, value in zip(bars, self_attention):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.grid(True, alpha=0.3)

    # 找出自注意力最强的词
    strongest_self_idx = np.argmax(self_attention)
    strongest_word = token_labels[strongest_self_idx]

    # 添加解释
    explanation = f"""自注意力强度说明：
• 自注意力 = 每个词对自己的关注程度
• 最强自注意力："{strongest_word}" = {self_attention[strongest_self_idx]:.3f}

自注意力的意义：
• 高自注意力：词汇语义独立性强，不太依赖上下文
• 低自注意力：词汇更依赖其他词来确定含义
• 内容词(如dogs, cushion)通常比功能词(如a, the)有更高自注意力

为什么"two"分值比"dogs"高？
• "two"的自注意力 = {self_attention[0]:.3f}
• "dogs"的自注意力 = {self_attention[1]:.3f}
• 数量词"two"在这个语境中语义相对独立，不需要太多上下文信息"""

    plt.figtext(0.5, 0.02, explanation, ha='center', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "04_self_attention_strength.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 自注意力强度图已保存")

def create_attention_network(attention_matrix, token_labels, target_text, output_dir):
    """创建注意力网络图"""
    plt.figure(figsize=(12, 12))
    plt.suptitle(f'Attention Network\n"{target_text}"', fontsize=16, fontweight='bold')

    threshold = 0.15  # 只显示注意力权重大于阈值的连接
    dogs_idx = 1

    # 绘制token节点
    positions = {}
    for i in range(len(token_labels)):
        angle = 2 * np.pi * i / len(token_labels)
        x = np.cos(angle)
        y = np.sin(angle)
        positions[i] = (x, y)

        # 节点大小根据总注意力权重（接收到的注意力）
        incoming_attention = attention_matrix[:, i].sum()  # 列和：其他词对这个词的总注意力
        node_size = incoming_attention * 1000

        # 节点颜色：dogs用红色，其他用蓝色
        color = 'red' if i == dogs_idx else 'lightblue'
        plt.scatter(x, y, s=node_size, alpha=0.8, c=color, edgecolors='black', linewidth=2)
        plt.text(x*1.3, y*1.3, token_labels[i], ha='center', va='center',
                fontsize=12, fontweight='bold')

    # 绘制注意力连接（有向边）
    connection_count = 0
    for i in range(len(token_labels)):
        for j in range(len(token_labels)):
            if i != j and attention_matrix[i, j] > threshold:
                x1, y1 = positions[i]  # 起点（query）
                x2, y2 = positions[j]  # 终点（key）

                # 连线透明度和粗细根据注意力权重
                alpha = attention_matrix[i, j]
                linewidth = alpha * 5

                # 绘制箭头表示方向
                plt.annotate('', xy=(x2*0.9, y2*0.9), xytext=(x1*0.9, y1*0.9),
                           arrowprops=dict(arrowstyle='->', color='gray',
                                         alpha=alpha, lw=linewidth))
                connection_count += 1

    plt.xlim(-1.8, 1.8)
    plt.ylim(-1.8, 1.8)
    plt.axis('off')

    # 添加解释
    explanation = f"""注意力网络图说明：
• 节点：每个token，大小表示接收到的总注意力
• 红色节点：dogs（关键词）
• 蓝色节点：其他词
• 箭头：注意力方向（A→B表示A关注B）
• 线条粗细/透明度：注意力权重大小

网络特点：
• 显示了{connection_count}个强连接（权重>{threshold}）
• 节点大小反映词汇的"重要性"（被关注程度）
• 箭头方向显示注意力流向

为什么只看到不同颜色的点？
• 大部分注意力权重<{threshold}，被过滤掉了
• 这表明词汇间的强关联相对较少
• 每个词主要关注自己（自注意力占主导）"""

    plt.figtext(0.5, 0.02, explanation, ha='center', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "05_attention_network.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 注意力网络图已保存")

def create_multi_head_analysis(final_attention, content_start, display_length, target_text, output_dir):
    """创建多头注意力分析图"""
    plt.figure(figsize=(14, 8))
    plt.suptitle(f'Multi-Head Attention Patterns\n"{target_text}"', fontsize=16, fontweight='bold')

    num_heads = min(8, final_attention.shape[0])  # 显示前8个头
    head_patterns = []
    head_names = []

    for head in range(num_heads):
        head_att = final_attention[head, content_start:content_start+display_length,
                                  content_start:content_start+display_length].numpy()

        # 计算不同类型的注意力强度
        diag_strength = np.diag(head_att).mean()  # 自注意力平均强度
        off_diag_strength = (head_att.sum() - np.diag(head_att).sum()) / (display_length*display_length - display_length)  # 跨词注意力平均强度

        head_patterns.append([diag_strength, off_diag_strength])
        head_names.append(f'Head{head+1}')

    head_patterns = np.array(head_patterns)
    x = np.arange(num_heads)
    width = 0.35

    bars1 = plt.bar(x - width/2, head_patterns[:, 0], width, label='Self-Attention (自注意力)',
                   alpha=0.8, color='orange')
    bars2 = plt.bar(x + width/2, head_patterns[:, 1], width, label='Cross-Attention (跨词注意力)',
                   alpha=0.8, color='lightblue')

    plt.xlabel('Attention Head (注意力头)', fontsize=14)
    plt.ylabel('Attention Strength (注意力强度)', fontsize=14)
    plt.xticks(x, head_names)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    # 找出特化模式
    self_attention_head = np.argmax(head_patterns[:, 0])
    cross_attention_head = np.argmax(head_patterns[:, 1])

    # 添加解释
    explanation = f"""多头注意力模式说明：
• CLIP文本编码器有{final_attention.shape[0]}个注意力头，这里显示前{num_heads}个
• 每个头学习不同的注意力模式

头部特化：
• Head{self_attention_head+1}：最强自注意力 ({head_patterns[self_attention_head, 0]:.3f})
• Head{cross_attention_head+1}：最强跨词注意力 ({head_patterns[cross_attention_head, 1]:.3f})

不同头的作用：
• 自注意力强的头：关注词汇本身的语义
• 跨词注意力强的头：关注词汇间的关系
• 多头机制让模型同时捕获多种语言模式"""

    plt.figtext(0.5, 0.02, explanation, ha='center', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "06_multi_head_attention.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 多头注意力分析图已保存")

def create_analysis_report(attention_matrix, token_labels, input_ids, content_start, target_text, output_dir):
    """创建详细分析报告"""
    plt.figure(figsize=(14, 10))
    plt.suptitle(f'Detailed Analysis Report\n"{target_text}"', fontsize=16, fontweight='bold')
    plt.axis('off')

    # 计算各种统计信息
    dogs_idx = 1
    self_attention = np.diag(attention_matrix)
    most_attended_by_dogs = np.argmax(attention_matrix[dogs_idx, :])
    strongest_self_attention = np.argmax(self_attention)
    max_cross_attention = attention_matrix.max()
    avg_attention = attention_matrix.mean()

    # 生成详细报告
    report_text = f"""COMPREHENSIVE ATTENTION ANALYSIS REPORT

目标文本: "{target_text}"
分析的Token数量: {len(token_labels)} 个内容tokens（已过滤CLS/SEP）

=== TOKEN分解详情 ===
"""

    for i, (token, token_id) in enumerate(zip(token_labels, input_ids[content_start:content_start+len(token_labels)])):
        self_att = self_attention[i]
        incoming_att = attention_matrix[:, i].sum()  # 被其他词关注的总量
        outgoing_att = attention_matrix[i, :].sum()  # 关注其他词的总量
        report_text += f"[{i}] {token:8} (ID: {token_id:5}) | 自注意力: {self_att:.3f} | 被关注: {incoming_att:.3f} | 关注他人: {outgoing_att:.3f}\n"

    report_text += f"""
=== 关键发现 ===
• 最强自注意力: "{token_labels[strongest_self_attention]}" = {self_attention[strongest_self_attention]:.3f}
• 最强跨词注意力: {max_cross_attention:.3f}
• 平均注意力权重: {avg_attention:.3f}
• "dogs"最关注: "{token_labels[most_attended_by_dogs]}"

=== 问题解答 ===

Q1: 为什么"two"的自注意力比"dogs"高？
A1: • "two"自注意力 = {self_attention[0]:.3f}，"dogs"自注意力 = {self_attention[1]:.3f}
    • 数量词"two"在语义上相对独立，不需要太多上下文
    • "dogs"作为核心名词，需要更多与其他词的交互来确定完整语义

Q2: 为什么Key Attention显示为"dogs"？
A2: • 这里分析的是"dogs"作为Query时对其他词的注意力分布
    • 即"dogs"这个词关注句子中其他哪些词
    • 这与自注意力强度是不同的概念

Q3: 为什么Attention Distribution中其他词的值很低？
A3: • 注意力权重经过softmax归一化，所有权重总和 = 1.0
    • "dogs"主要关注自己({attention_matrix[dogs_idx, dogs_idx]:.3f})
    • 剩余权重({1-attention_matrix[dogs_idx, dogs_idx]:.3f})分配给其他{len(token_labels)-1}个词
    • 平均每个其他词获得约{(1-attention_matrix[dogs_idx, dogs_idx])/(len(token_labels)-1):.3f}的注意力

Q4: Self-Attention Strength指什么？
A4: • 每个词对自己的注意力权重（注意力矩阵的对角线元素）
    • 反映词汇的语义独立性和重要性
    • 高自注意力 = 词汇语义相对独立，不太依赖上下文

Q5: Multi-Head Attention Pattern的几个Head分别是什么？
A5: • CLIP文本编码器有8个注意力头，每个头学习不同模式
    • 有些头专注自注意力（词汇本身语义）
    • 有些头专注跨词注意力（词汇间关系）
    • 多头机制让模型同时捕获多种语言理解模式
"""

    plt.text(0.05, 0.95, report_text, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "07_detailed_analysis_report.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 详细分析报告已保存")

def run_detailed_english_attention_demo():
    """
    运行详细的英文文本注意力分析演示
    """
    print("=" * 60)
    print("详细英文文本注意力机制分析")
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
    texts = [target_text]
    print(f"✓ 目标文本: {target_text}")

    # 4. 提取注意力数据
    print("提取注意力权重...")
    attention_data = extract_attention_weights(model, tokenizer, image, texts, device)

    if not attention_data:
        print("❌ 注意力数据提取失败")
        return

    print("✓ 注意力数据提取成功")

    # 5. 生成详细可视化
    output_dir = Path(__file__).parent.parent / "outputs" / "detailed_english_attention"
    print(f"生成详细可视化到: {output_dir}")

    create_individual_attention_plots(attention_data, str(output_dir))

    print("=" * 60)
    print("✅ 详细英文文本注意力分析完成！")
    print(f"📁 结果保存在: {output_dir}")
    print("📄 生成的文件:")
    for i, filename in enumerate([
        "01_token_sequence.png",
        "02_attention_heatmap.png",
        "03_dogs_attention_distribution.png",
        "04_self_attention_strength.png",
        "05_attention_network.png",
        "06_multi_head_attention.png",
        "07_detailed_analysis_report.png"
    ], 1):
        print(f"   {i}. {filename}")
    print("=" * 60)

    return output_dir

if __name__ == "__main__":
    # 直接运行详细演示
    run_detailed_english_attention_demo()

