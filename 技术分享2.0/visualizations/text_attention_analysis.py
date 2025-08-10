from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def visualize_text_inputs(attention_data, output_dir="outputs"):
    """
    可视化具体文本的Token分解过程

    Args:
        attention_data: 注意力数据
        output_dir: 输出目录
    """
    print("生成具体文本Token分解可视化...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle('具体文本Token分解分析', fontsize=16, fontweight='bold')

    if attention_data and attention_data.get('texts') and attention_data.get('text_inputs'):
        texts = attention_data['texts']
        text_inputs = attention_data['text_inputs']

        # 选择第一个文本进行详细分析，优先选择英文文本
        target_text = texts[0]
        # 如果有英文文本，优先使用
        for text in texts:
            if any(word in text.lower() for word in ['two', 'dogs', 'lying', 'cushion', 'sun']):
                target_text = text
                break
        input_ids = text_inputs['input_ids'][0].numpy()
        attention_mask = text_inputs['attention_mask'][0].numpy()
        valid_length = int(attention_mask.sum())

        # 上半部分：原始文本和Token分解
        axes[0].set_title(f'文本Token分解: "{target_text}"', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # 尝试重建token对应的文本（简化版本）
        token_info = f"原始文本: {target_text}\n\n"
        token_info += f"Token序列 (共{valid_length}个):\n"

        # 显示token ID和位置
        token_positions = []
        for i in range(min(valid_length, 15)):  # 显示前15个token
            token_id = input_ids[i]
            if i == 0:
                token_info += f"[{i}] {token_id} (CLS)\n"
            elif i == valid_length - 1:
                token_info += f"[{i}] {token_id} (SEP)\n"
            else:
                # 根据token ID推测可能的内容
                if token_id < 1000:
                    token_info += f"[{i}] {token_id} (特殊符号)\n"
                else:
                    token_info += f"[{i}] {token_id} (文本片段)\n"
            token_positions.append(i)

        if valid_length > 15:
            token_info += f"... 还有 {valid_length - 15} 个tokens\n"

        axes[0].text(0.05, 0.95, token_info, transform=axes[0].transAxes, fontsize=11,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

        # 下半部分：Token序列可视化
        axes[1].set_title('Token序列结构', fontsize=14, fontweight='bold')

        # 创建token序列的可视化
        display_length = min(valid_length, 20)
        token_values = input_ids[:display_length]

        # 绘制token条形图
        colors = []
        labels = []
        for i, token_id in enumerate(token_values):
            if i == 0:
                colors.append('red')
                labels.append(f'CLS\n{token_id}')
            elif i == valid_length - 1 and i < display_length:
                colors.append('orange')
                labels.append(f'SEP\n{token_id}')
            else:
                colors.append('lightblue')
                labels.append(f'T{i}\n{token_id}')

        bars = axes[1].bar(range(display_length), token_values, color=colors, alpha=0.7)
        axes[1].set_xlabel('Token位置', fontsize=12)
        axes[1].set_ylabel('Token ID', fontsize=12)
        axes[1].set_xticks(range(display_length))
        axes[1].set_xticklabels([f'{i}' for i in range(display_length)], rotation=0)

        # 添加token标签
        for i, (bar, label) in enumerate(zip(bars, labels)):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + max(token_values) * 0.01,
                        label, ha='center', va='bottom', fontsize=9, fontweight='bold')

        # 添加说明
        explanation = f"分析文本: {target_text}\n"
        explanation += f"总Token数: {valid_length}, 显示前{display_length}个\n"
        explanation += "红色=CLS token, 橙色=SEP token, 蓝色=文本tokens"

        axes[1].text(0.02, 0.98, explanation, transform=axes[1].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

        print(f"✓ 显示文本 '{target_text}' 的Token分解，共{valid_length}个tokens")
    else:
        for ax in axes:
            ax.text(0.5, 0.5, '需要完整的注意力数据\n包括texts和text_inputs',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.axis('off')
        print("⚠️ 无法显示文本Token分解 - 数据缺失")

    plt.tight_layout()

    # 保存图像
    output_path = Path(output_dir) / "07_text_inputs.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"文本Token分解可视化已保存: {output_path}")

def visualize_token_length_distribution(attention_data, output_dir="outputs"):
    """
    可视化Token长度分布

    Args:
        attention_data: 注意力数据
        output_dir: 输出目录
    """
    print("生成Token长度分布可视化...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.title('Token长度分布', fontsize=16, fontweight='bold')

    if attention_data and attention_data.get('text_inputs') and 'attention_mask' in attention_data['text_inputs']:
        text_inputs = attention_data['text_inputs']
        token_lengths = text_inputs['attention_mask'].sum(dim=1).numpy()

        # 绘制条形图
        bars = plt.bar(range(len(token_lengths)), token_lengths, color='skyblue')
        plt.xlabel('文本索引', fontsize=12)
        plt.ylabel('Token数量', fontsize=12)

        # 添加数值标签
        for i, length in enumerate(token_lengths):
            plt.text(i, length + 0.5, str(int(length)), ha='center', va='bottom')

        # 添加文本标签
        if attention_data.get('texts'):
            plt.xticks(range(len(token_lengths)),
                      [f"T{i+1}" for i in range(len(token_lengths))],
                      rotation=0)

            # 添加文本说明
            text_info = "文本详情:\n\n"
            for i, text in enumerate(attention_data['texts']):
                text_info += f"T{i+1}: {text[:30]}{'...' if len(text) > 30 else ''}\n"

            plt.figtext(0.5, 0.01, text_info, ha='center', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

        print(f"✓ 显示token长度分布: {token_lengths}")
    else:
        plt.text(0.5, 0.5, '需要text_inputs数据\n(tokenization结果)', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        print("⚠️ 无法显示token长度 - text_inputs缺失")

    # 保存图像
    output_path = Path(output_dir) / "08_token_length_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Token长度分布可视化已保存: {output_path}")


def visualize_attention_layer(attention_data, layer_idx, layer_name, output_dir="outputs", file_prefix="09"):
    """
    可视化特定层的文本注意力模式

    Args:
        attention_data: 注意力数据
        layer_idx: 层索引
        layer_name: 层名称
        output_dir: 输出目录
        file_prefix: 文件前缀
    """
    print(f"生成{layer_name}文本注意力可视化...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{layer_name}文本注意力分析', fontsize=16, fontweight='bold')

    if attention_data and attention_data.get('text_attentions') and len(attention_data['text_attentions']) > abs(layer_idx):
        text_attentions = attention_data['text_attentions']
        texts = attention_data.get('texts', [])

        # 获取注意力矩阵 (batch_size, num_heads, seq_len, seq_len)
        if layer_idx == -1:
            layer_idx = len(text_attentions) - 1

        attention_matrix = text_attentions[layer_idx][0]  # 取第一个batch
        num_heads, seq_len, _ = attention_matrix.shape

        # 1. 平均注意力模式 (左上)
        avg_attention = attention_matrix.mean(dim=0).numpy()
        max_display = min(seq_len, 25)  # 最多显示25x25

        im1 = axes[0, 0].imshow(avg_attention[:max_display, :max_display],
                               cmap='Blues', interpolation='nearest')
        axes[0, 0].set_title('平均注意力模式', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Key位置')
        axes[0, 0].set_ylabel('Query位置')
        plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)

        # 2. 多头注意力对比 (右上)
        head_indices = [0, num_heads//4, num_heads//2, num_heads-1] if num_heads >= 4 else [0]
        head_colors = ['red', 'blue', 'green', 'orange']

        axes[0, 1].set_title('多头注意力对比', fontsize=12, fontweight='bold')
        for i, head_idx in enumerate(head_indices[:4]):
            if head_idx < num_heads:
                # 计算每个头的注意力强度（对角线附近的平均值）
                head_attention = attention_matrix[head_idx].numpy()
                diagonal_strength = np.mean([head_attention[j, j] for j in range(min(seq_len, 20))])
                off_diagonal_strength = np.mean(head_attention[:20, :20]) - diagonal_strength

                axes[0, 1].bar([f'Head{head_idx}'], [diagonal_strength],
                              color=head_colors[i], alpha=0.7, label=f'Head {head_idx}')

        axes[0, 1].set_ylabel('注意力强度')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 注意力分布统计 (左下)
        attention_values = avg_attention.flatten()
        axes[1, 0].hist(attention_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_title('注意力值分布', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('注意力值')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].axvline(np.mean(attention_values), color='red', linestyle='--',
                          label=f'均值: {np.mean(attention_values):.3f}')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Token级别注意力分析 (右下)
        if texts and len(texts) > 0:
            # 分析第一个文本的token注意力
            text_inputs = attention_data.get('text_inputs', {})
            if 'attention_mask' in text_inputs:
                attention_mask = text_inputs['attention_mask'][0].numpy()
                valid_length = int(attention_mask.sum())

                # 计算每个token接收到的总注意力
                token_attention_received = avg_attention[:valid_length, :valid_length].sum(axis=0)
                # 计算每个token给出的总注意力
                token_attention_given = avg_attention[:valid_length, :valid_length].sum(axis=1)

                x_pos = range(min(valid_length, 15))  # 最多显示15个token
                width = 0.35

                axes[1, 1].bar([x - width/2 for x in x_pos],
                              token_attention_received[:len(x_pos)], width,
                              label='接收注意力', alpha=0.7, color='lightcoral')
                axes[1, 1].bar([x + width/2 for x in x_pos],
                              token_attention_given[:len(x_pos)], width,
                              label='给出注意力', alpha=0.7, color='lightblue')

                axes[1, 1].set_title('Token注意力分析', fontsize=12, fontweight='bold')
                axes[1, 1].set_xlabel('Token位置')
                axes[1, 1].set_ylabel('注意力强度')
                axes[1, 1].set_xticks(x_pos)
                axes[1, 1].set_xticklabels([f'T{i}' for i in x_pos])
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'Token分析需要\nattention_mask数据',
                               ha='center', va='center', transform=axes[1, 1].transAxes)
        else:
            axes[1, 1].text(0.5, 0.5, 'Token分析需要\n文本数据',
                           ha='center', va='center', transform=axes[1, 1].transAxes)

        # 添加文本信息
        if texts:
            text_info = f"分析文本: {texts[0][:50]}{'...' if len(texts[0]) > 50 else ''}\n"
            text_info += f"层索引: {layer_idx}, 注意力头数: {num_heads}, 序列长度: {seq_len}"
            fig.text(0.5, 0.02, text_info, ha='center', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

        print(f"✓ 显示{layer_name}文本注意力 - {num_heads}头, 序列长度{seq_len}")
    else:
        for ax in axes.flat:
            ax.text(0.5, 0.5, f'需要text_attentions数据\n或层数不足',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.axis('off')
        print(f"⚠️ 无法显示{layer_name}文本注意力 - 数据缺失或层数不足")

    plt.tight_layout()

    # 保存图像
    output_path = Path(output_dir) / f"{file_prefix}_{layer_name.replace(' ', '_')}_attention.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{layer_name}文本注意力可视化已保存: {output_path}")

def visualize_multi_head_attention(attention_data, output_dir="outputs"):
    """
    可视化多头注意力模式和头部特化分析

    Args:
        attention_data: 注意力数据
        output_dir: 输出目录
    """
    print("生成多头注意力模式可视化...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('多头注意力深度分析', fontsize=16, fontweight='bold')

    if attention_data and attention_data.get('text_attentions'):
        text_attentions = attention_data['text_attentions']
        texts = attention_data.get('texts', [])

        # 使用最后一层的注意力
        final_attention = text_attentions[-1][0]  # [num_heads, seq_len, seq_len]
        num_heads, seq_len, _ = final_attention.shape

        # 1. 选择代表性头部的注意力模式 (上半部分)
        heads_to_show = min(6, num_heads)
        if num_heads >= 6:
            head_indices = [0, num_heads//5, 2*num_heads//5, 3*num_heads//5, 4*num_heads//5, num_heads-1]
        else:
            head_indices = list(range(num_heads))

        for i, head_idx in enumerate(head_indices[:heads_to_show]):
            ax = plt.subplot(3, 3, i+1)
            head_attention = final_attention[head_idx, :20, :20].numpy()

            im = plt.imshow(head_attention, cmap='Blues', interpolation='nearest')
            plt.title(f'Head {head_idx}', fontsize=11, fontweight='bold')
            plt.xlabel('Key位置', fontsize=9)
            plt.ylabel('Query位置', fontsize=9)

            # 添加注意力强度统计
            max_att = head_attention.max()
            mean_att = head_attention.mean()
            plt.text(0.02, 0.98, f'Max: {max_att:.3f}\nMean: {mean_att:.3f}',
                    transform=ax.transAxes, fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

        # 2. 头部特化分析 (右上)
        ax_spec = plt.subplot(3, 3, 7)

        # 计算每个头的特化指标
        head_specializations = []
        head_entropies = []

        for head_idx in range(num_heads):
            head_att = final_attention[head_idx, :min(seq_len, 20), :min(seq_len, 20)].numpy()

            # 计算注意力熵 (衡量注意力分散程度)
            head_att_flat = head_att.flatten()
            head_att_flat = head_att_flat[head_att_flat > 1e-8]  # 避免log(0)
            entropy = -np.sum(head_att_flat * np.log(head_att_flat + 1e-8))
            head_entropies.append(entropy)

            # 计算对角线注意力比例 (自注意力强度)
            diagonal_sum = np.sum([head_att[j, j] for j in range(min(head_att.shape))])
            total_sum = np.sum(head_att)
            diagonal_ratio = diagonal_sum / (total_sum + 1e-8)
            head_specializations.append(diagonal_ratio)

        # 绘制头部特化散点图
        scatter = ax_spec.scatter(head_specializations, head_entropies,
                                 c=range(num_heads), cmap='viridis', s=60, alpha=0.7)
        ax_spec.set_xlabel('自注意力比例', fontsize=10)
        ax_spec.set_ylabel('注意力熵', fontsize=10)
        ax_spec.set_title('头部特化分析', fontsize=11, fontweight='bold')
        ax_spec.grid(True, alpha=0.3)

        # 添加头部标签
        for i, (x, y) in enumerate(zip(head_specializations, head_entropies)):
            if i % 2 == 0:  # 只标注部分头部避免拥挤
                ax_spec.annotate(f'H{i}', (x, y), xytext=(5, 5),
                               textcoords='offset points', fontsize=8)

        # 3. 注意力层次分析 (右中)
        ax_layer = plt.subplot(3, 3, 8)

        # 分析不同层的注意力模式
        layer_indices = [0, len(text_attentions)//3, 2*len(text_attentions)//3, -1]
        layer_names = ['早期层', '中早期层', '中晚期层', '最终层']
        layer_colors = ['lightblue', 'skyblue', 'steelblue', 'darkblue']

        layer_avg_attentions = []
        for layer_idx in layer_indices:
            if layer_idx == -1:
                layer_idx = len(text_attentions) - 1
            if layer_idx < len(text_attentions):
                layer_att = text_attentions[layer_idx][0].mean(dim=0).numpy()  # 平均所有头
                avg_attention = layer_att[:min(seq_len, 15), :min(seq_len, 15)].mean()
                layer_avg_attentions.append(avg_attention)
            else:
                layer_avg_attentions.append(0)

        bars = ax_layer.bar(layer_names, layer_avg_attentions, color=layer_colors, alpha=0.7)
        ax_layer.set_title('层次注意力强度', fontsize=11, fontweight='bold')
        ax_layer.set_ylabel('平均注意力', fontsize=10)
        ax_layer.tick_params(axis='x', rotation=45)
        ax_layer.grid(True, alpha=0.3)

        # 添加数值标签
        for bar, value in zip(bars, layer_avg_attentions):
            height = bar.get_height()
            ax_layer.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                         f'{value:.3f}', ha='center', va='bottom', fontsize=9)

        # 4. 注意力模式类型分析 (右下)
        ax_pattern = plt.subplot(3, 3, 9)

        # 分析不同类型的注意力模式
        pattern_types = ['局部注意力', '全局注意力', '稀疏注意力']
        pattern_counts = [0, 0, 0]

        for head_idx in range(num_heads):
            head_att = final_attention[head_idx, :min(seq_len, 15), :min(seq_len, 15)].numpy()

            # 计算局部注意力 (对角线附近)
            local_attention = 0
            for i in range(min(head_att.shape[0]-1, 10)):
                local_attention += head_att[i, i] + head_att[i, i+1] + head_att[i+1, i]

            # 计算全局注意力 (第一个token的注意力)
            global_attention = head_att[0, :].sum() if head_att.shape[0] > 0 else 0

            # 计算稀疏度
            non_zero_ratio = np.sum(head_att > 0.01) / head_att.size

            # 分类
            if local_attention > global_attention and non_zero_ratio > 0.3:
                pattern_counts[0] += 1  # 局部
            elif global_attention > local_attention:
                pattern_counts[1] += 1  # 全局
            else:
                pattern_counts[2] += 1  # 稀疏

        wedges, texts_pie, autotexts = ax_pattern.pie(pattern_counts, labels=pattern_types,
                                                     autopct='%1.1f%%', startangle=90,
                                                     colors=['lightcoral', 'lightgreen', 'lightskyblue'])
        ax_pattern.set_title('注意力模式分布', fontsize=11, fontweight='bold')

        # 添加文本信息
        if texts:
            text_info = f"分析文本: {texts[0][:40]}{'...' if len(texts[0]) > 40 else ''}\n"
            text_info += f"注意力头数: {num_heads}, 序列长度: {seq_len}, 层数: {len(text_attentions)}"
            fig.text(0.5, 0.02, text_info, ha='center', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

        print(f"✓ 显示多头注意力深度分析 - {num_heads}个头, {len(text_attentions)}层")
    else:
        plt.text(0.5, 0.5, '需要text_attentions数据\n显示多头注意力模式',
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        print("⚠️ 无法显示多头注意力模式 - 数据缺失")

    plt.tight_layout()

    # 保存图像
    output_path = Path(output_dir) / "12_multi_head_attention.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"多头注意力模式可视化已保存: {output_path}")

def visualize_text_image_similarities(attention_data, output_dir="outputs"):
    """
    可视化图文相似度分布

    Args:
        attention_data: 注意力数据
        output_dir: 输出目录
    """
    print("生成图文相似度分布可视化...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.title('图文相似度分布', fontsize=16, fontweight='bold')

    try:
        similarities = attention_data.get('similarities')
        if similarities is not None:
            similarities = similarities.numpy()
            texts = attention_data.get('texts', [f'Text{i}' for i in range(len(similarities))])

            # 绘制热力图
            sim_matrix = similarities.reshape(1, -1)
            im = plt.imshow(sim_matrix, cmap='RdYlGn', aspect='auto')
            plt.colorbar(im, shrink=0.8)
            plt.xlabel('文本索引', fontsize=12)
            plt.yticks([])
            plt.ylabel('图像', fontsize=12)

            # 添加数值标签 - 修复：将NumPy数组元素转换为浮点数
            for i, sim in enumerate(similarities):
                # 将NumPy数组元素转换为浮点数
                try:
                    sim_value = float(sim.item()) if hasattr(sim, 'item') else float(sim)
                    plt.text(i, 0, f'{sim_value:.3f}', ha='center', va='center',
                            fontweight='bold', fontsize=10,
                            color='white' if sim_value < 0.5 else 'black')
                except (ValueError, TypeError) as e:
                    print(f"警告: 无法转换相似度值 {sim}: {e}")
                    continue

            # 如果有文本信息，添加说明
            if attention_data.get('texts'):
                plt.xticks(range(len(similarities)),
                          [f"T{i+1}" for i in range(len(similarities))],
                          rotation=0)

                text_info = "文本详情:\n\n"
                for i, text in enumerate(texts):
                    text_info += f"T{i+1}: {text[:30]}{'...' if len(text) > 30 else ''}\n"

                plt.figtext(0.5, 0.01, text_info, ha='center', fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

            print(f"✓ 显示相似度分布: {similarities}")
        else:
            plt.text(0.5, 0.5, '需要similarities数据', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
            print("⚠️ 无法显示相似度分布 - similarities缺失")
    except Exception as e:
        plt.text(0.5, 0.5, f'绘制出错:\n{str(e)[:50]}...', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        print(f"⚠️ 无法显示相似度分布 - {e}")

    # 保存图像
    output_path = Path(output_dir) / "13_text_image_similarities.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图文相似度分布可视化已保存: {output_path}")

def visualize_feature_space_projection(attention_data, output_dir="outputs"):
    """
    可视化特征空间投影

    Args:
        attention_data: 注意力数据
        output_dir: 输出目录
    """
    print("生成特征空间投影可视化...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.title('特征空间投影 (前两个维度)', fontsize=16, fontweight='bold')

    try:
        if attention_data and attention_data.get('image_embeds') is not None and attention_data.get('text_embeds') is not None:
            image_embed = attention_data['image_embeds'][0].numpy()
            text_embeds = attention_data['text_embeds'].numpy()

            print(f"特征维度检查: image_embed={image_embed.shape}, text_embeds={text_embeds.shape}")

            # 使用前两个特征维度进行2D投影
            if image_embed.shape[0] >= 2 and text_embeds.shape[1] >= 2:
                print("使用前两个特征维度进行2D投影...")

                # 提取前两个维度
                image_2d = image_embed[:2]
                text_2d = text_embeds[:, :2]

                # 绘制图像特征点
                plt.scatter(image_2d[0], image_2d[1], c='red', s=200, marker='*',
                           label='图像特征', zorder=5)

                # 绘制文本特征点
                for i in range(len(text_2d)):
                    plt.scatter(text_2d[i, 0], text_2d[i, 1], c='blue', s=100, alpha=0.7)
                    plt.annotate(f'T{i+1}', (text_2d[i, 0], text_2d[i, 1]),
                                xytext=(5, 5), textcoords='offset points', fontsize=10)

                # 绘制连接线（相似度）
                if attention_data.get('similarities') is not None:
                    similarities = attention_data['similarities'].numpy()
                    for i in range(len(text_2d)):
                        try:
                            alpha_value = float(similarities[i].item()) if hasattr(similarities[i], 'item') else float(similarities[i])
                            alpha_value = max(0.1, min(1.0, alpha_value))  # 确保alpha在有效范围内
                            plt.plot([image_2d[0], text_2d[i, 0]],
                                    [image_2d[1], text_2d[i, 1]],
                                    'gray', alpha=alpha_value, linewidth=2)
                        except (ValueError, TypeError) as e:
                            print(f"警告: 无法处理相似度值 {similarities[i]}: {e}")
                            plt.plot([image_2d[0], text_2d[i, 0]],
                                    [image_2d[1], text_2d[i, 1]],
                                    'gray', alpha=0.5, linewidth=2)

                plt.xlabel('特征维度1', fontsize=12)
                plt.ylabel('特征维度2', fontsize=12)
                plt.legend()
                plt.grid(True, alpha=0.3)

                # 如果有文本信息，添加说明
                if attention_data.get('texts'):
                    text_info = "文本详情:\n\n"
                    for i, text in enumerate(attention_data['texts']):
                        text_info += f"T{i+1}: {text[:30]}{'...' if len(text) > 30 else ''}\n"

                    plt.figtext(0.5, 0.01, text_info, ha='center', fontsize=10,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

                print("✓ 特征空间投影完成")
            else:
                plt.text(0.5, 0.5, f'特征维度不足\n需要至少2维', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
                print("⚠️ 无法显示特征投影 - 特征维度不足")
        else:
            plt.text(0.5, 0.5, '需要image_embeds和\ntext_embeds数据', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
            print("⚠️ 无法显示特征投影 - 嵌入数据缺失")
    except Exception as e:
        plt.text(0.5, 0.5, f'特征空间投影出错:\n{str(e)[:30]}...', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        print(f"⚠️ 特征空间投影出错: {e}")

    # 保存图像
    output_path = Path(output_dir) / "14_feature_space_projection.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"特征空间投影可视化已保存: {output_path}")

def create_english_token_attention_analysis(attention_data, output_dir="outputs"):
    """
    专门针对英文文本"two dogs lying on a cushion in the sun"的token级别注意力分析
    过滤掉CLS和SEP token，只显示实际内容token之间的关系

    Args:
        attention_data: 注意力数据
        output_dir: 输出目录
    """
    print("生成英文Token级别注意力关系可视化（过滤CLS/SEP）...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('English Content Tokens Attention Analysis (No CLS/SEP)', fontsize=16, fontweight='bold')

    if attention_data and attention_data.get('text_attentions') and attention_data.get('texts'):
        texts = attention_data['texts']
        text_attentions = attention_data['text_attentions']
        text_inputs = attention_data.get('text_inputs', {})

        # 寻找英文文本
        target_text = None
        target_idx = 0
        for i, text in enumerate(texts):
            if any(word in text.lower() for word in ['two', 'dogs', 'lying', 'cushion', 'sun']):
                target_text = text
                target_idx = i
                break

        if target_text is None:
            target_text = texts[0]  # 如果没找到英文文本，使用第一个

        if 'input_ids' in text_inputs and 'attention_mask' in text_inputs:
            input_ids = text_inputs['input_ids'][target_idx].numpy()
            attention_mask = text_inputs['attention_mask'][target_idx].numpy()
            valid_length = int(attention_mask.sum())

            # 使用最后一层的注意力
            final_attention = text_attentions[-1][target_idx]  # [num_heads, seq_len, seq_len]
            avg_attention = final_attention.mean(dim=0).numpy()  # 平均所有头

            # 过滤掉CLS和SEP token，只保留实际内容token
            content_start = 1  # 跳过CLS
            content_end = valid_length - 1  # 跳过SEP
            content_length = content_end - content_start

            # 限制显示长度（只显示内容token）
            display_length = min(content_length, 9)  # 显示9个内容token

            # 提取内容token之间的注意力矩阵
            attention_matrix = avg_attention[content_start:content_start+display_length,
                                           content_start:content_start+display_length]

            # 为英文文本创建更准确的token标签（只包含内容token）
            english_content_tokens = ['two', 'dogs', 'lying', 'on', 'a', 'cushion', 'in', 'the', 'sun']

            token_labels = []
            for i in range(display_length):
                token_id = input_ids[content_start + i]  # 调整索引到内容token
                if i < len(english_content_tokens):
                    token_labels.append(f'{english_content_tokens[i]}\n{token_id}')
                else:
                    token_labels.append(f'T{i+1}\n{token_id}')

            # 1. 主要的注意力热力图 (左上，占据较大空间)
            ax_main = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=2)
            im = ax_main.imshow(attention_matrix, cmap='Blues', interpolation='nearest')
            ax_main.set_title(f'Content Token Attention Matrix\n"{target_text}"', fontsize=14, fontweight='bold')
            ax_main.set_xlabel('Key Token', fontsize=12)
            ax_main.set_ylabel('Query Token', fontsize=12)
            ax_main.set_xticks(range(display_length))
            ax_main.set_yticks(range(display_length))
            ax_main.set_xticklabels([label.split('\n')[0] for label in token_labels], rotation=45, ha='right')
            ax_main.set_yticklabels([label.split('\n')[0] for label in token_labels])

            # 添加数值标签
            for i in range(display_length):
                for j in range(display_length):
                    value = attention_matrix[i, j]
                    color = 'white' if value > 0.5 else 'black'
                    ax_main.text(j, i, f'{value:.2f}', ha='center', va='center',
                               color=color, fontsize=8, fontweight='bold')

            plt.colorbar(im, ax=ax_main, shrink=0.8)

            # 2. 关键词"dogs"的注意力分布 (右上)
            ax_key = plt.subplot2grid((3, 4), (0, 2), colspan=2)

            # 找到"dogs"这个token的位置（在过滤后的内容token中）
            dogs_idx = 1  # 在内容token中："dogs"是第1个token (0:two, 1:dogs, 2:lying...)
            if dogs_idx < display_length:
                key_attention = attention_matrix[dogs_idx, :]

                bars = ax_key.bar(range(display_length), key_attention,
                                 color=['red' if i == dogs_idx else 'lightblue' for i in range(display_length)],
                                 alpha=0.7)
                ax_key.set_title(f'Key Token "dogs" Attention Distribution', fontsize=12, fontweight='bold')
                ax_key.set_xlabel('Target Token')
                ax_key.set_ylabel('Attention Weight')
                ax_key.set_xticks(range(display_length))
                ax_key.set_xticklabels([label.split('\n')[0] for label in token_labels], rotation=45)

                # 添加数值标签
                for bar, value in zip(bars, key_attention):
                    height = bar.get_height()
                    ax_key.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{value:.2f}', ha='center', va='bottom', fontsize=9)

            # 3. Token自注意力强度 (左下)
            ax_self = plt.subplot2grid((3, 4), (1, 2))

            self_attention = np.diag(attention_matrix)
            bars = ax_self.bar(range(display_length), self_attention,
                              color='orange', alpha=0.7)
            ax_self.set_title('Token Self-Attention Strength', fontsize=12, fontweight='bold')
            ax_self.set_xlabel('Token')
            ax_self.set_ylabel('Self-Attention')
            ax_self.set_xticks(range(display_length))
            ax_self.set_xticklabels([label.split('\n')[0] for label in token_labels], rotation=45)

            # 4. 注意力网络图 (右下)
            ax_network = plt.subplot2grid((3, 4), (1, 3))

            # 创建简化的注意力网络
            threshold = 0.1  # 只显示注意力权重大于阈值的连接

            # 绘制token节点
            positions = {}
            for i in range(display_length):
                angle = 2 * np.pi * i / display_length
                x = np.cos(angle)
                y = np.sin(angle)
                positions[i] = (x, y)

                # 节点大小根据总注意力权重
                node_size = attention_matrix[i, :].sum() * 1000
                color = 'red' if i == dogs_idx else 'lightblue'
                ax_network.scatter(x, y, s=node_size, alpha=0.7, c=color)
                ax_network.text(x*1.2, y*1.2, token_labels[i].split('\n')[0],
                               ha='center', va='center', fontsize=9)

            # 绘制注意力连接
            for i in range(display_length):
                for j in range(display_length):
                    if i != j and attention_matrix[i, j] > threshold:
                        x1, y1 = positions[i]
                        x2, y2 = positions[j]
                        alpha = attention_matrix[i, j]
                        ax_network.plot([x1, x2], [y1, y2], 'gray',
                                       alpha=alpha, linewidth=alpha*3)

            ax_network.set_title('Token Attention Network', fontsize=12, fontweight='bold')
            ax_network.set_xlim(-1.5, 1.5)
            ax_network.set_ylim(-1.5, 1.5)
            ax_network.axis('off')

            # 5. 详细说明 (底部)
            ax_info = plt.subplot2grid((3, 4), (2, 0), colspan=4)
            ax_info.axis('off')

            info_text = f"Analysis Text: {target_text}\n\n"
            info_text += f"Content Token Breakdown (CLS/SEP filtered, showing {display_length} tokens):\n"
            for i, label in enumerate(token_labels):
                token_type = label.split('\n')[0]
                token_id = label.split('\n')[1]
                info_text += f"[{i}] {token_type} (ID: {token_id})  "
                if (i + 1) % 4 == 0:
                    info_text += "\n"

            info_text += f"\n\nKey Findings:\n"
            info_text += f"• Strongest self-attention: Token {np.argmax(self_attention)} ({token_labels[np.argmax(self_attention)].split()[0]})\n"
            info_text += f"• Strongest cross-token attention: {attention_matrix.max():.3f}\n"
            info_text += f"• Average attention weight: {attention_matrix.mean():.3f}\n"
            if dogs_idx < display_length:
                most_attended = np.argmax(attention_matrix[dogs_idx, :])
                info_text += f"• 'dogs' pays most attention to: {token_labels[most_attended].split()[0]}"

            ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes, fontsize=11,
                        verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))

            print(f"✓ 显示英文内容Token级别注意力分析 - {display_length}个tokens（已过滤CLS/SEP）")
        else:
            plt.text(0.5, 0.5, 'Need text_inputs data\nfor Token-level analysis',
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
            print("⚠️ 无法显示Token级别分析 - text_inputs缺失")
    else:
        plt.text(0.5, 0.5, 'Need text_attentions and texts data',
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        print("⚠️ 无法显示Token级别分析 - 数据缺失")

    plt.tight_layout()

    # 保存图像
    output_path = Path(output_dir) / "08_english_token_attention_details.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"英文Token级别注意力关系可视化已保存: {output_path}")

def create_text_attention_visualizations(attention_data, output_dir="outputs"):
    """
    创建所有文本注意力相关的可视化

    Args:
        attention_data: 注意力数据
        output_dir: 输出目录
    """
    print("生成文本注意力分析可视化...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    # 1. 文本Token分解
    visualize_text_inputs(attention_data, output_dir)

    # 2. 英文Token级别注意力关系详细分析
    create_english_token_attention_analysis(attention_data, output_dir)

    # 3. 早期层文本注意力
    if attention_data and attention_data.get('text_attentions') and len(attention_data['text_attentions']) > 1:
        visualize_attention_layer(attention_data, 1, "早期层", output_dir, "09")

    # 4. 中期层文本注意力
    if attention_data and attention_data.get('text_attentions') and len(attention_data['text_attentions']) > 5:
        visualize_attention_layer(attention_data, 5, "中期层", output_dir, "10")

    # 5. 最终层文本注意力
    if attention_data and attention_data.get('text_attentions'):
        visualize_attention_layer(attention_data, -1, "最终层", output_dir, "11")

    # 6. 多头注意力深度分析
    visualize_multi_head_attention(attention_data, output_dir)

    # 7. 图文相似度分布
    visualize_text_image_similarities(attention_data, output_dir)

    # 8. 特征空间投影
    visualize_feature_space_projection(attention_data, output_dir)

    print("文本注意力分析可视化完成")

