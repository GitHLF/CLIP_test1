from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def visualize_text_inputs(attention_data, output_dir="outputs"):
    """
    可视化文本输入

    Args:
        attention_data: 注意力数据
        output_dir: 输出目录
    """
    print("生成文本输入可视化...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.title('文本输入详情', fontsize=16, fontweight='bold')

    if attention_data and attention_data.get('texts'):
        texts = attention_data['texts']
        text_display = "注意力分析文本:\n\n"
        for i, text in enumerate(texts):
            text_display += f"{i+1}. {text}\n\n"

        plt.text(0.05, 0.95, text_display, transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        print(f"✓ 显示 {len(texts)} 个注意力分析文本")
    else:
        plt.text(0.5, 0.5, '需要注意力数据\n显示具体分析文本', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        print("⚠️ 无法显示注意力分析文本 - 数据缺失")

    plt.axis('off')

    # 保存图像
    output_path = Path(output_dir) / "07_text_inputs.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"文本输入可视化已保存: {output_path}")

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
    可视化特定层的文本注意力

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

    plt.figure(figsize=(10, 8))
    plt.title(f'{layer_name}文本注意力', fontsize=16, fontweight='bold')

    if attention_data and attention_data.get('text_attentions') and len(attention_data['text_attentions']) > layer_idx:
        text_attentions = attention_data['text_attentions']

        # 获取注意力矩阵
        text_att = text_attentions[layer_idx][0, 0, :20, :20].numpy()

        # 绘制热力图
        im = plt.imshow(text_att, cmap='Blues', interpolation='nearest')
        plt.colorbar(im, shrink=0.8)
        plt.xlabel('Key位置', fontsize=12)
        plt.ylabel('Query位置', fontsize=12)

        # 如果有文本信息，添加说明
        if attention_data.get('texts'):
            text_info = f"显示文本: {attention_data['texts'][0][:30]}{'...' if len(attention_data['texts'][0]) > 30 else ''}"
            plt.figtext(0.5, 0.01, text_info, ha='center', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

        print(f"✓ 显示{layer_name}文本注意力")
    else:
        plt.text(0.5, 0.5, f'需要text_attentions数据\n或层数不足', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        print(f"⚠️ 无法显示{layer_name}文本注意力 - 数据缺失或层数不足")

    # 保存图像
    output_path = Path(output_dir) / f"{file_prefix}_{layer_name.replace(' ', '_')}_attention.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{layer_name}文本注意力可视化已保存: {output_path}")

def visualize_multi_head_attention(attention_data, output_dir="outputs"):
    """
    可视化多头注意力模式

    Args:
        attention_data: 注意力数据
        output_dir: 输出目录
    """
    print("生成多头注意力模式可视化...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    plt.figure(figsize=(12, 8))
    plt.title('多头注意力模式', fontsize=16, fontweight='bold')

    if attention_data and attention_data.get('text_attentions'):
        text_attentions = attention_data['text_attentions']

        # 获取最后一层的多头注意力
        num_heads = min(4, text_attentions[-1].shape[1])
        head_patterns = []

        for head in range(num_heads):
            head_att = text_attentions[-1][0, head, :15, :15].numpy()
            diag_strength = np.diag(head_att).mean()
            off_diag_strength = (head_att.sum() - np.diag(head_att).sum()) / (15*15 - 15)
            head_patterns.append([diag_strength, off_diag_strength])

        head_patterns = np.array(head_patterns)
        x = np.arange(num_heads)
        width = 0.35

        # 绘制条形图
        plt.bar(x - width/2, head_patterns[:, 0], width, label='对角线注意力', alpha=0.8)
        plt.bar(x + width/2, head_patterns[:, 1], width, label='非对角线注意力', alpha=0.8)
        plt.xlabel('注意力头', fontsize=12)
        plt.ylabel('注意力强度', fontsize=12)
        plt.xticks(x, [f'头{i+1}' for i in range(num_heads)])
        plt.legend()
        plt.grid(True, alpha=0.3)

        print(f"✓ 显示{num_heads}个注意力头的模式")
    else:
        plt.text(0.5, 0.5, '需要text_attentions数据', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        print("⚠️ 无法显示多头注意力模式 - 数据缺失")

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
                sim_value = float(sim)
                plt.text(i, 0, f'{sim_value:.3f}', ha='center', va='center',
                        fontweight='bold', fontsize=10,
                        color='white' if sim_value < 0.5 else 'black')

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
                        alpha = similarities[i]
                        plt.plot([image_2d[0], text_2d[i, 0]],
                                [image_2d[1], text_2d[i, 1]],
                                'gray', alpha=alpha, linewidth=2)

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

    # 1. 文本输入
    visualize_text_inputs(attention_data, output_dir)

    # 2. Token长度分布
    visualize_token_length_distribution(attention_data, output_dir)

    # 3. 早期层文本注意力
    if attention_data and attention_data.get('text_attentions') and len(attention_data['text_attentions']) > 1:
        visualize_attention_layer(attention_data, 1, "早期层", output_dir, "09")

    # 4. 中期层文本注意力
    if attention_data and attention_data.get('text_attentions') and len(attention_data['text_attentions']) > 5:
        visualize_attention_layer(attention_data, 5, "中期层", output_dir, "10")

    # 5. 最终层文本注意力
    if attention_data and attention_data.get('text_attentions'):
        visualize_attention_layer(attention_data, -1, "最终层", output_dir, "11")

    # 6. 多头注意力模式
    visualize_multi_head_attention(attention_data, output_dir)

    # 7. 图文相似度分布
    visualize_text_image_similarities(attention_data, output_dir)

    # 8. 特征空间投影
    visualize_feature_space_projection(attention_data, output_dir)

    print("文本注意力分析可视化完成")

