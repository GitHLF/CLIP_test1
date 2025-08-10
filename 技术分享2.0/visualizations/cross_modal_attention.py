from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def visualize_image_attention_regions(original_image, attention_data, output_dir="outputs"):
    """
    可视化图像关注区域

    Args:
        original_image: 原始图像
        attention_data: 注意力数据
        output_dir: 输出目录
    """
    print("生成图像关注区域可视化...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    plt.figure(figsize=(10, 10))
    plt.title('图像关注区域 (Vision Transformer)', fontsize=16, fontweight='bold')

    try:
        if attention_data and attention_data.get('vision_attentions'):
            # 获取图像最关注的区域
            vision_att = attention_data['vision_attentions'][-1][0, :, 0, 1:].mean(dim=0)

            # 动态计算patch配置
            num_patches_total = len(vision_att)
            num_patches = int(np.sqrt(num_patches_total))
            patch_size = 224 // num_patches if num_patches > 0 else 16

            # 显示原图
            resized_image = original_image.resize((224, 224))
            plt.imshow(resized_image, alpha=0.7)

            # 叠加注意力热力图
            if num_patches * num_patches == num_patches_total:
                attention_map = vision_att.reshape(num_patches, num_patches).numpy()
                attention_resized = np.kron(attention_map, np.ones((patch_size, patch_size)))
                plt.imshow(attention_resized, cmap='hot', alpha=0.5, extent=[0, 224, 224, 0])

                # 标注最关注的前3个区域
                top3_patches = torch.topk(vision_att, min(3, len(vision_att))).indices
                colors = ['yellow', 'cyan', 'lime']
                for i, patch_idx in enumerate(top3_patches):
                    if patch_idx < num_patches_total:
                        row = patch_idx // num_patches
                        col = patch_idx % num_patches
                        if row < num_patches and col < num_patches:
                            rect = plt.Rectangle((col*patch_size, row*patch_size), patch_size, patch_size,
                                               linewidth=3, edgecolor=colors[i], facecolor='none')
                            plt.gca().add_patch(rect)
                            plt.text(col*patch_size + 8, row*patch_size + 8, f'R{i+1}',
                                    color=colors[i], fontweight='bold', fontsize=12)

            plt.xlim(0, 224)
            plt.ylim(224, 0)
            plt.axis('off')
            print("✓ 显示图像关注区域")
        else:
            plt.text(0.5, 0.5, '需要视觉注意力数据', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
            print("⚠️ 无法显示图像关注区域 - 数据缺失")
    except Exception as e:
        plt.text(0.5, 0.5, f'绘制出错:\n{str(e)[:50]}...', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        print(f"⚠️ 绘制图像关注区域时出错: {e}")

    # 保存图像
    output_path = Path(output_dir) / "22_image_attention_regions.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图像关注区域可视化已保存: {output_path}")

def visualize_text_attention_patterns(attention_data, output_dir="outputs"):
    """
    可视化文本注意力模式

    Args:
        attention_data: 注意力数据
        output_dir: 输出目录
    """
    print("生成文本注意力模式可视化...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.title('文本注意力模式', fontsize=16, fontweight='bold')

    try:
        if attention_data and attention_data.get('text_attentions') and attention_data.get('texts'):
            text_att = attention_data['text_attentions'][-1][0, 0, :, :].numpy()
            text_length = attention_data['text_inputs']['attention_mask'][0].sum().item()

            valid_att = text_att[:text_length, :text_length]
            im = plt.imshow(valid_att, cmap='Blues', interpolation='nearest')
            plt.colorbar(im, shrink=0.8)
            plt.xlabel('Key Token位置', fontsize=12)
            plt.ylabel('Query Token位置', fontsize=12)

            # 标记重要的token位置
            important_tokens = np.unravel_index(np.argpartition(valid_att.ravel(), -3)[-3:], valid_att.shape)
            for i, (row, col) in enumerate(zip(important_tokens[0], important_tokens[1])):
                plt.scatter(col, row, c='red', s=100, alpha=0.7, marker='x')

            # 添加文本说明
            text_info = f"显示文本: {attention_data['texts'][0][:30]}{'...' if len(attention_data['texts'][0]) > 30 else ''}"
            plt.figtext(0.5, 0.01, text_info, ha='center', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

            print("✓ 显示文本注意力模式")
        else:
            plt.text(0.5, 0.5, '需要文本注意力数据', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
            print("⚠️ 无法显示文本注意力模式 - 数据缺失")
    except Exception as e:
        plt.text(0.5, 0.5, f'绘制出错:\n{str(e)[:50]}...', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        print(f"⚠️ 绘制文本注意力模式时出错: {e}")

    # 保存图像
    output_path = Path(output_dir) / "23_text_attention_patterns.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"文本注意力模式可视化已保存: {output_path}")

def visualize_cross_modal_similarity_matrix(attention_data, output_dir="outputs"):
    """
    可视化跨模态相似度矩阵

    Args:
        attention_data: 注意力数据
        output_dir: 输出目录
    """
    print("生成跨模态相似度矩阵可视化...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.title('跨模态相似度矩阵', fontsize=16, fontweight='bold')

    try:
        similarities = attention_data.get('similarities')
        if similarities is not None:
            similarities = similarities.numpy()
            texts = attention_data.get('texts', [f'Text{i}' for i in range(len(similarities))])

            sim_matrix = similarities.reshape(-1, 1).T
            im = plt.imshow(sim_matrix, cmap='RdYlGn', aspect='auto')
            plt.colorbar(im, shrink=0.8)

            plt.xlabel('文本索引', fontsize=12)
            plt.ylabel('图像', fontsize=12)
            plt.xticks(range(len(texts)), [f'T{i+1}' for i in range(len(texts))])
            plt.yticks([0], ['Image'])

            for i, sim in enumerate(similarities):
                # 将NumPy数组元素转换为浮点数
                try:
                    sim_value = float(sim.item()) if hasattr(sim, 'item') else float(sim)
                    color = 'white' if sim_value < 0.5 else 'black'
                    plt.text(i, 0, f'{sim_value:.3f}', ha='center', va='center',
                            fontweight='bold', color=color)
                except (ValueError, TypeError) as e:
                    print(f"警告: 无法转换相似度值 {sim}: {e}")
                    continue

            # 添加文本说明
            text_info = "文本详情:\n\n"
            for i, text in enumerate(texts):
                text_info += f"T{i+1}: {text[:30]}{'...' if len(text) > 30 else ''}\n"

            plt.figtext(0.5, 0.01, text_info, ha='center', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

            print("✓ 显示跨模态相似度矩阵")
        else:
            plt.text(0.5, 0.5, '需要相似度数据', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
            print("⚠️ 无法显示跨模态相似度矩阵 - 数据缺失")
    except Exception as e:
        plt.text(0.5, 0.5, f'绘制出错:\n{str(e)[:50]}...', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        print(f"⚠️ 绘制跨模态相似度矩阵时出错: {e}")

    # 保存图像
    output_path = Path(output_dir) / "24_cross_modal_similarity_matrix.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"跨模态相似度矩阵可视化已保存: {output_path}")

def visualize_feature_alignment(attention_data, output_dir="outputs"):
    """
    可视化特征对齐程度

    Args:
        attention_data: 注意力数据
        output_dir: 输出目录
    """
    print("生成特征对齐程度可视化...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    plt.figure(figsize=(12, 8))
    plt.title('特征对齐程度 (图像-文本特征相关性)', fontsize=16, fontweight='bold')

    try:
        image_embeds = attention_data.get('image_embeds')
        text_embeds = attention_data.get('text_embeds')
        if image_embeds is not None and text_embeds is not None:
            image_embed = image_embeds[0].numpy()
            text_embeds_np = text_embeds.numpy()

            correlations = []
            for text_embed in text_embeds_np:
                corr = np.corrcoef(image_embed, text_embed)[0, 1]
                correlations.append(corr)

            texts = attention_data.get('texts', [f'Text{i}' for i in range(len(correlations))])
            colors = plt.cm.RdYlGn(np.linspace(0.3, 1, len(correlations)))

            bars = plt.bar(range(len(correlations)), correlations, color=colors)
            plt.xlabel('文本索引', fontsize=12)
            plt.ylabel('特征相关性', fontsize=12)
            plt.xticks(range(len(texts)), [f'T{i+1}' for i in range(len(texts))])
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            plt.grid(True, alpha=0.3)

            for bar, corr in zip(bars, correlations):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height > 0 else height - 0.03,
                        f'{corr:.3f}', ha='center', va='bottom' if height > 0 else 'top',
                        fontweight='bold', fontsize=10)

            # 添加文本说明
            text_info = "文本详情:\n\n"
            for i, text in enumerate(texts):
                text_info += f"T{i+1}: {text[:30]}{'...' if len(text) > 30 else ''}\n"

            plt.figtext(0.5, 0.01, text_info, ha='center', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

            print("✓ 显示特征对齐程度")
        else:
            plt.text(0.5, 0.5, '需要特征嵌入数据', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
            print("⚠️ 无法显示特征对齐程度 - 数据缺失")
    except Exception as e:
        plt.text(0.5, 0.5, f'绘制出错:\n{str(e)[:50]}...', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        print(f"⚠️ 绘制特征对齐程度时出错: {e}")

    # 保存图像
    output_path = Path(output_dir) / "25_feature_alignment.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"特征对齐程度可视化已保存: {output_path}")

def visualize_attention_flow(attention_data, output_dir="outputs"):
    """
    可视化注意力流向图

    Args:
        attention_data: 注意力数据
        output_dir: 输出目录
    """
    print("生成注意力流向图可视化...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.title('跨模态注意力流向', fontsize=16, fontweight='bold')

    try:
        if (attention_data and attention_data.get('vision_attentions') and
            attention_data.get('text_attentions') and attention_data.get('text_inputs')):

            # 提取视觉注意力中的重要区域
            vision_att = attention_data['vision_attentions'][-1][0, :, 0, 1:].mean(dim=0)
            top_patches = torch.topk(vision_att, 3).indices.numpy()

            # 提取文本注意力中的重要token
            text_att = attention_data['text_attentions'][-1][0, 0, :, :].numpy()
            text_length = attention_data['text_inputs']['attention_mask'][0].sum().item()
            avg_att_per_token = text_att[:text_length, :text_length].mean(axis=1)
            top_tokens = np.argsort(avg_att_per_token)[-3:]

            # 绘制流向图
            for i, patch in enumerate(top_patches):
                y_pos = 0.8 - i * 0.3
                plt.scatter(0.2, y_pos, s=200, c='red', alpha=0.7)
                plt.text(0.1, y_pos, f'区域{i+1}', ha='right', va='center', fontweight='bold')

            for i, token in enumerate(top_tokens):
                y_pos = 0.8 - i * 0.3
                plt.scatter(0.8, y_pos, s=200, c='blue', alpha=0.7)
                plt.text(0.9, y_pos, f'Token{token}', ha='left', va='center', fontweight='bold')

            # 绘制连接线
            similarities = attention_data['similarities'].numpy()
            max_sim_idx = np.argmax(similarities)

            for i in range(min(3, len(top_patches))):
                for j in range(min(3, len(top_tokens))):
                    y1 = 0.8 - i * 0.3
                    y2 = 0.8 - j * 0.3
                    alpha = similarities[max_sim_idx] if i == 0 and j == 0 else 0.3
                    plt.plot([0.2, 0.8], [y1, y2], 'gray', alpha=alpha, linewidth=2)

            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.text(0.2, 0.95, '图像区域', ha='center', fontweight='bold', fontsize=14)
            plt.text(0.8, 0.95, '文本Token', ha='center', fontweight='bold', fontsize=14)
            plt.axis('off')

            # 添加文本说明
            if attention_data.get('texts'):
                text_info = f"最匹配文本: {attention_data['texts'][max_sim_idx][:30]}{'...' if len(attention_data['texts'][max_sim_idx]) > 30 else ''}"
                plt.figtext(0.5, 0.01, text_info, ha='center', fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

            print("✓ 显示注意力流向图")
        else:
            plt.text(0.5, 0.5, '需要完整注意力数据\n绘制流向图', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
            print("⚠️ 无法显示注意力流向图 - 数据缺失")
    except Exception as e:
        plt.text(0.5, 0.5, f'绘制出错:\n{str(e)[:50]}...', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        print(f"⚠️ 绘制注意力流向图时出错: {e}")

    # 保存图像
    output_path = Path(output_dir) / "26_attention_flow.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"注意力流向图可视化已保存: {output_path}")

def visualize_confidence_analysis(attention_data, output_dir="outputs"):
    """
    可视化匹配置信度分析

    Args:
        attention_data: 注意力数据
        output_dir: 输出目录
    """
    print("生成匹配置信度分析可视化...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.title('匹配置信度分析', fontsize=16, fontweight='bold')

    try:
        similarities = attention_data.get('similarities')
        if similarities is not None:
            similarities = similarities.numpy()
            texts = attention_data.get('texts', [f'Text{i}' for i in range(len(similarities))])

            # 将NumPy数组元素转换为浮点数
            max_sim = float(similarities.max())
            max_idx = np.argmax(similarities)
            second_max = float(np.partition(similarities, -2)[-2])
            confidence_gap = max_sim - second_max
            entropy = float(-np.sum(similarities * np.log(similarities + 1e-8)))
            normalized_entropy = entropy / np.log(len(similarities))

            metrics = ['最高相似度', '次高相似度', '置信度差距', '熵值']
            values = [max_sim, second_max, confidence_gap, normalized_entropy]
            colors = ['green', 'orange', 'blue', 'purple']

            bars = plt.bar(metrics, values, color=colors, alpha=0.7)
            plt.ylabel('数值', fontsize=12)
            plt.tick_params(axis='x', rotation=45)
            plt.grid(True, alpha=0.3)

            for bar, value in zip(bars, values):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

            # 添加解释
            interpretation = f"最佳匹配: {texts[max_idx][:30]}{'...' if len(texts[max_idx]) > 30 else ''}"
            plt.figtext(0.5, 0.01, interpretation, ha='center', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

            print("✓ 显示匹配置信度分析")
        else:
            plt.text(0.5, 0.5, '需要相似度数据\n进行置信度分析', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
            print("⚠️ 无法显示匹配置信度分析 - 数据缺失")
    except Exception as e:
        plt.text(0.5, 0.5, f'绘制出错:\n{str(e)[:50]}...', ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        print(f"⚠️ 绘制匹配置信度分析时出错: {e}")

    # 保存图像
    output_path = Path(output_dir) / "27_confidence_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"匹配置信度分析可视化已保存: {output_path}")

def create_cross_modal_attention_visualizations(original_image, attention_data, output_dir="outputs"):
    """
    创建所有跨模态注意力相关的可视化

    Args:
        original_image: 原始图像
        attention_data: 注意力数据
        output_dir: 输出目录
    """
    print("生成跨模态注意力对应关系可视化...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    if not attention_data:
        print("⚠️ 无法生成跨模态注意力可视化 - 缺少注意力数据")
        return

    # 1. 图像关注区域
    visualize_image_attention_regions(original_image, attention_data, output_dir)

    # 2. 文本注意力模式
    visualize_text_attention_patterns(attention_data, output_dir)

    # 3. 跨模态相似度矩阵
    visualize_cross_modal_similarity_matrix(attention_data, output_dir)

    # 4. 特征对齐程度
    visualize_feature_alignment(attention_data, output_dir)

    # 5. 注意力流向图
    visualize_attention_flow(attention_data, output_dir)

    # 6. 匹配置信度分析
    visualize_confidence_analysis(attention_data, output_dir)

    print("跨模态注意力对应关系可视化完成")

