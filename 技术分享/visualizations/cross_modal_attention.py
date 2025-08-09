from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_cross_modal_attention_visualization(original_image, attention_data, output_dir="outputs"):
    """创建跨模态注意力对应关系可视化"""
    print("生成跨模态注意力对应关系可视化...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    try:
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        fig.suptitle('跨模态注意力对应关系分析', fontsize=18, fontweight='bold')

        if not attention_data:
            for ax in axes.flat:
                ax.text(0.5, 0.5, '需要注意力数据', ha='center', va='center', transform=ax.transAxes)
            plt.tight_layout()
            output_path = Path(output_dir) / "03_cross_modal_attention.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"跨模态注意力可视化已保存: {output_path}")
            plt.close()
            return

        # 1. 图像区域与文本token对应关系
        ax1 = axes[0, 0]
        try:
            if attention_data.get('vision_attentions') and attention_data.get('text_attentions'):
                # 获取图像最关注的区域
                vision_att = attention_data['vision_attentions'][-1][0, :, 0, 1:].mean(dim=0)

                # 动态计算patch配置
                num_patches_total = len(vision_att)
                num_patches = int(np.sqrt(num_patches_total))
                patch_size = 224 // num_patches if num_patches > 0 else 16

                # 显示原图
                resized_image = original_image.resize((224, 224))
                ax1.imshow(resized_image, alpha=0.7)

                # 叠加注意力热力图
                if num_patches * num_patches == num_patches_total:
                    attention_map = vision_att.reshape(num_patches, num_patches).numpy()
                    attention_resized = np.kron(attention_map, np.ones((patch_size, patch_size)))
                    ax1.imshow(attention_resized, cmap='hot', alpha=0.5, extent=[0, 224, 224, 0])

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
                                ax1.add_patch(rect)
                                ax1.text(col*patch_size + 8, row*patch_size + 8, f'R{i+1}',
                                        color=colors[i], fontweight='bold', fontsize=12)

                ax1.set_title('1. 图像关注区域\n(Vision Transformer)', fontsize=12, fontweight='bold')
                ax1.set_xlim(0, 224)
                ax1.set_ylim(224, 0)
            else:
                ax1.text(0.5, 0.5, '需要视觉注意力数据', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('1. 图像关注区域', fontsize=12, fontweight='bold')
        except Exception as e:
            print(f"绘制图像关注区域时出错: {e}")
            ax1.text(0.5, 0.5, f'绘制出错:\n{str(e)[:50]}...', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('1. 图像关注区域', fontsize=12, fontweight='bold')

        # 2. 文本关注模式
        ax2 = axes[0, 1]
        try:
            if attention_data.get('text_attentions') and attention_data.get('texts'):
                text_att = attention_data['text_attentions'][-1][0, 0, :, :].numpy()
                text_length = attention_data['text_inputs']['attention_mask'][0].sum().item()

                valid_att = text_att[:text_length, :text_length]
                im2 = ax2.imshow(valid_att, cmap='Blues', interpolation='nearest')
                ax2.set_title(f'2. 文本注意力模式\n"{attention_data["texts"][0][:20]}..."', fontsize=12, fontweight='bold')
                ax2.set_xlabel('Key Token位置')
                ax2.set_ylabel('Query Token位置')
                plt.colorbar(im2, ax=ax2, shrink=0.8)

                important_tokens = np.unravel_index(np.argpartition(valid_att.ravel(), -3)[-3:], valid_att.shape)
                for i, (row, col) in enumerate(zip(important_tokens[0], important_tokens[1])):
                    ax2.scatter(col, row, c='red', s=100, alpha=0.7, marker='x')
            else:
                ax2.text(0.5, 0.5, '需要文本注意力数据', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('2. 文本注意力模式', fontsize=12, fontweight='bold')
        except Exception as e:
            print(f"绘制文本注意力模式时出错: {e}")
            ax2.text(0.5, 0.5, f'绘制出错:\n{str(e)[:50]}...', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('2. 文本注意力模式', fontsize=12, fontweight='bold')

        # 3. 跨模态相似度热力图
        ax3 = axes[0, 2]
        try:
            similarities = attention_data.get('similarities')
            if similarities is not None:  # 修复：正确检查tensor
                similarities = similarities[0].numpy()
                texts = attention_data.get('texts', [f'Text{i}' for i in range(len(similarities))])

                sim_matrix = similarities.reshape(-1, 1).T
                im3 = ax3.imshow(sim_matrix, cmap='RdYlGn', aspect='auto')

                ax3.set_title('3. 图文相似度矩阵', fontsize=12, fontweight='bold')
                ax3.set_xlabel('文本索引')
                ax3.set_ylabel('图像')
                ax3.set_xticks(range(len(texts)))
                ax3.set_xticklabels([f'T{i+1}' for i in range(len(texts))])
                ax3.set_yticks([0])
                ax3.set_yticklabels(['Image'])

                for i, sim in enumerate(similarities):
                    color = 'white' if sim < 0.5 else 'black'
                    ax3.text(i, 0, f'{sim:.3f}', ha='center', va='center',
                            fontweight='bold', color=color)

                plt.colorbar(im3, ax=ax3, shrink=0.8)
            else:
                ax3.text(0.5, 0.5, '需要相似度数据', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('3. 图文相似度矩阵', fontsize=12, fontweight='bold')
        except Exception as e:
            print(f"绘制相似度矩阵时出错: {e}")
            ax3.text(0.5, 0.5, f'绘制出错:\n{str(e)[:50]}...', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('3. 图文相似度矩阵', fontsize=12, fontweight='bold')

        # 4. 特征对齐分析
        ax4 = axes[1, 0]
        try:
            image_embeds = attention_data.get('image_embeds')
            text_embeds = attention_data.get('text_embeds')
            if image_embeds is not None and text_embeds is not None:  # 修复：正确检查tensor
                image_embed = image_embeds[0].numpy()
                text_embeds_np = text_embeds.numpy()

                correlations = []
                for text_embed in text_embeds_np:
                    corr = np.corrcoef(image_embed, text_embed)[0, 1]
                    correlations.append(corr)

                texts = attention_data.get('texts', [f'Text{i}' for i in range(len(correlations))])
                colors = plt.cm.RdYlGn(np.linspace(0.3, 1, len(correlations)))

                bars = ax4.bar(range(len(correlations)), correlations, color=colors)
                ax4.set_title('4. 特征对齐程度\n(图像-文本特征相关性)', fontsize=12, fontweight='bold')
                ax4.set_xlabel('文本索引')
                ax4.set_ylabel('特征相关性')
                ax4.set_xticks(range(len(texts)))
                ax4.set_xticklabels([f'T{i+1}' for i in range(len(texts))])
                ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax4.grid(True, alpha=0.3)

                for bar, corr in zip(bars, correlations):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height > 0 else height - 0.03,
                            f'{corr:.3f}', ha='center', va='bottom' if height > 0 else 'top',
                            fontweight='bold', fontsize=9)
            else:
                ax4.text(0.5, 0.5, '需要特征嵌入数据', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('4. 特征对齐程度', fontsize=12, fontweight='bold')
        except Exception as e:
            print(f"绘制特征对齐分析时出错: {e}")
            ax4.text(0.5, 0.5, f'绘制出错:\n{str(e)[:50]}...', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('4. 特征对齐程度', fontsize=12, fontweight='bold')

        # 5. 注意力流向图
        ax5 = axes[1, 1]
        try:
            if (attention_data.get('vision_attentions') and attention_data.get('text_attentions')
                and attention_data.get('similarities') is not None):

                vision_att = attention_data['vision_attentions'][-1][0, :, 0, 1:].mean(dim=0)
                top_patches = torch.topk(vision_att, 3).indices.numpy()

                text_att = attention_data['text_attentions'][-1][0, 0, :, :].numpy()
                text_length = attention_data['text_inputs']['attention_mask'][0].sum().item()
                avg_att_per_token = text_att[:text_length, :text_length].mean(axis=1)
                top_tokens = np.argsort(avg_att_per_token)[-3:]

                for i, patch in enumerate(top_patches):
                    y_pos = 0.8 - i * 0.3
                    ax5.scatter(0.2, y_pos, s=200, c='red', alpha=0.7)
                    ax5.text(0.1, y_pos, f'区域{i+1}', ha='right', va='center', fontweight='bold')

                for i, token in enumerate(top_tokens):
                    y_pos = 0.8 - i * 0.3
                    ax5.scatter(0.8, y_pos, s=200, c='blue', alpha=0.7)
                    ax5.text(0.9, y_pos, f'Token{token}', ha='left', va='center', fontweight='bold')

                similarities = attention_data['similarities'][0].numpy()
                max_sim_idx = np.argmax(similarities)

                for i in range(min(3, len(top_patches))):
                    for j in range(min(3, len(top_tokens))):
                        y1 = 0.8 - i * 0.3
                        y2 = 0.8 - j * 0.3
                        alpha = similarities[max_sim_idx] if i == 0 and j == 0 else 0.3
                        ax5.plot([0.2, 0.8], [y1, y2], 'gray', alpha=alpha, linewidth=2)

                ax5.set_xlim(0, 1)
                ax5.set_ylim(0, 1)
                ax5.set_title('5. 跨模态注意力流向', fontsize=12, fontweight='bold')
                ax5.text(0.2, 0.95, '图像区域', ha='center', fontweight='bold', fontsize=12)
                ax5.text(0.8, 0.95, '文本Token', ha='center', fontweight='bold', fontsize=12)
                ax5.axis('off')
            else:
                ax5.text(0.5, 0.5, '需要完整注意力数据\n绘制流向图', ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title('5. 跨模态注意力流向', fontsize=12, fontweight='bold')
        except Exception as e:
            print(f"绘制注意力流向图时出错: {e}")
            ax5.text(0.5, 0.5, f'绘制出错:\n{str(e)[:50]}...', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('5. 跨模态注意力流向', fontsize=12, fontweight='bold')

        # 6. 匹配置信度分析
        ax6 = axes[1, 2]
        try:
            similarities = attention_data.get('similarities')
            if similarities is not None:  # 修复：正确检查tensor
                similarities = similarities[0].numpy()
                texts = attention_data.get('texts', [f'Text{i}' for i in range(len(similarities))])

                max_sim = similarities.max()
                second_max = np.partition(similarities, -2)[-2]
                confidence_gap = max_sim - second_max
                entropy = -np.sum(similarities * np.log(similarities + 1e-8))

                metrics = ['最高相似度', '次高相似度', '置信度差距', '熵值']
                values = [max_sim, second_max, confidence_gap, entropy/np.log(len(similarities))]
                colors = ['green', 'orange', 'blue', 'purple']

                bars = ax6.bar(metrics, values, color=colors, alpha=0.7)
                ax6.set_title('6. 匹配置信度分析', fontsize=12, fontweight='bold')
                ax6.set_ylabel('数值')
                ax6.tick_params(axis='x', rotation=45)

                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

                ax6.grid(True, alpha=0.3)

                interpretation = f"最佳匹配: {texts[np.argmax(similarities)][:20]}..."
                ax6.text(0.5, -0.15, interpretation, ha='center', va='top', transform=ax6.transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"), fontsize=10)
            else:
                ax6.text(0.5, 0.5, '需要相似度数据\n进行置信度分析', ha='center', va='center', transform=ax6.transAxes)
                ax6.set_title('6. 匹配置信度分析', fontsize=12, fontweight='bold')
        except Exception as e:
            print(f"绘制置信度分析时出错: {e}")
            ax6.text(0.5, 0.5, f'绘制出错:\n{str(e)[:50]}...', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('6. 匹配置信度分析', fontsize=12, fontweight='bold')

        plt.tight_layout()
        output_path = Path(output_dir) / "03_cross_modal_attention.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"跨模态注意力可视化已保存: {output_path}")

    except Exception as e:
        print(f"创建跨模态注意力可视化时出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        plt.close()

