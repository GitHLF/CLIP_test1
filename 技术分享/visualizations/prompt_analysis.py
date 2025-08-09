from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_prompt_analysis_visualization(group_results, all_individual_results, attention_data=None, output_dir="outputs"):
    """创建prompt分析和文本注意力可视化"""
    print("生成prompt分析和文本注意力可视化...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    # 调试信息
    if attention_data:
        print("调试信息 - attention_data 内容:")
        print(f"  - keys: {list(attention_data.keys())}")
        if 'texts' in attention_data:
            print(f"  - texts: {attention_data['texts']}")
        if 'text_inputs' in attention_data:
            print(f"  - text_inputs keys: {list(attention_data['text_inputs'].keys())}")
            if 'input_ids' in attention_data['text_inputs']:
                print(f"  - input_ids shape: {attention_data['text_inputs']['input_ids'].shape}")
    else:
        print("调试信息 - attention_data 为空")

    try:
        fig, axes = plt.subplots(3, 4, figsize=(24, 18))
        fig.suptitle('文本处理和注意力机制分析', fontsize=18, fontweight='bold')

        # 第一行：文本预处理和Tokenization
        # 1. 原始文本展示 - 显示所有概念组
        ax1 = axes[0, 0]
        ax1.axis('off')

        # 显示所有概念组的信息，而不仅仅是attention_data中的文本
        text_display = "概念组分析结果:\n\n"
        sorted_groups = sorted(group_results.items(), key=lambda x: x[1]['average_score'], reverse=True)

        for i, (group_name, group_data) in enumerate(sorted_groups[:5]):  # 显示前5个组
            avg_score = group_data['average_score']
            text_display += f"{i+1}. {group_name[:25]}...\n"
            text_display += f"   平均分: {avg_score:.3f}\n"
            text_display += f"   变体数: {len(group_data['prompts'])}\n\n"

        ax1.text(0.05, 0.95, text_display, transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        print(f"✓ 显示 {len(group_results)} 个概念组信息")
        ax1.set_title('1. 概念组分析概览', fontsize=12, fontweight='bold')

        # 2. 详细文本展示 - 显示注意力分析的具体文本
        ax2 = axes[0, 1]
        ax2.axis('off')

        if attention_data and attention_data.get('texts'):
            texts = attention_data['texts']
            detailed_text_display = "注意力分析文本:\n\n"
            for i, text in enumerate(texts):
                detailed_text_display += f"{i+1}. {text}\n\n"
            ax2.text(0.05, 0.95, detailed_text_display, transform=ax2.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            print(f"✓ 显示 {len(texts)} 个注意力分析文本")
        else:
            ax2.text(0.5, 0.5, '需要注意力数据\n显示具体分析文本', ha='center', va='center', transform=ax2.transAxes)
            print("⚠️ 无法显示注意力分析文本 - 数据缺失")
        ax2.set_title('2. 注意力分析文本', fontsize=12, fontweight='bold')

        # 3. Tokenization过程
        ax3 = axes[0, 2]
        if attention_data and attention_data.get('text_inputs') and 'attention_mask' in attention_data['text_inputs']:
            text_inputs = attention_data['text_inputs']
            token_lengths = text_inputs['attention_mask'].sum(dim=1).numpy()
            ax3.bar(range(len(token_lengths)), token_lengths, color='skyblue')
            ax3.set_title('3. Token长度分布', fontsize=12, fontweight='bold')
            ax3.set_xlabel('文本索引')
            ax3.set_ylabel('Token数量')
            for i, length in enumerate(token_lengths):
                ax3.text(i, length + 0.5, str(int(length)), ha='center', va='bottom')
            print(f"✓ 显示token长度分布: {token_lengths}")
        else:
            ax3.text(0.5, 0.5, '需要text_inputs数据\n(tokenization结果)', ha='center', va='center', transform=ax3.transAxes)
            print("⚠️ 无法显示token长度 - text_inputs缺失")
            ax3.set_title('3. Token长度分布', fontsize=12, fontweight='bold')

        # 4. 所有prompts的分数分布
        ax4 = axes[0, 3]
        all_scores = [result[1] for result in all_individual_results]
        ax4.hist(all_scores, bins=15, color='orange', alpha=0.7, edgecolor='black')
        ax4.set_title(f'4. 所有Prompts分数分布\n(总计{len(all_scores)}个)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('匹配分数')
        ax4.set_ylabel('频次')
        ax4.axvline(np.mean(all_scores), color='red', linestyle='--', linewidth=2, label=f'平均值: {np.mean(all_scores):.3f}')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        print(f"✓ 显示 {len(all_scores)} 个prompts的分数分布")

        # 第二行：文本注意力机制详细分析
        if attention_data and attention_data.get('text_attentions'):
            text_attentions = attention_data['text_attentions']
            print(f"✓ 处理 {len(text_attentions)} 层文本注意力")

            # 5. 早期层文本注意力
            ax5 = axes[1, 0]
            if len(text_attentions) > 1:
                early_text_att = text_attentions[1][0, 0, :20, :20].numpy()
                im5 = ax5.imshow(early_text_att, cmap='Blues', interpolation='nearest')
                ax5.set_title('5. 早期层文本注意力\n(第2层，前20个token)', fontsize=12, fontweight='bold')
                ax5.set_xlabel('Key位置')
                ax5.set_ylabel('Query位置')
                plt.colorbar(im5, ax=ax5, shrink=0.6)
            else:
                ax5.text(0.5, 0.5, '层数不足', ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title('5. 早期层文本注意力', fontsize=12, fontweight='bold')

            # 6. 中期层文本注意力
            ax6 = axes[1, 1]
            if len(text_attentions) > 5:
                mid_text_att = text_attentions[5][0, 0, :20, :20].numpy()
                im6 = ax6.imshow(mid_text_att, cmap='Blues', interpolation='nearest')
                ax6.set_title('6. 中期层文本注意力\n(第6层，前20个token)', fontsize=12, fontweight='bold')
                ax6.set_xlabel('Key位置')
                ax6.set_ylabel('Query位置')
                plt.colorbar(im6, ax=ax6, shrink=0.6)
            else:
                ax6.text(0.5, 0.5, '层数不足', ha='center', va='center', transform=ax6.transAxes)
                ax6.set_title('6. 中期层文本注意力', fontsize=12, fontweight='bold')

            # 7. 最终层文本注意力
            ax7 = axes[1, 2]
            final_text_att = text_attentions[-1][0, 0, :20, :20].numpy()
            im7 = ax7.imshow(final_text_att, cmap='Blues', interpolation='nearest')
            ax7.set_title('7. 最终层文本注意力\n(最后层，前20个token)', fontsize=12, fontweight='bold')
            ax7.set_xlabel('Key位置')
            ax7.set_ylabel('Query位置')
            plt.colorbar(im7, ax=ax7, shrink=0.6)

            # 8. 多头注意力对比
            ax8 = axes[1, 3]
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

            ax8.bar(x - width/2, head_patterns[:, 0], width, label='对角线注意力', alpha=0.8)
            ax8.bar(x + width/2, head_patterns[:, 1], width, label='非对角线注意力', alpha=0.8)
            ax8.set_title('8. 多头注意力模式', fontsize=12, fontweight='bold')
            ax8.set_xlabel('注意力头')
            ax8.set_ylabel('注意力强度')
            ax8.set_xticks(x)
            ax8.set_xticklabels([f'头{i+1}' for i in range(num_heads)])
            ax8.legend()
            ax8.grid(True, alpha=0.3)

        else:
            # 如果没有文本注意力数据
            for i, ax in enumerate([axes[1, 0], axes[1, 1], axes[1, 2], axes[1, 3]]):
                ax.text(0.5, 0.5, f'需要text_attentions\n注意力数据\n({5+i})',
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{5+i}. 文本注意力分析', fontsize=12, fontweight='bold')
            print("⚠️ 无法显示文本注意力分析 - text_attentions缺失")

        # 第三行：跨模态匹配分析
        # 9. 概念组匹配结果
        ax9 = axes[2, 0]
        group_names = list(group_results.keys())
        avg_scores = [group_results[name]['average_score'] for name in group_names]
        colors = plt.cm.RdYlGn(np.linspace(0.3, 1, len(group_names)))

        bars = ax9.bar(range(len(group_names)), avg_scores, color=colors)
        ax9.set_title('9. 概念组匹配结果', fontsize=12, fontweight='bold')
        ax9.set_xlabel('概念组')
        ax9.set_ylabel('平均匹配分数')
        ax9.set_xticks(range(len(group_names)))
        ax9.set_xticklabels([name[:10] + '...' if len(name) > 10 else name
                            for name in group_names], rotation=45, ha='right')

        for bar, score in zip(bars, avg_scores):
            height = bar.get_height()
            ax9.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

        # 10. 图文特征相似度矩阵
        ax10 = axes[2, 1]
        if attention_data and attention_data.get('similarities') is not None:
            similarities = attention_data['similarities'][0].numpy()

            sim_matrix = similarities.reshape(1, -1)
            im10 = ax10.imshow(sim_matrix, cmap='RdYlGn', aspect='auto')
            ax10.set_title('10. 图文相似度分布', fontsize=12, fontweight='bold')
            ax10.set_xlabel('文本索引')
            ax10.set_yticks([])
            ax10.set_ylabel('图像')

            for i, sim in enumerate(similarities):
                ax10.text(i, 0, f'{sim:.3f}', ha='center', va='center',
                         fontweight='bold', fontsize=8,
                         color='white' if sim < 0.5 else 'black')

            plt.colorbar(im10, ax=ax10, shrink=0.6)
            print(f"✓ 显示相似度分布: {similarities}")
        else:
            ax10.text(0.5, 0.5, '需要similarities数据', ha='center', va='center', transform=ax10.transAxes)
            print("⚠️ 无法显示相似度分布 - similarities缺失")
            ax10.set_title('10. 图文相似度分布', fontsize=12, fontweight='bold')

        # 11. 特征空间投影（使用简单方法）
        ax11 = axes[2, 2]
        try:
            print("开始特征空间投影...")

            if (attention_data and attention_data.get('image_embeds') is not None
                and attention_data.get('text_embeds') is not None):

                # 安全地提取特征
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
                    ax11.scatter(image_2d[0], image_2d[1], c='red', s=100, marker='*',
                                label='图像特征', zorder=5)

                    # 绘制文本特征点
                    for i in range(len(text_2d)):
                        ax11.scatter(text_2d[i, 0], text_2d[i, 1], c='blue', s=60, alpha=0.7)
                        ax11.annotate(f'T{i+1}', (text_2d[i, 0], text_2d[i, 1]),
                                     xytext=(5, 5), textcoords='offset points', fontsize=8)

                    # 绘制连接线（相似度）
                    if attention_data.get('similarities') is not None:
                        similarities = attention_data['similarities'][0].numpy()
                        for i in range(len(text_2d)):
                            alpha = similarities[i]
                            ax11.plot([image_2d[0], text_2d[i, 0]],
                                     [image_2d[1], text_2d[i, 1]],
                                     'gray', alpha=alpha, linewidth=2)

                    ax11.set_title('11. 特征空间投影\n(前两个维度)', fontsize=12, fontweight='bold')
                    ax11.set_xlabel('特征维度1')
                    ax11.set_ylabel('特征维度2')
                    ax11.legend()
                    ax11.grid(True, alpha=0.3)
                    print("✓ 特征空间投影完成")

                else:
                    ax11.text(0.5, 0.5, f'特征维度不足\n需要至少2维', ha='center', va='center', transform=ax11.transAxes)
                    ax11.set_title('11. 特征空间投影', fontsize=12, fontweight='bold')

            else:
                ax11.text(0.5, 0.5, '需要image_embeds和\ntext_embeds数据', ha='center', va='center', transform=ax11.transAxes)
                print("⚠️ 无法显示特征投影 - 嵌入数据缺失")
                ax11.set_title('11. 特征空间投影', fontsize=12, fontweight='bold')

        except Exception as e:
            print(f"⚠️ 特征空间投影出错: {e}")
            ax11.text(0.5, 0.5, f'特征投影出错:\n{str(e)[:30]}...', ha='center', va='center', transform=ax11.transAxes, fontsize=10)
            ax11.set_title('11. 特征空间投影', fontsize=12, fontweight='bold')

        # 12. Top匹配详细分析
        ax12 = axes[2, 3]
        try:
            sorted_results = sorted(all_individual_results, key=lambda x: x[1], reverse=True)
            top_8_results = sorted_results[:8]  # 显示更多结果

            prompts_top8 = [r[0] for r in top_8_results]
            scores_top8 = [r[1] for r in top_8_results]

            bars = ax12.barh(range(len(prompts_top8)), scores_top8,
                            color=plt.cm.viridis(np.linspace(0, 1, len(prompts_top8))))

            ax12.set_title('12. Top-8 最佳匹配', fontsize=12, fontweight='bold')
            ax12.set_xlabel('匹配分数')
            ax12.set_yticks(range(len(prompts_top8)))
            ax12.set_yticklabels([f"{i+1}. {p[:15]}..." if len(p) > 15 else f"{i+1}. {p}"
                                 for i, p in enumerate(prompts_top8)], fontsize=8)

            for i, (bar, score) in enumerate(zip(bars, scores_top8)):
                ax12.text(score + 0.001, i, f'{score:.4f}',
                         va='center', fontsize=7, fontweight='bold')

            print("✓ Top匹配分析完成")

        except Exception as e:
            print(f"⚠️ Top匹配分析出错: {e}")
            ax12.text(0.5, 0.5, f'Top匹配分析出错:\n{str(e)[:30]}...', ha='center', va='center', transform=ax12.transAxes)
            ax12.set_title('12. Top匹配分析', fontsize=12, fontweight='bold')

        plt.tight_layout()
        output_path = Path(output_dir) / "02_text_attention_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"文本注意力分析可视化已保存: {output_path}")

    except Exception as e:
        print(f"创建文本分析可视化时出错: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 确保图形被关闭和内存被释放
        plt.close('all')
        import gc
        gc.collect()
        print("✓ 内存清理完成")

