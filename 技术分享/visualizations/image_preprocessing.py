from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def get_patch_info_from_attention(vision_attentions):
    """从注意力权重推断patch信息"""
    if vision_attentions and len(vision_attentions) > 0:
        # 获取序列长度（包含CLS token）
        seq_len = vision_attentions[0].shape[-1]
        num_patches_total = seq_len - 1  # 减去CLS token

        # 计算patch网格大小
        num_patches_per_side = int(np.sqrt(num_patches_total))

        # 根据patch数量推断patch大小
        if num_patches_per_side == 14:  # 14x14 = 196 patches
            patch_size = 224 // 14  # 16
        elif num_patches_per_side == 16:  # 16x16 = 256 patches
            patch_size = 224 // 16  # 14
        else:
            # 默认值
            patch_size = 224 // num_patches_per_side if num_patches_per_side > 0 else 16

        print(f"检测到patch配置: {num_patches_per_side}x{num_patches_per_side} patches, patch_size={patch_size}")
        return num_patches_per_side, patch_size
    else:
        # 默认配置
        return 14, 16

def create_image_preprocessing_visualization(original_image, processed_image_tensor, attention_data=None, output_dir="outputs"):
    """创建图像预处理和Vision Transformer处理过程可视化"""
    print("生成图像预处理和处理过程可视化...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    # 从注意力数据推断patch配置
    if attention_data and attention_data.get('vision_attentions'):
        num_patches, patch_size = get_patch_info_from_attention(attention_data['vision_attentions'])
    else:
        # 默认配置（ViT-Base）
        num_patches, patch_size = 14, 16
        print(f"使用默认patch配置: {num_patches}x{num_patches} patches, patch_size={patch_size}")

    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    fig.suptitle('图像预处理和Vision Transformer处理过程', fontsize=18, fontweight='bold')

    # 第一行：预处理过程
    # 1. 原始图像
    ax1 = axes[0, 0]
    ax1.imshow(original_image)
    ax1.set_title(f'1. 原始图像\n尺寸: {original_image.size}', fontsize=12, fontweight='bold')
    ax1.axis('off')

    # 2. 调整大小后的图像
    ax2 = axes[0, 1]
    resized_image = original_image.resize((224, 224))
    ax2.imshow(resized_image)
    ax2.set_title('2. 调整大小\n224×224', fontsize=12, fontweight='bold')
    ax2.axis('off')

    # 3. 标准化后的图像
    ax3 = axes[0, 2]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    denormalized = processed_image_tensor * std + mean
    denormalized = torch.clamp(denormalized, 0, 1)
    denormalized_img = denormalized.permute(1, 2, 0).numpy()
    ax3.imshow(denormalized_img)
    ax3.set_title('3. ImageNet标准化', fontsize=12, fontweight='bold')
    ax3.axis('off')

    # 4. Patch分割示意图
    ax4 = axes[0, 3]
    ax4.imshow(resized_image)
    for i in range(num_patches + 1):
        ax4.axhline(y=i * patch_size, color='red', linewidth=1.5, alpha=0.8)
        ax4.axvline(x=i * patch_size, color='red', linewidth=1.5, alpha=0.8)
    ax4.set_title(f'4. Patch分割\n{num_patches}×{num_patches}={num_patches**2}个patch', fontsize=12, fontweight='bold')
    ax4.set_xlim(0, 224)
    ax4.set_ylim(224, 0)

    # 第二行：Vision Transformer处理过程
    # 5. Patch Embedding过程
    ax5 = axes[1, 0]
    # 显示几个patch的示例
    positions = [(3, 3), (7, 7), (10, 10)]  # 选择几个不同位置的patch
    colors = ['red', 'blue', 'green']

    ax5.imshow(resized_image, alpha=0.7)
    for i, ((row, col), color) in enumerate(zip(positions, colors)):
        # 确保patch位置在有效范围内
        if row < num_patches and col < num_patches:
            # 绘制patch边框
            rect = plt.Rectangle((col*patch_size, row*patch_size), patch_size, patch_size,
                               linewidth=3, edgecolor=color, facecolor='none')
            ax5.add_patch(rect)
            ax5.text(col*patch_size + patch_size//2, row*patch_size + patch_size//2, f'P{i+1}',
                    color=color, fontweight='bold', fontsize=10, ha='center', va='center')

    ax5.set_title('5. Patch Embedding\n选中的patch示例', fontsize=12, fontweight='bold')
    ax5.set_xlim(0, 224)
    ax5.set_ylim(224, 0)

    # 6. 位置编码可视化
    ax6 = axes[1, 1]
    position_map = np.zeros((num_patches, num_patches))
    for i in range(num_patches):
        for j in range(num_patches):
            position_map[i, j] = i * num_patches + j

    im6 = ax6.imshow(position_map, cmap='viridis')
    ax6.set_title(f'6. 位置编码\n每个patch的位置ID', fontsize=12, fontweight='bold')
    plt.colorbar(im6, ax=ax6, shrink=0.6)

    # 7. CLS Token + Patches序列
    ax7 = axes[1, 2]
    # 创建序列示意图
    sequence_length = num_patches**2 + 1  # patches + 1 CLS
    sequence_visual = np.zeros((1, min(sequence_length, 100)))  # 限制显示长度
    sequence_visual[0, 0] = 1  # CLS token标记为1
    sequence_visual[0, 1:] = 0.5  # patches标记为0.5

    im7 = ax7.imshow(sequence_visual, cmap='RdYlBu', aspect='auto')
    ax7.set_title(f'7. 输入序列\nCLS + {num_patches**2}个Patches', fontsize=12, fontweight='bold')
    ax7.set_xlabel('序列位置')
    ax7.set_yticks([])
    ax7.text(0, 0, 'CLS', ha='center', va='center', fontweight='bold', color='white', fontsize=8)
    if sequence_visual.shape[1] > 20:
        ax7.text(sequence_visual.shape[1]//2, 0, 'Patches...', ha='center', va='center', fontweight='bold', fontsize=8)

    # 8. Transformer层处理示意
    ax8 = axes[1, 3]
    if attention_data and attention_data.get('vision_attentions'):
        # 显示不同层的CLS token特征变化
        vision_attentions = attention_data['vision_attentions']
        layer_features = []

        for layer_idx in range(min(6, len(vision_attentions))):  # 显示前6层
            # 计算CLS token的平均注意力强度
            cls_attention = vision_attentions[layer_idx][0, :, 0, :]  # [heads, seq_len]
            avg_attention = cls_attention[:, 1:].mean().item()  # 对patches的平均注意力
            layer_features.append(avg_attention)

        ax8.plot(range(1, len(layer_features)+1), layer_features, 'o-', linewidth=2, markersize=8)
        ax8.set_title('8. Transformer层处理\nCLS token注意力变化', fontsize=12, fontweight='bold')
        ax8.set_xlabel('Transformer层')
        ax8.set_ylabel('平均注意力强度')
        ax8.grid(True, alpha=0.3)
    else:
        ax8.text(0.5, 0.5, '需要注意力数据\n才能显示', ha='center', va='center',
                transform=ax8.transAxes, fontsize=12)
        ax8.set_title('8. Transformer层处理', fontsize=12, fontweight='bold')

    # 第三行：注意力可视化
    if attention_data and attention_data.get('vision_attentions'):
        vision_attentions = attention_data['vision_attentions']

        # 9. 早期层注意力（第2层）
        ax9 = axes[2, 0]
        if len(vision_attentions) > 1:
            early_attention = vision_attentions[1][0, :, 0, 1:].mean(dim=0)  # 第2层，CLS对patches的注意力

            # 检查注意力向量长度是否匹配
            expected_length = num_patches * num_patches
            if len(early_attention) == expected_length:
                attention_map = early_attention.reshape(num_patches, num_patches).numpy()
                im9 = ax9.imshow(attention_map, cmap='hot', interpolation='bilinear')
                ax9.set_title('9. 早期层注意力\n(第2层CLS→Patches)', fontsize=12, fontweight='bold')
                plt.colorbar(im9, ax=ax9, shrink=0.6)
            else:
                ax9.text(0.5, 0.5, f'注意力维度不匹配\n期望:{expected_length}, 实际:{len(early_attention)}',
                        ha='center', va='center', transform=ax9.transAxes, fontsize=10)
                ax9.set_title('9. 早期层注意力', fontsize=12, fontweight='bold')
        else:
            ax9.text(0.5, 0.5, '层数不足', ha='center', va='center', transform=ax9.transAxes)
            ax9.set_title('9. 早期层注意力', fontsize=12, fontweight='bold')

        # 10. 中期层注意力（第6层）
        ax10 = axes[2, 1]
        if len(vision_attentions) > 5:
            mid_attention = vision_attentions[5][0, :, 0, 1:].mean(dim=0)  # 第6层

            expected_length = num_patches * num_patches
            if len(mid_attention) == expected_length:
                attention_map = mid_attention.reshape(num_patches, num_patches).numpy()
                im10 = ax10.imshow(attention_map, cmap='hot', interpolation='bilinear')
                ax10.set_title('10. 中期层注意力\n(第6层CLS→Patches)', fontsize=12, fontweight='bold')
                plt.colorbar(im10, ax=ax10, shrink=0.6)
            else:
                ax10.text(0.5, 0.5, f'注意力维度不匹配\n期望:{expected_length}, 实际:{len(mid_attention)}',
                         ha='center', va='center', transform=ax10.transAxes, fontsize=10)
                ax10.set_title('10. 中期层注意力', fontsize=12, fontweight='bold')
        else:
            ax10.text(0.5, 0.5, '层数不足', ha='center', va='center', transform=ax10.transAxes)
            ax10.set_title('10. 中期层注意力', fontsize=12, fontweight='bold')

        # 11. 最终层注意力（最后一层）
        ax11 = axes[2, 2]
        final_attention = vision_attentions[-1][0, :, 0, 1:].mean(dim=0)  # 最后一层

        expected_length = num_patches * num_patches
        if len(final_attention) == expected_length:
            attention_map = final_attention.reshape(num_patches, num_patches).numpy()
            im11 = ax11.imshow(attention_map, cmap='hot', interpolation='bilinear')
            ax11.set_title('11. 最终层注意力\n(最后层CLS→Patches)', fontsize=12, fontweight='bold')
            plt.colorbar(im11, ax=ax11, shrink=0.6)

            # 12. 注意力叠加在原图上
            ax12 = axes[2, 3]
            ax12.imshow(resized_image, alpha=0.7)
            # 将注意力图调整到原图尺寸
            attention_resized = np.kron(attention_map, np.ones((patch_size, patch_size)))
            ax12.imshow(attention_resized, cmap='hot', alpha=0.5, extent=[0, 224, 224, 0])
            ax12.set_title('12. 注意力叠加图\n最关注的区域', fontsize=12, fontweight='bold')
            ax12.set_xlim(0, 224)
            ax12.set_ylim(224, 0)

            # 标注最关注的区域
            max_attention_idx = final_attention.argmax().item()
            max_row = max_attention_idx // num_patches
            max_col = max_attention_idx % num_patches
            rect = plt.Rectangle((max_col*patch_size, max_row*patch_size), patch_size, patch_size,
                               linewidth=3, edgecolor='yellow', facecolor='none')
            ax12.add_patch(rect)
            ax12.text(max_col*patch_size + patch_size//2, max_row*patch_size + patch_size//2, 'MAX',
                     color='yellow', fontweight='bold', fontsize=10, ha='center', va='center')
        else:
            ax11.text(0.5, 0.5, f'注意力维度不匹配\n期望:{expected_length}, 实际:{len(final_attention)}',
                     ha='center', va='center', transform=ax11.transAxes, fontsize=10)
            ax11.set_title('11. 最终层注意力', fontsize=12, fontweight='bold')

            # 12. 占位符
            ax12 = axes[2, 3]
            ax12.text(0.5, 0.5, '需要有效的\n注意力数据', ha='center', va='center', transform=ax12.transAxes)
            ax12.set_title('12. 注意力叠加图', fontsize=12, fontweight='bold')

    else:
        # 如果没有注意力数据，显示占位符
        for i, ax in enumerate([axes[2, 0], axes[2, 1], axes[2, 2], axes[2, 3]]):
            ax.text(0.5, 0.5, f'需要注意力数据\n才能显示\n({9+i})',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{9+i}. 注意力可视化', fontsize=12, fontweight='bold')

    plt.tight_layout()
    output_path = Path(output_dir) / "01_image_processing_detailed.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"详细图像处理可视化已保存: {output_path}")
    plt.close()

