import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_image_preprocessing_visualization(original_image, processed_image_tensor, output_dir="outputs"):
    """创建图像预处理过程可视化"""
    print("生成图像预处理可视化...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('图像预处理和Patch分割过程', fontsize=16, fontweight='bold')

    # 1. 原始图像
    ax1 = axes[0, 0]
    ax1.imshow(original_image)
    ax1.set_title(f'原始图像\n尺寸: {original_image.size}', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # 2. 调整大小后的图像
    ax2 = axes[0, 1]
    resized_image = original_image.resize((224, 224))
    ax2.imshow(resized_image)
    ax2.set_title('调整大小后\n224×224', fontsize=14, fontweight='bold')
    ax2.axis('off')

    # 3. 标准化后的图像（反标准化显示）
    ax3 = axes[0, 2]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    denormalized = processed_image_tensor * std + mean
    denormalized = torch.clamp(denormalized, 0, 1)
    denormalized_img = denormalized.permute(1, 2, 0).numpy()

    ax3.imshow(denormalized_img)
    ax3.set_title('标准化后\n(ImageNet标准)', fontsize=14, fontweight='bold')
    ax3.axis('off')

    # 4. Patch分割示意图
    ax4 = axes[1, 0]
    ax4.imshow(resized_image)

    patch_size = 16
    num_patches = 224 // patch_size

    for i in range(num_patches + 1):
        ax4.axhline(y=i * patch_size, color='red', linewidth=1.5, alpha=0.8)
        ax4.axvline(x=i * patch_size, color='red', linewidth=1.5, alpha=0.8)

    ax4.set_title(f'Patch分割\n{num_patches}×{num_patches} = {num_patches ** 2}个patch',
                  fontsize=14, fontweight='bold')
    ax4.set_xlim(0, 224)
    ax4.set_ylim(224, 0)

    # 5. 单个patch示例
    ax5 = axes[1, 1]
    patch_row, patch_col = 7, 7
    patch_img = np.array(resized_image)[
                patch_row * patch_size:(patch_row + 1) * patch_size,
                patch_col * patch_size:(patch_col + 1) * patch_size
                ]

    ax5.imshow(patch_img)
    ax5.set_title(f'单个Patch示例\n位置: ({patch_row}, {patch_col})\n尺寸: {patch_size}×{patch_size}',
                  fontsize=14, fontweight='bold')
    ax5.axis('off')

    # 6. Patch位置编码示意
    ax6 = axes[1, 2]
    position_map = np.zeros((num_patches, num_patches))
    for i in range(num_patches):
        for j in range(num_patches):
            position_map[i, j] = i * num_patches + j

    im = ax6.imshow(position_map, cmap='viridis')
    ax6.set_title('Patch位置编码\n(0-195 + CLS token)', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Patch列')
    ax6.set_ylabel('Patch行')
    plt.colorbar(im, ax=ax6, shrink=0.8)

    plt.tight_layout()
    output_path = Path(output_dir) / "01_image_preprocessing.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图像预处理可视化已保存: {output_path}")
    plt.close()

