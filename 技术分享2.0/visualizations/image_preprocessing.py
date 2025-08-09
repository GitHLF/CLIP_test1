from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def visualize_original_image(image, output_dir="outputs"):
    """
    可视化原始图像

    Args:
        image: 原始图像
        output_dir: 输出目录
    """
    print("生成原始图像可视化...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    plt.figure(figsize=(8, 8))
    plt.title('原始输入图像', fontsize=16, fontweight='bold')
    plt.imshow(image)
    plt.axis('off')

    # 保存图像
    output_path = Path(output_dir) / "15_original_image.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"原始图像可视化已保存: {output_path}")

def visualize_preprocessing_steps(image, output_dir="outputs"):
    """
    可视化图像预处理步骤

    Args:
        image: 原始图像
        output_dir: 输出目录
    """
    print("生成图像预处理步骤可视化...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    plt.figure(figsize=(15, 5))
    plt.suptitle('图像预处理步骤', fontsize=16, fontweight='bold')

    # 1. 原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('1. 原始图像', fontsize=12)
    plt.axis('off')

    # 2. 调整大小
    resize_transform = transforms.Resize(224)
    resized_image = resize_transform(image)
    plt.subplot(1, 3, 2)
    plt.imshow(resized_image)
    plt.title('2. 调整大小 (224x224)', fontsize=12)
    plt.axis('off')

    # 3. 中心裁剪
    crop_transform = transforms.CenterCrop(224)
    cropped_image = crop_transform(resized_image)
    plt.subplot(1, 3, 3)
    plt.imshow(cropped_image)
    plt.title('3. 中心裁剪', fontsize=12)
    plt.axis('off')

    # 保存图像
    output_path = Path(output_dir) / "16_preprocessing_steps.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图像预处理步骤可视化已保存: {output_path}")

def visualize_normalization(image, output_dir="outputs"):
    """
    可视化图像归一化

    Args:
        image: 原始图像
        output_dir: 输出目录
    """
    print("生成图像归一化可视化...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    plt.figure(figsize=(15, 5))
    plt.suptitle('图像归一化过程', fontsize=16, fontweight='bold')

    # 1. 调整大小和裁剪
    resize_crop = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224)
    ])
    resized_image = resize_crop(image)
    plt.subplot(1, 3, 1)
    plt.imshow(resized_image)
    plt.title('1. 调整大小和裁剪', fontsize=12)
    plt.axis('off')

    # 2. 转换为张量
    to_tensor = transforms.ToTensor()
    tensor_image = to_tensor(resized_image)
    plt.subplot(1, 3, 2)
    plt.imshow(tensor_image.permute(1, 2, 0))
    plt.title('2. 转换为张量\n[0, 1]范围', fontsize=12)
    plt.axis('off')

    # 3. 归一化
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
    normalized_image = normalize(tensor_image)

    # 为了可视化，我们需要反归一化
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    denormalized = normalized_image * std + mean
    denormalized = torch.clamp(denormalized, 0, 1)

    plt.subplot(1, 3, 3)
    plt.imshow(denormalized.permute(1, 2, 0))
    plt.title('3. 归一化\n标准化处理', fontsize=12)
    plt.axis('off')

    # 保存图像
    output_path = Path(output_dir) / "17_normalization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图像归一化可视化已保存: {output_path}")

def visualize_patch_embedding(image, output_dir="outputs"):
    """
    可视化Patch Embedding过程

    Args:
        image: 原始图像
        output_dir: 输出目录
    """
    print("生成Patch Embedding可视化...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    # 调整大小和裁剪
    resize_crop = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224)
    ])
    resized_image = resize_crop(image)

    # 创建图像
    plt.figure(figsize=(10, 10))
    plt.title('Vision Transformer Patch Embedding', fontsize=16, fontweight='bold')

    # 显示图像
    plt.imshow(resized_image)

    # 添加patch网格
    patch_size = 32  # CLIP ViT-B/32使用32x32的patch
    num_patches = 224 // patch_size

    # 绘制网格
    for i in range(num_patches + 1):
        plt.axhline(y=i * patch_size, color='white', linestyle='-', linewidth=1)
        plt.axvline(x=i * patch_size, color='white', linestyle='-', linewidth=1)

    # 标记几个特定的patch
    positions = [(1, 1), (3, 3), (5, 5)]
    colors = ['red', 'blue', 'green']

    for (row, col), color in zip(positions, colors):
        rect = plt.Rectangle((col*patch_size, row*patch_size), patch_size, patch_size,
                           linewidth=3, edgecolor=color, facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(col*patch_size + patch_size//2, row*patch_size + patch_size//2, f'P{row},{col}',
                color=color, fontweight='bold', fontsize=12, ha='center', va='center')

    plt.axis('off')

    # 保存图像
    output_path = Path(output_dir) / "18_patch_embedding.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Patch Embedding可视化已保存: {output_path}")

def visualize_position_encoding(output_dir="outputs"):
    """
    可视化位置编码

    Args:
        output_dir: 输出目录
    """
    print("生成位置编码可视化...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.title('Vision Transformer位置编码', fontsize=16, fontweight='bold')

    # 假设使用32x32的patch，总共有7x7=49个patch
    patch_size = 32
    num_patches = 224 // patch_size

    # 创建位置ID矩阵
    position_map = np.zeros((num_patches, num_patches))
    for i in range(num_patches):
        for j in range(num_patches):
            position_map[i, j] = i * num_patches + j

    # 绘制位置编码热力图
    im = plt.imshow(position_map, cmap='viridis')
    plt.colorbar(im, shrink=0.8, label='位置ID')

    # 添加位置ID标签
    for i in range(num_patches):
        for j in range(num_patches):
            plt.text(j, i, f'{int(position_map[i, j])}', ha='center', va='center',
                    color='white' if position_map[i, j] > num_patches**2/2 else 'black',
                    fontsize=9)

    plt.xlabel('列索引', fontsize=12)
    plt.ylabel('行索引', fontsize=12)

    # 保存图像
    output_path = Path(output_dir) / "19_position_encoding.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"位置编码可视化已保存: {output_path}")

def visualize_cls_token_sequence(output_dir="outputs"):
    """
    可视化CLS Token + Patches序列

    Args:
        output_dir: 输出目录
    """
    print("生成CLS Token + Patches序列可视化...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    plt.figure(figsize=(12, 4))
    plt.title('Vision Transformer输入序列', fontsize=16, fontweight='bold')

    # 假设使用32x32的patch，总共有7x7=49个patch，加上1个CLS token
    patch_size = 32
    num_patches = 224 // patch_size
    sequence_length = num_patches**2 + 1

    # 创建序列示意图
    sequence_visual = np.zeros((1, sequence_length))
    sequence_visual[0, 0] = 1  # CLS token标记为1
    sequence_visual[0, 1:] = 0.5  # patches标记为0.5

    # 绘制序列
    im = plt.imshow(sequence_visual, cmap='RdYlBu', aspect='auto')
    plt.colorbar(im, shrink=0.8)
    plt.xlabel('序列位置', fontsize=12)
    plt.yticks([])

    # 标记CLS token
    plt.text(0, 0, 'CLS', ha='center', va='center', fontweight='bold', color='white', fontsize=12)

    # 标记一些patch位置
    for i in [1, 10, 20, 30, 40, sequence_length-1]:
        if i > 0:
            plt.text(i, 0, f'P{i-1}', ha='center', va='center', fontweight='bold', fontsize=10)

    # 添加说明
    plt.figtext(0.5, 0.01, f"序列结构: [CLS] + {num_patches**2}个Patches = {sequence_length}个tokens",
               ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    # 保存图像
    output_path = Path(output_dir) / "20_cls_token_sequence.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"CLS Token + Patches序列可视化已保存: {output_path}")

def visualize_attention_map(attention_data, image, output_dir="outputs"):
    """
    可视化注意力图

    Args:
        attention_data: 注意力数据
        image: 原始图像
        output_dir: 输出目录
    """
    print("生成注意力图可视化...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    if not attention_data or not attention_data.get('vision_attentions'):
        print("⚠️ 无法生成注意力图 - 缺少vision_attentions数据")
        return

    # 调整大小和裁剪
    resize_crop = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224)
    ])
    resized_image = resize_crop(image)

    # 获取最后一层的注意力
    vision_attentions = attention_data['vision_attentions']
    final_attention = vision_attentions[-1][0, :, 0, 1:].mean(dim=0)

    # 计算patch配置
    patch_size = 32  # CLIP ViT-B/32使用32x32的patch
    num_patches = 224 // patch_size
    expected_length = num_patches * num_patches

    if len(final_attention) == expected_length:
        # 重塑为二维注意力图
        attention_map = final_attention.reshape(num_patches, num_patches).numpy()

        # 创建图像
        plt.figure(figsize=(12, 5))
        plt.suptitle('Vision Transformer注意力可视化', fontsize=16, fontweight='bold')

        # 1. 注意力热力图
        plt.subplot(1, 2, 1)
        im = plt.imshow(attention_map, cmap='hot', interpolation='bilinear')
        plt.colorbar(im, shrink=0.8)
        plt.title('注意力热力图', fontsize=12)
        plt.axis('off')

        # 2. 注意力叠加在原图上
        plt.subplot(1, 2, 2)
        plt.imshow(resized_image, alpha=0.7)

        # 将注意力图调整到原图尺寸
        attention_resized = np.kron(attention_map, np.ones((patch_size, patch_size)))
        plt.imshow(attention_resized, cmap='hot', alpha=0.5, extent=[0, 224, 224, 0])
        plt.title('注意力叠加图', fontsize=12)
        plt.axis('off')

        # 标注最关注的区域
        max_attention_idx = final_attention.argmax().item()
        max_row = max_attention_idx // num_patches
        max_col = max_attention_idx % num_patches
        rect = plt.Rectangle((max_col*patch_size, max_row*patch_size), patch_size, patch_size,
                           linewidth=3, edgecolor='yellow', facecolor='none')
        plt.gca().add_patch(rect)
        plt.text(max_col*patch_size + patch_size//2, max_row*patch_size + patch_size//2, 'MAX',
                color='yellow', fontweight='bold', fontsize=12, ha='center', va='center')

        # 保存图像
        output_path = Path(output_dir) / "21_attention_map.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"注意力图可视化已保存: {output_path}")
    else:
        print(f"⚠️ 注意力维度不匹配 - 期望:{expected_length}, 实际:{len(final_attention)}")

def create_image_preprocessing_visualizations(image, processed_image_tensor, attention_data, output_dir="outputs"):
    """
    创建所有图像预处理相关的可视化

    Args:
        image: 原始图像
        processed_image_tensor: 处理后的图像张量
        attention_data: 注意力数据
        output_dir: 输出目录
    """
    print("生成图像预处理可视化...")

    # 确保输出目录存在
    Path(output_dir).mkdir(exist_ok=True)

    # 1. 原始图像
    visualize_original_image(image, output_dir)

    # 2. 预处理步骤
    visualize_preprocessing_steps(image, output_dir)

    # 3. 归一化
    visualize_normalization(image, output_dir)

    # 4. Patch Embedding
    visualize_patch_embedding(image, output_dir)

    # 5. 位置编码
    visualize_position_encoding(output_dir)

    # 6. CLS Token + Patches序列
    visualize_cls_token_sequence(output_dir)

    # 7. 注意力图
    if attention_data:
        visualize_attention_map(attention_data, image, output_dir)

    print("图像预处理可视化完成")

