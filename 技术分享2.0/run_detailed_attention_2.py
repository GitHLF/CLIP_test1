#!/usr/bin/env python3
"""
英文文本注意力机制过程可视化 v2.0
按照：分词 → 查询向量 → 位置编码 → transformer → 句子语义向量 的路径进行详细分析
专门针对 "two dogs lying on a cushion in the sun" 这句话
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from utils.model_loader import load_local_clip_model
from utils.attention_extractor import extract_attention_weights

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def step1_tokenization_analysis(tokenizer, target_text, output_dir):
    """
    步骤1: 分词过程分析
    """
    print("=" * 50)
    print("步骤1: 分词过程分析")
    print("=" * 50)

    # 进行分词
    text_inputs = tokenizer(
        target_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )

    input_ids = text_inputs['input_ids'][0].numpy()
    attention_mask = text_inputs['attention_mask'][0].numpy()
    valid_length = int(attention_mask.sum())

    print(f"原始文本: {target_text}")
    print(f"Token序列长度: {valid_length}")
    print(f"Input IDs: {input_ids[:valid_length]}")

    # 创建可视化
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle('Step 1: Tokenization Process Analysis\n"two dogs lying on a cushion in the sun"',
                 fontsize=16, fontweight='bold')

    # 1.1 原始文本展示
    axes[0].set_title('1.1 Original Text', fontsize=14, fontweight='bold')
    axes[0].text(0.5, 0.5, f'"{target_text}"', ha='center', va='center',
                 fontsize=18, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].axis('off')

    # 1.2 Token ID序列
    axes[1].set_title('1.2 Token ID Sequence', fontsize=14, fontweight='bold')

    # 绘制token ID条形图
    token_positions = range(valid_length)
    bars = axes[1].bar(token_positions, input_ids[:valid_length],
                       color=['red' if i == 0 else 'orange' if i == valid_length - 1 else 'lightblue'
                              for i in range(valid_length)], alpha=0.7)

    axes[1].set_xlabel('Token Position')
    axes[1].set_ylabel('Token ID')
    axes[1].set_xticks(token_positions)

    # 添加token ID标签
    for i, (bar, token_id) in enumerate(zip(bars, input_ids[:valid_length])):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2., height + 100,
                     f'{token_id}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 1.3 Token类型说明
    axes[2].set_title('1.3 Token Type Analysis', fontsize=14, fontweight='bold')

    # 预期的token标签
    expected_tokens = ['[CLS]', 'two', 'dogs', 'lying', 'on', 'a', 'cushion', 'in', 'the', 'sun', '[SEP]']
    token_labels = expected_tokens[:valid_length]

    # 绘制token类型
    colors = ['red', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink',
              'lightgray', 'lightcyan', 'wheat', 'lavender', 'lightsteelblue', 'orange']

    for i, (token, color) in enumerate(zip(token_labels, colors)):
        rect = plt.Rectangle((i, 0), 1, 1, facecolor=color, edgecolor='black', linewidth=2)
        axes[2].add_patch(rect)
        axes[2].text(i + 0.5, 0.5, token, ha='center', va='center', fontweight='bold', fontsize=11)
        axes[2].text(i + 0.5, -0.3, f'ID:{input_ids[i]}', ha='center', va='center', fontsize=9)

    axes[2].set_xlim(0, valid_length)
    axes[2].set_ylim(-0.5, 1.2)
    axes[2].set_xlabel('Token Position')
    axes[2].axis('off')

    # 添加说明
    explanation = f"""分词过程说明：
• 原始文本被分解为 {valid_length} 个tokens
• [CLS]: 分类标记，用于句子级别的表示
• [SEP]: 分隔标记，表示句子结束
• 内容tokens: {valid_length - 2} 个实际单词token
• 每个token都有唯一的ID，用于后续的embedding查找"""

    plt.figtext(0.5, 0.02, explanation, ha='center', fontsize=11,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "step1_tokenization.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ 分词过程分析完成")
    return text_inputs, token_labels


def step2_query_vector_analysis(model, text_inputs, token_labels, output_dir):
    """
    步骤2: 查询向量生成分析
    """
    print("=" * 50)
    print("步骤2: 查询向量生成分析")
    print("=" * 50)

    device = next(model.parameters()).device
    text_inputs_device = {k: v.to(device) for k, v in text_inputs.items()}

    # 获取embedding层的输出
    with torch.no_grad():
        # 获取token embeddings
        token_embeddings = model.text_model.embeddings.token_embedding(text_inputs_device['input_ids'])
        print(f"Token embeddings shape: {token_embeddings.shape}")  # [1, seq_len, hidden_size]

        # 获取位置embeddings
        position_ids = torch.arange(text_inputs_device['input_ids'].shape[1], device=device).unsqueeze(0)
        position_embeddings = model.text_model.embeddings.position_embedding(position_ids)
        print(f"Position embeddings shape: {position_embeddings.shape}")

        # 合并embeddings
        embeddings = token_embeddings + position_embeddings
        print(f"Combined embeddings shape: {embeddings.shape}")

    # 转换为numpy用于可视化
    token_embeddings_np = token_embeddings[0].cpu().numpy()  # [seq_len, hidden_size]
    position_embeddings_np = position_embeddings[0].cpu().numpy()
    embeddings_np = embeddings[0].cpu().numpy()

    valid_length = len(token_labels)

    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Step 2: Query Vector Generation Analysis', fontsize=16, fontweight='bold')

    # 2.1 Token Embeddings热力图
    axes[0, 0].set_title('2.1 Token Embeddings (First 64 dims)', fontsize=12, fontweight='bold')
    im1 = axes[0, 0].imshow(token_embeddings_np[:valid_length, :64].T, cmap='RdBu', aspect='auto')
    axes[0, 0].set_xlabel('Token Position')
    axes[0, 0].set_ylabel('Embedding Dimension')
    axes[0, 0].set_xticks(range(valid_length))
    axes[0, 0].set_xticklabels([label.replace('[', '').replace(']', '') for label in token_labels], rotation=45)
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)

    # 2.2 Position Embeddings热力图
    axes[0, 1].set_title('2.2 Position Embeddings (First 64 dims)', fontsize=12, fontweight='bold')
    im2 = axes[0, 1].imshow(position_embeddings_np[:valid_length, :64].T, cmap='RdBu', aspect='auto')
    axes[0, 1].set_xlabel('Token Position')
    axes[0, 1].set_ylabel('Embedding Dimension')
    axes[0, 1].set_xticks(range(valid_length))
    axes[0, 1].set_xticklabels([f'Pos{i}' for i in range(valid_length)], rotation=45)
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)

    # 2.3 Combined Embeddings热力图
    axes[1, 0].set_title('2.3 Combined Embeddings (Token + Position)', fontsize=12, fontweight='bold')
    im3 = axes[1, 0].imshow(embeddings_np[:valid_length, :64].T, cmap='RdBu', aspect='auto')
    axes[1, 0].set_xlabel('Token Position')
    axes[1, 0].set_ylabel('Embedding Dimension')
    axes[1, 0].set_xticks(range(valid_length))
    axes[1, 0].set_xticklabels([label.replace('[', '').replace(']', '') for label in token_labels], rotation=45)
    plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)

    # 2.4 向量相似度分析
    axes[1, 1].set_title('2.4 Token Embedding Similarities', fontsize=12, fontweight='bold')

    # 计算token embeddings之间的余弦相似度
    token_emb_norm = F.normalize(torch.from_numpy(token_embeddings_np[:valid_length]), dim=1)
    similarity_matrix = torch.mm(token_emb_norm, token_emb_norm.t()).numpy()

    im4 = axes[1, 1].imshow(similarity_matrix, cmap='Blues', vmin=0, vmax=1)
    axes[1, 1].set_xlabel('Token')
    axes[1, 1].set_ylabel('Token')
    axes[1, 1].set_xticks(range(valid_length))
    axes[1, 1].set_yticks(range(valid_length))
    axes[1, 1].set_xticklabels([label.replace('[', '').replace(']', '') for label in token_labels], rotation=45)
    axes[1, 1].set_yticklabels([label.replace('[', '').replace(']', '') for label in token_labels])
    plt.colorbar(im4, ax=axes[1, 1], shrink=0.8)

    # 添加相似度数值
    for i in range(valid_length):
        for j in range(valid_length):
            if i != j:  # 不显示对角线
                axes[1, 1].text(j, i, f'{similarity_matrix[i, j]:.2f}',
                                ha='center', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "step2_query_vectors.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ 查询向量生成分析完成")
    return embeddings


def step3_position_encoding_analysis(embeddings, token_labels, output_dir):
    """
    步骤3: 位置编码详细分析
    """
    print("=" * 50)
    print("步骤3: 位置编码详细分析")
    print("=" * 50)

    embeddings_np = embeddings[0].cpu().numpy()
    valid_length = len(token_labels)

    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Step 3: Position Encoding Analysis', fontsize=16, fontweight='bold')

    # 3.1 位置编码模式
    axes[0, 0].set_title('3.1 Position Encoding Patterns', fontsize=12, fontweight='bold')

    # 模拟正弦余弦位置编码的模式
    pos_encoding = np.zeros((valid_length, 64))
    for pos in range(valid_length):
        for i in range(32):
            pos_encoding[pos, 2 * i] = np.sin(pos / (10000 ** (2 * i / 64)))
            pos_encoding[pos, 2 * i + 1] = np.cos(pos / (10000 ** (2 * i / 64)))

    im1 = axes[0, 0].imshow(pos_encoding.T, cmap='RdBu', aspect='auto')
    axes[0, 0].set_xlabel('Position')
    axes[0, 0].set_ylabel('Encoding Dimension')
    axes[0, 0].set_xticks(range(valid_length))
    axes[0, 0].set_xticklabels([f'P{i}' for i in range(valid_length)])
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)

    # 3.2 不同位置的编码向量
    axes[0, 1].set_title('3.2 Position Vectors (First 32 dims)', fontsize=12, fontweight='bold')

    positions_to_show = [0, 2, 5, 8, valid_length - 1]  # CLS, dogs, cushion, sun, SEP
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    for i, (pos, color) in enumerate(zip(positions_to_show, colors)):
        if pos < valid_length:
            axes[0, 1].plot(pos_encoding[pos, :32], color=color, linewidth=2,
                            label=f'{token_labels[pos]} (pos {pos})', alpha=0.8)

    axes[0, 1].set_xlabel('Encoding Dimension')
    axes[0, 1].set_ylabel('Encoding Value')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3.3 位置距离对相似度的影响
    axes[1, 0].set_title('3.3 Position Distance vs Similarity', fontsize=12, fontweight='bold')

    # 计算不同位置间的相似度
    distances = []
    similarities = []

    for i in range(valid_length):
        for j in range(i + 1, valid_length):
            distance = abs(i - j)
            similarity = np.dot(pos_encoding[i], pos_encoding[j]) / (
                    np.linalg.norm(pos_encoding[i]) * np.linalg.norm(pos_encoding[j]))
            distances.append(distance)
            similarities.append(similarity)

    axes[1, 0].scatter(distances, similarities, alpha=0.6, s=50)
    axes[1, 0].set_xlabel('Position Distance')
    axes[1, 0].set_ylabel('Position Encoding Similarity')
    axes[1, 0].grid(True, alpha=0.3)

    # 添加趋势线
    z = np.polyfit(distances, similarities, 1)
    p = np.poly1d(z)
    axes[1, 0].plot(distances, p(distances), "r--", alpha=0.8, linewidth=2)

    # 3.4 位置编码的作用说明
    axes[1, 1].axis('off')

    explanation = f"""位置编码的作用机制：

1. 绝对位置信息：
   • 每个位置都有唯一的编码向量
   • 使用正弦余弦函数生成周期性模式
   • 不同频率捕获不同尺度的位置关系

2. 相对位置关系：
   • 相邻位置的编码相似度较高
   • 距离越远，相似度越低
   • 帮助模型理解词序关系

3. 在我们的句子中：
   • "two" (pos 1) 和 "dogs" (pos 2) 位置相邻
   • "dogs" 和 "cushion" (pos 5) 有一定距离
   • 位置编码帮助模型理解语法结构

4. 与Token Embedding结合：
   • Token embedding: 词汇语义信息
   • Position embedding: 位置序列信息
   • 两者相加得到完整的输入表示"""

    axes[1, 1].text(0.05, 0.95, explanation, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "step3_position_encoding.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ 位置编码分析完成")


def step4_transformer_analysis(model, text_inputs, token_labels, output_dir):
    """
    步骤4: Transformer层详细分析
    """
    print("=" * 50)
    print("步骤4: Transformer层详细分析")
    print("=" * 50)

    device = next(model.parameters()).device
    text_inputs_device = {k: v.to(device) for k, v in text_inputs.items()}

    # 获取所有层的注意力
    with torch.no_grad():
        outputs = model.text_model(
            input_ids=text_inputs_device['input_ids'],
            attention_mask=text_inputs_device['attention_mask'],
            output_attentions=True,
            output_hidden_states=True
        )

    attentions = outputs.attentions  # 12层注意力
    hidden_states = outputs.hidden_states  # 13个hidden states (embedding + 12层)

    valid_length = len(token_labels)
    content_start = 1  # 跳过CLS
    content_end = valid_length - 1  # 跳过SEP
    content_length = content_end - content_start

    print(f"Transformer层数: {len(attentions)}")
    print(f"注意力头数: {attentions[0].shape[1]}")
    print(f"内容token数: {content_length}")

    # 创建可视化
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Step 4: Transformer Layer Analysis', fontsize=16, fontweight='bold')

    # 4.1 不同层的注意力模式演变
    ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=4)
    ax1.set_title('4.1 Attention Pattern Evolution Across Layers', fontsize=14, fontweight='bold')

    # 选择几个关键层进行展示
    layers_to_show = [0, 3, 7, 11]  # 第1、4、8、12层
    layer_names = ['Layer 1', 'Layer 4', 'Layer 8', 'Layer 12']

    for i, (layer_idx, layer_name) in enumerate(zip(layers_to_show, layer_names)):
        ax = plt.subplot2grid((4, 4), (1, i))

        # 平均所有注意力头，只看内容token
        layer_attention = attentions[layer_idx][0].mean(dim=0)  # 平均所有头
        content_attention = layer_attention[content_start:content_end, content_start:content_end]

        im = ax.imshow(content_attention.cpu().numpy(), cmap='Blues', vmin=0, vmax=1)
        ax.set_title(layer_name, fontsize=12)
        ax.set_xticks(range(content_length))
        ax.set_yticks(range(content_length))
        content_labels = token_labels[content_start:content_end]
        ax.set_xticklabels(content_labels, rotation=45, fontsize=8)
        ax.set_yticklabels(content_labels, fontsize=8)

        if i == len(layers_to_show) - 1:
            plt.colorbar(im, ax=ax, shrink=0.8)

    # 4.2 多头注意力分析
    ax2 = plt.subplot2grid((4, 4), (2, 0), colspan=2)
    ax2.set_title('4.2 Multi-Head Attention Patterns (Layer 12)', fontsize=14, fontweight='bold')

    # 分析最后一层的多头注意力
    final_attention = attentions[-1][0]  # [num_heads, seq_len, seq_len]
    num_heads = min(8, final_attention.shape[0])

    head_patterns = []
    for head in range(num_heads):
        head_att = final_attention[head, content_start:content_end, content_start:content_end]
        diag_strength = torch.diag(head_att).mean().item()
        off_diag_strength = (head_att.sum() - torch.diag(head_att).sum()).item() / (
                    content_length * content_length - content_length)
        head_patterns.append([diag_strength, off_diag_strength])

    head_patterns = np.array(head_patterns)
    x = np.arange(num_heads)
    width = 0.35

    bars1 = ax2.bar(x - width / 2, head_patterns[:, 0], width, label='Self-Attention', alpha=0.8, color='orange')
    bars2 = ax2.bar(x + width / 2, head_patterns[:, 1], width, label='Cross-Attention', alpha=0.8, color='lightblue')

    ax2.set_xlabel('Attention Head')
    ax2.set_ylabel('Attention Strength')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'H{i + 1}' for i in range(num_heads)])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 4.3 Hidden States演变
    ax3 = plt.subplot2grid((4, 4), (2, 2), colspan=2)
    ax3.set_title('4.3 Hidden States Evolution', fontsize=14, fontweight='bold')

    # 计算每层hidden states的变化
    layer_changes = []
    for i in range(1, len(hidden_states)):
        prev_hidden = hidden_states[i - 1][0, content_start:content_end]  # 内容token
        curr_hidden = hidden_states[i][0, content_start:content_end]

        # 计算变化幅度（L2范数）
        change = torch.norm(curr_hidden - prev_hidden, dim=1).mean().item()
        layer_changes.append(change)

    ax3.plot(range(1, len(hidden_states)), layer_changes, 'o-', linewidth=2, markersize=6)
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Hidden State Change')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Hidden State Changes Between Layers')

    # 4.4 关键词"dogs"的注意力演变
    ax4 = plt.subplot2grid((4, 4), (3, 0), colspan=4)
    ax4.set_title('4.4 "dogs" Token Attention Evolution Across Layers', fontsize=14, fontweight='bold')

    dogs_idx = 1  # "dogs"在内容token中的位置 (0:two, 1:dogs, ...)
    dogs_attention_evolution = []

    for layer_idx in range(len(attentions)):
        layer_attention = attentions[layer_idx][0].mean(dim=0)  # 平均所有头
        dogs_attention = layer_attention[content_start + dogs_idx, content_start:content_end]
        dogs_attention_evolution.append(dogs_attention.cpu().numpy())

    dogs_attention_evolution = np.array(dogs_attention_evolution)

    im4 = ax4.imshow(dogs_attention_evolution.T, cmap='Blues', aspect='auto')
    ax4.set_xlabel('Layer')
    ax4.set_ylabel('Target Token')
    ax4.set_xticks(range(len(attentions)))
    ax4.set_xticklabels([f'L{i + 1}' for i in range(len(attentions))])
    ax4.set_yticks(range(content_length))
    ax4.set_yticklabels(content_labels)
    plt.colorbar(im4, ax=ax4, shrink=0.8)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "step4_transformer_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Transformer层分析完成")
    return outputs


def step5_sentence_vector_analysis(model, text_inputs, token_labels, outputs, output_dir):
    """
    步骤5: 句子语义向量生成分析
    """
    print("=" * 50)
    print("步骤5: 句子语义向量生成分析")
    print("=" * 50)

    device = next(model.parameters()).device
    text_inputs_device = {k: v.to(device) for k, v in text_inputs.items()}

    # 获取最终的句子表示
    with torch.no_grad():
        # 获取CLS token的表示（句子级别表示）
        cls_hidden = outputs.last_hidden_state[0, 0]  # CLS token

        # 通过projection head得到最终的文本embedding
        text_embeds = model.text_projection(cls_hidden)

        # 归一化
        text_embeds_norm = F.normalize(text_embeds, dim=0)

        print(f"CLS hidden state shape: {cls_hidden.shape}")
        print(f"Text embedding shape: {text_embeds.shape}")
        print(f"Text embedding norm: {torch.norm(text_embeds_norm).item():.4f}")

    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Step 5: Sentence Semantic Vector Analysis', fontsize=16, fontweight='bold')

    # 5.1 CLS token演变过程
    axes[0, 0].set_title('5.1 CLS Token Evolution Across Layers', fontsize=12, fontweight='bold')

    # 收集所有层的CLS token表示
    cls_evolution = []
    for hidden_state in outputs.hidden_states:
        cls_token = hidden_state[0, 0].cpu().numpy()  # CLS token
        cls_evolution.append(cls_token)

    cls_evolution = np.array(cls_evolution)  # [num_layers+1, hidden_size]

    # 显示前64维的演变
    im1 = axes[0, 0].imshow(cls_evolution[:, :64].T, cmap='RdBu', aspect='auto')
    axes[0, 0].set_xlabel('Layer')
    axes[0, 0].set_ylabel('Hidden Dimension')
    axes[0, 0].set_xticks(range(len(cls_evolution)))
    axes[0, 0].set_xticklabels(['Embed'] + [f'L{i + 1}' for i in range(len(cls_evolution) - 1)])
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)

    # 5.2 CLS token与其他token的相似度
    axes[0, 1].set_title('5.2 CLS Token Similarity with Content Tokens', fontsize=12, fontweight='bold')

    final_hidden = outputs.last_hidden_state[0].cpu()  # [seq_len, hidden_size]
    cls_final = final_hidden[0]  # CLS token
    content_tokens = final_hidden[1:-1]  # 内容tokens（去掉CLS和SEP）

    # 计算相似度
    cls_norm = F.normalize(cls_final.unsqueeze(0), dim=1)
    content_norm = F.normalize(content_tokens, dim=1)
    similarities = torch.mm(cls_norm, content_norm.t())[0].numpy()

    content_labels = token_labels[1:-1]  # 去掉CLS和SEP
    bars = axes[0, 1].bar(range(len(similarities)), similarities,
                          color='lightcoral', alpha=0.7)
    axes[0, 1].set_xlabel('Content Token')
    axes[0, 1].set_ylabel('Similarity with CLS')
    axes[0, 1].set_xticks(range(len(similarities)))
    axes[0, 1].set_xticklabels(content_labels, rotation=45)
    axes[0, 1].grid(True, alpha=0.3)

    # 添加数值标签
    for bar, sim in zip(bars, similarities):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f'{sim:.3f}', ha='center', va='bottom', fontsize=9)

    # 5.3 最终文本embedding的特征分布
    axes[1, 0].set_title('5.3 Final Text Embedding Distribution', fontsize=12, fontweight='bold')

    text_embeds_np = text_embeds_norm.cpu().numpy()

    # 绘制embedding值的分布
    axes[1, 0].hist(text_embeds_np, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 0].set_xlabel('Embedding Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)

    # 添加统计信息
    mean_val = np.mean(text_embeds_np)
    std_val = np.std(text_embeds_np)
    axes[1, 0].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
    axes[1, 0].axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.7, label=f'±Std: {std_val:.3f}')
    axes[1, 0].axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7)
    axes[1, 0].legend()

    # 5.4 句子语义向量生成过程总结
    axes[1, 1].axis('off')

    # 计算一些关键统计信息
    max_sim_idx = np.argmax(similarities)
    max_sim_token = content_labels[max_sim_idx]
    max_sim_value = similarities[max_sim_idx]

    explanation = f"""句子语义向量生成过程：

1. CLS Token的作用：
   • 初始化为特殊的[CLS] embedding
   • 通过12层Transformer逐步聚合信息
   • 最终成为整个句子的语义表示

2. 信息聚合机制：
   • CLS token通过注意力机制关注所有内容token
   • 最相似的内容token: "{max_sim_token}" ({max_sim_value:.3f})
   • 逐层融合语法和语义信息

3. 最终表示特征：
   • 维度: {text_embeds.shape[0]}
   • L2范数: {torch.norm(text_embeds_norm).item():.4f} (已归一化)
   • 均值: {mean_val:.3f}, 标准差: {std_val:.3f}

4. 语义编码结果：
   • 捕获了"两只狗在垫子上晒太阳"的完整语义
   • 可用于与图像特征进行跨模态匹配
   • 包含了词汇、语法、语义的综合信息

5. 从Token到句子的转换：
   • Token level: 单词级别的语义
   • Sentence level: 句子级别的语义
   • 通过CLS token实现信息的全局整合"""

    axes[1, 1].text(0.05, 0.95, explanation, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "step5_sentence_vector.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ 句子语义向量分析完成")
    return text_embeds_norm


def create_process_summary(output_dir):
    """
    创建整个过程的总结图
    """
    print("=" * 50)
    print("创建过程总结")
    print("=" * 50)

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    fig.suptitle('Complete Process: From Tokenization to Sentence Vector\n"two dogs lying on a cushion in the sun"',
                 fontsize=16, fontweight='bold')

    ax.axis('off')

    # 绘制流程图
    steps = [
        "1. Tokenization\n分词",
        "2. Query Vectors\n查询向量",
        "3. Position Encoding\n位置编码",
        "4. Transformer\nTransformer层",
        "5. Sentence Vector\n句子语义向量"
    ]

    descriptions = [
        "• 文本 → Token序列\n• 11个tokens\n• [CLS] + 9内容 + [SEP]",
        "• Token → Embedding\n• 512维向量\n• 词汇 + 位置信息",
        "• 绝对位置信息\n• 相对位置关系\n• 正弦余弦编码",
        "• 12层注意力机制\n• 8个注意力头\n• 自注意力 + 跨词注意力",
        "• CLS token表示\n• 512维语义向量\n• 句子级别语义"
    ]

    # 绘制步骤框
    box_width = 0.15
    box_height = 0.3
    y_center = 0.6

    for i, (step, desc) in enumerate(zip(steps, descriptions)):
        x_center = 0.1 + i * 0.2

        # 绘制步骤框
        rect = plt.Rectangle((x_center - box_width / 2, y_center - box_height / 2),
                             box_width, box_height,
                             facecolor='lightblue', edgecolor='black', linewidth=2)
        ax.add_patch(rect)

        # 添加步骤标题
        ax.text(x_center, y_center + 0.05, step, ha='center', va='center',
                fontsize=12, fontweight='bold')

        # 添加描述
        ax.text(x_center, y_center - 0.05, desc, ha='center', va='center',
                fontsize=10, style='italic')

        # 绘制箭头（除了最后一个）
        if i < len(steps) - 1:
            ax.annotate('', xy=(x_center + box_width / 2 + 0.02, y_center),
                        xytext=(x_center + box_width / 2, y_center),
                        arrowprops=dict(arrowstyle='->', lw=2, color='red'))

    # 添加详细说明
    summary_text = """完整的文本理解过程：

输入: "two dogs lying on a cushion in the sun"

步骤1 - 分词 (Tokenization):
• 将原始文本分解为可处理的token单元
• 添加特殊标记[CLS]和[SEP]用于句子级别处理
• 每个token映射到唯一的ID

步骤2 - 查询向量 (Query Vectors):
• 将token ID转换为高维embedding向量
• Token embedding包含词汇语义信息
• Position embedding包含位置序列信息

步骤3 - 位置编码 (Position Encoding):
• 为每个位置生成唯一的编码向量
• 使用正弦余弦函数捕获位置关系
• 帮助模型理解词序和语法结构

步骤4 - Transformer处理:
• 12层自注意力机制逐步处理信息
• 每层8个注意力头捕获不同类型的关系
• 词汇间的依赖关系逐步建立和强化

步骤5 - 句子语义向量:
• CLS token聚合所有词汇信息
• 生成固定长度的句子级别表示
• 可用于下游任务（如图文匹配）

最终结果: 512维的句子语义向量，包含了完整的语义信息"""

    ax.text(0.5, 0.25, summary_text, ha='center', va='top',
            fontsize=11, fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "step6_process_summary.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ 过程总结创建完成")


def run_detailed_attention_analysis_v2():
    """
    运行详细的注意力分析 v2.0
    按照：分词 → 查询向量 → 位置编码 → transformer → 句子语义向量 的路径
    """
    print("🚀 启动英文文本注意力机制过程分析 v2.0")
    print("📊 分析路径：分词 → 查询向量 → 位置编码 → transformer → 句子语义向量")
    print("🎯 目标文本：'two dogs lying on a cushion in the sun'")
    print("=" * 80)

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

    # 2. 准备文本
    target_text = "two dogs lying on a cushion in the sun"
    print(f"✓ 目标文本: {target_text}")

    # 3. 创建输出目录
    output_dir = Path(__file__).parent / "outputs" / "detailed_attention_v2"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ 输出目录: {output_dir}")

    # 执行分析步骤
    try:
        # 步骤1: 分词分析
        text_inputs, token_labels = step1_tokenization_analysis(tokenizer, target_text, output_dir)

        # 步骤2: 查询向量分析
        embeddings = step2_query_vector_analysis(model, text_inputs, token_labels, output_dir)

        # 步骤3: 位置编码分析
        step3_position_encoding_analysis(embeddings, token_labels, output_dir)

        # 步骤4: Transformer分析
        outputs = step4_transformer_analysis(model, text_inputs, token_labels, output_dir)

        # 步骤5: 句子语义向量分析
        sentence_vector = step5_sentence_vector_analysis(model, text_inputs, token_labels, outputs, output_dir)

        # 步骤6: 过程总结
        create_process_summary(output_dir)

        print("=" * 80)
        print("✅ 详细注意力分析 v2.0 完成！")
        print(f"📁 所有结果保存在: {output_dir}")
        print("📄 生成的文件:")
        print("   1. step1_tokenization.png - 分词过程分析")
        print("   2. step2_query_vectors.png - 查询向量生成")
        print("   3. step3_position_encoding.png - 位置编码分析")
        print("   4. step4_transformer_analysis.png - Transformer层分析")
        print("   5. step5_sentence_vector.png - 句子语义向量")
        print("   6. step6_process_summary.png - 完整过程总结")
        print("=" * 80)

        # 尝试打开文件夹（macOS）
        import subprocess
        import sys
        try:
            if sys.platform == "darwin":  # macOS
                subprocess.run(["open", str(output_dir)], check=False)
                print("📂 文件夹已自动打开")
        except:
            pass

        return output_dir

    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 直接运行详细演示
    run_detailed_attention_analysis_v2();
