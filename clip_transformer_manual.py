import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class MultiHeadAttention(nn.Module):
    """多头注意力机制实现"""

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None, debug=False):
        batch_size, seq_len, d_model = x.shape

        if debug:
            print(f"    🔍 MultiHeadAttention 详细计算过程:")
            print(f"       输入 x shape: {x.shape}")
            print(f"       输入 x 前3个token的前5维数值:")
            for i in range(min(3, seq_len)):
                print(f"         token {i}: {x[0, i, :5].detach().numpy()}")

        # 计算Q, K, V
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if debug:
            print(f"       Q shape: {Q.shape} (batch, heads, seq_len, head_dim)")
            print(f"       K shape: {K.shape}")
            print(f"       V shape: {V.shape}")
            print(f"       第1个头的Q矩阵前3个token的前3维:")
            for i in range(min(3, seq_len)):
                print(f"         Q[0,0,{i},:3] = {Q[0, 0, i, :3].detach().numpy()}")

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if debug:
            print(f"       注意力分数 scores shape: {scores.shape}")
            print(f"       缩放因子 √d_k = √{self.head_dim} = {math.sqrt(self.head_dim):.3f}")
            print(f"       第1个头的注意力分数矩阵 (前3x3):")
            print(f"         {scores[0, 0, :3, :3].detach().numpy()}")

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            mask = mask.expand(batch_size, self.num_heads, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)
            if debug:
                print(f"       应用mask后的分数 (前3x3):")
                print(f"         {scores[0, 0, :3, :3].detach().numpy()}")

        attention_weights = F.softmax(scores, dim=-1)

        if debug:
            print(f"       注意力权重 (softmax后, 前3x3):")
            print(f"         {attention_weights[0, 0, :3, :3].detach().numpy()}")
            print(f"       第1行权重和: {attention_weights[0, 0, 0, :].sum().item():.6f}")

        # 应用注意力权重
        context = torch.matmul(attention_weights, V)

        if debug:
            print(f"       上下文向量 context shape: {context.shape}")
            print(f"       第1个头第1个token的上下文向量前5维:")
            print(f"         {context[0, 0, 0, :5].detach().numpy()}")

        # 重新组织输出
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        output = self.out_linear(context)

        if debug:
            print(f"       最终输出 shape: {output.shape}")
            print(f"       输出第1个token前5维: {output[0, 0, :5].detach().numpy()}")

        return output


class TransformerBlock(nn.Module):
    """Transformer编码器块"""

    def __init__(self, d_model, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None, debug=False):
        if debug:
            print(f"  🔍 TransformerBlock 详细计算过程:")
            print(f"     输入 x shape: {x.shape}")
            print(f"     输入第1个token前5维: {x[0, 0, :5].detach().numpy()}")

        # 多头注意力 + 残差连接
        attn_output = self.attention(x, mask, debug=debug)

        if debug:
            print(f"     注意力输出第1个token前5维: {attn_output[0, 0, :5].detach().numpy()}")

        x_after_attn = x + attn_output

        if debug:
            print(f"     残差连接后第1个token前5维: {x_after_attn[0, 0, :5].detach().numpy()}")

        x_norm1 = self.norm1(x_after_attn)

        if debug:
            print(f"     LayerNorm1后第1个token前5维: {x_norm1[0, 0, :5].detach().numpy()}")

        # MLP + 残差连接
        mlp_output = self.mlp(x_norm1)

        if debug:
            print(f"     MLP输出第1个token前5维: {mlp_output[0, 0, :5].detach().numpy()}")

        x_after_mlp = x_norm1 + mlp_output

        if debug:
            print(f"     MLP残差连接后第1个token前5维: {x_after_mlp[0, 0, :5].detach().numpy()}")

        x_final = self.norm2(x_after_mlp)

        if debug:
            print(f"     LayerNorm2后(最终输出)第1个token前5维: {x_final[0, 0, :5].detach().numpy()}")

        return x_final


class VisionTransformer(nn.Module):
    """Vision Transformer 图像编码器"""

    def __init__(self, image_size=224, patch_size=16, num_channels=3,
                 d_model=768, num_heads=12, num_layers=12, mlp_dim=3072):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.d_model = d_model

        # Patch embedding层
        self.patch_embedding = nn.Conv2d(
            num_channels, d_model, kernel_size=patch_size, stride=patch_size
        )

        # 位置编码
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, d_model)
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer编码器层
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, mlp_dim)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, debug=False):
        batch_size = x.shape[0]

        if debug:
            print(f"🔍 VisionTransformer 详细计算过程:")
            print(f"   输入图像 shape: {x.shape}")
            print(f"   图像像素值范围: [{x.min().item():.3f}, {x.max().item():.3f}]")
            print(f"   左上角3x3像素值 (第1个通道):")
            print(f"     {x[0, 0, :3, :3].detach().numpy()}")

        # 将图像分割成patches并嵌入
        patches = self.patch_embedding(x)  # (B, d_model, H/P, W/P)

        if debug:
            print(f"   Patch embedding后 shape: {patches.shape}")
            print(f"   第1个patch的前5维特征: {patches[0, :5, 0, 0].detach().numpy()}")

        x = patches.flatten(2).transpose(1, 2)  # (B, num_patches, d_model)

        if debug:
            print(f"   展平后 shape: {x.shape}")
            print(f"   第1个patch前5维: {x[0, 0, :5].detach().numpy()}")

        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        if debug:
            print(f"   CLS token shape: {cls_tokens.shape}")
            print(f"   CLS token前5维: {cls_tokens[0, 0, :5].detach().numpy()}")

        x = torch.cat([cls_tokens, x], dim=1)

        if debug:
            print(f"   添加CLS token后 shape: {x.shape}")

        # 添加位置编码
        x = x + self.position_embedding

        if debug:
            print(f"   位置编码 shape: {self.position_embedding.shape}")
            print(f"   位置编码前5维 (CLS): {self.position_embedding[0, 0, :5].detach().numpy()}")
            print(f"   位置编码前5维 (第1个patch): {self.position_embedding[0, 1, :5].detach().numpy()}")
            print(f"   添加位置编码后CLS token前5维: {x[0, 0, :5].detach().numpy()}")

        # 通过Transformer层
        for i, block in enumerate(self.transformer_blocks):
            if debug and i < 2:  # 只显示前2层的详细信息
                print(f"\n   === Transformer Block {i+1} ===")
                x = block(x, debug=True)
            else:
                x = block(x, debug=False)

        x = self.norm(x)

        if debug:
            print(f"   最终LayerNorm后 shape: {x.shape}")
            print(f"   CLS token最终特征前5维: {x[0, 0, :5].detach().numpy()}")

        # 返回CLS token的特征
        return x[:, 0]


class TextTransformer(nn.Module):
    """文本Transformer编码器"""

    def __init__(self, vocab_size=49408, max_length=77, d_model=512,
                 num_heads=8, num_layers=12, mlp_dim=2048):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # 位置编码
        self.position_embedding = nn.Parameter(
            torch.randn(1, max_length, d_model)
        )

        # Transformer编码器层
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, mlp_dim)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids, attention_mask=None, debug=False):
        batch_size, seq_len = input_ids.shape

        if debug:
            print(f"🔍 TextTransformer 详细计算过程:")
            print(f"   输入token IDs shape: {input_ids.shape}")
            print(f"   前10个token IDs: {input_ids[0, :10].detach().numpy()}")
            if attention_mask is not None:
                print(f"   attention_mask前10位: {attention_mask[0, :10].detach().numpy()}")

        # Token embedding + 位置编码
        token_embeds = self.token_embedding(input_ids)

        if debug:
            print(f"   Token embedding shape: {token_embeds.shape}")
            print(f"   第1个token embedding前5维: {token_embeds[0, 0, :5].detach().numpy()}")
            print(f"   第2个token embedding前5维: {token_embeds[0, 1, :5].detach().numpy()}")

        pos_embed = self.position_embedding[:, :seq_len, :]

        if debug:
            print(f"   位置编码 shape: {pos_embed.shape}")
            print(f"   位置0编码前5维: {pos_embed[0, 0, :5].detach().numpy()}")
            print(f"   位置1编码前5维: {pos_embed[0, 1, :5].detach().numpy()}")

        x = token_embeds + pos_embed

        if debug:
            print(f"   Token + 位置编码后第1个token前5维: {x[0, 0, :5].detach().numpy()}")

        # 通过Transformer层
        for i, block in enumerate(self.transformer_blocks):
            if debug and i < 2:  # 只显示前2层的详细信息
                print(f"\n   === Text Transformer Block {i+1} ===")
                x = block(x, attention_mask, debug=True)
            else:
                x = block(x, attention_mask, debug=False)

        x = self.norm(x)

        # 返回最后一个有效token的特征
        if attention_mask is not None:
            last_token_indices = (attention_mask.sum(dim=1) - 1).long()
            batch_indices = torch.arange(x.shape[0], device=x.device)
            result = x[batch_indices, last_token_indices]

            if debug:
                print(f"   最后有效token位置: {last_token_indices.detach().numpy()}")
                print(f"   最后有效token特征前5维: {result[0, :5].detach().numpy()}")
        else:
            result = x[:, -1]
            if debug:
                print(f"   使用最后一个token特征前5维: {result[0, :5].detach().numpy()}")

        return result


class SimpleCLIP(nn.Module):
    """简化版CLIP模型实现"""

    def __init__(self, image_encoder_config=None, text_encoder_config=None, projection_dim=512):
        super().__init__()

        # 默认配置
        if image_encoder_config is None:
            image_encoder_config = {
                'image_size': 224, 'patch_size': 16, 'num_channels': 3,
                'd_model': 768, 'num_heads': 12, 'num_layers': 12, 'mlp_dim': 3072
            }

        if text_encoder_config is None:
            text_encoder_config = {
                'vocab_size': 49408, 'max_length': 77, 'd_model': 512,
                'num_heads': 8, 'num_layers': 12, 'mlp_dim': 2048
            }

        # 图像编码器
        self.vision_model = VisionTransformer(**image_encoder_config)

        # 文本编码器
        self.text_model = TextTransformer(**text_encoder_config)

        # 投影层
        self.visual_projection = nn.Linear(image_encoder_config['d_model'], projection_dim)
        self.text_projection = nn.Linear(text_encoder_config['d_model'], projection_dim)

        # 温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_image(self, image, debug=False):
        """编码图像"""
        if debug:
            print(f"🔍 图像编码过程:")

        image_features = self.vision_model(image, debug=debug)

        if debug:
            print(f"   Vision模型输出 shape: {image_features.shape}")
            print(f"   Vision特征前5维: {image_features[0, :5].detach().numpy()}")

        projected_features = self.visual_projection(image_features)

        if debug:
            print(f"   投影后特征 shape: {projected_features.shape}")
            print(f"   投影后特征前5维: {projected_features[0, :5].detach().numpy()}")

        normalized_features = F.normalize(projected_features, dim=-1)

        if debug:
            print(f"   归一化后特征前5维: {normalized_features[0, :5].detach().numpy()}")
            print(f"   特征向量长度: {torch.norm(normalized_features[0]).item():.6f}")

        return normalized_features

    def encode_text(self, input_ids, attention_mask=None, debug=False):
        """编码文本"""
        if debug:
            print(f"🔍 文本编码过程:")

        text_features = self.text_model(input_ids, attention_mask, debug=debug)

        if debug:
            print(f"   Text模型输出 shape: {text_features.shape}")
            print(f"   Text特征前5维: {text_features[0, :5].detach().numpy()}")

        projected_features = self.text_projection(text_features)

        if debug:
            print(f"   投影后特征 shape: {projected_features.shape}")
            print(f"   投影后特征前5维: {projected_features[0, :5].detach().numpy()}")

        normalized_features = F.normalize(projected_features, dim=-1)

        if debug:
            print(f"   归一化后特征前5维: {normalized_features[0, :5].detach().numpy()}")
            print(f"   特征向量长度: {torch.norm(normalized_features[0]).item():.6f}")

        return normalized_features

    def forward(self, image, input_ids, attention_mask=None, debug=False):
        """前向传播"""
        if debug:
            print(f"🚀 CLIP完整前向传播过程:")
            print(f"=" * 80)

        image_features = self.encode_image(image, debug=debug)

        if debug:
            print(f"\n" + "=" * 80)

        text_features = self.encode_text(input_ids, attention_mask, debug=debug)

        if debug:
            print(f"\n🔍 相似度计算过程:")
            print(f"   图像特征 shape: {image_features.shape}")
            print(f"   文本特征 shape: {text_features.shape}")

        # 计算相似度
        logit_scale = self.logit_scale.exp()

        if debug:
            print(f"   温度参数 logit_scale: {logit_scale.item():.6f}")

        # 计算点积
        dot_product = image_features @ text_features.t()

        if debug:
            print(f"   点积结果 shape: {dot_product.shape}")
            print(f"   点积矩阵:")
            print(f"     {dot_product.detach().numpy()}")

        logits_per_image = logit_scale * dot_product
        logits_per_text = logits_per_image.t()

        if debug:
            print(f"   缩放后的logits_per_image:")
            print(f"     {logits_per_image.detach().numpy()}")
            print(f"   logits_per_text:")
            print(f"     {logits_per_text.detach().numpy()}")

            # 计算概率
            probs_i2t = F.softmax(logits_per_image, dim=-1)
            probs_t2i = F.softmax(logits_per_text, dim=-1)

            print(f"   图像到文本概率:")
            print(f"     {probs_i2t.detach().numpy()}")
            print(f"   文本到图像概率:")
            print(f"     {probs_t2i.detach().numpy()}")

        return {
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text,
            'image_embeds': image_features,
            'text_embeds': text_features
        }


class CLIPExplanation:
    """
    CLIP (Contrastive Language-Image Pre-training) 模型详细讲解

    这是一个从零实现的CLIP模型，不依赖transformers库
    """

    def __init__(self):
        # 创建简化版CLIP模型
        self.model = SimpleCLIP()
        print("✅ 成功创建简化版CLIP模型（无需transformers库）")

    def detailed_computation_example(self):
        """详细的计算过程示例"""
        print("=" * 80)
        print("CLIP 详细计算过程示例")
        print("=" * 80)

        # 创建具体的示例数据
        print("\n📝 创建示例数据:")

        # 创建一个简单的图像（红色方块）
        image = torch.zeros(1, 3, 224, 224)
        image[0, 0, 50:150, 50:150] = 1.0  # 红色通道
        print(f"   创建了一个红色方块图像 shape: {image.shape}")

        # 创建文本tokens（模拟 "a red square" 的tokenization）
        input_ids = torch.tensor([[49406, 320, 1000, 5000, 49407] + [0] * 72])  # 模拟token序列
        attention_mask = torch.tensor([[1, 1, 1, 1, 1] + [0] * 72])  # 前5个token有效

        print(f"   文本tokens: {input_ids[0, :10].numpy()} (前10个)")
        print(f"   注意力mask: {attention_mask[0, :10].numpy()} (前10个)")

        print(f"\n🚀 开始CLIP完整计算过程:")
        print("=" * 80)

        # 执行前向传播，开启debug模式
        with torch.no_grad():
            outputs = self.model(image, input_ids, attention_mask, debug=True)

        print(f"\n📊 最终结果:")
        print(f"   图像-文本相似度: {outputs['logits_per_image'][0, 0].item():.6f}")
        print(f"   图像特征向量长度: {torch.norm(outputs['image_embeds'][0]).item():.6f}")
        print(f"   文本特征向量长度: {torch.norm(outputs['text_embeds'][0]).item():.6f}")

    def attention_visualization_example(self):
        """注意力机制可视化示例"""
        print("\n" + "=" * 80)
        print("注意力机制详细计算示例")
        print("=" * 80)

        # 创建一个小的示例来演示注意力计算
        print("\n📝 创建简化的注意力示例:")
        d_model = 8  # 简化的模型维度
        num_heads = 2
        seq_len = 4
        batch_size = 1

        # 创建简单的输入
        x = torch.randn(batch_size, seq_len, d_model)
        print(f"   输入序列 shape: {x.shape}")
        print(f"   输入矩阵:")
        for i in range(seq_len):
            print(f"     token {i}: {x[0, i].detach().numpy()}")

        # 创建注意力层
        attention = MultiHeadAttention(d_model, num_heads)

        print(f"\n🔍 注意力计算详细过程:")
        with torch.no_grad():
            output = attention(x, debug=True)

        print(f"\n📊 注意力计算总结:")
        print(f"   输入shape: {x.shape}")
        print(f"   输出shape: {output.shape}")
        print(f"   输出矩阵:")
        for i in range(seq_len):
            print(f"     token {i}: {output[0, i].detach().numpy()}")

    def step_by_step_training_example(self):
        """逐步训练过程示例"""
        print("\n" + "=" * 80)
        print("CLIP训练过程详细示例")
        print("=" * 80)

        print("\n📝 模拟训练batch:")
        batch_size = 3

        # 创建batch数据
        images = torch.randn(batch_size, 3, 224, 224)
        input_ids = torch.randint(1, 1000, (batch_size, 77))
        attention_mask = torch.ones(batch_size, 77)

        print(f"   Batch大小: {batch_size}")
        print(f"   图像batch shape: {images.shape}")
        print(f"   文本batch shape: {input_ids.shape}")

        print(f"\n🔍 训练前向传播:")
        with torch.no_grad():
            # 编码图像和文本
            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(input_ids, attention_mask)

            print(f"   图像特征 shape: {image_features.shape}")
            print(f"   文本特征 shape: {text_features.shape}")

            # 计算相似度矩阵
            logit_scale = self.model.logit_scale.exp()
            similarity_matrix = logit_scale * image_features @ text_features.t()

            print(f"\n📊 相似度矩阵 ({batch_size}x{batch_size}):")
            print(f"   {similarity_matrix.detach().numpy()}")

            # 创建标签（对角线为正样本）
            labels = torch.arange(batch_size)
            print(f"   标签: {labels.numpy()}")

            # 计算损失
            loss_i2t = F.cross_entropy(similarity_matrix, labels)
            loss_t2i = F.cross_entropy(similarity_matrix.t(), labels)
            total_loss = (loss_i2t + loss_t2i) / 2

            print(f"\n💰 损失计算:")
            print(f"   图像到文本损失: {loss_i2t.item():.6f}")
            print(f"   文本到图像损失: {loss_t2i.item():.6f}")
            print(f"   总损失: {total_loss.item():.6f}")

            # 显示概率分布
            probs_i2t = F.softmax(similarity_matrix, dim=-1)
            probs_t2i = F.softmax(similarity_matrix.t(), dim=-1)

            print(f"\n📈 概率分布:")
            print(f"   图像到文本概率:")
            for i in range(batch_size):
                print(f"     图像{i}: {probs_i2t[i].detach().numpy()}")
            print(f"   文本到图像概率:")
            for i in range(batch_size):
                print(f"     文本{i}: {probs_t2i[i].detach().numpy()}")


def main():
    """主函数：运行CLIP详细讲解"""
    print("CLIP (Contrastive Language-Image Pre-training) 详细计算过程讲解")
    print("🚀 从零实现版本 - 包含所有中间值")
    print("=" * 80)

    try:
        # 创建讲解实例
        clip_demo = CLIPExplanation()

        # 1. 详细计算过程示例
        clip_demo.detailed_computation_example()

        # 2. 注意力机制可视化
        clip_demo.attention_visualization_example()

        # 3. 训练过程示例
        clip_demo.step_by_step_training_example()

        print("\n" + "=" * 80)
        print("✅ CLIP 详细计算过程讲解完成！")
        print("💡 所有中间值都已显示，帮助理解每一步的计算过程")
        print("=" * 80)

    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()