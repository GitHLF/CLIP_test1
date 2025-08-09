import numpy as np
import torch
import torch.nn.functional as F

from .image_processor import SimpleImageProcessor


def extract_attention_weights(model, tokenizer, image, texts, device="cpu"):
    """提取CLIP模型的注意力权重"""
    print("提取注意力权重...")

    # 图像预处理
    processor = SimpleImageProcessor()
    image_tensor = processor(image).unsqueeze(0).to(device)

    # 文本tokenization
    text_inputs = tokenizer(
        texts,  # 使用正确的参数名
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    attention_data = {}

    with torch.no_grad():
        # 获取图像注意力
        try:
            vision_outputs = model.vision_model(
                pixel_values=image_tensor,
                output_attentions=True
            )

            if hasattr(vision_outputs, 'attentions') and vision_outputs.attentions:
                attention_data['vision_attentions'] = [att.cpu() for att in vision_outputs.attentions]
                attention_data['vision_hidden_states'] = vision_outputs.last_hidden_state.cpu()
                print(f"获取到 {len(attention_data['vision_attentions'])} 层图像注意力")

                # 打印调试信息
                first_att = attention_data['vision_attentions'][0]
                print(f"图像注意力shape: {first_att.shape}")  # 应该是 [batch, heads, seq_len, seq_len]
                seq_len = first_att.shape[-1]
                num_patches = seq_len - 1  # 减去CLS token
                print(f"检测到 {num_patches} 个patches ({int(np.sqrt(num_patches))}x{int(np.sqrt(num_patches))})")

        except Exception as e:
            print(f"获取图像注意力失败: {e}")
            attention_data['vision_attentions'] = None

        # 获取文本注意力
        try:
            text_outputs = model.text_model(
                input_ids=text_inputs['input_ids'],
                attention_mask=text_inputs['attention_mask'],
                output_attentions=True
            )

            if hasattr(text_outputs, 'attentions') and text_outputs.attentions:
                attention_data['text_attentions'] = [att.cpu() for att in text_outputs.attentions]
                attention_data['text_hidden_states'] = text_outputs.last_hidden_state.cpu()
                print(f"获取到 {len(attention_data['text_attentions'])} 层文本注意力")

                # 打印调试信息
                first_text_att = attention_data['text_attentions'][0]
                print(f"文本注意力shape: {first_text_att.shape}")

        except Exception as e:
            print(f"获取文本注意力失败: {e}")
            attention_data['text_attentions'] = None

        # 获取跨模态相似度
        try:
            full_outputs = model(
                pixel_values=image_tensor,
                input_ids=text_inputs['input_ids'],
                attention_mask=text_inputs['attention_mask']
            )

            attention_data['image_embeds'] = full_outputs.image_embeds.cpu()
            attention_data['text_embeds'] = full_outputs.text_embeds.cpu()
            attention_data['logits_per_image'] = full_outputs.logits_per_image.cpu()
            attention_data['similarities'] = F.softmax(full_outputs.logits_per_image, dim=-1).cpu()

            print(f"跨模态特征shape: image_embeds={full_outputs.image_embeds.shape}, text_embeds={full_outputs.text_embeds.shape}")

        except Exception as e:
            print(f"获取跨模态特征失败: {e}")

    attention_data['texts'] = texts  # 使用正确的参数名
    attention_data['text_inputs'] = {k: v.cpu() for k, v in text_inputs.items()}

    return attention_data

