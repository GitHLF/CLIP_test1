import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 设置HuggingFace镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 忽略警告
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
from PIL import Image
import json
from pathlib import Path


# 直接导入核心组件，绕过image_processing问题
def load_clip_directly():
    """直接加载CLIP模型，绕过image_processing问题"""
    try:
        # 直接从modeling模块导入，避免processing问题
        from transformers.models.clip.modeling_clip import CLIPModel
        from transformers.models.clip.configuration_clip import CLIPConfig
        from transformers import AutoTokenizer

        print("✅ 成功导入核心CLIP组件")
        return CLIPModel, CLIPConfig, AutoTokenizer, True
    except ImportError as e:
        print(f"❌ 核心组件导入失败: {e}")
        return None, None, None, False

# 创建简化的处理器
class SimpleProcessor:
    """简化的CLIP处理器，避免image_processing问题"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text=None, images=None, return_tensors="pt", padding=True, debug=False):
        if debug:
            print("🔍 SimpleProcessor 详细处理过程:")

        # 处理文本
        if text is not None:
            if isinstance(text, str):
                text = [text]

            if debug:
                print(f"   输入文本: {text}")

            # 使用tokenizer处理文本
            text_inputs = self.tokenizer(
                text,
                return_tensors=return_tensors,
                padding=padding,
                truncation=True,
                max_length=77
            )

            if debug:
                print(f"   Tokenization结果:")
                for i, txt in enumerate(text):
                    tokens = text_inputs['input_ids'][i]
                    print(f"     文本{i}: '{txt}'")
                    print(f"     Token IDs: {tokens[:10].tolist()}... (前10个)")
                    print(f"     Token长度: {tokens.shape[0]}")
                    print(f"     实际有效长度: {text_inputs['attention_mask'][i].sum().item()}")
        else:
            text_inputs = {}

        # 处理图像
        if images is not None:
            if isinstance(images, Image.Image):
                images = [images]

            if debug:
                print(f"   输入图像数量: {len(images)}")
                print(f"   第1张图像尺寸: {images[0].size}")

            # 简单的图像预处理
            pixel_values = []
            for i, img in enumerate(images):
                if debug and i == 0:  # 只显示第一张图像的详细处理
                    print(f"   图像{i}预处理过程:")
                    print(f"     原始尺寸: {img.size}")

                # 调整大小到224x224
                img = img.resize((224, 224))
                if debug and i == 0:
                    print(f"     调整后尺寸: {img.size}")

                # 转换为tensor
                img_array = torch.tensor(list(img.getdata())).float()
                img_array = img_array.view(224, 224, 3)

                if debug and i == 0:
                    print(f"     转换为tensor shape: {img_array.shape}")
                    print(f"     像素值范围: [{img_array.min():.1f}, {img_array.max():.1f}]")
                    print(f"     左上角3x3像素 (R通道):")
                    print(f"       {img_array[:3, :3, 0].numpy()}")

                # 转换为CHW格式并归一化
                img_tensor = img_array.permute(2, 0, 1) / 255.0

                if debug and i == 0:
                    print(f"     CHW格式 shape: {img_tensor.shape}")
                    print(f"     归一化后范围: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")

                # 标准化 (ImageNet标准)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_tensor = (img_tensor - mean) / std

                if debug and i == 0:
                    print(f"     ImageNet标准化后:")
                    print(f"       均值: {mean.flatten().tolist()}")
                    print(f"       标准差: {std.flatten().tolist()}")
                    print(f"       标准化后范围: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
                    print(f"       R通道左上角3x3:")
                    print(f"         {img_tensor[0, :3, :3].numpy()}")

                pixel_values.append(img_tensor)

            pixel_values = torch.stack(pixel_values)
            text_inputs['pixel_values'] = pixel_values

            if debug:
                print(f"   最终pixel_values shape: {pixel_values.shape}")

        return text_inputs

# 尝试导入组件
CLIPModel, CLIPConfig, AutoTokenizer, clip_available = load_clip_directly()

class CLIPExplanation:
    """
    CLIP (Contrastive Language-Image Pre-training) 模型详细讲解

    CLIP是OpenAI开发的多模态模型，能够理解图像和文本之间的关系
    通过对比学习的方式，将图像和文本映射到同一个特征空间中
    """

    def __init__(self, use_mirror=True, mirror_url='https://hf-mirror.com'):
        print("🔄 正在加载预训练的CLIP模型...")

        if not clip_available:
            print("❌ CLIP组件不可用，使用演示模式")
            self._create_demo_version()
            return

        model_loaded = False

        # 检查本地模型文件
        model_dir = Path("./models")
        possible_paths = [
            model_dir / "models--openai--clip-vit-base-patch32" / "snapshots",
            model_dir / "clip-model",
            model_dir
        ]

        local_model_path = None
        for path in possible_paths:
            if path.exists():
                # 查找包含config.json的子目录
                for subdir in path.rglob("*"):
                    if subdir.is_dir() and (subdir / "config.json").exists():
                        local_model_path = subdir
                        break
                if local_model_path:
                    break

        if local_model_path:
            print(f"📁 找到本地模型文件: {local_model_path}")

            try:
                print("🔧 直接从本地文件加载模型...")

                # 加载配置
                config_path = local_model_path / "config.json"
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                config = CLIPConfig.from_dict(config_dict)

                # 加载模型
                self.model = CLIPModel.from_pretrained(
                    str(local_model_path),
                    config=config,
                    local_files_only=True
                )

                # 加载tokenizer
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        str(local_model_path),
                        local_files_only=True
                    )
                except:
                    # 如果本地tokenizer加载失败，使用在线版本
                    print("⚠️  本地tokenizer加载失败，使用在线版本...")
                    self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

                # 创建简化处理器
                self.processor = SimpleProcessor(self.tokenizer)

                print("✅ 成功从本地文件加载CLIP模型")
                model_loaded = True

            except Exception as e:
                print(f"❌ 本地文件加载失败: {e}")
                print("详细错误信息:")
                import traceback
                traceback.print_exc()

        # 如果本地加载失败，尝试在线加载
        if not model_loaded:
            try:
                print("🌐 尝试在线加载...")

                # 直接使用模型名称加载
                self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
                self.processor = SimpleProcessor(self.tokenizer)

                print("✅ 成功在线加载CLIP模型")
                model_loaded = True

            except Exception as e:
                print(f"❌ 在线加载也失败: {e}")

        # 如果所有方法都失败，创建演示版本
        if not model_loaded:
            print("❌ 无法加载真实模型，创建演示版本")
            self._create_demo_version()
            model_loaded = True

        if model_loaded and not hasattr(self, 'demo_mode'):
            print(f"📊 模型信息:")
            print(f"   - 模型大小: ~605MB")
            print(f"   - 缓存位置: ./models/")
            print(f"   - 图像编码器: Vision Transformer")
            print(f"   - 文本编码器: Transformer")

            # 测试模型
            self._test_model()

    def _test_model(self):
        """测试模型是否正常工作"""
        try:
            print("🧪 测试模型功能...")

            # 创建测试数据
            test_image = Image.new('RGB', (224, 224), color='red')
            test_texts = ["a red image", "a blue image"]

            # 处理输入
            inputs = self.processor(text=test_texts, images=test_image)

            # 模型推理
            with torch.no_grad():
                outputs = self.model(**inputs)

            print("✅ 模型测试成功！")

        except Exception as e:
            print(f"⚠️  模型测试失败: {e}")

    def _create_demo_version(self):
        """创建一个用于演示的简化版本"""
        print("🔧 创建演示版CLIP模型...")

        class DemoModel:
            def __init__(self):
                self.config = type('Config', (), {
                    'hidden_size': 768,
                    'num_attention_heads': 12,
                    'image_size': 224,
                    'patch_size': 16,
                    'vocab_size': 49408,
                    'max_position_embeddings': 77
                })()

                self.vision_model = type('VisionModel', (), {
                    'config': self.config,
                    'encoder': type('Encoder', (), {
                        'layers': [None] * 12
                    })()
                })()

                self.text_model = type('TextModel', (), {
                    'config': self.config,
                    'encoder': type('Encoder', (), {
                        'layers': [None] * 12
                    })()
                })()

                self.visual_projection = type('Projection', (), {
                    'in_features': 768,
                    'out_features': 512
                })()

                self.text_projection = type('Projection', (), {
                    'in_features': 512,
                    'out_features': 512
                })()

                self.logit_scale = torch.tensor(2.6593)

            def parameters(self):
                return [torch.randn(1000) for _ in range(100)]

            def __call__(self, **kwargs):
                # 模拟模型输出
                batch_size = 1
                if 'pixel_values' in kwargs:
                    batch_size = kwargs['pixel_values'].shape[0]

                text_batch_size = 1
                if 'input_ids' in kwargs:
                    text_batch_size = kwargs['input_ids'].shape[0]

                return type('Output', (), {
                    'image_embeds': torch.randn(batch_size, 512),
                    'text_embeds': torch.randn(text_batch_size, 512),
                    'logits_per_image': torch.randn(batch_size, text_batch_size),
                    'logits_per_text': torch.randn(text_batch_size, batch_size)
                })()

        class DemoProcessor:
            def __call__(self, text=None, images=None, return_tensors="pt", padding=True, debug=False):
                batch_size = len(text) if text else 1
                return {
                    'pixel_values': torch.randn(1, 3, 224, 224),
                    'input_ids': torch.randint(0, 1000, (batch_size, 77)),
                    'attention_mask': torch.ones(batch_size, 77)
                }

        self.model = DemoModel()
        self.processor = DemoProcessor()
        self.demo_mode = True
        print("✅ 演示版CLIP模型创建完成")

    def explain_clip_architecture(self):
        """详细解释CLIP的内部架构"""
        print("=" * 60)
        print("CLIP 模型架构详解")
        print("=" * 60)

        if hasattr(self, 'demo_mode'):
            print("⚠️  当前为演示模式，显示理论架构")

        print("\n1. 整体架构:")
        print("   CLIP = 图像编码器 + 文本编码器 + 对比学习")
        print("   - 图像编码器: Vision Transformer (ViT) 或 ResNet")
        print("   - 文本编码器: Transformer")
        print("   - 目标: 将图像和文本映射到同一个特征空间")

        print("\n2. 图像编码器 (Vision Transformer):")
        print("   - 将图像分割成固定大小的patches (如16x16)")
        print("   - 每个patch通过线性投影变成embedding")
        print("   - 添加位置编码")
        print("   - 通过多层Transformer编码器处理")
        print("   - 输出: 图像的特征向量表示")

        print("\n3. 文本编码器 (Transformer):")
        print("   - 文本tokenization和embedding")
        print("   - 添加位置编码")
        print("   - 通过多层Transformer编码器处理")
        print("   - 输出: 文本的特征向量表示")

        print("\n4. 对比学习机制:")
        print("   - 计算图像和文本特征的余弦相似度")
        print("   - 正样本对(匹配的图像-文本)相似度最大化")
        print("   - 负样本对(不匹配的图像-文本)相似度最小化")

    def demonstrate_clip_components(self):
        """演示CLIP各个组件的工作原理"""
        print("\n" + "=" * 60)
        print("CLIP 组件演示")
        print("=" * 60)

        if hasattr(self, 'demo_mode'):
            print("⚠️  当前为演示模式，显示模拟数据")

        # 1. 图像编码器演示
        print("\n1. 图像编码器工作流程:")
        dummy_image = torch.randn(1, 3, 224, 224)
        print(f"   输入图像shape: {dummy_image.shape}")

        vision_model = self.model.vision_model
        print(f"   Vision Transformer层数: {len(vision_model.encoder.layers)}")
        print(f"   隐藏层维度: {vision_model.config.hidden_size}")
        print(f"   注意力头数: {vision_model.config.num_attention_heads}")
        print(f"   图像尺寸: {vision_model.config.image_size}")
        print(f"   Patch尺寸: {vision_model.config.patch_size}")

        # 2. 文本编码器演示
        print("\n2. 文本编码器工作流程:")
        dummy_text = torch.randint(0, 1000, (1, 77))
        print(f"   输入文本tokens shape: {dummy_text.shape}")

        text_model = self.model.text_model
        print(f"   Text Transformer层数: {len(text_model.encoder.layers)}")
        print(f"   隐藏层维度: {text_model.config.hidden_size}")
        print(f"   词汇表大小: {text_model.config.vocab_size}")
        print(f"   最大序列长度: {text_model.config.max_position_embeddings}")

        # 3. 投影层信息
        print("\n3. 投影层信息:")
        print(f"   视觉投影维度: {self.model.visual_projection.in_features} → {self.model.visual_projection.out_features}")
        print(f"   文本投影维度: {self.model.text_projection.in_features} → {self.model.text_projection.out_features}")
        print(f"   温度参数: {self.model.logit_scale.item():.4f}")

    def detailed_inference_walkthrough(self):
        """详细的推理过程演示"""
        print("\n" + "=" * 80)
        print("CLIP 详细推理过程演示")
        print("=" * 80)

        if hasattr(self, 'demo_mode'):
            print("⚠️  当前为演示模式，显示模拟推理过程")
            self._demo_detailed_inference()
            return
        else:
            print("🎉 使用真实CLIP模型进行详细推理")

        # 准备具体的示例数据
        print("\n📝 1. 准备示例数据:")

        # 创建一个红色方块图像
        image = Image.new('RGB', (224, 224), color=(255, 0, 0))  # 纯红色
        texts = ["a red square", "a blue circle", "a green triangle"]

        print(f"   图像: 224x224 红色方块")
        print(f"   候选文本: {texts}")

        # 详细的数据预处理
        print(f"\n🔍 2. 数据预处理详细过程:")
        inputs = self.processor(text=texts, images=image, debug=True)

        print(f"\n🧠 3. 模型推理详细过程:")

        with torch.no_grad():
            # 分步骤进行推理，获取中间结果
            print(f"   🖼️  图像编码过程:")

            # 获取图像特征
            vision_outputs = self.model.vision_model(pixel_values=inputs['pixel_values'])
            image_features_raw = vision_outputs.last_hidden_state[:, 0, :]  # CLS token

            print(f"     Vision Transformer输出 shape: {vision_outputs.last_hidden_state.shape}")
            print(f"     CLS token特征 shape: {image_features_raw.shape}")
            print(f"     CLS token前5维: {image_features_raw[0, :5].detach().numpy()}")

            # 图像投影
            image_embeds = self.model.visual_projection(image_features_raw)
            print(f"     投影后图像特征 shape: {image_embeds.shape}")
            print(f"     投影后前5维: {image_embeds[0, :5].detach().numpy()}")

            # L2归一化
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            print(f"     归一化后前5维: {image_embeds[0, :5].detach().numpy()}")
            print(f"     特征向量长度: {torch.norm(image_embeds[0]).item():.6f}")

            print(f"\n   📝 文本编码过程:")

            # 获取文本特征
            text_outputs = self.model.text_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )

            # 获取最后一个token的特征（通常是EOS token）
            sequence_lengths = inputs['attention_mask'].sum(dim=1) - 1
            text_features_raw = text_outputs.last_hidden_state[
                torch.arange(text_outputs.last_hidden_state.shape[0]),
                sequence_lengths
            ]

            print(f"     Text Transformer输出 shape: {text_outputs.last_hidden_state.shape}")
            print(f"     最后token位置: {sequence_lengths.detach().numpy()}")
            print(f"     文本特征 shape: {text_features_raw.shape}")

            for i, text in enumerate(texts):
                print(f"     文本{i} '{text}' 特征前3维: {text_features_raw[i, :3].detach().numpy()}")

            # 文本投影
            text_embeds = self.model.text_projection(text_features_raw)
            print(f"     投影后文本特征 shape: {text_embeds.shape}")

            # L2归一化
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            print(f"     归一化后文本特征:")
            for i, text in enumerate(texts):
                print(f"       '{text}' 前3维: {text_embeds[i, :3].detach().numpy()}")
                print(f"       向量长度: {torch.norm(text_embeds[i]).item():.6f}")

            print(f"\n   🔗 相似度计算过程:")

            # 计算相似度矩阵
            logit_scale = self.model.logit_scale.exp()
            print(f"     温度参数 exp(logit_scale): {logit_scale.item():.6f}")

            # 点积计算
            similarity_raw = torch.matmul(image_embeds, text_embeds.t())
            print(f"     原始点积相似度:")
            for i, text in enumerate(texts):
                print(f"       与 '{text}': {similarity_raw[0, i].item():.6f}")

            # 温度缩放
            logits_per_image = logit_scale * similarity_raw
            print(f"     温度缩放后logits:")
            for i, text in enumerate(texts):
                print(f"       与 '{text}': {logits_per_image[0, i].item():.6f}")

            # Softmax概率
            probs = F.softmax(logits_per_image, dim=-1)
            print(f"     Softmax概率分布:")
            for i, text in enumerate(texts):
                print(f"       '{text}': {probs[0, i].item():.6f}")

            # 验证概率和
            print(f"     概率和: {probs.sum().item():.6f}")

            print(f"\n📊 4. 最终结果分析:")

            # 找到最佳匹配
            best_match_idx = torch.argmax(probs, dim=-1)
            best_text = texts[best_match_idx[0]]
            best_prob = probs[0, best_match_idx[0]].item()

            print(f"   最佳匹配: '{best_text}' (概率: {best_prob:.4f})")

            # 计算置信度
            sorted_probs, sorted_indices = torch.sort(probs[0], descending=True)
            confidence = sorted_probs[0] - sorted_probs[1]
            print(f"   置信度 (最高-次高): {confidence.item():.4f}")

            # 显示排序结果
            print(f"   完整排序:")
            for i, idx in enumerate(sorted_indices):
                text = texts[idx]
                prob = sorted_probs[i].item()
                print(f"     {i+1}. '{text}': {prob:.4f}")

    def _demo_detailed_inference(self):
        """演示模式的详细推理"""
        print("🎭 演示模式详细推理过程:")

        # 模拟数据
        texts = ["a red square", "a blue circle", "a green triangle"]

        print(f"\n📝 模拟数据:")
        print(f"   图像: 224x224 红色方块")
        print(f"   候选文本: {texts}")

        # 模拟特征
        image_features = torch.randn(1, 512)
        text_features = torch.randn(len(texts), 512)

        # 归一化
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        print(f"\n🧠 模拟推理过程:")
        print(f"   图像特征 shape: {image_features.shape}")
        print(f"   文本特征 shape: {text_features.shape}")

        # 相似度计算
        similarity = torch.matmul(image_features, text_features.t())
        logit_scale = 2.6593  # 典型的CLIP温度参数
        logits = logit_scale * similarity
        probs = F.softmax(logits, dim=-1)

        print(f"   相似度计算:")
        for i, text in enumerate(texts):
            print(f"     '{text}': 相似度={similarity[0,i].item():.4f}, 概率={probs[0,i].item():.4f}")

    def clip_inference_example(self):
        """CLIP推理过程的完整示例"""
        print("\n" + "=" * 60)
        print("CLIP 推理示例")
        print("=" * 60)

        if hasattr(self, 'demo_mode'):
            print("⚠️  当前为演示模式，显示模拟推理过程")
        else:
            print("🎉 使用真实CLIP模型进行推理")

        # 准备示例数据
        image = Image.new('RGB', (224, 224), color='red')
        texts = ["a red image", "a blue image", "a green image", "a cat", "a dog"]

        print("\n1. 数据预处理:")
        print(f"   图像尺寸: {image.size}")
        print(f"   候选文本: {texts}")

        # 使用processor处理输入
        inputs = self.processor(text=texts, images=image, return_tensors="pt", padding=True)
        print(f"   处理后的图像tensor shape: {inputs['pixel_values'].shape}")
        print(f"   处理后的文本tokens shape: {inputs['input_ids'].shape}")
        print(f"   注意力mask shape: {inputs['attention_mask'].shape}")

        print("\n2. 模型推理过程:")

        if hasattr(self, 'demo_mode'):
            # 演示模式：创建模拟输出
            print("   🔧 生成模拟推理结果...")

            # 模拟特征
            image_features = torch.randn(1, 512)
            text_features = torch.randn(len(texts), 512)

            print(f"   图像特征维度: {image_features.shape}")
            print(f"   文本特征维度: {text_features.shape}")

            # 模拟相似度计算
            similarity_matrix = torch.matmul(image_features, text_features.t())
            print(f"   图像-文本相似度矩阵: {similarity_matrix.shape}")

            # 模拟概率分布
            probs = F.softmax(similarity_matrix, dim=-1)

            print("\n3. 推理结果 (模拟):")
            for i, text in enumerate(texts):
                print(f"   '{text}': {probs[0][i].item():.4f}")
        else:
            # 真实模式：使用实际模型
            with torch.no_grad():
                outputs = self.model(**inputs)

                image_features = outputs.image_embeds
                text_features = outputs.text_embeds

                print(f"   图像特征维度: {image_features.shape}")
                print(f"   文本特征维度: {text_features.shape}")

                logits_per_image = outputs.logits_per_image
                print(f"   图像-文本相似度矩阵: {logits_per_image.shape}")

                probs = logits_per_image.softmax(dim=-1)

                print("\n3. 推理结果 (真实模型):")
                for i, text in enumerate(texts):
                    print(f"   '{text}': {probs[0][i].item():.4f}")

    def analyze_attention_patterns(self):
        """分析注意力模式（如果可能的话）"""
        print("\n" + "=" * 60)
        print("CLIP 注意力模式分析")
        print("=" * 60)

        if hasattr(self, 'demo_mode'):
            print("⚠️  演示模式：无法获取真实注意力权重")
            print("💡 注意力机制说明:")
            print("   - Vision Transformer中，每个patch都会与其他patch交互")
            print("   - CLS token通过注意力机制聚合所有patch的信息")
            print("   - 文本Transformer中，每个token关注上下文信息")
            print("   - 注意力权重反映了模型关注的重点区域")
            return

        print("🔍 尝试分析注意力模式...")

        # 创建测试数据
        image = Image.new('RGB', (224, 224), color='red')
        text = "a red image"

        inputs = self.processor(text=text, images=image)

        try:
            # 尝试获取注意力权重（需要模型支持输出注意力）
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)

            if hasattr(outputs, 'vision_model_output') and hasattr(outputs.vision_model_output, 'attentions'):
                vision_attentions = outputs.vision_model_output.attentions
                print(f"   Vision注意力层数: {len(vision_attentions)}")
                print(f"   最后一层注意力 shape: {vision_attentions[-1].shape}")

                # 分析CLS token的注意力
                cls_attention = vision_attentions[-1][0, :, 0, :]  # [num_heads, num_patches+1]
                print(f"   CLS token注意力权重 shape: {cls_attention.shape}")

                # 显示每个头的注意力分布
                for head in range(min(3, cls_attention.shape[0])):  # 只显示前3个头
                    attention_weights = cls_attention[head]
                    print(f"   注意力头{head}:")
                    print(f"     对自己(CLS): {attention_weights[0].item():.4f}")
                    print(f"     对patch的平均注意力: {attention_weights[1:].mean().item():.4f}")
                    print(f"     最大注意力patch: {attention_weights[1:].argmax().item()}")

            if hasattr(outputs, 'text_model_output') and hasattr(outputs.text_model_output, 'attentions'):
                text_attentions = outputs.text_model_output.attentions
                print(f"   Text注意力层数: {len(text_attentions)}")
                print(f"   最后一层注意力 shape: {text_attentions[-1].shape}")

        except Exception as e:
            print(f"   ⚠️  无法获取注意力权重: {e}")
            print("   💡 可能需要在模型调用时设置 output_attentions=True")

def main():
    """主函数：运行CLIP详细讲解"""
    print("CLIP (Contrastive Language-Image Pre-training) 详细讲解")
    print("🚀 直接加载版本 - 包含详细中间量分析")
    print("=" * 80)

    try:
        # 创建讲解实例
        clip_demo = CLIPExplanation()

        # 1. 架构讲解
        clip_demo.explain_clip_architecture()

        # 2. 组件演示
        clip_demo.demonstrate_clip_components()

        # 3. 详细推理过程演示
        clip_demo.detailed_inference_walkthrough()

        # 4. 注意力模式分析
        clip_demo.analyze_attention_patterns()

        # 5. 基础推理示例
        clip_demo.clip_inference_example()

        print("\n" + "=" * 80)
        print("✅ CLIP 详细讲解完成！")
        if hasattr(clip_demo, 'demo_mode'):
            print("📝 当前为演示模式，已展示CLIP的核心概念和架构")
            print("💡 模型文件已下载，但版本兼容性问题导致无法加载")
            print("🔧 建议降级transformers版本: pip install transformers==4.21.0")
        else:
            print("🎉 成功使用真实CLIP模型！")
            print("💾 模型已成功加载并缓存到本地")
            print("🔍 已展示详细的中间量计算过程")
        print("=" * 80)

    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
