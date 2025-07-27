import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 忽略NumPy警告
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor
from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path

print(f"🔍 检查transformers版本...")
import transformers

print(f"transformers版本: {transformers.__version__}")


class VideoTextModels:
    """基于CLIP的视频-文字处理SOTA模型集合"""

    def __init__(self):
        self.models = {}
        self.processors = {}

    def load_clip_video1_videoclip(self):
        """模型1: VideoCLIP - 基于CLIP的视频理解模型"""
        print("🔄 加载 VideoCLIP 模型...")
        try:
            model_name = "openai/clip-vit-base-patch32"

            # 使用CLIPModel和CLIPProcessor（更稳定）
            self.models['clip_video1'] = CLIPModel.from_pretrained(
                model_name,
                cache_dir="./models/clip_video1"
            )

            self.processors['clip_video1'] = CLIPProcessor.from_pretrained(
                model_name,
                cache_dir="./models/clip_video1"
            )

            print("✅ VideoCLIP (基础版) 加载成功")
            return True
        except Exception as e:
            print(f"❌ VideoCLIP 加载失败: {e}")
            return False

    def load_clip_video2_xclip(self):
        """模型2: X-CLIP - 大模型版本"""
        print("🔄 加载 X-CLIP 模型...")
        try:
            model_name = "openai/clip-vit-large-patch14"

            self.models['clip_video2'] = CLIPModel.from_pretrained(
                model_name,
                cache_dir="./models/clip_video2"
            )

            self.processors['clip_video2'] = CLIPProcessor.from_pretrained(
                model_name,
                cache_dir="./models/clip_video2"
            )
            print("✅ X-CLIP (大模型版) 加载成功")
            return True
        except Exception as e:
            print(f"❌ X-CLIP 加载失败: {e}")
            return False

    def load_clip_video3_patch16(self):
        """模型3: CLIP-Patch16 - 更细粒度的patch版本"""
        print("🔄 加载 CLIP-Patch16 模型...")
        try:
            model_name = "openai/clip-vit-base-patch16"

            self.models['clip_video3'] = CLIPModel.from_pretrained(
                model_name,
                cache_dir="./models/clip_video3"
            )

            self.processors['clip_video3'] = CLIPProcessor.from_pretrained(
                model_name,
                cache_dir="./models/clip_video3"
            )
            print("✅ CLIP-Patch16 加载成功")
            return True
        except Exception as e:
            print(f"❌ CLIP-Patch16 加载失败: {e}")
            return False

    def load_all_models(self):
        """加载所有模型"""
        print("🚀 开始加载所有视频-文字处理模型...")
        print("=" * 80)

        models_info = [
            ("VideoCLIP (Base-Patch32)", self.load_clip_video1_videoclip),
            ("X-CLIP (Large-Patch14)", self.load_clip_video2_xclip),
            ("CLIP (Base-Patch16)", self.load_clip_video3_patch16),
        ]

        success_count = 0
        for name, load_func in models_info:
            print(f"\n📦 正在加载 {name}...")
            if load_func():
                success_count += 1
            print("-" * 40)

        print(f"\n✅ 成功加载 {success_count}/{len(models_info)} 个模型")
        return success_count > 0


class VideoProcessor:
    """视频处理工具类"""

    @staticmethod
    def create_sample_images(output_dir="./test_materials", num_images=8):
        """创建示例图像序列模拟视频帧"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        images = []

        for i in range(num_images):
            # 创建640x480的图像
            img = Image.new('RGB', (640, 480), color=(50, 50, 50))
            draw = ImageDraw.Draw(img)

            # 添加移动的圆形
            center_x = int(320 + 200 * np.sin(2 * np.pi * i / num_images))
            center_y = int(240 + 100 * np.cos(2 * np.pi * i / num_images))

            # 绘制圆形
            radius = 50
            draw.ellipse([
                center_x - radius, center_y - radius,
                center_x + radius, center_y + radius
            ], fill=(255, 255, 0))

            # 添加文字
            draw.text((50, 50), f"Frame {i + 1}/{num_images}",
                      fill=(255, 255, 255))

            # 调整大小到224x224（CLIP标准输入尺寸）
            img_resized = img.resize((224, 224))
            images.append(img_resized)

            # 保存图像文件
            img_path = output_path / f"frame_{i:03d}.jpg"
            img_resized.save(img_path)

        print(f"✅ 创建了 {num_images} 个示例图像帧")
        return images


class VideoTextDemo:
    """视频-文字处理演示类"""

    def __init__(self):
        self.video_models = VideoTextModels()
        self.video_processor = VideoProcessor()

    def setup_test_materials(self):
        """准备测试素材"""
        print("🎬 准备测试素材...")

        # 创建测试目录
        test_dir = Path("./test_materials")
        test_dir.mkdir(exist_ok=True)

        # 1. 创建示例图像序列（模拟视频帧）
        frames = self.video_processor.create_sample_images(str(test_dir))

        # 2. 准备测试文本
        test_texts = [
            "a person walking in the park",
            "a moving colorful circle",
            "animated graphics with text",
            "a ball bouncing around",
            "computer generated animation"
        ]

        # 保存测试文本
        text_file = test_dir / "test_texts.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            for text in test_texts:
                f.write(text + '\n')

        print(f"✅ 测试素材准备完成:")
        print(f"   - 图像帧: {len(frames)} 个")
        print(f"   - 文本文件: {text_file}")

        return frames, test_texts

    def demonstrate_video_text_matching(self, model_name="clip_video1"):
        """演示视频-文本匹配"""
        print(f"\n🎯 使用 {model_name} 进行视频-文本匹配演示")
        print("=" * 60)

        # 准备测试素材
        frames, test_texts = self.setup_test_materials()

        # 检查模型是否加载
        if model_name not in self.video_models.models:
            print(f"❌ 模型 {model_name} 未加载")
            return

        model = self.video_models.models[model_name]
        processor = self.video_models.processors[model_name]

        print("🎞️ 处理视频帧...")
        print(f"   使用 {len(frames)} 帧进行分析")

        # 处理输入
        print("🔄 处理输入数据...")
        try:
            # 使用第一帧作为代表图像
            representative_frame = frames[0]

            inputs = processor(
                images=representative_frame,
                text=test_texts,
                return_tensors="pt",
                padding=True
            )

            print("✅ 输入数据处理完成")
            print(f"   图像tensor shape: {inputs['pixel_values'].shape}")
            print(f"   文本tokens shape: {inputs['input_ids'].shape}")

            # 模型推理
            print("🧠 进行模型推理...")
            with torch.no_grad():
                outputs = model(**inputs)

                # 获取相似度
                logits_per_image = outputs.logits_per_image
                probs = F.softmax(logits_per_image, dim=-1)

                print("\n📊 视频-文本匹配结果:")
                for i, text in enumerate(test_texts):
                    score = probs[0][i].item()
                    print(f"   '{text}': {score:.4f}")

        except Exception as e:
            print(f"❌ 推理过程出错: {e}")
            import traceback
            traceback.print_exc()

    def run_comprehensive_demo(self):
        """运行完整的演示"""
        print("🎬 视频-文字处理模型综合演示")
        print("=" * 80)

        # 加载模型
        if not self.video_models.load_all_models():
            print("❌ 没有成功加载任何模型，退出演示")
            return

        # 对每个成功加载的模型进行演示
        for model_name in self.video_models.models.keys():
            self.demonstrate_video_text_matching(model_name)
            print("\n" + "=" * 60 + "\n")


def show_model_papers():
    """显示推荐的模型和论文"""
    print("\n📚 推荐的视频-文字处理SOTA模型:")
    print("=" * 80)

    papers = [
        {
            "name": "CLIP",
            "paper": "Learning Transferable Visual Models From Natural Language Supervision",
            "url": "https://arxiv.org/abs/2103.00020",
            "description": "OpenAI的原始CLIP模型，奠定了视觉-语言理解基础"
        },
        {
            "name": "VideoCLIP",
            "paper": "VideoCLIP: Contrastive Pre-training for Zero-shot Video-Text Understanding",
            "url": "https://arxiv.org/abs/2109.14084",
            "description": "将CLIP扩展到视频理解领域"
        },
        {
            "name": "X-CLIP",
            "paper": "Expanding Language-Image Pretrained Models for General Video Recognition",
            "url": "https://arxiv.org/abs/2208.02816",
            "description": "微软提出的跨模态视频理解模型"
        },
        {
            "name": "VideoMAE",
            "paper": "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training",
            "url": "https://arxiv.org/abs/2203.12602",
            "description": "基于掩码自编码器的视频理解"
        }
    ]

    for i, paper in enumerate(papers, 1):
        print(f"\n{i}. **{paper['name']}**")
        print(f"   📄 论文: {paper['paper']}")
        print(f"   🔗 链接: {paper['url']}")
        print(f"   💡 描述: {paper['description']}")


def main():
    """主函数"""
    print("🎬 基于CLIP的视频-文字处理SOTA模型演示")
    print("🔧 使用升级后的transformers库")
    print("=" * 80)

    # 显示推荐模型和论文
    show_model_papers()

    try:
        # 创建演示实例
        demo = VideoTextDemo()

        # 运行综合演示
        demo.run_comprehensive_demo()

        print("\n" + "=" * 80)
        print("✅ 演示完成！")
        print("\n💡 使用指南:")
        print("1. 视频素材: 支持图像序列模拟视频")
        print("2. 文本素材: 准备描述性文本列表")
        print("3. 处理流程: 图像→特征编码→文本匹配")
        print("4. 应用场景: 视频检索、内容理解、自动标注")

    except Exception as e:
        print(f"❌ 演示运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()