import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 忽略NumPy警告
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import numpy as np
from pathlib import Path
import json
import cv2

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

    def __call__(self, text=None, images=None, return_tensors="pt", padding=True):
        # 处理文本
        if text is not None:
            if isinstance(text, str):
                text = [text]

            # 使用tokenizer处理文本
            text_inputs = self.tokenizer(
                text,
                return_tensors=return_tensors,
                padding=padding,
                truncation=True,
                max_length=77
            )
        else:
            text_inputs = {}

        # 处理图像
        if images is not None:
            if isinstance(images, Image.Image):
                images = [images]

            # 简单的图像预处理
            pixel_values = []
            for img in images:
                # 调整大小到224x224
                img = img.resize((224, 224))
                # 转换为tensor
                img_array = torch.tensor(list(img.getdata())).float()
                img_array = img_array.view(224, 224, 3)
                # 转换为CHW格式并归一化
                img_tensor = img_array.permute(2, 0, 1) / 255.0
                # 标准化 (ImageNet标准)
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_tensor = (img_tensor - mean) / std
                pixel_values.append(img_tensor)

            pixel_values = torch.stack(pixel_values)
            text_inputs['pixel_values'] = pixel_values

        return text_inputs

# 尝试导入组件
CLIPModel, CLIPConfig, AutoTokenizer, clip_available = load_clip_directly()

# 设置设备 - 优先使用MPS
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

device = get_device()
print(f"🔧 使用设备: {device}")

class VideoProcessor:
    """视频处理工具类 - 创建视频帧序列并生成MP4"""

    @staticmethod
    def create_sample_video_frames(output_dir="./test_materials", num_frames=16):
        """创建示例视频帧序列"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        frames = []
        frame_descriptions = []
        full_size_frames = []  # 保存完整尺寸的帧用于视频生成

        print(f"🎬 创建 {num_frames} 个视频帧...")

        for i in range(num_frames):
            # 创建640x480的图像
            img = Image.new('RGB', (640, 480), color=(30, 30, 30))
            draw = ImageDraw.Draw(img)

            # 场景1: 移动的圆形 (前8帧)
            if i < 8:
                # 添加移动的黄色圆形
                center_x = int(100 + 400 * (i / 7))
                center_y = int(240 + 100 * np.sin(2 * np.pi * i / 8))

                radius = 40
                draw.ellipse([
                    center_x - radius, center_y - radius,
                    center_x + radius, center_y + radius
                ], fill=(255, 255, 0))

                # 添加文字描述
                frame_text = f"Moving Ball - Frame {i+1}"
                draw.text((50, 50), frame_text, fill=(255, 255, 255))

                # 添加场景描述
                scene_desc = f"Scene 1: Yellow ball moving from left to right"
                draw.text((50, 420), scene_desc, fill=(200, 200, 200))

                frame_descriptions.append("a moving yellow ball")

            # 场景2: 跳跃的方块 (后8帧)
            else:
                frame_in_scene = i - 8
                # 添加跳跃的红色方块
                center_x = 320
                # 抛物线运动
                center_y = int(400 - 200 * (4 * frame_in_scene * (7 - frame_in_scene)) / 49)

                size = 60
                draw.rectangle([
                    center_x - size//2, center_y - size//2,
                    center_x + size//2, center_y + size//2
                ], fill=(255, 100, 100))

                # 添加文字描述
                frame_text = f"Jumping Box - Frame {i+1}"
                draw.text((50, 50), frame_text, fill=(255, 255, 255))

                # 添加场景描述
                scene_desc = f"Scene 2: Red box jumping up and down"
                draw.text((50, 420), scene_desc, fill=(200, 200, 200))

                frame_descriptions.append("a jumping red box")

            # 保存完整尺寸的帧用于视频生成
            full_size_frames.append(img.copy())

            # 调整大小到224x224（CLIP标准输入尺寸）
            img_resized = img.resize((224, 224))
            frames.append(img_resized)

            # 保存原始大小的图像文件（用于调试）
            img_path = output_path / f"frame_{i:03d}.jpg"
            img.save(img_path)

        print(f"✅ 创建了 {num_frames} 个视频帧")
        return frames, frame_descriptions, full_size_frames

    @staticmethod
    def create_video_from_frames(full_size_frames, output_path="./videos/clip_test_1.mp4", fps=2):
        """从帧序列创建MP4视频"""
        # 创建输出目录
        output_dir = Path(output_path).parent
        output_dir.mkdir(exist_ok=True)

        print(f"🎥 创建视频: {output_path}")

        if not full_size_frames:
            print("❌ 没有帧数据")
            return False

        # 转换PIL图像为OpenCV格式
        cv_frames = []
        for pil_frame in full_size_frames:
            # 转换PIL图像为numpy数组
            frame_array = np.array(pil_frame)
            # 转换RGB到BGR（OpenCV格式）
            frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            cv_frames.append(frame_bgr)

        # 获取帧尺寸
        height, width, layers = cv_frames[0].shape

        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # 写入所有帧
        for frame in cv_frames:
            video_writer.write(frame)

        video_writer.release()
        print(f"✅ 视频已保存: {output_path}")
        return True

    @staticmethod
    def create_analysis_video(frames, texts, results, output_path="./videos/clip_analysis.mp4", fps=2):
        """创建带有分析结果的视频"""
        output_dir = Path(output_path).parent
        output_dir.mkdir(exist_ok=True)

        print(f"📊 创建分析视频: {output_path}")

        # 创建带分析结果的帧
        analysis_frames = []

        for i, (frame, frame_results) in enumerate(zip(frames, results)):
            # 放大帧到640x480用于显示
            display_frame = frame.resize((640, 480))
            draw = ImageDraw.Draw(display_frame)

            # 添加帧号
            draw.text((10, 10), f"Frame {i+1}", fill=(255, 255, 255))

            # 添加分析结果
            y_offset = 50
            for j, (text, score) in enumerate(zip(texts, frame_results)):
                score_val = score.item()
                color = (255, 255, 255) if score_val > 0.3 else (150, 150, 150)

                # 绘制文本和分数
                result_text = f"{text}: {score_val:.3f}"
                draw.text((10, y_offset + j * 25), result_text, fill=color)

                # 绘制分数条
                bar_width = int(score_val * 200)
                if bar_width > 0:
                    draw.rectangle([
                        250, y_offset + j * 25 + 5,
                        250 + bar_width, y_offset + j * 25 + 15
                    ], fill=(0, 255, 0) if score_val > 0.3 else (100, 100, 100))

            # 转换为OpenCV格式
            frame_array = np.array(display_frame)
            frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            analysis_frames.append(frame_bgr)

        # 创建视频
        height, width, layers = analysis_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        for frame in analysis_frames:
            video_writer.write(frame)

        video_writer.release()
        print(f"✅ 分析视频已保存: {output_path}")
        return True

class CLIPVideoDemo:
    """CLIP视频处理演示类"""

    def __init__(self):
        print("🔄 初始化CLIP视频演示...")

        if not clip_available:
            print("❌ CLIP组件不可用，使用演示模式")
            self._create_demo_model()
            return

        # 加载CLIP模型
        try:
            print("📦 加载CLIP模型...")

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
                ).to(device)

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

                print(f"✅ CLIP模型加载成功，运行在 {device}")
                self.model_loaded = True

                # 测试模型
                self._test_model()

            else:
                print("❌ 未找到本地模型文件")
                raise Exception("本地模型文件不存在")

        except Exception as e:
            print(f"❌ CLIP模型加载失败: {e}")
            print("🔧 使用演示模式...")
            self.model_loaded = False
            self._create_demo_model()

        self.video_processor = VideoProcessor()

    def _test_model(self):
        """测试模型是否正常工作"""
        try:
            print("🧪 测试模型功能...")

            # 创建测试数据
            test_image = Image.new('RGB', (224, 224), color='red')
            test_texts = ["a red image", "a blue image"]

            # 处理输入
            inputs = self.processor(text=test_texts, images=test_image)

            # 将输入移动到设备
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in inputs.items()}

            # 模型推理
            with torch.no_grad():
                outputs = self.model(**inputs)

            print("✅ 模型测试成功！")

        except Exception as e:
            print(f"⚠️  模型测试失败: {e}")

    def _create_demo_model(self):
        """创建演示模型"""
        class DemoModel:
            def __call__(self, **kwargs):
                batch_size = kwargs.get('pixel_values', torch.randn(1, 3, 224, 224)).shape[0]
                text_batch_size = kwargs.get('input_ids', torch.randint(0, 1000, (1, 77))).shape[0]

                return type('Output', (), {
                    'image_embeds': torch.randn(batch_size, 512),
                    'text_embeds': torch.randn(text_batch_size, 512),
                    'logits_per_image': torch.randn(batch_size, text_batch_size),
                    'logits_per_text': torch.randn(text_batch_size, batch_size)
                })()

        class DemoProcessor:
            def __call__(self, images=None, text=None, return_tensors="pt", padding=True):
                batch_size = len(text) if text else 1
                return {
                    'pixel_values': torch.randn(1, 3, 224, 224),
                    'input_ids': torch.randint(0, 1000, (batch_size, 77)),
                    'attention_mask': torch.ones(batch_size, 77)
                }

        self.model = DemoModel()
        self.processor = DemoProcessor()
        self.model_loaded = False
        print("✅ 演示模型创建完成")

    def analyze_single_frame(self, frame, texts):
        """分析单个视频帧"""
        inputs = self.processor(
            images=frame,
            text=texts,
            return_tensors="pt",
            padding=True
        )

        if self.model_loaded:
            # 将输入移动到设备
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits_per_image, dim=-1)

        return probs.cpu() if self.model_loaded else probs

    def analyze_video_sequence(self, frames, texts):
        """分析整个视频序列"""
        print(f"🎞️ 分析 {len(frames)} 个视频帧...")

        frame_results = []

        for i, frame in enumerate(frames):
            print(f"   处理帧 {i+1}/{len(frames)}")
            probs = self.analyze_single_frame(frame, texts)
            frame_results.append(probs[0])  # 取第一个图像的结果

        return torch.stack(frame_results)

    def create_comprehensive_video_demo(self):
        """创建完整的视频演示"""
        print("\n🎬 创建完整的CLIP视频演示")
        print("=" * 60)

        # 创建视频帧和描述
        frames, frame_descriptions, full_size_frames = self.video_processor.create_sample_video_frames()

        # 创建基础视频 - 使用完整尺寸的帧
        self.video_processor.create_video_from_frames(full_size_frames, "./videos/clip_test_1.mp4", fps=2)

        # 定义候选文本
        texts = [
            "a moving yellow ball",
            "a jumping red box",
            "animated graphics",
            "computer animation",
            "geometric shapes"
        ]

        print(f"\n📝 分析文本: {texts}")

        # 分析视频序列（使用224x224的帧）
        results = self.analyze_video_sequence(frames, texts)

        # 创建分析视频
        self.video_processor.create_analysis_video(frames, texts, results, "./videos/clip_analysis.mp4", fps=2)

        # 显示详细分析结果
        print(f"\n📊 详细分析结果:")
        print("=" * 80)

        # 创建视频描述文件
        description_file = Path("./videos/video_description.txt")
        with open(description_file, 'w', encoding='utf-8') as f:
            f.write("CLIP视频分析结果\n")
            f.write("=" * 50 + "\n\n")
            f.write("视频文件: clip_test_1.mp4\n")
            f.write("分析视频: clip_analysis.mp4\n\n")
            f.write("视频内容描述:\n")
            f.write("- 前8帧: 黄色圆球从左到右移动\n")
            f.write("- 后8帧: 红色方块上下跳跃\n\n")
            f.write("候选文本:\n")
            for i, text in enumerate(texts, 1):
                f.write(f"{i}. {text}\n")
            f.write("\n")

            # 逐帧分析结果
            f.write("逐帧分析结果:\n")
            f.write("-" * 30 + "\n")
            for i, frame_probs in enumerate(results):
                f.write(f"\n帧 {i+1:2d} ({frame_descriptions[i]}):\n")
                for j, text in enumerate(texts):
                    score = frame_probs[j].item()
                    f.write(f"  {text:25s}: {score:.4f}\n")

            # 平均分数
            f.write(f"\n整体平均分数:\n")
            f.write("-" * 20 + "\n")
            avg_scores = results.mean(dim=0)
            for i, text in enumerate(texts):
                score = avg_scores[i].item()
                f.write(f"{text:25s}: {score:.4f}\n")

        # 在控制台显示摘要
        print("📋 视频内容摘要:")
        print("  🎯 场景1 (帧1-8): 黄色圆球从左到右移动")
        print("  🎯 场景2 (帧9-16): 红色方块上下跳跃")
        print(f"\n📁 生成的文件:")
        print(f"  🎥 原始视频: ./videos/clip_test_1.mp4")
        print(f"  📊 分析视频: ./videos/clip_analysis.mp4")
        print(f"  📝 详细报告: ./videos/video_description.txt")

        # 显示最佳匹配
        print(f"\n🏆 最佳文本匹配:")
        avg_scores = results.mean(dim=0)
        best_idx = torch.argmax(avg_scores)
        best_text = texts[best_idx]
        best_score = avg_scores[best_idx].item()
        print(f"  '{best_text}': {best_score:.4f}")

        # 场景切换分析
        scene1_scores = results[:8].mean(dim=0)  # 前8帧
        scene2_scores = results[8:].mean(dim=0)  # 后8帧

        print(f"\n🎬 场景分析:")
        print(f"  场景1 (移动球) 最佳匹配: '{texts[torch.argmax(scene1_scores)]}' ({scene1_scores.max():.4f})")
        print(f"  场景2 (跳跃盒) 最佳匹配: '{texts[torch.argmax(scene2_scores)]}' ({scene2_scores.max():.4f})")

        return True

    def run_comprehensive_demo(self):
        """运行完整演示"""
        print("🎬 CLIP视频处理综合演示")
        print(f"🔧 运行设备: {device}")
        if self.model_loaded:
            print("🎉 使用真实CLIP模型")
        else:
            print("🎭 使用演示模式")
        print("=" * 80)

        # 创建视频演示
        self.create_comprehensive_video_demo()

        print("\n" + "=" * 80)
        print("✅ CLIP视频处理演示完成！")
        print(f"💡 使用了 {device} 进行加速计算")
        if self.model_loaded:
            print("🎉 成功使用真实CLIP模型进行推理")
        else:
            print("🎭 使用演示模式完成功能展示")

        print("\n📺 查看结果:")
        print("  1. 打开 ./videos/clip_test_1.mp4 查看原始视频")
        print("  2. 打开 ./videos/clip_analysis.mp4 查看分析结果")
        print("  3. 查看 ./videos/video_description.txt 了解详细分析")
        print("=" * 80)

def show_video_papers():
    """显示视频相关的SOTA论文"""
    print("\n📚 视频-文字处理SOTA模型论文:")
    print("=" * 80)

    papers = [
        {
            "name": "CLIP",
            "paper": "Learning Transferable Visual Models From Natural Language Supervision",
            "url": "https://arxiv.org/abs/2103.00020",
            "description": "OpenAI的原始CLIP模型，可用于视频帧理解"
        },
        {
            "name": "VideoCLIP",
            "paper": "VideoCLIP: Contrastive Pre-training for Zero-shot Video-Text Understanding",
            "url": "https://arxiv.org/abs/2109.14084",
            "description": "专门针对视频理解的CLIP扩展"
        },
        {
            "name": "X-CLIP",
            "paper": "Expanding Language-Image Pretrained Models for General Video Recognition",
            "url": "https://arxiv.org/abs/2208.02816",
            "description": "微软提出的跨模态视频理解模型"
        },
        {
            "name": "Video-ChatGPT",
            "paper": "Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models",
            "url": "https://arxiv.org/abs/2306.05424",
            "description": "结合ChatGPT的视频理解模型"
        }
    ]

    for i, paper in enumerate(papers, 1):
        print(f"\n{i}. **{paper['name']}**")
        print(f"   📄 {paper['paper']}")
        print(f"   🔗 {paper['url']}")
        print(f"   💡 {paper['description']}")

def main():
    """主函数"""
    print("🎬 基于CLIP的视频-文字处理演示")
    print(f"🚀 支持Apple Silicon MPS加速")
    print("🔧 绕过NumPy兼容性问题")
    print("🎥 生成可视化视频文件")
    print("=" * 80)

    # 显示相关论文
    show_video_papers()

    try:
        # 创建并运行演示
        demo = CLIPVideoDemo()
        demo.run_comprehensive_demo()

    except Exception as e:
        print(f"❌ 演示运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

