import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 忽略NumPy警告
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import torch
import torch.nn.functional as F
from PIL import Image
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

class VideoAnalyzer:
    """视频分析器 - 读取真实视频文件并进行CLIP分析"""

    def __init__(self):
        self.model_loaded = False
        self._load_clip_model()

    def _load_clip_model(self):
        """加载CLIP模型"""
        if not clip_available:
            print("❌ CLIP组件不可用，使用演示模式")
            self._create_demo_model()
            return

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
                    print("⚠️  本地tokenizer加载失败，使用在线版本...")
                    self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

                # 创建简化处理器
                self.processor = SimpleProcessor(self.tokenizer)

                print(f"✅ CLIP模型加载成功，运行在 {device}")
                self.model_loaded = True

            else:
                print("❌ 未找到本地模型文件")
                raise Exception("本地模型文件不存在")

        except Exception as e:
            print(f"❌ CLIP模型加载失败: {e}")
            print("🔧 使用演示模式...")
            self.model_loaded = False
            self._create_demo_model()

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
        print("✅ 演示模型创建完成")

    def extract_frames_from_video(self, video_path, max_frames=30, sample_rate=1):
        """
        从视频文件中提取帧

        Args:
            video_path: 视频文件路径
            max_frames: 最大提取帧数
            sample_rate: 采样率（每sample_rate帧取一帧）

        Returns:
            frames: PIL图像列表
            frame_info: 帧信息（时间戳等）
        """
        print(f"🎞️ 从视频提取帧: {video_path}")

        if not Path(video_path).exists():
            print(f"❌ 视频文件不存在: {video_path}")
            return [], []

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"❌ 无法打开视频文件: {video_path}")
            return [], []

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        print(f"📊 视频信息:")
        print(f"   - 总帧数: {total_frames}")
        print(f"   - 帧率: {fps:.2f} FPS")
        print(f"   - 时长: {duration:.2f} 秒")

        frames = []
        frame_info = []
        frame_count = 0
        extracted_count = 0

        while cap.isOpened() and extracted_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # 按采样率提取帧
            if frame_count % sample_rate == 0:
                # 转换BGR到RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 转换为PIL图像
                pil_frame = Image.fromarray(frame_rgb)
                frames.append(pil_frame)

                # 记录帧信息
                timestamp = frame_count / fps if fps > 0 else frame_count
                frame_info.append({
                    'frame_number': frame_count,
                    'timestamp': timestamp,
                    'extracted_index': extracted_count
                })
                extracted_count += 1

            frame_count += 1

        cap.release()
        print(f"✅ 提取了 {len(frames)} 帧")
        return frames, frame_info

    def load_text_queries(self, text_path):
        """
        从文本文件加载查询文本

        Args:
            text_path: 文本文件路径

        Returns:
            texts: 文本列表
        """
        print(f"📝 加载文本查询: {text_path}")

        if not Path(text_path).exists():
            print(f"❌ 文本文件不存在: {text_path}")
            return []

        texts = []
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):  # 忽略空行和注释
                        texts.append(line)

            print(f"✅ 加载了 {len(texts)} 个文本查询:")
            for i, text in enumerate(texts, 1):
                print(f"   {i}. {text}")

            return texts

        except Exception as e:
            print(f"❌ 读取文本文件失败: {e}")
            return []

    def analyze_frame(self, frame, texts):
        """分析单个帧"""
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

    def analyze_video(self, video_path, text_path, output_dir):
        """
        分析视频文件

        Args:
            video_path: 视频文件路径
            text_path: 文本文件路径
            output_dir: 输出目录
        """
        print(f"\n🎬 开始分析视频")
        print("=" * 60)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 1. 提取视频帧
        frames, frame_info = self.extract_frames_from_video(video_path, max_frames=50, sample_rate=5)
        if not frames:
            print("❌ 无法提取视频帧")
            return False

        # 2. 加载文本查询
        texts = self.load_text_queries(text_path)
        if not texts:
            print("❌ 无法加载文本查询")
            return False

        # 3. 分析每一帧
        print(f"\n🔍 开始CLIP分析...")
        results = []

        for i, (frame, info) in enumerate(zip(frames, frame_info)):
            print(f"   分析帧 {i+1}/{len(frames)} (时间: {info['timestamp']:.2f}s)")
            probs = self.analyze_frame(frame, texts)
            results.append(probs[0])  # 取第一个图像的结果

        results = torch.stack(results)

        # 4. 生成分析报告
        self._generate_analysis_report(
            video_path, text_path, texts, results, frame_info, output_path
        )

        # 5. 创建可视化视频
        self._create_analysis_video(
            frames, texts, results, frame_info, output_path / "analysis_video.mp4"
        )

        # 6. 保存关键帧
        self._save_key_frames(frames, texts, results, frame_info, output_path)

        print(f"\n✅ 分析完成！结果保存在: {output_path}")
        return True

    def _generate_analysis_report(self, video_path, text_path, texts, results, frame_info, output_path):
        """生成详细的分析报告"""
        report_path = output_path / "description.txt"

        print(f"📊 生成分析报告: {report_path}")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("CLIP视频分析报告\n")
            f.write("=" * 50 + "\n\n")

            # 基本信息
            f.write("📁 文件信息:\n")
            f.write(f"   视频文件: {Path(video_path).name}\n")
            f.write(f"   文本文件: {Path(text_path).name}\n")
            f.write(f"   分析帧数: {len(results)}\n")
            f.write(f"   使用设备: {device}\n")
            f.write(f"   模型状态: {'真实CLIP模型' if self.model_loaded else '演示模式'}\n\n")

            # 文本查询
            f.write("📝 文本查询:\n")
            for i, text in enumerate(texts, 1):
                f.write(f"   {i}. {text}\n")
            f.write("\n")

            # 整体分析结果
            f.write("📊 整体分析结果:\n")
            f.write("-" * 30 + "\n")
            avg_scores = results.mean(dim=0)
            sorted_indices = torch.argsort(avg_scores, descending=True)

            for rank, idx in enumerate(sorted_indices, 1):
                text = texts[idx]
                score = avg_scores[idx].item()
                f.write(f"   {rank}. {text:30s}: {score:.4f}\n")
            f.write("\n")

            # 逐帧详细结果
            f.write("🎞️ 逐帧分析结果:\n")
            f.write("-" * 50 + "\n")

            for i, (frame_probs, info) in enumerate(zip(results, frame_info)):
                f.write(f"\n帧 {i+1:3d} (时间: {info['timestamp']:6.2f}s, 原始帧号: {info['frame_number']}):\n")

                # 按分数排序
                sorted_indices = torch.argsort(frame_probs, descending=True)
                for rank, idx in enumerate(sorted_indices, 1):
                    text = texts[idx]
                    score = frame_probs[idx].item()
                    f.write(f"   {rank}. {text:25s}: {score:.4f}\n")

            # 时间段分析
            f.write(f"\n⏰ 时间段分析:\n")
            f.write("-" * 30 + "\n")

            # 将视频分成几个时间段
            num_segments = min(5, len(results))
            segment_size = len(results) // num_segments

            for seg in range(num_segments):
                start_idx = seg * segment_size
                end_idx = min((seg + 1) * segment_size, len(results))

                if start_idx >= len(results):
                    break

                segment_results = results[start_idx:end_idx]
                segment_avg = segment_results.mean(dim=0)
                best_idx = torch.argmax(segment_avg)
                best_text = texts[best_idx]
                best_score = segment_avg[best_idx].item()

                start_time = frame_info[start_idx]['timestamp']
                end_time = frame_info[end_idx-1]['timestamp'] if end_idx > 0 else start_time

                f.write(f"   时间段 {seg+1} ({start_time:.1f}s - {end_time:.1f}s): {best_text} ({best_score:.4f})\n")

    def _create_analysis_video(self, frames, texts, results, frame_info, output_path, fps=2):
        """创建带有分析结果的视频"""
        print(f"🎥 创建分析视频: {output_path}")

        analysis_frames = []

        for i, (frame, frame_probs, info) in enumerate(zip(frames, results, frame_info)):
            # 放大帧到640x480用于显示
            display_frame = frame.resize((800, 600))
            from PIL import ImageDraw
            draw = ImageDraw.Draw(display_frame)

            # 添加帧信息
            draw.text((10, 10), f"帧 {i+1}/{len(frames)}", fill=(255, 255, 255))
            draw.text((10, 35), f"时间: {info['timestamp']:.2f}s", fill=(255, 255, 255))

            # 添加分析结果
            y_offset = 70
            sorted_indices = torch.argsort(frame_probs, descending=True)

            for rank, idx in enumerate(sorted_indices[:5]):  # 只显示前5个结果
                text = texts[idx]
                score = frame_probs[idx].item()
                color = (255, 255, 255) if score > 0.2 else (150, 150, 150)

                # 绘制文本和分数
                result_text = f"{rank+1}. {text}: {score:.3f}"
                draw.text((10, y_offset + rank * 25), result_text, fill=color)

                # 绘制分数条
                bar_width = int(score * 300)
                if bar_width > 0:
                    draw.rectangle([
                        400, y_offset + rank * 25 + 5,
                        400 + bar_width, y_offset + rank * 25 + 15
                    ], fill=(0, 255, 0) if score > 0.2 else (100, 100, 100))

            # 转换为OpenCV格式
            frame_array = np.array(display_frame)
            frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            analysis_frames.append(frame_bgr)

        # 创建视频
        if analysis_frames:
            height, width, layers = analysis_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            for frame in analysis_frames:
                video_writer.write(frame)

            video_writer.release()
            print(f"✅ 分析视频已保存")

    def _save_key_frames(self, frames, texts, results, frame_info, output_path):
        """保存关键帧"""
        key_frames_dir = output_path / "key_frames"
        key_frames_dir.mkdir(exist_ok=True)

        print(f"🖼️ 保存关键帧到: {key_frames_dir}")

        # 为每个文本查询找到最佳匹配的帧
        for text_idx, text in enumerate(texts):
            text_scores = results[:, text_idx]
            best_frame_idx = torch.argmax(text_scores)
            best_score = text_scores[best_frame_idx].item()

            if best_score > 0.1:  # 只保存分数较高的帧
                best_frame = frames[best_frame_idx]
                best_info = frame_info[best_frame_idx]

                # 清理文件名
                safe_text = "".join(c for c in text if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_text = safe_text.replace(' ', '_')[:50]  # 限制长度

                frame_filename = f"{text_idx+1:02d}_{safe_text}_{best_score:.3f}.jpg"
                frame_path = key_frames_dir / frame_filename

                best_frame.save(frame_path)
                print(f"   保存: {frame_filename} (时间: {best_info['timestamp']:.2f}s)")

def main():
    """主函数"""
    print("🎬 CLIP视频分析器 v2.0")
    print("=" * 80)

    # 设置输入路径
    video_dir = Path("./videos/test/test1")
    video_path = video_dir / "test1.mp4"
    text_path = video_dir / "test1.txt"

    print(f"📂 输入文件:")
    print(f"   视频: {video_path}")
    print(f"   文本: {text_path}")

    # 检查文件是否存在
    if not video_path.exists():
        print(f"❌ 视频文件不存在: {video_path}")
        print("💡 请将视频文件放在 ./videos/test/test1/test1.mp4")
        return

    if not text_path.exists():
        print(f"❌ 文本文件不存在: {text_path}")
        print("💡 请创建文本文件 ./videos/test/test1/test1.txt")
        print("   文件格式：每行一个查询文本，例如：")
        print("   a person walking")
        print("   a car driving")
        print("   a dog running")
        return

    try:
        # 创建分析器
        analyzer = VideoAnalyzer()

        # 分析视频
        success = analyzer.analyze_video(
            video_path=str(video_path),
            text_path=str(text_path),
            output_dir=str(video_dir)
        )

        if success:
            print("\n" + "=" * 80)
            print("✅ 视频分析完成！")
            print(f"📁 结果文件:")
            print(f"   📊 详细报告: {video_dir}/description.txt")
            print(f"   🎥 分析视频: {video_dir}/analysis_video.mp4")
            print(f"   🖼️ 关键帧: {video_dir}/key_frames/")
            print("=" * 80)
        else:
            print("❌ 视频分析失败")

    except Exception as e:
        print(f"❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

