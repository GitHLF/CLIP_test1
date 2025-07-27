import time
import urllib.request
from pathlib import Path


def create_test_directories():
    """创建测试目录结构"""
    test_dirs = [
        "./videos/test/test1",
        "./videos/test/test2",
        "./videos/test/test3"
    ]

    for dir_path in test_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ 创建目录: {dir_path}")

def download_file(url, filepath, description="文件"):
    """下载文件的通用函数"""
    try:
        print(f"📥 下载 {description}: {url}")

        # 添加请求头，模拟浏览器
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        req = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(req) as response:
            with open(filepath, 'wb') as f:
                f.write(response.read())

        print(f"✅ 下载完成: {filepath}")
        return True

    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def download_sample_videos():
    """从开源数据库下载示例视频"""
    print("🎬 开始下载开源视频样本...")

    # 使用一些公开可用的测试视频
    video_sources = [
        {
            "name": "test1",
            "description": "人物行走视频",
            "video_url": "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4",
            "backup_urls": [
                "https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4",
                "https://file-examples.com/storage/fe68c1f7d7e5bb45bb0b2e8/2017/10/file_example_MP4_480_1_5MG.mp4"
            ],
            "texts": [
                "a person walking",
                "people moving",
                "human activity",
                "outdoor scene",
                "pedestrian movement"
            ]
        },
        {
            "name": "test2",
            "description": "交通场景视频",
            "video_url": "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_2mb.mp4",
            "backup_urls": [
                "https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4"
            ],
            "texts": [
                "cars driving",
                "traffic on road",
                "vehicles moving",
                "urban scene",
                "transportation"
            ]
        },
        {
            "name": "test3",
            "description": "自然场景视频",
            "video_url": "https://sample-videos.com/zip/10/mp4/SampleVideo_640x360_1mb.mp4",
            "backup_urls": [
                "https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4"
            ],
            "texts": [
                "nature scene",
                "trees and plants",
                "outdoor landscape",
                "natural environment",
                "green vegetation"
            ]
        }
    ]

    for video_info in video_sources:
        download_video_and_create_text(video_info)

def download_video_and_create_text(video_info):
    """下载视频并创建对应的文本文件"""
    video_name = video_info["name"]
    video_dir = Path(f"./videos/test/{video_name}")
    video_path = video_dir / f"{video_name}.mp4"
    text_path = video_dir / f"{video_name}.txt"

    print(f"\n🎥 处理视频: {video_name} - {video_info['description']}")

    # 跳过已存在的文件
    if video_path.exists():
        print(f"⏭️ 跳过已存在的视频文件: {video_name}")
    else:
        # 尝试下载视频
        video_downloaded = False

        # 首先尝试主URL
        if download_file(video_info["video_url"], video_path, f"{video_name}视频"):
            video_downloaded = True
        else:
            # 尝试备用URL
            for i, backup_url in enumerate(video_info.get("backup_urls", []), 1):
                print(f"🔄 尝试备用URL {i}...")
                if download_file(backup_url, video_path, f"{video_name}视频(备用{i})"):
                    video_downloaded = True
                    break

        if not video_downloaded:
            print(f"❌ 无法下载视频: {video_name}")
            return False

    # 创建对应的文本文件
    if not text_path.exists():
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(f"# {video_info['description']}\n")
            f.write(f"# 视频文件: {video_name}.mp4\n")
            f.write(f"# 下载时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            for text in video_info["texts"]:
                f.write(f"{text}\n")
        print(f"✅ 创建文本文件: {text_path}")
    else:
        print(f"⏭️ 跳过已存在的文本文件: {video_name}")

    print(f"✅ 完成: {video_name}")
    return True

def verify_downloads():
    """验证下载的文件"""
    print("\n🔍 验证下载的文件...")

    test_dirs = ["test1", "test2", "test3"]

    for test_name in test_dirs:
        video_dir = Path(f"./videos/test/{test_name}")
        video_path = video_dir / f"{test_name}.mp4"
        text_path = video_dir / f"{test_name}.txt"

        print(f"\n📂 检查 {test_name}:")

        if video_path.exists():
            file_size = video_path.stat().st_size / (1024 * 1024)  # MB
            print(f"   ✅ 视频文件: {video_path} ({file_size:.2f} MB)")
        else:
            print(f"   ❌ 视频文件缺失: {video_path}")

        if text_path.exists():
            with open(text_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            print(f"   ✅ 文本文件: {text_path} ({len(lines)} 个查询)")
            for i, line in enumerate(lines[:3], 1):  # 显示前3个查询
                print(f"      {i}. {line}")
            if len(lines) > 3:
                print(f"      ... 还有 {len(lines) - 3} 个查询")
        else:
            print(f"   ❌ 文本文件缺失: {text_path}")

def main():
    """主函数"""
    print("📥 CLIP视频测试数据下载器")
    print("🌐 从开源示例视频网站下载测试文件")
    print("=" * 80)

    # 1. 创建目录
    create_test_directories()

    # 2. 下载示例视频
    download_sample_videos()

    # 3. 验证下载结果
    verify_downloads()

    print("\n" + "=" * 80)
    print("✅ 下载完成！")
    print("💡 现在可以运行 python clip_videos_mps_2.py 进行分析")
    print("📁 文件结构:")
    print("   ./videos/test/test1/test1.mp4 + test1.txt")
    print("   ./videos/test/test2/test2.mp4 + test2.txt")
    print("   ./videos/test/test3/test3.mp4 + test3.txt")
    print("=" * 80)

if __name__ == "__main__":
    main()

