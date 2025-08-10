#!/usr/bin/env python3
"""
详细英文文本注意力分析入口脚本
将每个subplot分别保存为单独图片，并提供详细解释
"""

from visualizations.english_text_attention_detailed import run_detailed_english_attention_demo

if __name__ == "__main__":
    print("🚀 启动详细英文文本注意力分析...")
    print("📊 将生成7个单独的分析图片，每个都有详细解释")

    result_dir = run_detailed_english_attention_demo()

    if result_dir:
        print(f"\n🎉 详细分析完成！")
        print(f"📁 所有图片保存在: {result_dir}")
        print("\n📄 生成的图片说明:")
        print("   1. 01_token_sequence.png - Token序列展示")
        print("   2. 02_attention_heatmap.png - 注意力热力图")
        print("   3. 03_dogs_attention_distribution.png - Dogs注意力分布")
        print("   4. 04_self_attention_strength.png - 自注意力强度")
        print("   5. 05_attention_network.png - 注意力网络图")
        print("   6. 06_multi_head_attention.png - 多头注意力分析")
        print("   7. 07_detailed_analysis_report.png - 详细分析报告")

        # 尝试打开文件夹（macOS）
        import subprocess
        import sys
        try:
            if sys.platform == "darwin":  # macOS
                subprocess.run(["open", str(result_dir)], check=False)
                print("\n📂 文件夹已自动打开")
        except:
            pass
    else:
        print("❌ 分析失败")

