#!/usr/bin/env python3
"""
英文文本注意力分析入口脚本
专门分析 "two dogs lying on a cushion in the sun" 这句话的注意力机制
"""

from visualizations.english_text_attention import run_english_text_attention_demo

if __name__ == "__main__":
    print("🚀 启动英文文本注意力分析...")
    result_path = run_english_text_attention_demo()

    if result_path:
        print(f"\n🎉 分析完成！请查看结果图片:")
        print(f"📄 {result_path}")

        # 尝试打开图片（macOS）
        import subprocess
        import sys
        try:
            if sys.platform == "darwin":  # macOS
                subprocess.run(["open", str(result_path)], check=False)
                print("🖼️  图片已自动打开")
        except:
            pass
    else:
        print("❌ 分析失败")

