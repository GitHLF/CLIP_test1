import json
from pathlib import Path

import torch


def load_local_clip_model(model_name="openai/clip-vit-base-patch32"):
    """加载本地CLIP模型

    Args:
        model_name: 模型名称或路径

    Returns:
        model: CLIP模型
        tokenizer: 分词器
        success: 是否成功加载
    """
    print("步骤1: 搜索本地CLIP模型...")

    # 检查可能的模型路径
    model_paths = [
        "./models/models--openai--clip-vit-base-patch32/snapshots",
        "./models/models--openai--clip-vit-large-patch14/snapshots",
        "./models/clip-model",
        "./models",
        "../models/models--openai--clip-vit-base-patch32/snapshots",
        "../models/models--openai--clip-vit-large-patch14/snapshots",
        "../models",
        "../../models/models--openai--clip-vit-base-patch32/snapshots",
        "../../models/models--openai--clip-vit-large-patch14/snapshots",
        "../../models"
    ]

    local_model_path = None
    for path in model_paths:
        path_obj = Path(path)
        if path_obj.exists():
            for subdir in path_obj.rglob("*"):
                if subdir.is_dir() and (subdir / "config.json").exists():
                    local_model_path = subdir
                    print(f"检查模型路径: {local_model_path}")

                    required_files = ["config.json", "pytorch_model.bin"]
                    missing_files = []
                    for file in required_files:
                        if not (local_model_path / file).exists():
                            missing_files.append(file)

                    if missing_files:
                        print(f"缺少文件: {missing_files}")
                        continue

                    break
            if local_model_path and not missing_files:
                break

    if local_model_path:
        print(f"找到本地模型: {local_model_path}")

        try:
            from transformers.models.clip.modeling_clip import CLIPModel
            from transformers.models.clip.configuration_clip import CLIPConfig
            from transformers import AutoTokenizer, CLIPTokenizer

            with open(local_model_path / "config.json", 'r') as f:
                config_dict = json.load(f)
            config = CLIPConfig.from_dict(config_dict)

            print("步骤2: 加载CLIP模型...")
            model = CLIPModel.from_pretrained(str(local_model_path), config=config, local_files_only=True)

            # 尝试加载tokenizer
            tokenizer = None
            tokenizer_methods = [
                lambda: AutoTokenizer.from_pretrained(str(local_model_path), local_files_only=True),
                lambda: CLIPTokenizer.from_pretrained(str(local_model_path), local_files_only=True),
                lambda: create_basic_tokenizer(),
            ]

            for i, method in enumerate(tokenizer_methods, 1):
                try:
                    print(f"尝试tokenizer加载方法 {i}...")
                    tokenizer = method()
                    print(f"tokenizer加载成功 (方法 {i})")
                    break
                except Exception as e:
                    print(f"方法 {i} 失败: {str(e)[:100]}...")
                    continue

            if tokenizer is None:
                print("所有tokenizer加载方法都失败")
                return None, None, False

            print("✓ 本地CLIP模型加载成功")
            return model, tokenizer, True

        except Exception as e:
            print(f"✗ 本地模型加载失败: {e}")
            return None, None, False
    else:
        print("未找到本地模型文件，尝试在线加载...")
        try:
            # 尝试在线加载，但设置较长的超时时间
            from transformers import CLIPModel, CLIPTokenizer
            import os

            # 设置较长的超时时间
            os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'  # 5分钟超时

            print(f"在线加载CLIP模型: {model_name}")
            model = CLIPModel.from_pretrained(model_name)
            tokenizer = CLIPTokenizer.from_pretrained(model_name)
            print("✓ 在线CLIP模型加载成功")
            return model, tokenizer, True
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            return None, None, False

def create_basic_tokenizer():
    """创建基础的tokenizer"""
    print("创建基础tokenizer...")

    class BasicTokenizer:
        def __init__(self):
            self.vocab_size = 49408
            self.max_length = 77
            self.bos_token_id = 49406
            self.eos_token_id = 49407
            self.pad_token_id = 0

        def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=77):
            if isinstance(texts, str):
                texts = [texts]

            input_ids = []
            attention_masks = []

            for text in texts:
                tokens = [self.bos_token_id]
                for char in text[:max_length - 2]:
                    token_id = min(ord(char), self.vocab_size - 1)
                    tokens.append(token_id)
                tokens.append(self.eos_token_id)

                while len(tokens) < max_length:
                    tokens.append(self.pad_token_id)

                attention_mask = [1 if token != self.pad_token_id else 0 for token in tokens]
                input_ids.append(tokens[:max_length])
                attention_masks.append(attention_mask[:max_length])

            return {
                'input_ids': torch.tensor(input_ids),
                'attention_mask': torch.tensor(attention_masks)
            }

    return BasicTokenizer()

