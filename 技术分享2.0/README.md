# CLIP多Prompt优化分析演示 2.0

这是CLIP多Prompt优化分析演示的2.0版本，采用单图可视化方式，每个分析结果都保存为单独的图像文件，便于查看和理解。

## 功能特点

- 支持单Prompt模式和多Prompt模式
- 每个概念组可配置使用的prompt数量
- 支持自定义图像文件和prompt配置
- 详细的可视化分析，每个图单独保存
- 完整展示概念组中的多个prompt变体及其匹配分数
- 包含图像预处理、文本注意力、跨模态注意力等多种可视化

## 可视化内容

本项目生成的可视化内容包括：

### 1. 概念组分析
- 概念组分析概览
- 概念组得分比较
- 每个概念组内的prompt变体得分
- 所有prompt的分数分布
- Top-N最佳匹配
- 概念组性能比较

### 2. 图像预处理
- 原始图像
- 预处理步骤
- 归一化
- Patch Embedding
- 位置编码
- CLS Token + Patches序列
- 注意力图

### 3. 文本注意力分析
- 文本输入
- Token长度分布
- 早期层文本注意力
- 中期层文本注意力
- 最终层文本注意力
- 多头注意力模式
- 图文相似度分布
- 特征空间投影

### 4. 跨模态注意力
- 图像关注区域
- 文本注意力模式
- 跨模态相似度矩阵
- 特征对齐程度
- 注意力流向图
- 匹配置信度分析

### 5. 最终结果
- 最终图像和结果
- 概念组排序
- Top-3概念组详细对比
- 最佳匹配详情

## 使用方法

1. 确保已安装所需依赖：
   ```
   pip install torch torchvision transformers pillow matplotlib numpy
   ```

2. 运行主程序：
   ```
   python main.py
   ```

3. 命令行参数：
   ```
   python main.py --help
   ```

   可用参数：
   - `--image IMAGE`: 要分析的图像文件路径（默认：dogs_sun_patio.jpeg）
   - `--prompt_file PROMPT_FILE`: 自定义prompt配置文件路径（JSON格式）
   - `--prompts_per_group PROMPTS_PER_GROUP`: 每个概念组使用的prompt数量（1=单prompt模式，默认：4）

4. 示例：
   ```
   # 使用默认配置
   python main.py

   # 分析自定义图像
   python main.py --image my_image.jpg

   # 使用自定义prompt配置
   python main.py --prompt_file my_prompts.json

   # 单prompt模式
   python main.py --prompts_per_group 1

   # 完整自定义
   python main.py --image my_image.jpg --prompt_file my_prompts.json --prompts_per_group 3
   ```

5. 输出结果将保存在`outputs_single_prompt_v2`或`outputs_{N}prompts_v2`目录中。

## 自定义Prompt配置

可以创建JSON格式的自定义prompt配置文件，格式如下：

```json
{
  "概念组名称1": [
    "prompt变体1",
    "prompt变体2",
    "prompt变体3"
  ],
  "概念组名称2": [
    "prompt变体1",
    "prompt变体2"
  ]
}
```

例如：

```json
{
  "猫咪": [
    "一只猫坐在窗台上",
    "猫咪在窗户旁边",
    "a cat sitting by the window"
  ],
  "狗狗": [
    "一只狗在草地上奔跑",
    "狗狗在户外玩耍",
    "dog running on grass"
  ]
}
```

## 注意事项

- 所有可视化图像按照编号顺序排列，便于查看
- 如果未指定自定义prompt文件，将使用默认的概念组配置

