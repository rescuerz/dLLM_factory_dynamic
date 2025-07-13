"""
基础数据集加载和测试框架

这个文件提供了数据集加载的基础功能，包括：
1. 从HuggingFace加载数据集（如 simplescaling/s1K）
2. 从本地JSON文件加载数据集
3. 数据预处理和格式转换
4. 创建训练和验证数据集

使用方法：
    python test_data_processing.py --train_data simplescaling/s1K --max_length 4096
"""

import torch
import argparse
import os
import sys
from datasets import load_dataset

# 添加父目录到路径以导入SFT模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import dLLMSFTDataset, preprocess_dataset
from utils import TransformerModelLoader


def load_data(args, tokenizer):
    """
    加载和预处理数据集

    Args:
        args: 命令行参数，包含数据路径和配置
        tokenizer: 分词器对象

    Returns:
        tuple: (train_dataset, eval_dataset) 训练和验证数据集
    """
    print(f"正在加载数据集: {args.train_data}")

    # 如果是本地JSON文件，则直接加载
    if args.train_data.endswith('.json'):
        from datasets import Dataset
        import json
        print("检测到本地JSON文件，正在加载...")
        with open(args.train_data, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data = Dataset.from_list(data)
        print(f"从本地JSON文件加载了 {len(data)} 个样本")

    # 如果是HuggingFace数据集（如 simplescaling/s1K），则使用load_dataset从huggingface下载并加载数据集
    else:
        print("从HuggingFace Hub加载数据集...")

        # 处理特殊数据集的配置
        dataset_config = None
        if args.train_data == "gsm8k":
            dataset_config = "main"  # gsm8k 默认使用 main 配置
            print(f"检测到 gsm8k 数据集，使用配置: {dataset_config}")

        # 加载数据集
        if dataset_config:
            data = load_dataset(args.train_data, dataset_config, split="train")
        else:
            data = load_dataset(args.train_data, split="train")

        data_len = len(data)  # type: ignore
        print(f"成功从 {args.train_data} 加载了 {data_len} 个训练样本")

    # 显示数据集基本信息
    try:
        features = data.features  # type: ignore
        if features is not None:
            print(f"数据集特征: {list(features.keys())}")
        else:
            print("数据集特征信息为空")
    except AttributeError:
        print("无法获取数据集特征信息")

    try:
        first_sample = data[0]  # type: ignore
        if isinstance(first_sample, dict):
            print(f"第一个样本的键: {list(first_sample.keys())}")
        else:
            print(f"第一个样本类型: {type(first_sample)}")
    except (IndexError, KeyError, TypeError, AttributeError):
        print("无法获取第一个样本信息")

    # 对数据进行预处理
    print("开始数据预处理...")
    train_data, test_data = preprocess_dataset(data, tokenizer, args.max_length)
    print(f"训练数据长度: {len(train_data)}")
    print(f"验证数据长度: {len(test_data)}")

    # 创建PyTorch数据集对象
    train_dataset = dLLMSFTDataset(train_data, tokenizer, args.max_length)
    test_dataset = dLLMSFTDataset(test_data, tokenizer, args.max_length, eval=True)

    return train_dataset, test_dataset


def analyze_dataset(dataset, dataset_name="数据集", tokenizer=None):
    """
    分析数据集的基本统计信息

    Args:
        dataset: 数据集对象
        dataset_name: 数据集名称
    """
    print(f"\n{'='*50}")
    print(f"{dataset_name} 分析")
    print(f"{'='*50}")

    if len(dataset) == 0:
        print("数据集为空")
        return

    # 获取第一个样本
    sample = dataset[0]
    print(f"样本数量: {len(dataset)}")
    print(f"样本字段: {list(sample.keys())}")

    # 分析input_ids
    if 'input_ids' in sample:
        input_ids = sample['input_ids']
        print(f"input_ids 形状: {input_ids.shape}")
        print(f"input_ids 数据类型: {input_ids.dtype}")

        # 统计非零token数量（实际内容长度）
        non_zero_count = (input_ids != tokenizer.pad_token_id).sum().item()
        print(f"非零token数量: {non_zero_count}")
        print(f"填充token数量: {len(input_ids) - non_zero_count}")

    # 分析prompt_lengths
    if 'prompt_lengths' in sample:
        prompt_lengths = sample['prompt_lengths']
        print(f"prompt_lengths: {prompt_lengths.item()}")



def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="数据集加载和预处理测试")

    # 数据相关参数
    parser.add_argument("--train_data", type=str, default="simplescaling/s1K",
                       help="训练数据路径（HuggingFace数据集名称或本地JSON文件路径）")

    # 模型相关参数
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct",
                       help="模型名称")
    parser.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct",
                       help="模型路径")

    # 处理参数
    parser.add_argument("--max_length", type=int, default=4096,
                       help="最大序列长度")
    parser.add_argument("--analyze", action="store_true",
                       help="是否进行详细的数据集分析")

    args = parser.parse_args()

    try:
        print("正在加载模型和分词器...")
        tokenizer, _ = TransformerModelLoader(args.model_name, args.model_path).load_model_tokenizer()
        print("模型和分词器加载成功")

        # 加载和预处理数据
        train_dataset, eval_dataset = load_data(args, tokenizer)

        # 基本信息输出
        print(f"\n{'='*50}")
        print("数据加载完成")
        print(f"{'='*50}")
        print(f"训练集样本数: {len(train_dataset)}")
        print(f"验证集样本数: {len(eval_dataset)}")

        # 显示样本示例
        if len(train_dataset) > 0:
            print(f"\n训练集第一个样本:")
            sample = train_dataset[0]
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape} {value.dtype}")
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value}")

        if len(eval_dataset) > 0:
            print(f"\n验证集第一个样本:")
            sample = eval_dataset[0]
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape} {value.dtype}")
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {value}")

        # 详细分析（可选）
        if args.analyze:
            analyze_dataset(train_dataset, "训练集", tokenizer)
            analyze_dataset(eval_dataset, "验证集", tokenizer)

        print(f"\n{'='*50}")
        print("✅ 数据加载和预处理测试完成!")
        print(f"{'='*50}")

    except Exception as e:
        print(f"\n{'='*50}")
        print(f"❌ 错误: {e}")
        print(f"{'='*50}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)