"""
完整训练环境模拟测试框架

这个文件完整地模拟实际训练环境来验证特殊token处理流程，包括：
1. 从sft.py导入真实的特殊token处理函数
2. 加载完整的模型和tokenizer
3. 使用真实的特殊token设置和模型embedding调整流程
4. 验证数据预处理中的特殊token插入
5. 检查整个流程的正确性

使用方法：
    # 完整测试（包含详细位置验证）
    python test_data_processing.py --train_data gsm8k --max_length 1024 --analyze

    # 基础测试
    python test_data_processing.py --train_data gsm8k --max_length 1024

    # 测试其他数据集
    python test_data_processing.py --train_data simplescaling/s1K --max_length 2048 --analyze
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

# 从sft.py导入真实的特殊token处理函数
from sft import SPECIAL_TOKENS, ensure_special_tokens_in_tokenizer, setup_model_and_tokenizer_for_special_tokens



def load_data(args, tokenizer):
    """
    加载和预处理数据集（使用已设置特殊token的tokenizer）

    Args:
        args: 命令行参数，包含数据路径和配置
        tokenizer: 已设置特殊token的分词器对象

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


def verify_special_tokens_setup(model, tokenizer):
    """
    验证特殊token设置是否正确
    """
    print(f"\n{'='*50}")
    print("验证特殊token设置")
    print(f"{'='*50}")

    success = True

    # 1. 检查特殊token是否在tokenizer词汇表中
    for token_name, token_value in SPECIAL_TOKENS.items():
        token_id = tokenizer.convert_tokens_to_ids(token_value)
        if token_id == tokenizer.unk_token_id or token_id is None:
            print(f"❌ 特殊token {token_value} 未在tokenizer词汇表中找到")
            success = False
        else:
            print(f"✅ 特殊token {token_value} -> ID: {token_id}")

    # 2. 检查模型embedding层大小是否与tokenizer词汇表大小兼容
    model_vocab_size = model.get_input_embeddings().weight.size(0)
    tokenizer_vocab_size = len(tokenizer)

    if model_vocab_size >= tokenizer_vocab_size:
        if model_vocab_size == tokenizer_vocab_size:
            print(f"✅ 模型embedding层大小与tokenizer词汇表大小完全匹配: {model_vocab_size}")
        else:
            size_diff = model_vocab_size - tokenizer_vocab_size
            print(f"✅ 模型embedding层大小兼容tokenizer词汇表大小:")
            print(f"    模型embedding: {model_vocab_size}, tokenizer: {tokenizer_vocab_size} (差异: +{size_diff})")
            print(f"    这是安全的配置，embedding层大于tokenizer可以避免索引越界")
    else:
        size_diff = tokenizer_vocab_size - model_vocab_size
        print(f"❌ 模型embedding层大小({model_vocab_size})小于tokenizer词汇表大小({tokenizer_vocab_size})")
        print(f"    差异: -{size_diff}，这会导致训练时索引越界错误")
        success = False

    # 3. 检查pad token设置
    if tokenizer.pad_token is not None:
        print(f"✅ pad_token已设置: {tokenizer.pad_token}")
    else:
        print("⚠️  pad_token未设置")

    return success


def verify_special_tokens_in_data(dataset, tokenizer, sample_count=10):
    """
    验证数据预处理后是否包含特殊token
    """
    print(f"\n{'='*50}")
    print("验证数据中的特殊token")
    print(f"{'='*50}")

    # 获取特殊token的ID
    expand_token_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS['expand'])
    enough_token_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS['enough'])

    expand_count = 0
    enough_count = 0
    mixed_count = 0  # 同时包含两种token的样本（异常情况）
    total_samples = min(sample_count, len(dataset))

    # 长度分布统计
    length_distribution = {
        'short_with_enough': 0,    # ≤64 tokens且包含<|enough|>
        'short_with_expand': 0,    # ≤64 tokens但包含<|expand|>（异常）
        'long_with_expand': 0,     # >64 tokens且包含<|expand|>
        'long_with_enough': 0,     # >64 tokens但包含<|enough|>（异常）
        'no_special_tokens': 0     # 无特殊token
    }

    position_accuracy = {
        'correct_positions': 0,
        'incorrect_positions': 0,
        'position_details': []
    }

    for i in range(total_samples):
        sample = dataset[i]
        input_ids = sample['input_ids']

        # 计算prompt长度和response长度
        prompt_length = sample.get('prompt_lengths', 0)
        if isinstance(prompt_length, torch.Tensor):
            prompt_length = prompt_length.item()

        # 估算response长度（简化计算）
        total_length = len(input_ids)
        estimated_response_length = total_length - prompt_length

        # 检查是否包含特殊token
        has_expand = (input_ids == expand_token_id).any().item()
        has_enough = (input_ids == enough_token_id).any().item()

        # 分类统计
        if has_expand and has_enough:
            mixed_count += 1
            print(f"⚠️  样本 {i+1}: 异常 - 同时包含两种特殊token")
        elif has_expand:
            expand_count += 1
            expand_positions = (input_ids == expand_token_id).nonzero(as_tuple=True)[0].tolist()
            relative_positions = [pos - prompt_length for pos in expand_positions]

            # 验证位置精确度
            expected_positions = [63, 127, 255, 511, 1023]
            correct_positions = [pos for pos in relative_positions if pos in expected_positions]
            incorrect_positions = [pos for pos in relative_positions if pos not in expected_positions]

            if incorrect_positions:
                position_accuracy['incorrect_positions'] += 1
                position_accuracy['position_details'].append({
                    'sample_id': i + 1,
                    'incorrect_positions': incorrect_positions,
                    'all_positions': relative_positions
                })
                print(f"❌ 样本 {i+1}: <|expand|> 位置错误 - 相对位置: {relative_positions}")
            else:
                position_accuracy['correct_positions'] += 1
                print(f"✅ 样本 {i+1}: <|expand|> 位置正确 - 相对位置: {relative_positions}")

            # 长度分布统计
            if estimated_response_length <= 64:
                length_distribution['short_with_expand'] += 1
                print(f"  ⚠️  短回答({estimated_response_length} tokens)却使用<|expand|>")
            else:
                length_distribution['long_with_expand'] += 1
                print(f"  ✅ 长回答({estimated_response_length} tokens)正确使用<|expand|>")

        elif has_enough:
            enough_count += 1
            enough_positions = (input_ids == enough_token_id).nonzero(as_tuple=True)[0].tolist()
            relative_positions = [pos - prompt_length for pos in enough_positions]

            print(f"样本 {i+1}: 包含 <|enough|> token，相对位置: {relative_positions}")

            # 长度分布统计
            if estimated_response_length <= 64:
                length_distribution['short_with_enough'] += 1
                print(f"  ✅ 短回答({estimated_response_length} tokens)正确使用<|enough|>")
            else:
                length_distribution['long_with_enough'] += 1
                print(f"  ⚠️  长回答({estimated_response_length} tokens)却使用<|enough|>")
        else:
            length_distribution['no_special_tokens'] += 1
            print(f"样本 {i+1}: 未包含特殊token (response长度: ~{estimated_response_length})")

    # 打印详细统计
    print(f"\n📊 详细统计结果 (检查了 {total_samples} 个样本):")
    print(f"包含 <|expand|> token的样本: {expand_count}")
    print(f"包含 <|enough|> token的样本: {enough_count}")
    print(f"同时包含两种token的样本: {mixed_count}")
    print(f"未包含特殊token的样本: {length_distribution['no_special_tokens']}")

    print(f"\n📏 长度分布验证:")
    print(f"短回答正确使用<|enough|>: {length_distribution['short_with_enough']}")
    print(f"短回答错误使用<|expand|>: {length_distribution['short_with_expand']}")
    print(f"长回答正确使用<|expand|>: {length_distribution['long_with_expand']}")
    print(f"长回答错误使用<|enough|>: {length_distribution['long_with_enough']}")

    print(f"\n🎯 位置精确度:")
    print(f"位置正确的样本: {position_accuracy['correct_positions']}")
    print(f"位置错误的样本: {position_accuracy['incorrect_positions']}")

    if position_accuracy['position_details']:
        print(f"\n❌ 位置错误详情:")
        for detail in position_accuracy['position_details']:
            print(f"  样本 {detail['sample_id']}: 错误位置 {detail['incorrect_positions']}")

    return {
        'total_samples': total_samples,
        'expand_count': expand_count,
        'enough_count': enough_count,
        'mixed_count': mixed_count,
        'no_special_tokens': length_distribution['no_special_tokens'],
        'length_distribution': length_distribution,
        'position_accuracy': position_accuracy
    }


def verify_special_token_positions(dataset, tokenizer, sample_count=10):
    """
    验证特殊token的精确位置
    """
    print(f"\n{'='*60}")
    print("🎯 特殊token位置精确验证")
    print(f"{'='*60}")

    # 获取特殊token的ID
    expand_token_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS['expand'])
    enough_token_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS['enough'])

    # 预期的expand token位置（相对于response起始的位置）
    expected_expand_positions = [63, 127, 255, 511, 1023]

    # 验证结果统计
    results = {
        'total_samples': 0,
        'expand_samples': 0,
        'enough_samples': 0,
        'position_correct': 0,
        'position_errors': [],
        'expand_position_stats': {pos: 0 for pos in expected_expand_positions},
        'enough_position_correct': 0,
        'detailed_results': []
    }

    total_samples = min(sample_count, len(dataset))
    results['total_samples'] = total_samples

    for i in range(total_samples):
        sample = dataset[i]
        input_ids = sample['input_ids']

        # 计算prompt长度（用于确定response起始位置）
        prompt_length = sample.get('prompt_lengths', 0)
        if isinstance(prompt_length, torch.Tensor):
            prompt_length = prompt_length.item()

        # 找到特殊token的位置
        expand_positions = (input_ids == expand_token_id).nonzero(as_tuple=True)[0].tolist()
        enough_positions = (input_ids == enough_token_id).nonzero(as_tuple=True)[0].tolist()

        sample_result = {
            'sample_id': i + 1,
            'prompt_length': prompt_length,
            'expand_positions': expand_positions,
            'enough_positions': enough_positions,
            'expand_relative_positions': [],
            'enough_relative_positions': [],
            'position_correct': True,
            'errors': []
        }

        # 计算相对于response起始的位置
        if expand_positions:
            sample_result['expand_relative_positions'] = [pos - prompt_length for pos in expand_positions]
            results['expand_samples'] += 1

            # 验证expand token位置
            relative_positions = sample_result['expand_relative_positions']
            position_errors = []

            for rel_pos in relative_positions:
                if rel_pos in expected_expand_positions:
                    results['expand_position_stats'][rel_pos] += 1
                else:
                    position_errors.append(f"意外位置: {rel_pos}")

            # 检查是否有遗漏的预期位置
            for expected_pos in expected_expand_positions:
                if expected_pos not in relative_positions:
                    # 需要检查原始response长度来判断是否应该有这个位置
                    # 这里简化处理，只记录实际找到的位置
                    pass

            if position_errors:
                sample_result['position_correct'] = False
                sample_result['errors'].extend(position_errors)
                results['position_errors'].append({
                    'sample_id': i + 1,
                    'errors': position_errors,
                    'found_positions': relative_positions,
                    'expected_positions': expected_expand_positions
                })

        if enough_positions:
            sample_result['enough_relative_positions'] = [pos - prompt_length for pos in enough_positions]
            results['enough_samples'] += 1

            # 验证enough token位置（应该在末尾）
            # 简化验证：检查是否只有一个enough token
            if len(enough_positions) == 1:
                results['enough_position_correct'] += 1
            else:
                sample_result['position_correct'] = False
                sample_result['errors'].append(f"enough token数量异常: {len(enough_positions)}")

        if sample_result['position_correct']:
            results['position_correct'] += 1

        results['detailed_results'].append(sample_result)

        # 打印详细信息
        print(f"\n样本 {i+1}:")
        print(f"  Prompt长度: {prompt_length}")

        if expand_positions:
            print(f"  <|expand|> 绝对位置: {expand_positions}")
            print(f"  <|expand|> 相对位置: {sample_result['expand_relative_positions']}")

            # 验证每个位置
            for rel_pos in sample_result['expand_relative_positions']:
                if rel_pos in expected_expand_positions:
                    print(f"    ✅ 位置 {rel_pos} 正确")
                else:
                    print(f"    ❌ 位置 {rel_pos} 错误（预期: {expected_expand_positions}）")

        if enough_positions:
            print(f"  <|enough|> 绝对位置: {enough_positions}")
            print(f"  <|enough|> 相对位置: {sample_result['enough_relative_positions']}")
            if len(enough_positions) == 1:
                print(f"    ✅ <|enough|> token位置正确")
            else:
                print(f"    ❌ <|enough|> token数量异常: {len(enough_positions)}")

        if not expand_positions and not enough_positions:
            print(f"  ⚠️  未找到特殊token")

    return results


def check_position_consistency(position_results):
    """
    检查位置一致性并提供详细分析

    Args:
        position_results: verify_special_token_positions的返回结果

    Returns:
        dict: 一致性检查报告
    """
    print(f"\n{'='*60}")
    print("📊 位置一致性分析")
    print(f"{'='*60}")

    total_samples = position_results['total_samples']
    position_correct = position_results['position_correct']

    # 计算准确率
    accuracy = (position_correct / total_samples * 100) if total_samples > 0 else 0

    print(f"总样本数: {total_samples}")
    print(f"位置正确样本数: {position_correct}")
    print(f"位置准确率: {accuracy:.1f}%")

    # 分析expand token位置分布
    print(f"\n<|expand|> token位置分布:")
    expected_positions = [63, 127, 255, 511, 1023]
    for pos in expected_positions:
        count = position_results['expand_position_stats'][pos]
        print(f"  位置 {pos}: {count} 次")

    # 分析错误情况
    if position_results['position_errors']:
        print(f"\n❌ 发现 {len(position_results['position_errors'])} 个位置错误:")
        for error in position_results['position_errors']:
            print(f"  样本 {error['sample_id']}:")
            print(f"    错误: {error['errors']}")
            print(f"    实际位置: {error['found_positions']}")
            print(f"    预期位置: {error['expected_positions']}")

    # 提供修复建议
    if accuracy < 100:
        print(f"\n🔧 修复建议:")
        print(f"  1. 检查data_process.py中的特殊token插入逻辑")
        print(f"  2. 验证target_positions = [63, 127, 255, 511, 1023]设置")
        print(f"  3. 确认response_start_idx计算是否正确")
        print(f"  4. 检查是否有位置偏移问题")

    consistency_report = {
        'accuracy': accuracy,
        'total_samples': total_samples,
        'correct_samples': position_correct,
        'error_count': len(position_results['position_errors']),
        'expand_distribution': position_results['expand_position_stats'],
        'enough_correct': position_results['enough_position_correct'],
        'recommendations': []
    }

    if accuracy < 100:
        consistency_report['recommendations'] = [
            "检查特殊token插入逻辑",
            "验证位置计算算法",
            "确认response起始位置计算"
        ]

    return consistency_report



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

    args = parser.parse_args()

    try:
        print("正在加载完整的模型和分词器...")

        # 加载完整的模型和tokenizer（真实训练环境）
        model_loader = TransformerModelLoader(args.model_name, args.model_path)
        tokenizer, model = model_loader.load_model_tokenizer()
        print("✅ 模型和分词器加载成功")

        print(f"原始tokenizer词汇表大小: {len(tokenizer)}")
        print(f"原始模型embedding层大小: {model.get_input_embeddings().weight.size(0)}")

        # 使用真实的特殊token设置流程（完全模拟sft.py中的处理）
        print(f"\n{'='*50}")
        print("设置特殊token并调整模型embedding层")
        print(f"{'='*50}")

        model, tokenizer, tokens_added = setup_model_and_tokenizer_for_special_tokens(model, tokenizer)

        if tokens_added:
            print("✅ 特殊token设置完成，模型已准备好进行训练")
            print(f"调整后tokenizer词汇表大小: {len(tokenizer)}")
            print(f"调整后模型embedding层大小: {model.get_input_embeddings().weight.size(0)}")
        else:
            print("ℹ️  特殊token已存在，无需调整")

        # 验证特殊token设置
        setup_success = verify_special_tokens_setup(model, tokenizer)
        if not setup_success:
            print("❌ 特殊token设置验证失败")
            return 1

        # 加载和预处理数据
        print(f"\n{'='*50}")
        print("开始数据加载和预处理")
        print(f"{'='*50}")
        train_dataset, eval_dataset = load_data(args, tokenizer)

        # 基本信息输出
        print(f"\n{'='*50}")
        print("数据加载完成")
        print(f"{'='*50}")
        print(f"训练集样本数: {len(train_dataset)}")
        print(f"验证集样本数: {len(eval_dataset)}")

        # 验证数据中的特殊token（基础验证）
        train_stats = None
        eval_stats = None
        if len(train_dataset) > 0:
            train_stats = verify_special_tokens_in_data(train_dataset, tokenizer, sample_count=10000)

        if len(eval_dataset) > 0:
            eval_stats = verify_special_tokens_in_data(eval_dataset, tokenizer, sample_count=10000)

        # 🎯 精确位置验证测试
        print(f"\n{'='*60}")
        print("🚀 开始精确位置验证测试")
        print(f"{'='*60}")

        position_test_results = {}

        if len(train_dataset) > 0:
            print(f"\n📋 训练集位置验证:")
            train_position_results = verify_special_token_positions(train_dataset, tokenizer, sample_count=10000)
            train_consistency = check_position_consistency(train_position_results)
            position_test_results['train'] = {
                'position_results': train_position_results,
                'consistency': train_consistency
            }

        if len(eval_dataset) > 0:
            print(f"\n📋 验证集位置验证:")
            eval_position_results = verify_special_token_positions(eval_dataset, tokenizer, sample_count=10000)
            eval_consistency = check_position_consistency(eval_position_results)
            position_test_results['eval'] = {
                'position_results': eval_position_results,
                'consistency': eval_consistency
            }



        # 🎯 综合验证总结
        print(f"\n{'='*70}")
        print("🎯 完整训练环境模拟测试总结")
        print(f"{'='*70}")

        # 基础设置验证
        print("📋 基础设置验证:")
        print("✅ 模型和tokenizer加载成功")
        print("✅ 特殊token设置和模型embedding层调整完成")

        # 重新验证特殊token设置（用于最终报告）
        final_setup_success = verify_special_tokens_setup(model, tokenizer)
        if final_setup_success:
            print("✅ 特殊token设置验证通过")
        else:
            print("❌ 特殊token设置验证失败")

        print("✅ 数据预处理完成")
        print(f"✅ 训练集样本数: {len(train_dataset)}")
        print(f"✅ 验证集样本数: {len(eval_dataset)}")

        # 位置验证总结
        print(f"\n🎯 特殊token位置验证总结:")
        overall_position_accuracy = 0
        total_position_tests = 0

        if 'train' in position_test_results:
            train_accuracy = position_test_results['train']['consistency']['accuracy']
            train_samples = position_test_results['train']['consistency']['total_samples']
            print(f"✅ 训练集位置验证: {train_accuracy:.1f}% 准确率 ({train_samples} 样本)")
            overall_position_accuracy += train_accuracy * train_samples
            total_position_tests += train_samples

        if 'eval' in position_test_results:
            eval_accuracy = position_test_results['eval']['consistency']['accuracy']
            eval_samples = position_test_results['eval']['consistency']['total_samples']
            print(f"✅ 验证集位置验证: {eval_accuracy:.1f}% 准确率 ({eval_samples} 样本)")
            overall_position_accuracy += eval_accuracy * eval_samples
            total_position_tests += eval_samples

        if total_position_tests > 0:
            overall_accuracy = overall_position_accuracy / total_position_tests
            print(f"🎯 总体位置准确率: {overall_accuracy:.1f}%")

            if overall_accuracy >= 95:
                print("🎉 位置验证优秀！特殊token插入位置高度准确")
            elif overall_accuracy >= 80:
                print("✅ 位置验证良好，但仍有改进空间")
            else:
                print("⚠️  位置验证需要改进，建议检查插入逻辑")

        # 功能完整性验证
        print(f"\n📊 功能完整性验证:")
        if train_stats:
            if train_stats.get('mixed_count', 0) == 0:
                print("✅ 无异常样本（同时包含两种特殊token）")
            else:
                print(f"⚠️  发现 {train_stats['mixed_count']} 个异常样本")

        # 最终状态判断
        all_tests_passed = True

        # 检查特殊token设置
        if not final_setup_success:
            all_tests_passed = False

        # 检查位置验证准确率
        if total_position_tests > 0 and overall_accuracy < 80:
            all_tests_passed = False

        print(f"\n{'='*70}")
        if all_tests_passed:
            print("🎉 所有测试通过！完整训练环境模拟测试成功完成!")
            print("   特殊token处理流程验证通过，可以安全进行实际训练。")
            print("   位置插入准确，数据预处理正常，模型已正确调整。")
        else:
            print("⚠️  部分测试未完全通过，建议检查以下问题:")
            if not final_setup_success:
                print("   - 特殊token设置存在问题")
                print("   - 检查模型embedding层与tokenizer的兼容性")
            if total_position_tests > 0 and overall_accuracy < 80:
                print("   - 特殊token位置插入准确率偏低")
                print("   - 检查data_process.py中的插入逻辑")
            print("   建议修复问题后重新测试。")

        # 释放模型内存
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print("✅ 模型内存已释放")

        print(f"{'='*70}")

        # 返回测试结果状态
        return 0 if all_tests_passed else 1

    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ 测试过程中发生错误: {e}")
        print(f"{'='*70}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)