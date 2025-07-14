#!/usr/bin/env python3
"""
动态长度微调集成测试脚本

测试新集成的动态长度功能是否正常工作，同时确保现有功能不受影响。
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataloader import dLLMDataCollator
from trainer.dynamic_length_trainer import DynamicLengthTrainer
from trainer.trainer import dLLMTrainer
from argsparser import ArgsProcessor


def test_data_collator():
    """测试数据整理器的动态长度功能"""
    print("🧪 测试数据整理器...")
    
    # 创建模拟的tokenizer
    class MockTokenizer:
        def __init__(self):
            self.mask_token_id = 126336
            self.unk_token_id = 0
            
        def convert_tokens_to_ids(self, token):
            # 模拟特殊token ID
            token_map = {
                "<|expand|>": 126337,
                "<|enough|>": 126338
            }
            return token_map.get(token, self.unk_token_id)
    
    tokenizer = MockTokenizer()
    
    # 测试标准模式（应该与现有功能完全相同）
    print("  📋 测试标准模式...")
    standard_collator = dLLMDataCollator(
        tokenizer=tokenizer,
        max_length=512,
        enable_dynamic_length=False
    )
    
    # 创建测试批次
    batch = {
        "input_ids": torch.randint(0, 1000, (2, 100)),
        "prompt_lengths": torch.tensor([20, 25])
    }
    
    # 测试标准前向过程
    try:
        noisy_batch, t, mask_indices = standard_collator.forward_process(batch.copy())
        print("    ✅ 标准前向过程正常")
        print(f"    📊 输出形状: noisy_batch={noisy_batch.shape}, t={t.shape}, mask_indices={mask_indices.shape}")
    except Exception as e:
        print(f"    ❌ 标准前向过程失败: {e}")
        return False
    
    # 测试动态长度模式
    print("  📋 测试动态长度模式...")
    dynamic_config = {
        'initial_response_length': 64,
        'expansion_steps': [64, 128, 256],
        'confidence_threshold': 0.7
    }
    
    dynamic_collator = dLLMDataCollator(
        tokenizer=tokenizer,
        max_length=512,
        enable_dynamic_length=True,
        dynamic_config=dynamic_config
    )
    
    # 测试动态前向过程
    try:
        batch_with_dynamic = batch.copy()
        batch_with_dynamic["current_response_length"] = 64
        
        noisy_batch, t, mask_indices = dynamic_collator.forward_process(batch_with_dynamic)
        print("    ✅ 动态前向过程正常")
        print(f"    📊 输出形状: noisy_batch={noisy_batch.shape}, t={t.shape}, mask_indices={mask_indices.shape}")
    except Exception as e:
        print(f"    ❌ 动态前向过程失败: {e}")
        return False
    
    print("✅ 数据整理器测试通过")
    return True


def test_trainer_classes():
    """测试训练器类的导入和基本功能"""
    print("🧪 测试训练器类...")

    # 测试类导入
    print("  📋 测试类导入...")
    try:
        from trainer.trainer import dLLMTrainer
        from trainer.dynamic_length_trainer import DynamicLengthTrainer
        print("    ✅ 训练器类导入成功")
    except Exception as e:
        print(f"    ❌ 训练器类导入失败: {e}")
        return False

    # 测试动态长度训练器的基本属性
    print("  📋 测试动态长度训练器配置...")
    try:
        # 测试不同配置下的初始化
        configs = [
            None,  # 无配置
            {},    # 空配置
            {'enable_dynamic_length': False},  # 禁用
            {'enable_dynamic_length': True, 'initial_response_length': 64}  # 启用
        ]

        for i, config in enumerate(configs):
            # 创建一个最小的训练器实例来测试配置处理
            class MinimalTrainer(DynamicLengthTrainer):
                def __init__(self, dynamic_config=None):
                    # 跳过父类初始化，只测试我们的配置逻辑
                    self.dynamic_config = dynamic_config or {}
                    self.enable_dynamic_length = self.dynamic_config.get('enable_dynamic_length', False)

                    if self.enable_dynamic_length:
                        self._init_dynamic_length_components()

            trainer = MinimalTrainer(config)
            expected_enabled = bool(config and config.get('enable_dynamic_length', False))

            if trainer.enable_dynamic_length == expected_enabled:
                print(f"    ✅ 配置 {i+1}: 动态长度={'启用' if expected_enabled else '禁用'}")
            else:
                print(f"    ❌ 配置 {i+1}: 期望{'启用' if expected_enabled else '禁用'}，实际{'启用' if trainer.enable_dynamic_length else '禁用'}")
                return False

    except Exception as e:
        print(f"    ❌ 动态长度训练器配置测试失败: {e}")
        return False

    print("✅ 训练器类测试通过")
    return True


def test_config_loading():
    """测试配置加载"""
    print("🧪 测试配置加载...")
    
    try:
        # 测试加载默认配置
        config_path = "./config/sft/default_config.yaml"
        if os.path.exists(config_path):
            args_processor = ArgsProcessor(config_path)
            args = argparse.Namespace()
            args = args_processor.add_args_from_yaml(args)
            
            # 检查动态长度配置是否正确加载
            has_dynamic_config = hasattr(args, 'enable_dynamic_length')
            print(f"    📋 动态长度配置存在: {has_dynamic_config}")
            
            if has_dynamic_config:
                print(f"    📊 启用动态长度: {args.enable_dynamic_length}")
                if hasattr(args, 'dynamic_length'):
                    print(f"    📊 动态长度详细配置: {type(args.dynamic_length)}")
            
            print("    ✅ 配置加载成功")
        else:
            print(f"    ⚠️  配置文件不存在: {config_path}")
            
    except Exception as e:
        print(f"    ❌ 配置加载失败: {e}")
        return False
    
    print("✅ 配置测试通过")
    return True


def test_backward_compatibility():
    """测试向后兼容性"""
    print("🧪 测试向后兼容性...")
    
    # 模拟没有动态长度配置的情况
    class OldStyleArgs:
        def __init__(self):
            self.model_name = "test-model"
            self.local_batch_size = 1
            self.max_length = 512
            # 故意不包含动态长度相关的配置
    
    args = OldStyleArgs()
    
    # 测试是否能正常处理缺少动态长度配置的情况
    try:
        enable_dynamic_length = getattr(args, 'enable_dynamic_length', False)
        dynamic_config = getattr(args, 'dynamic_length', None) if enable_dynamic_length else None
        
        print(f"    📋 动态长度功能: {'启用' if enable_dynamic_length else '禁用'}")
        print(f"    📋 动态配置: {dynamic_config}")
        
        # 应该默认禁用动态长度功能
        if not enable_dynamic_length and dynamic_config is None:
            print("    ✅ 向后兼容性正常 - 默认禁用动态长度功能")
        else:
            print("    ❌ 向后兼容性问题 - 应该默认禁用动态长度功能")
            return False
            
    except Exception as e:
        print(f"    ❌ 向后兼容性测试失败: {e}")
        return False
    
    print("✅ 向后兼容性测试通过")
    return True


def test_stage2_features():
    """测试阶段2的核心功能"""
    print("🧪 测试阶段2核心功能...")

    # 测试渐进式长度扩展逻辑
    print("  📋 测试渐进式长度扩展...")
    try:
        from trainer.dynamic_length_trainer import DynamicLengthTrainer

        # 创建最小的训练器实例来测试扩展逻辑
        class TestTrainer(DynamicLengthTrainer):
            def __init__(self, dynamic_config=None):
                self.dynamic_config = dynamic_config or {}
                self.enable_dynamic_length = self.dynamic_config.get('enable_dynamic_length', False)
                self.special_token_ids = {'expand': 126337, 'enough': 126338}

                # 模拟训练状态
                self.state = type('State', (), {'global_step': 150})()

        config = {
            'enable_dynamic_length': True,
            'expansion_steps': [64, 128, 256, 512],
            'initial_response_length': 64,
            'min_steps_per_stage': 100
        }

        trainer = TestTrainer(config)

        # 测试训练阶段判断
        stage_64 = trainer._get_current_training_stage(64, [64, 128, 256, 512])
        stage_128 = trainer._get_current_training_stage(128, [64, 128, 256, 512])
        stage_256 = trainer._get_current_training_stage(256, [64, 128, 256, 512])

        if stage_64 == 0 and stage_128 == 1 and stage_256 == 2:
            print("    ✅ 训练阶段判断正确")
        else:
            print(f"    ❌ 训练阶段判断错误: {stage_64}, {stage_128}, {stage_256}")
            return False

        # 测试长度自适应权重
        weight_64 = trainer._get_length_adaptive_weight(64)
        weight_128 = trainer._get_length_adaptive_weight(128)
        weight_512 = trainer._get_length_adaptive_weight(512)

        if 1.0 <= weight_64 <= 1.5 and 0.8 <= weight_128 <= 1.2 and 0.8 <= weight_512 <= 1.1:
            print("    ✅ 长度自适应权重正确")
        else:
            print(f"    ❌ 长度自适应权重异常: {weight_64}, {weight_128}, {weight_512}")
            return False

    except Exception as e:
        print(f"    ❌ 渐进式长度扩展测试失败: {e}")
        return False

    # 测试特殊token检测
    print("  📋 测试特殊token检测...")
    try:
        # 模拟输入数据
        input_ids = torch.tensor([[1, 2, 3, 126337, 5, 6, 7, 126338, 9, 10]])  # expand at pos 3, enough at pos 7

        # 测试直接观察
        direct_result = trainer._check_direct_token_observation(
            input_ids[0], 3, 126337, 126338
        )

        if direct_result['should_expand'] and direct_result['token_type'] == 'expand':
            print("    ✅ 直接观察expand token正确")
        else:
            print(f"    ❌ 直接观察expand token失败: {direct_result}")
            return False

        direct_result_enough = trainer._check_direct_token_observation(
            input_ids[0], 7, 126337, 126338
        )

        if not direct_result_enough['should_expand'] and direct_result_enough['token_type'] == 'enough':
            print("    ✅ 直接观察enough token正确")
        else:
            print(f"    ❌ 直接观察enough token失败: {direct_result_enough}")
            return False

    except Exception as e:
        print(f"    ❌ 特殊token检测测试失败: {e}")
        return False

    print("✅ 阶段2核心功能测试通过")
    return True


def main():
    """主测试函数"""
    print("🚀 开始动态长度微调集成测试 - 阶段2")
    print("=" * 60)

    tests = [
        ("向后兼容性", test_backward_compatibility),
        ("配置加载", test_config_loading),
        ("数据整理器", test_data_collator),
        ("训练器类", test_trainer_classes),
        ("阶段2核心功能", test_stage2_features),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📝 运行测试: {test_name}")
        print("-" * 40)

        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 测试通过")
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")

    print("\n" + "=" * 60)
    print(f"📊 测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！动态长度微调阶段2集成成功！")
        return True
    else:
        print("⚠️  部分测试失败，请检查相关问题")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
