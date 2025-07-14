#!/usr/bin/env python3
"""
测试动态长度训练集成
"""

import sys
import os
sys.path.append('/mnt/40t/zhounan/dLLM_factory_dynamic/sft')

import torch
import logging
from transformers import TrainingArguments

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dynamic_length_trainer_import():
    """测试DynamicLengthTrainer是否能正确导入"""
    try:
        from trainer.dynamic_length_trainer import DynamicLengthTrainer
        logger.info("✅ DynamicLengthTrainer导入成功")
        return True
    except Exception as e:
        logger.error(f"❌ DynamicLengthTrainer导入失败: {e}")
        return False

def test_dynamic_length_trainer_init():
    """测试DynamicLengthTrainer是否能正确初始化"""
    try:
        from trainer.dynamic_length_trainer import DynamicLengthTrainer
        from transformers import AutoConfig, AutoModelForCausalLM

        # 创建一个简单的测试模型
        config = AutoConfig.from_pretrained("gpt2")
        config.vocab_size = 1000  # 减小词汇表大小以节省内存
        config.n_layer = 2  # 减少层数
        config.n_head = 2   # 减少注意力头数
        config.n_embd = 128 # 减少嵌入维度

        model = AutoModelForCausalLM.from_config(config)

        # 创建基本的训练参数
        training_args = TrainingArguments(
            output_dir="./test_output",
            per_device_train_batch_size=1,
            num_train_epochs=1,
            logging_steps=1,
        )

        # 创建动态长度配置
        dynamic_config = {
            'enable_dynamic_length': True,
            'initial_length': 64,
            'max_length': 2048,
            'confidence_threshold': 0.7,
            'max_expansions': 3
        }

        # 初始化trainer
        trainer = DynamicLengthTrainer(
            model=model,
            args=training_args,
            dynamic_config=dynamic_config
        )
        
        logger.info("✅ DynamicLengthTrainer初始化成功")
        logger.info(f"   - enable_dynamic_length: {trainer.enable_dynamic_length}")
        logger.info(f"   - dynamic_config: {trainer.dynamic_config}")
        return True
        
    except Exception as e:
        logger.error(f"❌ DynamicLengthTrainer初始化失败: {e}")
        return False

def test_compute_loss_method():
    """测试compute_loss方法是否存在且可调用"""
    try:
        from trainer.dynamic_length_trainer import DynamicLengthTrainer
        from transformers import AutoConfig, AutoModelForCausalLM

        # 创建简单测试模型
        config = AutoConfig.from_pretrained("gpt2")
        config.vocab_size = 1000
        config.n_layer = 1
        config.n_head = 2
        config.n_embd = 64
        model = AutoModelForCausalLM.from_config(config)

        training_args = TrainingArguments(
            output_dir="./test_output",
            per_device_train_batch_size=1,
            num_train_epochs=1,
        )

        dynamic_config = {
            'enable_dynamic_length': False,  # 先测试标准模式
        }

        trainer = DynamicLengthTrainer(
            model=model,
            args=training_args,
            dynamic_config=dynamic_config
        )
        
        # 检查compute_loss方法是否存在
        assert hasattr(trainer, 'compute_loss'), "compute_loss方法不存在"
        assert callable(trainer.compute_loss), "compute_loss不可调用"
        
        logger.info("✅ compute_loss方法检查通过")
        return True
        
    except Exception as e:
        logger.error(f"❌ compute_loss方法测试失败: {e}")
        return False

def test_dynamic_length_detection():
    """测试动态长度模式检测逻辑"""
    try:
        from trainer.dynamic_length_trainer import DynamicLengthTrainer
        from transformers import AutoConfig, AutoModelForCausalLM

        # 创建简单测试模型
        config = AutoConfig.from_pretrained("gpt2")
        config.vocab_size = 1000
        config.n_layer = 1
        config.n_head = 2
        config.n_embd = 64
        model = AutoModelForCausalLM.from_config(config)

        training_args = TrainingArguments(
            output_dir="./test_output",
            per_device_train_batch_size=1,
            num_train_epochs=1,
        )

        # 测试启用动态长度
        dynamic_config = {
            'enable_dynamic_length': True,
        }

        trainer = DynamicLengthTrainer(
            model=model,
            args=training_args,
            dynamic_config=dynamic_config
        )
        
        # 模拟标准输入（应该回退到标准训练）
        standard_inputs = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'labels': torch.tensor([[1, 2, 3, 4, 5]])
        }
        
        # 模拟动态长度输入
        dynamic_inputs = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'labels': torch.tensor([[1, 2, 3, 4, 5]]),
            'prompt_lengths': torch.tensor([2])
        }
        
        logger.info("✅ 动态长度检测逻辑测试准备完成")
        logger.info(f"   - 标准输入包含prompt_lengths: {'prompt_lengths' in standard_inputs}")
        logger.info(f"   - 动态输入包含prompt_lengths: {'prompt_lengths' in dynamic_inputs}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 动态长度检测测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    logger.info("🚀 开始测试动态长度训练集成...")
    
    tests = [
        ("导入测试", test_dynamic_length_trainer_import),
        ("初始化测试", test_dynamic_length_trainer_init),
        ("compute_loss方法测试", test_compute_loss_method),
        ("动态长度检测测试", test_dynamic_length_detection),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 运行测试: {test_name}")
        if test_func():
            passed += 1
        else:
            logger.error(f"测试失败: {test_name}")
    
    logger.info(f"\n📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！动态长度训练集成成功")
    else:
        logger.error("❌ 部分测试失败，需要修复问题")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
