
import torch
import argparse
from transformers import TrainingArguments
import os
from data import dLLMSFTDataset,dLLMDataCollator,dLLMDataCollator_dynamic_length,preprocess_dataset
from trainer import dLLMTrainer
from argsparser import ArgsProcessor
from utils import TransformerModelLoader,LoraBuilder
from datasets import load_dataset
import logging

# 配置日志系统
def setup_logging(debug=False):
    """设置日志配置"""
    level = logging.DEBUG if debug else logging.INFO

    # 配置根日志器
    logging.basicConfig(
        level=level,
        format='%(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    # 设置特定模块的日志级别
    logging.getLogger('sft.trainer.dynamic_length_trainer').setLevel(level)
    logging.getLogger('transformers').setLevel(logging.WARNING)  # 减少transformers的日志噪音

    if debug:
        print("🐛 DEBUG模式已启用 - 将显示详细的日志信息")

    return logging.getLogger(__name__)

logger = logging.getLogger(__name__)

# Special Token定义
SPECIAL_TOKENS = {
    "expand": "<|expand|>",  # 扩展token
    "enough": "<|enough|>"   # 结束token
}

def ensure_special_tokens_in_tokenizer(tokenizer):
    """
    确保特殊token在tokenizer词汇表中

    Returns:
        bool: 是否添加了新的特殊token
    """
    special_tokens = list(SPECIAL_TOKENS.values())
    existing_tokens = set(tokenizer.get_vocab().keys())
    new_tokens = [token for token in special_tokens if token not in existing_tokens]

    if new_tokens:
        special_tokens_dict = {'additional_special_tokens': new_tokens}
        num_added = tokenizer.add_special_tokens(special_tokens_dict)
        logger.info(f"训练初始化：添加了 {num_added} 个特殊token: {new_tokens}")
        return True
    return False

def setup_model_and_tokenizer_for_special_tokens(model, tokenizer):
    """
    为训练脚本提供的工具函数：设置模型和tokenizer以支持特殊token

    Args:
        model: 预训练模型
        tokenizer: 预训练tokenizer

    Returns:
        tuple: (model, tokenizer, tokens_added) - 是否添加了新token
    """
    # 记录原始状态
    original_tokenizer_size = len(tokenizer)
    original_model_embedding_size = model.get_input_embeddings().weight.size(0)
    logger.info(f"原始状态 Tokenizer词汇表大小: {original_tokenizer_size} 模型embedding层大小: {original_model_embedding_size}")

    # 检查原始状态是否正常
    if original_model_embedding_size != original_tokenizer_size:
        size_diff = original_model_embedding_size - original_tokenizer_size
        logger.warning(f"⚠️  警告: 模型embedding层与tokenizer大小不匹配 (差异: {size_diff})")
        if size_diff < 0:
            logger.error(f"❌ 严重错误: 模型embedding层小于tokenizer词汇表，这会导致训练错误")
            raise ValueError(f"模型embedding层({original_model_embedding_size}) < tokenizer词汇表({original_tokenizer_size})")

    # 添加特殊token
    special_tokens = list(SPECIAL_TOKENS.values())
    existing_tokens = set(tokenizer.get_vocab().keys())
    new_tokens = [token for token in special_tokens if token not in existing_tokens]
    if new_tokens:
        tokens_added = True
        special_tokens_dict = {'additional_special_tokens': new_tokens}
        num_added = tokenizer.add_special_tokens(special_tokens_dict)
        logger.info(f"训练初始化：添加了 {num_added} 个特殊token: {new_tokens}")

        new_tokenizer_size = len(tokenizer)
        actual_new_embedding_size = max(original_model_embedding_size, new_tokenizer_size)

        # 安全的embedding层调整
        if new_tokenizer_size > original_model_embedding_size:
            # 只有当tokenizer变大时才调整模型
            model.resize_token_embeddings(new_tokenizer_size)
            actual_new_embedding_size = model.get_input_embeddings().weight.size(0)
            logger.info(f"模型embedding层已扩展: {original_model_embedding_size} -> {actual_new_embedding_size}")
        elif new_tokenizer_size == original_model_embedding_size:
            logger.info(f"模型embedding层大小已匹配，无需调整")

        logger.info(f"插入特殊token后，Tokenizer词汇表大小: {new_tokenizer_size} 模型embedding层大小: {actual_new_embedding_size}")

    else:
        tokens_added = False
        logger.info(f"特殊token已存在，无需调整模型embedding层")

    return model, tokenizer, tokens_added

def load_data(args, tokenizer):
    # 如果是本地json文件，则直接加载
    if args.train_data.endswith('.json'):
        from datasets import Dataset
        import json
        with open(args.train_data, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data = Dataset.from_list(data)
    # 如果是HuggingFace数据集，则使用load_dataset从huggingface下载并加载数据集
    else:
        # 处理特殊数据集的配置
        dataset_config = None
        if args.train_data == "gsm8k":
            dataset_config = "main"  # gsm8k 默认使用 main 配置

        # 加载数据集
        if dataset_config:
            data = load_dataset(args.train_data, dataset_config, split="train")
        else:
            data = load_dataset(args.train_data, split="train")

    # 对数据进行预处理
    train_data, eval_data = preprocess_dataset(data, tokenizer, args.max_length)
    logger.info(f"Train data length: {len(train_data)}, Eval data length: {len(eval_data)}")
    train_dataset = dLLMSFTDataset(train_data, tokenizer, args.max_length)
    eval_dataset = dLLMSFTDataset(eval_data, tokenizer, args.max_length, eval=True)
    return train_dataset, eval_dataset

def train_model(args, model, tokenizer, train_dataset, eval_dataset):
    # 检查是否启用动态长度微调
    enable_dynamic_length = getattr(args, 'enable_dynamic_length', False)
    dynamic_config = getattr(args, 'dynamic_length', None) if enable_dynamic_length else None

    # 将enable_dynamic_length添加到dynamic_config中
    if enable_dynamic_length and dynamic_config:
        dynamic_config['enable_dynamic_length'] = enable_dynamic_length

    logger.info(f"🔧 训练模式: {'动态长度微调（enable_dynamic_length=True）' if enable_dynamic_length else '标准SFT训练'}")
    if enable_dynamic_length:
        logger.info(f"动态长度配置: {dynamic_config}")

    # 创建训练参数
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.job_name),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.local_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        learning_rate=args.learning_rate,
        load_best_model_at_end=args.load_best_model_at_end,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        bf16=args.bf16,
        report_to=args.report_to,
        remove_unused_columns=args.remove_unused_columns,
    )

    # 根据配置选择训练器和数据整理器
    if enable_dynamic_length:
        # 使用动态长度训练器
        from trainer.dynamic_length_trainer import DynamicLengthTrainer

        # 创建动态长度专用的数据整理器（提供干净数据）
        data_collator = dLLMDataCollator_dynamic_length(
            tokenizer=tokenizer,
            mask_token_id=126336,
            max_length=args.max_length
        )

        # 创建动态长度训练器
        trainer = DynamicLengthTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dynamic_config=dynamic_config,
            processing_class=tokenizer  
        )

    else:
        # 使用标准训练器（保持现有逻辑不变）
        data_collator = dLLMDataCollator(
            tokenizer=tokenizer,
            mask_token_id=126336,
            max_length=args.max_length
        )

        trainer = dLLMTrainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

    # 开始训练
    logger.info("开始训练...")
    trainer.train()
    logger.info("训练完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configuration parser")
    parser.add_argument("--debug",dest="debug",action="store_true",help="debug mode")
    parser.add_argument("--enable_lora",default=True,help="enable lora")
    parser.add_argument("--train_config_path",type=str,default="./config/sft/default_config.yaml",help="Path to the Train YAML configuration file")
    parser.add_argument("--lora_config_path",type=str,default="./config/lora/default_config.yaml",help="Path to the Lora YAML configuration file")
    args = parser.parse_args()
    args_processor = ArgsProcessor(args.train_config_path)
    args = args_processor.add_args_from_yaml(args)

    logger = setup_logging(debug=False)
    model_loader = TransformerModelLoader(tokenizer_path=args.model_name,model_path=args.model_name)
    tokenizer, model = model_loader.load_model_tokenizer()

    # 设置特殊token并调整模型embedding层
    model, tokenizer, tokens_added = setup_model_and_tokenizer_for_special_tokens(model, tokenizer)

    if args.enable_lora:
        lora_args =  argparse.ArgumentParser(description="Lora Configuration parser").parse_args()
        lora_args_processor = ArgsProcessor(args.lora_config_path)
        lora_args = lora_args_processor.add_args_from_yaml(lora_args)
        lora_bulider = LoraBuilder(lora_args)
        model = lora_bulider.get_Lora(model)
    train_dataset, eval_dataset = load_data(args, tokenizer)
    logger.info(f"Global Batch Size: {args.local_batch_size * args.grad_accum_steps * torch.cuda.device_count()}")
    train_model(args,model,tokenizer,train_dataset,eval_dataset)
    