import torch
import torch.nn.functional as F
import numpy as np
import logging
from dllm_cache.cache import dLLMCache

eval_logger = logging.getLogger(__name__)


def add_gumbel_noise(logits, temperature):
    """添加Gumbel噪声到logits"""
    if temperature == 0:
        return logits.exp()
    noise = torch.rand_like(logits)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """计算每个样本中每步需要unmask的token数量"""
    # 计算每个样本中待生成位置的数量
    mask_num = mask_index.sum(dim=1, keepdim=True)
    # 计算每个样本中待生成位置的平均数量
    base = mask_num // steps
    # 计算剩余的待生成位置数量
    remainder = mask_num % steps
    # 初始化每个样本的转移token数量
    num_transfer_tokens = base.expand(-1, steps).clone()
    # 如果剩余的待生成位置数量大于0，则将剩余的待生成位置分配给前几个样本
    if remainder.sum() > 0:
        indices = torch.arange(steps, device=mask_index.device)
        mask = indices.unsqueeze(0) < remainder
        num_transfer_tokens[mask] += 1
    return num_transfer_tokens.to(torch.int64)

def _calculate_steps_per_block(total_steps, num_blocks):
    """将总步数分配给每个块"""
    if num_blocks == 0:
        return []
    base_steps = total_steps // num_blocks
    remainder = total_steps % num_blocks
    distribution = [base_steps] * num_blocks
    for i in range(remainder):
        distribution[i] += 1
    return distribution


def _get_special_token_ids(tokenizer):
    """获取特殊token的ID映射"""
    SPECIAL_TOKENS = {
        "expand": "<|expand|>",  # 扩展token
        "enough": "<|enough|>"   # 结束token
    }

    token_ids = {}
    if tokenizer is not None:
        for key, token in SPECIAL_TOKENS.items():
            token_id = tokenizer.convert_tokens_to_ids(token)
            # 检查token是否在词汇表中
            if token_id != tokenizer.unk_token_id:
                token_ids[key] = token_id
            else:
                eval_logger.warning(f"Special token '{token}' not found in tokenizer vocabulary")
                token_ids[key] = None
        eval_logger.info(f"Special token IDs: {token_ids}")
    else:
        eval_logger.warning("Tokenizer not available, special token IDs will be empty")

    return token_ids


def _detect_special_tokens_in_decoded(decoded_tokens, special_token_ids):
    """检测在新解码的token中是否包含特殊token"""
    result = {'expand': False, 'enough': False}

    # 只有当特殊token ID存在时才进行检测
    if special_token_ids.get('expand') is not None and torch.any(decoded_tokens == special_token_ids['expand']):
        result['expand'] = True
    if special_token_ids.get('enough') is not None and torch.any(decoded_tokens == special_token_ids['enough']):
        result['enough'] = True
    return result


def _expand_sequence_length(x, attention_mask, current_gen_length, expansion_steps, mask_id):
    """动态扩展序列长度 和 attention_mask"""
    # 找到下一个扩展长度
    next_length = next((step for step in expansion_steps if step > current_gen_length), None)

    if next_length is None:
        eval_logger.warning(f"No expansion step found for current length {current_gen_length}")
        return x, current_gen_length

    batch_size = x.shape[0]
    expansion_size = next_length - current_gen_length

    # 创建扩展部分，用mask_id填充
    expansion_part = torch.full(
        (batch_size, expansion_size),
        mask_id,
        dtype=torch.long,
        device=x.device
    )

    # 拼接扩展部分
    new_x = torch.cat([x, expansion_part], dim=1)
    new_attention_mask = torch.ones((batch_size, new_x.shape[1]), dtype=attention_mask.dtype, device=attention_mask.device)
    new_attention_mask[:, :attention_mask.shape[1]] = attention_mask

    eval_logger.info(f"Expanded sequence from {current_gen_length} to {next_length} tokens")

    return new_x, new_attention_mask, next_length


def generate_dynamic_length(
    input_ids,
    attention_mask,
    model,
    tokenizer,
    gen_kwargs,
    temperature=0.0,
    mask_id=126336,
):
    """
    基于LLaDA架构的动态长度生成函数

    Args:
        input_ids: 输入token序列 [batch_size, prompt_length]
        attention_mask: 注意力掩码
        model: 生成模型
        tokenizer: 分词器
        gen_kwargs: 生成参数字典

    Returns:
        torch.Tensor: 生成的token序列 [batch_size, actual_gen_length]
    """
    # 从gen_kwargs中提取参数
    total_steps = gen_kwargs.get("steps")
    block_length = gen_kwargs.get("block_length", 64)  # 块长度，适配初始长度
    cfg_scale = gen_kwargs.get("cfg_scale")
    remasking=gen_kwargs.get("remasking","low_confidence")
    # 动态长度相关参数
    initial_gen_length = gen_kwargs.get("initial_length", 64)  # 初始生成长度
    expansion_steps = gen_kwargs.get("expansion_steps", [64, 128, 256, 512, 1024, 2048])
    max_expansions = gen_kwargs.get("max_expansions", 5)

    # 获取特殊token ID
    special_token_ids = _get_special_token_ids(tokenizer)
    enough_token_id = special_token_ids.get('enough')
    expand_token_id = special_token_ids.get('expand')

    with torch.no_grad():
        batch_size, prompt_length = input_ids.shape
        current_gen_length = initial_gen_length
        expansion_count = 0
        steps_consumed = 0
        early_stop_triggered = False
        eval_logger.info(f"Starting dynamic generation: prompt_length={prompt_length}, initial_gen_length={current_gen_length}")

        # 创建初始序列：prompt + 待生成部分（全部用mask_id填充）
        x = torch.full(
            (batch_size, prompt_length + current_gen_length),
            mask_id,
            dtype=torch.long,
            device=model.device,
        )
        # 保留原始的prompt部分
        x[:, :prompt_length] = input_ids
        # 创建一个布尔掩码，用于标记哪些位置是待生成的部分
        prompt_index = x != mask_id

        # 初始化缓存
        feature_cache = dLLMCache()
        feature_cache.reset_cache(prompt_length)

        num_block = 0
        num_blocks = current_gen_length // block_length

        # 处理特殊情况：当生成长度小于块长度时
        if num_blocks == 0:
            eval_logger.info(f"Generation length ({current_gen_length}) is smaller than block length ({block_length}). Processing as single block.")
            # 将整个生成长度作为一个块处理
            num_blocks = 1
            block_length = current_gen_length

        # 初始分配steps_per_block
        steps_per_block_plan = _calculate_steps_per_block(total_steps, num_blocks)

        eval_logger.info(f"Initial generation plan: num_blocks={num_blocks}, total_steps={total_steps}, steps_per_block={steps_per_block_plan}")

        # Main generation loop: iterates over blocks dynamically
        while num_block < num_blocks:
            if early_stop_triggered:
                eval_logger.info("Early stop triggered. Halting generation.")
                break
            
            expansion_triggered_in_block = False
            steps_for_this_block = steps_per_block_plan[num_block]


            # 计算当前块的起始和结束索引
            start_idx = prompt_length + num_block * block_length
            end_idx = prompt_length + (num_block + 1) * block_length

            # 获取当前块的输入和掩码
            block_x = x[:, start_idx:end_idx]
            block_mask_index = block_x == mask_id

            if block_mask_index.sum() == 0:
                eval_logger.info(f"Block {num_block} is fully generated. Skipping.")
                num_block += 1
                continue
            
            # 计算每一步需要unmask的token数量
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_for_this_block)
            eval_logger.info(f"Block {num_block} has {block_mask_index.sum()} masked tokens, {steps_for_this_block} steps to unmask")

            # 对当前块进行steps_for_this_block次迭代
            for i in range(steps_for_this_block):
                if steps_consumed >= total_steps:
                    eval_logger.info("Total steps consumed. Halting generation.")
                    early_stop_triggered = True
                    break
                
                # 标记哪些位置是待生成的部分
                mask_index = x == mask_id
                prompt_index = ~mask_index

                # CFG处理（与原generate函数相同）
                if cfg_scale > 0.0:
                    if hasattr(feature_cache, "cfg_interval_steps"):
                        feature_cache.update_step(layer_id=33)
                        if feature_cache.refresh_cfg(layer_id=33):
                            # 创建无条件输入（prompt也被mask）
                            cfg_x = x.clone()
                            cfg_x[prompt_index] = mask_id

                            # 计算条件生成的logits
                            logits = model(x, attention_mask=attention_mask).logits[
                                :, prompt_length:
                            ]
                            feature_cache.cache_type = "cfg"
                            # 计算无条件生成的logits
                            cfg_logits = model(
                                cfg_x, attention_mask=attention_mask
                            ).logits[:, prompt_length:]
                            # 计算并缓存CFG残差
                            cfg_residual = logits - cfg_logits
                            feature_cache.set_cache(
                                layer_id=33,
                                feature_name="cfg_residual",
                                features=cfg_residual,
                                cache_type="gen",
                            )
                            feature_cache.cache_type = "no_cfg"
                        else:
                            # 如果CFG缓存未刷新，则从缓存中获取CFG残差
                            feature_cache.cache_type = "cfg"
                            cfg_residual = feature_cache.get_cache(
                                layer_id=33,
                                feature_name="cfg_residual",
                                cache_type="gen",
                            )
                            feature_cache.cache_type = "no_cfg"
                            logits = model(x, attention_mask=attention_mask).logits[
                                :, prompt_length:
                            ]
                    else:
                        # 无缓存的CFG
                        cfg_x = x.clone()
                        cfg_x[prompt_index] = mask_id
                        logits = model(x, attention_mask=attention_mask).logits[
                            :, prompt_length:
                        ]
                        cfg_logits = model(cfg_x, attention_mask=attention_mask).logits[
                            :, prompt_length:
                        ]
                        cfg_residual = logits - cfg_logits
                    # CFG公式：logits_final = logits + cfg_scale * cfg_residual
                    logits = (logits - cfg_residual) + (cfg_scale + 1) * cfg_residual
                else:
                    # 无CFG，直接计算logits
                    logits = model(x, attention_mask=attention_mask).logits[
                        :, prompt_length:
                    ]

                # 添加Gumbel噪声
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                # 选择概率最高的token
                x0 = torch.argmax(logits_with_noise, dim=-1)

                # 低置信度重掩蔽（与原generate函数相同）
                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )
                elif remasking == "random":
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                # 确保不会更新当前块之外的位置
                x0_p[:, (num_block + 1) * block_length :] = -np.inf

                # 只在mask位置应用预测，非mask位置保持不变
                x0 = torch.where(
                    mask_index[:, prompt_length:], x0, x[:, prompt_length:]
                )
                # 计算置信度，只在mask位置应用预测，非mask位置置信度为-inf
                confidence = torch.where(mask_index[:, prompt_length:], x0_p, -np.inf)

                # 计算需要转移的token索引
                transfer_index = torch.zeros_like(
                    x0, dtype=torch.bool, device=x0.device
                )
                for j in range(confidence.shape[0]):
                    # 选择置信度最高的num_transfer_tokens[j, i]个token
                    select_index = torch.topk(
                        confidence[j], k=num_transfer_tokens[j, i]
                    ).indices
                    transfer_index[j, select_index] = True

                # 实际更新选中的位置
                x[:, prompt_length:][transfer_index] = x0[transfer_index]

                # 关键：在token解码后检测特殊token
                special_detection = _detect_special_tokens_in_decoded(x0[transfer_index], special_token_ids)

                if special_detection['enough']:
                    early_stop_triggered = True
                    eval_logger.info("Detected <|enough|> token, current length is sufficient")

                if special_detection['expand'] and expansion_count < max_expansions:
                    expansion_triggered_in_block = True
                    eval_logger.info(f"Detected <|expand|> token, expanding sequence (expansion {expansion_count + 1}/{max_expansions})")

                # 增加步数计数
                steps_consumed += 1

            # 块内生成结束后，检查是否需要扩展
            if expansion_triggered_in_block and expansion_count < max_expansions and not early_stop_triggered:
                x[x == expand_token_id] = mask_id
                x, attention_mask, new_gen_length = _expand_sequence_length(x, attention_mask, current_gen_length, expansion_steps, mask_id)
                if new_gen_length > current_gen_length:
                    expansion_count += 1
                    current_gen_length = new_gen_length
                    
                    # --- 核心：重新规划剩余步数 ---
                    old_num_blocks = num_blocks
                    num_blocks = current_gen_length // block_length
                    remaining_steps = total_steps - steps_consumed
                    remaining_blocks_to_plan = num_blocks - (num_block + 1)
                    
                    new_plan_for_remainder = _calculate_steps_per_block(remaining_steps, remaining_blocks_to_plan)
                    # 更新总计划：保留已完成块的计划，拼接新计划
                    steps_per_block_plan = steps_per_block_plan[:num_block+1] + new_plan_for_remainder
                    eval_logger.info(f"Expanded sequence from {current_gen_length} to {new_gen_length} tokens")
                    eval_logger.info(f"New generation plan: finished {num_block} blocks, remaining_steps={remaining_steps}")
            num_block += 1

    eval_logger.info(f"Dynamic generation completed: final_length={current_gen_length}, expansions={expansion_count}, steps_consumed={steps_consumed}")

    # 返回生成的部分（去除prompt）
    return x[:, prompt_length:]