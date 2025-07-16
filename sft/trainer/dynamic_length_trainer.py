import torch
import logging

from .trainer import dLLMTrainer

# 设置日志
logger = logging.getLogger(__name__)

# 确保日志能正常工作的函数
def set_logger_works():
    if not logger.handlers:
        # 如果没有handler，添加一个
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')  # 使用简单格式，与sft.py保持一致
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger

set_logger_works()

class DynamicLengthTrainer(dLLMTrainer):

    def __init__(self, dynamic_config=None, special_token_ids=None, **kwargs):
        # 提取special_token_ids，避免传递给父类
        self.special_token_ids = special_token_ids or {}

        super().__init__(**kwargs)

        # 动态长度配置
        self.dynamic_config = dynamic_config or {}
        self.enable_dynamic_length = self.dynamic_config.get('enable_dynamic_length', False)
        
        # 获取特殊token ID映射
        self.special_token_ids = self._get_special_token_ids()
        self.mask_token_id = 126336
        
    
    def _get_special_token_ids(self):
        """
        获取特殊token的ID映射
        与现有的data_process.py保持一致
        """
        # 使用与data_process.py相同的特殊token定义
        SPECIAL_TOKENS = {
            "expand": "<|expand|>",  # 扩展token
            "enough": "<|enough|>"   # 结束token
        }
        
        token_ids = {}
        tokenizer = getattr(self, 'processing_class', None) or getattr(self, 'tokenizer', None)

        if tokenizer is not None:
            for key, token in SPECIAL_TOKENS.items():
                token_id = tokenizer.convert_tokens_to_ids(token)
                token_ids[key] = token_id
            logger.info(f"Special token IDs: {token_ids}")
        else:
            logger.warning("Tokenizer not available, special token IDs will be empty")
            
        return token_ids


    def _forward_diffusion_process_with_dynamic_length(self, input_ids, mask_token_id, eps=1e-3,
                                                 current_length=None, _special_token_ids=None,
                                                 timestep=None, _config=None):
        """
        支持动态长度的前向扩散过程，将输入序列的部分token随机替换为mask token，模拟扩散过程中的噪声添加。

        Args:
            input_ids: 输入token序列（response部分） [batch_size, seq_len]
            mask_token_id: mask token的ID，用于替换被mask的位置
            eps: 最小mask概率
            current_length: 当前有效长度(response 长度)（支持动态长度）
            special_token_ids: 特殊token ID映射（用于后续检查）
            timestep: 指定的时间步，如果为None则随机采样
            config: 动态长度配置对象

        Returns:
            tuple: (noisy_input, p_mask, mask_indices, actual_timestep)
        """
        b, l = input_ids.shape
        device = input_ids.device

        # 动态长度处理，从短序列开始，逐步扩展到长序列
        if current_length is not None:
            effective_length = min(current_length, l)
        else:
            effective_length = l

        # 时间步处理：可以指定或随机采样
        if timestep is not None:
            # 使用指定的时间步
            if isinstance(timestep, (int, float)):
                t = torch.full((b,), timestep, device=device)
            else:
                t = timestep
        else:
            # 随机采样时间步
            t = torch.rand((b,), device=device)

        # 计算mask概率
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, effective_length)

        # 创建完整的mask概率张量
        full_p_mask = torch.zeros((b, l), device=device)
        full_p_mask[:, :effective_length] = p_mask

        # 随机mask
        mask_indices = torch.rand((b, effective_length), device=device) < p_mask

        # Special token与普通token一视同仁，不需要额外保护
        # 这样可以让模型学会在适当的位置生成special token

        # 生成噪声输入
        noisy_input = input_ids.clone()
        noisy_input[:, :effective_length] = torch.where(
            mask_indices,
            mask_token_id,  # 使用mask_token_id作为mask token
            input_ids[:, :effective_length]
        )

        return noisy_input, full_p_mask, mask_indices, t

    def _apply_diffusion_noise(self, input_ids, prompt_length, current_length, special_token_ids, _device, config, mask_token_id):
        """
        对单个样本应用扩散噪声，只对response部分进行扩散，保持prompt部分不变

        Returns:
            tuple: (noisy_input, mask_indices, p_mask, current_input) - 加噪后的输入、mask位置、mask概率、原始截取输入
        """
        # 截取到当前长度
        current_input = input_ids[:, :current_length]

        # 计算response长度
        prompt_len = prompt_length[0].item()
        response_length = max(1, current_length - prompt_len)

        # 随机采样解码比例
        detection_min = config.expansion_check_ratio - 0.05
        detection_max = config.expansion_check_ratio + 0.05
        target_decoded_ratio = torch.rand(1).item() * (detection_max - detection_min) + detection_min

        # 计算mask参数
        target_decoded_tokens = int(response_length * target_decoded_ratio)
        target_mask_tokens = response_length - target_decoded_tokens
        mask_ratio = target_mask_tokens / response_length

        # 计算时间步（已知mask_ratio，解出timestep）
        eps = 1e-3
        timestep = max(0.0, min(1.0, (mask_ratio - eps) / (1 - eps)))

        logger.info(f"Diffusion noise: prompt_len={prompt_len}, response_length={response_length}, "
                    f"decode_ratio={target_decoded_ratio:.2f}, mask_ratio={mask_ratio:.2f}, timestep={timestep:.2f}")

        # 只对response部分应用扩散
        noisy_input = current_input.clone()
        mask_indices = torch.zeros_like(current_input, dtype=torch.bool)  # 初始化mask indices

        if response_length > 0:
            response_part = current_input[0, prompt_len:prompt_len + response_length].unsqueeze(0)
            noisy_response, _, response_mask_indices, _ = self._forward_diffusion_process_with_dynamic_length(
                response_part, mask_token_id=mask_token_id,
                current_length=response_length, _special_token_ids=special_token_ids,
                timestep=timestep, _config=config
            )
            noisy_input[0, prompt_len:prompt_len + response_length] = noisy_response[0]
            # 记录response部分的mask信息
            mask_indices[0, prompt_len:prompt_len + response_length] = response_mask_indices[0]

        # 计算p_mask用于损失计算
        p_mask = torch.zeros_like(noisy_input, dtype=torch.float)
        if response_length > 0:
            response_mask_prob = (1 - eps) * timestep + eps
            p_mask[0, prompt_len:prompt_len + response_length] = response_mask_prob

        return noisy_input, mask_indices, p_mask, current_input

    def _train_single_sample_at_length(self, noisy_input, _mask_indices, p_mask, current_input, model, loss_func,
                                    special_token_ids, device, config, mask_token_id):
        """
        基于已加噪的数据进行损失计算和模型前向传播

        Args:
            noisy_input: 已加噪的输入数据
            mask_indices: mask位置信息
            p_mask: mask概率信息
            current_input: 原始截取的输入数据

        Returns:
            tuple: (loss, outputs) - 损失值和模型输出
        """

        # 模型前向传播
        actual_mask_indices = (noisy_input == mask_token_id)
        mask_count = actual_mask_indices.sum().item()
        logger.debug(f"Forward pass: mask_count={mask_count}, input_shape={noisy_input.shape}")

        # 构建attention_mask：在训练阶段可选择性排除special token
        attention_mask = torch.ones_like(noisy_input, dtype=torch.long, device=device)

        # 新增配置：是否在训练时也排除special token attention
        exclude_in_training = getattr(config, 'exclude_special_tokens_in_training', False)

        if config.exclude_special_tokens_from_attention and special_token_ids and exclude_in_training:
            for token_name, token_id in special_token_ids.items():
                if token_id is not None:
                    # 将special token位置的attention_mask设为0
                    special_positions = (noisy_input == token_id)
                    if special_positions.any():
                        attention_mask = attention_mask & (~special_positions)
                        logger.debug(f"Excluded special token '{token_name}' (id={token_id}) from attention at {special_positions.sum().item()} positions")

            # 额外检查：确保原始输入中的special token也被排除
            # 考虑special token可能未被mask的情况
            for token_name, token_id in special_token_ids.items():
                if token_id is not None:
                    original_special_positions = (current_input == token_id)
                    if original_special_positions.any():
                        attention_mask = attention_mask & (~original_special_positions)
                        logger.debug(f"Excluded original special token '{token_name}' (id={token_id}) from attention at {original_special_positions.sum().item()} positions")

        outputs = model(input_ids=noisy_input, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs

        # 计算损失 - 与标准compute_loss保持一致的缩放方式
        if actual_mask_indices.any():
            # 1. 计算未缩放的损失（与标准方法一致）
            unscaled_loss = loss_func(logits.view(-1, logits.shape[-1]), current_input.view(-1), reduction="none").view(logits.shape[0], -1)

            # 2. 只对被mask的位置计算损失，并按时间步缩放（与标准方法的 unscaled_loss / t 一致）
            # 这里p_mask相当于标准方法中的t，表示mask概率/时间步
            masked_unscaled_loss = unscaled_loss[actual_mask_indices]
            masked_p_mask = p_mask[actual_mask_indices]
            scaled_loss = masked_unscaled_loss / masked_p_mask

            # 3. 按response中可能被加噪声的token总数进行归一化（与标准方法一致）
            # 标准方法: loss.sum() / (inputs["input_ids"].numel() - num_prompt_tokens)
            # 这里: loss.sum() / response_length (response_length是可能被加噪声的token总数)
            # 从p_mask中计算response长度（非零元素的数量）
            response_length = (p_mask > 0).sum().item()
            loss = scaled_loss.sum() / max(response_length, 1)  # 避免除零

            logger.info(f"Loss calculation: unscaled_loss={masked_unscaled_loss.sum().item():.4f}, "
                        f"scaled_loss={scaled_loss.sum().item():.4f}, final_loss={loss.item():.4f}, response_length={response_length}")
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            logger.info("Loss calculation: no masked positions, loss=0")

        return loss, outputs


    def _check_unmasked_special_tokens(self, input_ids, mask_positions, special_token_ids, current_response_length, prompt_length, sample_idx=0):
        """
        检查未被mask的special token，直接进行扩展决策

        Args:
            input_ids: 输入序列 [1, seq_len] - 单个样本
            mask_positions: mask位置 [1, seq_len] 布尔张量，可以为None（快速检查模式）
            special_token_ids: 特殊token ID映射
            current_response_length: 当前response的训练长度（不包括prompt）
            prompt_length: 单个样本的prompt长度 [1]
            sample_idx: 样本索引（用于日志显示）

        Returns:
            dict or None: 扩展决策字典，如果没有直接决策则返回None
        """
        # 由于现在处理单个样本，batch_size 应该是 1
        assert input_ids.size(0) == 1, f"Expected single sample, got batch_size={input_ids.size(0)}"

        expand_token_id = special_token_ids.get('expand')
        enough_token_id = special_token_ids.get('enough')

        # 定义special token应该出现的位置（0基索引）
        special_token_positions = [63, 127, 255, 511, 1023, 2047]

        # 处理单个样本（索引为0）
        i = 0  # 在tensor中的索引始终为0

        # 计算response部分的范围
        if isinstance(prompt_length, torch.Tensor):
            sample_prompt_len = prompt_length[i].item() if len(prompt_length.shape) > 0 else prompt_length.item()
        else:
            sample_prompt_len = prompt_length

        response_start = sample_prompt_len

        # 找到当前response长度范围内的special token位置
        valid_positions = []
        for pos in special_token_positions:
            if pos < current_response_length:  # 只检查当前response长度范围内的位置
                absolute_pos = response_start + pos
                if absolute_pos < input_ids.size(1):  # 确保不超出序列边界
                    valid_positions.append((pos, absolute_pos))

        if not valid_positions:
            return None  # 当前response长度范围内没有special token位置

        # 检查最大位置的special token（最接近当前扩展边界）
        valid_positions.sort(key=lambda x: x[0], reverse=True)
        max_pos, max_absolute_pos = valid_positions[0]

        logger.info(f"Sample {sample_idx}: Checking max position {max_pos} (absolute {max_absolute_pos}) within response length {current_response_length}")

        # 检查最大位置的token
        token_id = input_ids[i, max_absolute_pos].item()
        # 如果有mask_positions，检查是否被mask；如果没有，直接检查token类型
        is_masked = mask_positions[i, max_absolute_pos] if mask_positions is not None else False
        logger.info(f"Sample {sample_idx}: Token ID at max position {max_absolute_pos}: {token_id}, is_masked: {is_masked}")

        if not is_masked:
            if token_id == expand_token_id:
                # 发现未被mask的expand token，直接决策扩展
                logger.info(f"Sample {sample_idx}: Found unmasked <expand> at max position {max_absolute_pos} (relative {max_pos}), direct expand")
                return {
                    'expand': True,
                    'confidence': 1.0,  # 直接观察到，置信度最高
                    'method': 'direct_unmasked',
                    'absolute_position': max_absolute_pos,
                    'relative_position': max_pos,
                    'token_type': 'expand'
                }

            elif token_id == enough_token_id:
                # 发现未被mask的enough token，直接决策停止
                logger.info(f"Sample {sample_idx}: Found unmasked <enough> at max position {max_absolute_pos} (relative {max_pos}), direct stop")
                return {
                    'expand': False,
                    'confidence': 1.0,  # 直接观察到，置信度最高
                    'method': 'direct_unmasked',
                    'absolute_position': max_absolute_pos,
                    'relative_position': max_pos,
                    'token_type': 'enough'
                }

        # 没有找到直接决策
        return None



    def _detect_sample_expansion_with_outputs(self, outputs, input_ids, prompt_length, current_length, special_token_ids, config):
        """
        基于已有的模型输出检测单个样本是否需要扩展

        Args:
            outputs: 已计算的模型输出
            input_ids: 输入序列 [batch_size, seq_len]
            prompt_length: prompt长度
            current_length: 当前总序列长度（prompt + response）
            special_token_ids: 特殊token ID映射
            config: 配置对象

        Returns:
            dict: 扩展决策字典
        """
        # 计算当前response长度
        prompt_len = prompt_length[0].item() if isinstance(prompt_length, torch.Tensor) else prompt_length
        current_response_length = current_length - prompt_len

        # 定义special token应该出现的位置（0基索引，相对于response开始位置）
        special_token_positions = [63, 127, 255, 511, 1023, 2047]

        # 找到当前response长度范围内的最大special token位置
        max_special_pos = None
        for pos in reversed(special_token_positions):  # 从大到小检查
            if pos < current_response_length:
                max_special_pos = pos
                break

        if max_special_pos is None:
            logger.info(f"No special token position within current response length {current_response_length}")
            return {'expand': False, 'confidence': 0.0, 'reason': 'no_special_position'}

        # 计算绝对位置
        max_absolute_pos = prompt_len + max_special_pos

        logger.info(f"Using existing outputs for expansion detection at response position {max_special_pos} (absolute {max_absolute_pos})")

        # 使用已有的模型输出
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs

        # 检查最大special token位置的概率
        if max_absolute_pos < logits.size(1):
            position_logits = logits[0, max_absolute_pos, :]
            probabilities = torch.softmax(position_logits, dim=-1)

            expand_token_id = special_token_ids.get('expand')
            enough_token_id = special_token_ids.get('enough')

            # 记录所有特殊token的概率
            token_probs = {}
            for token_name, token_id in special_token_ids.items():
                if token_id is not None and token_id < logits.size(-1):
                    token_probs[token_name] = probabilities[token_id].item()

            logger.info(f"Model prediction at position {max_absolute_pos}: special_token_probs={token_probs}")

            # 检查expand token概率
            if expand_token_id is not None and expand_token_id < logits.size(-1):
                expand_prob = probabilities[expand_token_id].item()
                if expand_prob > config.confidence_threshold:
                    logger.info(f"Model predicts <expand> token, prob={expand_prob:.4f} > threshold={config.confidence_threshold}")
                    return {'expand': True, 'confidence': expand_prob, 'method': 'model_prediction_reused', 'position': max_absolute_pos}

            # 检查enough token概率
            if enough_token_id is not None and enough_token_id < logits.size(-1):
                enough_prob = probabilities[enough_token_id].item()
                if enough_prob > config.confidence_threshold:
                    logger.info(f"Model predicts <enough> token, prob={enough_prob:.4f} > threshold={config.confidence_threshold}")
                    return {'expand': False, 'confidence': enough_prob, 'reason': 'enough', 'method': 'model_prediction_reused', 'position': max_absolute_pos}

        logger.info(f"Model prediction confidence too low, stopping expansion")
        return {'expand': False, 'confidence': 0.0, 'reason': 'low_confidence', 'method': 'model_prediction_reused'}

    def _get_next_response_expansion_length(self, current_response_length, response_expansion_steps):
        """
        获取下一个response扩展长度
        """
        for step in response_expansion_steps:
            if step > current_response_length:
                return step
        return current_response_length  # 无法扩展




    def _train_dynamic_diffusion_step_multi_expansion(self, input_ids, prompt_length, model, loss_func,
                                                special_token_ids, device, config, tokenizer):
        """
        样本级独立动态扩展的扩散训练

        核心改进：
        1. 训练目标：response长度而不是整个序列长度
        2. 扩展步骤：64 -> 128 -> 256 -> 512 -> 1024 -> 2048 (response长度)
        3. 每个样本独立决定是否需要扩展
        4. 检查逻辑基于当前response长度范围内的最大special token位置
        """
        # 动态获取mask token ID
        mask_token_id = self.mask_token_id

        # Response长度扩展步骤
        response_expansion_steps = config.expansion_steps
        initial_response_length = config.initial_response_length
        max_expansions = config.max_expansions

        batch_size = input_ids.shape[0]
        logger.info(f"input_ids.shape: {input_ids.shape}")
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # 样本级状态跟踪（存储response长度而不是整个序列长度）
        sample_current_response_lengths = [initial_response_length] * batch_size
        sample_active = [True] * batch_size  # 哪些样本还需要继续训练
        sample_losses = []  # 记录每个样本的真实损失（不包括直接扩展的0损失）
        direct_expansion_count = 0  # 记录直接扩展的次数
        training_rounds = 0

        logger.info(f"Starting response-based training: batch_size={batch_size}, initial_response_length={initial_response_length}, max_expansions={max_expansions}")

        # 样本级独立训练循环
        while any(sample_active) and training_rounds < max_expansions:
            training_rounds += 1
            active_samples = [i for i, active in enumerate(sample_active) if active]

            if not active_samples:
                break

            logger.info(f"Training round {training_rounds}/{max_expansions}: active_samples={len(active_samples)}/{batch_size}")

            # 对每个活跃样本进行独立训练
            for sample_idx in active_samples:
                current_response_length = sample_current_response_lengths[sample_idx]

                # 计算实际可用的response长度
                prompt_len = prompt_length[sample_idx].item() if isinstance(prompt_length, torch.Tensor) else prompt_length
                max_possible_response_length = input_ids.size(1) - prompt_len

                # 确保不超过实际可用长度
                if current_response_length > max_possible_response_length:
                    sample_active[sample_idx] = False
                    logger.info(f"Sample {sample_idx}: response length {current_response_length} exceeds max possible {max_possible_response_length}, stopping")
                    continue

                # 计算当前整个序列的训练长度
                current_total_length = prompt_len + current_response_length

                # 阶段1：先对数据进行加噪处理（扩散过程）
                noisy_input, mask_indices, p_mask, current_input = self._apply_diffusion_noise(
                    input_ids[sample_idx:sample_idx+1],
                    prompt_length[sample_idx:sample_idx+1],
                    current_total_length,
                    special_token_ids,
                    device,
                    config,
                    mask_token_id
                )

                # 阶段2：基于加噪后的数据检查未被mask的special token
                direct_decision = self._check_unmasked_special_tokens(
                    noisy_input,  # 使用加噪后的数据
                    mask_indices,
                    special_token_ids,
                    current_response_length,
                    prompt_length[sample_idx:sample_idx+1],
                    sample_idx=sample_idx  # 传递正确的样本索引用于日志显示
                )

                if direct_decision is not None and direct_decision.get('expand', False):
                    expansion_decision = direct_decision
                    sample_loss = torch.tensor(0.0, device=device, requires_grad=True)  # 跳过损失计算
                    sample_outputs = None
                    direct_expansion_count += 1  # 记录直接扩展次数
                    logger.info(f"Sample {sample_idx}: Direct expansion, skipping loss calculation")
                else:
                    # 阶段3：如果没有明确扩展信号，进行损失计算和模型前向传播
                    sample_loss, sample_outputs = self._train_single_sample_at_length(
                        noisy_input, mask_indices, p_mask, current_input,
                        model, loss_func, special_token_ids, device, config, mask_token_id
                    )

                    if direct_decision is None:
                        expansion_decision = self._detect_sample_expansion_with_outputs(
                            sample_outputs,
                            noisy_input,  # 使用加噪后的输入
                            prompt_length[sample_idx:sample_idx+1],
                            current_total_length,
                            special_token_ids,
                            config
                        )
                    else:
                        expansion_decision = direct_decision

                    # 只记录真实计算的损失（不包括直接扩展的0损失）
                    sample_losses.append(sample_loss.item())

                # 累加损失（包括直接扩展的0损失，用于梯度计算）
                total_loss = total_loss + sample_loss

                # 根据扩展决策更新样本状态（修正：基于response长度扩展）
                if expansion_decision['expand']:
                    # 需要扩展，更新到下一个response长度
                    next_response_length = self._get_next_response_expansion_length(current_response_length, response_expansion_steps)
                    if next_response_length > current_response_length and next_response_length <= max_possible_response_length:
                        sample_current_response_lengths[sample_idx] = next_response_length
                        logger.info(f"Sample {sample_idx}: expanding response from {current_response_length} to {next_response_length}")
                    else:
                        # 无法再扩展，标记为完成
                        sample_active[sample_idx] = False
                        logger.info(f"Sample {sample_idx}: cannot expand response further, current={current_response_length}, stopping")
                else:
                    # 内容已足够，停止该样本的训练
                    sample_active[sample_idx] = False
                    logger.info(f"Sample {sample_idx}: response content sufficient, stopping")

        # 损失归一化
        total_samples_processed = len(sample_losses)

        if sample_losses:
            # 真实损失（不包括直接扩展的0损失）
            total_loss = total_loss / total_samples_processed

            logger.info(f"动态长度训练完成: rounds={training_rounds}, "
                       f"total_samples={total_samples_processed}, "
                       f"computed_losses={len(sample_losses)}, "
                       f"direct_expansions={direct_expansion_count}, "
                       f"total_loss={total_loss.item():.4f}")
        else:
            logger.warning("动态长度训练完成，但没有计算任何损失")
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        return total_loss


    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        重写损失计算方法，支持动态长度训练

        Args:
            model: 模型实例
            inputs: 输入数据
            num_items_in_batch: 批次中的项目数量
            return_outputs: 是否返回模型输出

        Returns:
            损失值或(损失值, 模型输出)的元组
        """
        # 如果启用动态长度训练且输入包含必要字段，使用动态长度训练
        if (self.enable_dynamic_length):
            return self._compute_dynamic_length_loss(model, inputs, num_items_in_batch, return_outputs)
        else:
            # 回退到标准dLLM训练
            return super().compute_loss(model, inputs, num_items_in_batch, return_outputs)

    def _compute_dynamic_length_loss(self, model, inputs, num_items_in_batch, return_outputs):
        """
        动态长度训练的损失计算
        """
        try:
            # 提取必要的输入数据
            input_ids = inputs['input_ids']
            prompt_lengths = inputs['prompt_lengths']

            # 获取tokenizer（优先使用processing_class，回退到tokenizer）
            tokenizer = getattr(self, 'processing_class', None) or getattr(self, 'tokenizer', None)

            class DynamicConfig:
                def __init__(self, config_dict):
                    # 映射YAML配置文件中的键名到代码中使用的属性名
                    self.expansion_check_ratio = config_dict.get('expansion_check_ratio', 0.35)
                    self.confidence_threshold = config_dict.get('confidence_threshold', 0.7)
                    self.max_expansions = config_dict.get('max_expansions', 5)
                    # 映射exclude_from_attention到exclude_special_tokens_from_attention
                    self.exclude_special_tokens_from_attention = config_dict.get('exclude_from_attention', False)
                    self.exclude_special_tokens_in_training = config_dict.get('exclude_special_tokens_in_training', False)

                    # 添加其他可能的配置项
                    self.initial_response_length = config_dict.get('initial_response_length', 64)
                    self.expansion_steps = config_dict.get('expansion_steps', [64, 128, 256, 512, 1024, 2048])

            # 创建配置对象以保持代码兼容性
            config_obj = DynamicConfig(self.dynamic_config)

            # 直接调用内置的完整动态长度训练函数
            loss = self._train_dynamic_diffusion_step_multi_expansion(
                input_ids=input_ids,
                prompt_length=prompt_lengths,
                model=model,
                loss_func=torch.nn.functional.cross_entropy,
                special_token_ids=self.special_token_ids,
                device=input_ids.device,
                config=config_obj,
                tokenizer=tokenizer
            )

            logger.debug(f"Dynamic length loss computed: {loss.item():.4f}")

            # 如果需要返回outputs，执行标准前向传播获取outputs
            if return_outputs:
                with torch.no_grad():
                    # 对于LLaDA模型，不传递labels以避免损失计算警告
                    # 使用原始输入进行标准前向传播，确保output长度与原始序列一致
                    # 传入的inputs要去除prompt_lengths
                    # 对于LLaDA模型，在评估时不传递labels以避免警告
                    eval_inputs = {k: v for k, v in inputs.items() if k not in ['prompt_lengths', 't', 'num_prompt_tokens', 'labels']}
                    outputs = model(**eval_inputs)
                    logger.debug(f"Standard forward pass for evaluation: logits.shape={outputs.logits.shape}")
                return (loss, outputs)
            else:
                return loss

        except Exception as e:
            logger.warning(f"Dynamic length training failed: {e}, falling back to standard training")
            # 自动回退到标准训练
            return super().compute_loss(model, inputs, num_items_in_batch, return_outputs)


    
    