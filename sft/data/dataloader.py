import torch
import torch.nn.functional as F
from transformers import DefaultDataCollator
import random
from tqdm import tqdm
import pickle
import torch.distributed as dist

class dLLMDataCollator(DefaultDataCollator):
    """
    dLLM（Diffusion Language Model）数据整理器

    这是dLLM训练框架的核心组件，负责实现前向噪声过程（Forward Noising Process）。


    核心功能：
    1. 向输入序列添加随机噪声（用mask token替换部分原始token）
    2. 根据时间步长t控制噪声强度（t越大，噪声越多）
    3. 保护prompt部分不被加噪声（确保用户输入完整性）
    4. 设置训练标签，只对被mask的token计算损失
    """

    def __init__(self, *args, **kwargs):
        """
        初始化dLLM数据整理器

        Args:
            *args: 传递给父类的位置参数
            **kwargs: 关键字参数，必须包含：
                - tokenizer: 分词器对象，用于获取mask_token_id
                - max_length (可选): 序列最大长度
                - mask_token_id (可选): 如果tokenizer没有mask_token_id时手动指定
                - enable_dynamic_length (可选): 是否启用动态长度功能
                - dynamic_config (可选): 动态长度配置字典
        """
        super().__init__()
        # 获取mask token的ID，这是dLLM噪声过程的核心token
        # mask token用于替换原始token，模拟扩散过程中的噪声
        self.mask_token_id = kwargs["tokenizer"].mask_token_id
        self.tokenizer = kwargs["tokenizer"]

        # 可选的最大序列长度参数
        if "max_length" in kwargs:
            self.max_length = kwargs["max_length"]

        # 如果分词器没有mask_token_id，则必须手动提供
        # 这对于dLLM模型是必需的，因为噪声过程依赖于mask token
        if kwargs["tokenizer"].mask_token_id is None:
            assert (
                "mask_token_id" in kwargs
            ), "For dLLM models, pass a mask_token_id or set it equal to tokenizer.mask_token_id"
            self.mask_token_id = kwargs["mask_token_id"]

        # 动态长度功能配置（阶段1：基础集成）
        self.enable_dynamic_length = kwargs.get("enable_dynamic_length", False)
        self.dynamic_config = kwargs.get("dynamic_config", None)

        # 初始化动态长度相关的状态
        if self.enable_dynamic_length and self.dynamic_config:
            self._init_dynamic_length_config()

    def _init_dynamic_length_config(self):
        """
        初始化动态长度配置
        设置默认值并验证配置参数的有效性
        """
        # 设置默认配置值
        default_config = {
            'initial_response_length': 64,
            'expansion_steps': [64, 128, 256, 512, 1024, 2048],
            'max_expansions': 5,
            'confidence_threshold': 0.7,
            'expansion_check_ratio': 0.35,
            'exclude_special_tokens_from_attention': False
        }

        # 确保 dynamic_config 不为 None
        if self.dynamic_config is None:
            self.dynamic_config = {}

        # 合并用户配置和默认配置
        for key, default_value in default_config.items():
            if key not in self.dynamic_config:
                self.dynamic_config[key] = default_value

        # 获取特殊token ID映射（用于动态长度处理）
        self.special_token_ids = self._get_special_token_ids()

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
        for key, token in SPECIAL_TOKENS.items():
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id != self.tokenizer.unk_token_id:
                token_ids[key] = token_id
            else:
                # 如果特殊token不存在，记录警告但不中断训练
                print(f"Warning: Special token '{token}' not found in tokenizer vocabulary")

        return token_ids

    def forward_process(self, batch, eps=1e-3):
        """
        前向噪声过程的统一入口
        根据配置选择使用标准或动态长度的前向过程

        Args:
            batch (dict): 包含input_ids的批次数据
            eps (float): 防止t=0的小常数，确保总有一定的噪声

        Returns:
            tuple: (noisy_batch, t, mask_indices)
                - noisy_batch: 添加噪声后的token序列
                - t: 扩展后的时间步长张量 [B, N]
                - mask_indices: 布尔掩码，标记哪些位置被添加了噪声
        """
        if self.enable_dynamic_length and self.dynamic_config:
            # 使用动态长度的前向过程
            return self._dynamic_forward_process(batch, eps)
        else:
            # 使用标准的前向过程
            return self._standard_forward_process(batch, eps)

    def _standard_forward_process(self, batch, eps=1e-3):
        """
        标准的dLLM前向噪声过程（保持原有逻辑不变）

        dLLM的核心机制：根据时间步长t，随机将部分原始token替换为mask token，
        模拟扩散过程中信息的逐渐丢失。时间步长t控制噪声强度：
        - t接近0：几乎不添加噪声，保持原始序列
        - t接近1：大量添加噪声，大部分token被mask

        Args:
            batch (dict): 包含input_ids的批次数据
            eps (float): 防止t=0的小常数，确保总有一定的噪声

        Returns:
            tuple: (noisy_batch, t, mask_indices)
                - noisy_batch: 添加噪声后的token序列
                - t: 扩展后的时间步长张量 [B, N]
                - mask_indices: 布尔掩码，标记哪些位置被添加了噪声
        """
        # 原始token序列 [B, N]
        input_ids = batch["input_ids"]
        # B=批次大小, N=序列长度
        B, N = input_ids.shape

        # 生成或获取时间步长t
        # t控制噪声强度，每个样本可以有不同的噪声水平
        if "t" not in batch:
            # 如果批次中没有预设的t，则随机生成 [0, 1] 之间的值
            t = torch.rand((B,), device=input_ids.device)
        else:
            # 使用预设的时间步长（通常在评估时使用固定值）
            t = batch["t"]

        # 时间步长缩放和扩展
        # (1-eps)*t + eps 确保t不会完全为0，总是保持最小噪声水平
        t = (1 - eps) * t + eps
        # 将t从 [B] 扩展到 [B, N]，使每个token位置都有对应的噪声概率
        t = t[:, None].repeat(1, N)

        # 生成噪声掩码：随机决定哪些token位置需要被mask
        # 对每个位置生成 [0,1] 随机数，如果小于t则该位置被mask
        # t越大，被mask的token越多（噪声越强）
        mask_indices = torch.rand((B, N), device=input_ids.device) < t

        # 应用噪声：将被选中的位置替换为mask_token_id
        # torch.where(condition, x, y): 如果condition为True选择x，否则选择y
        noisy_batch = torch.where(mask_indices, self.mask_token_id, input_ids)

        return noisy_batch, t, mask_indices

    def __call__(self, batch):
        """
        这个方法将原始批次数据转换为dLLM训练所需的格式：
        1. 创建训练标签（labels）
        2. 应用前向噪声过程
        3. 保护prompt部分不被加噪声
        4. 设置损失计算掩码
        """
        batch = super().__call__(batch)

        # labels用于计算损失，指示模型应该预测什么token
        batch["labels"] = batch["input_ids"].clone()

        # 应用前向噪声过程，获取加噪后的序列和掩码信息
        noisy_batch, batch["t"], mask_indices = self.forward_process(batch)

        # 设置损失计算掩码：只对被mask的token计算损失
        # 将未被mask的位置标记为-100，这些位置在损失计算时会被忽略
        batch["labels"][~mask_indices] = -100

        # 初始化prompt token计数器
        batch["num_prompt_tokens"] = 0

        # 处理prompt保护机制：确保用户输入部分不被加噪声
        if "prompt_lengths" in batch:
            # 获取每个样本的prompt长度
            prompt_lengths = batch.pop("prompt_lengths")

            # 创建位置索引矩阵 [1, N]，用于标识每个位置的索引
            prompt_length_indices = torch.arange(noisy_batch.shape[1]).unsqueeze(0)

            # 创建prompt掩码：标记哪些位置属于prompt部分
            # prompt_mask[i, j] = True 表示样本i的第j个位置属于prompt
            prompt_mask = prompt_length_indices < prompt_lengths

            # 恢复prompt部分的原始token：将prompt区域的噪声token替换回原始token
            noisy_batch[prompt_mask] = batch["input_ids"][prompt_mask].clone()

            # 将prompt部分的标签设为-100，表示不对prompt部分计算损失
            batch["labels"][prompt_mask] = -100

            # 统计prompt部分的token数量，用于损失归一化
            batch["num_prompt_tokens"] = prompt_mask.sum()

        # 更新批次数据：使用加噪后的序列作为模型输入
        batch["input_ids"] = noisy_batch.long()

        return batch

    def _dynamic_forward_process(self, batch, eps=1e-3):
        """
        动态长度的前向噪声过程 - 阶段2：支持样本级独立训练
        支持渐进式长度扩展和特殊token感知的噪声添加

        Args:
            batch (dict): 包含input_ids的批次数据，可能包含：
                - input_ids: 输入token序列 [B, N]
                - prompt_lengths: 每个样本的prompt长度 [B]
                - current_response_length: 当前训练的response长度（可选）
                - sample_expansion_lengths: 每个样本的独立扩展长度 [B]（可选）
            eps (float): 防止t=0的小常数

        Returns:
            tuple: (noisy_batch, t, mask_indices)
        """
        input_ids = batch["input_ids"]
        B, N = input_ids.shape
        device = input_ids.device

        # 获取样本级独立的扩展长度
        sample_expansion_lengths = batch.get("sample_expansion_lengths", None)
        current_response_length = batch.get("current_response_length", None)
        prompt_lengths = batch.get("prompt_lengths", None)

        # 确定每个样本的训练长度
        if sample_expansion_lengths is not None:
            # 使用样本级独立长度
            training_lengths = sample_expansion_lengths
        elif current_response_length is not None:
            # 使用统一的当前长度
            if isinstance(current_response_length, (int, float)):
                training_lengths = torch.full((B,), current_response_length, device=device)
            else:
                training_lengths = current_response_length
        else:
            # 使用配置中的初始长度
            initial_length = self.dynamic_config['initial_response_length'] if self.dynamic_config else 64
            training_lengths = torch.full((B,), initial_length, device=device)

        # 生成或获取时间步长t
        if "t" not in batch:
            # 为每个样本生成适应其训练长度的时间步长
            t = self._generate_sample_adaptive_timesteps(training_lengths, device)
        else:
            t = batch["t"]

        # 时间步长缩放
        t = (1 - eps) * t + eps

        # 创建样本级有效长度掩码
        effective_length_mask = self._create_sample_adaptive_mask(
            B, N, training_lengths, prompt_lengths, device
        )

        # 扩展时间步长到序列维度，但只对有效部分生效
        t_expanded = t[:, None].repeat(1, N)
        t_expanded = t_expanded * effective_length_mask

        # 生成噪声掩码，考虑特殊token位置
        mask_indices = self._generate_special_token_aware_mask(
            input_ids, t_expanded, effective_length_mask, training_lengths, prompt_lengths
        )

        # 应用噪声
        noisy_batch = torch.where(mask_indices, self.mask_token_id, input_ids)

        return noisy_batch, t_expanded, mask_indices

    def _generate_sample_adaptive_timesteps(self, training_lengths, device):
        """
        为每个样本生成适应其训练长度的时间步长

        Args:
            training_lengths: 每个样本的训练长度 [B]
            device: 设备

        Returns:
            torch.Tensor: 时间步长 [B]
        """
        batch_size = training_lengths.shape[0]

        # 基础随机时间步长
        t = torch.rand((batch_size,), device=device)

        # 根据每个样本的训练长度调整时间步长分布
        # 较短的序列使用较小的时间步长，较长的序列使用较大的时间步长
        length_factors = torch.clamp(training_lengths.float() / 512.0, 0.1, 1.0)  # 归一化到[0.1,1]
        t = t * (0.3 + 0.7 * length_factors)  # 调整时间步长范围

        return t

    def _create_sample_adaptive_mask(self, batch_size, seq_length, training_lengths,
                                   prompt_lengths, device):
        """
        创建样本级自适应的有效长度掩码

        Args:
            batch_size: 批次大小
            seq_length: 序列长度
            training_lengths: 每个样本的训练长度 [B]
            prompt_lengths: 每个样本的prompt长度 [B]
            device: 设备

        Returns:
            torch.Tensor: 有效长度掩码 [B, N]
        """
        # 创建位置索引
        position_indices = torch.arange(seq_length, device=device).unsqueeze(0).repeat(batch_size, 1)

        if prompt_lengths is not None:
            # 如果提供了prompt长度，只对response部分的有效长度添加噪声
            prompt_lengths = prompt_lengths.unsqueeze(1) if prompt_lengths.dim() == 1 else prompt_lengths
            training_lengths = training_lengths.unsqueeze(1) if training_lengths.dim() == 1 else training_lengths

            response_start = prompt_lengths
            response_end = prompt_lengths + training_lengths

            # 创建response区域的掩码
            mask = (position_indices >= response_start) & (position_indices < response_end)
        else:
            # 如果没有prompt长度信息，对整个序列的有效长度添加噪声
            training_lengths = training_lengths.unsqueeze(1) if training_lengths.dim() == 1 else training_lengths
            mask = position_indices < training_lengths

        return mask.float()

    def _generate_special_token_aware_mask(self, input_ids, t_expanded, effective_length_mask,
                                         training_lengths, prompt_lengths):
        """
        生成特殊token感知的噪声掩码

        在特殊token位置应用特殊的噪声策略

        Args:
            input_ids: 输入token序列 [B, N]
            t_expanded: 扩展的时间步长 [B, N]
            effective_length_mask: 有效长度掩码 [B, N]
            training_lengths: 每个样本的训练长度 [B]
            prompt_lengths: 每个样本的prompt长度 [B]

        Returns:
            torch.Tensor: 噪声掩码 [B, N]
        """
        B, N = input_ids.shape
        device = input_ids.device

        # 生成基础噪声掩码
        base_mask = torch.rand((B, N), device=device) < t_expanded

        # 应用有效长度掩码
        mask_indices = base_mask & effective_length_mask.bool()

        # 如果有特殊token ID，应用特殊token感知的策略
        if hasattr(self, 'special_token_ids') and self.special_token_ids:
            mask_indices = self._apply_special_token_mask_strategy(
                input_ids, mask_indices, training_lengths, prompt_lengths
            )

        return mask_indices

    def _apply_special_token_mask_strategy(self, input_ids, mask_indices, training_lengths, prompt_lengths):
        """
        应用特殊token的掩码策略

        Args:
            input_ids: 输入token序列 [B, N]
            mask_indices: 基础掩码 [B, N]
            training_lengths: 每个样本的训练长度 [B]
            prompt_lengths: 每个样本的prompt长度 [B]

        Returns:
            torch.Tensor: 调整后的掩码 [B, N]
        """
        expand_token_id = self.special_token_ids.get('expand', None)
        enough_token_id = self.special_token_ids.get('enough', None)

        if expand_token_id is None or enough_token_id is None:
            return mask_indices

        # 获取特殊token掩码策略配置
        if self.dynamic_config is not None:
            exclude_from_attention = self.dynamic_config.get('special_tokens', {}).get('exclude_from_attention', False)
        else:
            exclude_from_attention = False

        if exclude_from_attention:
            # 如果配置为排除特殊token，则不对特殊token位置添加噪声
            special_token_positions = (input_ids == expand_token_id) | (input_ids == enough_token_id)
            mask_indices = mask_indices & (~special_token_positions)

        return mask_indices

    def _generate_dynamic_timesteps(self, batch_size, current_response_length, device):
        """
        为动态长度训练生成时间步长
        可以根据当前训练长度调整时间步长的分布

        Args:
            batch_size: 批次大小
            current_response_length: 当前训练的response长度
            device: 设备

        Returns:
            torch.Tensor: 时间步长 [B]
        """
        # 基础随机时间步长
        t = torch.rand((batch_size,), device=device)

        # 可以根据当前长度调整时间步长分布
        # 例如：较短的序列使用较小的时间步长，较长的序列使用较大的时间步长
        length_factor = min(current_response_length / 512.0, 1.0)  # 归一化到[0,1]
        t = t * (0.5 + 0.5 * length_factor)  # 调整时间步长范围

        return t

    def _create_effective_length_mask(self, batch_size, seq_length, current_response_length,
                                    prompt_lengths, device):
        """
        创建有效长度掩码，确定哪些位置应该参与噪声过程

        Args:
            batch_size: 批次大小
            seq_length: 序列长度
            current_response_length: 当前训练的response长度
            prompt_lengths: 每个样本的prompt长度
            device: 设备

        Returns:
            torch.Tensor: 有效长度掩码 [B, N]
        """
        # 创建位置索引
        position_indices = torch.arange(seq_length, device=device).unsqueeze(0).repeat(batch_size, 1)

        if prompt_lengths is not None:
            # 如果提供了prompt长度，只对response部分的有效长度添加噪声
            prompt_lengths = prompt_lengths.unsqueeze(1)  # [B, 1]
            response_start = prompt_lengths
            response_end = prompt_lengths + current_response_length

            # 创建response区域的掩码
            mask = (position_indices >= response_start) & (position_indices < response_end)
        else:
            # 如果没有prompt长度信息，对整个序列的有效长度添加噪声
            mask = position_indices < current_response_length

        return mask.float()



