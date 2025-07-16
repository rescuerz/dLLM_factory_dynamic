import torch
import torch.nn.functional as F
from transformers import DefaultDataCollator
import random
from tqdm import tqdm
import pickle
import torch.distributed as dist

class dLLMDataCollator(DefaultDataCollator):
    """
    Adds the forward noising process to the batch.
    Modify forward_process to change the noise schedule
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mask_token_id = kwargs["tokenizer"].mask_token_id
        self.tokenizer = kwargs["tokenizer"]
        if "max_length" in kwargs:
            self.max_length = kwargs["max_length"]
        if kwargs["tokenizer"].mask_token_id is None:
            assert (
                "mask_token_id" in kwargs
            ), "For dLLM models, pass a mask_token_id or set it equal to tokenizer.mask_token_id"
            self.mask_token_id = kwargs["mask_token_id"]

    def forward_process(self, batch, eps=1e-3):
        input_ids = batch["input_ids"]
        B, N = input_ids.shape
        if "t" not in batch:
            t = torch.rand((B,), device=input_ids.device)
        else:
            t = batch["t"]

        t = (1 - eps) * t + eps
        t = t[:, None].repeat(1, N)

        mask_indices = torch.rand((B, N), device=input_ids.device) < t
        noisy_batch = torch.where(mask_indices, self.mask_token_id, input_ids)
        return noisy_batch, t, mask_indices

    def __call__(self, batch):
        batch = super().__call__(batch)
        batch["labels"] = batch["input_ids"].clone()
        noisy_batch, batch["t"], mask_indices = self.forward_process(batch)
        batch["labels"][~mask_indices] = -100
        batch["num_prompt_tokens"] = 0
        if "prompt_lengths" in batch:
            prompt_lengths = batch.pop("prompt_lengths")
            prompt_length_indices = torch.arange(noisy_batch.shape[1]).unsqueeze(0)
            prompt_mask = prompt_length_indices < prompt_lengths
            noisy_batch[prompt_mask] = batch["input_ids"][prompt_mask].clone()
            batch["labels"][prompt_mask] = -100
            batch["num_prompt_tokens"] = prompt_mask.sum()
        batch["input_ids"] = noisy_batch.long()
        return batch
    
class dLLMDataCollator_dynamic_length(DefaultDataCollator):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mask_token_id = kwargs["tokenizer"].mask_token_id
        self.tokenizer = kwargs["tokenizer"]
        if "max_length" in kwargs:
            self.max_length = kwargs["max_length"]
        if kwargs["tokenizer"].mask_token_id is None:
            assert (
                "mask_token_id" in kwargs
            ), "For dLLM models, pass a mask_token_id or set it equal to tokenizer.mask_token_id"
            self.mask_token_id = kwargs["mask_token_id"]

    def __call__(self, batch):
        """
        动态长度训练专用数据整理器

        关键：手动处理padding，然后跳过加噪过程
        """
        # # 1. 手动处理padding（因为DefaultDataCollator不会自动padding）
        # input_ids_list = []
        # prompt_lengths_list = []
        # attention_mask_list = []

        # for item in batch:
        #     # 处理input_ids
        #     if isinstance(item['input_ids'], list):
        #         input_ids_list.append(torch.tensor(item['input_ids']))
        #     else:
        #         input_ids_list.append(item['input_ids'])

        #     # 处理prompt_lengths
        #     prompt_lengths_list.append(item.get('prompt_lengths', 0))

        #     # 处理attention_mask
        #     if 'attention_mask' in item:
        #         if isinstance(item['attention_mask'], list):
        #             attention_mask_list.append(torch.tensor(item['attention_mask']))
        #         else:
        #             attention_mask_list.append(item['attention_mask'])
        #     else:
        #         attention_mask_list.append(torch.ones_like(input_ids_list[-1]))

        # # 2. 执行padding
        # max_length = max(len(ids) for ids in input_ids_list)
        # batch_size = len(input_ids_list)

        # # 检测设备（从第一个tensor获取设备信息）
        # device = input_ids_list[0].device if hasattr(input_ids_list[0], 'device') else torch.device('cpu')

        # padded_input_ids = torch.full((batch_size, max_length), self.tokenizer.pad_token_id, dtype=torch.long, device=device)
        # padded_attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long, device=device)

        # for i, (input_ids, attention_mask) in enumerate(zip(input_ids_list, attention_mask_list)):
        #     seq_len = len(input_ids)
        #     padded_input_ids[i, :seq_len] = input_ids
        #     padded_attention_mask[i, :seq_len] = attention_mask[:seq_len] if len(attention_mask) >= seq_len else torch.ones(seq_len)

        # # 3. 构建batch字典（模拟super().__call__的结果）
        # batch = {
        #     'input_ids': padded_input_ids,
        #     'attention_mask': padded_attention_mask,
        #     'prompt_lengths': torch.tensor(prompt_lengths_list, dtype=torch.long, device=device)
        # }
        batch = super().__call__(batch)
        batch["labels"] = batch["input_ids"].clone()
        # noisy_batch, batch["t"], mask_indices = self.forward_process(batch)  # 注释掉
        noisy_batch = batch["input_ids"].clone()  # 使用干净数据替代加噪数据

        mask_indices = torch.zeros_like(batch["input_ids"], dtype=torch.bool)

        # 5. 设置labels（与标准版本逻辑相同，但由于没有真正的mask，所以labels保持完整）
        batch["labels"][~mask_indices] = -100  # 由于mask_indices全为False，这行实际上会把所有labels设为-100

        # 6. 处理prompt部分（与标准版本相同）
        batch["num_prompt_tokens"] = 0
        if "prompt_lengths" in batch:
            prompt_lengths = batch.pop("prompt_lengths")
            prompt_length_indices = torch.arange(noisy_batch.shape[1]).unsqueeze(0)
            # 确保prompt_lengths的维度正确
            if prompt_lengths.dim() == 1:
                prompt_lengths = prompt_lengths.unsqueeze(1)  # [batch_size, 1]
            prompt_mask = prompt_length_indices < prompt_lengths
            noisy_batch[prompt_mask] = batch["input_ids"][prompt_mask].clone()
            batch["labels"][prompt_mask] = -100
            batch["num_prompt_tokens"] = prompt_mask.sum()

            # 重新添加prompt_lengths供动态长度训练使用
            batch["prompt_lengths"] = prompt_lengths

        # 7. 设置最终的input_ids（干净数据）
        batch["input_ids"] = noisy_batch.long()

        # 8. 添加虚拟的t字段以保持格式兼容性
        batch["t"] = torch.ones(batch["input_ids"].size(0), device=batch["input_ids"].device)

        return batch
