import torch
import torch.nn.functional as F
from transformers import Trainer
from transformers import DefaultDataCollator
import random
from tqdm import tqdm
import pickle
import torch.distributed as dist
import json

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
...
</answer>
"""



def insert_special_tokens(tokenizer, inputs: str, prompt_text: str, original_response: str) -> str:
    """
    在apply_chat_template后插入特殊token

    Args:
        tokenizer: 分词器
        inputs: apply_chat_template后的完整文本 (prompt + response)
        prompt_text: apply_chat_template后的prompt部分
        original_response: 原始response文本内容

    Returns:
        插入特殊token后的完整文本
    """
    # 特殊token定义（在数据处理中直接定义，避免循环导入）
    SPECIAL_TOKENS = {
        "expand": "<|expand|>",  # 扩展token
        "enough": "<|enough|>"   # 结束token
    }

    try:
        # 1. 计算原始response的token长度（用于决定插入策略）
        original_response_tokens = tokenizer.encode(original_response, add_special_tokens=False)
        original_response_length = len(original_response_tokens)

        # 2. 对完整文本进行tokenization
        full_tokens = tokenizer.encode(inputs, add_special_tokens=False)
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

        # 3. 计算response部分在完整token序列中的起始位置
        response_start_idx = len(prompt_tokens)

        # 4. 获取特殊token的ID，并检查是否有效
        enough_token_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS['enough'])
        expand_token_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS['expand'])

        if enough_token_id is None or expand_token_id is None:
            print(f"Warning: Special tokens not found in tokenizer vocabulary")
            print(f"请确保在训练脚本中调用了 setup_model_and_tokenizer_for_special_tokens 函数")
            return inputs

        # 5. 根据原始response长度决定插入策略
        if original_response_length <= 64:
            # 短回答：在末尾添加<|enough|>token
            modified_tokens = full_tokens.copy()
            modified_tokens.append(enough_token_id)
            # print(f"Inserted <|enough|> at the end for short response (length: {original_response_length})")
        else:
            # 长回答：在指定位置插入<|expand|>token
            target_positions = [63, 127, 255, 511, 1023]  # 第64、128、256、512、1024个token

            # 先提取response部分的token
            response_tokens = full_tokens[response_start_idx:].copy()

            # 从前往后插入，确保每个<|expand|> token最终位于正确位置
            for final_pos in target_positions:
                # 检查原始response长度是否足够长，需要插入这个位置的token
                if final_pos < original_response_length:
                    # 直接在目标最终位置插入
                    # 由于我们从前往后插入，当前的final_pos就是我们要插入的位置
                    insert_pos = final_pos

                    # 确保插入位置在有效范围内
                    if insert_pos >= 0 and insert_pos < len(response_tokens):
                        # 在计算出的位置插入expand token ID
                        response_tokens.insert(insert_pos, expand_token_id)
                        # print(f"Inserted <|expand|> at response position {insert_pos}, final position will be {final_pos}")

            # 重新组合完整的token序列
            modified_tokens = full_tokens[:response_start_idx] + response_tokens

        # 6. 解码为文本
        modified_inputs = tokenizer.decode(modified_tokens, skip_special_tokens=False)
        return modified_inputs

    except Exception as e:
        print(f"Error inserting special tokens: {e}")
        return inputs  # 出错时返回原始文本

def preprocess_dataset_original(data, tokenizer, max_length, test_split=0.01):
    preprocessed_data = []
    for i in tqdm(range(len(data)), desc="Preprocessing dataset"):
        question = SYSTEM_PROMPT + "\n\n" + data[i]["question"]
        trajectory = f"<reasoning>{data[i]['thinking_trajectories'][0]}</reasoning>\n<answer>{data[i]['attempt']}</answer>"
        prompt = [{"role": "user", "content": question}]
        response = [{"role": "assistant", "content": trajectory}]
        inputs = tokenizer.apply_chat_template(prompt + response, tokenize=False)
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False) + "\n"
        tokenized_input = tokenizer(
            inputs, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length"
        ).input_ids.squeeze(0)
        num_tokens = tokenized_input.shape[0]
        tokenized_prompt = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        preprocessed_data.append(
            {
                "input_ids": tokenized_input,
                "prompt_lengths": tokenized_prompt.attention_mask.sum(-1),
            }
        )

    random.shuffle(preprocessed_data)
    test_data = preprocessed_data[: int(len(preprocessed_data) * test_split)]
    train_data = preprocessed_data[int(len(preprocessed_data) * test_split) :]
    return train_data, test_data

def preprocess_dataset(data, tokenizer, max_length, test_split=0.01):
    """
    数据格式：
    {
        'question': 'Solve: 2x + 3 = 7',
        'thinking_trajectories': ['First, subtract 3 from both sides: 2x = 4. Then divide by 2: x = 2.'],
        'attempt': 'x = 2'
    }
    # 步骤1：构建question
    question = 
    Respond in the following format:
    <reasoning>
    Your reasoning here
    </reasoning>
    <answer>
    ...
    </answer>

    Solve: 2x + 3 = 7
    # 步骤2：构建trajectory
    trajectory = "<reasoning>First, subtract 3 from both sides: 2x = 4. Then divide by 2: x = 2.</reasoning>\n<answer>x = 2</answer>"

    # 步骤3-4：应用聊天模板
    inputs = "<|im_start|>user\n[question内容]\n<|im_end|>\n<|im_start|>assistant\n[trajectory内容]\n<|im_end|>"

    # 步骤5：分词
    tokenized_input = tensor([1, 123, 456, 789, ..., 0, 0, 0])  # [4096]

    # 步骤6：输出
    output_sample = {
        "input_ids": tensor([1, 123, 456, 789, ..., 0, 0, 0]),
        "prompt_lengths": tensor(156)  # 用户输入部分长度
    }
    """
    preprocessed_data = []

    for i in tqdm(range(len(data)), desc="Preprocessing dataset"):
        # 构建系统提示词 + 问题
        question = SYSTEM_PROMPT + "\n\n" + data[i]["question"]
        # 根据数据集格式构建推理轨迹 + 答案
        if 'thinking_trajectories' in data[i] and 'attempt' in data[i]:
            # simplescaling/s1K 格式
            # print("simplescaling/s1K 格式")
            reasoning = data[i]['thinking_trajectories'][0]
            answer = data[i]['attempt']
        elif 'answer' in data[i]:
            # GSM8K 等标准格式，从answer中提取推理过程和最终答案
            full_answer = data[i]['answer']
            # 尝试分离推理过程和最终答案
            # 如果answer中包含 #### 开头，则认为 #### 后面的内容是最终答案
            if '####' in full_answer:
                # print("GSM8K 格式")
                answer = full_answer.split('####')[1].strip()
                reasoning = full_answer.split('####')[0].strip()
            else:
                if '\n' in full_answer:
                    parts = full_answer.split('\n')
                    reasoning = '\n'.join(parts[:-1]).strip()
                    answer = parts[-1].strip()
                else:
                    reasoning = full_answer
                    answer = full_answer
        else:
            raise ValueError(f"不支持的数据格式，样本 {i} 缺少必要字段")

        trajectory = f"<reasoning>{reasoning}</reasoning>\n<answer>{answer}</answer>"
        # 构建对话格式
        prompt = [{"role": "user", "content": question}]
        response = [{"role": "assistant", "content": trajectory}]
        
        sample = {
            "sample_id": i,
            "prompt": prompt,
            "response": response,
        }

        # apply_chat_template 作用机制：将对话格式转换为模型特定的聊天格式
        # 添加特殊标记（如 <|im_start|>, <|im_end|> 等）
        inputs = tokenizer.apply_chat_template(prompt + response, tokenize=False)
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False) + "\n"

        # 插入特殊token：<|expand|>在长回答的指定位置，<|enough|>在短回答末尾
        inputs = insert_special_tokens(tokenizer, inputs, prompt, trajectory)

        # 将处理好的数据保存到example.jsonl文件中
        processed_sample = {
            "sample_id": i,
            "processed_inputs": inputs,
        }

        tokenized_input = tokenizer(
            inputs, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length"
        ).input_ids.squeeze(0)
        num_tokens = tokenized_input.shape[0]
        tokenized_prompt = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        # 单独的prompt用于计算用户输入部分的长度，在训练时屏蔽这部分的损失
        preprocessed_data.append(
            {
                "input_ids": tokenized_input,
                "prompt_lengths": tokenized_prompt.attention_mask.sum(-1),
            }
        )

    random.shuffle(preprocessed_data)
    test_data = preprocessed_data[: int(len(preprocessed_data) * test_split)]
    train_data = preprocessed_data[int(len(preprocessed_data) * test_split) :]

    return train_data, test_data






