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
        # 将prompt和response打印到example.jsonl文件中
        with open('example.jsonl', 'a') as f:
            f.write(json.dumps({"prompt": prompt, "response": response}) + '\n')
        # apply_chat_template 作用机制：将对话格式转换为模型特定的聊天格式
        # 添加特殊标记（如 <|im_start|>, <|im_end|> 等）
        inputs = tokenizer.apply_chat_template(prompt + response, tokenize=False)
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False) + "\n"
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






