import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union,Type,TypeVar

import jinja2
import torch
import torch.nn.functional as F
import transformers
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs,
    find_executable_batch_size,
)
from datasets import Dataset
from accelerate.utils import get_max_memory
from huggingface_hub import HfApi
from packaging import version
from peft import PeftModel
from peft import __version__ as PEFT_VERSION
from tqdm import tqdm
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,

)

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import (
    Collator,
    clear_torch_cache,
    configure_pad_token,
    get_dtype,
    handle_stop_sequences,
    pad_and_concat,
    stop_sequences_criteria,
)

eval_logger = logging.getLogger(__name__)

# 导入动态长度生成功能
try:
    from .dynamic_generation import generate_dynamic_length
    DYNAMIC_GENERATION_AVAILABLE = True
except ImportError:
    DYNAMIC_GENERATION_AVAILABLE = False
    eval_logger.warning("Dynamic generation module not available. Dynamic length generation will be disabled.")
from dllm_cache.cache import  dLLMCacheConfig,dLLMCache
from dllm_cache.hooks import  register_cache_LLaDA
from dataclasses import asdict
T = TypeVar("T", bound="LM")
from lm_eval.api.model import LM



import torch
from dllm_cache.cache import dLLMCache
import torch.nn.functional as F
import numpy as np


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits.exp()
    noise = torch.rand_like(logits)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
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


def generate(
    input_ids,
    attention_mask,
    model,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
):
    with torch.no_grad():
        batch_size, prompt_length = input_ids.shape
        # 创建完整序列：prompt + 待生成部分（全部用mask_id填充）
        x = torch.full(
            (batch_size, prompt_length + gen_length),
            mask_id,
            dtype=torch.long,
            device=model.device,
        )
        # 保留原始的prompt部分
        x[:, :prompt_length] = input_ids
        # 创建一个布尔掩码，用于标记哪些位置是待生成的部分
        prompt_index = x != mask_id

        # 确保生成长度是block长度的整数倍， 计算生成块数num_blocks
        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        # 确保steps是num_blocks的整数倍， 计算每个块的生成步数steps_per_block
        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks

        feature_cache = dLLMCache()
        feature_cache.reset_cache(prompt_length)
        # 对每个块进行 steps_per_block次迭代：
        for num_block in range(num_blocks):
            # 计算当前块的起始和结束索引
            start_idx = prompt_length + num_block * block_length
            end_idx = prompt_length + (num_block + 1) * block_length
            # 获取当前块的输入和掩码
            block_x = x[:, start_idx:end_idx]
            block_mask_index = block_x == mask_id
            # 计算每一步需要unmask的token数量
            num_transfer_tokens = get_num_transfer_tokens(
                block_mask_index, steps_per_block
            )

            for i in range(steps_per_block):
                # 标记哪些位置是待生成的部分
                mask_index = x == mask_id
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
                        # 无缓存的CFG， 计算CFG残差
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
                    # 无CFG， 直接计算logits
                    logits = model(x, attention_mask=attention_mask).logits[
                        :, prompt_length:
                    ]
                # 添加Gumbel噪声
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                # 选择概率最高的token
                x0 = torch.argmax(logits_with_noise, dim=-1)

                # 低置信度重掩蔽
                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )
                elif remasking == "random":
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                # 确保不会更新当前快之外的位置
                x0_p[:, (num_block + 1) * block_length :] = -np.inf

                # 只在mask位置应用预测，非mask位置保持不变
                x0 = torch.where(
                    mask_index[:, prompt_length:], x0, x[:, prompt_length:]
                )
                # 计算置信度，只在mask位置应用预测，非mask位置置信度为-inf
                confidence = torch.where(mask_index[:, prompt_length:], x0_p, -np.inf)

                # 计算需要转移的token索引，为每个样本选择置信度最高的num_transfer_tokens[j, i]个token
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
        return x[:, prompt_length:]



@register_model("LLaDA")
class LLaDA(TemplateLM):
    AUTO_MODEL_CLASS = None
    _DEFAULT_MAX_LENGTH = 20480

    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        backend: Literal["default", "causal", "seq2seq"] = "causal",
        revision: Optional[str] = "main",
        subfolder: Optional[str] = None,
        tokenizer: Optional[
            Union[
                str,
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
            ]
        ] = None,
        truncation: Optional[bool] = False,
        logits_cache: bool = True,
        max_length: Optional[int] = None,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[Union[int]] = 1,
        max_batch_size: Optional[int] = 64,
        trust_remote_code: Optional[bool] = True,
        use_fast_tokenizer: Optional[bool] = True,
        add_bos_token: Optional[bool] = False,
        escape_until:Optional[bool] = False,
        prefix_token_id: Optional[int] = None,
        # arguments used for splitting a model across GPUs naively.
        # only used if `parallelize=True`.
        parallelize: Optional[bool] = False,
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[Union[str, os.PathLike]] = "./offload",
        # PEFT, delta weights and quantization options
        peft: Optional[str] = None,
        delta: Optional[str] = None,
        autogptq: Optional[Union[bool, str]] = False,
        gptqmodel: Optional[bool] = False,
        gguf_file: Optional[str] = None,
        is_feature_cache: bool = False,
        is_cfg_cache: bool = False,
        prompt_interval_steps: int = 1,
        gen_interval_steps: int = 1,
        cfg_interval_steps: int = 1,
        transfer_ratio:float = 0.0,
        mc_num: int = 1024,
        remasking: str = "low_confidence",
        mask_id: int = 126336,
        is_check_greedy : bool =True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.mc_num = mc_num
        self.mask_id = mask_id
        self.remasking = remasking
        self.pretrained = pretrained
        self.prompt_interval_steps = prompt_interval_steps
        self.gen_interval_steps = gen_interval_steps
        self.cfg_interval_steps = cfg_interval_steps
        self.transfer_ratio = transfer_ratio
        self.is_check_greedy = is_check_greedy
        self.add_bos_token = add_bos_token
        self.escape_until = escape_until
        if not isinstance(pretrained, str):
            eval_logger.warning(
                "`pretrained` model kwarg is not of type `str`. Many other model arguments may be ignored. Please do not launch via accelerate or use `parallelize=True` if passing an existing model this way."
            )
            assert not parallelize, (
                "`parallelize=True` is not compatible with passing pre-initialized model to `pretrained`"
            )
            self._model = pretrained
            self._device = self._model.device
            self._config = self._model.config
            gpus = 0

        else:
            assert isinstance(device, str)
            assert isinstance(pretrained, str)
            assert isinstance(batch_size, (int, str))

            gpus = torch.cuda.device_count()
            accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
            accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
            if accelerator.num_processes > 1:
                self.accelerator = accelerator

            if "npu" in accelerator.device.type:
                gpus = torch.npu.device_count()

            # using one process with no model parallelism
            if not (parallelize or accelerator.num_processes > 1):
                # use user-passed device
                device_list = set(
                    ["cuda", "cpu"]
                    + [f"cuda:{i}" for i in range(gpus)]
                    + ["mps", "mps:0"]
                    + [f"npu:{i}" for i in range(gpus)]
                )
                if device and device in device_list:
                    self._device = torch.device(device)
                    eval_logger.info(f"Using device '{device}'")
                    if device in ("mps", "mps:0") and version.parse(
                        torch.__version__
                    ) < version.parse("2.1"):
                        raise RuntimeError(
                            f"mps requires torch >= 2.1. You have {torch.__version__}"
                        )
                else:
                    eval_logger.info("Device not specified")
                    eval_logger.info(f"Cuda Available? {torch.cuda.is_available()}")
                    self._device = (
                        torch.device("cuda")
                        if torch.cuda.is_available()
                        else torch.device("cpu")
                    )
            else:  # Parallelism managed by accelerate
                if device != "cuda":
                    eval_logger.info(
                        f"Using `accelerate launch` or `parallelize=True`, device '{device}' will be overridden when placing model."
                    )
                # TODO: include in warning that `load_in_8bit` etc. affect this too
                self._device = (
                    self.accelerator.device
                    if hasattr(self, "accelerator")
                    else torch.device(device)
                )

            revision = str(revision)  # cast to string if not already one
            # TODO: update this to be less of a hack once subfolder is fixed in HF
            revision = revision + ("/" + subfolder if subfolder is not None else "")

            self._get_config(
                pretrained,
                revision=revision,
                trust_remote_code=trust_remote_code,
                gguf_file=gguf_file,
            )
            # determine which of 'causal' and 'seq2seq' backends to use for HF models
        self._get_backend(
            config=self.config, backend=backend, trust_remote_code=trust_remote_code
        )
        self._create_tokenizer(
            pretrained,
            tokenizer,
            revision=revision,
            trust_remote_code=trust_remote_code,
            use_fast_tokenizer=use_fast_tokenizer,
            gguf_file=gguf_file,
            add_bos_token=add_bos_token,
        )
        # load tokenizer so we know tokenizer vocabulary size before loading model and PEFT
        if isinstance(pretrained, str):
            self._create_model(
                pretrained=pretrained,
                revision=revision,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                parallelize=parallelize,
                gpus=gpus,
                max_memory_per_gpu=max_memory_per_gpu,
                max_cpu_memory=max_cpu_memory,
                offload_folder=offload_folder,
                peft=peft,
                delta=delta,
                autogptq=autogptq,
                gptqmodel=gptqmodel,
                gguf_file=gguf_file,
                **kwargs,
            )

        # access self._model through self.model property outside this method
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()
            self.model.tie_weights()

        self.truncation = truncation
        self.logits_cache = logits_cache
        self.vocab_size = self.tokenizer.vocab_size
        # select (or create) a pad token to use
        self.tokenizer = configure_pad_token(self.tokenizer, model_config=self.config)

        self.add_bos_token = add_bos_token
        if "gemma" in getattr(self.config, "model_type", ""):
            self.add_bos_token = True
            eval_logger.info(
                f"Model type is '{self.config.model_type}', part of the Gemma family--a BOS token will be used as Gemma underperforms without it."
            )

        self._max_length = max_length
        self.pretrained = pretrained
        self.delta = delta
        self.peft = peft
        self.revision = revision
        self.batch_schedule = 1
        self.batch_sizes = {}
        self.max_batch_size = max_batch_size

        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self.batch_size_per_gpu = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self.batch_size_per_gpu = int(batch_size)

        if isinstance(pretrained, str):
            if gpus >= 1 or str(self.device) == "mps":
                # TODO: can remove this whole snippet except in the mps case, perhaps?
                if not (parallelize or autogptq or hasattr(self, "accelerator")):
                    # place model onto device requested manually,
                    # if not using HF Accelerate or device_map
                    # or any other option that preloads model onto device
                    try:
                        self.model.to(self.device)
                    except ValueError:
                        eval_logger.debug(
                            "Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes` or `device_map` is provided. If the desired GPU is being used, this message is safe to ignore."
                        )
            # multigpu data-parallel support when launched with accelerate
            if gpus > 1:
                if accelerator.num_processes > 1:
                    if parallelize:
                        eval_logger.warning(
                            "You are both using a HF Accelerate `device_map` (`--model_args parallelize=True`) and launching via `accelerate launch`. This will attempt to do model and data parallelism depending on the resources available."
                        )
                    elif gpus > accelerator.num_processes:
                        eval_logger.warning(
                            "WARNING: The number of total system GPUs does not match the number of spawned processes. "
                            "If you would like to use data parallelism, please launch the script "
                            "with 'accelerate launch *script*'. "
                            f"Current run will proceed with {accelerator.num_processes} devices."
                        )
                        if self.accelerator.is_local_main_process:
                            eval_logger.info(
                                f"Using {gpus} devices with data parallelism"
                            )

                    self._device = torch.device(f"{accelerator.device}")
                    self.accelerator = accelerator
                    self._rank = self.accelerator.local_process_index
                    self._world_size = self.accelerator.num_processes
                else:
                    # if we aren't launching via accelerate, ditch
                    self._rank = 0
                    self._world_size = 1
        else:
            # if a PreTrainedModel was passed into HFLM, we forgo distributed setup.
            eval_logger.warning(
                "Passed an already-initialized model through `pretrained`, assuming single-process call to evaluate() or custom distributed integration"
            )
            self._rank = 0
            self._world_size = 1
        self.custom_prefix_token_id = prefix_token_id
        if prefix_token_id is not None:
            eval_logger.info(
                f"Loglikelihood prefix token id used in evaluation: {self.prefix_token_id}"
            )
        if is_feature_cache:
            dLLMCache.new_instance(**asdict(dLLMCacheConfig(
                    prompt_interval_steps=prompt_interval_steps,
                    gen_interval_steps=gen_interval_steps,
                    transfer_ratio=transfer_ratio,
                    cfg_interval_steps=cfg_interval_steps if is_cfg_cache else 1,
                )))
            register_cache_LLaDA(self.model,"model.transformer.blocks")
        else:
            dLLMCache.new_instance(**asdict(dLLMCacheConfig(
                    prompt_interval_steps=1,
                    gen_interval_steps=1,
                    transfer_ratio=0,
                    cfg_interval_steps=cfg_interval_steps if is_cfg_cache else 1,
                )))

        if self.rank == 0:
                print(f"Feature Cache is {is_feature_cache}.CFG Cache is {is_cfg_cache},prompt_interval_steps={prompt_interval_steps}, gen_interval_steps={gen_interval_steps}, cfg_interval_steps={cfg_interval_steps},transfer_ratio={transfer_ratio}")
    def _get_accelerate_args(
        self,
        parallelize: Optional[bool] = None,
        device_map: Optional[str] = "auto",
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
        gpus: Optional[int] = None,
    ) -> dict:
        """Returns the kwargs needed to apply `accelerate` in `AutoModel.from_pretrained`."""
        num_local_processes = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        num_machines = int(os.environ.get("WORLD_SIZE", 0)) // num_local_processes
        if (
            num_machines == 0
            and hasattr(self, "accelerator")
            and self.accelerator is not None
        ):
            eval_logger.info("We are not in a distributed setting for accelerate. Setting model_parallel to False.")
            parallelize = False

        if parallelize is None:
            # If parallelism is unset by the user, we automatically assign model parallelism
            # if enough extra GPUs are available
            max_memory_all_gpus = get_max_memory()
            # We just want gpu, not cpu, max memory
            if "cpu" in max_memory_all_gpus:
                del max_memory_all_gpus["cpu"]
            parallelize = bool(num_local_processes < len(max_memory_all_gpus))
            eval_logger.info(
                f"Setting model parallel to {parallelize} since "
                f"the number of local processes is {num_local_processes} "
                f"and the number of GPUs is {len(max_memory_all_gpus)}"
            )

        args = {}
        if parallelize:  # Model parallelism will be used
            max_memory = {}
            if max_memory_per_gpu is not None:  # Using the provided memory requirements
                max_memory_per_gpu_map = {
                    device_idx: max_memory_per_gpu for device_idx in range(gpus)
                }
            else:  # Estimating the possible memory requirements
                max_memory_all_gpus = get_max_memory()
                if "cpu" in max_memory_all_gpus:
                    del max_memory_all_gpus["cpu"]
                if not hasattr(self, "accelerator"):
                    max_memory_per_gpu_map = {
                        k: v for k, v in max_memory_all_gpus.items()
                    }
                else:
                    # use only 1 / num_processes of the GPUs if we are running under accelerate launch
                    max_memory_per_gpu_map = {
                        k: v
                        for k, v in max_memory_all_gpus.items()
                        if k % num_local_processes
                        == (self.accelerator.process_index % num_local_processes)
                    }
            args["max_memory"] = max_memory_per_gpu_map
            args["device_map"] = "auto" if device_map is None else device_map
            eval_logger.info(
                f"Model parallel was set to True, setting max memory per GPU to {max_memory_per_gpu_map} and device map to {args.get('device_map')}"
            )

            if max_cpu_memory is not None:
                max_memory["cpu"] = max_cpu_memory

            args["offload_folder"] = offload_folder
        elif (
            device_map is None
        ):  # No model parallelism, we use the default provided device for our model
            if hasattr(self, "accelerator"):
                device_map = {"": f"{self.accelerator.device}"}
            else:
                device_map = {"": str(self.device)}
            args["max_memory"] = None
            args["device_map"] = device_map
            eval_logger.info(
                f"Model parallel was set to False, max memory was not set, and device map was set to {device_map}"
            )
        else:
            args["max_memory"] = None
            args["device_map"] = None
            eval_logger.info("Model parallel was set to False.")

        return args

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self): 
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self):
        # it is used as prefix for loglikelihood
        if self.custom_prefix_token_id is not None:
            return self.custom_prefix_token_id
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        
        return self._device
    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def _get_backend(
        self,
        config: Union[transformers.PretrainedConfig, transformers.AutoConfig],
        backend: Literal["default", "causal", "seq2seq"] = "default",
        trust_remote_code: Optional[bool] = False,
    ) -> None:
        """
        Helper method during initialization.
        Determines the backend ("causal" (decoder-only) or "seq2seq" (encoder-decoder)) model type to be used.
        sets `self.AUTO_MODEL_CLASS` appropriately if not already set.

        **If not calling HFLM.__init__() or HFLM._get_backend() within a subclass of HFLM,
        user must set `self.backend` to be either "causal" or "seq2seq" manually!**
        """

        assert backend in ["default", "causal", "seq2seq"]

        if backend != "default":
            # if we've settled on non-default backend, use that manually
            if backend == "causal":
                self.backend = backend
            elif backend == "seq2seq":
                self.backend = backend
            eval_logger.info(
                f"Overrode HF model backend type, and using type '{self.backend}'"
            )
        else:
            # determine and use the default HF backend for this model, based on its config + metadata.
            if (
                getattr(config, "model_type")
                in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
            ):
                # first check if model type is listed under seq2seq models, since some
                # models like MBart are listed in both seq2seq and causal mistakenly in HF transformers.
                # these special cases should be treated as seq2seq models.
                self.backend = "seq2seq"
                eval_logger.debug(f"Using model type '{self.backend}'")
            elif (
                getattr(self.config, "model_type") in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
            ):
                self.backend = "causal"
                eval_logger.debug(f"Using model type '{self.backend}'")
            else:
                if not trust_remote_code:
                    eval_logger.warning(
                        "HF model type is neither marked as CausalLM or Seq2SeqLM. \
                    This is expected if your model requires `trust_remote_code=True` but may be an error otherwise."
                        "Setting backend to causal"
                    )
                # if model type is neither in HF transformers causal or seq2seq model registries
                # then we default to assuming AutoModelForCausalLM
                self.backend = "causal"
                eval_logger.info(
                    f"Model type cannot be determined. Using default model type '{self.backend}'"
                )

        if self.AUTO_MODEL_CLASS is None:
            if self.backend == "causal":
                self.AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM
            elif self.backend == "seq2seq":
                self.AUTO_MODEL_CLASS = transformers.AutoModelForSeq2SeqLM

    def _get_config(
        self,
        pretrained: str,
        revision: str = "main",
        trust_remote_code: bool = False,
        gguf_file: Optional[str] = None,
    ) -> None:
        """Return the model config for HuggingFace models"""

        # 检查是否是 LoRA 检查点目录
        import os
        import json
        adapter_config_path = os.path.join(pretrained, "adapter_config.json")
        is_lora_checkpoint = os.path.exists(adapter_config_path)

        if is_lora_checkpoint:
            # 对于 LoRA 检查点，从 adapter_config.json 获取基础模型路径
            eval_logger.info(f"🔧 Detected LoRA checkpoint, loading config from base model")
            try:
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                base_model_path = adapter_config.get("base_model_name_or_path", "GSAI-ML/LLaDA-8B-Instruct")
                eval_logger.info(f"📦 Using base model config from: {base_model_path}")

                # 从基础模型加载配置
                self._config = transformers.AutoConfig.from_pretrained(
                    base_model_path,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                    gguf_file=gguf_file,
                )
            except Exception as e:
                eval_logger.warning(f"⚠️  Failed to load config from LoRA checkpoint: {e}")
                eval_logger.info("🔄 Falling back to standard config loading...")
                # 回退到标准配置加载
                self._config = transformers.AutoConfig.from_pretrained(
                    pretrained,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                    gguf_file=gguf_file,
                )
        else:
            # 标准配置加载
            self._config = transformers.AutoConfig.from_pretrained(
                pretrained,
                revision=revision,
                trust_remote_code=trust_remote_code,
                gguf_file=gguf_file,
            )

    def _create_model(
        self,
        pretrained: str,
        revision: Optional[str] = "main",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        trust_remote_code: Optional[bool] = False,
        # arguments used for splitting a model across GPUs naively.
        # only used if `parallelize=True`.
        # (accelerate naive PP (device_map) options)
        parallelize: Optional[bool] = False,
        gpus: Optional[int] = None,
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
        # PEFT, delta weights and quantization options
        peft: Optional[str] = None,
        delta: Optional[str] = None,
        autogptq: Optional[Union[bool, str]] = False,
        gptqmodel: Optional[bool] = False,
        gguf_file: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initializes an HF or HF-compatible PreTrainedModel from scratch
        inside HFLM, using the kwargs passed into self.__init__().

        Also handles functionality such as AutoGPTQ usage and PEFT wrapping.

        For future similar extensions to AutoGPTQ that are not core to HF's ecosystem,
        (such as PyTorch models that are nearly, but not quite, fully mirroring
        HF's public interface relied on in this HFLM class)
        please consider subclassing HFLM and overriding this and other methods as needed.
        """
        if autogptq or gptqmodel:
            raise ValueError("'autogptq' and 'gptqmodel' are not supported yet")

        # 移除 PEFT 限制，现在支持 LoRA 检查点加载
        if delta:
            raise ValueError("'delta' weights are not supported yet")

        model_kwargs = kwargs if kwargs else {}

        model_kwargs.update(
            self._get_accelerate_args(
                parallelize=parallelize,
                device_map=kwargs.get("device_map", None),
                max_memory_per_gpu=max_memory_per_gpu,
                max_cpu_memory=max_cpu_memory,
                offload_folder=offload_folder,
                gpus=gpus,
            )
        )

        # 检查是否是 LoRA 检查点目录
        import os
        import json
        adapter_config_path = os.path.join(pretrained, "adapter_config.json")
        is_lora_checkpoint = os.path.exists(adapter_config_path)

        if is_lora_checkpoint:
            # 加载 LoRA 检查点
            eval_logger.info(f"🔧 Detected LoRA checkpoint at: {pretrained}")

            # 读取 adapter_config.json 获取基础模型路径
            try:
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                base_model_path = adapter_config.get("base_model_name_or_path", "GSAI-ML/LLaDA-8B-Instruct")
                eval_logger.info(f"📦 Base model path: {base_model_path}")
                eval_logger.info(f"🎯 LoRA config - rank: {adapter_config.get('r', 'N/A')}, alpha: {adapter_config.get('lora_alpha', 'N/A')}")

                # 先加载基础模型（激进内存优化）
                eval_logger.info("🚀 Loading base model with aggressive memory optimization...")

                # 清理 GPU 缓存
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    eval_logger.info(f"💾 GPU memory before loading: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

                # 激进的内存优化参数
                memory_optimized_kwargs = model_kwargs.copy()
                memory_optimized_kwargs.update({
                    "low_cpu_mem_usage": True,
                    "torch_dtype": get_dtype(dtype),
                    "device_map": "auto",
                    "max_memory": {0: "35GB"},  # 限制 GPU 0 使用 35GB，留出 12GB 缓冲
                    "offload_folder": "./temp_offload",  # 临时卸载目录
                    "offload_state_dict": True,
                })

                self._model = transformers.AutoModel.from_pretrained(
                    base_model_path,
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                    **memory_optimized_kwargs
                )

                if torch.cuda.is_available():
                    eval_logger.info(f"💾 GPU memory after base model: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

                # 然后加载 LoRA 适配器（内存优化）
                eval_logger.info("🔗 Loading LoRA adapter with memory optimization...")

                # 再次清理缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                from peft import PeftModel
                self._model = PeftModel.from_pretrained(
                    self._model,
                    pretrained,
                    torch_dtype=get_dtype(dtype),
                    is_trainable=False,
                    adapter_name="default"  # 明确指定适配器名称
                )

                if torch.cuda.is_available():
                    eval_logger.info(f"💾 GPU memory after LoRA: {torch.cuda.memory_allocated()/1024**3:.2f}GB")

                eval_logger.info("✅ LoRA adapter loaded successfully")

            except Exception as e:
                eval_logger.error(f"❌ Failed to load LoRA checkpoint: {e}")
                eval_logger.info("🔄 Falling back to standard model loading...")
                # 回退到标准加载
                self._model = transformers.AutoModel.from_pretrained(
                    pretrained,
                    revision=revision,
                    torch_dtype=get_dtype(dtype),
                    trust_remote_code=trust_remote_code,
                    **model_kwargs
                )

        elif peft:
            # 处理通过 peft 参数指定的 LoRA 路径
            eval_logger.info(f"🔧 Loading base model and applying PEFT from: {peft}")
            self._model = transformers.AutoModel.from_pretrained(
                pretrained,
                revision=revision,
                torch_dtype=get_dtype(dtype),
                trust_remote_code=trust_remote_code,
                **model_kwargs
            )
            from peft import PeftModel
            self._model = PeftModel.from_pretrained(
                self._model,
                peft,
                torch_dtype=get_dtype(dtype),
                is_trainable=False
            )
            eval_logger.info("✅ PEFT adapter loaded successfully")

        else:
            # 标准模型加载（保持向后兼容）
            eval_logger.info(f"📦 Loading standard model from: {pretrained}")
            self._model = transformers.AutoModel.from_pretrained(
                pretrained,
                revision=revision,
                torch_dtype=get_dtype(dtype),
                trust_remote_code=trust_remote_code,
                **model_kwargs
            )

        self._model = self._model.to(self.device).eval()
        eval_logger.info(f"🎯 Model loaded and moved to device: {self.device}")
            


    def _create_tokenizer(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        tokenizer: Optional[
            Union[
                str,
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
            ]
        ],
        revision: Optional[str] = "main",
        trust_remote_code: Optional[bool] = False,
        use_fast_tokenizer: Optional[bool] = True,
        gguf_file: Optional[str] = None,
        add_bos_token: Optional[bool] = False,
    ) -> None:
        """
        Helper method during initialization.

        Create a tokenizer object corresponding to the correct
        tokenizer for value of `pretrained`, or use the pre-initialized tokenizer passed.
        """
        kwargs = {
            "revision": revision,
            "trust_remote_code": trust_remote_code,
        }

        # gguf format embeds tokenizer and is not compatible with hf tokenizer `use_fast` param
        if gguf_file is not None:
            kwargs["gguf_file"] = gguf_file
        else:
            kwargs["use_fast"] = use_fast_tokenizer

        if add_bos_token:
            kwargs["add_bos_token"] = True

        if tokenizer:
            if isinstance(tokenizer, str):
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    tokenizer, **kwargs
                )
            else:
                assert isinstance(
                    tokenizer, transformers.PreTrainedTokenizer
                ) or isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
                self.tokenizer = tokenizer
        else:
            # Get tokenizer based on 'pretrained'
            if isinstance(pretrained, str):
                model_name = pretrained

                # 检查是否是 LoRA 检查点目录
                import os
                import json
                adapter_config_path = os.path.join(pretrained, "adapter_config.json")
                is_lora_checkpoint = os.path.exists(adapter_config_path)

                if is_lora_checkpoint:
                    # 对于 LoRA 检查点，检查是否有分词器文件
                    tokenizer_files = ["tokenizer.json", "tokenizer_config.json"]
                    has_tokenizer = any(os.path.exists(os.path.join(pretrained, f)) for f in tokenizer_files)

                    if has_tokenizer:
                        eval_logger.info(f"🔤 Loading tokenizer from LoRA checkpoint: {pretrained}")
                        model_name = pretrained
                    else:
                        # 从基础模型加载分词器
                        try:
                            with open(adapter_config_path, 'r') as f:
                                adapter_config = json.load(f)
                            base_model_path = adapter_config.get("base_model_name_or_path", "GSAI-ML/LLaDA-8B-Instruct")
                            eval_logger.info(f"🔤 Loading tokenizer from base model: {base_model_path}")
                            model_name = base_model_path
                        except Exception as e:
                            eval_logger.warning(f"⚠️  Failed to get base model path: {e}")
                            model_name = pretrained
            else:
                # get the HF hub name via accessor on model
                model_name = self.model.name_or_path

            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name, **kwargs
            )
        return None


    def tok_encode(
        self, string: str, left_truncate_len=None, add_special_tokens=None
    ) -> List[int]:
        """ """
        # default for None - empty dict, use predefined tokenizer param
        # used for all models except for CausalLM or predefined value
        special_tokens_kwargs = {}

        # by default for CausalLM - false or self.add_bos_token is set
        if add_special_tokens is None:
            if self.backend == "causal":
                special_tokens_kwargs = {
                    "add_special_tokens": False or self.add_bos_token
                }
        # otherwise the method explicitly defines the value
        else:
            special_tokens_kwargs = {"add_special_tokens": add_special_tokens}

        encoding = self.tokenizer.encode(string, **special_tokens_kwargs)

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_batch_encode(
        self,
        strings: List[str],
        padding_side: str = "left",
        left_truncate_len: int = None,
        truncation: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # encode a batch of strings. converts to tensors and pads automatically, unlike tok_encode.
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side

        add_special_tokens = {}
        if self.backend == "causal":
            add_special_tokens = {"add_special_tokens": False or self.add_bos_token}

        encoding = self.tokenizer(
            strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            **add_special_tokens,
        )
        if left_truncate_len:
            original_lengths = encoding["input_ids"].size(1)
            if original_lengths > left_truncate_len:
                eval_logger.warn(
                    f"Left truncation applied. Original sequence length was {original_lengths}, "
                    f"truncating to last {left_truncate_len} tokens. Some content will be lost.",
                )
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][
                :, -left_truncate_len:
            ]
        self.tokenizer.padding_side = old_padding_side

        return encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _model_call(self, inps, attn_mask=None, labels=None):
        """
        :param inps: torch.Tensor
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)] or of shape
            [batch, sequence_ctx]. the size of sequence may vary from call to call
        :param attn_mask: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :param labels: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :return
            A torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model's decoder
        """
        with torch.no_grad():
            if attn_mask is not None or labels is not None:
                assert attn_mask is not None and labels is not None
                assert self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM
                return self.model(
                    input_ids=inps, attention_mask=attn_mask, labels=labels
                ).logits
            else:
                assert self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM
                return self.model(inps).logits

    def _loglikelihood_tokens(self, requests, **kwargs) -> List[Tuple[float, bool]]:
        raise NotImplementedError
    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        raise NotImplementedError


    def loglikelihood(self, requests):
        raise NotImplementedError
    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        req = []
        bar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Running generate_until requests")
        ds = [{"text": req.args[0]} for req in requests]
        ds = Dataset.from_list(ds)
        gen_kwargs = requests[0].args[1]

        # 检查是否启用动态长度生成
        enable_dynamic = gen_kwargs.get("enable_dynamic_length", False)

        if enable_dynamic and not DYNAMIC_GENERATION_AVAILABLE:
            eval_logger.warning("Dynamic length generation requested but module not available. Falling back to fixed length generation.")
            enable_dynamic = False

        if enable_dynamic:
            eval_logger.info("Using dynamic length generation")

        for batch in ds.iter(self.batch_size):
            contexts = batch["text"]
            if self.add_bos_token:
                contexts = [self.tokenizer.bos_token + p for p in contexts]
            context_enc, attn_masks = self.tok_batch_encode(
                contexts,
                truncation=self.truncation,
            )

            # 选择生成方法
            if enable_dynamic:
                # 为动态长度生成添加必要的默认参数
                dynamic_gen_kwargs = self._prepare_dynamic_gen_kwargs(gen_kwargs)

                # 使用动态长度生成
                out = self._generate_dynamic_length_batch(
                    input_ids=context_enc,
                    attention_mask=attn_masks,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    gen_kwargs=dynamic_gen_kwargs
                )
                # 动态长度生成返回的是tensor列表，需要转换为适合batch_decode的格式
                if isinstance(out, list) and len(out) > 0:
                    # 由于每个样本的生成长度可能不同，需要分别处理
                    # 将每个tensor转换为列表，然后组合成一个列表
                    out_list = []
                    for tensor in out:
                        # tensor形状是[1, gen_length]，转换为[gen_length]
                        out_list.append(tensor.squeeze(0))
                    out = out_list
            else:
                # 使用原有固定长度生成
                out = generate(
                    input_ids=context_enc,
                    attention_mask=attn_masks,
                    model=self.model,
                    steps=gen_kwargs.get("steps"),
                    gen_length=gen_kwargs.get("gen_length"),
                    block_length=gen_kwargs.get("block_length"),
                    cfg_scale=gen_kwargs.get("cfg_scale"),
                    remasking=gen_kwargs.get("remasking",None) if gen_kwargs.get("remasking",None) else "low_confidence"
                )

            cont_toks_list = self.tokenizer.batch_decode(out, skip_special_tokens=True)
            for s in cont_toks_list:
                if not self.escape_until:
                    for term in gen_kwargs.get("until"):
                        if len(term) > 0:
                            s = s.split(term)[0]
                res.append(s)
                bar.update(1)
            req.append(contexts)
        bar.close()
        return res

    def _prepare_dynamic_gen_kwargs(self, gen_kwargs: dict) -> dict:
        """
        为动态长度生成准备参数，添加必要的默认值

        Args:
            gen_kwargs: 原始生成参数

        Returns:
            dict: 包含动态长度生成所需参数的字典
        """
        dynamic_kwargs = gen_kwargs.copy()

        # 添加动态长度生成的必需参数默认值
        if "initial_length" not in dynamic_kwargs:
            dynamic_kwargs["initial_length"] = 64

        if "expansion_steps" not in dynamic_kwargs:
            dynamic_kwargs["expansion_steps"] = [64, 128, 256, 512, 1024, 2048]

        if "max_expansions" not in dynamic_kwargs:
            dynamic_kwargs["max_expansions"] = 5

        # 确保基本参数有默认值
        if "steps" not in dynamic_kwargs:
            dynamic_kwargs["steps"] = 128

        if "block_length" not in dynamic_kwargs:
            dynamic_kwargs["block_length"] = 64

        if "cfg_scale" not in dynamic_kwargs:
            dynamic_kwargs["cfg_scale"] = 0.0

        if "remasking" not in dynamic_kwargs:
            dynamic_kwargs["remasking"] = "low_confidence"

        if "mask_id" not in dynamic_kwargs:
            dynamic_kwargs["mask_id"] = 126336

        eval_logger.debug(f"Dynamic generation parameters: {dynamic_kwargs}")
        return dynamic_kwargs

    def _generate_dynamic_length_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        model,
        tokenizer,
        gen_kwargs: dict
    ) -> torch.Tensor:
        """
        使用动态长度生成处理批次数据

        Args:
            input_ids: 输入token序列 [batch_size, prompt_length]
            attention_mask: 注意力掩码
            model: 生成模型
            tokenizer: 分词器
            gen_kwargs: 生成参数字典

        Returns:
            torch.Tensor: 生成的token序列 [batch_size, actual_gen_length]
        """
        batch_size = input_ids.shape[0]
        eval_logger.info(f"Processing batch of size {batch_size} with dynamic length generation")
        results = []
        try:
            for i in range(batch_size):
                single_input = input_ids[i:i+1]  # 保持维度 [1, prompt_length]
                single_mask = attention_mask[i:i+1] if attention_mask is not None else torch.ones_like(single_input)
                single_result = generate_dynamic_length(
                    input_ids=single_input,
                    attention_mask=single_mask,
                    model=model,
                    tokenizer=tokenizer,
                    gen_kwargs=gen_kwargs
                )
                results.append(single_result)
        except Exception as e:
            raise RuntimeError(f"Dynamic length generation failed: {str(e)}") from e

        
        return results

    def apply_chat_template(
        self, chat_history: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        try:
            chat_templated = self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
            )
        except jinja2.exceptions.TemplateError:
            eval_logger.warning(
                "Failed to apply chat template. removing the system role in chat history."
            )
            chat_history = [msg for msg in chat_history if msg["role"] != "system"]
            chat_templated = self.tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
            )

        return chat_templated
