import asyncio
import logging
import os
import warnings
from functools import partial

import torch
import torch.distributed
import verl.utils.torch_functional as verl_F
from omegaconf import DictConfig
from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.debug import DistProfiler, log_gpu_memory_usage, simple_timer
from verl.utils.debug.performance import reduce_timing
from verl.utils.device import get_device_id, get_device_name, get_torch_device
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
)
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker, RewardModelWorker, get_sharding_strategy
from vllm import SamplingParams


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

class MARCHActorFactCheckWorker(ActorRolloutRefWorker):
    def _build_model_optimizer(self, *args, **kwargs):
        results = super()._build_model_optimizer(*args, **kwargs)

        if self.config.model.custom_chat_template == "quark_chat":
            from verl.utils import hf_tokenizer
            from verl.utils.fs import copy_to_local
            
            use_shm = self.config.model.get("use_shm", False)
            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            temp_tokenizer = hf_tokenizer(local_path, trust_remote_code=self.config.model.get("trust_remote_code", False))
            original_chat_template = getattr(temp_tokenizer, 'chat_template', None)
            
            if self.processor is not None:
                self.processor.chat_template = original_chat_template
            else:
                self.tokenizer.chat_template = original_chat_template

        return results

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def _update_rollout_config(self, params):
        """
        Update the rollout config with the given parameters.
        """
        old_params = {}
        if hasattr(self.rollout, "sampling_params"):
            if isinstance(self.rollout.sampling_params, SamplingParams):
                for k, v in params.items():
                    old_params[k] = getattr(self.rollout.sampling_params, k, None)
                    setattr(self.rollout.sampling_params, k, v)
            else:
                for k, v in params.items():
                    old_params[k] = self.rollout.sampling_params.get(k)
                    if v is None:
                        self.rollout.sampling_params.pop(k)
                    else:
                        self.rollout.sampling_params[k] = v
        elif hasattr(self.rollout, "config"):
            for k, v in params.items():
                old_params[k] = self.rollout.config.get(k)
                if v in None:
                    self.rollout.config.pop(k)
                else:
                    self.rollout.config[k] = v
        else:
            old_params = params
        return old_params

