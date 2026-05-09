import os
from collections import defaultdict
from pprint import pprint
from copy import deepcopy
import uuid
import re
import tempfile
from typing import Dict, List

import numpy as np
from tqdm import tqdm
import torch
import ray
import pandas as pd
import json

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.metric_utils import process_validation_metrics
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.ray_trainer import compute_response_mask, apply_kl_penalty, compute_advantage
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.utils.debug import marked_timer
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.metric import reduce_metrics
from quarl.utils.data_utils import extract_thought_and_answer, extract_docs_and_query


question_prompt = """You are a data annotation expert. Your task is to analyze a model's response to a user query. For every critical number found in the model's response, you must create a question where the answer is precisely and exactly that number.

Rule 1: Clarity and Specificity of Questions:
Each question must be clear, complete, and unambiguous. It must contain enough context from the response to ensure the number is the only possible correct answer.
For example, if the model's response contains "In 2025, 50 people will take the bar exam in Beijing," a good question would be "How many people will take the bar exam in the Beijing area in 2025?", and the answer is 50.
A bad example would be "How many people will take the bar exam in 2025?". This is ambiguous because it lacks the specific region ('Beijing area').

Rule 2: Format of the Answer:
The answer must be a pure number, either an integer or a decimal. It must NOT include any other characters or formats, such as percentage signs (%), 'k' to denote thousands (e.g., 10k), ranges (e.g., 10-20), or words (e.g., fifty).

Please provide your output as an unordered list using the following format for each question-answer pair:

- Question: xxx [Answer: n]
- Question: xxx [Answer: n]

Now, generate questions for the user query and model response below. You need to generate at least three question-answer pairs, and up to ten. Do not output any other content or explanations.

User Query: {query}
Model Response: {response}
"""

answer_question_prompt = """You are a data validation expert. You will be given a set of reference materials and a list of user questions. Your task is to retrieve information from the reference materials to answer all questions. Before providing each answer, you must state the evidence for it.
You must adhere strictly to the content of the reference materials and must not fabricate or infer any information that is not explicitly mentioned. If the relevant information cannot be found in the materials, you must state "Cannot answer".
It is important to note that all answers are pure numbers. Therefore, your answer must also be a pure number and must not contain any other content, such as percentage signs (%), 'k' to denote thousands, ranges, or numbers written as words.
Here is an example:
Reference Materials:
Document 1: ...in 2024, 50 people will take the bar exam in Beijing. \nDocument 2: ...(omitted)

- Question 1: How many people will take the bar exam in Beijing in 2024?
- Question 2: How many people will take the bar exam in Beijing in 2025?

Example Response:

1. Evidence: Document 1 states that 50 people will take the bar exam in Beijing in 2024.
[Answer: 50]
2. Evidence: The materials do not contain information about the bar exam in Beijing for 2025, therefore I cannot answer.
[Answer: Cannot answer]

Now, based on the reference materials below, answer all the following questions. Strictly adhere to the example format provided.

Reference Materials:
{docs}

Questions:
{questions}
"""

class MARCHFactCheckTrainer(RayPPOTrainer):

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.trainer.profile_steps
            if self.config.trainer.profile_steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.trainer.profile_continuous_steps
                        else curr_step_profile
                    )

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "interaction_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                if "index" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("index")
                if "agent_name" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("agent_name")

                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )
                
                if "reward_model" in batch.non_tensor_batch:
                    gen_batch.non_tensor_batch["reward_model"] = deepcopy(batch.non_tensor_batch["reward_model"])

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):

                    # ===== 1. rollout: 生成 response =====
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # ===== 2. fact-check: 给 response 打分，并构造 checker_batch (T=0.5) =====

                    with marked_timer("gen_fact_check", timing_raw, color="orange"):
                        fact_check_results, fact_check_reward_tensor, fact_check_reward_extra_info, checker_batch = \
                            self.generate_fact_check_batch(batch, timing_raw)

                    # ===== 3. reward：只对 response batch 算 RM + 自定义 reward =====
                    with marked_timer("reward", timing_raw, color="yellow"):
                        reward_extra_infos_dict = {}
                        
                        if self.use_rm:
                            rm_reward_tensor = self.rm_wg.compute_rm_score(batch)
                            # print("RM dp keys:", rm_reward_tensor.batch.keys())
                            rm_reward_tensor = rm_reward_tensor.batch["rm_scores"]
                        else:
                            rm_reward_tensor = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            rule_reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                        if self.config.reward_model.launch_reward_fn_async:
                            # 等待异步 reward
                            rule_reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                                            
                        baseline_type = self.config.get("rlhf_baseline", 'rlhf')
                        print(f"[Baseline Type:] {baseline_type}")
                        if baseline_type == 'rlhf':
                            response_reward_tensor = rm_reward_tensor
                            batch.batch["token_level_scores"] = response_reward_tensor
                        else:
                            response_reward_tensor = rm_reward_tensor + fact_check_reward_tensor
                            batch.batch["token_level_scores"] = response_reward_tensor
                            
                        print(f"[RM Reward Tensor]: {rm_reward_tensor}")
                        print(f"[Rule Reward Tensor]: {rule_reward_tensor}")
                        print(f"[Fact Check Reward Tensor]: {fact_check_reward_tensor}")
                        print(f"[Response Reward Tensor]: {response_reward_tensor}")
                    
                    # ===== 4. 构建 big_batch：response + checker，一次性完成 =====
                    use_checker_ppo = self.config.get("use_checker_ppo", True)

                    with marked_timer("build_big_batch_and_logprob", timing_raw, color="blue"):
                        if len(checker_batch) > 0 and use_checker_ppo:
                            if "response_mask" not in checker_batch.batch:
                                checker_batch.batch["response_mask"] = compute_response_mask(checker_batch)
                            if "token_level_scores" not in checker_batch.batch:
                                checker_batch.batch["token_level_scores"] = torch.zeros_like(
                                    checker_batch.batch["responses"], dtype=torch.float32
                                )
                            big_batch = self._concat_dataprotos(batch, checker_batch)
                            print("big_batch size:", big_batch.batch["prompts"].shape[0],
                                "uid size:", len(big_batch.non_tensor_batch.get("uid", [])))
                        else:
                            # 干净版：只用 response 部分做 PPO
                            # print("Only Response Reward, big_batch size:", batch.batch["prompts"].shape[0])
                            big_batch = batch

                        # 计算 old_log_probs
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(big_batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = big_batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(
                            loss_mat=entropys,
                            loss_mask=response_masks,
                            loss_agg_mode=loss_agg_mode
                        )
                        metrics.update({"actor/entropy": entropy_agg.detach().item()})
                        old_log_prob.batch.pop("entropys")
                        big_batch = big_batch.union(old_log_prob)

                        # rollout_log_probs diff 监控
                        if "rollout_log_probs" in big_batch.batch.keys():
                            rollout_old_log_probs = big_batch.batch["rollout_log_probs"]
                            actor_old_log_probs = big_batch.batch["old_log_probs"]
                            attention_mask = big_batch.batch["attention_mask"]
                            responses = big_batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff.max().detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff.mean().detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff.std().detach().item(),
                                }
                            )

                    # ===== 5. reference log prob、value =====
                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(big_batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(big_batch)
                            big_batch = big_batch.union(ref_log_prob)

                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(big_batch)
                            big_batch = big_batch.union(values)

                    # ===== 6. advantage / PPO loss =====
                    with marked_timer("adv", timing_raw, color="brown"):

                        # 合并 reward_extra_info（response + fact-check）（checker 自己的 info 如果有也可并入）
                        reward_extra_infos_dict.update(fact_check_reward_extra_info)
                        if reward_extra_infos_dict:
                            big_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # 把 token_level_scores 作为 reward，其上已经包含 response + checker 的 reward
                        if self.config.algorithm.use_kl_in_reward:
                            big_batch, kl_metrics = apply_kl_penalty(
                                big_batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            big_batch.batch["token_level_rewards"] = big_batch.batch["token_level_scores"]

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        big_batch = compute_advantage(
                            big_batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                        # ===== fact-check 统计 =====
                        fc_total = np.array(fact_check_reward_extra_info["fact_check_sp_total_num"], dtype=float)
                        fc_fault = np.array(fact_check_reward_extra_info["fact_check_sp_fault_num"], dtype=float)
                        fc_true  = np.array(fact_check_reward_extra_info["fact_check_sp_true_num"], dtype=float)
                        fc_score = np.array(fact_check_reward_extra_info["fact_check_sp_score"], dtype=float)

                        mask = fc_total > 0
                        if mask.any():
                            metrics["fact_check_sp/total_num_mean"] = float(fc_total[mask].mean())
                            metrics["fact_check_sp/fault_num_mean"] = float(fc_fault[mask].mean())
                            metrics["fact_check_sp/true_num_mean"]  = float(fc_true[mask].mean())
                            metrics["fact_check_sp/score_mean"]     = float(fc_score[mask].mean())
                            metrics["fact_check_sp/score_std"]     = float(fc_score[mask].std()) if fc_score[mask].size > 1 else 0.0
                            metrics["fact_check_sp/has_question_rate"] = float((fc_total > 0).mean())
                            acc = np.zeros_like(fc_total)
                            acc[mask] = fc_true[mask] / fc_total[mask]
                            metrics["fact_check_sp/acc_mean"] = float(acc[mask].mean())

                        else:
                            metrics["fact_check_sp/total_num_mean"] = 0.0
                            metrics["fact_check_sp/fault_num_mean"] = 0.0
                            metrics["fact_check_sp/true_num_mean"]  = 0.0
                            metrics["fact_check_sp/score_mean"]     = 0.0
                            metrics["fact_check_sp/has_question_rate"] = 0.0
                            metrics["fact_check_sp/acc_mean"] = 0.0

                        # ===== checker reward 统计 =====
                        if len(checker_batch) > 0:
                            checker_scores = checker_batch.batch["token_level_scores"].sum(dim=-1).detach().cpu().numpy().astype(float)
                            metrics["checker/sp_score_mean"] = float(checker_scores.mean())
                            metrics["checker/sp_score_std"]  = float(checker_scores.std()) if checker_scores.size > 1 else 0.0
                        else:
                            metrics["checker/sp_score_mean"] = 0.0
                            metrics["checker/sp_score_std"]  = 0.0

                   # ===== 7. update critic / actor =====
                    # 在传给 critic / actor 之前，清理 big_batch.non_tensor_batch 中长度不等于 batch_size 的字段
                    bs = big_batch.batch["prompts"].shape[0]
                    clean_non_tensor = {}
                    for k, v in big_batch.non_tensor_batch.items():
                        try:
                            arr = np.array(v)
                            if arr.shape[0] != bs:
                                print(f"[cleanup] drop non_tensor key {k}: len={arr.shape[0]}, batch_size={bs}")
                                continue
                            clean_non_tensor[k] = arr
                        except Exception:
                            # 无法 np.array 的（比如标量或别的类型），也直接丢掉
                            print(f"[cleanup] drop non_tensor key {k}: type={type(v)}")
                            continue
                    big_batch.non_tensor_batch = clean_non_tensor

                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(big_batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            big_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(big_batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # ===== 8. dump / validate / ckpt 后面的逻辑按你原来的继续 =====
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            # 这里仍旧只 dump response 部分
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate、save ckpt、metrics 等保留原逻辑
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                # 下面 stop_profile / metrics / logger / step++ 等完全保持不变
                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.trainer.profile_steps
                        if self.config.trainer.profile_steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.trainer.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )

                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=batch)

    def _prepare_data(self, msgs, idx, is_padding=False):
        return {
            "prompt": msgs,
            "answer": "no_answer",
            "label": "no_label",
            "data_source": "fact_check",
            "ability": "general",
            "reward_model": {"ground_truth": "no_answer", "style": "model"},
            "extra_info": {
                'index': idx,
                'split': 'train',
                'is_padding': is_padding,
            }
        }
        
    def _prepare_fact_check_data(self, data):
        df = pd.DataFrame(data)

        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            temp_file = tmp.name
            df.to_parquet(temp_file, index=False)
        
        try:
            from verl.utils.dataset.rl_dataset import RLHFDataset

            temp_config = deepcopy(self.config.data)
            temp_config.filter_overlong_prompts = False

            temp_dataset = RLHFDataset(
                data_files=[temp_file], tokenizer=self.tokenizer, config=temp_config, processor=self.processor
            )

            from torchdata.stateful_dataloader import StatefulDataLoader
            from verl.utils.dataset.rl_dataset import collate_fn

            temp_dataloader = StatefulDataLoader(
                dataset=temp_dataset,
                batch_size=len(data),
                num_workers=0,
                drop_last=False,
                collate_fn=collate_fn,
                shuffle=False,
            )

            batch_dict = next(iter(temp_dataloader))
            batch = DataProto.from_single_dict(batch_dict)

            return batch
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def _extract_qa_pairs(self, text):
        """提取问题和"""
        pattern = r"-\s*Question:\s*(.*?)\s*\[Answer:\s*(.*?)\]"
        matches = re.findall(pattern, text)

        qa_pairs = []
        for question, answer in matches:
            qa_pairs.append({"question": question.strip(), "answer": answer.strip()})

        return qa_pairs
    
    def _extract_detailed_info(self, text):
        """提取详细信息包括序号、依据和答案"""
        pattern = r"(\d+)\.\s*Evidence:\s*(.*?)\s*\[Answer:\s*(.*?)\]"
        matches = re.findall(pattern, text, re.DOTALL)

        result = {}
        
        print(matches)
        for index, match in enumerate(matches):
            number, basis, answer = match[0], match[-2], match[-1]
            result[int(number)] = {"number": int(number), "basis": basis.strip(), "answer": answer.strip()}

        return result

    def _check_ans_eq(self, ans1, ans2):
        """比较答案是否相等，忽略格式差异"""
        # 清理特殊字符
        for char in ['%', 'k', 'K', 'w', 'W', ',']:
            ans1 = ans1.replace(char, "")
            ans2 = ans2.replace(char, "")
        # 有些case是这样的，一个字符串"5.18, 6.30"
        # 直接判断是否完全相等
        if ans1 == ans2:
            return True
        
        # 转换为数字
        try:
            num1 = float(ans1) if "." in ans1 else int(ans1)
        except (ValueError, AttributeError):
            return False
        
        try:
            num2 = float(ans2) if "." in ans2 else int(ans2)
        except (ValueError, AttributeError):
            return False
        
        # 比较
        if isinstance(num1, int) and isinstance(num2, int):
            return num1 == num2
        elif isinstance(num1, float) and isinstance(num2, float):
            return abs(num1 - num2) < 0.1
        else:
            # 混合类型比较
            return abs(float(num1) - float(num2)) < 0.1
        
    @torch.no_grad()
    def generate_fact_check_batch(self, gen_batch_output: DataProto, timing_raw: Dict):
        """
        Generate a fact check batch based on the generated responses.
        同时构造一个 checker_batch（只含温度 0.5 的 QA 回答轨迹），用于 PPO。
        """
        results = [None] * len(gen_batch_output)
        rule_reward_tensor = torch.zeros_like(gen_batch_output.batch["responses"], dtype=torch.float32)
        rewards_show = [0.] * rule_reward_tensor.shape[0]
        reward_extra_info = {
            "fact_check_sp_total_num": [0] * rule_reward_tensor.shape[0],
            "fact_check_sp_fault_num": [0] * rule_reward_tensor.shape[0],
            "fact_check_sp_true_num": [0] * rule_reward_tensor.shape[0],
            "fact_check_sp_score": [0] * rule_reward_tensor.shape[0],
            "fact_check_sp_bad_format_flag": [False] * rule_reward_tensor.shape[0],
            "fact_check_sp_bad_request_flag": [False] * rule_reward_tensor.shape[0],
        }
        cases = []
        questions = []
        n_completed = 0
        n_not_label, n_bad_format = 0, 0

        # 1) 从 gen_batch_output 抽出 response / docs / query，准备自出题
        for batch_idx, data in enumerate(gen_batch_output):
            data_label = data.non_tensor_batch.get("extra_info", {}).get("label", "unknown")
            
            prompt_ids = data.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data.batch["responses"]
            valid_response_length = data.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            docs, query = extract_docs_and_query(prompt_str, data.non_tensor_batch.get("raw_prompt", []))

            think_and_answer = extract_thought_and_answer(response_str)
            answer = think_and_answer[1] if think_and_answer else response_str

            cases.append({
                "response": answer,
                "docs": docs,
                "query": query,
                "valid_response_length": int(valid_response_length),
                "batch_idx": batch_idx,
            })

            is_padding, no_label = True, 0.
            if data_label not in self.config.get("custom_rewards_fact_check_sp_labels", ["rag_for_digit_fact", "gaokao_fact"]):
                no_label = 1.
                n_not_label += 1
            elif docs is not None and query is not None and answer is not None:
                is_padding = False
            
            if is_padding:
                # 这条样本不参与 fact-check，自定义一个惩罚
                score = (no_label - 1.0) * 1.5
                results[batch_idx] = {
                    "total_num": 1.0 - no_label,
                    "fault_num": 1.0 - no_label,
                    "true_num": 0.,
                    "score": score,
                    "bad_format_flag": no_label == 0.,
                    "bad_request_flag": False,
                    "idx": batch_idx,
                }
                rule_reward_tensor[batch_idx, cases[batch_idx]['valid_response_length'] - 1] = score
                rewards_show[batch_idx] = score
                for key, val in results[batch_idx].items():
                    if key == 'idx':
                        continue
                    reward_extra_info[f"fact_check_sp_{key}"][batch_idx] = val
                n_completed += 1
                n_bad_format += 1 - int(no_label)

            # 为自出题准备 prompt
            msgs = [{"role": "user", "content": question_prompt.format(query=query, response=answer)}]
            question_prompt_data = self._prepare_data(msgs, batch_idx, is_padding)
            
            questions.append(question_prompt_data)
        
        if n_completed >= len(gen_batch_output):
            print(f"[FactCheckSelfPlay] All samples in the batch are no label or bad format, n_not_label: {n_not_label}, n_bad_format: {n_bad_format}, n_completed: {n_completed}!")
            # 没有 checker batch，返回一个空 DataProto
            empty_checker_batch = DataProto.from_single_dict(
                {
                    "prompts": torch.empty(0, 1, dtype=torch.long),
                    "responses": torch.empty(0, 1, dtype=torch.long),
                    "attention_mask": torch.empty(0, 1, dtype=torch.long),
                }
            )
            empty_checker_batch.non_tensor_batch = {}
            empty_checker_batch.meta_info = {}
            
            return results, rule_reward_tensor, reward_extra_info, empty_checker_batch
        
        # 2) 自出题：从回答中抽数字问题
        question_batch = self._prepare_fact_check_data(questions)
        if not self.async_rollout_mode:
            question_batch_output = self.actor_rollout_wg.generate_sequences(question_batch)
        else:
            question_batch_output = self.async_rollout_manager.generate_sequences(question_batch)
        
        questions, q_a_cases = [], []
        n_no_qa_pairs = 0
        for idx, data in enumerate(question_batch_output):
            if question_batch[idx].non_tensor_batch['extra_info']['is_padding']:
                question_extract_summary = None
            else:
                prompt_ids = data.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]

                valid_prompt_length = data.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data.batch["responses"]
                valid_response_length = data.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

                question_extract_think_and_summary = extract_thought_and_answer(response_str)
                question_extract_summary = question_extract_think_and_summary[1] if question_extract_think_and_summary else response_str
            
            if question_extract_summary is not None:
                q_a_pairs = self._extract_qa_pairs(question_extract_summary)
            else:
                q_a_pairs = []
            
            q_a_cases.append({
                "response": cases[idx]["response"],
                "query": cases[idx]["query"],
                "batch_idx": cases[idx]["batch_idx"],
                "q_a_pairs": q_a_pairs,
            })

            if q_a_pairs:
                is_padding = False
            else:
                is_padding = True
                # 没有可检验的问题，score 设为 0
                if results[cases[idx]["batch_idx"]] is None:
                    results[cases[idx]["batch_idx"]] = {
                        "total_num": 0,
                        "fault_num": 0,
                        "true_num": 0,
                        "score": 0.0,
                        "bad_format_flag": False,
                        "bad_request_flag": False,
                        "idx": cases[idx]["batch_idx"],
                    }
                    rule_reward_tensor[cases[idx]["batch_idx"], cases[idx]['valid_response_length'] - 1] = 0.0
                    rewards_show[cases[idx]["batch_idx"]] = 0.0
                    for key, val in results[cases[idx]["batch_idx"]].items():
                        if key == 'idx':
                            continue
                        reward_extra_info[f"fact_check_sp_{key}"][cases[idx]["batch_idx"]] = val
                    n_completed += 1
                    n_no_qa_pairs += 1

            # 构造给 checker 的问题列表
            question = "\n".join([f"- Question {i + 1}:{q['question']}" for i, q in enumerate(q_a_pairs)])
            msgs = [{"role": "user", "content": answer_question_prompt.format(docs=cases[idx]['docs'], questions=question)}]
            question_prompt_data = self._prepare_data(msgs, cases[idx]["batch_idx"], is_padding)
            questions.append(question_prompt_data)
        
        if n_completed >= len(gen_batch_output):
            print(f"[FactCheckSelfPlay] All samples in the batch are no label, bad format or no qa pairs, n_not_label: {n_not_label}, n_bad_format: {n_bad_format}, n_no_qa_pairs: {n_no_qa_pairs}, n_completed: {n_completed}!")
            empty_checker_batch = DataProto.from_single_dict(
                {
                    "prompts": torch.empty(0, 1, dtype=torch.long),
                    "responses": torch.empty(0, 1, dtype=torch.long),
                    "attention_mask": torch.empty(0, 1, dtype=torch.long),
                }
            )
            empty_checker_batch.non_tensor_batch = {}
            empty_checker_batch.meta_info = {}

            return results, rule_reward_tensor, reward_extra_info, empty_checker_batch
        
        question_batch = self._prepare_fact_check_data(questions)
        
        # 3) checker 回答：三个温度
        majority_voting_nums = 3

        answer_batch_output_list = []
        for think_params in range(majority_voting_nums):
            if not self.async_rollout_mode:
                answer_batch_output = self.actor_rollout_wg.generate_sequences(question_batch)
            else:
                answer_batch_output = self.async_rollout_manager.generate_sequences(question_batch)
            answer_batch_output_list.append(answer_batch_output)

        # 4) majority voting + 额外 (answer_i=2) 构造 checker reward
        lines = []
        n_scores = 0

        # 记录 0.5 温度那条的 per-question 正误
        checker_correct_info = [None] * len(question_batch)  # idx -> list[1/-1]

        coef = self.config.get("custom_rewards_fact_check_sp_coef", 0.1)

        for idx in range(len(question_batch)):
            if question_batch[idx].non_tensor_batch['extra_info']['is_padding']:
                continue

            q_a_pairs = q_a_cases[idx]["q_a_pairs"]
            correct_count = np.zeros(len(q_a_pairs), dtype=int)

            checker_flags_for_this_idx = None  # 0.5 温度的正确性

            for answer_i in range(majority_voting_nums):
                answer = answer_batch_output_list[answer_i][idx]
                prompt_ids = answer.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]

                response_ids = answer.batch["responses"]
                valid_response_length = answer.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
                if not response_str.strip().startswith(f"{idx+1}."):
                    response_str = f"{idx+1}." + response_str.strip()
                re_ans = self._extract_detailed_info(response_str)

                correct_info = []
                for q_idx, q_a in enumerate(q_a_pairs):
                    if q_idx + 1 not in re_ans:
                        correct_info.append(-1)
                    elif self._check_ans_eq(re_ans[q_idx + 1]["answer"], q_a["answer"]):
                        correct_info.append(1)
                    else:
                        correct_info.append(-1)

                correct_count += np.array(correct_info, dtype=int)

                if answer_i == 2: # 和上面保持一致即可
                    checker_flags_for_this_idx = correct_info

                q_a_cases[idx][f'answer_{answer_i}'] = re_ans
                q_a_cases[idx][f'correct_info_{answer_i}'] = correct_info

            q_a_cases[idx]["correct_count"] = correct_count.tolist()
            checker_correct_info[idx] = checker_flags_for_this_idx

            fault_num = int((correct_count < 0).sum())
            true_num = int((correct_count > 0).sum())
            total_num = len(correct_count)
            
            use_ztr = self.config.get("use_ztr", True)
            
            if not use_ztr:    
                # error rate reward
                base_score = -coef * min(fault_num, 13)
            else:
                # zero tolerance reward
                if fault_num > 0:
                    base_score = -1.0
                else:  # fault_num == 0 and total_num > 0
                    base_score = 0.0   # 或者 full_correct_bonus

            results[cases[idx]["batch_idx"]] = {
                "total_num": total_num,
                "fault_num": fault_num,
                "true_num": true_num,
                "score": base_score,
                "bad_format_flag": False,
                "bad_request_flag": False,
                "idx": cases[idx]["batch_idx"],
            }
            rule_reward_tensor[cases[idx]["batch_idx"], cases[idx]['valid_response_length'] - 1] = base_score
            rewards_show[cases[idx]["batch_idx"]] = base_score
            for key, val in results[cases[idx]["batch_idx"]].items():
                if key == 'idx':
                    continue
                reward_extra_info[f"fact_check_sp_{key}"][cases[idx]["batch_idx"]] = val
            n_completed += 1
            n_scores += 1
            lines.append(json.dumps(q_a_cases[idx], ensure_ascii=False))

        # 5) 构造 checker_batch + 为其打 reward
        checker_answer_output = answer_batch_output_list[2] # 也可以random一下，都OK

        # 深拷一份，避免修改原对象
        checker_batch = deepcopy(checker_answer_output)

        # 为 non_tensor_batch 里的 extra_info 加一个 role 标志（可选）
        extra_infos = checker_batch.non_tensor_batch.get("extra_info", None)
        if extra_infos is not None:
            for i in range(len(extra_infos)):
                extra_infos[i]["role"] = "checker"

        # 构造 checker_reward_tensor，形状和 checker_batch.batch["responses"] 一致
        checker_reward_tensor = torch.zeros_like(checker_batch.batch["responses"], dtype=torch.float32)

        for idx in range(len(checker_batch)):
            # 这一条是否是 padding
            if question_batch[idx].non_tensor_batch['extra_info']['is_padding']:
                continue

            # 1) 找到对应的 Answerer 原始 batch index
            batch_idx = question_batch[idx].non_tensor_batch['extra_info']['index']
            # 注意：这里用的是你 _prepare_data 里写入的 'index'

            # 2) 从 rule_reward_tensor 中读取该 Answerer 的 fact-check reward
            ans_rewards = rule_reward_tensor[batch_idx]  # shape: [T_res]
            # 我们只在最后一个 valid token 上写过 base_score，其余为 0
            non_zero_positions = (ans_rewards != 0)
            if non_zero_positions.any():
                answerer_score = ans_rewards[non_zero_positions][-1].item()
            else:
                # 没有 fact-check 信号时（例如 no QA pairs 或 bad format），就用 0
                answerer_score = 0.0

            # 3) 把这个 score 打在这一条 checker 回答的最后一个 token 上
            responses = checker_batch.batch["responses"][idx]
            resp_len = responses.shape[-1]
            last_token_idx = resp_len - 1  # 0-based index

            checker_reward_tensor[idx, last_token_idx] = answerer_score

        checker_batch.batch["token_level_scores"] = checker_reward_tensor


        # 6) 保存 rollout 内容
        os.makedirs(os.path.join(os.environ["CUSTOM_OUTPUT_DIR"], "fact_check_rollout"), exist_ok=True)
        with open(os.path.join(os.environ["CUSTOM_OUTPUT_DIR"], f"fact_check_rollout/{self.global_steps}.jsonl"), "w") as f:
            f.write("\n".join(lines) + "\n")
        
        print(f"[FactCheckSelfPlay] Finished fact check for {n_scores} samples, n_not_label: {n_not_label}, n_bad_format: {n_bad_format}, n_no_qa_pairs: {n_no_qa_pairs}, n_completed: {n_completed}!")
        
        return results, rule_reward_tensor, reward_extra_info, checker_batch
    
    def _concat_dataprotos(self, dp1: DataProto, dp2: DataProto) -> DataProto:
        """
        在 batch 维度上拼接两个 DataProto。
        假设二者 tensor 部分的 keys 一致（至少 prompts/responses/attention_mask/response_mask/token_level_scores）。
        non_tensor_batch 逐 key 尝试拼接，如果某个 key 在 dp1/dp2 中长度与各自 batch_size 不一致，则忽略该 key。
        uid 在拼接后统一重建，避免长度不一致。
        """
        # 1) 拼 tensor batch
        new_batch = {}
        for k, v in dp1.batch.items():
            if k in dp2.batch:
                new_batch[k] = torch.cat([v, dp2.batch[k]], dim=0)
            else:
                new_batch[k] = v

        # 2) 构造 DataProto（只含 tensor 部分）
        new = DataProto.from_single_dict(new_batch)

        n1 = dp1.batch["prompts"].shape[0]
        n2 = dp2.batch["prompts"].shape[0]
        n = n1 + n2

        # 3) 拼 non_tensor_batch（除 uid 外）
        new.non_tensor_batch = {}
        # 所有 key 的集合
        all_keys = set(dp1.non_tensor_batch.keys()) | set(dp2.non_tensor_batch.keys())
        for k in all_keys:
            if k == "uid":
                continue  # uid 统一重建

            v1 = dp1.non_tensor_batch.get(k, None)
            v2 = dp2.non_tensor_batch.get(k, None)

            arr1 = np.array(v1, dtype=object) if v1 is not None else np.empty((n1,), dtype=object)
            arr2 = np.array(v2, dtype=object) if v2 is not None else np.empty((n2,), dtype=object)

            # 如果长度不匹配各自 batch size，说明这个 key 的数据不可靠，直接跳过
            if arr1.shape[0] != n1 or arr2.shape[0] != n2:
                # 也可以在这里 print 一下看看是哪些 key 出问题
                # print(f"[concat] skip key {k}: len1={arr1.shape[0]}, len2={arr2.shape[0]}, n1={n1}, n2={n2}")
                continue

            new.non_tensor_batch[k] = np.concatenate([arr1, arr2], axis=0)

        # 4) 统一重建 uid，保证长度一致
        batch_size = new.batch["prompts"].shape[0]
        new.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(batch_size)],
            dtype=object,
        )

        # 5) meta_info 沿用 dp1 的（如果有需要可以再融合）
        new.meta_info = deepcopy(dp1.meta_info)
        return new
