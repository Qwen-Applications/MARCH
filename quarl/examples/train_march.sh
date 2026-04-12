#!/bin/bash

set -e        # Exit on error
set -o pipefail  # Exit on pipe failure
set -x
ulimit -n 65535

# Setup NCCL
export NCCL_WORK_FIFO_DEPTH=4194304
export LANG=C.UTF-8
export LANGUAGE=C.UTF-8

pip3 install tensorboardX qwen_vl_utils
pip3 install transformers==4.52.4 vllm==0.8.5.post1

pip3 install "pyarrow>=19.0.1"
pip3 install math-verify
pip3 install "optree>=0.13.0"
pip3 install torchdata

# sglang
pip3 install sglang==0.4.6.post5 sgl_kernel==0.1.5 cuda-python cuda-bindings torch_memory_saver torchao

# update ray
pip install --upgrade --force-reinstall 'ray[default]'
pip install click==8.2.1

apt install -y rclone

# Setup workspace dir
export WORKSPACE=/root/code/my_workspace
bash /root/code/quarl/scripts/setup_workspace.sh

# Install VeRL
export PYTHONPATH=$WORKSPACE/verl
export REPO_DIR=$WORKSPACE/quarl
export PYTHONPATH=$WORKSPACE/verl:$REPO_DIR:$PYTHONPATH

# Setup tensorboard logging
export TENSORBOARD_DIR=${YOUR_TENSORBOARD_LOG_DIR}

# Setup training data, checkpoint paths, and other custom paths
export CUSTOM_TRAIN_DATA_PATH=${YOUR_TRAINING_DATA_PATH}
export CUSTOM_TEST_DATA_PATH=${YOUR_TEST_DATA_PATH}
export CUSTOM_TRAIN_CHECKPOINT_DIR=${YOUR_CHECKPOINT_SAVE_DIR}
export CUSTOM_SOURCE_ACTOR_CHECKPOINT_DIR=${YOUR_ACTOR_MODEL_CHECKPOINT_DIR}
export CUSTOM_SOURCE_REWARD_MODEL_CHECKPOINT_DIR=${YOUR_REWARD_MODEL_CHECKPOINT_DIR}
export CUSTOM_SOURCE_CRITIC_CHECKPOINT_DIR=${YOUR_CRITIC_MODEL_CHECKPOINT_DIR}
export CUSTOM_OUTPUT_DIR=${YOUR_ROLLOUT_OUTPUT_DIR}  # For saving rollout content and other outputs

# Resource info
export NNODES=${NNODES} 
export NODE_RANK=${RANK}
export MASTER_ADDR=${MASTER_ADDR} 

# Model path definitions
ACTOR_PATH=${CUSTOM_SOURCE_ACTOR_CHECKPOINT_DIR}
REWARD_PATH=${CUSTOM_SOURCE_REWARD_MODEL_CHECKPOINT_DIR}
CRITIC_PATH=${CUSTOM_SOURCE_CRITIC_CHECKPOINT_DIR}

echo "Actor path: $ACTOR_PATH"
echo "Reward path: $REWARD_PATH"
echo "Critic path: $CRITIC_PATH"


# Fixed parameters based on usage
TRAIN_METHOD=ppo
BATCH_SIZE=32
MINI_BATCH_SIZE=32
MAX_PROMPT_LENGTH=24567
MAX_RESPONSE_LENGTH=8192
PPO_MAX_TOKEN_LEN_PER_GPU=$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH ))
PPO_MICRO_BATCH_PER_GPU=1
LOG_PROB_MIRCO_BATCH_SIZE_PER_GPU=1
REWARD_MIRCO_BATCH_SIZE_PER_GPU=1
CRITIC_MIRCO_BATCH_SIZE_PER_GPU=1
ROLLOUT_TP=4
GRM_TP=1
ACTOR_SP=1
REWARD_SP=1
CRITIC_SP=1
ACTOR_FSDP_SIZE=-1
FSDP_TYPE=fsdp
CHAT_TEMPLATE=null
RM_ENABLE=True
TOTAL_EPOCHS=1
ACTOR_DTYPE=null

# PPO specific settings
ROLLOUT_N=1
ADV_ESTIMATOR=gae
ACTOR_USE_KL_LOSS=False
USE_KL_IN_REWARD=True

# quark custom arguments
TASK_TYPE=fact_check_sp_march
RM_TYPE=default
DATASET_CLASS=null
ROLLOUT_ENGINE=vllm

# Common RM Args
CUSTOM_RM_ARGS="reward_model.reward_manager=quark \
    +custom_reward_functions.bad_pattern.labels=['rag_for_digit_fact','rag_not_for_digit_fact','rag','fortune_telling','emotion','fuzzy_intention','medical'] \
    +custom_reward_functions.bad_pattern.integration=sum \
    +custom_reward_functions.bad_pattern.kwargs.bad_pattern_coef=0.1
    +custom_rewards_fact_check_sp_coef=0.1 \
    +custom_rewards_fact_check_sp_labels=['rag_for_digit_fact']"

USE_CHECKER_REWARD=True
USE_QA_NUM_REWARD=False
RLHF_BASELINE=factcheck
USE_ZTR=True

# Sync nodes
sh $REPO_DIR/scripts/sync_nodes.sh

if [ $NODE_RANK -eq 0 ]; then
    # Primary node runs training
    ray start --block --head --port=6379 &

    if [ ! $NNODES -eq 1 ]; then sleep 240; fi
    
    python3 $REPO_DIR/quarl/main_rl.py \
        quark.task_type=${TASK_TYPE} \
        quark.reward_type=${RM_TYPE} \
        algorithm.adv_estimator=${ADV_ESTIMATOR} \
        data.train_files=${CUSTOM_TRAIN_DATA_PATH}.parquet \
        data.val_files=${CUSTOM_TEST_DATA_PATH}.parquet \
        data.train_batch_size=${BATCH_SIZE} \
        data.max_prompt_length=${MAX_PROMPT_LENGTH} \
        data.max_response_length=${MAX_RESPONSE_LENGTH} \
        data.return_raw_chat=True \
        data.filter_overlong_prompts=True \
        data.truncation='right' \
        data.custom_cls.name=${DATASET_CLASS} \
        actor_rollout_ref.model.path=${ACTOR_PATH} \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.strategy=${FSDP_TYPE} \
        actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE} \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_PER_GPU} \
        actor_rollout_ref.actor.use_dynamic_bsz=False \
        actor_rollout_ref.actor.use_kl_loss=${ACTOR_USE_KL_LOSS} \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ACTOR_SP} \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        +actor_rollout_ref.actor.fsdp_config.model_dtype=$ACTOR_DTYPE \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.actor.fsdp_config.offload_policy=True \
        actor_rollout_ref.actor.fsdp_config.fsdp_size=$ACTOR_FSDP_SIZE \
        actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
        actor_rollout_ref.rollout.name=${ROLLOUT_ENGINE} \
        actor_rollout_ref.rollout.max_num_batched_tokens=$PPO_MAX_TOKEN_LEN_PER_GPU \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MIRCO_BATCH_SIZE_PER_GPU} \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
        actor_rollout_ref.rollout.n=${ROLLOUT_N} \
        actor_rollout_ref.type=quark_fact_check_sp \
        actor_rollout_ref.ref.strategy=${FSDP_TYPE} \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MIRCO_BATCH_SIZE_PER_GPU} \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.model.custom_chat_template=${CHAT_TEMPLATE} \
        algorithm.use_kl_in_reward=${USE_KL_IN_REWARD} \
        algorithm.kl_ctrl.kl_coef=0.1 \
        algorithm.gamma=0.998 \
        critic.optim.lr=1e-5 \
        critic.ppo_micro_batch_size_per_gpu=${CRITIC_MIRCO_BATCH_SIZE_PER_GPU} \
        critic.strategy=${FSDP_TYPE} \
        critic.model.use_remove_padding=True \
        critic.model.path=${CRITIC_PATH} \
        critic.model.enable_gradient_checkpointing=True \
        critic.model.fsdp_config.param_offload=True \
        critic.model.fsdp_config.optimizer_offload=True \
        critic.ulysses_sequence_parallel_size=${CRITIC_SP} \
        reward_model.enable=${RM_ENABLE} \
        reward_model.model.input_tokenizer=${REWARD_PATH} \
        reward_model.strategy=${FSDP_TYPE} \
        reward_model.model.path=${REWARD_PATH} \
        reward_model.model.use_remove_padding=True \
        reward_model.model.type=${RM_TYPE} \
        reward_model.max_length=$PPO_MAX_TOKEN_LEN_PER_GPU \
        reward_model.micro_batch_size_per_gpu=${REWARD_MIRCO_BATCH_SIZE_PER_GPU} \
        reward_model.model.fsdp_config.param_offload=True \
        reward_model.ulysses_sequence_parallel_size=${REWARD_SP} \
        ${CUSTOM_RM_ARGS} \
        +rlhf_baseline=${RLHF_BASELINE} \
        +use_checker_ppo=${USE_CHECKER_REWARD} \
        +use_qa_num_reward=${USE_QA_NUM_REWARD} \
        +use_ztr=${USE_ZTR} \
        trainer.val_before_train=False \
        trainer.critic_warmup=10 \
        trainer.logger=['console','tensorboard'] \
        trainer.project_name=quark_RLHF_${TRAIN_METHOD} \
        trainer.experiment_name=march_train_${TRAIN_METHOD} \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=${NNODES} \
        trainer.save_freq=10 \
        trainer.test_freq=5 \
        trainer.default_local_dir=${CUSTOM_TRAIN_CHECKPOINT_DIR} \
        trainer.total_epochs=${TOTAL_EPOCHS}
else
    ray start --block --address=${MASTER_ADDR}:6379
    sleep 120
fi
