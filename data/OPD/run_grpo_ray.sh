#!/usr/bin/env bash

set -euo pipefail

: "${WANDB_API_KEY:?WANDB_API_KEY must be set before launching GRPO}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="${GRPO_PROFILE:-smoke}"
SHARED_ROOT="${SHARED_ROOT:-/workspace/data}"
RAY_ADDRESS="${RAY_ADDRESS:-auto}"
WANDB_BASE_URL="${WANDB_BASE_URL:-http://10.154.24.41:8082}"
PROJECT_NAME="${WANDB_PROJECT:-OPD-GRPO}"

export RAY_ADDRESS WANDB_BASE_URL

case "${PROFILE}" in
    smoke)
        TRAIN_DATASET="${TRAIN_DATASET:-${SHARED_ROOT}/datasets/OPD/dapo-math-17k-processed-smoke-32.parquet}"
        GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
        MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-32}"
        N_RESPONSES="${N_RESPONSES:-2}"
        MAX_RESP_LENGTH="${MAX_RESP_LENGTH:-512}"
        MAX_VAL_RESP_LENGTH="${MAX_VAL_RESP_LENGTH:-512}"
        TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}"
        SAVE_FREQ="${SAVE_FREQ:--1}"
        TEST_FREQ="${TEST_FREQ:--1}"
        ;;
    paper)
        TRAIN_DATASET="${TRAIN_DATASET:-${SHARED_ROOT}/datasets/OPD/dapo-math-17k-processed.parquet}"
        GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
        MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-64}"
        N_RESPONSES="${N_RESPONSES:-8}"
        MAX_RESP_LENGTH="${MAX_RESP_LENGTH:-7168}"
        MAX_VAL_RESP_LENGTH="${MAX_VAL_RESP_LENGTH:-31744}"
        TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
        SAVE_FREQ="${SAVE_FREQ:-20}"
        TEST_FREQ="${TEST_FREQ:-20}"
        ;;
    *)
        echo "Unknown GRPO_PROFILE=${PROFILE}; expected smoke or paper" >&2
        exit 2
        ;;
esac

MODEL_PATH="${MODEL_PATH:-${SHARED_ROOT}/models/Qwen3-4B-Base}"
VERL_SOURCE="${VERL_SOURCE:-/workspace/code/verl}"
REWARD_PATH="${REWARD_PATH:-${SHARED_ROOT}/code/OPD/ttrl_math/__init__.py}"
TEST_DATA_DIR="${TEST_DATA_DIR:-${SHARED_ROOT}/datasets/OPD/test_data}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESP_LENGTH > MAX_PROMPT_LENGTH + MAX_VAL_RESP_LENGTH ? MAX_PROMPT_LENGTH + MAX_RESP_LENGTH : MAX_PROMPT_LENGTH + MAX_VAL_RESP_LENGTH))}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-$((MAX_PROMPT_LENGTH + MAX_RESP_LENGTH > 32768 ? MAX_PROMPT_LENGTH + MAX_RESP_LENGTH : 32768))}"

# Eight GPUs preserve the paper's batch arithmetic. Offload and TP=2 adapt the 80 GB recipe to 24 GB cards.
NUM_NODES="${NUM_NODES:-2}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
PARALLEL_SIZE="${PARALLEL_SIZE:-2}"
MODEL_DTYPE="${MODEL_DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.5}"
NETWORK_INTERFACE="${NETWORK_INTERFACE:-ens5f0}"

ROLLOUT_BATCH_SIZE=$((GLOBAL_BATCH_SIZE * N_RESPONSES))
WORLD_SIZE=$((NUM_NODES * GPUS_PER_NODE))
if (( WORLD_SIZE % PARALLEL_SIZE != 0 )); then
    echo "WORLD_SIZE=${WORLD_SIZE} must be divisible by PARALLEL_SIZE=${PARALLEL_SIZE}" >&2
    exit 2
fi
DATA_PARALLEL_SIZE=$((WORLD_SIZE / PARALLEL_SIZE))
if (( ROLLOUT_BATCH_SIZE % DATA_PARALLEL_SIZE != 0 )); then
    echo "GLOBAL_BATCH_SIZE*N_RESPONSES=${ROLLOUT_BATCH_SIZE} must be divisible by data parallel size ${DATA_PARALLEL_SIZE}" >&2
    exit 2
fi

python "${SCRIPT_DIR}/scripts/setup/check_grpo_environment.py" \
    --expected-nodes 3 \
    --expected-gpus 18 \
    --shared-root "${SHARED_ROOT}" \
    --wandb-base-url "${WANDB_BASE_URL}" \
    --login-wandb

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3-4b-base-grpo-${PROFILE}-${RUN_STAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SHARED_ROOT}/checkpoints/Qwen3-4B-Base-GRPO/${EXPERIMENT_NAME}}"
VALIDATION_DIR="${VALIDATION_DIR:-${SHARED_ROOT}/validation/GRPO/${EXPERIMENT_NAME}}"
LOG_DIR="${LOG_DIR:-${SHARED_ROOT}/logs/grpo}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${EXPERIMENT_NAME}.log}"
mkdir -p "${OUTPUT_DIR}" "${VALIDATION_DIR}" "${LOG_DIR}"

export PYTHONPATH="${VERL_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=true
export RAY_memory_usage_threshold=0.99
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${NETWORK_INTERFACE}}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NETWORK_INTERFACE}}"
export TORCH_NCCL_BLOCKING_WAIT=1

echo "profile=${PROFILE} experiment=${EXPERIMENT_NAME} world_size=${WORLD_SIZE} dp=${DATA_PARALLEL_SIZE} tp/sp=${PARALLEL_SIZE}"
VERL_REVISION="$(git -C "${VERL_SOURCE}" rev-parse HEAD 2>/dev/null || printf 'unknown')"
echo "model=${MODEL_PATH} dataset=${TRAIN_DATASET} output=${OUTPUT_DIR} log=${LOG_FILE}"
echo "verl_source=${VERL_SOURCE} verl_revision=${VERL_REVISION} reward=${REWARD_PATH}"

cd "${VERL_SOURCE}"
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.shuffle=False \
    data.train_files="${TRAIN_DATASET}" \
    data.val_files="['${TEST_DATA_DIR}/AIME25/test.parquet','${TEST_DATA_DIR}/AMC23/test.parquet','${TEST_DATA_DIR}/AIME24/test.parquet']" \
    data.train_batch_size="${GLOBAL_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESP_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${PARALLEL_SIZE}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype="${MODEL_DTYPE}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.dtype="${MODEL_DTYPE}" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${PARALLEL_SIZE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.n="${N_RESPONSES}" \
    actor_rollout_ref.rollout.repetition_penalty=1.0 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype="${MODEL_DTYPE}" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    reward.reward_model.enable=False \
    reward.custom_reward_function.path="${REWARD_PATH}" \
    reward.custom_reward_function.name=reward_func \
    trainer.val_before_train=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.validation_data_dir="${VALIDATION_DIR}" \
    trainer.n_gpus_per_node="${GPUS_PER_NODE}" \
    trainer.nnodes="${NUM_NODES}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs=1 \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.resume_mode=disable \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    +ray_kwargs.ray_init.runtime_env.working_dir="${VERL_SOURCE}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.WANDB_BASE_URL="${WANDB_BASE_URL}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME}" \
    "$@" 2>&1 | tee -a "${LOG_FILE}"
