#!/usr/bin/env bash
# Paper-compatible student-Top-K OPD on verl 0.9 (legacy trainer).

set -euo pipefail

WANDB_ENV_FILE=${WANDB_ENV_FILE:-/workspace/data/secrets/opd_wandb.env}
if [[ -f "${WANDB_ENV_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${WANDB_ENV_FILE}"
    set +a
fi
WANDB_BASE_URL=${WANDB_BASE_URL:-http://10.154.24.41:8082}
WANDB_MODE=${WANDB_MODE:-online}
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "WANDB_API_KEY is not set; provide it through ${WANDB_ENV_FILE} or the environment." >&2
    exit 2
fi
export WANDB_API_KEY WANDB_BASE_URL WANDB_MODE

TEACHER_VARIANT=${TEACHER_VARIANT:-qwen3-4b}
case "${TEACHER_VARIANT}" in
    qwen3-4b)
        TEACHER_MODEL=${TEACHER_MODEL:-/workspace/data/models/Qwen3-4B}
        ;;
    qwen3-4b-base-grpo)
        TEACHER_MODEL=${TEACHER_MODEL:-/workspace/data/models/Qwen3-4B-Base-GRPO}
        ;;
    *)
        echo "Unknown TEACHER_VARIANT=${TEACHER_VARIANT}; use qwen3-4b or qwen3-4b-base-grpo." >&2
        exit 2
        ;;
esac

STUDENT_MODEL=${STUDENT_MODEL:-/workspace/data/models/Qwen3-1.7B-Base}
TRAIN_FILE=${TRAIN_FILE:-/workspace/data/datasets/OPD/dapo-math-17k-processed.parquet}
EVAL_DATA_DIR=${EVAL_DATA_DIR:-/workspace/data/datasets/OPD/test_data}
VAL_FILES=${VAL_FILES:-"['${EVAL_DATA_DIR}/AIME24/test.parquet','${EVAL_DATA_DIR}/AIME25/test.parquet','${EVAL_DATA_DIR}/AMC23/test.parquet']"}

ACTOR_NNODES=${ACTOR_NNODES:-2}
ACTOR_GPUS_PER_NODE=${ACTOR_GPUS_PER_NODE:-4}
ACTOR_NODE_IPS=${ACTOR_NODE_IPS:-10.154.24.42,10.154.24.43}
SCORER_NNODES=${SCORER_NNODES:-2}
SCORER_GPUS_PER_NODE=${SCORER_GPUS_PER_NODE:-2}
TEACHER_NODE_IPS=${TEACHER_NODE_IPS:-10.154.24.41,10.154.24.42}
NUM_SCORERS=$((SCORER_NNODES * SCORER_GPUS_PER_NODE))

IFS=',' read -r -a ACTOR_NODE_IP_ARRAY <<<"${ACTOR_NODE_IPS}"
IFS=',' read -r -a TEACHER_NODE_IP_ARRAY <<<"${TEACHER_NODE_IPS}"
if (( ${#ACTOR_NODE_IP_ARRAY[@]} != ACTOR_NNODES )); then
    echo "ACTOR_NODE_IPS=${ACTOR_NODE_IPS} must contain ACTOR_NNODES=${ACTOR_NNODES} entries." >&2
    exit 2
fi
if (( ${#TEACHER_NODE_IP_ARRAY[@]} != SCORER_NNODES )); then
    echo "TEACHER_NODE_IPS=${TEACHER_NODE_IPS} must contain SCORER_NNODES=${SCORER_NNODES} entries." >&2
    exit 2
fi
ACTOR_NODE_IPS_CONFIG="[\"${ACTOR_NODE_IPS//,/\",\"}\"]"
TEACHER_NODE_IPS_CONFIG="[\"${TEACHER_NODE_IPS//,/\",\"}\"]"

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
ROLLOUT_N=${ROLLOUT_N:-4}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-7168}
MAX_VAL_RESPONSE_LENGTH=${MAX_VAL_RESPONSE_LENGTH:-31744}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-4096}
DYNAMIC_METRICS_FREQ=${DYNAMIC_METRICS_FREQ:-1}
PARAM_OFFLOAD=${PARAM_OFFLOAD:-True}
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-True}
NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-1}
NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}

ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1.0}
TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-1.0}
TOPK=${TOPK:-16}
ACTOR_LR=${ACTOR_LR:-1e-6}

ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
SAVE_FREQ=${SAVE_FREQ:-20}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-2}
TEST_FREQ=${TEST_FREQ:--1}
PROJECT_NAME=${PROJECT_NAME:-opd_verl09_qwen3_1p7b}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${TEACHER_VARIANT}_student-topk}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/workspace/data/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}
METRICS_FILE=${METRICS_FILE:-/workspace/data/metrics/${PROJECT_NAME}/${EXPERIMENT_NAME}/train_metrics.jsonl}
export VERL_FILE_LOGGER_PATH=${VERL_FILE_LOGGER_PATH:-${METRICS_FILE}}
mkdir -p "$(dirname "${VERL_FILE_LOGGER_PATH}")"

MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_VAL_RESPONSE_LENGTH))

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="['${TRAIN_FILE}']"
    data.val_files="${VAL_FILES}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation=error
    data.shuffle=False
)

MODEL=(
    actor_rollout_ref.model.path="${STUDENT_MODEL}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=False
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.shuffle=False
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD}
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIMIZER_OFFLOAD}
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}
    actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE}
    actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=16
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95
)

DISTILLATION=(
    distillation.enabled=True
    distillation.n_gpus_per_node=${SCORER_GPUS_PER_NODE}
    distillation.nnodes=${SCORER_NNODES}
    "+distillation.resource_pool_node_ips=${TEACHER_NODE_IPS_CONFIG}"
    distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL}"
    distillation.distillation_loss.loss_mode=opd_student_topk
    distillation.distillation_loss.topk=${TOPK}
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=False
    distillation.distillation_loss.clip_ratio_low=0.2
    distillation.distillation_loss.clip_ratio_high=0.2
    distillation.opd_student_topk.num_scorers=${NUM_SCORERS}
    distillation.opd_student_topk.dtype=bfloat16
    distillation.opd_student_topk.temperature=${TEACHER_TEMPERATURE}
    distillation.opd_student_topk.logits_chunk_size=128
    distillation.opd_student_topk.dynamic_metrics_freq=${DYNAMIC_METRICS_FREQ}
)

TRAINER=(
    +ray_kwargs.ray_init.runtime_env.working_dir=/workspace/code/verl
    "+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_P2P_DISABLE=\"${NCCL_P2P_DISABLE}\""
    "+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_SHM_DISABLE=\"${NCCL_SHM_DISABLE}\""
    "+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_IB_DISABLE=\"${NCCL_IB_DISABLE}\""
    '+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_SOCKET_IFNAME="ens5f0"'
    '+ray_kwargs.ray_init.runtime_env.env_vars.GLOO_SOCKET_IFNAME="ens5f0"'
    trainer.use_v1=False
    trainer.balance_batch=True
    trainer.logger='["console","file","wandb"]'
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node=${ACTOR_GPUS_PER_NODE}
    trainer.nnodes=${ACTOR_NNODES}
    "+trainer.resource_pool_node_ips=${ACTOR_NODE_IPS_CONFIG}"
    trainer.val_before_train=False
    trainer.save_freq=${SAVE_FREQ}
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.default_local_dir="${CHECKPOINT_DIR}"
)

REWARD=(
    reward.custom_reward_function.path=pkg://verl.utils.reward_score.ttrl_math
    reward.custom_reward_function.name=reward_func
)

cd /workspace/code/verl
python -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${DISTILLATION[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "$@"
