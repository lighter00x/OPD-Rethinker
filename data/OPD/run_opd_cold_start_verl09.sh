#!/usr/bin/env bash
# Section 5.1 cold-start comparison on the OpenThoughts3 complement split.

set -euo pipefail

STUDENT_VARIANT=${STUDENT_VARIANT:-sft}
case "${STUDENT_VARIANT}" in
    base)
        STUDENT_MODEL=/workspace/data/models/Qwen3-1.7B-Base
        EXPERIMENT_NAME=${EXPERIMENT_NAME:-pure_opd_qwen3_1p7b_base}
        METRICS_WANDB_RUN_ID=${METRICS_WANDB_RUN_ID:-cs17base1}
        ;;
    sft)
        STUDENT_MODEL=/workspace/data/models/Qwen3-1.7B-SFT
        EXPERIMENT_NAME=${EXPERIMENT_NAME:-cold_start_qwen3_1p7b_sft}
        METRICS_WANDB_RUN_ID=${METRICS_WANDB_RUN_ID:-cs17sft1}
        ;;
    *)
        echo "Unknown STUDENT_VARIANT=${STUDENT_VARIANT}; use base or sft." >&2
        exit 2
        ;;
esac

PROJECT_NAME=${PROJECT_NAME:-opd_cold_start_qwen3_1p7b_openthoughts3_30k}
TRAIN_FILE=${TRAIN_FILE:-/workspace/data/datasets/OPD/openthoughts3_complement_30k.parquet}
TEACHER_MODEL=${TEACHER_MODEL:-/workspace/data/models/Qwen3-4B}

for path in "${STUDENT_MODEL}" "${TEACHER_MODEL}" "${TRAIN_FILE}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Required cold-start asset is missing: ${path}" >&2
        exit 2
    fi
done

export STUDENT_MODEL TEACHER_MODEL TRAIN_FILE PROJECT_NAME EXPERIMENT_NAME
export TEACHER_VARIANT=qwen3-4b
export SAVE_FREQ=${SAVE_FREQ:-10}
export MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-2}
export DYNAMIC_METRICS_FREQ=${DYNAMIC_METRICS_FREQ:-5}
export PARAM_OFFLOAD=${PARAM_OFFLOAD:-True}
export OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-True}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}

# The self-hosted W&B backend can buffer an active veRL run's history until
# finish. Mirror the durable JSONL metrics into a fixed, resumable run so that
# per-step curves remain visible across trainer and container restarts.
if [[ "${ENABLE_WANDB_METRICS_MIRROR:-1}" == "1" ]]; then
    WANDB_ENV_FILE=${WANDB_ENV_FILE:-/workspace/data/secrets/opd_wandb.env}
    if [[ -f "${WANDB_ENV_FILE}" ]]; then
        set -a
        # shellcheck disable=SC1090
        source "${WANDB_ENV_FILE}"
        set +a
    fi
    export WANDB_BASE_URL=${WANDB_BASE_URL:-http://10.154.24.41:8082}

    METRICS_FILE=${METRICS_FILE:-/workspace/data/metrics/${PROJECT_NAME}/${EXPERIMENT_NAME}/train_metrics.jsonl}
    SYNC_STATE_FILE=${SYNC_STATE_FILE:-/workspace/data/metrics/${PROJECT_NAME}/${EXPERIMENT_NAME}/wandb_sync_state.json}
    SYNC_DIR=${SYNC_DIR:-/workspace/data/logs/${PROJECT_NAME}}
    SYNC_PID_FILE=${SYNC_PID_FILE:-${SYNC_DIR}/${EXPERIMENT_NAME}_wandb_sync.pid}
    SYNC_LOG_FILE=${SYNC_LOG_FILE:-${SYNC_DIR}/${EXPERIMENT_NAME}_wandb_sync.log}
    mkdir -p "${SYNC_DIR}" "$(dirname "${SYNC_STATE_FILE}")"

    sync_pid=""
    sync_state=""
    if [[ -f "${SYNC_PID_FILE}" ]]; then
        read -r sync_pid < "${SYNC_PID_FILE}" || true
        if [[ "${sync_pid}" =~ ^[0-9]+$ ]] && kill -0 "${sync_pid}" 2>/dev/null; then
            sync_state=$(ps -o stat= -p "${sync_pid}" 2>/dev/null || true)
        fi
    fi
    if [[ -z "${sync_state}" || "${sync_state}" == Z* ]]; then
        nohup python /workspace/code/OPD/scripts/sync_metrics_to_wandb.py \
            --metrics-file "${METRICS_FILE}" \
            --project "${PROJECT_NAME}" \
            --run-name "${EXPERIMENT_NAME}_metrics" \
            --run-id "${METRICS_WANDB_RUN_ID}" \
            --state-file "${SYNC_STATE_FILE}" \
            --poll-seconds 15 \
            >> "${SYNC_LOG_FILE}" 2>&1 < /dev/null &
        echo $! > "${SYNC_PID_FILE}"
    fi
fi

exec bash /workspace/code/OPD/run_opd_verl09.sh \
    '+data.apply_chat_template_kwargs.enable_thinking=False' \
    "$@"
