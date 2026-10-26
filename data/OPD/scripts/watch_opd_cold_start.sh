#!/usr/bin/env bash
# Supervise the cold-start OPD run and resume it from the latest complete checkpoint.

set -uo pipefail

PROJECT_NAME=${PROJECT_NAME:-opd_cold_start_qwen3_1p7b_openthoughts3_30k}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-cold_start_qwen3_1p7b_sft_3090_v2}
METRICS_WANDB_RUN_ID=${METRICS_WANDB_RUN_ID:-cs17sft2}
WANDB_RUN_ID=${WANDB_RUN_ID:-rk6b6lmn}
WANDB_RESUME_MODE=${WANDB_RESUME_MODE:-must}
STUDENT_VARIANT=${STUDENT_VARIANT:-sft}
EXPECTED_GPUS=${EXPECTED_GPUS:-12}
EXPECTED_NODES=${EXPECTED_NODES:-3}
EXPECTED_GPU_LAYOUT=${EXPECTED_GPU_LAYOUT:-10.154.24.41=2,10.154.24.42=6,10.154.24.43=4}
ACTOR_NODE_IPS=${ACTOR_NODE_IPS:-10.154.24.42,10.154.24.43}
TEACHER_NODE_IPS=${TEACHER_NODE_IPS:-10.154.24.41,10.154.24.42}
RESTART_DELAY=${RESTART_DELAY:-60}
MAX_FAILURES_WITHOUT_CHECKPOINT=${MAX_FAILURES_WITHOUT_CHECKPOINT:-3}

LOG_DIR=${LOG_DIR:-/workspace/data/logs/${PROJECT_NAME}}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/workspace/data/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}
PID_FILE=${PID_FILE:-${LOG_DIR}/${EXPERIMENT_NAME}_watchdog.pid}
CHILD_PID_FILE=${CHILD_PID_FILE:-${LOG_DIR}/${EXPERIMENT_NAME}.pid}
SUPERVISOR_LOG=${SUPERVISOR_LOG:-${LOG_DIR}/${EXPERIMENT_NAME}_watchdog.log}
LOCK_FILE=${LOCK_FILE:-${LOG_DIR}/${EXPERIMENT_NAME}_watchdog.lock}

mkdir -p "${LOG_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "A watchdog for ${EXPERIMENT_NAME} is already running." >&2
    exit 3
fi
echo $$ >"${PID_FILE}"

child_pid=""
cleanup() {
    if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
        kill -TERM "${child_pid}" 2>/dev/null || true
        wait "${child_pid}" 2>/dev/null || true
    fi
    rm -f "${PID_FILE}" "${CHILD_PID_FILE}"
}
on_signal() {
    exit 143
}
trap cleanup EXIT
trap on_signal INT TERM

checkpoint_step() {
    local marker="${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt"
    if [[ -f "${marker}" ]]; then
        tr -cd '0-9' <"${marker}"
    else
        printf '0'
    fi
}

wait_for_cluster() {
    while ! timeout 150 python /workspace/code/OPD/scripts/check_ray_gpu_cluster.py \
        --expected-gpus "${EXPECTED_GPUS}" \
        --expected-nodes "${EXPECTED_NODES}" \
        --expected-layout "${EXPECTED_GPU_LAYOUT}"; do
        echo "$(date -Is) Ray/CUDA cluster is not ready; retrying in ${RESTART_DELAY}s."
        sleep "${RESTART_DELAY}"
    done
}

attempt=0
failures_without_checkpoint=0
last_checkpoint=$(checkpoint_step)

while true; do
    wait_for_cluster
    attempt=$((attempt + 1))
    attempt_log="${LOG_DIR}/${EXPERIMENT_NAME}_autoresume_$(date +%Y%m%d_%H%M%S)_attempt${attempt}.log"
    echo "$(date -Is) Starting attempt ${attempt} from checkpoint ${last_checkpoint}; log=${attempt_log}"

    env \
        STUDENT_VARIANT="${STUDENT_VARIANT}" \
        PROJECT_NAME="${PROJECT_NAME}" \
        EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
        METRICS_WANDB_RUN_ID="${METRICS_WANDB_RUN_ID}" \
        WANDB_RUN_ID="${WANDB_RUN_ID}" \
        WANDB_RESUME="${WANDB_RESUME_MODE}" \
        ACTOR_NNODES=2 \
        ACTOR_GPUS_PER_NODE=4 \
        ACTOR_NODE_IPS="${ACTOR_NODE_IPS}" \
        SCORER_NNODES=2 \
        SCORER_GPUS_PER_NODE=2 \
        TEACHER_NODE_IPS="${TEACHER_NODE_IPS}" \
        OPTIMIZER_OFFLOAD=True \
        TEST_FREQ=20 \
        bash /workspace/code/OPD/run_opd_cold_start_verl09.sh \
        trainer.resume_mode=auto \
        9>&- \
        >"${attempt_log}" 2>&1 &
    child_pid=$!
    echo "${child_pid}" >"${CHILD_PID_FILE}"

    wait "${child_pid}"
    status=$?
    child_pid=""
    rm -f "${CHILD_PID_FILE}"

    if [[ ${status} -eq 0 ]]; then
        echo "$(date -Is) Training completed successfully."
        exit 0
    fi

    current_checkpoint=$(checkpoint_step)
    if (( current_checkpoint > last_checkpoint )); then
        failures_without_checkpoint=0
        last_checkpoint=${current_checkpoint}
    else
        failures_without_checkpoint=$((failures_without_checkpoint + 1))
    fi

    echo "$(date -Is) Attempt ${attempt} exited with status ${status}; "\
         "checkpoint=${current_checkpoint}, consecutive failures without a new checkpoint=${failures_without_checkpoint}."
    if (( failures_without_checkpoint >= MAX_FAILURES_WITHOUT_CHECKPOINT )); then
        echo "$(date -Is) Refusing an infinite restart loop after ${failures_without_checkpoint} failures."
        exit "${status}"
    fi
    sleep "${RESTART_DELAY}"
done
