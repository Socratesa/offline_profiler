source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate fd_final

# fd_final is used for its newer paddle (dev20260820): the FastDeploy branch imports
# paddle.distributed.launch.utils.control_socket, which the older fd_afd_dev paddle
# (dev20260528) does not have.  But fd_final's own site-packages copy of fastdeploy
# has no MoE route-counting instrumentation, so PYTHONPATH must win over it or the
# /v1/save_moe_expert_stats flush this whole run depends on will not exist.
export PYTHONPATH="/workspace/fd_afd_dev/FastDeploy${PYTHONPATH:+:$PYTHONPATH}"

# Single-service GLM-4.5-Air launcher used by offline_exp/scripts/run_all_glm_datasets.sh.
# Every knob below is overridable from the environment: the queue runner exports
# CUDA_VISIBLE_DEVICES / FD_LOG_DIR / FD_MOE_STATS_* / the ports / the length
# limits per dataset, then backgrounds this script and waits on /health.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export FD_LOG_DIR="${FD_LOG_DIR:-test}"

# Per-expert route counting. The queue runner sets ENABLE=1 and points PATH at
# runs/<tag>/stats/<dataset>.npz, then POSTs /v1/save_moe_expert_stats to flush it.
export FD_MOE_STATS_ENABLE="${FD_MOE_STATS_ENABLE:-0}"
export FD_MOE_STATS_PATH="${FD_MOE_STATS_PATH:-/workspace/fd_afd_dev/offline_exp/runs/mixed_eplb_r0/moe_expert_stats.npz}"

MODEL_PATH="${MODEL_PATH:-/workspace/models/GLM-4.5-Air}"
API_PORT="${API_PORT:-1910}"
ENGINE_WORKER_QUEUE_PORT="${ENGINE_WORKER_QUEUE_PORT:-24710}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"


# The caller already wipes FD_LOG_DIR and the stats file before each dataset, so
# only make sure the directory exists here rather than removing anything.
mkdir -p "${FD_LOG_DIR}"

# EPLB is intentionally left off: the eval runs measure the model's natural
# routing, and rebalancing would perturb the very counts being collected.
# The expert-route-counting experiment (offline_exp/method/) enabled it with:
#   EPLB_CONFIG='{"redundant_experts_num": 0, "redundant_expert_dump_workload_interval": 1}'
#   --enable-eplb --eplb-config "${EPLB_CONFIG}"

# exec so the PID the caller records is the server itself, which makes its
# kill -TERM land on the API master instead of this wrapper shell.
exec python -m fastdeploy.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --port "${API_PORT}" \
  --engine-worker-queue-port "${ENGINE_WORKER_QUEUE_PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --data-parallel-size "${DATA_PARALLEL_SIZE}" \
  --gpu-memory-utilization 0.9 \
  --enable-chunked-prefill \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max-model-len "${MAX_MODEL_LEN}"
