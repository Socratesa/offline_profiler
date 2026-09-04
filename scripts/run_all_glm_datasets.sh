#!/usr/bin/env bash
set -u

# Run each dataset in isolation. A failed dataset is recorded and does not stop the queue.
ROOT="/workspace/fd_afd_dev"
PYTHON_BIN="${PYTHON_BIN:-/workspace/miniconda3/envs/fd_final/bin/python}"
# TAG="${TAG:-glm_think_32k_c512_20260817}"
TAG="${TAG:-glm_think_32k_c512_20260901}"
RUN_DIR="$ROOT/offline_exp/runs/$TAG"
LOG_DIR="$RUN_DIR/logs"
FD_LOG_ROOT="$RUN_DIR/fd_logs"
MODEL="${MODEL:-/workspace/models/GLM-4.5-Air}"
API_PORT="${API_PORT:-1910}"
ENGINE_PORT="${ENGINE_PORT:-24710}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
DATASET_WORKERS="${DATASET_WORKERS:-512}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-14400}"
DATASET_TIMEOUT="${DATASET_TIMEOUT:-172800}"
TRUNCATE_RETRIES="${TRUNCATE_RETRIES:-0}"
# Smoke-test knob: 0 = full dataset. When >0, only the first N records of each
# dataset are inferred, and N also replaces the EXPECTED count so that the
# completeness gate still lets scoring run.
LIMIT="${LIMIT:-0}"
# Five domains, calibration first then evaluation.  See offline_exp/data/README.md
# for why each set was picked.  Legacy queues kept for reference:
#   DATASETS="financebench humaneval asdiv aime mbpp openbookqa finqa livecodebench"
#   DATASETS="pubmedqa medqa_usmle legalqaeval mmlu_pro_law"
DATASETS="${DATASETS:-asdiv mbpp finqa medmcqa lexam_law gsm8k math500 humaneval livecodebench financebench medqa_usmle mmlu_pro_law}"
STATUS="$LOG_DIR/run_all_status.md"

declare -A EXPECTED=(
  # 校准集
  [asdiv]=2305 [mbpp]=374 [finqa]=883 [medmcqa]=4183 [lexam_law]=712
  # 评测集
  [gsm8k]=1319 [math500]=500 [humaneval]=164 [livecodebench]=1055
  [financebench]=150 [medqa_usmle]=1273 [mmlu_pro_law]=1101
  # Retired: [arc_challenge]=1172 [openbookqa]=4957 [aime]=120
  #          [pubmedqa]=1000 [legalqaeval]=1204
)

mkdir -p "$RUN_DIR/output" "$RUN_DIR/score" "$RUN_DIR/code" "$RUN_DIR/stats" "$FD_LOG_ROOT" "$LOG_DIR"
printf '# Dataset run status\n\n' > "$STATUS"

service_pid=""
valid_records() {
  local path="$1"
  "$PYTHON_BIN" - "$path" <<'PY'
import json
import sys

best = {}
try:
    stream = open(sys.argv[1], encoding="utf-8")
except FileNotFoundError:
    print(0)
    raise SystemExit
with stream:
    for line in stream:
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        idx = row.get("idx")
        if idx is None:
            continue
        rank = (row.get("error") is None, row.get("finish_reason", "stop") == "stop")
        if idx not in best or rank >= best[idx][0]:
            best[idx] = (rank, row)
print(sum(row.get("error") is None and row.get("finish_reason", "stop") == "stop"
          for _, row in best.values()))
PY
}

stop_service() {
  if [[ -n "$service_pid" ]] && kill -0 "$service_pid" 2>/dev/null; then
    kill -TERM "$service_pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$service_pid" 2>/dev/null || break
      sleep 1
    done
  fi
  # run_glm.sh launches a process tree; clean only the configured worker port.
  pkill -TERM -f "/worker/worker_process.py.*--engine_worker_queue_port $ENGINE_PORT" 2>/dev/null || true
  pkill -TERM -f "fastdeploy.entrypoints.openai.api_server.*--port $API_PORT" 2>/dev/null || true
  # The engine worker queue may keep the TCP port briefly after the API master exits.
  for port in "$API_PORT" "$ENGINE_PORT"; do
    # Allow up to three minutes for the API/engine process tree to release the port.
    for _ in $(seq 1 180); do
      if "$PYTHON_BIN" - "$port" <<'PY'
import socket
import sys

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind(("127.0.0.1", int(sys.argv[1])))
except OSError:
    raise SystemExit(1)
finally:
    s.close()
PY
      then
        break
      fi
      sleep 1
    done
  done
  service_pid=""
}

port_is_free() {
  "$PYTHON_BIN" - "$1" <<'PY'
import socket
import sys

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind(("127.0.0.1", int(sys.argv[1])))
except OSError:
    raise SystemExit(1)
finally:
    s.close()
PY
}

save_stats() {
  local dataset="$1"
  local stats="$RUN_DIR/stats/${dataset}.npz"
  if ! curl -fsS --max-time 360 -X POST \
      "http://127.0.0.1:${API_PORT}/v1/save_moe_expert_stats" >/dev/null; then
    return 1
  fi
  for _ in $(seq 1 60); do
    if [[ -s "$stats" ]]; then
      return 0
    fi
    sleep 1
  done
  return 1
}

start_service() {
  local dataset="$1"
  local service_log="$LOG_DIR/${dataset}.service.stdout.log"
  local fd_log="$FD_LOG_ROOT/${dataset}_fd_log"
  local stats="$RUN_DIR/stats/${dataset}.npz"
  if ! port_is_free "$API_PORT" || ! port_is_free "$ENGINE_PORT"; then
    printf 'ports unavailable before %s: api=%s engine=%s\n' \
      "$dataset" "$API_PORT" "$ENGINE_PORT" >> "$service_log"
    return 1
  fi
  rm -f "$stats"
  rm -rf "$fd_log"
  CUDA_VISIBLE_DEVICES="$GPU_IDS" \
    FD_LOG_DIR="$fd_log" \
    FD_MOE_STATS_ENABLE=1 \
    FD_MOE_STATS_PATH="$stats" \
    MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    MAX_NUM_SEQS="$MAX_NUM_SEQS" \
    MAX_NUM_BATCHED_TOKENS="$MAX_NUM_BATCHED_TOKENS" \
    API_PORT="$API_PORT" \
    ENGINE_WORKER_QUEUE_PORT="$ENGINE_PORT" \
    bash "$ROOT/offline_exp/scripts/run_glm.sh" >"$service_log" 2>&1 &
  service_pid=$!
  for _ in $(seq 1 180); do
    if curl -fsS --max-time 2 "http://127.0.0.1:${API_PORT}/health" >/dev/null 2>&1; then
      return 0
    fi
    if ! kill -0 "$service_pid" 2>/dev/null; then
      return 1
    fi
    sleep 5
  done
  return 1
}

run_one() {
  local dataset="$1"
  local infer_log="$LOG_DIR/${dataset}.infer.log"
  local stats="$RUN_DIR/stats/${dataset}.npz"
  local inference_ok=0
  local expected="${EXPECTED[$dataset]:-0}"
  local limit_args=()
  if [[ "$LIMIT" -gt 0 ]]; then
    expected="$LIMIT"
    limit_args+=(--limit "$LIMIT")
  fi
  printf '\n## %s\n' "$dataset" >> "$STATUS"
  if ! start_service "$dataset"; then
    printf -- '- status: service_start_failed\n- log: `%s`\n' "$LOG_DIR/${dataset}.service.stdout.log" >> "$STATUS"
    # Do not kill an unrelated service when the preflight found the port occupied.
    if [[ -n "$service_pid" ]]; then
      stop_service
    fi
    return 0
  fi
  if timeout --foreground "$DATASET_TIMEOUT" "$PYTHON_BIN" "$ROOT/offline_exp/eval/run_infer.py" \
      --dataset "$dataset" --tag "$TAG" \
      --base-url "http://127.0.0.1:${API_PORT}/v1" \
      --model "$MODEL" --workers "$DATASET_WORKERS" \
      --max-tokens "$MAX_TOKENS" --temperature 0 \
      --timeout "$REQUEST_TIMEOUT" --truncate-retries "$TRUNCATE_RETRIES" \
      "${limit_args[@]}" \
      >"$infer_log" 2>&1; then
    printf -- '- status: inference_ok\n' >> "$STATUS"
    inference_ok=1
  else
    printf -- '- status: inference_failed (skipped; next dataset continues)\n' >> "$STATUS"
  fi
  if [[ "$inference_ok" -eq 1 ]] && save_stats "$dataset"; then
    printf -- '- stats_status: saved\n' >> "$STATUS"
  elif [[ "$inference_ok" -eq 0 ]]; then
    printf -- '- stats_status: not_attempted\n' >> "$STATUS"
  else
    printf -- '- stats_status: save_failed\n' >> "$STATUS"
  fi
  local produced=0
  produced=$(valid_records "$RUN_DIR/output/${dataset}.jsonl")
  if [[ "$produced" -lt "$expected" ]]; then
    printf -- '- records: %s/%s (incomplete; scoring result is not treated as full-dataset result)\n' \
      "$produced" "$expected" >> "$STATUS"
  else
    printf -- '- records: %s/%s\n' "$produced" "$expected" >> "$STATUS"
  fi
  stop_service
  if [[ "$inference_ok" -eq 1 && "$produced" -ge "$expected" && -s "$RUN_DIR/output/${dataset}.jsonl" ]]; then
    local score_log="$LOG_DIR/${dataset}.judge.log"
    if [[ "$dataset" == "humaneval" || "$dataset" == "livecodebench" || "$dataset" == "mbpp" ]]; then
      if "$PYTHON_BIN" "$ROOT/offline_exp/eval/judge_code.py" --tag "$TAG" --dataset "$dataset" \
          --workers 32 --lcb-release 6 --lcb-timeout 6 >"$score_log" 2>&1; then
        printf -- '- scoring: ok\n' >> "$STATUS"
      else
        printf -- '- scoring: failed (recorded; next dataset continues)\n' >> "$STATUS"
      fi
    else
      if "$PYTHON_BIN" "$ROOT/offline_exp/eval/judge.py" --tag "$TAG" --dataset "$dataset" \
          --workers 32 --timeout 180 >"$score_log" 2>&1; then
        printf -- '- scoring: ok\n' >> "$STATUS"
      else
        printf -- '- scoring: failed (recorded; next dataset continues)\n' >> "$STATUS"
      fi
    fi
  else
    printf -- '- scoring: skipped (inference failed or dataset incomplete)\n' >> "$STATUS"
  fi
  printf -- '- inference_log: `%s`\n- stats: `%s`\n- fd_log: `%s`\n' "$infer_log" "$stats" "$FD_LOG_ROOT/${dataset}_fd_log" >> "$STATUS"
}

trap stop_service EXIT INT TERM
for dataset in $DATASETS; do
  run_one "$dataset"
done

printf '\nCompleted queue. Failed datasets are listed in this file and do not abort later datasets.\n' >> "$STATUS"
