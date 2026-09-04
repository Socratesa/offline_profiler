#!/usr/bin/env bash
set -u
# AFD 模式下只跑评测集：向 router 打请求，收集输出，不打分（后续单独跑 judge）
# 用法:
#   bash run_afd_eval_datasets.sh              # 复用已在跑的服务
#   bash run_afd_eval_datasets.sh --restart    # 先 kill 再重启全部服务
# 参数默认与 offline_exp/scripts/run_all_glm_datasets.sh 保持一致

ROOT="/workspace/fd_afd_dev"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/workspace/miniconda3/envs/fd_final/bin/python}"
TAG="${TAG:-afd_glm_think_32k_c512_$(date +%Y%m%d)}"
RUN_DIR="$ROOT/offline_exp/runs/$TAG"
LOG_DIR="$RUN_DIR/logs"
MODEL="${MODEL:-/workspace/models/GLM-4.5-Air}"
ROUTER="${ROUTER:-127.0.0.1:9123}"
BASE_URL="${BASE_URL:-http://${ROUTER}/v1}"
DATASET_WORKERS="${DATASET_WORKERS:-512}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-14400}"
DATASET_TIMEOUT="${DATASET_TIMEOUT:-172800}"
TRUNCATE_RETRIES="${TRUNCATE_RETRIES:-0}"
LIMIT="${LIMIT:-0}"
# 只跑评测集，校准集不跑
DATASETS="${DATASETS:-gsm8k math500 humaneval livecodebench financebench medqa_usmle mmlu_pro_law}"
STATUS="$LOG_DIR/run_afd_eval_status.md"
RESTART=0
[[ "${1:-}" == "--restart" ]] && RESTART=1

declare -A EXPECTED=(
  [gsm8k]=1319 [math500]=500 [humaneval]=164 [livecodebench]=1055
  [financebench]=150 [medqa_usmle]=1273 [mmlu_pro_law]=1101
)

mkdir -p "$RUN_DIR/output" "$RUN_DIR/score" "$RUN_DIR/code" "$LOG_DIR"

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

router_ready() {
  local n
  n="$(curl -s -m 5 "http://${ROUTER}/registered_number" 2>/dev/null)" || return 1
  [[ -n "$n" ]] || return 1
  echo "$n" | grep -q '"prefill":0' && return 1
  echo "$n" | grep -q '"attn":0' && return 1
  echo "$n" | grep -q '"ffn":0' && return 1
  echo "$n"
}

ensure_service() {
  if [[ "$RESTART" -eq 1 ]]; then
    echo "[afd_eval] --restart: kill + 重启全部服务"
    bash "$SCRIPT_DIR/run_all.sh" || return 1
    return 0
  fi
  if router_ready >/dev/null; then
    echo "[afd_eval] 复用已在跑的服务: $(router_ready)"
  else
    echo "[afd_eval] 服务未就绪，启动全部服务"
    bash "$SCRIPT_DIR/run_all.sh" || return 1
  fi
  # 服务在跑也要确认能正常出结果
  echo "[afd_eval] smoke test"
  bash "$SCRIPT_DIR/test.sh" >"$LOG_DIR/smoke_test.out" 2>&1 || {
    echo "[afd_eval] smoke test 失败，见 $LOG_DIR/smoke_test.out"
    tail -20 "$LOG_DIR/smoke_test.out"
    return 1
  }
  echo "[afd_eval] smoke test OK"
}

run_one() {
  local dataset="$1"
  local infer_log="$LOG_DIR/${dataset}.infer.log"
  local expected="${EXPECTED[$dataset]:-0}"
  local limit_args=()
  if [[ "$LIMIT" -gt 0 ]]; then
    expected="$LIMIT"
    limit_args+=(--limit "$LIMIT")
  fi
  printf '\n## %s\n' "$dataset" >>"$STATUS"

  if ! router_ready >/dev/null; then
    printf -- '- status: router_not_ready (skipped)\n' >>"$STATUS"
    echo "[afd_eval] ${dataset}: router 不可用，跳过"
    return 0
  fi

  echo "[afd_eval] ${dataset}: 开始推理 (workers=${DATASET_WORKERS})"
  if timeout --foreground "$DATASET_TIMEOUT" "$PYTHON_BIN" "$ROOT/offline_exp/eval/run_infer.py" \
      --dataset "$dataset" --tag "$TAG" \
      --base-url "$BASE_URL" \
      --model "$MODEL" --workers "$DATASET_WORKERS" \
      --max-tokens "$MAX_TOKENS" --temperature 0 \
      --timeout "$REQUEST_TIMEOUT" --truncate-retries "$TRUNCATE_RETRIES" \
      "${limit_args[@]}" \
      >"$infer_log" 2>&1; then
    printf -- '- status: inference_ok\n' >>"$STATUS"
  else
    printf -- '- status: inference_failed (skipped; next dataset continues)\n' >>"$STATUS"
  fi

  local produced=0
  produced=$(valid_records "$RUN_DIR/output/${dataset}.jsonl")
  if [[ "$produced" -lt "$expected" ]]; then
    printf -- '- records: %s/%s (incomplete)\n' "$produced" "$expected" >>"$STATUS"
  else
    printf -- '- records: %s/%s\n' "$produced" "$expected" >>"$STATUS"
  fi
  printf -- '- inference_log: `%s`\n- output: `%s`\n' \
    "$infer_log" "$RUN_DIR/output/${dataset}.jsonl" >>"$STATUS"
  echo "[afd_eval] ${dataset}: ${produced}/${expected}"
}

printf '# AFD eval datasets status\n\n- tag: `%s`\n- router: `%s`\n- workers: %s\n- max_tokens: %s\n' \
  "$TAG" "$ROUTER" "$DATASET_WORKERS" "$MAX_TOKENS" >"$STATUS"

if ! ensure_service; then
  printf -- '\n- status: service_not_ready, aborted\n' >>"$STATUS"
  echo "[afd_eval] 服务不可用，退出"
  exit 1
fi

for dataset in $DATASETS; do
  run_one "$dataset"
done

printf '\n推理完成。打分请单独跑 offline_exp/eval/judge.py / judge_code.py --tag %s\n' "$TAG" >>"$STATUS"
echo "[afd_eval] 全部完成，状态: $STATUS"
