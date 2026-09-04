#!/bin/bash
# 打请求验证 AFD 链路，并统计请求落在了哪些实例上
# 用法:
#   bash test.sh                      # 1 个请求
#   N=20 bash test.sh                 # 20 个并发请求
#   N=20 MAX_TOKENS=128 bash test.sh "你的 prompt"

ROUTER="${ROUTER:-127.0.0.1:9123}"
PROMPT="${1:-用一句话解释什么是注意力机制}"
MAX_TOKENS="${MAX_TOKENS:-64}"
N="${N:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT="${LOG_ROOT:-$(ls -td "${SCRIPT_DIR}"/run_log/*/ 2>/dev/null | head -1)}"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

echo "[test] router: ${ROUTER}, concurrency: ${N}, max_tokens: ${MAX_TOKENS}"
curl -s -m 5 "http://${ROUTER}/registered_number" || { echo "[test] router 不可达"; exit 1; }
echo

start=$(date +%s.%N)
for i in $(seq 1 "${N}"); do
  curl -s -m 300 -X POST "http://${ROUTER}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$(printf '{"model":"default","messages":[{"role":"user","content":"%s (#%s)"}],"max_tokens":%s,"stream":false}' "${PROMPT}" "${i}" "${MAX_TOKENS}")" \
    -o "${TMP}/resp_${i}.json" -w "%{http_code}" >"${TMP}/code_${i}" &
done
wait
end=$(date +%s.%N)

ok=0
for i in $(seq 1 "${N}"); do
  [ "$(cat "${TMP}/code_${i}")" = "200" ] && ok=$((ok + 1))
done
grep -ho '"id": *"chatcmpl-[^"]*"' "${TMP}"/resp_*.json | grep -o 'chatcmpl-[a-f0-9-]*' | sort -u >"${TMP}/ids"

echo "[test] 200: ${ok}/${N}, elapsed: $(echo "${end} - ${start}" | bc)s"
if [ "${N}" = "1" ]; then
  python -m json.tool "${TMP}/resp_1.json" 2>/dev/null || cat "${TMP}/resp_1.json"
fi
[ "${ok}" = "${N}" ] || { echo "[test] FAILED"; exit 1; }

if [ -z "${LOG_ROOT}" ] || [ ! -d "${LOG_ROOT}" ]; then
  echo "[test] OK (未找到 run_log 目录，跳过分布统计)"
  exit 0
fi

count_ids() {
  # $1: 日志文件；统计本次请求 id 在其中作为 HTTP 入口出现的个数
  local f="$1" n=0
  [ -f "${f}" ] || { echo 0; return; }
  while read -r id; do
    grep -q "create chat completion request: ${id}" "${f}" && n=$((n + 1))
  done <"${TMP}/ids"
  echo "${n}"
}

echo "[test] 请求分布 (log root: ${LOG_ROOT})"
echo "  -- decode / attn 实例"
for d in attn_0 attn_1; do
  echo "     ${d}: $(count_ids "${LOG_ROOT}/${d}/api_server.log")"
done
echo "  -- prefill DP rank"
for i in 0 1 2 3; do
  [ -d "${LOG_ROOT}/prefill/log_${i}" ] || continue
  echo "     dp${i}: $(count_ids "${LOG_ROOT}/prefill/log_${i}/api_server.log")"
done
echo "[test] OK"
