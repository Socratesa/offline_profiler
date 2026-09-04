#!/bin/bash
# 一键启动 AFD 全部服务：router(可复用) + ffn + attn_0 + attn_1 + prefill
# 本次启动的所有日志都在 run_log/<时间戳>/ 下

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${LOG_ROOT:-${SCRIPT_DIR}/run_log/${RUN_TAG}}"
ROUTER_PORT=9123
API_PORTS="1610 1611 1612 1613 1711 1712 1713"

mkdir -p "${LOG_ROOT}"
echo "[run_all] log root: ${LOG_ROOT}"

port_listening() {
  (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null
}

# gunicorn 会把 proctitle 改成 "gunicorn: master/worker"，pkill -f fastdeploy 匹配不到
api_port_owners() {
  local p
  for p in ${API_PORTS}; do
    ss -lntpH "sport = :${p}" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2
  done | sort -u | tr '\n' ' '
}

# 1. 先杀掉现有服务。fd-router 的进程名不含 fastdeploy，不会被误杀
echo "[run_all] pkill -f fastdeploy"
pkill -f "fastdeploy"
pkill -f "paddle.distributed.launch"
pkill -9 -f "fastdeploy|api_server|worker_process"

sleep 10

# 2. 按端口兜底清理漏杀的残留进程，最多等 1 分钟
for sig in TERM TERM KILL; do
  leftover="$(api_port_owners)"
  [ -z "${leftover// /}" ] && break
  echo "[run_all] kill -${sig} leftover on api ports: ${leftover}"
  kill -"${sig}" ${leftover} 2>/dev/null
  sleep 20
done

leftover="$(api_port_owners)"
if [ -n "${leftover// /}" ]; then
  echo "[run_all] api ports still occupied by: ${leftover}, abort"
  exit 1
fi

# 3. router 已在监听就直接复用
if port_listening "${ROUTER_PORT}"; then
  echo "[run_all] router already on ${ROUTER_PORT}, reuse it"
else
  echo "[run_all] starting router on ${ROUTER_PORT}"
  nohup bash "${SCRIPT_DIR}/run_router.sh" >"${LOG_ROOT}/router.out" 2>&1 &
  for _ in $(seq 60); do
    port_listening "${ROUTER_PORT}" && break
    sleep 1
  done
  port_listening "${ROUTER_PORT}" || {
    echo "[run_all] router failed to start, see ${LOG_ROOT}/router.out"
    exit 1
  }
fi

# 4. 启动各实例：FD_LOG_DIR 指到本次时间戳目录，stdout/stderr 落在同一目录
start_service() {
  local name="$1" script="$2"
  echo "[run_all] starting ${name} (${script})"
  FD_LOG_DIR="${LOG_ROOT}/${name}" nohup bash "${SCRIPT_DIR}/${script}" >"${LOG_ROOT}/${name}.out" 2>&1 &
}

# ffn 与两个 attn 要在 launch-port 29600 汇合，必须同时起
start_service ffn run_ffn.sh
start_service attn_0 run_attn_0.sh
start_service attn_1 run_attn_1.sh
start_service prefill run_prefill.sh

# 5. 等各 api_server 端口就绪（加载模型较慢，最多等 30 分钟）
echo "[run_all] waiting for api ports:${API_PORTS}"
pending=""
for _ in $(seq 180); do
  pending=""
  for p in ${API_PORTS}; do
    port_listening "${p}" || pending="${pending} ${p}"
  done
  [ -z "${pending}" ] && break
  sleep 10
done

if [ -n "${pending}" ]; then
  echo "[run_all] still not listening:${pending}"
  echo "[run_all] check ${LOG_ROOT}/*.out and ${LOG_ROOT}/<service>/"
  echo "[run_all] logs: ${LOG_ROOT}"
  exit 1
fi
echo "[run_all] all services up"
echo "[run_all] logs: ${LOG_ROOT}"

# 6. 冒烟测试：端口起来不等于能正常出结果，必须真打一个请求
echo "[run_all] smoke test"
if LOG_ROOT="${LOG_ROOT}" bash "${SCRIPT_DIR}/test.sh" >"${LOG_ROOT}/smoke_test.out" 2>&1; then
  echo "[run_all] smoke test OK"
else
  echo "[run_all] smoke test FAILED, see ${LOG_ROOT}/smoke_test.out"
  tail -20 "${LOG_ROOT}/smoke_test.out"
  exit 1
fi
