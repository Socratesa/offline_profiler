#!/bin/bash
# Stop the GLM api server without letting pgrep patterns match this script's own
# cmdline (which is what makes inline `pkill -f ...` kill its own shell).
PAT='fastdeploy.entrypoints.openai.api_server'
mapfile -t PIDS < <(pgrep -f "$PAT")
echo "pids: ${PIDS[*]:-none}"
for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null; done
for _ in $(seq 1 45); do
  sleep 2
  pgrep -f "$PAT" >/dev/null || break
done
mapfile -t LEFT < <(pgrep -f "$PAT")
if [[ ${#LEFT[@]} -gt 0 ]]; then
  echo "force killing: ${LEFT[*]}"
  for p in "${LEFT[@]}"; do kill -KILL "$p" 2>/dev/null; done
  sleep 5
fi
echo "remaining: $(pgrep -c -f "$PAT" || echo 0)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tail -4
