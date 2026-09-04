#!/bin/bash

source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate fd_final

rm -rf "${FD_LOG_DIR:-log_prefill}"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
FD_USE_PFCC_DEEP_EP=1 \
FD_LOG_DIR="${FD_LOG_DIR:-log_prefill}" \
python -m fastdeploy.entrypoints.openai.multi_api_server \
  --num-servers 4 --ports 1610,1611,1612,1613 \
  --args \
  --model /workspace/models/GLM-4.5-Air \
  --tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel \
  --splitwise-role prefill --router "0.0.0.0:9123" \
  --cache-transfer-protocol "rdma" \
  --gpu-memory-utilization 0.9 \
  --enable-chunked-prefill \
  --max-num-seqs 512 \
  --max-num-batched-tokens 16384 \
  --max-model-len 32768