#!/bin/bash

source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate fd_final

CUDA_VISIBLE_DEVICES=5 \
FD_ENABLE_EP_CONTROL_REQ_POLL=1 \
FD_ENABLE_CPU_GROUP=1 \
FD_MOE_A2A_BACKEND=mooncake \
FD_LOG_DIR="${FD_LOG_DIR:-log_attn_1}" \
python -m fastdeploy.entrypoints.openai.api_server \
  --port 1713 --model /workspace/models/GLM-4.5-Air \
  --tensor-parallel-size 1 --data-parallel-size 1 --enable-expert-parallel \
  --splitwise-role decode --router "0.0.0.0:9123" \
  --afd-role attn --afd-ninsts 3 --afd-inst-rank 2 \
  --enable-eplb --eplb-config '{"redundant_experts_num": 16, "redundant_expert_async_load_model_shmem_size_gb": 16}' \
  --ips "10.63.64.227,10.63.64.227,10.63.64.227" --launch-port 29600 \
  --cache-transfer-protocol "rdma" \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 512 \
  --max-num-batched-tokens 16384 \
  --max-model-len 32768