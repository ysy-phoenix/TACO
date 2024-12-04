#!/bin/bash

# 确保 vllm serve 命令在 PATH 中
# 确保 codellama/CodeLlama-7b-hf 模型已下载到本地

MODEL_NAME="Qwen/Qwen2.5-Coder-32B-Instruct"
BASE_PORT=8000

for GPU_ID in {0..7}; do
  PORT=$((BASE_PORT + GPU_ID))
  echo "Starting vLLM serve on GPU $GPU_ID at port $PORT"
  
  # 启动 vllm serve
  CUDA_VISIBLE_DEVICES=$GPU_ID vllm serve $MODEL_NAME \
    --port $PORT \
    --disable-log-requests \
    &
done

echo "All vLLM serve instances started."