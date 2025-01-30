FROM vllm/vllm-openai:latest

# ENV MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
# VLLM_ALLOW_RUNTIME_LORA_UPDATING=true

ENV MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

ENTRYPOINT python3 -m vllm.entrypoints.openai.api_server --model $MODEL_NAME $VLLM_ARGS