# Use the official vLLM OpenAI-compatible server image
FROM vllm/vllm-openai:latest

# Set working directory
WORKDIR /app

# Copy requirements and install them if they change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the custom tool parser and chat template
COPY lfm_tool_parser.py .
COPY tool_chat_template_lfm.jinja .

# RunPod default port is often 8000
ENV PORT=8000
ENV MODEL_NAME=LiquidAI/LFM2.5-1.2B-instruct
ENV MAX_MODEL_LEN=2048

# Expose the port
EXPOSE ${PORT}

# Use shell form so environment variables are expanded
ENTRYPOINT ["/bin/sh", "-c"]
CMD ["python3 -m vllm.entrypoints.openai.api_server \
  --model ${MODEL_NAME} \
  --max-model-len ${MAX_MODEL_LEN} \
  --host 0.0.0.0 \
  --port ${PORT} \
  --chat-template ./tool_chat_template_lfm.jinja \
  --enable-auto-tool-choice \
  --tool-parser-plugin ./lfm_tool_parser.py \
  --tool-call-parser lfm \
  --trust-remote-code \
  ${EXTRA_ARGS}"]
