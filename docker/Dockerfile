# Use the official vLLM OpenAI-compatible server image
FROM vllm/vllm-openai:latest

# Set working directory
WORKDIR /app

# Copy requirements and install them
# Note: vLLM is already installed in the base image, 
# but we install additional dependencies if any.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the tool parser and chat template
COPY lfm_tool_parser.py .
COPY tool_chat_template_lfm.jinja .

# Expose the port vLLM will run on
EXPOSE 8000

# Set environment variables if needed
ENV MODEL_NAME=LiquidAI/LFM2.5-1.2B-instruct

# Command to run the server
# We use the arguments identified in your README
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
CMD ["--model", "LiquidAI/LFM2.5-1.2B-instruct", \
     "--max-model-len", "2048", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--chat-template", "./tool_chat_template_lfm.jinja", \
     "--enable-auto-tool-choice", \
     "--tool-parser-plugin", "./lfm_tool_parser.py", \
     "--tool-call-parser", "lfm", \
     "--trust-remote-code"]

