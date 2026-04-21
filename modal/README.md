# Modal Deployment for vLLM

This directory contains the configuration and scripts required to deploy a high-performance vLLM server on [Modal](https://modal.com). The setup is specifically optimized for the Liquid Foundation Model (LFM) and includes integrated tool-calling capabilities.

## Prerequisites

Before deploying, ensure you have the Modal client installed and your environment authenticated.

### 1. Install the Modal Client
We recommend using `uv` for efficient dependency management:

```bash
uv pip install modal
```

### 2. Authentication
Authenticate your local environment with your Modal account:

```bash
modal setup
```

### 3. Move required files:
```bash
cp ../tool_chat_template_lfm.jinja .
cp ../lfm_tool_parser.py .          
```

## Deployment

To launch the vLLM server as a live development endpoint, execute:

```bash
modal serve server.py
```

For a permanent deployment, use `modal deploy server.py`.

## Implementation Details

The `server.py` script manages the entire infrastructure lifecycle:

- **Runtime Environment**: Constructs a custom container based on the `nvidia/cuda:12.9.0-devel-ubuntu22.04` image, pre-configured with Python 3.12.
- **Dependency Isolation**: Installs performance-critical libraries including `vllm`, `transformers`, and custom local plugins.
- **Efficient Caching**: Utilizes Modal Volumes (`huggingface-cache` and `vllm-cache`) to persist model weights across sessions, significantly reducing cold-start times.
- **Hardware Configuration**: Provisions NVIDIA A10 GPUs and optimizes container parameters such as `scaledown_window` and `startup_timeout`.
- **Advanced vLLM Configuration**:
    - **Model**: `LiquidAI/LFM2.5-1.2B-instruct`
    - **Tool Integration**: Loads a custom tool-calling parser (`lfm_tool_parser.py`) and a specialized chat template (`tool_chat_template_lfm.jinja`).
    - **Performance**: Enables asynchronous scheduling and supports large context windows (up to 4096 tokens).