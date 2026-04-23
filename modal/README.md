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



# Memory Snapshot: reducing cold start times
- We create a memory snapshot from your Function just before it calls for inputs.
- Your Function is then “frozen”, saved as an optimized format, and cached in our distributed file system. 
- Every time your program cold boots the program starts from this frozen state.
![alt text](image.png)

- **Memory Snapshots:** Instead of starting from scratch, Modal saves the entire CPU and GPU memory state to disk. When a new container starts, it "restores" this state instantly.
- **vLLM Integration:** It uses the vLLM engine but adds a "sleep" and "warmup" cycle to ensure the snapshot is clean and efficient.
- CPU snapshot: `enable_memory_snapshot=True`
- GPU snapshot: `experimental_options={"enable_gpu_snapshot": True}`

3. Implement Lifecycle Hooks (@modal.enter):
This is the most critical part of the snapshot process. You need two "enter" methods:
- `@modal.enter(snap=True)`: * This runs once during the creation of the snapshot.
    * **Action:** Load your SLM weights, start the vLLM engine, and run a "warmup" request (a dummy prompt) to compile CUDA kernels.
    * **Action:** Call a /sleep or similar endpoint to clear temporary GPU buffers before the snapshot is taken.
- `@modal.enter(snap=False)`: This runs every time a container is restored from a snapshot.
    * **Action:** "Wake up" the engine and ensure the network ports are ready to receive traffic.


VLLM Slep:
model.