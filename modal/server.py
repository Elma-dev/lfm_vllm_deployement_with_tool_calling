from asyncio import subprocess
import modal
import modal

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.19.0")
    .uv_pip_install("transformers==5.5.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_file("lfm_tool_parser.py", "/app/lfm_tool_parser.py")
    .add_local_file("tool_chat_template_lfm.jinja", "/app/tool_chat_template_lfm.jinja")
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("lfm-vllm-inference")

N_GPUS = 1
MINUTE = 60
VLLM_PORT = 8000
FAST_BOOT = False


@app.function(
    image=vllm_image,
    gpu=f"A10:{N_GPUS}",
    scaledown_window=3 * MINUTE,
    timeout=10 * MINUTE,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=100)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTE)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "LiquidAI/LFM2.5-1.2B-instruct",
        "--max-model-len",
        "4096",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--chat-template",
        "/app/tool_chat_template_lfm.jinja",
        "--enable-auto-tool-choice",
        "--tool-parser-plugin",
        "/app/lfm_tool_parser.py",
        "--tool-call-parser",
        "lfm",
        "--trust-remote-code",
        "--uvicorn-log-level=info",
        "--async-scheduling",
    ]
    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]
    subprocess.Popen(" ".join(cmd), shell=True)
