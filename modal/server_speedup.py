import modal
import requests
import time
import subprocess


vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.19.0")
    .uv_pip_install("transformers==5.5.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    .env(
        {"VLLM_SERVER_DEV_MODE": "1"}
    )  # this one is so important to activate vllm sleep mode
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
REGION = "us-east"

with vllm_image.imports():
    import requests


# ------------- Snapshot Utils -----------------------------#
def check_running(p: subprocess.Popen):
    if (rc := p.poll()) is not None:
        raise subprocess.CalledProcessError(rc, cmd=p.args)


def wait_ready(process: subprocess, timeout: int = 15 * 60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            check_running(process)
            requests.get(f"http://127.0.0.1:8000/health").raise_for_status()
            return
        except (
            subprocess.CalledProcessError,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
        ):
            time.sleep(5)
    raise TimeoutError(f"vLLM server not ready within {timeout} seconds")


def warmup():
    payload = {
        "model": "llm",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "max_tokens": 16,
    }

    for _ in range(3):
        requests.post(
            f"http://127.0.0.1:8000/v1/chat/completions",
            json=payload,
            timeout=60,
        ).raise_for_status()


def sleep(level: int = 1):
    requests.post(f"http://127.0.0.1:8000/sleep?level={level}").raise_for_status()


def wake_up():
    requests.post("http://127.0.0.1:8000/wake_up").raise_for_status()


# -----------------------------------------------------------#


@app.cls(
    image=vllm_image,
    gpu=f"A10:{N_GPUS}",
    scaledown_window=2 * MINUTE,
    timeout=10 * MINUTE,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    enable_memory_snapshot=True,  # CPU snapshot
    experimental_options={"enable_gpu_snapshot": True},  # GPU snapshot,
    min_containers=0,
    max_containers=2,
    region=REGION,
)
@modal.concurrent(max_inputs=50)
# @modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTE)
# @modal.experimental.http_server(port=VLLM_PORT, startup_timeout=10 * MINUTE)
class LfmVllmInferenceSpeedUp:
    @modal.enter(snap=True)
    def startup(self):
        """Start the vLLM server and block until it is healthy, then warm it up and put it to sleep."""
        import subprocess

        cmd = [
            "vllm",
            "serve",
            "LiquidAI/LFM2.5-1.2B-instruct",
            "--max-model-len",
            "4096",
            "--served-model-name",
            "LiquidAI/LFM2.5-1.2B-instruct",
            "--served-model-name",
            "llm",
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
            "--enable-sleep-mode",  # important for vllm sleep mode
        ]
        cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]
        self.process = subprocess.Popen(" ".join(cmd), shell=True)
        wait_ready(self.process)
        warmup()
        sleep(level=1)

    @modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTE)
    def serve(self):
        pass

    @modal.enter(snap=False)
    def restore(self):
        """Wake vLLM from sleep mode after restoring from a memory snapshot."""
        wake_up()

    @modal.exit()
    def stop(self):
        self.process.terminate()
