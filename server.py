import subprocess, time, os

# Kill any existing vLLM server
os.system("pkill -f 'vllm.entrypoints' 2>/dev/null")
time.sleep(2)

cmd = [
    "python",
    "-m",
    "vllm.entrypoints.openai.api_server",
    "--model",
    "LiquidAI/LFM2.5-1.2B-Instruct",  # ← swap to your model
    "--host",
    "0.0.0.0",
    "--port",
    "8000",
    "--dtype",
    "auto",
    "--max-model-len",
    "4096",
    "--chat-template",
    "/content/tool_chat_template_lfm.jinja",
    "--enable-auto-tool-choice",
    "--tool-parser-plugin",
    "/content/lfm_tool_parser.py",  # ← plugin file
    "--tool-call-parser",
    "lfm",  # ← registered name
    "--trust-remote-code",
]

log = open("/content/vllm_server.log", "w")
proc = subprocess.Popen(cmd, stdout=log, stderr=log)

print("⏳ Waiting for server to start...")
for i in range(60):
    time.sleep(3)
    try:
        import urllib.request

        urllib.request.urlopen("http://localhost:8000/health")
        print(f"✅ Server ready after {(i + 1) * 3}s")
        break
    except Exception:
        print(f"  ... {(i + 1) * 3}s", end="\r")
else:
    print("❌ Server didn't start — check /content/vllm_server.log")
