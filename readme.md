1. install requirements

```
uv pip install -r requirements.py
```

2. run server:

using server.py

```
uv run server.py
```

or dirce CLI command:

```
vllm serve --model LiquidAI/LFM2.5-1.2B-instruct --max-model-len 2048 --host 0.0.0.0 --port 8000 --chat-template ./tool_chat_template_lfm
.jinja --enable-auto-tool-choice --tool-parser-plugin ./lfm_tool_parser.py --tool-call-parser lfm --trust-remote-code 
```
