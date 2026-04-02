from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="LiquidAI/LFM2.5-1.2B-Instruct",
    messages=[
        {"role": "user", "content": "What's the weather in Agadir? and also in rabat?"}
    ],
    tools=tools,
    tool_choice="auto",
    temperature=0.1,
    max_tokens=256,
)

msg = response.choices[0].message
print("finish_reason :", response.choices[0].finish_reason)
print("content       :", msg.content)
print("tool_calls    :", msg.tool_calls)

if msg.tool_calls:
    import json

    tc = msg.tool_calls[0]
    print("\n✅ Tool call extracted successfully!")
    print(f"   name      : {tc.function.name}")
    print(f"   arguments : {json.loads(tc.function.arguments)}")
else:
    print(
        "\n⚠️  No tool_calls — check raw content above for <|tool_call_start|> markers"
    )
