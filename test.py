from openai import OpenAI
import json

base_url = "https://naweowphwim9.shares.zrok.io/v1"
key = "empty"

client = OpenAI(base_url=base_url, api_key=key)

# Get the first available model
model_id = client.models.list().data[0].id
print(f"Using model: {model_id}")

# Define sample tool (function)
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

# Run tool calling test
print("\n--- Running tool calling test ---")
response = client.chat.completions.create(
    model=model_id,
    messages=[
        {"role": "user", "content": "What's the weather like in Agadir and Rabat?"}
    ],
    tools=tools,
    tool_choice="auto",
    temperature=0.1,
)

message = response.choices[0].message

print(f"Role: {message.role}")
print(f"Content: {message.content}")

# Mock tool execution
def get_weather(location, unit="celsius"):
    """Mock weather service"""
    return {"location": location, "temperature": "22", "unit": unit, "description": "Sunny"}

if message.tool_calls:
    print("\n✅ Tool calls detected:")
    messages = [{"role": "user", "content": "What's the weather like in Agadir and Rabat?"}]
    messages.append(message)  # Add assistant message with tool calls

    for tool_call in message.tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        print(f"  - Executing: {function_name}({function_args})")
        
        if function_name == "get_weather":
            function_response = get_weather(
                location=function_args.get("location"),
                unit=function_args.get("unit", "celsius")
            )
            
            # Add tool response message
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(function_response),
                }
            )

    # Get final response from the model
    print("\n--- Getting final response from model ---")
    final_response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        tools=tools,
    )
    
    print(f"Final response: {final_response.choices[0].message.content}")
else:
    print("\n⚠️ No tool calls detected. Raw message:")
    print(message)
