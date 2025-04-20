# https://mer.vin/2025/01/ollama-reasoning-chatbot-code/

import ollama

# Create streaming completion
completion = ollama.chat(
    model="deepseek-r1:latest",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Why sky is blue?"}
    ],
    stream=True  # Enable streaming
)

# Print the response as it comes in
for chunk in completion:
    if 'message' in chunk and 'content' in chunk['message']:
        content = chunk['message']['content']
        print(content, end='', flush=True)