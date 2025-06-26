# https://mer.vin/2025/01/ollama-reasoning-chatbot-code/
__copyright__ = "Copyright (c) 2025 Hai Liang Wang<hailiang.hl.wang@gmail.com> All Rights Reserved"
__author__ = "Hai Liang Wang"
__date__ = "2025-06-24:21:20:00"

import os, sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, curdir)

if sys.version_info[0] < 3:
    raise RuntimeError("Must be using Python 3")
else:
    unicode = str

import env3, log5
ENV = env3.load_env(os.path.join(curdir, ".env"))
logger = log5.get_logger(log5.LN(__name__), output_mode=log5.OUTPUT_STDOUT)

OLLAMA_BASEURL = ENV.get("OLLAMA_BASEURL", "")
OLLAMA_HEADER_AUTH = f"Basic {ENV.get('OLLAMA_HEADER_AUTH', '')}"

from ollama import Client
client = Client(
  host=OLLAMA_BASEURL,
  headers={'Authorization': OLLAMA_HEADER_AUTH}
)

# Create streaming completion
completion = client.chat(
    model="deepseek-r1:14b",
    
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