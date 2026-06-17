import os
import threading

import ollama

os.environ["OLLAMA_NUM_PARALLEL"] = "9"

from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY", base_url="YOUR_OPENAI_BASE_URL")

calling_times = 0
calling_times_lock = threading.Lock()
input_tokens = 0
output_tokens = 0
token_lock = threading.Lock()
file_lock = threading.Lock()

args = None
