import json
import re

import ollama

from taras import config

GPT_MODELS = {"gpt-4o-mini", "gpt-4o-ca"}
ANALYST_SYSTEM = "You are a tabular data analyst, you are proficient in solving tabular data analysis problem."


def chat(model, messages):
    with config.calling_times_lock:
        config.calling_times += 1

    if model in GPT_MODELS:
        output = config.client.chat.completions.create(model=model, messages=messages)
        return output.choices[0].message.content
    else:
        output = ollama.chat(model=model, messages=messages, stream=False)
        with config.token_lock:
            config.input_tokens += output['prompt_eval_count']
            config.output_tokens += output['eval_count']
        return output['message']['content']


def parse_json_from_response(response):
    answer = re.findall(r'\{(.*?)\}', response, flags=re.DOTALL)
    return json.loads('{' + answer[-1] + '}')


def build_messages(system_content, prompt):
    return [
        {
            "role": "system",
            "content": system_content
        },
        {
            "role": "user",
            "content": prompt
        }
    ]


def normalize_tabfact_answer(answer):
    answer = answer.lower()
    if answer == "true":
        return 1
    elif answer == "false":
        return 0
    else:
        return "Analysis Fail"
