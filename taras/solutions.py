from taras import config
from taras.llm import ANALYST_SYSTEM, build_messages
from taras.logging_utils import log_prompt_section
from taras.parallel_utils import run_parallel
from taras.solvers import code_solver, direct_solver


def code_solution(prompt_tables, utterance, table_file_path):
    num_p = config.args.num_p

    prompt = f"""{{
"Persona": "You are a tabular data analyst, you are proficient in using code to solve tabular data analysis problem.",
"Instructions": [
    "Given a table preview, you need to write PYTHON CODE to solve the question.",
    "When writting code, assume the data was stored in a file called 'PATH.csv', directly use "pd.read_csv('PATH.csv')" to load it.",
    "At last, store your result in 'analysis_answer'. DO NOT use print functions in your code, I only get the answer from that variable."
    ],
"Data": {{
    "Table": "{prompt_tables}",
    "Question": "{utterance}"
}}
}}"""
    log_prompt_section("----------Code Solution----------", prompt)

    model = config.args.engine
    messages = build_messages(
    "You are a tabular data analyst, you are proficient in writing python code to solve tabular data analysis problem.",
        prompt,
    )

    return run_parallel(
        lambda: code_solver(model, messages, table_file_path),
        num_p,
        num_p,
    )


def direct_solution(num_p, prompt_tables, utterance):
    prompt = f"""{{
"Persona": "You are a tabular data analyst, you are proficient in solving tabular data analysis problem.",
"Instructions": [
    "Given a table preview, You need to judge whether the question can be answered or not using the preview I give you. If yes, you need to give the answer."
],
"Output Format": "Your output should first contain a detailed analysis process. After that, attach your result in the following JSON format: {{"judgement": "can be answered" or "can't be answered", "answer": ""}}.",
"Data": {{
    "Table": "{prompt_tables}",
    "Question": "{utterance}"
}}
}}"""
    log_prompt_section(f"----------Direct Solution {num_p}----------", prompt)

    model = config.args.engine
    messages = build_messages(ANALYST_SYSTEM, prompt)

    if num_p == 1:
        return [direct_solver(model, messages)]

    return run_parallel(
        lambda: direct_solver(model, messages),
        num_p,
        num_p,
    )
