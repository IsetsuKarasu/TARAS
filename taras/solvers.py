import re
import warnings

import pandas as pd

from taras import config
from taras.csv_io import make_exec_pandas
from taras.llm import chat, normalize_tabfact_answer, parse_json_from_response
from taras.logging_utils import log_response


def code_solver(model, messages, table_file_path):
    retries = 0
    while retries < config.args.max_retries:
        retries += 1
        try:
            response = chat(model, messages)
            answer = re.findall(r'\```(.*?)\```', response, flags=re.DOTALL)
            if len(answer) != 1:
                continue

            code = answer[0].strip('```').strip('python').replace("pd.read_csv('PATH.csv')", f"pd.read_csv('{table_file_path}')")

            error = "No Error"
            analysis_answer = "No Answer"
            exec_globals = {'__builtins__': __builtins__, 'pd': make_exec_pandas()}
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    exec(code, exec_globals)
                analysis_answer = exec_globals['analysis_answer']
                if not analysis_answer or isinstance(analysis_answer, (pd.Series, pd.DataFrame)):
                    continue
                analysis_answer = str(analysis_answer)
            except Exception as e:
                error = str(e) or repr(e)

            if error != "No Error":
                output = "Analysis Fail"
            else:
                output = analysis_answer
            break
        except Exception as e:
            pass
            # print("Analysing: ", str(e) or repr(e))

    if retries >= config.args.max_retries:
        output = "Analysis Fail"

    log_response(response, code=code, use_file_lock=True)

    if config.args.dataset == "TabFact":
        output = normalize_tabfact_answer(output)

    return output


def direct_solver(model, messages):
    retries = 0
    while retries < config.args.max_retries:
        retries += 1
        try:
            response = chat(model, messages)
            answer_json = parse_json_from_response(response)

            judgement = answer_json["judgement"]
            if judgement == "can be answered":
                final_answer = str(answer_json['answer'])
                if config.args.dataset == "TabFact":
                    final_answer = final_answer.lower()

                    if final_answer == "true":
                        final_answer = 1
                    elif final_answer == "false":
                        final_answer = 0
                    else:
                        continue
            elif judgement == "can't be answered":
                final_answer = "Analysis Fail"
            else:
                continue

            break
        except Exception as e:
            pass
            # print("Analysing: ", str(e) or repr(e))

    if retries >= config.args.max_retries:
        final_answer = "Analysis Fail"

    log_response(response, use_file_lock=True)

    return final_answer
