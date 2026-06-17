from taras import config
from taras.llm import chat, parse_json_from_response
from taras.logging_utils import append_run_log, log_prompt_section


def judge_answer(utterance, model_answer, gold_answer):
    prompt = f"""{{
"Persona": "You are a teacher who is grading students' homework.",
"Instructions": [
    "Given a question and its answer, the question is clear and unambiguous, and the value of the answer is also unique. But the answer does not require any format, so the way to express it can be various.",
    "What you need to do is to judge whether the candidate answer is also correct without considering the format.",
    "Note: Please DO NOT doubt the authenticity of the correct answer, you should consider it absolutely correct."
],
"Question": "{utterance}",
"Candidate Answer": "{model_answer}",
"Correct Answer": "{gold_answer}",
"Output Format": "Your output should first contain a detailed analysis process. After that, attach your result in the following JSON format: {{"judgement": "yes" or "no"}}."
}}"""
    log_prompt_section("----------Judge Answer----------", prompt)

    model = config.args.engine
    messages = [
        {
            "role": "system",
            "content": "You are a teacher who is grading students' homework."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    retries = 0
    while retries < config.args.max_retries:
        retries += 1
        try:
            response = chat(model, messages)
            answer_json = parse_json_from_response(response)

            judgement = answer_json["judgement"]
            if judgement not in {"yes", "no"}:
                continue

            break
        except Exception as e:
            pass
            # print("Judging: ", str(e) or repr(e))
    if retries >= config.args.max_retries:
        judgement = "no"

    append_run_log("----------Response----------\n" + response + '\n')

    if judgement == "yes":
        return gold_answer
    else:
        return model_answer


def resolve_final_output(final_result, value, dataset, utterance):
    if str(final_result).isdigit() and str(value).isdigit():
        if str(final_result) == str(value):
            return value
        else:
            return final_result
    else:
        if final_result != "Analysis Fail":
            if dataset == 'FeTaQA':
                return final_result
            else:
                return judge_answer(utterance, final_result, value)
        else:
            return "Analysis Fail"
