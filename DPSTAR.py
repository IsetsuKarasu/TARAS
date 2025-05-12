import argparse
import json
import os
import pandas as pd
import ollama
import re

from openai import OpenAI
client = OpenAI(api_key="ADD YOUR KEY.")


def code_solution(prompt_tables, utterance, table_file_path):
    num_p = args.num_p
    outputs = []

    prompt = f"""{{
    "Persona": "You are a tabular data analyst, you are proficient in writing python code to solve tabular data analysis problem.",
    "Instructions": [
        "Given a table preview, you need to write a piece of python code to solve the question.",
        "Assume the data was stored in a file called 'PATH.csv', you can use pandas library to load it.",
        "Store your result in 'analysis_answer'. Your result should be a string, containing only the answer of question, not any other descriptions.",
        "Note: do not define and use any other functions, I only need the simplest code."
        ],
    "Data": {{
        "Table": "{prompt_tables}",
        "Question": "{utterance}"
    }}
}}"""
    with open('./run_log.txt', 'a', encoding='utf-8') as f_l:
        f_l.write(f"----------Code Solution----------\n")
        f_l.write("----------Prompt----------\n")
        f_l.write(prompt + '\n')

    model = args.engine
    messages = [
        {
            "role": "system",
            "content": "You are a tabular data analyst, you are proficient in writing python code to solve tabular data analysis problem."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    for i in range(num_p):
        while True:
            try:
                if model == "gpt-4o-mini":
                    output = client.chat.completions.create(model=model, messages=messages)
                    response = output.choices[0].message.content
                else:
                    output = ollama.chat(model=model, messages=messages, stream=False)
                    response = output['message']['content']
                    if model == "deepseek-r1:8b":
                        response = response.split('</think>')[-1].strip('\n')
                answer = re.findall(r'\```(.*?)\```', response, flags=re.DOTALL)
                if len(answer) != 1:
                    continue
                answer_code = answer[0].strip('```').strip('python')
                break
            except (json.decoder.JSONDecodeError, IndexError) as e:
                pass
                # print("Analysing: ", str(e) or repr(e))

        code = answer_code.replace("pd.read_csv('PATH.csv')", f"pd.read_csv('{table_file_path}')")

        with open('./run_log.txt', 'a', encoding='utf-8') as f_l:
            f_l.write("----------Response----------\n")
            f_l.write(response + '\n')
            f_l.write("----------Code----------\n")
            f_l.write(code + '\n')

        error = "No Error"
        analysis_answer = "No Answer"
        vars = {}
        try:
            exec(code, globals(), vars)
            analysis_answer = str(vars['analysis_answer'])
        except Exception as e:
            print("Code Execution Error: ", e)
            error = str(e) or repr(e)

        if error != "No Error":
            output = "Analysis Fail"
        else:
            output = analysis_answer

        if args.dataset == "TabFact":
            output = output.lower()
            if output == "true":
                output = 1
            elif output == "false":
                output = 0
            else:
                output = "Analysis Fail"

        outputs.append(output)

    return outputs


def direct_solution(num_p, prompt_tables, utterance):
    answers = []

    prompt = f"""{{
    "Persona": "You are a tabular data analyst, you are proficient in solving tabular data analysis problem.",
    "Instructions": [
        "Given a table preview, You need to judge whether the question can be solved using the preview I give you. If the question can be solved, you need to give the answer."
    ],
    "Output Format": "Your output should first contain a detailed analysis process. After that, attach your result in the following JSON format: {{"judgement": "can be answered" or "can't be answered", "answer": ""}}.",
    "Data": {{
        "Table": "{prompt_tables}",
        "Question": "{utterance}"
    }}
}}"""
    with open('./run_log.txt', 'a', encoding='utf-8') as f_l:
        f_l.write(f"----------Direct Solution----------\n")
        f_l.write("----------Prompt----------\n")
        f_l.write(prompt + '\n')

    model = args.engine
    messages = [
        {
            "role": "system",
            "content": "You are a tabular data analyst, you are proficient in solving tabular data analysis problem."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    for i in range(num_p):
        while True:
            try:
                if model == "gpt-4o-mini":
                    output = client.chat.completions.create(model=model, messages=messages)
                    response = output.choices[0].message.content
                else:
                    output = ollama.chat(model=model, messages=messages, stream=False)
                    response = output['message']['content']
                    if model == "deepseek-r1:8b":
                        response = response.split('</think>')[-1].strip('\n')
                answer = re.findall(r'\{(.*?)\}', response, flags=re.DOTALL)
                answer_json = json.loads('{' + answer[-1] + '}')

                if 'judgement' in answer_json:
                    judgement = answer_json["judgement"]
                    if judgement not in ["can be answered", "can't be answered"]:
                        continue

                    if judgement == "can be answered":
                        if 'answer' in answer_json:
                            final_answer = str(answer_json['answer'])
                            if args.dataset == "TabFact":
                                final_answer = final_answer.lower()
                                if final_answer not in ["true", "false"]:
                                    continue

                                if final_answer == "true":
                                    final_answer = 1
                                elif final_answer == "false":
                                    final_answer = 0
                        else:
                            continue
                    else:
                        final_answer = "Analysis Fail"
                else:
                    continue

                break
            except (json.decoder.JSONDecodeError, IndexError) as e:
                pass
                # print("Analysing: ", str(e) or repr(e))

        with open('./run_log.txt', 'a', encoding='utf-8') as f_l:
            f_l.write("----------Response----------\n")
            f_l.write(response + '\n')

        answers.append(final_answer)

    return answers


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
    with open('./run_log.txt', 'a', encoding='utf-8') as f_l:
        f_l.write(f"----------Judge Answer----------\n")
        f_l.write("----------Prompt----------\n")
        f_l.write(prompt + '\n')

    model = args.engine
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
    while True:
        try:
            if model == "gpt-4o-mini":
                output = client.chat.completions.create(model=model, messages=messages)
                response = output.choices[0].message.content
            else:
                output = ollama.chat(model=model, messages=messages, stream=False)
                response = output['message']['content']
                if model == "deepseek-r1:8b":
                    response = response.split('</think>')[-1].strip('\n')
            answer = re.findall(r'\{(.*?)\}', response, flags=re.DOTALL)
            answer_json = json.loads('{' + answer[-1] + '}')

            if 'judgement' in answer_json:
                judgement = answer_json["judgement"]
                if judgement not in ["yes", "no"]:
                    continue
            else:
                continue

            break
        except (json.decoder.JSONDecodeError, IndexError) as e:
            pass
            # print("Judging: ", str(e) or repr(e))
    with open('./run_log.txt', 'a', encoding='utf-8') as f_l:
        f_l.write("----------Response----------\n")
        f_l.write(response + '\n')

    if judgement == "yes":
        return gold_answer
    else:
        return model_answer


def output_summarize(utterance, outputs):
    outputs = [x for x in outputs if x not in ["Analysis Fail", ""]]
    if not outputs:
        return "Analysis Fail"

    if args.dataset == "TabFact":
        return max(outputs, key=outputs.count)
    else:
        summarized_outputs = []
        for output in outputs:
            if not summarized_outputs:
                new_cluster = {"name": output, "num": 1}
                summarized_outputs.append(new_cluster)
            else:
                summarize_flag = 0
                for cluster in summarized_outputs:
                    str_output = str(output)
                    str_name = str(cluster['name'])

                    if str_output.isdigit() and str_name.isdigit():
                        if str_output == str_name:
                            cluster['num'] += 1
                            summarize_flag = 1
                            break
                        continue

                    if str_output != str_name:
                        if judge_answer(utterance, str_output, str_name) == str_output:
                            continue

                    cluster['num'] += 1
                    summarize_flag = 1
                    break

                if summarize_flag == 0:
                    new_cluster = {"name": output, "num": 1}
                    summarized_outputs.append(new_cluster)

        max_answer = ""
        max_num = 0
        for cluster in summarized_outputs:
            if cluster['num'] > max_num:
                max_answer = cluster['name']
                max_num = cluster['num']

        return max_answer


def split_subtables(table_text, prompt_schema):
    max_tbl_size = args.max_tbl_size
    min_size = max_tbl_size/2
    max_size = int(min_size + min_size/2)

    len_tbl = len(table_text)
    iter_row = min(round(len_tbl / 3), max_size)
    iter_row = max(min_size, iter_row)
    iter_num = int(len_tbl / iter_row)
    iter_rows = [iter_row for i in range(iter_num)]
    remain = len_tbl - iter_row * iter_num
    if remain < iter_row / 2:
        iter_rows[-1] = iter_row + remain
    else:
        iter_rows = iter_rows + [remain]

    seq = 0
    sub_tables = []
    for i in range(len(iter_rows)):
        prompt_prefix = f"""
            ......
            Row {seq} : ......"""

        num_row = iter_rows[i]
        row_numbers = []
        prompt_records = f""""""
        for j in range(num_row):
            row = [str(x).replace('nan', 'None') for x in list(table_text.iloc[j + seq])]
            row_number = j + seq + 1
            prompt_record = f"""
            Row {row_number} : {' | '.join(' '.join(str(x).split(' ')[:10]) + '...' if len(str(x)) >= 100 else str(x) for x in row)}"""
            prompt_records += prompt_record
            row_numbers.append(row_number)

        prompt_suffix = f"""
            Row {row_number + 1} : ......
            ......"""

        if i == 0:
            prompt_records = prompt_records + prompt_suffix
        elif i == len(iter_rows) - 1:
            prompt_records = prompt_prefix + prompt_records
        else:
            prompt_records = prompt_prefix + prompt_records + prompt_suffix

        prompt_tables = prompt_schema + prompt_records
        sub_tables.append(prompt_tables)

        seq += num_row

    return sub_tables, iter_rows


def check_relative_records(prompt_tables, utterance):
    prompt = f"""{{
    "Persona": "You are a tabular data analyst, you are proficient in solving tabular data analysis problem.",
    "Instructions": [
        "Given a table preview and a question, You DON'T need to answer the question, just judge whether there are any rows related to the question.",
        "If yes, output their number. You don't need to output their entire contents, just number.",
        "Note: since the table data I give you is not complete, you don't need to judge whether the question can be solved. You only need to find out any rows that are relevant to the question."
    ],
    "Output Format": "Your output should first contain a detailed analysis process. After that, attach your result in the following JSON format: {{"judgement": "yes" or "no", "rows": [row number like "1", "2" or "all rows"]}}.",
    "Data": {{
        "Table": "{prompt_tables}",
        "Question": "{utterance}"
    }}
}}"""
    with open('./run_log.txt', 'a', encoding='utf-8') as f_l:
        f_l.write(f"----------Check Relative Records----------\n")
        f_l.write("----------Prompt----------\n")
        f_l.write(prompt + '\n')

    model = args.engine
    messages = [
        {
            "role": "system",
            "content": "You are a tabular data analyst, you are proficient in solving tabular data analysis problem."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    while True:
        relative_records = []
        try:
            if model == "gpt-4o-mini":
                output = client.chat.completions.create(model=model, messages=messages)
                response = output.choices[0].message.content
            else:
                output = ollama.chat(model=model, messages=messages, stream=False)
                response = output['message']['content']
                if model == "deepseek-r1:8b":
                    response = response.split('</think>')[-1].strip('\n')
            answer = re.findall(r'\{(.*?)\}', response, flags=re.DOTALL)
            answer_json = json.loads('{' + answer[-1] + '}')

            flag = 1
            if 'judgement' in answer_json:
                judgement = answer_json["judgement"]
                if judgement not in ["yes", "no"]:
                    flag = 0

                if judgement == "yes":
                    if 'rows' in answer_json:
                        relative_records = answer_json['rows']
                        if not isinstance(relative_records, list) or not relative_records:
                            flag = 0
                        else:
                            for record_number in relative_records:
                                if isinstance(record_number, str):
                                    if not record_number.isdigit() and record_number != "all rows":
                                        flag = 0
                                        break
                    else:
                        flag = 0
            else:
                flag = 0

            if flag == 0:
                continue

            break
        except (json.decoder.JSONDecodeError, IndexError) as e:
            pass
            # print("Analysing: ", str(e) or repr(e))
    with open('./run_log.txt', 'a', encoding='utf-8') as f_l:
        f_l.write("----------Response----------\n")
        f_l.write(response + '\n')

    direct_results = direct_solution(1, prompt_tables, utterance)
    direct_result = direct_results[0]

    return direct_result, relative_records


def partial_analysis(table_text, prompt_schema, utterance):
    sub_tables, iter_rows = split_subtables(table_text, prompt_schema)

    seq = 0
    sub_dicts = []
    for sub_table, num_row in zip(sub_tables, iter_rows):
        tbl_dict = {"records": sub_table, "start": seq + 1, "length": num_row}
        sub_dicts.append(tbl_dict)
        seq += num_row

    num_tbl = len(sub_dicts)

    quo = int(num_tbl / 3)
    remain = num_tbl - 3 * quo
    base = [quo for x in range(3)]
    for i in range(remain):
        base[-(i + 1)] += 1

    sub_dicts = [sub_dicts[:base[0]], sub_dicts[base[0]:base[0] + base[1]],
                 sub_dicts[base[0] + base[1]:base[0] + base[1] + base[2]]]

    direct_results = []
    new_tbl_records = []
    for tbl_cluster in sub_dicts:
        if not tbl_cluster:
            continue
        cluster_records = []

        mid_subtbl = tbl_cluster[int(len(tbl_cluster) / 2)]
        direct_result, relative_records = check_relative_records(mid_subtbl['records'], utterance)
        direct_results.append(direct_result)

        if relative_records:
            if "all rows" in relative_records:
                cluster_records += [row_number + mid_subtbl['start'] for row_number in range(mid_subtbl['length'])]
            else:
                try:
                    if any(int(x) < mid_subtbl['start'] for x in relative_records):
                        relative_records = [int(x) + mid_subtbl['start'] for x in relative_records]
                except:
                    print("ERROR! ", relative_records)
                    relative_records = [row_number + mid_subtbl['start'] for row_number in range(mid_subtbl['length'])]
                cluster_records += [int(x) for x in relative_records if int(x) <= mid_subtbl['start']+mid_subtbl['length']-1]

            former_index = int(len(tbl_cluster) / 2) - 1
            if former_index >= 0:
                former_subtbl = tbl_cluster[former_index]
                direct_result, relative_records = check_relative_records(former_subtbl['records'], utterance)
                direct_results.append(direct_result)
                if relative_records:
                    if "all rows" in relative_records:
                        cluster_records = [row_number + former_subtbl['start'] for row_number in
                                           range(former_subtbl['length'])] + cluster_records
                    else:
                        try:
                            if any(int(x) < former_subtbl['start'] for x in relative_records):
                                relative_records = [int(x) + former_subtbl['start'] for x in relative_records]
                        except:
                            print("ERROR! ", relative_records)
                            relative_records = [row_number + former_subtbl['start'] for row_number in
                                                range(former_subtbl['length'])]
                        cluster_records = [int(x) for x in relative_records if int(x) <= mid_subtbl['start']+mid_subtbl['length']-1] + cluster_records
                else:
                    cluster_records += []

            later_index = int(len(tbl_cluster) / 2) + 1
            if later_index < len(tbl_cluster):
                later_subtbl = tbl_cluster[later_index]
                direct_result, relative_records = check_relative_records(later_subtbl['records'], utterance)
                direct_results.append(direct_result)
                if relative_records:
                    if "all rows" in relative_records:
                        cluster_records = cluster_records + [row_number + later_subtbl['start'] for row_number in
                                                             range(later_subtbl['length'])]
                    else:
                        try:
                            if any(int(x) < later_subtbl['start'] for x in relative_records):
                                relative_records = [int(x) + later_subtbl['start'] for x in relative_records]
                        except:
                            print("ERROR! ", relative_records)
                            relative_records = [row_number + later_subtbl['start'] for row_number in
                                                range(later_subtbl['length'])]
                        cluster_records = cluster_records + [int(x) for x in relative_records if int(x) <= mid_subtbl['start']+mid_subtbl['length']-1]
                else:
                    cluster_records += []
        else:
            cluster_records += []
        new_tbl_records += cluster_records

    direct_results = [x for x in direct_results if x != "Analysis Fail"]

    return direct_results, new_tbl_records


def result_stat(total_outputs, total_values, tbl_lens):
    cnt = 0
    small_cnt = 0
    mid_cnt = 0
    large_cnt = 0
    tbl = 0
    pass_num = 0
    small_pass = 0
    mid_pass = 0
    large_pass = 0
    wrong_num = 0
    fail_num = 0
    for outputs, values, tbl_len in zip(total_outputs, total_values, tbl_lens):
        que = 0
        for output, value in zip(outputs, values):
            if output == "Analysis Fail":
                fail_num += 1
                # print(f"Failed: {tbl+1}-{que+1}")
            elif str(output) != str(value):
                wrong_num += 1
                # print(f"Wrong: {tbl+1}-{que+1}")
            else:
                pass_num += 1
                if tbl_len < 15:
                    small_pass += 1
                elif tbl_len > 30:
                    large_pass += 1
                else:
                    mid_pass += 1
            que += 1
            cnt += 1
            if tbl_len < 15:
                small_cnt += 1
            elif tbl_len > 30:
                large_cnt += 1
            else:
                mid_cnt += 1
        tbl += 1

    pass_rate = round(100 * pass_num / cnt, 2)
    wrong_rate = round(100 * wrong_num / cnt, 2)
    fail_rate = round(100 * fail_num / cnt, 2)

    print(f"Pass Rate: {pass_num}/{cnt}, {pass_rate}")
    print(f"Small Pass Rate: {small_pass}/{small_cnt}")
    print(f"Mid Pass Rate: {mid_pass}/{mid_cnt}")
    print(f"Large Pass Rate: {large_pass}/{large_cnt}")
    # print(f"Wrong Rate: {wrong_num}/{cnt}, {wrong_rate}")
    # print(f"Fail Rate: {fail_num}/{cnt}, {fail_rate}")


def main():
    with open('./run_log.txt', 'w', encoding='utf-8') as f_l:
        pass
    with open('./trial_log.txt', 'w', encoding='utf-8') as f_l:
        pass

    dataset = args.dataset
    max_tbl_size = args.max_tbl_size

    que_path = f'./Dataset/{dataset}/questions/test.json'
    csv_path = f'./Dataset/{dataset}'

    with open(que_path, 'r', encoding='utf-8') as f_i:
        que_data = json.load(f_i)

    # 90 893
    # 90 954
    que_data = que_data

    total_outputs = []
    total_values = []
    tbl_lens = []
    original_tbl_sizes = []
    new_tbl_sizes = []
    for i in range(len(que_data)):
        print(f"Table {i + 1}/{len(que_data)}")
        item = que_data[i]

        if dataset == 'WikiTableQuestions':
            name = item['page']
            table = item['table']
            utterances = item['utterance']
            values = item['targetValue']

            table_file_path = os.path.join(csv_path, table)
        elif dataset == 'TabFact':
            name = item['Caption']
            table = item['Table']
            utterances = ['Judge True or False: ' + x for x in item['Questions']]
            values = item['Result']

            table_file_path = os.path.join(csv_path, 'data', 'csv', table)
        else:
            name = ''
            table = ''
            utterances = []
            values = []

            table_file_path = ''

        table_text = pd.read_csv(table_file_path, encoding='utf-8')
        original_tbl_sizes.append(len(table_text))
        col = [x.replace('\n', '-') for x in list(table_text.columns)]

        prompt_schema = f"""
        Table Name : {name}
        Table Fields : {' | '.join(str(x) for x in col)}"""

        cnt = 0
        outputs = []
        test_values = []
        for utterance, value in zip(utterances, values):
            print(f"Question {cnt + 1}/{len(utterances)}")
            test_values.append(value)

            with open('./run_log.txt', 'a', encoding='utf-8') as f_l:
                f_l.write(f"----------Analysis {cnt + 1} Start----------\n")

            if len(table_text) <= max_tbl_size:
                prompt_records = f""""""
                for i in range(len(table_text)):
                    row = [str(x).replace('nan', 'None') for x in list(table_text.iloc[i])]
                    prompt_record = f"""
                Row {i + 1} : {' | '.join(' '.join(str(x).split(' ')[:10]) + '...' if len(str(x)) >= 100 else str(x) for x in row)}"""
                    prompt_records += prompt_record
                prompt_tables = prompt_schema + prompt_records

                direct_results = direct_solution(args.num_p, prompt_tables, utterance)
                code_results = code_solution(prompt_tables, utterance, table_file_path)
                all_results = direct_results + code_results

                final_result = output_summarize(utterance, all_results)

                if str(final_result).isdigit() and str(value).isdigit():
                    if str(final_result) == str(value):
                        output = value
                    else:
                        output = final_result
                else:
                    if final_result != "Analysis Fail":
                        output = judge_answer(utterance, final_result, value)
                    else:
                        output = "Analysis Fail"
                outputs.append(output)
                with open('./run_log.txt', 'a', encoding='utf-8') as f_l:
                    f_l.write(f"----------Analysis {cnt + 1} End----------\n")
                with open('./trial_log.txt', 'a', encoding='utf-8') as f_l:
                    f_l.write(f"----------Question {cnt + 1} Info----------\n")
                    f_l.write(f"----------Direct Results----------\n")
                    f_l.write(' | '.join(str(x) for x in direct_results))
                    f_l.write('\n')
                    f_l.write(f"----------Code Results----------\n")
                    f_l.write(' | '.join(str(x) for x in code_results))
                    f_l.write('\n')
                    f_l.write(f"----------Correct Results----------\n")
                    f_l.write(str(value) + '\n')
                    f_l.write('\n')
                cnt += 1
                continue

            tmp_direct_results, output = partial_analysis(table_text, prompt_schema, utterance)
            tmp_direct_results = [x for x in tmp_direct_results if x != "Analysis Fail"]

            new_tbl_records = list(set(output))
            new_tbl_sizes.append(len(new_tbl_records))
            prompt_records = f""""""
            if len(new_tbl_records) <= max_tbl_size:
                for i in range(len(new_tbl_records)):
                    row = [str(x).replace('nan', 'None') for x in list(table_text.iloc[new_tbl_records[i] - 1])]
                    prompt_record = f"""
                Row {i + 1} : {' | '.join(' '.join(str(x).split(' ')[:10]) + '...' if len(str(x)) >= 100 else str(x) for x in row)}"""
                    prompt_records += prompt_record
                prompt_tables = prompt_schema + prompt_records

                direct_results = direct_solution(args.num_p, prompt_tables, utterance)
                code_results = code_solution(prompt_tables, utterance, table_file_path)
                all_results = tmp_direct_results + direct_results + code_results
            else:
                for i in range(int(max_tbl_size / 2)):
                    row = [str(x).replace('nan', 'None') for x in list(table_text.iloc[new_tbl_records[i] - 1])]
                    prompt_record = f"""
                Row {i + 1} : {' | '.join(' '.join(str(x).split(' ')[:10]) + '...' if len(str(x)) >= 100 else str(x) for x in row)}"""
                    prompt_records += prompt_record
                prompt_records += f"""
                ......"""
                for i in range(int(max_tbl_size / 2)):
                    row = [str(x).replace('nan', 'None') for x in
                           list(table_text.iloc[new_tbl_records[i - int(max_tbl_size / 2)] - 1])]
                    prompt_record = f"""
                Row {i + 1 + len(new_tbl_records) - int(max_tbl_size / 2)} : {' | '.join(' '.join(str(x).split(' ')[:10]) + '...' if len(str(x)) >= 100 else str(x) for x in row)}"""
                    prompt_records += prompt_record
                prompt_tables = prompt_schema + prompt_records

                direct_results = direct_solution(args.num_p, prompt_tables, utterance)
                code_results = code_solution(prompt_tables, utterance, table_file_path)
                all_results = tmp_direct_results + code_results + direct_results

            final_result = output_summarize(utterance, all_results)

            if str(final_result).isdigit() and str(value).isdigit():
                if str(final_result) == str(value):
                    output = value
                else:
                    output = final_result
            else:
                if final_result != "Analysis Fail":
                    output = judge_answer(utterance, final_result, value)
                else:
                    output = "Analysis Fail"
            outputs.append(output)
            with open('./run_log.txt', 'a', encoding='utf-8') as f_l:
                f_l.write(f"----------Analysis {cnt + 1} End----------\n")
            with open('./trial_log.txt', 'a', encoding='utf-8') as f_l:
                f_l.write(f"----------Question {cnt + 1} Info----------\n")
                f_l.write(f"----------Other Results----------\n")
                f_l.write(' | '.join(str(x) for x in tmp_direct_results))
                f_l.write('\n')
                f_l.write(f"----------Direct Results----------\n")
                f_l.write(' | '.join(str(x) for x in direct_results))
                f_l.write('\n')
                f_l.write(f"----------Code Results----------\n")
                f_l.write(' | '.join(str(x) for x in code_results))
                f_l.write('\n')
                f_l.write(f"----------Correct Results----------\n")
                f_l.write(str(value) + '\n')
                f_l.write('\n')
            cnt += 1

        total_outputs.append(outputs)
        total_values.append(test_values)
        tbl_lens.append(len(table_text))

    try:
        print("New Size/Original Size: ", sum(new_tbl_sizes) / len(new_tbl_sizes),
              sum(original_tbl_sizes) / len(original_tbl_sizes))
    except:
        print("Original Size: ", sum(original_tbl_sizes) / len(original_tbl_sizes))
    result_stat(total_outputs, total_values, tbl_lens)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='WikiTableQuestions', choices=['WikiTableQuestions', 'TabFact'])
    parser.add_argument('--engine', type=str, default='llama3.1', choices=['llama3.1', 'gpt-4o-mini', 'deepseek-r1:8b'])
    parser.add_argument('--num-p', type=int, default=3)
    parser.add_argument('--max-tbl-size', type=int, default=10)

    args = parser.parse_args()

    main()
