import concurrent.futures

from taras import config
from taras.llm import ANALYST_SYSTEM, build_messages, chat, parse_json_from_response
from taras.logging_utils import append_run_log
from taras.solutions import direct_solution
from taras.table_prompt import format_row_line, normalize_row_values


def split_subtables(table_text, prompt_schema):
    max_tbl_size = config.args.max_tbl_size
    min_size = int(max_tbl_size/2)
    max_size = int(1.5*min_size)

    len_tbl = len(table_text)
    iter_row = max(min_size, min(round(len_tbl / 3), max_size))
    iter_num = int(len_tbl / iter_row)
    iter_rows = [iter_row for _ in range(iter_num)]
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
            row = normalize_row_values(table_text.iloc[j + seq], replace_newline=True)
            row_number = j + seq + 1
            prompt_records += format_row_line(row, row_number)
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


def resolve_relative_records(subtbl, relative_records, mid_subtbl):
    if "all rows" in relative_records:
        return [row_number + subtbl['start'] for row_number in range(subtbl['length'])]
    try:
        if any(int(x) < subtbl['start'] for x in relative_records):
            relative_records = [int(x) + subtbl['start'] for x in relative_records]
    except:
        print("ERROR! ", relative_records)
        relative_records = [row_number + subtbl['start'] for row_number in range(subtbl['length'])]
    return [int(x) for x in relative_records if int(x) <= mid_subtbl['start']+mid_subtbl['length']-1]


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

    model = config.args.engine
    messages = build_messages(ANALYST_SYSTEM, prompt)

    retries = 0
    while retries < config.args.max_retries:
        retries += 1
        relative_records = []
        try:
            response = chat(model, messages)
            answer_json = parse_json_from_response(response)

            judgement = answer_json["judgement"]
            if judgement not in {"yes", "no"}:
                continue

            if judgement == "yes":
                relative_records = answer_json['rows']
                if not isinstance(relative_records, list) or not relative_records:
                    continue
                else:
                    relative_records = [str(x) for x in relative_records]
                    flag = True
                    for record_number in relative_records:
                        if not record_number.isdigit() and record_number != "all rows":
                            flag = False
                            break
                    if not flag:
                        continue
            break
        except Exception as e:
            pass
            # print("Extracting: ", str(e) or repr(e))
    if retries >= config.args.max_retries:
        relative_records = []

    append_run_log(
        f"----------Check Relative Records----------\n"
        f"----------Prompt----------\n"
        f"{prompt}\n"
        f"----------Response----------\n"
        f"{response}\n",
        use_file_lock=True,
    )

    direct_results = direct_solution(1, prompt_tables, utterance)
    direct_result = direct_results[0]

    return direct_result, relative_records


def cluster_analysis(tbl_cluster, utterance):
    if not tbl_cluster:
        return [], []

    adjacency_span = 1

    direct_results = []
    new_tbl_records = []
    cluster_records = []

    mid_subtbl = tbl_cluster[int(len(tbl_cluster) / 2)]
    direct_result, relative_records = check_relative_records(mid_subtbl['records'], utterance)
    direct_results.append(direct_result)

    if relative_records:
        cluster_records += resolve_relative_records(mid_subtbl, relative_records, mid_subtbl)

        former_index = int(len(tbl_cluster) / 2) - adjacency_span
        if former_index >= 0:
            former_subtbl = tbl_cluster[former_index]
            direct_result, relative_records = check_relative_records(former_subtbl['records'], utterance)
            direct_results.append(direct_result)
            if relative_records:
                cluster_records = resolve_relative_records(former_subtbl, relative_records, mid_subtbl) + cluster_records
            else:
                cluster_records += []

        later_index = int(len(tbl_cluster) / 2) + adjacency_span
        if later_index < len(tbl_cluster):
            later_subtbl = tbl_cluster[later_index]
            direct_result, relative_records = check_relative_records(later_subtbl['records'], utterance)
            direct_results.append(direct_result)
            if relative_records:
                cluster_records = cluster_records + resolve_relative_records(later_subtbl, relative_records, mid_subtbl)
            else:
                cluster_records += []
    else:
        cluster_records += []

    new_tbl_records += cluster_records

    return direct_results, new_tbl_records


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

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(cluster_analysis, sub_dicts[i], utterance) for i in range(3)]

        for future in concurrent.futures.as_completed(futures):
            try:
                output1, output2 = future.result()
                direct_results += output1
                new_tbl_records += output2
            except Exception as exc:
                print(f'Task generated an exception: {exc}')

    direct_results = [x for x in direct_results if x != "Analysis Fail"]

    return direct_results, new_tbl_records
