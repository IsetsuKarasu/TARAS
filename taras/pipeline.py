import json
import os
import time

from taras import config
from taras.csv_io import read_table_csv
from taras.analysis import run_direct_and_code_analysis
from taras.dataset import load_dataset_item
from taras.efficiency_stats import (
    compute_efficiency_averages,
    init_efficiency_log,
    record_sample_efficiency,
)
from taras.evaluation import result_stat
from taras.logging_utils import append_run_log, write_trial_log_question
from taras.output import resolve_final_output
from taras.refinement import partial_analysis
from taras.summarize import output_summarize
from taras.table_prompt import (
    build_full_table_prompt,
    build_prompt_schema,
    build_subset_table_prompt,
    build_truncated_table_prompt,
)


def main():
    with open('./run_log.txt', 'w', encoding='utf-8') as f_l:
        pass
    with open('./trial_log.txt', 'w', encoding='utf-8') as f_l:
        pass
    init_efficiency_log()

    dataset = config.args.dataset
    max_tbl_size = config.args.max_tbl_size

    que_path = f'./Dataset/{dataset}/questions/test.json'
    csv_path = f'./Dataset/{dataset}'

    with open(que_path, 'r', encoding='utf-8') as f_i:
        que_data = json.load(f_i)
    
    que_data = [que_data[5]]

    total_outputs = []
    total_values = []
    total_lens = []
    for i in range(len(que_data)):
        item_info = load_dataset_item(dataset, que_data[i], csv_path)
        if item_info is None:
            continue

        name = item_info['name']
        utterances = item_info['utterances']
        values = item_info['values']
        table_file_path = item_info['table_file_path']

        print(f"Table {i + 1}/{len(que_data)}")

        table_text = read_table_csv(table_file_path, encoding='utf-8')
        prompt_schema = build_prompt_schema(name, table_text.columns)

        cnt = 0
        outputs = []
        test_values = []
        for utterance, value in zip(utterances, values):
            print(f"Question {cnt + 1}/{len(utterances)}")
            test_values.append(value)

            sample_start_time = time.time()
            prev_input_tokens = config.input_tokens
            prev_output_tokens = config.output_tokens
            prev_calling_times = config.calling_times

            append_run_log(f"----------Analysis {cnt + 1} Start----------\n")

            if len(table_text) <= max_tbl_size:
                prompt_tables = build_full_table_prompt(table_text, prompt_schema, replace_newline=True)

                direct_results, code_results = run_direct_and_code_analysis(
                    prompt_tables, utterance, table_file_path, max_workers=3,
                )

                all_results = direct_results + code_results
                final_result = output_summarize(utterance, all_results)
                output = resolve_final_output(final_result, value, dataset, utterance)

                outputs.append(output)
                append_run_log(f"----------Analysis {cnt + 1} End----------\n")
                write_trial_log_question(
                    cnt, final_result, output, value, direct_results, code_results,
                )
                record_sample_efficiency(
                    cnt + 1,
                    config.input_tokens - prev_input_tokens,
                    config.output_tokens - prev_output_tokens,
                    time.time() - sample_start_time,
                    config.calling_times - prev_calling_times,
                )
                cnt += 1
                continue

            tmp_direct_results, output = partial_analysis(table_text, prompt_schema, utterance)
            tmp_direct_results = [x for x in tmp_direct_results if x != "Analysis Fail"]

            new_tbl_records = list(set(output))
            if len(new_tbl_records) <= max_tbl_size:
                prompt_tables = build_subset_table_prompt(
                    table_text, prompt_schema, new_tbl_records, replace_newline=True,
                )

                direct_results, code_results = run_direct_and_code_analysis(
                    prompt_tables, utterance, table_file_path, max_workers=3,
                )

                all_results = tmp_direct_results + direct_results + code_results
            else:
                prompt_tables = build_truncated_table_prompt(
                    table_text, prompt_schema, new_tbl_records, max_tbl_size,
                )

                direct_results, code_results = run_direct_and_code_analysis(
                    prompt_tables, utterance, table_file_path, max_workers=3,
                )

                all_results = tmp_direct_results + code_results + direct_results

            final_result = output_summarize(utterance, all_results)
            output = resolve_final_output(final_result, value, dataset, utterance)

            outputs.append(output)
            append_run_log(f"----------Analysis {cnt + 1} End----------\n")
            write_trial_log_question(
                cnt, final_result, output, value, direct_results, code_results,
                other_results=tmp_direct_results,
            )
            record_sample_efficiency(
                cnt + 1,
                config.input_tokens - prev_input_tokens,
                config.output_tokens - prev_output_tokens,
                time.time() - sample_start_time,
                config.calling_times - prev_calling_times,
            )
            cnt += 1

        total_outputs.append(outputs)
        total_values.append(test_values)
        total_lens.append(len(table_text))

        break

    result_stat(dataset, total_outputs, total_values, total_lens)

    ait, aot, aet, act = compute_efficiency_averages()
    print("AIT: ", ait)
    print("AOT: ", aot)
    print("AET: ", aet)
    print("ACT: ", act)
