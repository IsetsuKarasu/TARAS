import concurrent.futures

from taras import config
from taras.solutions import code_solution, direct_solution


def run_direct_and_code_analysis(prompt_tables, utterance, table_file_path, max_workers):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_direct = executor.submit(direct_solution, config.args.num_p, prompt_tables, utterance)
        future_code = executor.submit(code_solution, prompt_tables, utterance, table_file_path)

        try:
            direct_results = future_direct.result()
            code_results = future_code.result()
        except Exception as exc:
            print(f'Task generated an exception: {exc}')

    return direct_results, code_results
