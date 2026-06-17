from taras import config


def append_run_log(content, use_file_lock=False):
    if use_file_lock:
        with config.file_lock:
            with open('./run_log.txt', 'a', encoding='utf-8') as f_l:
                f_l.write(content)
    else:
        with open('./run_log.txt', 'a', encoding='utf-8') as f_l:
            f_l.write(content)


def log_prompt_section(header, prompt, use_file_lock=False):
    append_run_log(
        f"{header}\n"
        f"----------Prompt----------\n"
        f"{prompt}\n",
        use_file_lock=use_file_lock,
    )


def log_response(response, code=None, use_file_lock=False):
    content = "----------Response----------\n" + response + '\n'
    if code is not None:
        content += "----------Code----------\n" + code + '\n'
    append_run_log(content, use_file_lock=use_file_lock)


def write_trial_log_question(cnt, final_result, output, value, direct_results, code_results, other_results=None):
    with open('./trial_log.txt', 'a', encoding='utf-8') as f_l:
        f_l.write(f"----------Question {cnt + 1} Info----------\n")
        if other_results is not None:
            f_l.write(f"----------Other Results----------\n")
            f_l.write(' | '.join(str(x) for x in other_results))
            f_l.write('\n')
        f_l.write(f"----------Direct Results----------\n")
        f_l.write(' | '.join(str(x) for x in direct_results))
        f_l.write('\n')
        f_l.write(f"----------Code Results----------\n")
        f_l.write(' | '.join(str(x) for x in code_results))
        f_l.write('\n')
        f_l.write(f"----------Vote Result----------\n")
        f_l.write(str(final_result))
        f_l.write('\n')
        f_l.write(f"----------Final Result----------\n")
        f_l.write(str(output))
        f_l.write('\n')
        f_l.write(f"----------Correct Result----------\n")
        f_l.write(str(value) + '\n')
        f_l.write('\n')
