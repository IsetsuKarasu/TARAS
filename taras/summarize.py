from taras import config
from taras.output import judge_answer


def output_summarize(utterance, outputs):
    outputs = [x for x in outputs if x not in {"Analysis Fail", ""}]
    if not outputs:
        return "Analysis Fail"

    if config.args.dataset == "TabFact":
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
