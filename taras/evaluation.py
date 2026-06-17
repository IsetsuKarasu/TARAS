from rouge import Rouge


def result_stat(dataset, total_outputs, total_values, total_lens):
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

    rouge1_scores = 0
    rouge2_scores = 0
    rougel_scores = 0

    for outputs, values, len in zip(total_outputs, total_values, total_lens):
        que = 0
        for output, value in zip(outputs, values):
            if output == "Analysis Fail":
                fail_num += 1
                # print(f"Failed: {tbl+1}-{que+1}")
            elif dataset == 'FeTaQA':
                rouge = Rouge()
                rouge1_scores += rouge.get_scores(str(output), str(value))[0]['rouge-1']['r']
                rouge2_scores += rouge.get_scores(str(output), str(value))[0]['rouge-2']['r']
                rougel_scores += rouge.get_scores(str(output), str(value))[0]['rouge-l']['r']
            elif str(output).lower() != str(value).lower():
                wrong_num += 1
                # print(f"Wrong: {tbl+1}-{que+1}")
            else:
                if len <= 15:
                    small_pass += 1
                elif len <= 30:
                    mid_pass += 1
                else:
                    large_pass += 1
                pass_num += 1
            que += 1
            if len <= 15:
                small_cnt += 1
            elif len <= 30:
                mid_cnt += 1
            else:
                large_cnt += 1
            cnt += 1
        tbl += 1

    if dataset == 'FeTaQA':
        print(f"ROUGE-1: {rouge1_scores}/{cnt}, {round(rouge1_scores / cnt if cnt > 0 else 0, 2)}")
        print(f"ROUGE-2: {rouge2_scores}/{cnt}, {round(rouge2_scores / cnt if cnt > 0 else 0, 2)}")
        print(f"ROUGE-L: {rougel_scores}/{cnt}, {round(rougel_scores / cnt if cnt > 0 else 0, 2)}")
    else:
        pass_rate = round(100 * pass_num / cnt if cnt > 0 else 0, 2)
        wrong_rate = round(100 * wrong_num / cnt if cnt > 0 else 0, 2)
        fail_rate = round(100 * fail_num / cnt if cnt > 0 else 0, 2)
        print(f"Pass Rate: {pass_num}/{cnt}, {pass_rate}")
        print(f"Wrong Rate: {wrong_num}/{cnt}, {wrong_rate}")
        print(f"Fail Rate: {fail_num}/{cnt}, {fail_rate}")

        # small_pass_rate = round(100 * small_pass / small_cnt, 2)
        # mid_pass_rate = round(100 * mid_pass / mid_cnt, 2)
        # large_pass_rate = round(100 * large_pass / large_cnt, 2)
        # print(f"Small Pass Rate: {small_pass}/{small_cnt}, {small_pass_rate}")
        # print(f"Mid Pass Rate: {mid_pass}/{mid_cnt}, {mid_pass_rate}")
        # print(f"Large Pass Rate: {large_pass}/{large_cnt}, {large_pass_rate}")
