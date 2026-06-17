def format_cell(value):
    cell = str(value)
    if len(cell) >= 100:
        return ' '.join(cell.split(' ')[:10]) + '...'
    return cell


def format_row_line(row_values, row_number):
    return f"""
    Row {row_number} : {' | '.join(format_cell(x) for x in row_values)}"""


def normalize_row_values(row_series, replace_newline=False):
    if replace_newline:
        return [str(x).replace('nan', 'None').replace('\n', ' ') for x in list(row_series)]
    return [str(x).replace('nan', 'None') for x in list(row_series)]


def build_prompt_schema(name, columns):
    col = [x.replace('\n', '-') for x in list(columns)]
    return f"""
    Table Name : {name}
    Table Fields : {' | '.join(str(x) for x in col)}"""


def build_full_table_prompt(table_text, prompt_schema, replace_newline=True):
    prompt_records = f""""""
    for i in range(len(table_text)):
        row = normalize_row_values(table_text.iloc[i], replace_newline=replace_newline)
        prompt_records += format_row_line(row, i + 1)
    return prompt_schema + prompt_records


def build_subset_table_prompt(table_text, prompt_schema, row_indices, replace_newline=False):
    prompt_records = f""""""
    for i in range(len(row_indices)):
        row = normalize_row_values(table_text.iloc[row_indices[i] - 1], replace_newline=replace_newline)
        prompt_records += format_row_line(row, i + 1)
    return prompt_schema + prompt_records


def build_truncated_table_prompt(table_text, prompt_schema, row_indices, max_tbl_size):
    prompt_records = f""""""
    for i in range(int(max_tbl_size / 2)):
        row = normalize_row_values(table_text.iloc[row_indices[i] - 1], replace_newline=False)
        prompt_records += format_row_line(row, i + 1)
    prompt_records += f"""
            ......"""
    for i in range(int(max_tbl_size / 2)):
        row = normalize_row_values(
            table_text.iloc[row_indices[i - int(max_tbl_size / 2)] - 1],
            replace_newline=False,
        )
        prompt_records += format_row_line(
            row,
            i + 1 + len(row_indices) - int(max_tbl_size / 2),
        )
    return prompt_schema + prompt_records
