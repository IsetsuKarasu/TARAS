import os


def load_dataset_item(dataset, item, csv_path):
    # 4344
    if dataset == 'WikiTableQuestions':
        name = item['page']
        table = item['table']
        utterances = item['utterance']
        values = item['targetValue']

        table_file_path = os.path.join(csv_path, table)
    # 2024
    elif dataset == 'TabFact':
        name = item['Caption']
        table = item['Table']
        utterances = ['Judge True or False: ' + x for x in item['Questions']]
        values = item['Result']

        table_file_path = os.path.join(csv_path, 'data', 'csv', table)
    # 2003
    elif dataset == 'FeTaQA':
        name = item['page']
        table = item['table']
        utterances = item['utterance']
        values = item['targetValue']

        table_file_path = os.path.join(csv_path, 'csv', table)
    elif dataset == 'TableBench':
        # if item["type"] != "NumericalReasoning":
        # if item["type"] != "FactChecking":
        # if item["type"] != "DataAnalysis":
        if item["type"] != "Visualization":
        # if item["type"] not in {"NumericalReasoning", "FactChecking", "DataAnalysis"}:
            return None

        name = item['name']
        table = f'{str(item['id'])}.csv'
        utterances = [x + " You don't need to actually draw the chart, just return the vertical axis value of the chart in the form of a multidimensional array." for x in item['utterance']]
        values = item['targetValue']

        table_file_path = os.path.join(csv_path, 'csv', table)
    else:
        name = ''
        table = ''
        utterances = []
        values = []

        table_file_path = ''

    return {
        'name': name,
        'table': table,
        'utterances': utterances,
        'values': values,
        'table_file_path': table_file_path,
    }
