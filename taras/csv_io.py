import io

import pandas as pd


def normalize_wikitable_csv_text(text: str) -> str:
    return text.replace('\\"', '""')


def read_table_csv(csv_path: str, encoding: str = 'utf-8', **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path, encoding=encoding, **kwargs)
    except pd.errors.ParserError:
        with open(csv_path, 'r', encoding=encoding) as f:
            text = normalize_wikitable_csv_text(f.read())
        return pd.read_csv(io.StringIO(text), **kwargs)


def make_exec_pandas():
    import types

    exec_pd = types.ModuleType('pandas')
    exec_pd.__dict__.update({
        name: getattr(pd, name)
        for name in dir(pd)
        if not name.startswith('_')
    })
    exec_pd.read_csv = read_table_csv
    return exec_pd
