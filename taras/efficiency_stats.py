import json

EFFICIENCY_LOG_PATH = './sample_efficiency.jsonl'


def init_efficiency_log():
    with open(EFFICIENCY_LOG_PATH, 'w', encoding='utf-8') as f:
        pass


def record_sample_efficiency(sample_idx, it, ot, et, ct):
    record = {
        'sample': sample_idx,
        'it': it,
        'ot': ot,
        'et': et,
        'ct': ct,
    }
    with open(EFFICIENCY_LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')


def compute_efficiency_averages():
    records = []
    with open(EFFICIENCY_LOG_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return 0.0, 0.0, 0.0, 0.0

    n = len(records)
    ait = sum(r['it'] for r in records) / n
    aot = sum(r['ot'] for r in records) / n
    aet = sum(r['et'] for r in records) / n
    act = sum(r['ct'] for r in records) / n
    return ait, aot, aet, act
