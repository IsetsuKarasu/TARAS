import argparse

from taras import config
from taras.pipeline import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='WikiTableQuestions', choices=['WikiTableQuestions', 'TabFact', 'FeTaQA', 'TableBench'])
    parser.add_argument('--engine', type=str, default='qwen3:8b', choices=['llama3.1', 'qwen3:8b', 'gpt-4o-mini', 'gpt-4o-ca'])
    parser.add_argument('--num-p', type=int, default=3)
    parser.add_argument('--max-tbl-size', type=int, default=10)
    parser.add_argument('--max-retries', type=int, default=3)
    parser.add_argument('--retrieval-strategy', type=str, default='vector',
                        choices=['llm', 'bm25', 'vector', 'sql'])
    parser.add_argument('--retrieval-top-k', type=int, default=None)

    config.args = parser.parse_args()

    main()
