# TARAS

TARAS is a table question-answering pipeline that combines **direct reasoning** and **code-based analysis** over tabular data. It supports multiple benchmarks and can run with local models via [Ollama](https://ollama.com/) or remote models via an OpenAI-compatible API.

## Requirements

- Python 3.10+
- One of the following LLM backends:
  - **Ollama** (for local models such as `qwen3:8b`, `llama3.1`)
  - **OpenAI-compatible API** (for `gpt-4o-mini`, `gpt-4o-ca`)
- Benchmark data placed under `./Dataset/` (see [Dataset layout](#dataset-layout))

## Installation

```bash
git clone <repository-url>
cd TARAS

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Configuration

Before running, edit `taras/config.py` and provide your own credentials. **Do not commit real API keys.**

```python
# taras/config.py

client = OpenAI(
    api_key="YOUR_API_KEY",          # required for GPT engines
    base_url="YOUR_OPENAI_BASE_URL", # OpenAI-compatible endpoint
)
```

| Setting | When needed |
|---------|-------------|
| `api_key` | Required when using `--engine gpt-4o-mini` or `gpt-4o-ca` |
| `base_url` | Required for non-default OpenAI-compatible providers |

Ollama engines (`llama3.1`, `qwen3:8b`) do not use the OpenAI client, but `config.py` is still imported at startup.

## Dataset layout

Expected structure:

```
Dataset/
├── TableBench/
│   ├── questions/test.json
│   └── csv/*.csv
├── WikiTableQuestions/
│   ├── questions/test.json
│   └── *.csv
├── FeTaQA/
│   ├── questions/test.json
│   └── csv/*.csv
└── TabFact/
    ├── questions/test.json
    └── data/csv/*.csv
```

## Usage

Run from the project root:

```bash
python TARAS.py
```

### Command-line arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `TableBench` | Benchmark: `WikiTableQuestions`, `TabFact`, `FeTaQA`, `TableBench` |
| `--engine` | `qwen3:8b` | LLM: `llama3.1`, `qwen3:8b`, `gpt-4o-mini`, `gpt-4o-ca` |
| `--num-p` | `3` | Number of parallel samples per direct/code solver |
| `--max-tbl-size` | `10` | Row threshold for full-table vs. partial analysis |
| `--max-retries` | `3` | Max retries per LLM call |

### Examples

```bash
# TableBench with local Qwen model (default)
python TARAS.py --dataset TableBench --engine qwen3:8b

# WikiTableQuestions with Llama 3.1
python TARAS.py --dataset WikiTableQuestions --engine llama3.1

# TabFact with GPT via OpenAI-compatible API (configure config.py first)
python TARAS.py --dataset TabFact --engine gpt-4o-mini --num-p 1
```

## Output

Each run writes logs in the project root:

| File | Contents |
|------|----------|
| `run_log.txt` | Prompts, LLM responses, generated code, and analysis steps |
| `trial_log.txt` | Per-question direct/code results, vote outcome, and gold answer |

Metrics (pass rate, ROUGE, token/time stats) are printed to the terminal when the run finishes.

## Project structure

```
TARAS.py              Entry point and CLI
taras/
├── config.py         API client and global runtime state
├── pipeline.py       Main evaluation loop
├── solvers.py        Single-shot LLM solvers (code / direct / judge)
├── solutions.py      Parallel solution orchestration
├── refinement.py     Partial-table analysis for large tables
├── llm.py            Shared LLM helpers
├── table_prompt.py   Table-to-prompt formatting
├── dataset.py        Dataset-specific item parsing
├── evaluation.py     Metric computation
└── ...
```
