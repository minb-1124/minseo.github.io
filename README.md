# Completion-Sense Benchmark

NASim optimal-path observations are converted into text, encoded semantically,
and evaluated as a completion-sense reward curve.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 scripts/completion_sense_benchmark.py
```

Outputs are written to `logs/`.
