# MLOps Batch Pipeline — T0 Technical Assessment

A minimal, production-style batch job that loads OHLCV data, computes a rolling-mean
signal on `close`, and writes structured metrics + logs.

---

## Project structure

```
mlops-pipeline/
├── run.py           # Main pipeline script
├── config.yaml      # Runtime config (seed, window, version)
├── data.csv         # 10 000-row OHLCV dataset
├── requirements.txt # Pinned dependencies
├── Dockerfile       # Docker image definition
├── metrics.json     # Sample output from a successful run
├── run.log          # Sample log from a successful run
└── README.md
```

---

## Local run

### Prerequisites
- Python 3.9+
- pip

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the pipeline

```bash
python run.py \
  --input    data.csv \
  --config   config.yaml \
  --output   metrics.json \
  --log-file run.log
```

All four flags are required — no hardcoded paths exist in the code.

---

## Docker

### Build

```bash
docker build -t mlops-task .
```

### Run

```bash
docker run --rm mlops-task
```

The container:
- Copies data.csv and config.yaml into the container image at build time
- Executes the pipeline with all required flags
- Prints `metrics.json` to **stdout**
- Exits **0** on success, **non-zero** on failure

---

## Example `metrics.json` (success)

```json
{
  "version": "v1",
  "rows_processed": 9996,
  "metric": "signal_rate",
  "value": 0.4991,
  "latency_ms": 27,
  "seed": 42,
  "status": "success"
}
```

> `rows_processed = 9996` because the first `window - 1 = 4` rows yield NaN rolling
> means and are excluded from both signal computation and the final count.

## Example `metrics.json` (error)

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Required column 'close' not found. Columns present: ['open', 'high']",
  "latency_ms": 3
}
```

---

## Config reference (`config.yaml`)

| Key       | Type    | Description                          |
|-----------|---------|--------------------------------------|
| `seed`    | int     | NumPy random seed for reproducibility|
| `window`  | int ≥ 1 | Rolling mean window size             |
| `version` | str     | Pipeline version tag in output JSON  |

---

## Signal logic

```
rolling_mean[t] = mean(close[t-window+1 : t+1])   # standard pandas rolling
signal[t]       = 1  if close[t] > rolling_mean[t]
                  0  otherwise
```

The first `window - 1` rows have `rolling_mean = NaN` and are **excluded** from
`rows_processed` and `signal_rate`.

---

## Reproducibility

Results are deterministic across runs:

```
Run 1 → signal_rate = 0.4991
Run 2 → signal_rate = 0.4991   ✓
```

Controlled by `seed: 42` in `config.yaml`.

---

## Validation / error handling

The pipeline validates:
- Config file exists and contains `seed`, `window`, `version` with correct types
- Input CSV exists, is parseable, is non-empty, and contains `close`
- Non-numeric `close` values are dropped with a warning (not a hard failure)

Errors always produce a `metrics.json` with `"status": "error"` and exit code 1.

## Tech Stack

- Python 3.9
- pandas
- numpy
- PyYAML
- Docker
- logging
- argparse


## Observability

The pipeline emits structured logs to run.log including:
- job start/end
- config validation
- rows loaded
- rolling mean computation
- signal generation
- metrics summary
- validation and runtime errors

- Exit code 0 on success
- Exit code 1 on failure
