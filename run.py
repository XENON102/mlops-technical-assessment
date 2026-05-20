"""
MLOps Batch Pipeline — run.py
Rolling-mean signal generator on OHLCV close prices.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("mlops_pipeline")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    # file handler — always DEBUG
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    # console handler — INFO+
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REQUIRED_CONFIG_KEYS = {"seed", "window", "version"}


def load_config(config_path: str, logger: logging.Logger) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must be a mapping (key: value) document.")

    missing = REQUIRED_CONFIG_KEYS - cfg.keys()
    if missing:
        raise ValueError(f"Config missing required keys: {sorted(missing)}")

    # type checks
    if not isinstance(cfg["seed"], int):
        raise ValueError(f"'seed' must be an integer, got: {type(cfg['seed']).__name__}")
    if not isinstance(cfg["window"], int) or cfg["window"] < 1:
        raise ValueError(f"'window' must be a positive integer, got: {cfg['window']}")
    if not isinstance(cfg["version"], str) or not cfg["version"].strip():
        raise ValueError(f"'version' must be a non-empty string, got: {cfg['version']!r}")

    logger.info(
        "Config loaded — seed=%s  window=%s  version=%s",
        cfg["seed"], cfg["window"], cfg["version"],
    )
    return cfg


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_dataset(input_path: str, logger: logging.Logger) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Cannot parse CSV: {exc}") from exc

    if df.empty:
        raise ValueError("Input CSV is empty.")

    if "close" not in df.columns:
        raise ValueError(
            f"Required column 'close' not found. Columns present: {list(df.columns)}"
        )

    # coerce close to numeric; flag non-parseable rows
    original_len = len(df)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    bad_rows = df["close"].isna().sum()
    if bad_rows == original_len:
        raise ValueError("Column 'close' contains no valid numeric values.")
    if bad_rows:
        logger.warning("Dropped %d row(s) with non-numeric 'close' values.", bad_rows)
        df = df.dropna(subset=["close"]).reset_index(drop=True)

    logger.info("Dataset loaded — %d rows, %d columns.", len(df), len(df.columns))
    logger.debug("Columns: %s", list(df.columns))
    return df


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def compute_rolling_mean(df: pd.DataFrame, window: int, logger: logging.Logger) -> pd.DataFrame:
    """
    Compute rolling mean on 'close'.
    The first (window-1) rows will be NaN; those rows are excluded from
    signal computation (they are not counted in rows_processed or signal_rate).
    """
    logger.info("Computing rolling mean with window=%d.", window)
    df = df.copy()
    df["rolling_mean"] = df["close"].rolling(window=window).mean()
    nan_count = df["rolling_mean"].isna().sum()
    logger.debug(
        "Rolling mean computed — %d warm-up rows excluded (NaN).", nan_count
    )
    return df


def compute_signal(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    signal = 1 if close > rolling_mean, else 0.
    Only rows where rolling_mean is not NaN are included.
    """
    logger.info("Generating binary signal.")
    valid = df.dropna(subset=["rolling_mean"]).copy()
    valid["signal"] = (valid["close"] > valid["rolling_mean"]).astype(int)
    logger.debug(
        "Signal generated — %d valid rows, %d signal=1, %d signal=0.",
        len(valid),
        valid["signal"].sum(),
        (valid["signal"] == 0).sum(),
    )
    return valid


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def write_metrics(output_path: str, payload: dict, logger: logging.Logger) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Metrics written to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLOps batch signal pipeline.")
    parser.add_argument("--input",    required=True, help="Path to input CSV file.")
    parser.add_argument("--config",   required=True, help="Path to YAML config file.")
    parser.add_argument("--output",   required=True, help="Path for output metrics JSON.")
    parser.add_argument("--log-file", required=True, dest="log_file",
                        help="Path for log file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.log_file)

    version = "unknown"
    t_start = time.time()
    logger.info("=== Job started ===")

    try:
        # 1. Config
        cfg = load_config(args.config, logger)
        version = cfg["version"]
        seed    = cfg["seed"]
        window  = cfg["window"]

        np.random.seed(seed)
        logger.debug("NumPy random seed set to %d.", seed)

        # 2. Dataset
        df = load_dataset(args.input, logger)

        # 3. Rolling mean
        df = compute_rolling_mean(df, window, logger)

        # 4. Signal
        valid = compute_signal(df, logger)

        # 5. Metrics
        rows_processed = len(valid)
        signal_rate    = float(valid["signal"].mean())
        latency_ms     = int((time.time() - t_start) * 1000)

        metrics = {
            "version":        version,
            "rows_processed": rows_processed,
            "metric":         "signal_rate",
            "value":          round(signal_rate, 4),
            "latency_ms":     latency_ms,
            "seed":           seed,
            "status":         "success",
        }

        logger.info(
            "Metrics — rows_processed=%d  signal_rate=%.4f  latency_ms=%d",
            rows_processed, signal_rate, latency_ms,
        )

        write_metrics(args.output, metrics, logger)
        logger.info("=== Job completed successfully ===")

        # Print final JSON to stdout (Docker requirement)
        print(json.dumps(metrics, indent=2))
        sys.exit(0)

    except Exception as exc:
        latency_ms = int((time.time() - t_start) * 1000)
        logger.error("Pipeline failed: %s", exc, exc_info=True)

        error_metrics = {
            "version":       version,
            "status":        "error",
            "error_message": str(exc),
            "latency_ms":    latency_ms,
        }
        write_metrics(args.output, error_metrics, logger)
        logger.info("=== Job ended with error ===")

        print(json.dumps(error_metrics, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
