# ── base image ───────────────────────────────────────────────────────────────
FROM python:3.9-slim

# ── metadata ─────────────────────────────────────────────────────────────────
LABEL maintainer="mlops-intern"
LABEL version="v1"

# ── working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── install dependencies first (cached layer) ────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── copy pipeline source + data ───────────────────────────────────────────────
COPY run.py       .
COPY config.yaml  .
COPY data.csv     .

# ── run the pipeline ─────────────────────────────────────────────────────────
# All paths are relative to /app — no hardcoding.
CMD ["python", "run.py", \
     "--input",    "data.csv", \
     "--config",   "config.yaml", \
     "--output",   "metrics.json", \
     "--log-file", "run.log"]
