# ╔══════════════════════════════════════════════════════════════════╗
# ║   RecSys Auto-Benchmark — Docker Image                          ║
# ║   Base: python:3.12-slim                                        ║
# ║   Supports: CLI benchmark mode + FastAPI server mode            ║
# ╚══════════════════════════════════════════════════════════════════╝

# ── Stage 1: Builder ────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# System deps needed to compile native extensions (scipy, lightfm, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only requirements first — maximises Docker layer cache
COPY requirements.txt .

# Upgrade pip, then install all dependencies into /install
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL maintainer="Jahidul Arafat <jahidularafat.dev@gmail.com>" \
      description="RecSys Auto-Benchmark: 14 algorithms, Optuna tuning, FastAPI" \
      python.version="3.12"

# Minimal runtime system libraries (BLAS for numpy/scipy)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libopenblas0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

# ── App setup ───────────────────────────────────────────────────────────────
WORKDIR /app

# Copy source code
COPY . .

# Create directory for user-uploaded data files
RUN mkdir -p /app/data /app/results

# Non-root user for security
RUN useradd --create-home --shell /bin/bash recsys && \
    chown -R recsys:recsys /app
USER recsys

# ── Environment ─────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# FastAPI port
EXPOSE 8000

# ── Default command: run FastAPI server ─────────────────────────────────────
# Override with:
#   docker run recsys python main.py --file /app/data/ratings.csv
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]