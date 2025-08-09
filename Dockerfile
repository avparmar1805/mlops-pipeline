# ---- builder ----
FROM python:3.9-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      gcc python3-dev build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /wheels
COPY requirements.txt .
# Build wheels for all deps
RUN pip wheel --wheel-dir . -r requirements.txt

# ---- final ----
FROM python:3.9-slim

WORKDIR /app
COPY --from=builder /wheels /wheels
COPY requirements.txt .

# Install from those wheels, no gcc needed here
RUN pip3 install --no-cache-dir --no-index \
      --find-links /wheels -r requirements.txt

COPY src/ ./src/
RUN mkdir -p ./logs

EXPOSE 8080
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8080"]
