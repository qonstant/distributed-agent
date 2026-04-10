FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install small system deps needed by numpy/faiss wheels (adjust if needed)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
       build-essential \
       ca-certificates \
       wget \
       libopenblas-dev \
 && rm -rf /var/lib/apt/lists/*

# Install python deps (layer cached)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the entire repository into the image (this will include .env if present)
COPY . /app/

# Ensure out exists and make it mountable
RUN mkdir -p /app/out

VOLUME ["/app/out"]

EXPOSE 8000

# Run the FastAPI app (uses rag_file_only_api.py)
CMD ["uvicorn", "rag_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
