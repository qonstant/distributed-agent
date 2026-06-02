FROM python:3.11-slim-bookworm

WORKDIR /app

# install dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y gcc libpq-dev make && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Keep graph-build dependencies isolated because LightRAG currently pulls
# Pydantic v2, while the admin app still runs on Pydantic v1.
RUN python -m venv /opt/lightrag-venv && \
    /opt/lightrag-venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/lightrag-venv/bin/pip install --no-cache-dir \
        "lightrag-hku>=1.4.16" \
        boto3 \
        mammoth \
        markdownify \
        pdfplumber \
        python-dotenv

# Copy the entire project
COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
