FROM python:3.11-slim-bookworm

WORKDIR /app

# install dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y gcc libpq-dev make && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
