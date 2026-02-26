FROM python:3.11-slim

WORKDIR /app

# Install build dependencies that might be needed by pandas/chromadb/pydantic
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Render exposes the port in the PORT environment variable
ENV PORT=8000

# Start Uvicorn, listening on 0.0.0.0 and the designated port
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}
