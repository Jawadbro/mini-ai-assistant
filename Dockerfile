# Slim Python base — faiss-cpu and pypdf both have manylinux wheels,
# so no build-essential/gcc needed here.
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so Docker can cache this layer
# separately from application code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render (and most PaaS providers) inject the port to bind via $PORT.
# Default to 8000 for local `docker run`.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
