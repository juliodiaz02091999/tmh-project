FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for opencv + basic build
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt /app/api/requirements.txt

RUN pip install --no-cache-dir -r /app/api/requirements.txt

# Copy backend code
COPY api/ /app/api/

# Copy model weights into /app (root_dir in app.py points here)
COPY model_a_fixed_iris.pth /app/model_a_fixed_iris.pth
COPY ["model_b_eyelid (2).pth", "/app/model_b_eyelid (2).pth"]

ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT}"]

