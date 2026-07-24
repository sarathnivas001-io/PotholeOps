FROM python:3.11-slim

WORKDIR /app

# System deps for Pillow/OpenCV-style image libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY params.yaml .
COPY models/ models/
COPY mlruns/ mlruns/
COPY dashboard/ dashboard/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
