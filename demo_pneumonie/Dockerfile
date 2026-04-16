FROM python:3.11-slim

WORKDIR /app

# Dépendances système pour OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Port HF Spaces
EXPOSE 7860

# Démarrage
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
