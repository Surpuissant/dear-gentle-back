# Image de base légère et stable
FROM python:3.11-slim

# Variables d'environnement utiles
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Dépendances système minimales (tiktoken/Numpy ont des wheels, mais on garde build-essential au cas où)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
 && rm -rf /var/lib/apt/lists/*

# Dossier de travail
WORKDIR /app

# Installer les deps en amont (layer cache-friendly)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copier le code (assure-toi que app.py est à la racine du contexte)
COPY . .

# Port d’écoute FastAPI
EXPOSE 8000

# Commande de lancement (non-streaming pour le MVP — passe en streaming plus tard si besoin)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]