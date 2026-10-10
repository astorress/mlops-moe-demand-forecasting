# Entorno de ejecución de los notebooks (CPU). En una máquina con GPU NVIDIA,
# los notebooks usan CUDA automáticamente si está disponible.
FROM python:3.12-slim

# libgomp1: lo necesitan LightGBM y XGBoost
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# src/ en el PYTHONPATH: permite importar moe_forecaster (y cargar moe_forecaster.pkl)
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg
