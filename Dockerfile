# Imagen base liviana de Python 3.11
FROM python:3.11-slim

# Evitar generación de archivos .pyc y asegurar salida inmediata en consola
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Instalar dependencias del sistema requeridas para compilación y curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente y artefactos del proyecto
COPY src/ ./src/
COPY plots/ ./plots/
COPY output/ ./output/
COPY app.py main.py reporte_final.md respuestas_teoricas.md ./

# Exponer el puerto predeterminado de Streamlit
EXPOSE 8501

# Healthcheck para verificar disponibilidad del contenedor
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Comando de ejecución
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
