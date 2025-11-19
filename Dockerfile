# 1. Base Image
FROM python:3.11-slim

# 2. Environment Variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# 3. System Dependencies
WORKDIR /app
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Application Code
# IMPORTANT: We copy app_v2.py, but we can rename it to app.py inside the container for simplicity, 
# or just call it app_v2.py in the CMD. Let's keep it explicit.
COPY app_v2.py .
COPY scripts/ scripts/

# 6. Data Ingestion (The "Self-Contained" Magic)
# This copies your local vector DB into the container image.
# Ensure these folders exist locally before building!
COPY data/chroma_db_advanced /app/data/chroma_db_advanced
COPY data/docstore_advanced.pkl /app/data/docstore_advanced.pkl

# 7. Network
EXPOSE 8501

# 8. Healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# 9. Run Command
# Note: We point to app_v2.py here
CMD ["streamlit", "run", "app_v2.py", "--server.headless", "true"]