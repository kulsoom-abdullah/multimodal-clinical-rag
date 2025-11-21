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
COPY app_v2.py .
COPY scripts/ scripts/

# 6. Data Ingestion (The "Self-Contained" Magic)
# Copy Database
COPY data/chroma_db_advanced /app/data/chroma_db_advanced
COPY data/docstore_advanced.pkl /app/data/docstore_advanced.pkl

# --- CRITICAL FIX: Copy the Images ---
COPY output/ /app/output/
# -------------------------------------

# 7. Network
EXPOSE 8501

# 8. Healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# 9. Run Command
CMD ["streamlit", "run", "app_v2.py", "--server.headless", "true"]