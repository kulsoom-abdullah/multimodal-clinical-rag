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

# 6. Data
# The index, docstore and extracted figures are build artifacts, not source:
# they are produced by scripts/extract_pdfs.py and scripts/ingest_data_advanced.py
# from PDFs that are not redistributed here, and they are gitignored. They are
# therefore mounted at run time rather than baked in -- COPYing them would make
# this image unbuildable from a clean clone.
#
#   docker build -t clinical-rag .
#   docker run --env-file .env -p 8501:8501 \
#     -v "$(pwd)/data:/app/data" \
#     -v "$(pwd)/output:/app/output" \
#     clinical-rag
#
# Expected at /app/data/chroma_db_advanced, /app/data/docstore_advanced.pkl
# and /app/output. The app reports a clear error if they are absent.

# 7. Network
EXPOSE 8501

# 8. Healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# 9. Run Command
CMD ["streamlit", "run", "app_v2.py", "--server.headless", "true"]