# =============================================================================
# Credit Default Risk Scoring API
# DDM501 - Lab 1: First ML Product
# =============================================================================
FROM python:3.11-slim

# 1. Environment: no .pyc, unbuffered logs, importable package root
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# 2. Working directory
WORKDIR /app

# 3. Dependencies first, so a source change does not invalidate this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Application code (models/ is mounted at runtime, not baked in)
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY data/ ./data/

# 5. Non-root user
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/models \
    && chown -R appuser:appuser /app
USER appuser

# 6. Port
EXPOSE 8000

# 7. Health check: /health must answer AND report model_loaded=true
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import json,sys,urllib.request; \
d=json.load(urllib.request.urlopen('http://localhost:8000/health', timeout=5)); \
sys.exit(0 if d.get('model_loaded') else 1)"

# 8. Start uvicorn bound to all interfaces
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
