FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv
# CPU-only torch keeps the image small; change the index URL for CUDA.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
 && pip install -r requirements.txt
COPY app app
COPY src src
COPY configs configs
# Model weights are NOT baked in; mount them (see docker-compose.yml).
RUN useradd -m appuser && mkdir -p /data && chown appuser /data
USER appuser
ENV MODEL_DIR=/models/patchcore/bottle DB_PATH=/data/inspections.db DEVICE=cpu \
    TORCH_HOME=/models/.torch
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
