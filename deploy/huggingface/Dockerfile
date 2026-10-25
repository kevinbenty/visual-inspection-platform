# Hugging Face Space image: FastAPI model service + Streamlit dashboard in one container.
# The GitHub Action copies this file to the Space root as "Dockerfile".
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv

COPY requirements.txt requirements-dashboard.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
 && pip install -r requirements.txt -r requirements-dashboard.txt huggingface_hub

COPY app app
COPY src src
COPY configs configs
COPY dashboard dashboard
COPY start.sh start.sh

# Spaces run containers as user 1000
RUN useradd -m -u 1000 user && mkdir -p /data /models && chown -R user /data /models /srv
USER user

ENV HOME=/home/user \
    MODEL_DIR=/models/patchcore \
    SAMPLES_DIR=/models/samples \
    DB_PATH=/data/inspections.db \
    DEVICE=cpu \
    TORCH_HOME=/models/.torch \
    API_URL=http://127.0.0.1:8000

EXPOSE 7860
CMD ["bash", "start.sh"]
