# HITDS - production image (CPU only).
# Build after: python scripts/06_export_deploy.py --dataset cicids2017
#   docker build -t hitds .
#   docker run -p 7860:7860 -e HITDS_USERS="analyst:change-me" -e HITDS_SECRET_KEY="long-random" hitds
# Small hosts (512 MB RAM): docker build --build-arg INSTALL_TORCH=0 -t hitds-lite .
#   Scoring never needs PyTorch (the MLP runs in NumPy); without torch, retraining swaps in
#   scikit-learn's MLP for that ensemble member.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=7860 HITDS_DATASET=cicids2017 HOST=0.0.0.0

# Hugging Face Spaces runs containers as uid 1000; the app writes its SQLite DB and retrained models
RUN useradd -m -u 1000 hitds
WORKDIR /app
RUN chown hitds:hitds /app

ARG INSTALL_TORCH=1
RUN if [ "$INSTALL_TORCH" = "1" ]; then pip install torch --index-url https://download.pytorch.org/whl/cpu; fi
COPY requirements-deploy.txt .
RUN pip install -r requirements-deploy.txt

COPY --chown=hitds:hitds hitds/ hitds/
COPY --chown=hitds:hitds app/ app/
COPY --chown=hitds:hitds serve.py ./
COPY --chown=hitds:hitds deploy/config.yaml ./config.yaml
COPY --chown=hitds:hitds deploy/data/ data/
COPY --chown=hitds:hitds deploy/models/ models/

USER hitds
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('PORT','7860'))"
CMD ["python", "serve.py"]
