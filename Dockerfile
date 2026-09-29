# HITDS - production image (CPU only).
# This version trains a small DEMO model (synthetic data) while the image is built,
# so no deploy/ folder is needed. For the real CIC-IDS2017 model, see docs/DEPLOY.md.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=7860 HITDS_DATASET=synthetic HOST=0.0.0.0

RUN useradd -m -u 1000 hitds
WORKDIR /app

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-deploy.txt .
RUN pip install -r requirements-deploy.txt

COPY hitds/ hitds/
COPY app/ app/
COPY serve.py config.yaml ./

# Build the demo model: synthetic flows -> clean split -> train ensemble -> save as live model v1
RUN python -c "\
import sys, pandas as pd; sys.path.insert(0, '.'); \
from hitds.common import load_config, use_dataset; \
from hitds.data import make_synthetic, split_and_save; \
from hitds.pipeline import train_bundle, save_bundle, set_live; \
base = load_config(); cfg = use_dataset(base, 'synthetic'); p = cfg['paths']['processed']; \
split_and_save(make_synthetic(20000, base['seed']), p, base, base['seed'], dataset='synthetic'); \
b = train_bundle(cfg, pd.read_pickle(p + '/train.pkl'), pd.read_pickle(p + '/val.pkl'), method='smote'); \
save_bundle(b, cfg['paths']['models']); set_live(cfg['paths']['models'], 1, 'demo model built in Docker'); \
print('demo model ready')"

RUN chown -R hitds:hitds /app
USER hitds
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('PORT','7860'))"
CMD ["python", "serve.py"]
