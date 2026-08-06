# Build context is the repo root (see build.sh) - the image needs both the shared
# skinbouncer_core package and the api/ code, which live in different top-level dirs.
FROM python:3.11-slim

WORKDIR /app

# opencv-python (not the -headless variant) dlopens libGL/libglib at import time, which
# a slim base image doesn't ship.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Installed before the api/ code so this layer only rebuilds when dependencies change,
# not on every code edit. No GUI extra here on purpose - see pyproject.toml.
COPY pyproject.toml ./
COPY skinbouncer_core ./skinbouncer_core
COPY labeling_tool ./labeling_tool
RUN pip install --no-cache-dir .

COPY 06_Deployment/api/main.py 06_Deployment/api/minecraft_skin_downloader.py ./
# Whatever detector folders exist here at build time (and only those) end up in the
# image - there's no runtime toggle. build.sh creates this directory if it doesn't
# exist yet, so a fresh clone with zero exported detectors still builds cleanly.
COPY 06_Deployment/api/models/detectors ./models/detectors

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
