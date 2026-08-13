# Hugging Face Spaces — Docker SDK
# Mirror of sets-warp-backend FastAPI service for migration from Render.
# Source of truth: https://github.com/<owner>/sets-warp-backend
FROM python:3.12-slim

# HF Spaces require user 1000 to avoid permission issues with the writable
# bind-mounts the platform sets up at runtime.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

# Install Python deps first so subsequent code changes don't bust the layer.
COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Runtime sources only — admin/training scripts stay out of the deployed
# image to keep it small and reduce the attack surface.
COPY --chown=user main.py ./
# The ingestion whitelist. Uploading it to the Space repo is not enough: it has
# to be copied into the image, or _load_labels_bundled() finds nothing and the
# backend fails open with validation disabled (which is what /health reports).
COPY --chown=user config/ ./config/

# Must match app_port in README.md YAML so HF's edge routes to the right port.
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
