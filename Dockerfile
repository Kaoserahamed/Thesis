FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

# Install the pinned project dependency ranges and notebook tooling. The
# Earth Engine collection dependencies remain optional by design.
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock \
    && pip install --no-cache-dir "jupyterlab>=4,<5"

COPY . .

# Keep the container usable without root for notebooks and generated outputs.
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /workspace
USER appuser

EXPOSE 8888

# The image is useful in CI-like environments by default. Override this with
# `jupyter lab --ip=0.0.0.0 --no-browser --allow-root` for interactive work.
CMD ["python", "scripts/ci_local.py"]
