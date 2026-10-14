# Base development image for MathVSE (Python 3.11, slim)
# Build with optional PyTorch CPU wheel: docker build --build-arg INSTALL_TORCH=1 -t mathvse:dev .
FROM python:3.11-slim

ARG INSTALL_TORCH=0
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/home/app/.local/bin:${PATH}"

# Install minimal OS packages needed for building wheels and common ops
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
WORKDIR /workspace
USER app

# Install Python runtime/dev dependencies commonly required for Phase 1.
# If your repo provides requirements.txt or pyproject.toml, prefer editing this file
# to `pip install -r requirements.txt` or `pip install .[dev]`.
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
      "numpy>=1.24,<2" \
      "opencv-python>=4.7.0,<5" \
      "Pillow>=9.4,<10" \
      "sympy>=1.11,<2" \
      "python-dotenv>=0.21,<1" \
      "streamlit>=1.24,<2" \
      "pytest>=7.2,<8"

# Optionally install CPU-only PyTorch (set INSTALL_TORCH=1 at build time).
# This uses the official PyTorch CPU wheel index.
RUN if [ "$INSTALL_TORCH" = "1" ] ; then \
    python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision --extra-index-url https://pypi.org/simple ; \
    fi

# Default entry: keep container alive so dev can exec into it.
# Typical usage: docker-compose run --rm dev bash
CMD ["bash"]