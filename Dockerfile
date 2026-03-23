FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first (much smaller than full torch - 500MB vs 2GB)
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    torchvision==0.16.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining packages
RUN pip install --no-cache-dir \
    gradio \
    "ultralytics==8.0.196" \
    opencv-python-headless \
    Pillow \
    numpy \
    pandas

COPY app.py .
COPY best.pt .

EXPOSE 10000

CMD ["python", "app.py"]
