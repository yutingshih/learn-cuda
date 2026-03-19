FROM nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git \
    cmake \
    ninja-build \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*
