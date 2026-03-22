FROM nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    DEBIAN_FRONTEND=noninteractive \
    apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    git \
    libopencv-dev \
    ninja-build \
    tree
