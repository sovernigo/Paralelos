FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    CUDAARCHS=75 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    clang-format \
    python3 \
    python3-matplotlib \
    cuda-nsight-compute-12-4 \
    && rm -rf /var/lib/apt/lists/*

# Remove cuda-compat para evitar conflito com o driver GeForce (RTX 2060 SUPER) no host/WSL2
RUN rm -rf /usr/local/cuda/compat /etc/ld.so.conf.d/00-cuda-compat.conf && ldconfig

# Garante o binario do Nsight Compute (ncu) no PATH
ENV PATH="/usr/local/cuda/bin:/opt/nvidia/nsight-compute:${PATH}"

WORKDIR /workspace

CMD ["/bin/bash"]