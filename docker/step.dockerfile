docker run -it nvidia/cuda:10.2-devel-ubuntu18.04 /bin/bash

sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list


apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*



apt-get update && apt-get install -y ffmpeg libsm6 libxext6 ninja-build libglib2.0-0 libxrender-dev  \
    vim \
    wget \
    python3.7 \
    python3.7-dev \
    python3.7-distutils \
    nano \
    htop \
    screen \
    nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* 

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        vim \
        wget \
        ca-certificates \
        libcudnn7=${CUDNN_VERSION} \
        libnccl2=${NCCL_VERSION} \
        libnccl-dev=${NCCL_VERSION} \
        libjpeg-dev \
        libpng-dev \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-distutils \
        librdmacm1 \
        libibverbs1 \
        ibverbs-providers
