FROM nvidia/cuda:10.2-devel-ubuntu18.04

# ADD clean-layer.sh  /tmp/clean-layer.sh
# TensorFlow version is tightly coupled to CUDA and cuDNN so it should be selected carefully
ENV PYTORCH_VERSION=1.8.0
ENV TORCHVISION_VERSION=0.9
ENV CUDNN_VERSION=7.6.5.32-1+cuda10.2

# Python 3.7 is supported by Ubuntu Bionic out of the box
ARG python=3.7
ENV PYTHON_VERSION=${python}

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]
RUN rm /etc/apt/sources.list.d/cuda.list
RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        cmake \
        g++-7 \
        sudo \
        zip \
        unzip \
        git \
        curl \
        vim \
        wget \
        ca-certificates \
        libjpeg-dev \
        libpng-dev \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-distutils \
        librdmacm1 \
        libibverbs1 \
        ibverbs-providers
        #/tmp/clean-layer.sh 

# for mmdetection
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libxrender-dev  libxext6 \
    nano \
    htop \
    screen \
    nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* 
    #/tmp/clean-layer.sh 

# Install python and pip
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# change pip source
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
RUN pip install --upgrade pip
RUN pip install future typing packaging

# install pytorch 
RUN PYTAGS=$(python -c "from packaging import tags; tag = list(tags.sys_tags())[0]; print(f'{tag.interpreter}-{tag.abi}')") && \
    pip install https://download.pytorch.org/whl/cu102/torch-${PYTORCH_VERSION}-${PYTAGS}-linux_x86_64.whl 
        # https://download.pytorch.org/whl/cu102/torchvision-${TORCHVISION_VERSION}-${PYTAGS}-linux_x86_64.whl

RUN pip install torchvision
# install mmcv, mmdetection, mmsegmentation
RUN pip install openmim
RUN mim install mmdet
RUN mim install mmcls
RUN pip install mmsegmentation # install the latest release


# Install python packages
RUN pip install numpy && \
    pip install Pillow && \
    pip install scipy && \
    pip install scikit-learn && \
    pip install networkx && \
    pip install tf-slim && \
    pip install pandas && \
    pip install tqdm && \
    pip install imgaug && \
    pip install shapely && \
    pip install requests && \
    pip install graphviz && \
    pip install cloudpickle && \
    pip install seaborn && \
    pip install flask && \
    pip install matplotlib
    #/tmp/clean-layer.sh

# Install tensorboard jypyterlab
RUN pip install jupyterlab && pip install tensorboard