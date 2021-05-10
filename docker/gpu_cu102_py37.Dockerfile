FROM nvidia/cuda:10.2-devel-ubuntu18.04
# ADD clean-layer.sh  /tmp/clean-layer.sh
# TensorFlow version is tightly coupled to CUDA and cuDNN so it should be selected carefully
ENV TENSORFLOW_VERSION=2.3.0
ENV PYTORCH_VERSION=1.5.1
ENV TORCHVISION_VERSION=0.6.1
ENV CUDNN_VERSION=7.6.5.32-1+cuda10.2
ENV NCCL_VERSION=2.7.8-1+cuda10.2
ENV MXNET_VERSION=1.5.1
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
        #/tmp/clean-layer.sh 

# for mmdetection
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
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

# Install TensorFlow, Keras, PyTorch and MXNet
RUN pip install future typing packaging
RUN pip install tensorflow==${TENSORFLOW_VERSION} \
                keras \
                h5py

RUN PYTAGS=$(python -c "from packaging import tags; tag = list(tags.sys_tags())[0]; print(f'{tag.interpreter}-{tag.abi}')") && \
    pip install https://download.pytorch.org/whl/cu102/torch-${PYTORCH_VERSION}-${PYTAGS}-linux_x86_64.whl \
        https://download.pytorch.org/whl/cu102/torchvision-${TORCHVISION_VERSION}-${PYTAGS}-linux_x86_64.whl

# https://download.pytorch.org/whl/cu102/torch-1.5.1-cp37-cp37m-linux_x86_64.whl
# https://download.pytorch.org/whl/cu102/torchvision-0.6.1-cp37-cp37m-linux_x86_64.whl

# Install Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz && \
    tar zxf openmpi-4.0.0.tar.gz && \
    cd openmpi-4.0.0 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi
    #/tmp/clean-layer.sh

# Install MMCV
RUN pip install mmcv-full -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html

# Install MMDetection
RUN git clone --depth 1 https://github.com/open-mmlab/mmdetection.git /openmmlab/mmdetection
WORKDIR /openmmlab/mmdetection
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .

# Install mmsegmentation
# RUN git clone --depth 1 https://github.com/open-mmlab/mmsegmentation.git /openmmlab/mmsegmentation
# WORKDIR /openmmlab/mmsegmentation
# ENV FORCE_CUDA="1"
# RUN pip install -r requirements/build.txt
# RUN pip install --no-cache-dir -e .
# RUN pip install git+https://github.com/open-mmlab/mmsegmentation.git # install the master branch
# WORKDIR /openmmlab
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
    pip install matplotlib && \
    pip install xgboost && \
    pip install lightgbm
    #/tmp/clean-layer.sh

# Install tensorboard jypyterlab
RUN pip install jupyterlab && \
    pip install tensorboard

# Install Horovod, temporarily using CUDA stubs
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MXNET=1 \
         pip install --no-cache-dir horovod[all-frameworks] && \
    ldconfig

# Install OpenSSH for MPI to communicate between containers
RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config