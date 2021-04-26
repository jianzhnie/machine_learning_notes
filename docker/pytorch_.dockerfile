FROM nvidia/cuda:10.2-devel-ubuntu18.04

ENV PYTORCH_VERSION=1.5.1
ENV TORCHVISION_VERSION=0.6.1
# Python 3.7 is supported by Ubuntu Bionic out of the box
ARG python=3.7
ENV PYTHON_VERSION=${python}

RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list

# for mmdetection
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 ninja-build libglib2.0-0 libxrender-dev  \
    vim \
    wget \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils \
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

# Install TensorFlow, Keras, PyTorch and MXNet
RUN PYTAGS=$(python -c "from packaging import tags; tag = list(tags.sys_tags())[0]; print(f'{tag.interpreter}-{tag.abi}')") && \
    pip install https://download.pytorch.org/whl/cu102/torch-${PYTORCH_VERSION}-${PYTAGS}-linux_x86_64.whl \
        https://download.pytorch.org/whl/cu102/torchvision-${TORCHVISION_VERSION}-${PYTAGS}-linux_x86_64.whl

# Install MMDetection
RUN conda clean --all
WORKDIR /openmmlab
RUN git clone https://github.com/open-mmlab/mmdetection.git mmdetection
WORKDIR /openmmlab/mmdetection
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .

# Install mmsegmentation
RUN WORKDIR /openmmlab
RUN git clone https://github.com/open-mmlab/mmsegmenation.git mmsegmentation
WORKDIR /openmmlab/mmsegmentation
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .
WORKDIR /openmmlab

# Install tensorboard jypyterlab
RUN pip install jupyterlab && \
    pip install tensorboard

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config