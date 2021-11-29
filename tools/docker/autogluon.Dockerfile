FROM  horovod/horovod:0.20.0-tf2.3.0-torch1.6.0-mxnet1.6.0.post0-py3.7-cuda10.1

RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list

# install some tools
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 ninja-build libglib2.0-0 libxrender-dev  \
    nano \
    htop \
    screen \
    nodejs \
    sudo \
    git \
    vim \
    wget \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# change pip source
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
RUN pip install --upgrade pip
RUN pip install future typing packaging

# install autogluon
RUN pip install -U setuptools wheel
# Here we assume CUDA 10.1 is installed.  You should change the number
# according to your own CUDA version (e.g. mxnet_cu100 for CUDA 10.0).
RUN pip install -U "mxnet_cu101<2.0.0"
RUN pip install autogluon

# install autokeras
RUN pip install git+https://github.com/keras-team/keras-tuner.git
RUN pip install autokeras

# install timm
RUN pip install timm

# Install python packages
RUN pip install numpy && \
    pip install Pillow && \
    pip install scipy && \
    pip install scikit-image && \
    pip install scikit-learn && \
    pip install networkx && \
    pip install pandas && \
    pip install tqdm && \
    pip install imgaug && \
    pip install shapely && \
    pip install requests && \
    pip install graphviz && \
    pip install cloudpickle && \
    pip install seaborn && \
    pip install matplotlib
    #/tmp/clean-layer.sh

# Install tensorboard jypyterlab
RUN pip install jupyterlab && \
    pip install tensorboard &&
