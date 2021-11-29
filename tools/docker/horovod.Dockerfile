FROM horovod/horovod:0.20.0-tf2.3.0-torch1.6.0-mxnet1.6.0.post0-py3.7-cuda10.1

RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list

# for mmdetection
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 ninja-build libglib2.0-0 libxrender-dev  \
    nano \
    htop \
    screen \
    nodejs \
    sudo \
    vim \
    wget \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# change pip source
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
RUN pip install --upgrade pip

# Install MMCV
RUN pip install mmcv-full==latest+torch1.6.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html

# Install MMDetection
RUN git clone --depth 1 https://gitee.com/mirrors/mmdetection.git /mmdetection
WORKDIR /mmdetection
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .

# Install python packages
RUN pip install numpy && \
    pip install Pillow && \
    pip install scipy && \
    pip install scikit-image && \
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
    pip install lightgbm && \
    pip install dm-sonnet==1.23
    #/tmp/clean-layer.sh

# Install tensorboard jypyterlab
RUN pip install jupyterlab && \
    pip install tensorboard && \
    pip install mlflow
