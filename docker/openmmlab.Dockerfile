ARG PYTORCH="1.5.1"
ARG CUDA="10.2"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
RUN pip install mmcv-full==latest+torch1.6.0+cu101 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html

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


# Install OpenSSH for MPI to communicate between containers
RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config