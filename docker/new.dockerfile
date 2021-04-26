docker pull nvidia/cuda:10.2-runtime-ubuntu18.04
docker run -it nvidia/cuda:10.2-runtime-ubuntu18.04 /bin/bash

# set apt source 
sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list

# install python & other dependencies
apt update -y
apt install -y  sudo curl ffmpeg libsm6 libxext6 vim wget libjpeg-dev libgl1-mesa-glx python3.8 python3.8-dev python3-distutils

# install pip
cp /usr/bin/python3.8 /usr/bin/python3
cp /usr/bin/python3.8 /usr/bin/python
curl -O https://bootstrap.pypa.io/get-pip.py && python get-pip.py && rm get-pip.py

# change pip source
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip install --upgrade pip

# install pytorch opencv
python -m pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 opencv-python 
python -m pip install jupyterlab 
python -m pip install tensorboard