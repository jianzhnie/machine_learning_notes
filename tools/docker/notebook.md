

# 1. docker build 过程中遇到 apt-get 出现以下报错

```sh
Reading package lists...
W: GPG error: https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu1804/x86_64  Release: The following signatures were invalid: BADSIG F60F4B3D7FA2AF80 cudatools <cudatools@nvidia.com>
E: The repository 'https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  Release' is not signed.
```

解决方案:
这个是
```sh
rm /etc/apt/sources.list.d/cuda.list
```
