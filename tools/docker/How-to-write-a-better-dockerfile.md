# 优雅的编写Dockerfile

### 一、语法

- 一条Dockerfile的命令，对应一个bulid的step，自上而下按顺序构建：
  - 首先拉取父镜像，留存一个镜像ID
  - 以上一步的镜像启动一个中间态容器，记录容器ID，执行step；执行完毕，删除这个ID的中间态容器，留存一个新的镜像ID
  - 每一次step执行，都会生成一个层的镜像，每层镜像会有唯一的 Hash 码，是增量式的。




### 二、常见样例

```dockerfile
FROM ubuntu:16.04
RUN apt-get update
RUN apt-get install -y apt-utils libjpeg-dev \
python-pip
RUN pip install --upgrade pip
RUN easy_install -U setuptools
RUN apt-get clean
```



- 好处：当正在执行的过程某一层出错，对其进行修正后再次Build，前面已经执行完成的层不会再次执行。这样能大大减少下次Build的时间。
- 坏处：如果Dockerfile文件多长，层数过多，会使镜像占用的空间也变得巨大。



```dockerfile
FROM ubuntu:16.04
RUN apt-get update && apt-get install -y apt-utils \
  libjpeg-dev python-pip \
           && pip install --upgrade pip \
      && easy_install -U setuptools \
    && apt-get clean
```



- 好处：把所有的组件全部在一层解决，这样做能一定程度上减少镜像的占用空间。
- 坏处：在制作基础镜像的时候若其中某个组编译出错，修正后再次Build就相当于重头再来了，比较消耗时间。



### 三、多阶构建

- 为了解决分层与空间占用大的问题，**最好的方法是使用多阶构建**

- 我们只需要在Dockerfile中多次使用FORM声明，每次FROM指令可以使用不同的基础镜像，我们可以选择将一个阶段的构建结果复制到另一个阶段，在最终的镜像中只会留下最后一次构建的结果，并且只需要编写一个Dockerfile文件。

- 最终效果在go环境中能让后端镜像从**300M减少到7M**



```dockerfile
FROM golang:1.11.4-alpine3.8 AS build-env

ENV GO111MODULE=off
ENV GO15VENDOREXPERIMENT=1
ENV GITPATH=https://github.com/apulis/AIArtsBackend
RUN mkdir -p /go/src/${GITPATH}
COPY ./ /go/src/${GITPATH}
RUN cd /go/src/${GITPATH} && CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go install -v

FROM alpine:latest
ENV apk –no-cache add ca-certificates
COPY --from=build-env /go/bin/AIArtsBackend /root/AIArtsBackend
```



### 四、注意事项

#### 合理使用docker的缓存机制
 - 把变化最少的部分放在 Dockerfile 的前面，提升dockerfile调试和创建效率。
 - 可能导致缓存失效的命令WORKDIR、CMD、ENV、ADD等，最好放到dockerfile底部。
 - 在dockerfile中使用RUN执行shell命令是，可以用"&&"将多条命令连接起来。**层数过多会导致镜像包过大**，需进行权衡。


#### 镜像最小化

- **不要把构建的过程也放到运行时的Dockerfile里**，可以把构建好的可执行文件通过copy命令放入镜像中。把常用的base_docker_image单独构建好。

- **尽量采用alpine版本镜像**，对于go语言基础镜像可以看到默认的latest镜像881MB而alpine仅仅只有不到50MB，同时通过减少镜像程序也可以**提高安全性**。

- 可以将容器拆分成多个子容器，使用docker-compose启动，容器内互相之间请求端口host为容器名。如 mysql://app_db:3306

- 合理使用`RUN rm` 构建出来的镜像不要包含不需要的内容，删除掉日志、安装临时文件等。

#### 其他建议


- 设置一些常用的国内加速镜像源。

  ```RUN python3 -m pip install --upgrade pip && python3 -m pip config set global.index-url http://mirrors.aliyun.com/pypi/simple && python3 -m pip config set install.trusted-host mirrors.aliyun.com
  ```

  pip源：`pip  install -i http://mirrors.aliyun.com/pypi/simple`

  apt源：`RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list`

- 对于一些需要在容器创建完成之后再运行的命令，如数据库的migrate操作。应该使用supervisor进程管理程序启动。

  `ENTRYPOINT ["/usr/bin/supervisord"]`


- **需要的pip包通过`pip freeze`命令，导出requirements**。

  - 对requirements文件进行拆分，并分次安装，避免每次都需要重新pip安装。

  - 对于较大的安装包，如TensorFlow可以导入whl包进行本地安装。

  - 对于arm架构中需要下载tar包进行本地编译的包，可以采用pip wheel <package or -i requirements.txt>，把所有依赖存储为whl包，再导入镜像中安装
    ```dockerfile
    COPY ./packages  /packages
    RUN pip install /packages/*.whl && rm -rf /packages/
    ```

  - 常用pip命令

    `pip install -i https://mirrors.ustc.edu.cn/pypi/web/simple  --default-timeout 600 --no-cache-dir   -r requirements.txt --user`
    -i 使用国内镜像源，default-timeout设置超时时间，避免安装大包时抛出timeout中断，--no-cache-dir避免出现缓存的BadZipfile，  --user会把包安装到home目录下，避免权限问题。
- 使用.dockerignore，避免当前目录下所有文件都拷入到docker deamon中
