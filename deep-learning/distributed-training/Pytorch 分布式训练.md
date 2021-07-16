初始化方式
分布式任务中，各节点之间需要进行协作，比如说控制数据同步等。因此，需要进行初始化，指定协作方式，同步规则等。

torch.distributed 提供了 3 种初始化方式，分别为 tcp、共享文件 和 环境变量初始化 等。

TCP 初始化
代码

TCP 方式初始化，需要指定进程 0 的 ip 和 port。这种方式需要手动为每个进程指定进程号。

import torch.distributed as dist

# Use address of one of the machines
dist.init_process_group(backend, init_method='tcp://10.1.1.20:23456',
                        rank=args.rank, world_size=4)
说明

不同进程内，均使用主进程的 ip 地址和 port，确保每个进程能够通过一个 master 进行协作。该 ip 一般为主进程所在的主机的 ip，端口号应该未被其他应用占用。

实际使用时，在每个进程内运行代码，并需要为每一个进程手动指定一个 rank，进程可以分布与相同或不同主机上。

多个进程之间，同步进行。若其中一个出现问题，其他的也马上停止。

使用

Node 1

python mnsit.py --init-method tcp://192.168.54.179:22225 --rank 0 --world-size 2
Node 2

python mnsit.py --init-method tcp://192.168.54.179:22225 --rank 1 --world-size 2
底层实现

在深入探讨初始化算法之前，先从 C/C++ 层面，大致浏览一下 init_process_group 背后发生了什么。

解析并验证参数
后端通过 name2channel.at() 函数进行解析，返回一个 channel 类，将用于执行数据传输
丢弃 GIL，并调用 THDProcessGroupInit() 函数，其实例化该 channel，并添加 master 节点的地址
rank 0 对应的进程将会执行 master 过程，而其他的进程则作为 workers
master
为所有的 worker 创建 sockets
等待所有的 worker 连接
发送给他们所有其他进程的位置
每一个 worker
创建连接 master 的 sockets
发送自己的位置信息
接受其他 workers 的信息
打开一个新的 socket，并与其他 wokers 进行握手信号
初始化结束，所有的进程之间相互连接
共享文件系统初始化
该初始化方式，要求共享的文件对于组内所有进程可见！

代码

设置方式如下：

import torch.distributed as dist

# rank should always be specified
dist.init_process_group(backend, init_method='file:///mnt/nfs/sharedfile',
                        world_size=4, rank=args.rank)
说明

其中，以 file:// 为前缀，表示文件系统各式初始化。/mnt/nfs/sharedfile 表示共享的文件，各个进程在共享文件系统中通过该文件进行同步或异步。因此，所有进程必须对该文件具有读写权限。

每一个进程将会打开这个文件，写入自己的信息，并等待直到其他所有进程完成该操作。在此之后，所有的请求信息将会被所有的进程可访问，为了避免 race conditions，文件系统必须支持通过 fcntl 锁定（大多数的 local 系统和 NFS 均支持该特性）。

说明：若指定为同一文件，则每次训练开始之前，该文件必须手动删除，但是文件所在路径必须存在！

与 tcp 初始化方式一样，也需要为每一个进程手动指定 rank。
使用

在主机 01 上：

python mnsit.py --init-method file://PathToShareFile/MultiNode --rank 0 --world-size 2
在主机 02 上：

python mnsit.py --init-method file://PathToShareFile/MultiNode --rank 1 --world-size 2
这里相比于 TCP 的方式麻烦一点的是运行完一次必须更换共享的文件名，或者删除之前的共享文件，不然第二次运行会报错。

环境变量初始化
默认情况下使用的都是环境变量来进行分布式通信，也就是指定 init_method="env://"。通过在所有机器上设置如下四个环境变量，所有的进程将会适当的连接到 master，获取其他进程的信息，并最终与它们握手(信号)。

MASTER_PORT: 必须指定，表示 rank0上机器的一个空闲端口（必须设置）
MASTER_ADDR: 必须指定，除了 rank0 主机，表示主进程 rank0 机器的地址（必须设置）
WORLD_SIZE: 可选，总进程数，可以这里指定，在 init 函数中也可以指定
RANK: 可选，当前进程的 rank，也可以在 init 函数中指定
配合 torch.distribution.launch 使用。

使用实例

Node 1: (IP: 192.168.1.1, and has a free port: 1234)

>>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
           --nnodes=2 --node_rank=0 --master_addr="192.168.1.1"
           --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
           and all other arguments of your training script)
Node 2

>>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
           --nnodes=2 --node_rank=1 --master_addr="192.168.1.1"
           --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
           and all other arguments of your training script)
Distributed Modules
DistributedDataParallel
原型
torch.nn.parallel.DistributedDataParallel(module, 
                                          device_ids=None, 
                                          output_device=None, 
                                          dim=0, 
                                          broadcast_buffers=True, 
                                          process_group=None, 
                                          bucket_cap_mb=25, 
                                          find_unused_parameters=False, 
                                          check_reduction=False)
功能
将给定的 module 进行分布式封装， 其将输入在 batch 维度上进行划分，并分配到指定的 devices 上。

module 会被复制到每台机器的每个 GPU 上，每一个模型的副本处理输入的一部分。

在反向传播阶段，每个机器的每个 GPU 上的梯度进行汇总并求平均。与 DataParallel 类似，batch size 应该大于 GPU 总数。

参数解析
module
要进行分布式并行的 module，一般为完整的 model

device_ids
int 列表或 torch.device 对象，用于指定要并行的设备。参考 DataParallel。

对于数据并行，即完整模型放置于一个 GPU 上（single-device module）时，需要提供该参数，表示将模型副本拷贝到哪些 GPU 上。

对于模型并行的情况，即一个模型，分散于多个 GPU 上的情况（multi-device module），以及 CPU 模型，该参数比必须为 None，或者为空列表。

与单机并行一样，输入数据及中间数据，必须放置于对应的，正确的 GPU 上。

output_device
int 或者 torch.device，参考 DataParallel。

对于 single-device 的模型，表示结果输出的位置。

对于 multi-device module 和 GPU 模型，该参数必须为 None 或空列表。

broadcast_buffers
bool 值，默认为 True

表示在 forward() 函数开始时，对模型的 buffer 进行同步 (broadcast)

process_group
对分布式数据（主要是梯度）进行 all-reduction 的进程组。

默认为 None，表示使用由 torch.distributed.init_process_group 创建的默认进程组 (process group)。