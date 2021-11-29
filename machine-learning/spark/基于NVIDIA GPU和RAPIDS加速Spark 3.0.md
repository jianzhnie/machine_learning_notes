**- 前言 -**

Spark 3.0 开始支持了数据的列式处理，同时能够将 GPU 作为资源进行调度。在此基础上，Nvidia/Spark-Rapids 开源项目基于 Rapids 库, 以 plugin 的方式提供了一组 GPU 上实现的 ETL 处理，利用 GPU 强大的并发能力加速 Join ,  Sort ,  Aggregate 等常见的 ETL 操作。


本次分享主要介绍该开源项目和目前取得的一些进展，以及使用到的一些相关技术。



**▌用于Apache Spark的RAPIDS加速器**

![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiarFQaj6NWOKQHnN8w2GnGljMzmvkEzuPbHhlR7KDvunwcd3XhCciaS9otkIRdcd9pNC1ELB0fu5nA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



大家看这张图都能联想到Hadoop很经典的一个标志一头大象，现在都是大数据时代了，面对海量数据的这种场景一直在不断地演进一些新的适应的硬件、一些新的软件架构。从最早的Google发的包括MapReduce、GFS等等的一些新的paper，然后到业界开源的一些新的软件生态体系，比如说Hadoop体系、基于Hadoop的文件系统、计算框架比如说HDFS、Hive、Spark。现在在各个互联网大厂，甚至不只是互联网公司，其他包括工业界的应用也非常的多。传统的这种大数据的处理框架都是基于CPU硬件的，GPU硬件已经发展了很多年，它其实在AI领域在深度学习领域已经取得了很好的效果。大家可能会有一个疑问，就是GPU能不能让大数据领域大数据处理跑得更快，对于传统的ETL流程能不能有一个比较好的加速效果？结果大家通过一些比较感知上的一些认识，可能会觉得还挺合适的，因为大数据天然的数据量非常的大，第二个特点是它的数据处理任务并行度非常高，这两个特点是非常适合GPU来执行的，对于GPU来说是非常亲和的。




![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiarFQaj6NWOKQHnN8w2GnGlsG5PGUnI7ovCKVeLibSesmol2Xq97tOibBBNmxklTKKRAGIs7bR1FLgA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

我们给的一个结论“是的”，通过GPU的加速，我们在TPCX-BB Like数据集上测试的一个结果（上图），可以看到相对于原始的CPU版本的query，我们测了这个图中大概四个query它的执行时间分别是25分钟、6分钟、7分钟、3分钟，经过GPU版本的执行，它的时间都缩短在一分钟上下左右，甚至最后的query只有0.14分钟。我们用的数据集是10TB的一个数据集，一个比较有参考性的大小，然后用的硬件规格是一个两节点的DGX-2集群，DGX-2是一个搭载了16张NVIDIA V100 显卡的 AI 服务器。

![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiarFQaj6NWOKQHnN8w2GnGlaDjicOgkVDjPZO9yGvnzrVLQZBNAjJhZZT408HInbsJFYsibvjQCqv7A/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

再比如说很多互联网场景推荐场景上，我们是不是能达到比较好加速效果？因为目前很多互联网公司推广搜推荐场景可能会涉及到，比如说短视频推荐、电商推荐、文本推荐，目前面临的一个问题是推荐场景本身，互联网公司它的业务覆盖的用户规模越来越大，内容本身也处在一个内容爆炸的一个时代，有海量的UGC的内容。一方面用户的数量的规模扩大，另一方面内容的数量量级的规模的扩大，对于整个的ETL训练的挑战都是非常大的。我们给出了一个DLRM经典的推荐模型在CRITEO数据集上的一个表现，达到了大概的一个加速效果是怎么样？我们依次看一下这四个版本的数据，最原始的版本还没有分布式的训练数据处理框架诞生之前，对于这种ETL的流程可能就是用一种单核或者说单机多核的这种方式去处理ETL的时间大概能到144小时，训练的时间我们用多核的去训练的达到45个小时。从最原始的版本的改进，我们可以说比如说用Spark这种形式比较先进的分布式计算框架去做ETL，这个时候它的ETL的流程能缩短到12小时。我们还可以怎么继续改进，比如说我们在训练的这一段，从传统的多核的CPU切换到GPU训练，我们这边举了一个例子，是用了单张的V100去做训练，训练的时间从之前的45个小时缩短到0.7个小时，最终其实就是今天要highlight的主题，就是说我能把ETL这部分如果也切到GPU训练，大概能达到一个怎么样的效果？我们这边举的一个例子是用了8张V100 显卡做ETL，最终的ETL的时间能缩短到0.5小时，整体的加速比从最早的时间是大概提升了160倍，到比较先进的CPU方案仍然有个48倍的提升效果，只用了4%的成本。比目前比较主流的方式就是CPU做ETL，然后用GPU做训练，我们仍然能达到一个10倍的加速效果，但是只有1/6的成本。


![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiarFQaj6NWOKQHnN8w2GnGl2EklejClB0fsaFhq6YkCc7KAy4U2wGqu8wc8RlXbwTIFp7ubmoQTOg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

这其实是去年的GTC2020，老黄发布的其实是一个比较经典的一个语录，就是“买得越多，省得越多”。这句话不无道理，对于一些原本的一些大数据的处理流程，是不是可以利用一些新的一些硬件特性，新的一些处理范式，取得更好的一个性价比，达到一个更小的成本？其实给的这个答案是的。

![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiarFQaj6NWOKQHnN8w2GnGlaYBRo3IREeJAyqiaGhfZv6NLtibPJjXZfUgfWkTqkMe9ND3Mhia1d1F7Q/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

如果我们想用GPU去做加速Spark处理，我们需要去改动多少的代码规模？其实这个是大数据工程师、数据分析师非常关心的一个问题，我们这块儿给的答案是对于Spark SQL和DataFrame代码其实并不需要做代码的任何的更改，也就是说你的业务代码是不用变，只不过是我们在配置项的时候，我们会看到第一行开头会把“spark.rapids.sql.enabled”设成true，就是一个配置项的改动，然后让spark-rapids生效，后面的这些业务代码都是保持不变，它的实施成本是非常低。

![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiarFQaj6NWOKQHnN8w2GnGlrvaBHO9uLRuB98f5wDicuq80ZewcfR0lIGQdXFVc48aba32ZHibQNuFw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

对于很多Spark SQL和DataFrame的里头的算子，这个算子的数目也是非常庞大的，我们也是需要去一个一个去做适配。目前我们可以整体的看一下这张图，就是说支持的算子规模应该是非常大的。没有再支持的这些算子，我们也非常欢迎大家反馈给我们，可以在github上去给我们提feature request。我们也非常迫切的想知道工业界里头具体哪些算子其实是用的频率非常高，但是实际上我们还没有去尽早的支持，这对于我们改进这个产品也是非常重要。

![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiarFQaj6NWOKQHnN8w2GnGlDCHwUibHvBbqBG3ZGWIn1H4dhbFs2Y1WbTM8Ijc9OmNw1RHQe0picHDw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

Spark如果是跑在GPU上是不是能解决传统的CPU上的所有的问题？其实是不一定的，我们有一个客观的分析，对于某一些场景图左边举的这些例子，其实并不一定说在GPU上跑就能达到一个很好的效果。比如说数据规模特别的小，整个的数据的size可能会特别小，但具体到每个partition的话，如果我设的partition数也是比较多一点，其实可能partition它的数据大小只有几百兆，不一定适合跑在GPU上。第二种场景就是说高缓存一致性的操作，这一类的操作如果在你的公司的业务的query里头包含的比例非常高的话，也不一定是GPU是十分合适的。第三类就是说包含特别多的数据移动，比如说我的整个的这些query有各种各样的shuffle，shuffle的比例非常的多，可能我的整个的操作是bound在IO层面，可能有网络、也可能有磁盘。还有一种可能就是UDF目前的实现可能会经常串到CPU，就是说还是会牵扯到CPU与GPU之间，可能会产生不断的一些数据的搬运。在这种情况下，就是数据移动特别多的情况下，GPU也不一定是很合适的。最后一种场景就是说我的GPU的内存十分有限，主流的英伟达GPU的显存也都是看具体型号，最新的A100也都能支持到80G，但是可能对于特定的场景来说，可能内存还有可能不够，如果是这种比较极端的情况下，也有可能说处理不了，也有可能说在GPU上的加速效果并不一定是十分的明显。右图非常清晰得展示了各个环节的吞吐大小是怎样的，从最左边看如果你是经常需要写磁盘、网络环境并不是十分高配的网络架构、数据移动比较多的话，经常会bound到这些地方。

![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiarFQaj6NWOKQHnN8w2GnGlPibmUzJjyBTjP9O2999VBuCuoRwicgT21vdtAYxrv0tp0Awa7xBrsPOw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

但是我们在GPU上跑spark仍然还是有很多的一些任务，它是十分的适合GPU场景。我们举一些具体的例子：

**1. 高散列度数据操作**

比如说高散列度的这三类操作，比如joins、aggregates、sort，这些其实都是一些基于shuffle的操作。高散列度具体指的是某一个column，它不同的值的数量除以整个的column的数量，或者简单理解为不同的值的数量是不是比较大的，如果是高散列度的这种情况的，是比较适合用GPU来跑Spark。

**2. window操作比较多**

第二类是说Windows的window的操作特别多，特别是Windows size的比较大的情况下，也是比较适合于GPU的。

**3. 复杂计算**

第三类的话就是说复杂的计算，比如说写一个特别复杂的UDF，也是比较适合GPU的。

**4. 数据编码**

最后的一个是说数据编码，或者说数据的序列化、反序列化读取和写入，比如说创建Parquet、读CSV，在这种情况下，因为GPU我们有一些特定的针对IO的一些优化，对于这一块来说性能加速比较好。

**▌Spark Rapids工作原理**

![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiarFQaj6NWOKQHnN8w2GnGlvSS4owQBPDQTBibqaEJxqnSHV9iaK7icR9ibrtOL7lTibuejRhSCLQl8fdg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



**1. ETL技术栈**

我们大概的介绍一下Rapids Accelerator，它的工作原理是怎么样，整个的ETL的技术站可以如下图所示：

![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiarFQaj6NWOKQHnN8w2GnGlHnARxldUJWCnrD0JK0zCVc75V56Xv3ASyyrJ70guviaUyCmK3ZpujVQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

图左边的可以看一下是Python为主的技术栈，传统的包括我们用Pandas Kaggle竞赛，或者说做数据分析的时候可能用Pandas会比较多，操作DataFrame的数据，我们对应的也提供了 GPU 版本的 Pandas-like 实现，叫做 cuDF。在 cuDF 的基础上我们提供了分布式的 Dataframe 支持，它就是 Dask cuDF。这些基础库底层依赖的是Python和Cython。最右边是spark技术栈上我们对应的一些优化，对于Spark Dataframe和Spark SQL都有了对应的加速，然后对于Scala和PySpark也都有一些对应的优化的实现。然后它底层依赖最上层是Java，底层调用实际上是cuDF的C++API，中间的通信是通过JNI来通信库。cuDF也是依赖Arrow的内存数据格式。对于这类列式存储，我们在CPU的基础上，也提供了GPU的支持。最底层是依赖于英伟达的显卡的一个计算平台CUDA，还有依赖CUDA基础上搭建的各种底层的一些实现的一些底层库。

**2. RAPIDS ACCELERATOR FOR APACHE SPAK**

![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiarFQaj6NWOKQHnN8w2GnGlbibTzyKkRdTBfI8ia0X97CFobW4M7RVq4icxX1EDnsg6XpibvicYlUos6zg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

我们今天要关注的RAPIDS Accelerator它的整个架构是怎么样？可以先从上图中最顶上看，最顶上是具体的算法工程师或者说数据分析师写的Spark任务在中间这一层是Spark core。左边这块我们目前已经实现加速的是spark SQL和DataFrame的API。刚才前面也讲到，我们是不需要去更改任何的业务代码，对于所有的这些业务代码之中描述的这些操作，这些算子来说，我们提供了RAPIDS Accelerator可以自动的去识别对应的操作数据类型，是不是可以调用Rapids来进行GPU加速，如果是可以的话，就会调用Rapids，如果是无法加速的话，就会执行标准的CPU操作，整个调度对于用户来说，对于实际写Spark应用的人来说是透明。右边这块是对于Shuffle的支持，Shuffle也是Spark很关键的一个性能瓶颈。对于Shuffle的流程，我们具体是做了哪些优化？对于GPU和RDMA/RoCE这种网络架构下，我们实现了一套新的Shuffle，在底层使用了UCX来达到一个更好的一个加速效果。

**3. SPARK SQL & DATAFRAME编译流程**

![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiarFQaj6NWOKQHnN8w2GnGliahiaMvw8yBZSyOUdro3MrSXiak98aFJYhmHeBOVLgfWsicObDfqx4NIicA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

整个的Spark SQL和DataFrame的一个编译流程是如上图所示，最上层是Dataframe在Logic Plan这一层还是不变，经过 Catalyst 优化，生成Physical Plan之后，对应到GPU的版本我们会生成GPU的Physical Plan，具体输出的数据是ColumnarBatch的RDD。如果需要把数据转回CPU处理的话，会再把RDD转回InternalRow的RDD。

![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiarFQaj6NWOKQHnN8w2GnGluuhuPFvgoUODFwdibFyF9wpRD5zzvibtJrficORAoeG9nzwLr6oUQcIUQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

具体的Spark SQL和DataFrame的执行计划，会对应到GPU的plugin，如果采用后会产生哪些变化，给出了一张比较详细的图。

![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiarFQaj6NWOKQHnN8w2GnGlQcF5mbeL4y594ztIEzARDWJkculuADkoE1DnmtkgwflHvmQN6BbYOw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

大家可以感兴趣的话可以自己测试一下。具体是对应到一个CPU的operation，如果是能用GPU优化的话，是能一对一的是去map到GPU的一个版本，如果说大家想自己去测一下GPU的版本Spark处理效果能达到一个怎么样的一个加速比，DataBricks提供了一个比较标准的Spark SQL生成数据的一个工具。我们主要也是依赖这个工具去做了一些 benchmark，主要的参数可以参考一下，我们用的选择scale factor是用的3TB，也用到了decimals和空类型。输入的input partitions的数目是400，shuffle的partitions是用的200，所有的输出的结果会写到S3上。

**4. 效果**

![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiarFQaj6NWOKQHnN8w2GnGl9ESULRfhEXlN8tVsBRAj5KZ7srEDicevbsiau7Eicia4yT1tBohARGpRsQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

整个的加速比的效果是怎么样？TPC-DS数据集上它加速比是怎么样？我们可以看一下QUERY 38的加速效果。具体选用的CPU的硬件的标准和GPU的硬件标准，都是AWS的标准的硬件单元，价格也都是非常透明。如果是从查询时间上来看的话，相比于CPU版本的话，大概有三倍的提升。虽然可以看到最底下GPU的硬件，我们用的是一个driver是一个CPU的driver，worker是用一个八节点的单GPU的配置，在这种情况下，每小时的cost会是高一点，但是整个query时间有了三倍的提升，最终算下来的话，我们大概节省了55%的成本。

![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiarFQaj6NWOKQHnN8w2GnGlCEN30l1OiaCl6aBibsxXfcuszWOlM1pc9QSI6UVR0q7icsZss8yQVRic5Q/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

Query 5也是一个比较经典的一个查询，它特别在哪儿？因为它重要部分都没有被GPU加速，只有少量部分被GPU加速，因为具体来说的话是它的Decimal Type还没有被GPU支持。在这种情况下，GPU版本也取得了一个比较好的性价比收益，相对于它的查询时间来说，是有1.8倍的速度的提升，成本上来说仍然能节省23%的成本。对于大家对于想从Spark3.0的集群CPU目前的架构过度到GPU架构的话，这是一个比较有参考性的一个例子，因为我们目前的Rapids Accelerator一直在紧锣密鼓的在迭代之中。目前的版本来说，即便不是所有的query都能被GPU加速，但是仍然还是能取得一个比较好的一个性价比。

**▌加速Shuffle**

![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiarFQaj6NWOKQHnN8w2GnGlvcYavFBLu0ic6Ak6cVNaxw4B9gwOqFtDKGCGEmwHTSlicvRwG5sRnInw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

我们会主要讲一下Shuffle，Spark Accelerator做了什么，它为什么要针对Shuffle做加速。传统的Shuffle大家如果是对 Spark比较熟悉的话，这块也不用再赘述了，其实就是牵涉到我们在某一些特定的一些操作，比如说join、sort、aggregate，要牵涉到节点是节点之间，或者说executor跟executor之间要做一些数据的交换，要通过网络做一些数据的一些传输，前一个stage跟后一个stage之间会产生一些数据的一些传输，就是牵涉到前一个stage要做一些数据的一些准备，然后把数据写到磁盘，然后通过网络把数据拉取过来，这中间可能也会牵扯到一些磁盘IO，然后把数据规整好。

![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiarFQaj6NWOKQHnN8w2GnGlYgzMjst7Ssury1sTZ3GbfC9kTib7ibl1vU7RVTPmjvSP1qIZ69icfTVTQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

传统的Shuffle的它的流程如上图所示，这张图是基于CPU的目前的一个硬件环境，Shuffle它的数据的搬移具体是怎么样一个流程，可以看到如果是我们不做任何的优化的话，即便是数据存在GPU的显存上，它也要经过比如经过PCI-e，然后才能去走网络，走本地的存储。可以看到有很多不必要的一些步骤，然后产生了一些额外的开销，比如说没有必要一定要经过PCI-e。

![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiarFQaj6NWOKQHnN8w2GnGlMFsQBX59gm3hUlLWv6mv6S6RJIFmXgx5GMez6azem2MliaPytn7GofA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

经过我们的优化的Shuffle之后，它大概的一个数据的移动是表现是怎么样？首先如上图所示，第一张图描述的是说GPU的memory是足够用的。这种情况下这时候的Shuffle是怎么走的？如果在同一个GPU内的话，数据本身不需要搬移。如果是在同一个节点，如果我们采用的节点也是有NVLink的情况下，这个数据可以直接通过NVLink来传输，而不用走PCI-e，也不用经过CPU。如果这个数据是存在本地存储NVMe接口的存储，可以通过GPU Direct Storage去直接做读取，如果是远程的数据，我们可以直接通过RMDA去做加速，同样用同样的也都是bypass掉了CPU和PCI-e。

![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiarFQaj6NWOKQHnN8w2GnGlvdViaH5wzJ4MY3R92Kib8dnyx0L4icfzDnXQESrKzXiaFw6dtPSNX3O0ew/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

如果说Shuffle的时候，这时候数据规模是超出了显存容量的话，大家也都比较熟悉Shuffle spill机制，我们的RAPDIS的Shuffle是不是还是能有一定的优化？这个答案也肯定的。首先如果是GPU的memory超了之后，会往主存里头去写一部分，如果主存之后也写不下，其实类似于之前的CPU的方案，会把这个数据写到本地的存储里。但是对于储存的这部分数据来说，仍然可以通过RDMA获得加速。

![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiarFQaj6NWOKQHnN8w2GnGlzSs4ic9MA62Wq82M09dzRSw4v5sibHNSpIyh8YzVT10dacmicS5a9d0BQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

我们底层依赖的具体依赖的技术的组件，我们是用了UCX这个库，它提供了一个抽象的通信接口，但是对于具体的底层的网络环境和资源，比如说是TCP还是RDMA、有没有用Shared Memory、是不是有GPU，它都会根据具体的状况去选择一个最优的一个网络处理方案，对于用户来说是透明的，你不需要去具体的关心这些网络细节。能达到的一个效果是，如果是能利用上最新的性能最优的RDMA的话，我们是能达到一个零拷贝的一个GPU的数据的传输。RDMA需要特定的一些硬件的一些支持。

![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiarFQaj6NWOKQHnN8w2GnGlIhnicxkl94uzsiaAwhojqjVZ2hueVBZtVa4micsjHflOnpk6FLSaoBVxg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

如果是采用UCX，能用RDMA这种网络架构的话，能达到一个怎么样的一个性能收益？我们这边举了一个具体的库存定价query的例子，CPU的执行时间不是228秒，相对于GPU它大概就能达到一个五倍的一个提升。如果在对于网络这块再做进一步的一些优化的话，其实可以看到是能缩短到8.4秒，整体看是有30倍左右的性能提升，这个提升还是非常明显。所以其实可以大家也可以看到，整个的计算流程其实主要是bound在网络这一块。

![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiarFQaj6NWOKQHnN8w2GnGlRrdyqNsItpibiaW96uqerOKCJWYUrzJ41OXbxCJrx3nEnq0w90ovebEA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

对于传统的逻辑回归模型，它的ETL的流程，也是能达到一个比较明显的一个收益，最原始版本是1556秒，最终优化的版本的话是76秒就可以执行完整个的ETL流程。

**▌0.2版本中的亮点**

![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiarFQaj6NWOKQHnN8w2GnGlPYz42fPwOZFHIrXtj7zBT8xWd5qnJL8BR8LU9B5lXCVvHyx8MIR1gA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

我们过一下，我们最近有Rapids的两个版本：0.2版本和0.3版本，大概都包含哪些新的一些特性。



**1. 多版本SPARK的支持**

对于从0.2版本开始，除了对于 Apache 社区版本的支持，对于Databricks 7.0ML和Google Dataproc 2.0，也都有对应的支持。

![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiarFQaj6NWOKQHnN8w2GnGlhat2x17pMgcLdwT9nnNbD417kZichNgnabcJFibs7BdtSZbEdpj0rBOg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**2. 读取小文件时的优化（PARQUET）**

![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiarFQaj6NWOKQHnN8w2GnGliafFgcB1DyNmibGWWIhnE0H57O38BYbByAgtOJXuhez8z7tnHquKJUWg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

第二个feature是对于读取较小的parquet文件的时候做了一些性能的优化，简单来说可以借助CPU的线程去并行化处理parquet文件的读取，实现CPU任务跟GPU任务能够互相覆盖，GPU在进行实际计算的时候，CPU也会同时去load data，能达到6倍的性能提升。

**3. 初步支持Scala UDF**

![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiarFQaj6NWOKQHnN8w2GnGlGRaHgbHHW1x779Vw1uVS577wfkspI4tNYuN0lv3q4j2cKe8CUUgM5w/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

对于Scala UDF的话，也有了一些初步的支持，目前支持的算子不是特别多，但是也可以具体的跑一些例子可以看一下，就是说对应的实际的用到的UDF是不是已经可以被编译成GPU版本，如果是GPU版本的话，其实应该是能达到一个比较好的一个性能收益。

**4. 加速Pandas UDFS**

![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiarFQaj6NWOKQHnN8w2GnGlqkG20SShiakMlCXZVNIOEGDFF0x1DJUPNB2IXTGVE7OKibCBcqVKquRA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

如果是Pandas用户的话可能会经常会用到Pandas UDF，对于这一块来说是RAPIDS的加速器也做了实现，具体实现的细节其实是可以使JVM进程和Python进程共享一个GPU，而且这样的一个好处是可以优化JVM进程跟Python进程之间的一些数据交换。

![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiarFQaj6NWOKQHnN8w2GnGl0F7dibJjfwtOX2icmkrcnhW1F7GShXXsuoU5JaNHFuLLWY66DGL0dzFA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

具体的实现细节可以如上图所示，目前的实现，单个GPU上我们可以跑一个JVM进程的同时，可以跑多个Python进程，可以配置具体的Python进程数，对于Python进程使用的GPU显存总量也都是可配置。

![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiarFQaj6NWOKQHnN8w2GnGlXeERwJVibMc4UexWPfKdkibqDPVXUDpVuEdy9ELWPcrLRib20Xdta0NpA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

相比于传统的CPU方法的优化来说，GPU版本其实是更加亲和的，因为都利用了列式存储，不牵扯到行式数据到列式数据转换的开销。在这种情况下加速收益还是比较明显。

![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiarFQaj6NWOKQHnN8w2GnGlHTzlLTCgdthF6Vwn0OUOF9GticrCG4I2SpibUH85B4ia6ibQTlLTgB6pCg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**▌RELEASE 0.3**

![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiarFQaj6NWOKQHnN8w2GnGlwiaOqwfMichdBrIEj61TOIXxAXEJjUOxRF3cJDvajic1ibmPqR96cxlpew/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

实现了一个per thread的默认流，然后通过这种方式能够更好地提高GPU的利用率。对于 AQE，也有了进一步的优化，对于3.0版本，如果这一部分的执行是可以跑在GPU上的话，会自动搬到GPU，在GPU上AQE总体达到可用状态。UCX版本升到最新的1.9.0。对于一些新的功能，对于parquet文件读取的支持，支持到lists和structs，然后对于窗口操作的一些算子，新增了对lead和lag的支持。对于普通的算子，添加了greatest跟least。

![图片](https://mmbiz.qpic.cn/mmbiz_png/zHbzQPKIBPiarFQaj6NWOKQHnN8w2GnGl9jtr1YoLa1S69oo4mxEGtViapia2tDkaWFEW5ia5hPwo5sl4DcEl2uUNw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

如果大家想要去获取rapids accelerate for spark的一些更多的一些信息的话，我们可以去直接通过NVIDIA的官网，可以直接联系到NVIDIA的 Spark Team，整个项目也是开源在github上。对于想获得Spark Accelerator比较新的、全面的信息的话，可以去下载Spark电子书，电子书目前是有中文版本。

**使用 RAPIDS 加速 Apache Spark 3.0**



- **Apache Spark在现代企业中的应用**



![图片](https://mmbiz.qpic.cn/mmbiz_png/QDdcCFX9vIeF7xDKCZ9oIG7LAUfpOb95YHnHRPRyejQIv1fW6e5nkApwXFWpOia92XY7OEBh9a28vBHgzorxanw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)





- **结合** **NVIDIA GP****U** **的** **Spark** **3.0**



![图片](https://mmbiz.qpic.cn/mmbiz_png/QDdcCFX9vIeF7xDKCZ9oIG7LAUfpOb95hxReojxlpylsibCDkpKhE2dNGAflS0UXbchAMdOxfW6EiazwaYofu2Pw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)





- **为什么要迁移到** **Spark** **3.0**

![图片](https://mmbiz.qpic.cn/mmbiz_png/QDdcCFX9vIeF7xDKCZ9oIG7LAUfpOb951S7V1HZrWyQ6uzdYTyNIfqHrnHU15Iq0TSyRUiayGEqtLyJajoHmxag/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)





- **英****伟达提供 G****PU 加****速的 APACHE SPARK**

![图片](https://mmbiz.qpic.cn/mmbiz_png/QDdcCFX9vIeF7xDKCZ9oIG7LAUfpOb95icScNA7VdUYjs1I9x7EsdGGRqrxG87XRqIDaUHLHSU2CsHpzQ4eg7ew/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



- **那么它是如****何****工作的****？**



![图片](https://mmbiz.qpic.cn/mmbiz_png/QDdcCFX9vIeF7xDKCZ9oIG7LAUfpOb95u0TaiabeWL8iba8DcPlkCagM4lFMicZHO8GgawSXnJKCjEKJ7weTTYq2Q/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



- **SPARK SQL &** **DATAFRAME 编译流程图**

![图片](https://mmbiz.qpic.cn/mmbiz_png/QDdcCFX9vIeF7xDKCZ9oIG7LAUfpOb95v0lyTUxQ1wic7UicsDoCNiayY5uVu7pE9x71FtpGaCVb8DZ79bJibjLohA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



- **SPARK SQL &** **DATAFRAME 编译流程图**

![图片](https://mmbiz.qpic.cn/mmbiz_png/QDdcCFX9vIeF7xDKCZ9oIG7LAUfpOb95mq3V9kacm5q322s2sCicMpFPlZxMibL0yBryOcObuPmyPUIZuMvchR2g/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



![图片](https://mmbiz.qpic.cn/mmbiz_png/QDdcCFX9vIeF7xDKCZ9oIG7LAUfpOb95pGHzpyI520JeJrCb8bFYzZaousX2EiabicBgF0qKWC6rWkosfEhrN01Q/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)





![图片](https://mmbiz.qpic.cn/mmbiz_png/QDdcCFX9vIeF7xDKCZ9oIG7LAUfpOb95DJQDsP5icQa5tevBxMhhHnjIk7icUiaFJqz42Xl3EKuPwFtvKNm44muDg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



- **适合 GPU 的场合**

-高散列度数据的 joins

-高散列度数据的 aggregates

-高散列度数据的 sort

-Window operations (特别是大型 windows)

-复杂计算

-数据编码（创建 Parquet 和 ORC 文件 读取 CSV)



- **并非适合于所有场合**

![图片](https://mmbiz.qpic.cn/mmbiz_png/QDdcCFX9vIeF7xDKCZ9oIG7LAUfpOb95yHE6MeJ6diaCxpOoIRLSq8zVQ8iamibRtp6sH9ibQcoBdLGRkR7ZkOmicxg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



**GPU** **处理加速的一些技术**



- **加速 SHUFFLE**

![图片](https://mmbiz.qpic.cn/mmbiz_png/QDdcCFX9vIeF7xDKCZ9oIG7LAUfpOb95Otj6X3vTb4opGkbMrN65a0ib6kcibtUeIQ9oTyAIWdFXXsAgmwY4XRGQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![图片](https://mmbiz.qpic.cn/mmbiz_png/QDdcCFX9vIeF7xDKCZ9oIG7LAUfpOb951nMJtwPokLRxfn43GM8D5hbicDE4ibypvBjjIoj2j6QEuicicgib9KusicnQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![图片](https://mmbiz.qpic.cn/mmbiz_png/QDdcCFX9vIeF7xDKCZ9oIG7LAUfpOb95PwcZ1X12BqAmM4XGIaf3Uvluop3a4uic7vZFia1xg7ZvoD3qIqWBzRmA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![图片](https://mmbiz.qpic.cn/mmbiz_png/QDdcCFX9vIeF7xDKCZ9oIG7LAUfpOb95SYHhKgQIZwHBV4HfcOk4B7V2ia6W2sDqbRmU2C631Ge9L6kF8pcPDYg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![图片](https://mmbiz.qpic.cn/mmbiz_png/QDdcCFX9vIeF7xDKCZ9oIG7LAUfpOb95KvWryalQOk4D1cew58BQvtU7kaXyXhlkxZ13QdlWtYlVgqZ7V0X7iag/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

- **UDF 的部分支持**

![图片](https://mmbiz.qpic.cn/mmbiz_png/QDdcCFX9vIeF7xDKCZ9oIG7LAUfpOb95ozD2mcziaQecNnk98GXCGYyEVIkY3Dmr6mwH06CTEeeg3xiaBZo1kpfg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![图片](https://mmbiz.qpic.cn/mmbiz_png/QDdcCFX9vIeF7xDKCZ9oIG7LAUfpOb95fiaVI22ibEUV9daezk4ibkIat90yQl72ibFia9JcejNZ545JiaMaJwHL73Ww/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

- **SCALA / JAVA / HIVE UDF 支持**

-提供 Scala/Java/Hive 定义的 UDF

-调用已有 cuDF 提供的 Java API 执行 GPU 处理

-自定义 GPU native 处理

-详细参考：rapids accelerated user defined functions



- **其他技术-加速 GPU 处理**

-小文件读取时，采用多线程并行以及合并 ( coalesce ）

-GPU 内存不够时， spill 到 Host 内存和磁盘

-采用最新的 Gpu Direct Storage 技术，加快 GPU 内存到磁盘 NVMe 的 I/O



**SPARK 3.0 性能测试结果**

![图片](https://mmbiz.qpic.cn/mmbiz_png/QDdcCFX9vIeF7xDKCZ9oIG7LAUfpOb955CE8NPnVjEic6DrK0UBpCDQAFuxnHD2xb4K2LYkONEibMia7hHu8AviadA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![图片](https://mmbiz.qpic.cn/mmbiz_png/QDdcCFX9vIeF7xDKCZ9oIG7LAUfpOb95b5pxBqVVLOABFRd2icqypWwIYYt10WLKgtZWHPdo4Zqjc5G0ZnicwykA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![图片](https://mmbiz.qpic.cn/mmbiz_png/QDdcCFX9vIeF7xDKCZ9oIG7LAUfpOb95ECdrWDr1tEE9Vqqmw4aXONN5ich4BlwOzUpJEbENXwhZDYicn5dHqOCA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![图片](https://mmbiz.qpic.cn/mmbiz_png/QDdcCFX9vIeF7xDKCZ9oIG7LAUfpOb95icP7tx3G5W9ichCokyLep2rf7ic3RrskIYwu8ZP7OiamE3mfO6WOd3ic80Q/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![图片](https://mmbiz.qpic.cn/mmbiz_png/QDdcCFX9vIeF7xDKCZ9oIG7LAUfpOb95evXzp4k0zJM0jT7K7mHzeoKPy6SlErOU2ziaZ5x3MsKaDD8icur7ec6g/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![图片](https://mmbiz.qpic.cn/mmbiz_png/QDdcCFX9vIeF7xDKCZ9oIG7LAUfpOb95FrFUjkGCwHgHsq4B0Po7jEOtbEIYgibgTkrwia9AvGL6Nuvic6rVXiaROw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![图片](https://mmbiz.qpic.cn/mmbiz_png/QDdcCFX9vIeF7xDKCZ9oIG7LAUfpOb95f7FibnnrcOTvuLJfYNSSmt6r8RpcCWQPlopxWH4Vrgdemiaga9Jge9KA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![图片](https://mmbiz.qpic.cn/mmbiz_png/QDdcCFX9vIeF7xDKCZ9oIG7LAUfpOb95xhA9E5eJjc30KwWsIWb9QqKQGAVPq19qU7pat3r9tOFXS4dcpcH8eA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![图片](https://mmbiz.qpic.cn/mmbiz_png/QDdcCFX9vIeF7xDKCZ9oIG7LAUfpOb95Ia9wIjBvMctuKWUc6jfNkzjZBCm5aNfJFyBFh3IreT6yzmHNtoSPFw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



对以上相关内容感兴趣，可以点击**阅读全文**，或扫描文章底部二维码加入钉群观看回放~
