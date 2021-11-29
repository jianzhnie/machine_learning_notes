作者：周威
链接：https://zhuanlan.zhihu.com/p/181709693
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。



MAML，全称呼叫做**Model-Agnostic Meta-Learning** ，意思就是**模型无关的元学习。**所以MAML可**并不是一个深度学习模型**，倒是更像一种**训练技巧**。

如果你对few-shot learning 或者meta learning的**基础知识**不懂，那么我并**不推荐**你去**直接**看论文，那会让你想放弃对这个领域的学习。

根据我的学习经验，我非常推荐你去看以下**李宏毅老师**的教学视频。链接如下

[https://www.bilibili.com/video/BV15b411g7Wd?p=57www.bilibili.com/video/BV15b411g7Wd?p=57](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV15b411g7Wd%3Fp%3D57)

我就是看了两遍视频后，并根据知乎上的文章，加深了对MAML的理解

[徐安言：Model-Agnostic Meta-Learning （MAML）模型介绍及算法详解879 赞同 · 126 评论文章![img](https://pic3.zhimg.com/equation_ipico.jpg)](https://zhuanlan.zhihu.com/p/57864886)

这里，我们根据的代码是**基于Pytorch**的，链接如下：

[https://github.com/dragen1860/MAML-Pytorchgithub.com/dragen1860/MAML-Pytorch](https://link.zhihu.com/?target=https%3A//github.com/dragen1860/MAML-Pytorch)

## 2. 简单谈谈MAML

MAML 的中文名就是**模型无关的元学习**。意思就是不论什么深度学习模型，都可以使用MAML来进行少样本学习。论文中提到该方法可以用在**分类**、**回归**，甚至**强化学习**上。

本文我们的代码是基于分类的，那么我们就从**分类的角度**展开对MAML的解析。

**2.1 Meta Learning的一些基础知识**

Meta Learning（元学习）,也可以称为“**learning to learn**”。常见的深度学习模型，比如对猫狗的分类模型，使用较多的是卷积神经网络模型，可以是VGG/ResNet等。

那么我们构建好了模型后，学习的就是**模型的参数**，学习的目的就是使得**最终的参数**能够在训练集上达到**最佳的精度**，**损失最小**。

但是元学习面向的是**学习的过程**，并不是**学习的结果**，也就是元学习不需要学出来最终的模型参数，学习的更像是**学习技巧**这种东西（这就是为什么叫做**learning to learn**）。

举个例子，人类在进行分类的时候，由于见过太多东西了，且已经学过太多东西的分类了。那么我们可能只需每个物体一张照片，就可以对物体做到很多的区分了，那么人是怎么根据少量的图片就能学习到如此好的成果呢？

- 显然 ，我们已经掌握了各种用于图片分类的较巧了，比如**根据物体的轮廓**、**纹理**等信息进行分类，那么**根据轮廓**、**根据纹理**等区分物体的方法，就是我们在meta learning中需要教机器进行学习的**学习技巧**。

本文介绍的MAML，其实是一种**固定模型**的meta learning ,可能会有人问

- 不是说MAML是模型无关的吗？为什么需要固定模型？

**模型无关**的意思是该方法可以用在**CNN**，也可以用在**RNN**，甚至可以用在**RL**中。但是MAML做的是固定模型的结构，只学习**初始化模型参数**这件事。

什么意思呢？就是我们希望通过meta-learning学习出一个**非常好**的**模型初始化参数**，有了这个初始化参数后，我们只需要**少量的样本**就可以快速在这个模型中进行收敛。

那么既然是**learning to learn**，那么输入就不再是单纯的数据了，而是一个个的**任务**（task)。就像人类在区分物体之前，已经看过了很多中不同物体的**区分任务**（task)，可能是猫狗分类，苹果香蕉分类，男女分类等等，这些都是一个个的任务task。那么MAML的输入是一个个的task，并不是一条条的数据，这与常见的机器学习和深度学习模型是不同的。

**2.2  N-way  K-shot learning**

用于**分类任务**的MAML，可以理解为一种**N-way  K-shot learning**，这里的N是用于分类的类别数量。K为每个类别的数据量（用于训练）。

什么意思呢？我觉着[这篇文章](https://zhuanlan.zhihu.com/p/57864886)解释的就很到位。以下作为引用：

> MAML的论文中多次出现名词**task**，模型的训练过程都是围绕task展开的，而作者并没有给它下一个明确的定义。要正确地理解task，我们需要了解的相关概念包括![[公式]](https://www.zhihu.com/equation?tex=%7B%5Cmathcal+D%7D_%7Bmeta-train%7D),![[公式]](https://www.zhihu.com/equation?tex=%7B%5Cmathcal+D%7D_%7Bmeta-test%7D),**support set**,**query set**,**meta-train classes**,**meta-test classes**等等。是不是有点眼花缭乱？不要着急，举个简单的例子，大家就可以很轻松地掌握这些概念。
>
> 我们假设这样一个场景：我们需要利用MAML训练一个数学模型模型![[公式]](https://www.zhihu.com/equation?tex=M_%7Bfine-tune%7D)，目的是对未知标签的图片做分类，类别包括![[公式]](https://www.zhihu.com/equation?tex=P_1%EF%BD%9EP_5)（每类5个已标注样本用于训练。另外每类有15个已标注样本用于测试）。我们的训练数据除了![[公式]](https://www.zhihu.com/equation?tex=P_1%EF%BD%9EP_5)中已标注的样本外，还包括另外10个类别的图片![[公式]](https://www.zhihu.com/equation?tex=C_1%EF%BD%9EC_%7B10%7D)（每类30个已标注样本），用于帮助训练元学习模型![[公式]](https://www.zhihu.com/equation?tex=M_%7Bmeta%7D)。我们的实验设置为**5-way 5-shot**。
>
> 关于具体的训练过程，会在下一节MAML算法详解中介绍。这里我们只需要有一个大概的了解：MAML首先利用![[公式]](https://www.zhihu.com/equation?tex=C_1%EF%BD%9EC_%7B10%7D)的数据集训练元模型![[公式]](https://www.zhihu.com/equation?tex=M_%7Bmeta%7D)，再在![[公式]](https://www.zhihu.com/equation?tex=P_1%EF%BD%9EP_5)的数据集上精调（fine-tune）得到最终的模型![[公式]](https://www.zhihu.com/equation?tex=M_%7Bfine-tune%7D)。
>
> 此时，![[公式]](https://www.zhihu.com/equation?tex=C_1%EF%BD%9EC_%7B10%7D)即**meta-train classes**，![[公式]](https://www.zhihu.com/equation?tex=C_1%EF%BD%9EC_%7B10%7D)包含的共计300个样本，即![[公式]](https://www.zhihu.com/equation?tex=%7B%5Cmathcal+D%7D_%7Bmeta-train%7D)，是用于训练![[公式]](https://www.zhihu.com/equation?tex=M_%7Bmeta%7D)的数据集。与之相对的，![[公式]](https://www.zhihu.com/equation?tex=P_1%EF%BD%9EP_5)即**meta-test classes**，![[公式]](https://www.zhihu.com/equation?tex=P_1%EF%BD%9EP_5)包含的共计100个样本，即![[公式]](https://www.zhihu.com/equation?tex=%7B%5Cmathcal+D%7D_%7Bmeta-test%7D)，是用于训练和测试![[公式]](https://www.zhihu.com/equation?tex=M_%7Bfine-tune%7D)的数据集。
>
> 根据5-way 5-shot的实验设置，我们在训练![[公式]](https://www.zhihu.com/equation?tex=M_%7Bmeta%7D)阶段，从![[公式]](https://www.zhihu.com/equation?tex=C_1%EF%BD%9EC_%7B10%7D)中随机取5个类别，每个类别再随机取20个已标注样本，组成一个**task**![[公式]](https://www.zhihu.com/equation?tex=%7B%5Cmathcal+T%7D)。其中的5个已标注样本称为![[公式]](https://www.zhihu.com/equation?tex=%7B%5Cmathcal+T%7D)的**support set**，另外15个样本称为![[公式]](https://www.zhihu.com/equation?tex=%7B%5Cmathcal+T%7D)的**query set**。这个**task**![[公式]](https://www.zhihu.com/equation?tex=%7B%5Cmathcal+T%7D)， 就相当于普通深度学习模型训练过程中的一条训练数据。那我们肯定要组成一个batch，才能做随机梯度下降SGD对不对？所以我们反复在训练数据分布中抽取若干个这样的**task**![[公式]](https://www.zhihu.com/equation?tex=%7B%5Cmathcal+T%7D)，组成一个batch。在训练![[公式]](https://www.zhihu.com/equation?tex=M_%7Bfine-tune%7D)阶段，**task**、**support set**、**query set**的含义与训练![[公式]](https://www.zhihu.com/equation?tex=M_%7Bmeta%7D)阶段均相同

作者的理解很到位，上面我们也说过MAML的数据是一个个的**任务**，而不是**数据**。

那么N-way K-shot就是一个个的任务。任务的类别为N，每个类别的Support set为K，至于query set大小需要**人为进行选择**（上例中选择了15，这是根据 ![[公式]](https://www.zhihu.com/equation?tex=P_1%EF%BD%9EP_5) 中“每类有15个已标注样本用于测试”决定的）。

**2.3 MAML算法流程**

MAML中是存在**两种梯度下降**的，也就是**gradient  by gradient**。**第一种梯度下降**是每个task都会执行的，而**第二种梯度下降**只有等batch size个task全部完成第一种梯度下降后才会执行的。

原文中是使用这样的**伪代码**进行MAML算法描述的。

![img](https://pic1.zhimg.com/v2-84ccb183dec8abaaab5866b5a860f278_b.jpg)

感觉看起来不是很直观，不妨看我下面的解析。

以上面的5-way 5-shot例子为例，这里我们简单叙述下MAML的算法流程。

- \1. 上面我们已经将数据区分成了 ![[公式]](https://www.zhihu.com/equation?tex=%7B%5Cmathcal+D%7D_%7Bmeta-train%7D) 和 ![[公式]](https://www.zhihu.com/equation?tex=%7B%5Cmathcal+D%7D_%7Bmeta-test%7D) ，在 ![[公式]](https://www.zhihu.com/equation?tex=%7B%5Cmathcal+D%7D_%7Bmeta-train%7D) 和和![[公式]](https://www.zhihu.com/equation?tex=%7B%5Cmathcal+D%7D_%7Bmeta-test%7D) 中我们又将数据区分了**support set**,**query set**
- \2. 我们用于训练的模型架构是 ![[公式]](https://www.zhihu.com/equation?tex=M_%7Bmeta%7D) （假设初始化参数为 ![[公式]](https://www.zhihu.com/equation?tex=%5Cphi) ），这可能是一个输出节点为5的CNN，训练的目的是为了使得模型有**较优秀**的**初始化参数。**最终我们想要学出可以**用于**数据集 ![[公式]](https://www.zhihu.com/equation?tex=%7B%5Cmathcal+D%7D_%7Bmeta-test%7D) **分类**的模型是 ![[公式]](https://www.zhihu.com/equation?tex=M_%7Bfine-tune%7D) ， ![[公式]](https://www.zhihu.com/equation?tex=M_%7Bfine-tune%7D) 和 ![[公式]](https://www.zhihu.com/equation?tex=M_%7Bmeta%7D) 的结构是**一模一样**的，不同的是**模型参数**。
- \3. 我们将**1个任务task**的**support set**去训练 ![[公式]](https://www.zhihu.com/equation?tex=M_%7Bmeta%7D) ，这里进行**第一种梯度下降**，假设**每个任务**只进行一次梯度下降，也就是 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7B%5Ctheta%7D%5E%7B1%7D%5CLeftarrow%5Cphi+-%5Ceta.%5Cpartial+l%28%5Cphi%29%2F%5Cpartial+%5Cphi) 。那么执行第2个task训练时，有 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7B%5Ctheta%7D%5E%7B2%7D%5CLeftarrow%5Cphi+-%5Ceta.%5Cpartial+l%28%5Cphi%29%2F%5Cpartial+%5Cphi) 。执行第batch size个task后，有 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7B%5Ctheta%7D%5E%7Bbz%7D%5CLeftarrow%5Cphi+-%5Ceta.%5Cpartial+l%28%5Cphi%29%2F%5Cpartial+%5Cphi) ，如下图所示。

![img](https://pic4.zhimg.com/v2-5679cdae47d9ae76ada9219b75d082cb_b.jpg)

- \4.  上述步骤3用了batch size个task对 ![[公式]](https://www.zhihu.com/equation?tex=M_%7Bmeta%7D) 进行了训练，然后我们使用上述batch个task中地**query set**去测试参数为![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7B%5Ctheta%7D%5E%7Bi%7D%2Ci%5Cin%5B1%2Cbatch+size%5D) 的 ![[公式]](https://www.zhihu.com/equation?tex=M_%7Bmeta%7D) 模型效果，获得总损失函数 ![[公式]](https://www.zhihu.com/equation?tex=L%28%5Cphi%29%3D%5Csum_%7Bi%3D1%7D%5E%7Bbs%7D%7Bl%5E%7Bi%7D%28%5Chat%7B%5Ctheta%7D%5E%7Bi%7D%29%7D) ，这个损失函数就是一个**batch task**中**每个task**的**query set**在各自参数为 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7B%5Ctheta%7D%5E%7Bi%7D%2Ci%5Cin%5B1%2Cbatch+size%5D) 的 ![[公式]](https://www.zhihu.com/equation?tex=M_%7Bmeta%7D) 中的损失 ![[公式]](https://www.zhihu.com/equation?tex=l%5E%7Bi%7D%28%5Chat%7B%5Ctheta%7D%5E%7Bi%7D%29) 之和。
- \5. 获得**总损失函数**后，我们就要对其进行**第二种的梯度下降**。即更新初始化参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Cphi) ，也就是  ![[公式]](https://www.zhihu.com/equation?tex=%5Cphi%5CLeftarrow%5Cphi+-%5Ceta.%5Cpartial+L%28%5Cphi%29%2F%5Cpartial+%5Cphi) 来更新初始化参数。这样不断地从步骤3开始训练，最终能够在数据集上获得该模型比较好的初始化参数。
- \6. 根据这个初始化的参数以及该模型，我们用数据集 ![[公式]](https://www.zhihu.com/equation?tex=%7B%5Cmathcal+D%7D_%7Bmeta-test%7D) 的support set对模型进行微调，这时候的**梯度下降步数**可以设置**更多**一点，不像训练时候（在第一次梯度下降过程中）只进行**一步**梯度下降。
- \7. 最后微调结束后，使用![[公式]](https://www.zhihu.com/equation?tex=%7B%5Cmathcal+D%7D_%7Bmeta-test%7D) 的query set进行模型的评估。

 至此，MAML的算法流程基本就结束了。

可以看出，每个batch个task中进行**batch次**第一种梯度下降以及**一次**第二种梯度下降。

**2.4 梯度近似计算**

上述算法流程结束后，我们可以获得三个等式，借用**李宏毅老师课堂ppt**，三个等式如下。

![img](https://pic2.zhimg.com/v2-b0882f25b44cfa921fcb97a826bf0351_b.jpg)

那么我们需要求解

![img](https://pic1.zhimg.com/v2-623bfab615fbbb20b3e3407b26fa0304_b.jpg)公式1

这三个等式的**第一个**就是最终我们需要求解的等式。那么这个等式中**最重要**的就是**总损失对 ![[公式]](https://www.zhihu.com/equation?tex=%5Cphi) 的梯度计算**，即

![img](https://pic2.zhimg.com/v2-4b17e0178babd6c990693150a063c3b9_b.jpg)公式2

然后我们对上述公式 ![[公式]](https://www.zhihu.com/equation?tex=%5Csum_%7Bn%3D1%7D%5E%7BN%7D%7B%7D) 里面的**梯度**进行**拆分计算**，即**![[公式]](https://www.zhihu.com/equation?tex=%7Bl%5E%7Bn%7D%28%5Chat%7B%5Ctheta%7D%5E%7Bn%7D%29%7D) 对 ![[公式]](https://www.zhihu.com/equation?tex=%5Cphi) 的梯度计算**，为

![img](https://pic4.zhimg.com/v2-7ab8a129f35e5911960698c4e96cf0eb_b.jpg)公式3

上面是把 ![[公式]](https://www.zhihu.com/equation?tex=%5Cphi) 拆成了一个个的标量，然后**分别计算后再整合**。

拆完 ![[公式]](https://www.zhihu.com/equation?tex=%5Cphi) 后，我们拆 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta) ，如下图所示。

![img](https://pic3.zhimg.com/v2-cd2643ef040ec0fc3607ad6d7451a3da_b.jpg)

那么根据链式法则，可得

![img](https://pic1.zhimg.com/v2-5fb1331edcceae7bb365ad2e6b9aee48_b.jpg)公式4

根据上述三个等式中的最后一个，也就是

![img](https://pic3.zhimg.com/v2-0e784b093cd3be2c03a85c8a0a10ab5a_b.jpg)公式5

我们将 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7B%5Ctheta_%7Bj%7D%7D) 和 ![[公式]](https://www.zhihu.com/equation?tex=%7B%5Cphi_%7Bj%7D%7D) 进行对应，获得以下公式

![img](https://pic3.zhimg.com/v2-1ce3ad8efa6c25679246656cdff70e72_b.jpg)公式6

接着分别公式4中的 i 和 j 进行分析。获得如下等式。

![img](https://pic2.zhimg.com/v2-f2c7dd1380e8a502d7dc803c3b350a01_b.jpg)公式7

那么作者考虑到这个二次微分不好计算，就假设这个**二次微分为0**来进行近似计算。如下

![img](https://pic3.zhimg.com/v2-bed57cf45a1713d14a56311c4f168886_b.jpg)公式8

那么将公式8的**近似结果**带入上面的公式4中。那么公式4就可以化简为

![img](https://pic1.zhimg.com/v2-77959b5d459d7d9bf4a412f4b3dab4c4_b.jpg)公式9

这种近似在论文中也体现了出来，如下

> we also include a comparison to dropping this backward pass and using **a first-order approximation。**

这个 **a first-order approximation**就是对**二次微分的忽略**。

那么根据公式9，公式3可以变化为

![img](https://pic2.zhimg.com/v2-22ef92b02e4b483b06ac900aee445931_b.jpg)公式10

那么将公式10带入公式2中，就可以**简化梯度计算**了。

至此有关MAML的解析就结束了。

## 3.总结

MAML是meta learning领域**非常重要**的一种算法。本文主要从**原理**的角度，结合了一些前人的**经验**，展开了对MAML的**解析**，我们发现本文最后的**梯度计算**中直接忽略**二次微分**，这样的这样的做法看似比较“**鲁莽**”，后面将结合下一个meta learning 算法，即**Reptile**，对MAML这个“**鲁莽**”行为带来的后果进行分析。
