## 《强化学习》第二讲 马尔科夫决策过程



在强化学习中，马尔科夫决策过程（Markov decision process, MDP）是对完全可观测的环境进行描述的，也就是说观测到的状态内容完整地决定了决策的需要的特征。几乎所有的强化学习问题都可以转化为MDP。本讲是理解强化学习问题的理论基础。

### **马尔科夫过程 Markov Process**

- **马尔科夫性 Markov Property**

某一状态信息包含了所有相关的历史，只要当前状态可知，所有的历史信息都不再需要，当前状态就可以决定未来，则认为该状态具有**马尔科夫性**。

可以用下面的状态转移概率公式来描述马尔科夫性：

![img](https://pic4.zhimg.com/80/v2-611b314b22f8e46f1c08d978717a9437_1440w.png)

下面状态转移矩阵定义了所有状态的转移概率：

![img](https://pic2.zhimg.com/80/v2-3ae1bf04c30fe0ed5aea3b3199265f55_1440w.png)

式中n为状态数量，矩阵中每一行元素之和为1.

- **马尔科夫过程 Markov Property**

**马尔科夫过程** 又叫马尔科夫链(Markov Chain)，它是一个无记忆的随机过程，可以用一个元组<S,P>表示，其中S是有限数量的状态集，P是状态转移概率矩阵。

- **示例——学生马尔科夫链**

本讲多次使用了学生马尔科夫链这个例子来讲解相关概念和计算。



![img](https://pic1.zhimg.com/80/v2-e1e894383536e4ff019f63e5507c2a18_1440w.png)

图中，圆圈表示学生所处的状态，方格Sleep是一个终止状态（吸收态），或者可以描述成自循环的状态，也就是Sleep状态的下一个状态100%的几率还是自己。箭头表示状态之间的转移，箭头上的数字表示当前转移的概率。

举例说明：

当学生处在第一节课（Class1）时，他/她有50%的几率会参加第2节课（Class2）；同时在也有50%的几率不在认真听课，进入到浏览facebook这个状态中。

在浏览facebook这个状态时，他/她有90%的几率在下一时刻继续浏览，也有10%的几率返回到课堂内容上来。

当学生进入到第二节课（Class2）时，会有80%的几率继续参加第三节课（Class3），也有20%的几率觉得课程较难而退出（Sleep）。

当学生处于第三节课这个状态时，他有60%的几率通过考试，继而100%的退出该课程，也有40%的可能性需要到去图书馆之类寻找参考文献，此后根据其对课堂内容的理解程度，又分别有20%、40%、40%的几率返回值第一、二、三节课重新继续学习。

一个可能的学生马尔科夫链从状态Class1开始，最终结束于Sleep，其间的过程根据状态转化图可以有很多种可能性，这些都称为**Sample Episodes**。以下四个Episodes都是可能的：

C1 - C2 - C3 - Pass - Sleep

C1 - FB - FB - C1 - C2 - Sleep

C1 - C2 - C3 - Pub - C2 - C3 - Pass - Sleep

C1 - FB - FB - C1 - C2 - C3 - Pub - C1 - FB - FB - FB - C1 - C2 - C3 - Pub - C2 - Sleep

该学生马尔科夫过程的状态转移矩阵如下图：

![img](https://pic1.zhimg.com/80/v2-23b6d59cfe253c4a678a1d9e8df43110_1440w.png)

### **马尔科夫奖励过程 Markov Reward Process**

马尔科夫奖励过程在马尔科夫过程的基础上增加了奖励 R 和衰减系数 *γ：<S,P,R,γ>*。R 是一个奖励函数。S 状态下的奖励是某一时刻(t)处在状态s下在下一个时刻(t+1)能获得的奖励期望：

![[公式]](https://www.zhihu.com/equation?tex=R_%7Bs%7D+%3D+E%5BR_%7Bt%2B1%7D+%7C+S_%7Bt%7D+%3D+s+%5D)

很多听众纠结为什么奖励是t+1时刻的。照此理解起来相当于离开这个状态才能获得奖励而不是进入这个状态即获得奖励。David指出这仅是一个约定，为了在描述 RL 问题中涉及到的观测O、行为A、和奖励R时比较方便。他同时指出如果把奖励改为 ![[公式]](https://www.zhihu.com/equation?tex=R_t) 而不是 ![[公式]](https://www.zhihu.com/equation?tex=R_%7Bt%2B1%7D) ，只要规定好，本质上意义是相同的，在表述上可以把奖励描述为“当进入某个状态会获得相应的奖励”。

**衰减系数 Discount Factor:** γ∈ [0, 1]，它的引入有很多理由，其中优达学城的“机器学习-强化学习”课程对其进行了非常有趣的数学解释。David也列举了不少原因来解释为什么引入衰减系数，其中有数学表达的方便，避免陷入无限循环，远期利益具有一定的不确定性，符合人类对于眼前利益的追求，符合金融学上获得的利益能够产生新的利益因而更有价值等等。

下图是一个“马尔科夫奖励过程”图示的例子，在“马尔科夫过程”基础上增加了针对每一个状态的奖励，由于不涉及衰减系数相关的计算，这张图并没有特殊交代衰减系数值的大小。



![img](https://pic2.zhimg.com/80/v2-4a4c2ccdb5911ec0125b75291a87dce5_1440w.png)

### **收获 Return**

定义：收获 ![[公式]](https://www.zhihu.com/equation?tex=G_%7Bt%7D) 为在一个马尔科夫奖励链上从t时刻开始往后所有的奖励的有衰减的总和。也有翻译成“收益”或"回报"。公式如下：

![img](https://pic1.zhimg.com/80/v2-e5e691ff4b754db8f893dfd367107600_1440w.png)

其中衰减系数体现了未来的奖励在当前时刻的价值比例，在k+1时刻获得的奖励R在t时刻的体现出的价值是 ![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma%5Ek+R) ，γ接近0，则表明趋向于“近视”性评估；γ接近1则表明偏重考虑远期的利益。

### **价值函数 Value Function**

价值函数给出了某一状态或某一行为的长期价值。

定义：一个马尔科夫奖励过程中某一状态的**价值函数**为**从该状态开始**的马尔可夫链收获的期望：

![[公式]](https://www.zhihu.com/equation?tex=v%28s%29+%3D+E+%5B+G_%7Bt%7D+%7C+S_%7Bt%7D+%3D+s+%5D)

注：价值可以仅描述状态，也可以描述某一状态下的某个行为，在一些特殊情况下还可以仅描述某个行为。在整个视频公开课中，除了特别指出，约定用**状态价值函数**或**价值函数**来描述针对状态的价值；用**行为价值函数**来描述某一状态下执行某一行为的价值，严格意义上说行为价值函数是“**状态行为对”价值函数**的简写。

### **举例说明收获和价值的计算**

为方便计算，把“学生马尔科夫奖励过程”示例图表示成下表的形式。表中第二行对应各状态的即时奖励值，蓝色区域数字为状态转移概率，表示为从所在行状态转移到所在列状态的概率：

![img](https://pic3.zhimg.com/80/v2-52c5d21082994b4cc1d4aac0fe4f58ba_1440w.png)

考虑如下4个马尔科夫链。现计算当γ= 1/2时，在t=1时刻（![[公式]](https://www.zhihu.com/equation?tex=S_%7B1%7D+%3D+C_%7B1%7D)）时状态 ![[公式]](https://www.zhihu.com/equation?tex=S_%7B1%7D) 的收获分别为：

![img](https://pic2.zhimg.com/80/v2-91921a745909435f7b984d1dae5ef271_1440w.png)

从上表也可以理解到，收获是针对一个马尔科夫链中的**某一个状态**来说的。

当γ= 0时，上表描述的MRP中，各状态的即时奖励就与该状态的价值相同。当γ≠ 0时，各状态的价值需要通过计算得到，这里先给出γ分别为0, 0.9,和1三种情况下各状态的价值，如下图所示。

各状态圈内的数字表示该状态的价值，圈外的R=-2等表示的是该状态的即时奖励。



![img](https://pic1.zhimg.com/80/v2-052d21809b80c136060f91e78b5e2278_1440w.png)



![img](https://pic3.zhimg.com/80/v2-ec423d06e1166af7bc0cbfac7f49e6a2_1440w.png)



![img](https://pic4.zhimg.com/80/v2-9c5b6a2e2c4082b83056078a0b1ced6b_1440w.png)



各状态价值的确定是很重要的，RL的许多问题可以归结为求状态的价值问题。因此如何求解各状态的价值，也就是寻找一个价值函数（从状态到价值的映射）就变得很重要了。

### **价值函数的推导**

- **Bellman方程 - MRP**

先尝试用价值的定义公式来推导看看能得到什么：

![img](https://pic3.zhimg.com/80/v2-fda247960872e2cb7653bcb89729626a_1440w.png)

这个推导过程相对简单，仅在导出最后一行时，将 ![[公式]](https://www.zhihu.com/equation?tex=G_%7Bt%2B1%7D) 变成了 ![[公式]](https://www.zhihu.com/equation?tex=v%28S_%7Bt%2B1%7D%29) 。其理由是收获的期望等于收获的期望的期望。下式是针对MRP的Bellman方程：

![img](https://pic3.zhimg.com/80/v2-d9bf4d39fba6d6afcb9e0e8cae734242_1440w.png)

通过方程可以看出 ![[公式]](https://www.zhihu.com/equation?tex=v%28s%29) 由两部分组成，一是该状态的即时奖励期望，即时奖励期望等于即时奖励，因为根据即时奖励的定义，它与下一个状态无关；另一个是下一时刻状态的价值期望，可以根据下一时刻状态的概率分布得到其期望。如果用s’表示s状态下一时刻任一可能的状态，那么Bellman方程可以写成：

![img](https://pic1.zhimg.com/80/v2-1164fecb7bf77d8210343e53c4fa7ac8_1440w.png)

- **方程的解释**

下图已经给出了γ=1时各状态的价值（该图没有文字说明γ=1，根据视频讲解和前面图示以及状态方程的要求，γ必须要确定才能计算），状态 ![[公式]](https://www.zhihu.com/equation?tex=C_%7B3%7D) 的价值可以通过状态Pub和Pass的价值以及他们之间的状态转移概率来计算：

![[公式]](https://www.zhihu.com/equation?tex=4.3+%3D+-2+%2B+1.0+%2A+%28+0.6+%2A+10+%2B+0.4+%2A+0.8+%29)



![img](https://pic4.zhimg.com/80/v2-a8997be4d72fcb8faaee4db82db495b3_1440w.png)



- **Bellman方程的矩阵形式和求解**

![img](https://pic4.zhimg.com/80/v2-444fc8bffca56f64f6599818800c54df_1440w.png)

结合矩阵的具体表达形式还是比较好理解的：

![img](https://pic3.zhimg.com/80/v2-071d680f97a7cfc7199f03a700b1f9a2_1440w.png)

Bellman方程是一个线性方程组，因此理论上解可以直接求解：

![img](https://pic4.zhimg.com/80/v2-ee67be43e30fababfab0fb4820db303f_1440w.png)

实际上，计算复杂度是 ![[公式]](https://www.zhihu.com/equation?tex=O%28n%5E%7B3%7D%29) ， ![[公式]](https://www.zhihu.com/equation?tex=n) 是状态数量。因此直接求解仅适用于小规模的MRPs。大规模MRP的求解通常使用迭代法。常用的迭代方法有：动态规划Dynamic Programming、蒙特卡洛评估Monte-Carlo evaluation、时序差分学习Temporal-Difference，后文会逐步讲解这些方法。

### **马尔科夫决策过程 Markov Decision Process**

相较于马尔科夫奖励过程，马尔科夫决定过程多了一个行为集合A，它是这样的一个元组: <S, A, P, R, γ>。看起来很类似马尔科夫奖励过程，但这里的P和R都与具体的**行为**a对应，而不像马尔科夫奖励过程那样仅对应于某个**状态**，A表示的是有限的行为的集合。具体的数学表达式如下：

![img](https://pic2.zhimg.com/80/v2-8d5223ece5e1c82928b164a7a7e589e9_1440w.png)



![img](https://pic4.zhimg.com/80/v2-0e748583bfa697a166935c91226fba6f_1440w.png)



- **示例——学生MDP**

下图给出了一个可能的MDP的状态转化图。图中红色的文字表示的是采取的行为，而不是先前的状态名。对比之前的学生MRP示例可以发现，即时奖励与行为对应了，同一个状态下采取不同的行为得到的即时奖励是不一样的。由于引入了Action，容易与状态名混淆，因此此图没有给出各状态的名称；此图还把Pass和Sleep状态合并成一个终止状态；另外当选择”去查阅文献”这个动作时，**主动**进入了一个临时状态（图中用黑色小实点表示），随后**被动的**被环境按照其动力学分配到另外三个状态，也就是说此时Agent没有选择权决定去哪一个状态。

![img](https://pic3.zhimg.com/80/v2-9f1230d02a301046332df0ef95e969de_1440w.png)



- **策略Policy**

策略 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi)是概率的集合或分布，其元素 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi%28a%7Cs%29) 为对过程中的某一状态s采取可能的行为a的概率。用 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi%28a%7Cs%29) 表示。

![img](https://pic4.zhimg.com/80/v2-4674dce375b920fa5750638b97f27ef7_1440w.png)

一个策略完整定义了个体的行为方式，也就是说定义了个体在各个状态下的各种可能的行为方式以及其概率的大小。Policy仅和当前的状态有关，与历史信息无关；同时某一确定的Policy是静态的，与时间无关；但是个体可以随着时间更新策略。

当给定一个MDP: ![[公式]](https://www.zhihu.com/equation?tex=M+%3D+%3CS%2C+A%2C+P%2C+R%2C+%5Cgamma%3E) 和一个策略π，那么状态序列 ![[公式]](https://www.zhihu.com/equation?tex=S_%7B1%7D%2CS_%7B2%7D%2C...) 是一个马尔科夫过程 ![[公式]](https://www.zhihu.com/equation?tex=%3CS%2C+P%5E%7B%5Cpi%7D%3E) ；同样，状态和奖励序列 ![[公式]](https://www.zhihu.com/equation?tex=S_%7B1%7D%2C+R_%7B2%7D%2C+S_%7B2%7D%2C+R_%7B3%7D%2C+S_%7B3%7D%2C+...) 是一个马尔科夫奖励过程 ![[公式]](https://www.zhihu.com/equation?tex=%3CS%2C+P%5E%7B%5Cpi%7D%2C+R%5E%7B%5Cpi%7D%2C+%5Cgamma%3E) ，并且在这个奖励过程中满足下面两个方程：

![img](https://pic1.zhimg.com/80/v2-b2d236b8f496f2159ecb8eb28c2d02d8_1440w.png)

用文字描述是这样的，在执行策略 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpi) 时，状态从s转移至 s' 的概率等于一系列概率的和，这一系列概率指的是在执行当前策略时，执行某一个行为的概率与该行为能使状态从s转移至s’的概率的乘积。

奖励函数表示如下：

![img](https://pic2.zhimg.com/80/v2-23a42adda5a08173205c1487827c824d_1440w.png)

用文字表述是这样的：当前状态s下执行某一指定策略得到的即时奖励是该策略下所有可能行为得到的奖励与该行为发生的概率的乘积的和。

策略在MDP中的作用相当于agent可以在某一个状态时做出选择，进而有形成各种马尔科夫过程的可能，而且基于策略产生的每一个马尔科夫过程是一个马尔科夫奖励过程，各过程之间的差别是不同的选择产生了不同的后续状态以及对应的不同的奖励。

- **基于策略π的价值函数**

定义![[公式]](https://www.zhihu.com/equation?tex=v_%5Cpi%28s%29) 是在MDP下的基于策略π的**状态价值函数**，表示从状态s开始，**遵循当前策略**时所获得的收获的期望；或者说在执行当前策略π时，衡量个体处在状态s时的价值大小。数学表示如下：

![img](https://pic1.zhimg.com/80/v2-160841859a045e3cfd68fd16df3d35f4_1440w.png)

注意策略是静态的、关于整体的概念，不随状态改变而改变；变化的是在某一个状态时，依据策略可能产生的具体行为，因为具体的行为是有一定的概率的，策略就是用来描述各个不同状态下执行各个不同行为的概率。



定义 ![[公式]](https://www.zhihu.com/equation?tex=q_%7B%5Cpi%7D%28s%2Ca%29)为**行为价值函数，表示**在执行策略π时，对当前状态s执行某一具体行为a所能的到的收获的期望；或者说在遵循当前策略π时，衡量对当前状态执行行为a的价值大小。行为价值函数一般都是与某一特定的状态相对应的，更精细的描述是**状态行为对**价值函数。行为价值函数的公式描述如下：

![img](https://pic2.zhimg.com/80/v2-5bf86a19e50ede3a82443a82c95087f9_1440w.png)

下图用例子解释了行为价值函数

![img](https://pic2.zhimg.com/80/v2-c3feec3c11cbbe6040c04ecb8931da59_1440w.png)



- **Bellman期望方程 Bellman Expectation Equation**

MDP下的状态价值函数和行为价值函数与MRP下的价值函数类似，可以改用下一时刻状态价值函数或行为价值函数来表达，具体方程如下：

![img](https://pic2.zhimg.com/80/v2-439072d457b39f007e41e0ec24a54f71_1440w.png)



![img](https://pic2.zhimg.com/80/v2-ffcf338639da9a40fea7096d371c25c9_1440w.png)



- ![[公式]](https://www.zhihu.com/equation?tex=v_%7B%5Cpi%7D%28s%29)**和**![[公式]](https://www.zhihu.com/equation?tex=q_%7B%5Cpi%7D%28s%2Ca%29)**的关系**

![img](https://pic1.zhimg.com/80/v2-afda4ee31b7ea7238f7c2bc15709e5a8_1440w.png)

上图中，空心较大圆圈表示状态，黑色实心小圆表示的是动作本身，连接状态和动作的线条仅仅把该状态以及该状态下可以采取的行为关联起来。可以看出，在遵循策略π时，状态s的价值体现为在该状态下遵循某一策略而采取所有可能行为的价值按行为发生概率的乘积求和。

![img](https://pic3.zhimg.com/80/v2-d81dabbf2059c2f7e5a499a72cd5f1a6_1440w.png)

类似的，一个行为价值函数也可以表示成状态价值函数的形式：



![img](https://pic2.zhimg.com/80/v2-3d02c6e2372658c839687493f8ddbfd1_1440w.png)

它表明，一个某一个状态下采取一个行为的价值，可以分为两部分：其一是离开这个状态的价值，其二是所有进入新的状态的价值于其转移概率乘积的和。

![img](https://pic4.zhimg.com/80/v2-5f4535af4300fa2228348c233724227b_1440w.png)

如果组合起来，可以得到下面的结果：

![img](https://pic1.zhimg.com/80/v2-aa100949ef199e5c917e526603999d1c_1440w.png)



![img](https://pic4.zhimg.com/80/v2-97c90182e2bfe4a2c83f58392f8d8f5f_1440w.png)

也可以得到下面的结果：

![img](https://pic2.zhimg.com/80/v2-0b1450e04071326f9cba4f3cce543121_1440w.png)



![img](https://pic4.zhimg.com/80/v2-be2a2778a9dfcfe2d9dce9bc64f45903_1440w.png)



- **学生MDP示例**

下图解释了红色空心圆圈状态的状态价值是如何计算的，遵循的策略随机策略，即所有可能的行为有相同的几率被选择执行。



![img](https://pic1.zhimg.com/80/v2-1ef95dc0d203c5f2e85986faf31464b0_1440w.png)



- **Bellman期望方程矩阵形式**

![img](https://pic1.zhimg.com/80/v2-110ff09a5debb86ba7864b4702d9c2e8_1440w.png)



![img](https://pic3.zhimg.com/80/v2-eadcbf77ddda92fc41c9e0dedceea94a_1440w.png)



- **最优价值函数**

最优状态价值函数 ![[公式]](https://www.zhihu.com/equation?tex=v_%2A%28s%29) 指的是在从所有策略产生的状态价值函数中，选取使状态s价值最大的函数：

![[公式]](https://www.zhihu.com/equation?tex=v_%7B%2A%7D+%3D+%5Cmax_%7B%5Cpi%7D+v_%7B%5Cpi%7D%28s%29)

类似的，最优行为价值函数 ![[公式]](https://www.zhihu.com/equation?tex=q_%7B%2A%7D%28s%2Ca%29) 指的是从所有策略产生的行为价值函数中，选取是状态行为对 ![[公式]](https://www.zhihu.com/equation?tex=%3Cs%2Ca%3E) 价值最大的函数：

![[公式]](https://www.zhihu.com/equation?tex=q_%7B%2A%7D%28s%2Ca%29+%3D+%5Cmax_%7B%5Cpi%7D+q_%7B%5Cpi%7D%28s%2Ca%29)

最优价值函数明确了MDP的最优可能表现，当我们知道了最优价值函数，也就知道了每个状态的最优价值，这时便认为这个MDP获得了解决。

学生MDP问题的最优状态价值



![img](https://pic3.zhimg.com/80/v2-87b7ba71377c2fde0dcb73c3612f0cee_1440w.png)



学生MDP问题的最优行为价值



![img](https://pic1.zhimg.com/80/v2-970c76c507a627a1e22a9cd3ee587608_1440w.png)



注：youtube留言认为Pub行为对应的价值是+9.4而不是+8.4



- **最优策略**

当对于任何状态 s，遵循策略π的价值不小于遵循策略 π' 下的价值，则策略π优于策略 π’：

![img](https://pic2.zhimg.com/80/v2-48baa212c40165d4545f39616a67d855_1440w.png)

**定理** 对于任何MDP，下面几点成立：1.存在一个最优策略，比任何其他策略更好或至少相等；2.所有的最优策略有相同的最优价值函数；3.所有的最优策略具有相同的行为价值函数。



- **寻找最优策略**

可以通过最大化最优行为价值函数来找到最优策略：

![img](https://pic1.zhimg.com/80/v2-b4f9455155559b42e8d4ee761935d428_1440w.png)

对于任何MDP问题，总存在一个确定性的最优策略；同时如果我们知道最优行为价值函数，则表明我们找到了最优策略。



- **学生MDP最优策略示例**

红色箭头表示的行为表示最优策略



![img](https://pic4.zhimg.com/80/v2-3f94c760b7b526670c0adcea9d4ae0f7_1440w.png)





- **Bellman最优方程 Bellman Optimality Equation**

针对 ![[公式]](https://www.zhihu.com/equation?tex=v_%7B%2A%7D) ，一个状态的最优价值等于从该状态出发采取的所有行为产生的行为价值中最大的那个行为价值：

![img](https://pic4.zhimg.com/80/v2-949feaf56d3564261819fb5d5768560f_1440w.png)



![img](https://pic3.zhimg.com/80/v2-a18e92f350dd20bdfdefd2a8e159d71a_1440w.png)



针对 ![[公式]](https://www.zhihu.com/equation?tex=q_%7B%2A%7D) ，在某个状态s下，采取某个行为的最优价值由2部分组成，一部分是离开状态 s 的即刻奖励，另一部分则是所有能到达的状态 s’ 的最优状态价值按出现概率求和：

![img](https://pic4.zhimg.com/80/v2-6bd526d85e0eac10c0ca1356fa76e2cf_1440w.png)



![img](https://pic4.zhimg.com/80/v2-87be3a34bcf2d7ddb303e92ef29a5ebb_1440w.png)



组合起来，针对 ![[公式]](https://www.zhihu.com/equation?tex=v_%7B%2A%7D) ，有：

![img](https://pic4.zhimg.com/80/v2-695fd1048f097b2331aba540c91358d3_1440w.png)



![img](https://pic1.zhimg.com/80/v2-42a29ffcbfd1fd5989c6bc3d9a7e9220_1440w.png)



针对![[公式]](https://www.zhihu.com/equation?tex=q_%7B%2A%7D) ，有：

![img](https://pic1.zhimg.com/80/v2-302240e13af99e181e4b23bf1e43f720_1440w.png)



![img](https://pic1.zhimg.com/80/v2-ebb8ee4e71dd9b6149481a334f738ba8_1440w.png)



- **Bellman最优方程学生MDP示例**



![img](https://pic4.zhimg.com/80/v2-001dc0c6e1c349369d5067969fc3385b_1440w.png)



- **求解Bellman最优方程**

Bellman最优方程是非线性的，没有固定的解决方案，通过一些迭代方法来解决：价值迭代、策略迭代、Q学习、Sarsa等。后续会逐步讲解展开。





## **MDP延伸——Extensions to MDPs**

简要提及：无限状态或连续MDP；部分可观测MDP；非衰减、平均奖励MDP
