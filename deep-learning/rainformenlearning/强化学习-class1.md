## [强化学习（一） ----- 基本概念]

### 1. 强化学习简介

强化学习（reinforcementlearning, RL）是近年来机器学习和智能控制领域的主要方法之一。定义: Reinforcement learning is learning what to do ----how to map situations to actions --- so as to maximize a numerical reward signal.[1]

也就是说强化学习关注的是智能体如何在环境中采取一系列行为，从而获得最大的累积回报。

通过强化学习，一个智能体应该知道在什么状态下应该采取什么行为。RL是从环境状态到动作的映射的学习，我们把这个映射称为策略。

![img](https://d33wubrfki0l68.cloudfront.net/67e23ccf2e1ba2d5b44e89d67b8be832af6b3a87/d75a6/images/cn/2020-05-09-introduction-of-reinforcement-learning/machine-learning-types.png)

强化学习同机器学习领域中的**有监督学习**和**无监督学习**不同，有监督学习是从外部监督者提供的带标注训练集中进行学习（任务驱动型），无监督学习是一个典型的寻找未标注数据中隐含结构的过程（数据驱动型）。强化学习是与两者并列的第三种机器学习范式，强化学习和监督学习的区别主要有以下两点：

1. 强化学习是试错学习(Trail-and-error)，由于没有直接的指导信息，智能体要以不断与环境进行交互，通过试错的方式来获得最佳策略。

2. 延迟回报，强化学习的指导信息很少，而且往往是在事后（最后一个状态）才给出的，这就导致了一个问题，就是获得正回报或者负回报以后，如何将回报分配给前面的状态。


在强化学习中，有两个可以进行交互的对象：**智能体（Agnet）**和**环境（Environment）**：

- 智能体：可以感知环境的**状态（State）**，并根据反馈的**奖励（Reward）**学习选择一个合适的**动作（Action）**，来最大化长期总收益。
- 环境：环境会接收智能体执行的一系列动作，对这一系列动作进行评价并转换为一种可量化的信号反馈给智能体。

![img](https://d33wubrfki0l68.cloudfront.net/3aeab88b7c8d9dad0807cb5e6b3a952e357ae1cd/89bc7/images/cn/2020-05-09-introduction-of-reinforcement-learning/reinforcement-learning.png)

图片来源：https://en.wikipedia.org/wiki/Reinforcement_learning

除了智能体和环境之外，强化学习系统有四个核心要素：**策略（Policy）**、**回报函数（收益信号，Reward Function）**、**价值函数（Value Function）**和**环境模型（Environment Model）**，其中环境模型是可选的。

- 策略：定义了智能体在特定时间的行为方式。策略是环境状态到动作的映射。
- 回报函数：定义了强化学习问题中的目标。在每一步中，环境向智能体发送一个称为收益的标量数值。
- 价值函数：表示了从长远的角度看什么是好的。一个状态的价值是一个智能体从这个状态开始，对将来累积的总收益的期望。
- 环境模型：是一种对环境的反应模式的模拟，它允许对外部环境的行为进行推断。

强化学习是一种对目标导向的学习与决策问题进行理解和自动化处理的计算方法。它强调智能体通过与环境的直接互动来学习，而不需要可效仿的监督信号或对周围环境的完全建模，因而与其他的计算方法相比具有不同的范式。

强化学习使用马尔可夫决策过程的形式化框架，使用**状态**，**动作**和**收益**定义学习型智能体与环境的互动过程。这个框架力图简单地表示人工智能问题的若干重要特征，这些特征包含了对**因果关系**的认知，对**不确定性**的认知，以及对**显式目标存在性**的认知。

价值与价值函数是强化学习方法的重要特征，价值函数对于策略空间的有效搜索来说十分重要。相比于进化方法以对完整策略的反复评估为引导对策略空间进行直接搜索，使用价值函数是强化学习方法与进化方法的不同之处。

### 2.  示例和应用

#### 游戏

例1. [flappy bird ](http://store.liebao.cn/game/LBGameCenter/?game=flappybird)是现在很流行的一款小游戏，不了解的同学可以点链接进去玩一会儿。现在我们让小鸟自行进行游戏，但是我们却没有小鸟的动力学模型，也不打算了解它的动力学。要怎么做呢？ 这时就可以给它设计一个强化学习算法，然后让小鸟不断的进行游戏，如果小鸟撞到柱子了，那就获得-1的回报，否则获得0回报。通过这样的若干次训练，我们最终可以得到一只飞行技能高超的小鸟，它知道在什么情况下采取什么动作来躲避柱子。

![img](https://d33wubrfki0l68.cloudfront.net/354f4ff4a54d8481949f35f325a9547608a58f70/65d9f/images/cn/2020-05-09-introduction-of-reinforcement-learning/flappy-bird-rl.png)

图片来源：https://easyai.tech/ai-definition/reinforcement-learning

目前，强化学习在包括**游戏**，**广告和推荐**，**对话系统**，**机器人**等多个领域均展开了广泛的应用。

例2. 假设我们要构建一个下国际象棋的机器，这种情况不能使用监督学习，首先，我们本身不是优秀的棋手，而请象棋老师来遍历每个状态下的最佳棋步则代价过于昂贵。其次，每个棋步好坏判断不是孤立的，要依赖于对手的选择和局势的变化。是一系列的棋步组成的策略决定了是否能赢得比赛。下棋过程的唯一的反馈是在最后赢得或是输掉棋局时才产生的。这种情况我们可以采用强化学习算法，通过不断的探索和试错学习，强化学习可以获得某种下棋的策略，并在每个状态下都选择最有可能获胜的棋步。目前这种算法已经在棋类游戏中得到了广泛应用。


![img](https://d33wubrfki0l68.cloudfront.net/fa75eeec83fa260b8895aab864223df15d1d592d/c8bdb/images/cn/2020-05-09-introduction-of-reinforcement-learning/alphago.png)

**AlphaGo** 是于 2014 年开始由英国伦敦 Google DeepMind 开发的人工智能围棋软件。AlphaGo 使用蒙特卡洛树搜索（Monte Carlo tree search），借助估值网络（value network）与走棋网络（policy network）这两种深度神经网络，通过估值网络来评估大量选点，并通过走棋网络选择落点。

**AlphaStar** 是由 DeepMind 开发的玩 [星际争霸 II](https://zh.wikipedia.org/wiki/星海爭霸II：自由之翼) 游戏的人工智能程序。AlphaStar 是由一个深度神经网路生成的，它接收来自原始游戏界面的输入数据，并输出一系列指令，构成游戏中的一个动作。

AlphaStar 还使用了一种新的多智能体学习算法。该神经网路最初是通过在 Blizzard 发布的匿名人类游戏中进行监督学习来训练的。这使得 AlphaStar 能够通过模仿学习星际争霸上玩家所使用的基本微观和宏观策略。这个初级智能体在 95% 的游戏中击败了内置的「精英」AI 关卡（相当于人类玩家的黄金级别）。

![img](https://d33wubrfki0l68.cloudfront.net/d6faef6a01b08a68d901bcdb2543e15dff0c7d91/e96c1/images/cn/2020-05-09-introduction-of-reinforcement-learning/alphastar.png)

**OpenAI Five** 是一个由 OpenAI 开发的用于多人视频游戏 [Dota 2](https://zh.wikipedia.org/zh-hans/Dota_2) 的人工智能程序。OpenAI Five 通过与自己进行超过 10,000 年时长的游戏进行优化学习，最终获得了专家级别的表现。

![img](https://d33wubrfki0l68.cloudfront.net/d7872d92aa8654c084ef7399ff0a1604d3b31113/05277/images/cn/2020-05-09-introduction-of-reinforcement-learning/openai-five.png)

**Pluribus** 是由 Facebook 开发的第一个在六人无限注德州扑克中击败人类专家的 AI 智能程序，其首次在复杂游戏中击败两个人或两个团队。

![img](https://d33wubrfki0l68.cloudfront.net/61bce36b57330d3fb2256ed991e20a91b6abf35d/9e185/images/cn/2020-05-09-introduction-of-reinforcement-learning/facebook-pluribus.jpg)


#### 广告和推荐

![img](https://d33wubrfki0l68.cloudfront.net/3584a8c4bfbe8927855a5402b3223a7b913708ad/f7797/images/cn/2020-05-09-introduction-of-reinforcement-learning/recommendation.png)

图片来源：A Reinforcement Learning Framework for Explainable Recommendation

#### 对话系统

![img](https://d33wubrfki0l68.cloudfront.net/23f876b6f2431eab47438538c625ee9a508c24ad/6277a/images/cn/2020-05-09-introduction-of-reinforcement-learning/dialogue-system.png)

图片来源：End-to-End Task-Completion Neural Dialogue Systems

#### 机器人

![img](https://d33wubrfki0l68.cloudfront.net/d25567c6877e6870f47df9b8aca89a46432eff2c/70a36/images/cn/2020-05-09-introduction-of-reinforcement-learning/robot.png)

图片来源：Learning Synergies between Pushing and Grasping with Self-supervised Deep Reinforcement Learning

### 3.开放资源

#### 开源实验平台

- [openai/gym](https://github.com/openai/gym)
- [MuJoCo](http://mujoco.org/)
- [openai/mujoco-py](https://github.com/openai/mujoco-py)
- [deepmind/lab](https://github.com/deepmind/lab)

#### 开源框架

- [deepmind/trfl/](https://github.com/deepmind/trfl/)
- [deepmind/open_spiel](https://github.com/deepmind/open_spiel)
- [google/dopamine](https://github.com/google/dopamine)
- [tensorflow/agents](https://github.com/tensorflow/agents)
- [keras-rl/keras-rl](https://github.com/keras-rl/keras-rl)
- [tensorforce/tensorforce](https://github.com/tensorforce/tensorforce)
- [facebookresearch/ReAgent](https://github.com/facebookresearch/ReAgent)
- [thu-ml/tianshou](https://github.com/thu-ml/tianshou)
- [astooke/rlpyt](https://github.com/astooke/rlpyt)
- [NervanaSystems/coach](https://github.com/NervanaSystems/coach)
- [PaddlePaddle/PARL](https://github.com/PaddlePaddle/PARL)

#### 开源模型

- [dennybritz/reinforcement-learning](https://github.com/dennybritz/reinforcement-learning)
- [openai/baselines](https://github.com/openai/baselines)

#### 其他资源

- [ShangtongZhang/reinforcement-learning-an-introduction](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
- [aikorea/awesome-rl](https://github.com/aikorea/awesome-rl)
- [openai/spinningup](https://github.com/openai/spinningup)
- [udacity/deep-reinforcement-learning](https://github.com/udacity/deep-reinforcement-learning)
