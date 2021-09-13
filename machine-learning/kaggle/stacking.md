

Stacking大杀器--StackNet

简介

![图片](https://mmbiz.qpic.cn/mmbiz_png/US10Gcd0tQEfcffueY0reDaT8agHibMbkl6VPJicIaSLBOMT46hKst5wjTztibed2dJsrke6B0nRpRPvJXnC2mlSg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

相信做数据竞赛的朋友对于StackNet肯定不陌生，早期的数据竞赛中，不夸张的说，Stacking技术可以帮助我们在排行榜上提升**50+**的名次，而其中最为著名的就是kaggle全球Top3的kaz-Anova开发的StackNet，StackNet是一个计算、可扩展和分析框架，它早期是采用Java软件实现的，使用StackNet拿下的金牌数也不下于100枚，

![图片](https://mmbiz.qpic.cn/mmbiz_png/ZQhHsg2x8fibsyBkJyXjlONLy4js5K5tWG8GACuar9FdYgdlbF1txRR2gx2qWPVib6hogp9ED664dsS0ZAUylCpQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

StackNet类似于前馈神经网络，与前馈神经网络不同，网络不是通过反向传播进行训练，而是一次迭代构建一层（使用叠加泛化），每个层都以最终目标作为其目标。

- StackNet（通常）会比它在每个第一层中包含的最好的单一模型要好，但是，它的最终性能仍然依赖于强大和diverse的单一模型的混合，以便从元建模方法中获得最佳效果。

StackNet

![图片](https://mmbiz.qpic.cn/mmbiz_png/US10Gcd0tQEfcffueY0reDaT8agHibMbkl6VPJicIaSLBOMT46hKst5wjTztibed2dJsrke6B0nRpRPvJXnC2mlSg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

01



工作原理

在一般的NN中，我们给定一些输入数据，神经网络通常应用感知器和变换函数，如relu、sigmoid、tanh或其他函数对其操作然后输入到下一层。

**StackNet模型则假设此函数可以采用任何有监督机器学习算法的形式来表示**：

- 我们只需要将其输出反馈到下一层即可。

此处的算法可以是分类器、回归器或任何产生输出的估计器。对于分类问题，要为response变量的任意数量的唯一类别创建输出预测分数，最后一层中的所有选定算法的输出维度必须等于这些唯一类别的数。如果存在多个此类分类器，则结果是所有这些输出预测的缩放平均值。

02



两种形式

StackNet有两种形式：

![图片](https://mmbiz.qpic.cn/mmbiz_png/ZQhHsg2x8fibsyBkJyXjlONLy4js5K5tW8eicsV2RUHibPtUhlHsyY36d80Dry2iaHWfa8XRfz3Vs1n6N0rnlQ38zw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

#### 1.传统模式

第一种为最为传统的Stack的模式，如上图的左侧所示。

#### 2.Restacking模式

第二种模式（也称为restacking）假设每一层都使用以前的神经元激活以及所有以前的层神经元（包括输入层）。

这种模式背后的直觉来自：

- 更高级别的算法已从输入数据中提取信息，但重新扫描输入空间可能会产生新的信息，这些信息在第一次扫描时并不明显。

这也是由下面讨论的正向训练方法所驱动的，并假设收敛需要在一个模型迭代中发生。

代码

![图片](https://mmbiz.qpic.cn/mmbiz_png/US10Gcd0tQEfcffueY0reDaT8agHibMbkl6VPJicIaSLBOMT46hKst5wjTztibed2dJsrke6B0nRpRPvJXnC2mlSg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

```
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from pystacknet.pystacknet import StackNetClassifier

models=[ 
        ######## First level ########
        [RandomForestClassifier (n_estimators=100, criterion="entropy", max_depth=5, max_features=0.5, random_state=1),
         ExtraTreesClassifier (n_estimators=100, criterion="entropy", max_depth=5, max_features=0.5, random_state=1),
         GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, max_features=0.5, random_state=1),
         LogisticRegression(random_state=1)
         ],
        ######## Second level ########
        [RandomForestClassifier (n_estimators=200, criterion="entropy", max_depth=5, max_features=0.5, random_state=1)]
        ]
model=StackNetClassifier(models, metric="auc", folds=4,
    restacking=False,use_retraining=True, use_proba=True, 
    random_state=12345,n_jobs=1, verbose=1)

model.fit(x,y)
preds=model.predict_proba(x_test) 
```

参考文献

![图片](https://mmbiz.qpic.cn/mmbiz_png/US10Gcd0tQEfcffueY0reDaT8agHibMbkl6VPJicIaSLBOMT46hKst5wjTztibed2dJsrke6B0nRpRPvJXnC2mlSg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

1. https://github.com/kaz-Anova/StackNet
2. https://medium.com/kaggle-blog
3. https://github.com/h2oai/pystacknet