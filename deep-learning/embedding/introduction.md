

YouTube 是利用 embedding 特征做推荐的开山之作，由于名声比较大，我们还是复用了他的网络结构，只不过在使用的特征上稍有差别。从一个 embedding 主义者的角度看，他的典型特点是把所有的特征（无论离散连续，单值多值）全部转化为 embedding，然后把各种 embedding 拼接在一起，构成一个一字长蛇阵的向量，然后送入 DNN，最后得到文章的向量。

Airbnb 主要贡献是在稀疏样本的构造上有所创新，个人感觉 Airbnb 这个操作部分弥补了 YouTube 在新闻推荐领域水土不服的问题。从一个 embedding 主义者的角度看，他的创新点主要有一下两点，一个是分群 embedding，另一个是用户和 item 混合训练。在移动腾讯网的动态规则聚类召回算法中就借鉴了 Airbnb 分群训练 embedding 的思想。

![img](https://image.jiqizhixin.com/uploads/editor/391aba6f-35b9-4595-8a0e-815d3de8a7b2/640.png)

在特征工程中，对于离散值，连续值，多值大致有以下几种 embedding 的方法。预先训练的 embedding 特征向量，训练样本大，参数学习更充分。end2end 是通过 embedding 层完成从高维稀疏向量到低维稠密特征向量的转换，优点是端到端，梯度统一，缺点是参数多，收敛速度慢，如果数据量少，参数很难充分训练。

![img](https://image.jiqizhixin.com/uploads/editor/664f3996-d26d-4994-b264-532907322287/640.png)

不同的深度学习模型中，除了对网络结构的各种优化外，在 embedding 的运算上也进行了各种优化的尝试，个人觉得对网络结构的各种优化本质上也是对 embedding 的运算的优化。

![img](https://image.jiqizhixin.com/uploads/editor/5502dc5a-28ab-4b66-8b92-227bba6b7cba/640.png)

embedding 作为一种技术，虽然很流行，但是他也存在一些缺陷，比如增量更新的语义不变性，很难同时包含多个特征，长尾数据难以训练等。

![img](https://image.jiqizhixin.com/uploads/editor/08171abc-8b5c-41b2-b013-d7c1bbb9dfc9/640.png)

针对 embedding 的空间分布影响模型的泛化误差的问题阿里和谷歌先后在 embedding 的表示和结构上进行了各种尝试，其中阿里提出了 residual embedding 的概念，希望把一个向量用中心向量和残差向量的形式去表示，以达到同一类别向量簇内高度聚集的目的。谷歌则希望对 embedding 的编码空间进行优化，简单来说就是为更高频更有效的特征分配更多的编码位置，反之则分配更少的编码位置。

![img](https://image.jiqizhixin.com/uploads/editor/07751d80-36f0-43a0-bdd4-71c69219cf07/640.png)

embedding 总体来说还是一种很有效的技术，在实践过程中大致经历了以下演进路线：

![img](https://image.jiqizhixin.com/uploads/editor/bb6eb01e-ca80-4f77-a400-d32bcb5ad9f5/640.png)