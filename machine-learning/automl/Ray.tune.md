## [Ray.tune](https://github.com/ray-project/ray/tree/master/python/ray/tune)

Ray.tune是一个超参数优化库，主要适用于深度学习和强化学习模型。它结合了许多先进算法，如Hyperband算法（最低限度地训练模型来确定超参数的影响）、基于群体的训练算法（Population Based Training，在共享超参数下同时训练和优化一系列网络）、Hyperopt方法和中值停止规则（如果模型性能低于中等性能则停止训练）。

这些都运行在Ray分布式计算平台上，这让它具有很强的扩展性。