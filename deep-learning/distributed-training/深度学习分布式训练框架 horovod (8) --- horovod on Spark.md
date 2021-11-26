# 深度学习分布式训练框架 horovod (8) --- TensorFlow On Spark





The `horovod.spark` package provides a convenient wrapper around Horovod that makes running distributed training jobs in Spark clusters easy.

In situations where training data originates from Spark, this enables a tight model design loop in which data processing, model training, and model evaluation are all done in Spark.

We provide two APIs for running Horovod on Spark: a high level **Estimator** API and a lower level **Run** API. Both use the same underlying mechanism to launch Horovod on Spark executors, but the Estimator API abstracts the data processing (from Spark DataFrames to deep learning datasets), model training loop, model checkpointing, metrics collection, and distributed training.

We recommend using Horovod Spark Estimators if you:

- Are using Keras (`tf.keras` or `keras`) or PyTorch for training.
- Want to train directly on a Spark DataFrame from `pyspark`.
- Are using a standard gradient descent optimization process as your training loop.

If for whatever reason the Estimator API does not meet your needs, the Run API offers more fine-grained control.



