# Data analysis and math

## [Breeze](http://www.scalanlp.org/)

Breeze is known as the primary scientific computing library for Scala. It scoops up ideas from MATLAB’s data structures and the NumPy classes for Python. Breeze provides fast and efficient manipulations with data arrays, and enables the implementation of many other operations, including the following:

- **Matrix and vector operations** for creating, transposing, filling with numbers, conducting element-wise operations, inversion, calculating determinants, and much more other options to meet almost every need**.**
- **Probability and statistic functions,** that vary from statistical distributions and calculating descriptive statistics (such as mean, variance and standard deviation) to Markov chain models. The primary packages for statistics are *breeze.stats* and *breeze.stats.distributions.*
- **Optimization,** which implies investigation of the function for a local or global minimum. Optimization methods are stored in the *breeze.optimize package*.
- **Linear algebra:** all basic operations rely on the netlib-java library, making Breeze extremely fast for algebraic computations.
- **Signal processing operations,** necessary for work with digital signals. The examples of important operations in Breeze are convolution and Fourier transformation, which decomposes the given function into a sum of sine and cosine components.

Breeze also provides plotting possibilities which we will discuss below.



## [Saddle](https://saddle.github.io/)

Another data manipulation toolkit for Scala is Saddle. It is a Scala analog of R and Python’s pandas library. Like the dataframes in pandas or R, Saddle is based on the Frame structure (2D indexed matrix).

The Vec and Mat classes are at the base of Series and Frame. You can implement different manipulations on these data structures, and use them for basic data analysis. Another great thing about Saddle is its robustness to missing values.



## [Scalalab](https://github.com/sterglee/scalalab)

ScalaLab is a Scala’s interpretation of MATLAB computing functionality**.** Moreover, ScalaLab can directly call and access the results of MATLAB scripts.

The main difference from the previous computation libraries is that ScalaLab uses its own domain-specific language called ScalaSci. Conveniently, Scalalab gets access to the variety of scientific Java and Scala libraries, so you can easily import your data and then use different methods to make manipulations and computations. Most of the techniques are similar to Breeze and Saddle. In addition, as in Breeze, there are plotting opportunities which allow further interpretation of the resulting data.



# NLP

## [Epic](http://www.scalanlp.org/)

Scala has some great natural language processing libraries as a part of ScalaNLP, including Epic and Puck. These libraries are mostly used as text parsers, with Puck being more convenient if you need to parse thousands of sentences due to its high-speed and GPU usage. Also, Epic is known as a prediction framework which employs structured prediction for building complex systems.

# Visualization

## [Breeze-vis](https://github.com/scalanlp/breeze/tree/master/viz)

As the name suggests, Breeze-viz is the plotting library developed by Breeze for Scala. It is based on the prominent Java charting library JFreeChart and has a MATLAB-like syntax. Although Breeze-viz has much fewer opportunities than MATLAB, matplotlib in Python, or R, it is still very helpful in the process of developing and establishing new models.

![img](https://miro.medium.com/max/60/1*cMqhh6sRGNipV3v-QiBFZw.png?q=20)

![img](https://miro.medium.com/max/2246/1*cMqhh6sRGNipV3v-QiBFZw.png)

## [Vegas](https://www.vegas-viz.org/)

Another Scala lib for data visualization is Vegas. It is much more functional than Breeze-viz and allows to make some plotting specifications such as filtering, transformations, and aggregations. It is similar in structure to Python’s Bokeh and Plotly.

Vegas provides declarative visualization that allows you to focus mainly on specifying what needs to be done with the data and conducting further analysis of the visualizations, without having to worry about the code implementation.

![img](https://miro.medium.com/max/60/1*tUP69MkZxYJiqAcmhA_aWA.png?q=20)

![img](https://miro.medium.com/max/1208/1*tUP69MkZxYJiqAcmhA_aWA.png)

# Machine Learning

## [Smile](https://haifengl.github.io/smile/)

Statistical Machine Intelligence and Learning Engine, or shortly Smile, is a promising modern machine learning system in some ways similar to Python’s scikit-learn. It is developed in Java and offers an API for Scala too. The library will amaze you with fast and extensive applications, efficient memory usage and a large set of machine learning algorithms for Classification, Regression, Nearest Neighbor Search, Feature Selection, etc.

![img](https://miro.medium.com/max/60/0*vWOV_SHxGu50BYtx.?q=20)

![img](https://miro.medium.com/max/1384/0*vWOV_SHxGu50BYtx.)

![img](https://miro.medium.com/max/60/0*Bp7YUJx4C-XRgSn6.?q=20)

![img](https://miro.medium.com/max/1478/0*Bp7YUJx4C-XRgSn6.)

## Apache Spark MLlib & ML

Built on top of Spark, MLlib library provides a vast variety of machine learning algorithms. Being written in Scala, it also provides highly functional API for Java, Python, and R, but opportunities for Scala are more flexible. The library consists of two separate packages: MLlib and ML. Let’s look at them in more detail one by one.

- MLlib is an RDD-based library that contains core machine learning algorithms for classification, clustering, unsupervised learning techniques supported by tools for implementing basic statistics such as correlations, hypothesis testing, and random data generation.
- ML is a newer library which, unlike MLlib, operates on data frames and datasets. The main purpose of the library is to give the ability to construct pipelines of different transformations on your data. The pipeline can be considered as a sequence of stages, where each stage is either a Transformer, that transforms one data frame into another data frame or an Estimator, an algorithm that can fit on a data frame to produce a Transformer.

Each package has its pros and cons and, in practice, it often proves more effective to apply both.

## [DeepLearning.scala](http://deeplearning.thoughtworks.school/)

DeepLearning.scala is an alternative machine learning toolkit that provides efficient solutions for deep learning. It utilizes mathematical formulas to create complex dynamic neural networks through a combination of object-oriented and functional programming. The library uses a wide range of types, as well as applicative type classes. The latter allows commencing multiple calculations simultaneously, which we consider crucial to have in a data scientist’s disposal. It’s worth mentioning that the library’s neural networks are programs and support all of Scala features.

## [Summing Bird](https://github.com/twitter/summingbird)

Summingbird is a domain-specific data processing framework which allows integration of batch and online MapReduce computations as well as the hybrid batch/online processing mode. The main catalyzer for designing the language came from Twitter developers who were often dealing with writing the same code twice: first for batch processing, then once more for online processing.

Summingbird consumes and generates two types of data: streams (infinite sequences of tuples), and snapshots regarded as the complete state of a dataset at some point in time. Finally, Summingbird provides platform implementations for Storm, Scalding, and an in-memory execution engine for testing purposes.

## [PredictionIO](http://predictionio.incubator.apache.org/index.html)

Of course, we can not ignore a machine learning server for constructing and deploying predictive engines called PredictionIO. It is built on Apache Spark, MLlib, and HBase and was even ranked on Github as the most popular Apache Spark-based machine learning product. It enables you to easily and efficiently build, evaluate and deploy engines, implement your own machine learning models, and incorporate them into your engine.

# Additional

# [Akka](https://akka.io/)

Developed by the Scala’s creator company, Akka is a concurrent framework for building distributed applications on a JVM. It uses an actor-based model, where an actor represents an object that receives messages and takes appropriate actions. Akka replaces the functionality of the Actor class that was available in the previous Scala versions.

The main difference, also considered as the most significant improvement, is the additional layer between the actors and the underlying system which only requires the actors to process messages, while the framework handles all other complications. All actors are hierarchically arranged, thus creating an Actor System which helps actors to interact with each other more efficiently and solve complex problems by dividing them into smaller tasks.

## [Spray](http://spray.io/)

Now let’s take a look at Spray — a suite of Scala libraries for constructing REST/HTTP web services built on top of Akka. It assures asynchronous, non-blocking actor-based high-performance request processing, while the internal Scala DSL provides a defining web service behavior, as well as efficient and convenient testing capabilities.

*UPD: Spray is no longer maintained and has been suspended by Akka HTTP. While most of the library functionality remains, there were some changes and improvements in streaming, module structure, routing DSL, etc. connected to this displacement. The* [*Migration Guide*](https://doc.akka.io/docs/akka-http/current/migration-guide/migration-from-spray.html?language=scala) *will help you find out about all of the developments.*

## [Slick](http://slick.lightbend.com/)

Last but not least on our list is Slick, which stands for Scala Language-Integrated Connection Kit. It is a library for creating and executing database queries that offer a variety of supported databases such as H2, MySQL, PostgreSQL, etc. Some databases are available via slick-extensions.

To build queries, Slick provides a powerful DSL, which makes the code look as if you were using the Scala collections. Slick supports both simple SQL queries, and strongly-typed joins of several tables. Moreover, simple subqueries can be used to construct more complex ones.
