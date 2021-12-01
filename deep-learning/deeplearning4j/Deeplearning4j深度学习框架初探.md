# Deeplearning4j深度学习框架初探

人工神经网络近年在机器学习算法中可谓一枝独秀，对自然语言处理（如机器翻译）和计算机视觉（如图像识别）等领域的进步起了重大作用。受惠于JVM久经考验的丰富生态，[Deeplearning4j](https://deeplearning4j.org/)（DL4J）深度学习库不但容易在各种平台上使用，而且性能优异（特别是方便借助Hadoop和Spark处理大数据），适合用于开发各类模式识别软件产品。

## 深度学习概论

### 机器学习与人工智能

映射是数学最核心的概念，众多问题都可以视为映射：

- 人脸识别可以视为一个从人脸图片集到人集的映射，把人脸图片对应到人脸所属的人
- 机器翻译可以视为一个从字符串集到字符串集的映射，把来自一个语言的字符串对应到另一语言中语义相同的某个字符串

为了解决问题，对于给定的映射f:X→Yf:X→Y，我们希望对于每个定义域中的值xx，都可以用计算机算出f(x)f(x)。对于一些对计算机来说是精确叙述的问题如排序，这是可能严格做到的。但对于人脸识别和机器翻译之类的问题，由于涉及到计算机以外的复杂概念，通常不能期望能完全准确地算出。于是，我们退而求其次，找一个能够计算的映射gg去近似ff，并且在某种意义下ff与gg接近。在过去，人们通过观察去设计这个近似映射gg，但对于复杂问题人手设计的启发式规则很快会变得难以维护，而且针对个别问题的方法往往过于特殊，导致重复劳动还不利于总结提高。为了用统一的框架解决不同的问题，人们又想办法自动地构造近似映射gg。当然我们必须给出关于ff的一些信息才可能完成这构造，通常给出映射在部分点处的值(x1,f(x1)),…,(xM,f(xM))(x1,f(x1)),…,(xM,f(xM))，由此构造gg的方法叫机器学习。显然即使给定了一个映射在一些点的值，它在其它点处的值仍然可以是任意的，机器学习只能建基于映射能够被相对“简单”的映射逼近的信念。机器学习通常的做法是先作出一类映射，然后从其中找出一个与已知数据最吻合的映射。可见，与基于大数定律的统计方法一样，只有在数据足够多时我们才能指望通过学习得出的映射gg确实能近似ff。幸运的是，随着互联网的兴起，数据源源不断地从人和各种传感器产生，收集数据变得容易，机器学习因而在许多问题变得可行。

机器学习现阶段还只能在一些特殊类型的问题上击败其它方法，不过它有更远大的愿景。在某种意义下，人也就是一个映射而已，考虑这个简单的模型：在时刻tt，人接受来自感官的刺激ItIt，结合经验EtEt作出反应OtOt，同时把经验丰富为Et+1Et+1。也就是说，人的行为就由映射(It,Et)↦(Et+1,Ot)(It,Et)↦(Et+1,Ot)完全刻画。假如我们能够完美地拟合这个映射，我们就能用机器去逼真地模仿人的全部行为。既然人们自认为人有智能，与人越接近就越智能，那么可以认为这机器具备所谓的智能（而且是比图灵的会话不可分性更强意义下的智能）。因此，机器学习的一个终极目标就是实现人工智能。

### 人工神经网络

由于数学方法一般在欧氏空间中最好处理，因此往往设计一种编码e:X→Rke:X→Rk把ff定义域中的对象对应到某个固定维数的欧氏空间，再设计一种解码单射d:Rl→Yd:Rl→Y把某个固定维数的欧氏空间中向量对应到ff值域中的对象，从而可以把问题归结为寻找近似映射h:Rk→Rlh:Rk→Rl再令g=d∘h∘eg=d∘h∘e即可。例如做图像识别时，图像可以对应到各像素亮度组成的向量；而做文章分类时，文章可以对应到各单词频率（或TF-IDF）组成的向量。对于机器翻译之类的问题，输入和输出的字符串都可以是任意长度的，但通过引入一个特殊的结束符和中间量，也可以和上述的人一样归结为输入和输出有固定维数的映射。

给定一列数据点{(Xi=e(xi),Yi=d−1(f(xi)))}Mi=1{(Xi=e(xi),Yi=d−1(f(xi)))}i=1M，机器学习首先假定hh可以被某族映射{h(⋅;Θ)}{h(⋅;Θ)}逼近，然后算出参数Θ=(θ1,…,θN)Θ=(θ1,…,θN)的估计值使ff与gg在已知点处的值接近。

首先，我们考虑映射族的构造，前馈神经网络的思想就是用一些称为神经元的基本的函数族的多重复合来构造。神经元形如h(z1,…,zk;a1,…,ak,b)=ϕ(a1z1+⋯+akzk+b)h(z1,…,zk;a1,…,ak,b)=ϕ(a1z1+⋯+akzk+b)，其中ϕϕ称为激活函数，只有单个神经元且ϕϕ为恒同时神经网络退化为线性回归。因为线性函数与线性函数的复合仍然线性，ϕϕ大多取非线性函数。而在有待训练时确定的参数中，各aiai称为权重，bb则称为偏移。

人工神经网络可以用图直观地表示，其中每个顶点是神经元，值沿着有向边在神经元间流动，直至到达输出神经元成为整个逼近映射的值的一个分量。设计人工神经网络时往往把它分为若干层，首层表示输入的各分量，最后一层表示输出的分量，边通常仅从一层指向下一层，一种粗暴的全连接设计就是让一层中各节点与下一层的所有节点都连接起来。所谓深度学习说是就是层数较多的意思，通常认为后面的层次保存了较高层次（更整体）的信息。反馈神经网络的图中也可能存在回路，与时序电路实现记忆的原理类似，这种设计被认为可模拟人“越想越像”的记忆，有利于上下文感知。神经网络的设计很大程度上是一门艺术，在实验前难以评判。

接下来，我们需要一个评估拟合好坏的指标，给定某损失函数LL，值L(Yi,h(Xi))L(Yi,h(Xi))越小越好。于是现在的问题是求出ΘΘ的值使1M∑Mi=1L(Yi,h(Xi;Θ))1M∑i=1ML(Yi,h(Xi;Θ))最小，这是一个优化问题。标准的数值解法是梯度下降法，从由启发式规则得出的参数值Θ0Θ0出发，然后逐步向负梯度方向修正Θk=Θk−1−ℓ∇Θ(1M∑Mi=1L(Yi,h(Xi;Θ)))|Θ=Θk−1Θk=Θk−1−ℓ∇Θ(1M∑i=1ML(Yi,h(Xi;Θ)))|Θ=Θk−1，其中ℓℓ称为学习率（太大容易不收敛，太小收敛则慢）。现实中因为MM太大，直接按上式计算会太慢且占内存，所以通常不会把所有样本一起用来算梯度，而是把样本分成多个小批次，每次迭代轮流用。另外，迭代公式也有一些变种，例如引入动量项。应当指出，由于hh不见得可微，而且往往不是凸函数，难以保证数值方法收敛于最小点。

最后指出，虽然上面主要谈监督学习。但某些非监督学习问题可转化为监督学习问题。例如有损压缩问题（类似的有语义散列问题）相当于寻找压缩函数f:X→Yf:X→Y和解压函数g:Y→Xg:Y→X使g∘fg∘f在某种意义下接近恒同映射，于是我们可以设计一个神经网络，各层中神经元个数先是递减再递增，输入数与输出数相同，数据集中数据同时用作输入和输出去训练网络，最后前半个神经网络就可作为压缩器而后半个神经网络就可作为解压器。类似技术还可以用于生成文本或图像之类。

## Deeplearning4j的基本用法

和使用其它库一样，首先需要满足依赖关系，例如对Maven项目，在`pom.xml`中加入以下样子的内容：

```yaml
<dependencies>
	<dependency>
		<groupId>org.nd4j</groupId>
		<artifactId>${nd4j.backend}</artifactId>
		<version>${nd4j.version}</version>
	</dependency>
	<dependency>
		<groupId>org.deeplearning4j</groupId>
		<artifactId>deeplearning4j-core</artifactId>
		<version>${dl4j.version}</version>
	</dependency>
	<dependency>
		<groupId>org.slf4j</groupId>
		<artifactId>slf4j-jdk14</artifactId>
		<version>1.7.25</version>
	</dependency>
</dependencies>
<properties>
	<!-- 如果你有支持CUDA的GPU可以后端应改为：nd4j-cuda-9.0-platform、nd4j-cuda-9.2-platform 或 nd4j-cuda-10.0-platform -->
	<nd4j.backend>nd4j-native-platform</nd4j.backend>
	<nd4j.version>1.0.0-beta4</nd4j.version>
	<dl4j.version>1.0.0-beta4</dl4j.version>
</properties>
```

接着我们用一个例子说明deeplearning4j的用法，这个例子使用卷积神经网络识别MNIST数据集中的手写数字图片。

```java
/** *****************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ***************************************************************************** */
package org.deeplearning4j.examples.mnist;
import java.io.*;
import java.util.*;
import org.datavec.api.io.labels.*;
import org.datavec.api.split.*;
import org.datavec.image.loader.*;
import org.datavec.image.recordreader.*;
import org.deeplearning4j.datasets.datavec.*;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.*;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.*;
import org.deeplearning4j.nn.weights.*;
import org.deeplearning4j.optimize.listeners.*;
import org.deeplearning4j.util.*;
import org.nd4j.evaluation.classification.*;
import org.nd4j.linalg.activations.*;
import org.nd4j.linalg.dataset.api.iterator.*;
import org.nd4j.linalg.dataset.api.preprocessor.*;
import org.nd4j.linalg.learning.config.*;
import org.nd4j.linalg.lossfunctions.*;
import org.nd4j.linalg.schedule.*;
import org.slf4j.*;
/**
 * Implementation of LeNet-5 for handwritten digits image classification on
 * MNIST dataset (99% accuracy)
 * <a href="http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf">[LeCun et al.,
 * 1998. Gradient based learning applied to document recognition]</a>
 * Some minor changes are made to the architecture like using ReLU and identity
 * activation instead of sigmoid/tanh, max pooling instead of avg pooling and
 * softmax output layer.
 * <p>
 * This example will download 15 Mb of data on the first run.
 *
 * @author hanlon
 * @author agibsonccc
 * @author fvaleri
 * @author dariuszzbyrad
 */
public class MnistClassifier{
	private static final Logger LOGGER=LoggerFactory.getLogger(MnistClassifier.class);
	//从http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz下载再解压到以下目录
	private static final String BASE_PATH="数据集路径";
	public static void main(String[] args) throws Exception{
		int height=28;    // height of the picture in px
		int width=28;     // width of the picture in px
		int channels=1;   // single channel for grayscale images
		int outputNum=10; // 10 digits classification
		int batchSize=54; // number of samples that will be propagated through the network in each iteration
		int nEpochs=1;    // number of training epochs
		int seed=1234;    // number used to initialize a pseudorandom number generator.
		Random randNumGen=new Random(seed);
		LOGGER.info("加载数据...");
		File trainData=new File(BASE_PATH+"/mnist_png/training");
		FileSplit trainSplit=new FileSplit(trainData,NativeImageLoader.ALLOWED_FORMATS,randNumGen);
		ParentPathLabelGenerator labelMaker=new ParentPathLabelGenerator(); // use parent directory name as the image label
		ImageRecordReader trainRR=new ImageRecordReader(height,width,channels,labelMaker);
		trainRR.initialize(trainSplit);
		DataSetIterator trainIter=new RecordReaderDataSetIterator(trainRR,batchSize,1,outputNum);
		// pixel values from 0-255 to 0-1 (min-max scaling)
		DataNormalization imageScaler=new ImagePreProcessingScaler();
		imageScaler.fit(trainIter);
		trainIter.setPreProcessor(imageScaler);
		// vectorization of test data
		File testData=new File(BASE_PATH+"/mnist_png/testing");
		FileSplit testSplit=new FileSplit(testData,NativeImageLoader.ALLOWED_FORMATS,randNumGen);
		ImageRecordReader testRR=new ImageRecordReader(height,width,channels,labelMaker);
		testRR.initialize(testSplit);
		DataSetIterator testIter=new RecordReaderDataSetIterator(testRR,batchSize,1,outputNum);
		testIter.setPreProcessor(imageScaler); // same normalization for better results
		LOGGER.info("网络配置和训练...");
		Map<Integer,Double> learningRateSchedule=new HashMap<>();
		learningRateSchedule.put(0,0.06);
		learningRateSchedule.put(200,0.05);
		learningRateSchedule.put(600,0.028);
		learningRateSchedule.put(800,0.0060);
		learningRateSchedule.put(1000,0.001);
		MultiLayerConfiguration conf=new NeuralNetConfiguration.Builder()
				.seed(seed)
				.l2(0.0005) // ridge regression value
				.updater(new Nesterovs(new MapSchedule(ScheduleType.ITERATION,learningRateSchedule)))
				.weightInit(WeightInit.XAVIER)
				.list()
				.layer(new ConvolutionLayer.Builder(5,5)
						.nIn(channels)
						.stride(1,1)
						.nOut(20)
						.activation(Activation.IDENTITY)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2,2)
						.stride(2,2)
						.build())
				.layer(new ConvolutionLayer.Builder(5,5)
						.stride(1,1) // nIn need not specified in later layers
						.nOut(50)
						.activation(Activation.IDENTITY)
						.build())
				.layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2,2)
						.stride(2,2)
						.build())
				.layer(new DenseLayer.Builder().activation(Activation.RELU)
						.nOut(500)
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
						.nOut(outputNum)
						.activation(Activation.SOFTMAX)
						.build())
				.setInputType(InputType.convolutionalFlat(height,width,channels)) // InputType.convolutional for normal image
				.build();
		MultiLayerNetwork net=new MultiLayerNetwork(conf);
		net.init();
		net.setListeners(new ScoreIterationListener(20));
		LOGGER.info("Total num of params: {}",net.numParams());
		// evaluation while training (the score should go down)
		for(int i=0;i<nEpochs;i++){
			net.fit(trainIter);
			LOGGER.info("Completed epoch {}",i);
			Evaluation eval=net.evaluate(testIter);
			LOGGER.info(eval.stats());
			trainIter.reset();
			testIter.reset();
		}
		File ministModelPath=new File(BASE_PATH+"/minist-model.zip");
		ModelSerializer.writeModel(net,ministModelPath,true);
		LOGGER.info("The MINIST model has been saved in {}",ministModelPath.getPath());
	}
}
```

下载并解压数据集后把`BASE_PATH`指向数据集所有目录，再运行上述类即可训练出一个卷积神经网络模型，模型会被保存到目录`BASE_PATH`下的`minist-model.zip`，另外会得到类似以下的测试结果（准确度约99%）：

```
========================Evaluation Metrics========================
 # of classes:    10
 Accuracy:        0.9900
 Precision:       0.9900
 Recall:          0.9899
 F1 Score:        0.9899
Precision, recall & F1: macro-averaged (equally weighted avg. of 10 classes)


=========================Confusion Matrix=========================
    0    1    2    3    4    5    6    7    8    9
---------------------------------------------------
  972    0    1    0    0    0    1    2    3    1 | 0 = 0
    0 1130    0    1    0    1    1    2    0    0 | 1 = 1
    0    2 1022    1    1    0    0    4    2    0 | 2 = 2
    0    0    1 1001    0    4    0    1    3    0 | 3 = 3
    0    0    1    0  974    0    1    0    2    4 | 4 = 4
    2    0    0    8    0  879    1    1    1    0 | 5 = 5
    3    2    0    0    2    3  948    0    0    0 | 6 = 6
    1    1    7    0    0    0    0 1014    1    4 | 7 = 7
    0    0    2    1    0    0    0    2  964    5 | 8 = 8
    0    0    1    2    5    0    0    4    1  996 | 9 = 9

Confusion matrix format: Actual (rowClass) predicted as (columnClass) N times
==================================================================
```

值得一提，由于DL4J用到了本地库，默认情况下Maven会把所有所有平台的本地库都纳入到类路径，这可能导致类加载失败。这时可以在`mvn`命令后加上类似`-Djavacpp.platform=linux-x86_64`（改成你的平台）的选项来限制只加载适用平台的本地库。

## 预处理

如前所述，原始数据如文本、声音、图像等等要转化为欧氏空间中张量形式的才能进入神经网络，因此需要提取、转换和加载的过程（ETL）。

### 提取

`InputSplit`接口管理数据位置，它的实现包括：

- `CollectionInputSplit`记录Uri数组或容器
- `FileSplit`记录根目录，并可设置递归、随机化和容许的文件格式
- `InputStreamInputSplit`记录`InputStream`
- `ListStringSplit`记录`java.util.List<java.util.List<java.lang.String>>`
- `NumberedFileInputSplit`记录含序号文件名模式（模式中用`%d`表示序号）和序号范围
- `OutputStreamInputSplit`记录`OutputStream`
- `StringSplit`记录`String`
- `TransformSplit`记录一个`BaseInputSplit`和到URI的转换

`RecordReader`接口用于把原始数据转换为一系列记录，其中使用前用`void initialize(Configuration conf, InputSplit split)`或`void initialize(InputSplit split)`初始化它。它的常用实现包括：

- `CollectionRecordReader`读入指定集合的元素集合作为记录
- `ComposableRecordReader`把多个指定`RecordReader`读入的对应记录连接成记录
- `ConcatenatingRecordReader`分别读入多个指定`RecordReader`读入的记录
- `CSVRecordReader`用于读入CSV记录，可指定跳过行数、分隔符和引号
- `CSVRegexRecordReader`用于读入CSV记录并对每个域按正则表达式的捕获组进一步分解，可指定跳过行数、分隔符、引号和各正则表达式
- `ExcelRecordReader`用于读入Excel记录，可指定跳过行数
- `FileRecordReader`用于读入文件
- `ImageRecordReader`用于读入图像，可指定高度、宽度、通道数、图像变换和标签生成器
- `InMemoryRecordReader`读入指定列表的元素列表作为记录
- `JacksonLineRecordReader`用于读入文件行，可指定域选择和对象映射
- `JacksonRecordReader`用于读入JSON、XML或YAML文件为记录，可指定域选择、对象映射、打乱与否、随机种子、标签生成器和标签位置
- `JDBCRecordReader`用于从关系式数据库读入记录，可指定SQL查询、数据元、元数据到记录查询和元数据索引
- `LibSvmRecordReader`用于读入libsvm记录
- `LineRecordReader`用于读入文件行
- `ListStringRecordReader`用于读入字符串列表
- `LocalTransformProcessRecordReader`用于转换各个记录可指定原`RecordReader`和`TransformProcess`
- `MapFileRecordReader`用于读入Hadoop MapFile对应于同一键的全体值，可指定键索引和随机化
- `MatlabRecordReader`用于读入Matlab记录
- `NativeAudioRecordReader`用于读入音频，可指定是否加标签和标签列表
- `ObjectDetectionRecordReader`用于读入图像及对象在其中位置，可指定高度、宽度、通道数、网格高度、网格宽度、图像变换和标签生成器
- `RegexLineRecordReader`用于把文本的各行按捕获组分解为记录，可指定正则表达式和跳过行数
- `SVMLightRecordReader`用于读入SVMLight格式的记录
- `TfidfRecordReader`用于把记录转换为TF-IDF向量
- `WavFileRecordReader`用于读入WAV音频，可指定是否加标签和标签列表

特别地`SequenceRecordReader`接口用于把原始数据转换为一系列记录列表：

| 类                                          | 记录列表                              | 记录               | 参数                                                         |
| :------------------------------------------ | :------------------------------------ | :----------------- | :----------------------------------------------------------- |
| `CodecRecordReader`                         | 视频                                  | 帧                 |                                                              |
| `CollectionSequenceRecordReader`            | 集合的集合                            | 集合               | 集合的集合的集合                                             |
| `CSVLineSequenceRecordReader`               | CSV文件的记录行                       | 域                 | 跳过行数、分隔符和引号                                       |
| `CSVMultiSequenceRecordReader`              | CSV文件由指定列表分隔符分隔的记录列表 | 域                 | 跳过行数、分隔符、引号、模式（`CONCAT`、`EQUAL_LENGTH`、`PAD`）、填充串 |
| `CSVNLinesSequenceRecordReader`             | CSV文件每若干条记录                   | 域                 | 列表行数、跳过行数和分隔符                                   |
| `CSVSequenceRecordReader`                   | CSV文件                               | 域                 | 跳过行数和分隔符                                             |
| `CSVVariableSlidingWindowRecordReader`      | CSV文件的滑动窗口中记录               | 域                 | 列表行数上限、跳过行数、分隔符和滑动距离                     |
| `InMemorySequenceRecordReader`              | 列表的列表                            | 列表               | 列表的列表的列表                                             |
| `JacksonLineSequenceRecordReader`           | 文件的行                              | 域                 | 可指定域选择和对象映射                                       |
| `LocalTransformProcessSequenceRecordReader` | 记录列表                              | 转换后记录         | 原`SequenceRecordReader`和`TransformProcess`                 |
| `MapFileSequenceRecordReader`               | Hadoop MapFile对应于同一键的全体值    | 值                 | 键索引和随机化                                               |
| `NativeCodecRecordReader`                   | 视频                                  | 帧                 |                                                              |
| `RegexSequenceRecordReader`                 | 文本文件的行                          | 正则表达式的捕获组 | 正则表达式、跳过行数、字符编码和错误处理器                   |
| `VideoRecordReader`                         | 图片目录                              | 图片               | 高度、宽度、是否加标签和标签列表                             |

### 转换

记录流往往需要经过转换才适合进入神经网络，这时就需要设置`TransformProcess`。

#### 设置输入模式

为此，我们需要描述转换前记录的模式，方法是先`new Schema.Builder()`，然后用以下方法增加列，最后调用`Schema build()`方法。

```
Schema.Builder 	addColumn(ColumnMetaData metaData)
Schema.Builder 	addColumnCategorical(java.lang.String name, java.util.List<java.lang.String> stateNames)
Schema.Builder 	addColumnCategorical(java.lang.String name, java.lang.String... stateNames)
Schema.Builder 	addColumnDouble(java.lang.String name)
Schema.Builder 	addColumnDouble(java.lang.String name, java.lang.Double minAllowedValue, java.lang.Double maxAllowedValue)
Schema.Builder 	addColumnDouble(java.lang.String name, java.lang.Double minAllowedValue, java.lang.Double maxAllowedValue, boolean allowNaN, boolean allowInfinite)
Schema.Builder 	addColumnFloat(java.lang.String name)
Schema.Builder 	addColumnFloat(java.lang.String name, java.lang.Float minAllowedValue, java.lang.Float maxAllowedValue)
Schema.Builder 	addColumnFloat(java.lang.String name, java.lang.Float minAllowedValue, java.lang.Float maxAllowedValue, boolean allowNaN, boolean allowInfinite)
Schema.Builder 	addColumnInteger(java.lang.String name)
Schema.Builder 	addColumnInteger(java.lang.String name, java.lang.Integer minAllowedValue, java.lang.Integer maxAllowedValue)
Schema.Builder 	addColumnLong(java.lang.String name)
Schema.Builder 	addColumnLong(java.lang.String name, java.lang.Long minAllowedValue, java.lang.Long maxAllowedValue)
Schema.Builder 	addColumnNDArray(java.lang.String columnName, long[] shape)
Schema.Builder 	addColumnsDouble(java.lang.String... columnNames)
Schema.Builder 	addColumnsDouble(java.lang.String pattern, int minIdxInclusive, int maxIdxInclusive)
Schema.Builder 	addColumnsDouble(java.lang.String pattern, int minIdxInclusive, int maxIdxInclusive, java.lang.Double minAllowedValue, java.lang.Double maxAllowedValue, boolean allowNaN, boolean allowInfinite)
Schema.Builder 	addColumnsFloat(java.lang.String... columnNames)
Schema.Builder 	addColumnsFloat(java.lang.String pattern, int minIdxInclusive, int maxIdxInclusive)
Schema.Builder 	addColumnsFloat(java.lang.String pattern, int minIdxInclusive, int maxIdxInclusive, java.lang.Float minAllowedValue, java.lang.Float maxAllowedValue, boolean allowNaN, boolean allowInfinite)
Schema.Builder 	addColumnsInteger(java.lang.String... names)
Schema.Builder 	addColumnsInteger(java.lang.String pattern, int minIdxInclusive, int maxIdxInclusive)
Schema.Builder 	addColumnsInteger(java.lang.String pattern, int minIdxInclusive, int maxIdxInclusive, java.lang.Integer minAllowedValue, java.lang.Integer maxAllowedValue)
Schema.Builder 	addColumnsLong(java.lang.String... names)
Schema.Builder 	addColumnsLong(java.lang.String pattern, int minIdxInclusive, int maxIdxInclusive)
Schema.Builder 	addColumnsLong(java.lang.String pattern, int minIdxInclusive, int maxIdxInclusive, java.lang.Long minAllowedValue, java.lang.Long maxAllowedValue)
Schema.Builder 	addColumnsString(java.lang.String... columnNames)
Schema.Builder 	addColumnsString(java.lang.String pattern, int minIdxInclusive, int maxIdxInclusive)
Schema.Builder 	addColumnsString(java.lang.String pattern, int minIdxInclusive, int maxIdxInclusive, java.lang.String regex, java.lang.Integer minAllowedLength, java.lang.Integer maxAllowedLength)
Schema.Builder 	addColumnString(java.lang.String name)
Schema.Builder 	addColumnString(java.lang.String name, java.lang.String regex, java.lang.Integer minAllowableLength, java.lang.Integer maxAllowableLength)
Schema.Builder 	addColumnTime(java.lang.String columnName, org.joda.time.DateTimeZone timeZone)
Schema.Builder 	addColumnTime(java.lang.String columnName, org.joda.time.DateTimeZone timeZone, java.lang.Long minValidValue, java.lang.Long maxValidValue)
Schema.Builder 	addColumnTime(java.lang.String columnName, java.util.TimeZone timeZone)
```

另外也可以借助`Join.Builder`把多个模式连接起来。

```
Join.Builder 	setJoinColumns(java.lang.String... joinColumnNames)
Join.Builder 	setJoinColumnsLeft(java.lang.String... joinColumnNames)
Join.Builder 	setJoinColumnsRight(java.lang.String... joinColumnNames)
Join.Builder 	setSchemas(Schema left, Schema right)
```

`Schema`（对应有`SequenceSchema`）类的以下静态方法可尝试推断模式：

```
Schema infer(java.util.List<Writable> record)
Schema inferMultiple(java.util.List<java.util.List<Writable>> record)
```

#### 设置转换

为设置`TransformProcess`，方法是先`new TransformProcessBuilder()`，然后用以下方法设置步骤，最后调用`TransformProcess build()`方法。

```
TransformProcess.Builder 	addConstantColumn(java.lang.String newColumnName, ColumnType newColumnType, Writable fixedValue)
TransformProcess.Builder 	addConstantDoubleColumn(java.lang.String newColumnName, double value)
TransformProcess.Builder 	addConstantIntegerColumn(java.lang.String newColumnName, int value)
TransformProcess.Builder 	addConstantLongColumn(java.lang.String newColumnName, long value)
TransformProcess.Builder 	appendStringColumnTransform(java.lang.String column, java.lang.String toAppend)
TransformProcess.Builder 	calculateSortedRank(java.lang.String newColumnName, java.lang.String sortOnColumn, WritableComparator comparator)
TransformProcess.Builder 	calculateSortedRank(java.lang.String newColumnName, java.lang.String sortOnColumn, WritableComparator comparator, boolean ascending)
TransformProcess.Builder 	categoricalToInteger(java.lang.String... columnNames)
TransformProcess.Builder 	categoricalToOneHot(java.lang.String... columnNames)
TransformProcess.Builder 	conditionalCopyValueTransform(java.lang.String columnToReplace, java.lang.String sourceColumn, Condition condition)
TransformProcess.Builder 	conditionalReplaceValueTransform(java.lang.String column, Writable newValue, Condition condition)
TransformProcess.Builder 	conditionalReplaceValueTransformWithDefault(java.lang.String column, Writable yesVal, Writable noVal, Condition condition)
TransformProcess.Builder 	convertFromSequence()
TransformProcess.Builder 	convertToDouble(java.lang.String inputColumn)
TransformProcess.Builder 	convertToInteger(java.lang.String inputColumn)
TransformProcess.Builder 	convertToSequence()
TransformProcess.Builder 	convertToSequence(java.util.List<java.lang.String> keyColumns, SequenceComparator comparator)
TransformProcess.Builder 	convertToSequence(java.lang.String keyColumn, SequenceComparator comparator)
TransformProcess.Builder 	convertToString(java.lang.String inputColumn)
TransformProcess.Builder 	doubleColumnsMathOp(java.lang.String newColumnName, MathOp mathOp, java.lang.String... columnNames)
TransformProcess.Builder 	doubleMathFunction(java.lang.String columnName, MathFunction mathFunction)
TransformProcess.Builder 	doubleMathOp(java.lang.String columnName, MathOp mathOp, double scalar)
TransformProcess.Builder 	duplicateColumn(java.lang.String column, java.lang.String newName)
TransformProcess.Builder 	duplicateColumns(java.util.List<java.lang.String> columnNames, java.util.List<java.lang.String> newNames)
TransformProcess.Builder 	filter(Condition condition)
TransformProcess.Builder 	filter(Filter filter)
TransformProcess.Builder 	floatColumnsMathOp(java.lang.String newColumnName, MathOp mathOp, java.lang.String... columnNames)
TransformProcess.Builder 	floatMathFunction(java.lang.String columnName, MathFunction mathFunction)
TransformProcess.Builder 	floatMathOp(java.lang.String columnName, MathOp mathOp, float scalar)
TransformProcess.Builder 	integerColumnsMathOp(java.lang.String newColumnName, MathOp mathOp, java.lang.String... columnNames)
TransformProcess.Builder 	integerMathOp(java.lang.String column, MathOp mathOp, int scalar)
TransformProcess.Builder 	integerToCategorical(java.lang.String columnName, java.util.List<java.lang.String> categoryStateNames)
TransformProcess.Builder 	integerToCategorical(java.lang.String columnName, java.util.Map<java.lang.Integer,java.lang.String> categoryIndexNameMap)
TransformProcess.Builder 	integerToOneHot(java.lang.String columnName, int minValue, int maxValue)
TransformProcess.Builder 	longColumnsMathOp(java.lang.String newColumnName, MathOp mathOp, java.lang.String... columnNames)
TransformProcess.Builder 	longMathOp(java.lang.String columnName, MathOp mathOp, long scalar)
TransformProcess.Builder 	ndArrayColumnsMathOpTransform(java.lang.String newColumnName, MathOp mathOp, java.lang.String... columnNames)
TransformProcess.Builder 	ndArrayDistanceTransform(java.lang.String newColumnName, Distance distance, java.lang.String firstCol, java.lang.String secondCol)
TransformProcess.Builder 	ndArrayMathFunctionTransform(java.lang.String columnName, MathFunction mathFunction)
TransformProcess.Builder 	ndArrayScalarOpTransform(java.lang.String columnName, MathOp op, double value)
TransformProcess.Builder 	normalize(java.lang.String column, Normalize type, DataAnalysis da)
TransformProcess.Builder 	offsetSequence(java.util.List<java.lang.String> columnsToOffset, int offsetAmount, SequenceOffsetTransform.OperationType operationType)
TransformProcess.Builder 	reduce(IAssociativeReducer reducer)
TransformProcess.Builder 	reduceSequence(IAssociativeReducer reducer)
TransformProcess.Builder 	reduceSequenceByWindow(IAssociativeReducer reducer, WindowFunction windowFunction)
TransformProcess.Builder 	removeAllColumnsExceptFor(java.util.Collection<java.lang.String> columnNames)
TransformProcess.Builder 	removeAllColumnsExceptFor(java.lang.String... columnNames)
TransformProcess.Builder 	removeColumns(java.util.Collection<java.lang.String> columnNames)
TransformProcess.Builder 	removeColumns(java.lang.String... columnNames)
TransformProcess.Builder 	renameColumn(java.lang.String oldName, java.lang.String newName)
TransformProcess.Builder 	renameColumns(java.util.List<java.lang.String> oldNames, java.util.List<java.lang.String> newNames)
TransformProcess.Builder 	reorderColumns(java.lang.String... newOrder)
TransformProcess.Builder 	replaceStringTransform(java.lang.String columnName, java.util.Map<java.lang.String,java.lang.String> mapping)
TransformProcess.Builder 	sequenceMovingWindowReduce(java.lang.String columnName, int lookback, ReduceOp op)
TransformProcess.Builder 	splitSequence(SequenceSplit split)
TransformProcess.Builder 	stringMapTransform(java.lang.String columnName, java.util.Map<java.lang.String,java.lang.String> mapping)
TransformProcess.Builder 	stringRemoveWhitespaceTransform(java.lang.String columnName)
TransformProcess.Builder 	stringToCategorical(java.lang.String columnName, java.util.List<java.lang.String> stateNames)
TransformProcess.Builder 	stringToTimeTransform(java.lang.String column, java.lang.String format, org.joda.time.DateTimeZone dateTimeZone)
TransformProcess.Builder 	timeMathOp(java.lang.String columnName, MathOp mathOp, long timeQuantity, java.util.concurrent.TimeUnit timeUnit)
TransformProcess.Builder 	transform(Transform transform)
TransformProcess.Builder 	trimSequence(int numStepsToTrim, boolean trimFromStart)
```

### 加载

`DataSetIterator`接口用于迭代小批次数据（一个小批次应该足够大以保证有代表性，同时不宜太大以减低内存需求，通常每小批次32至128个样本比较合理），每次返回一个`DataSet`，最多有一个输入和一个输出数组。`MultiDataSetIterator`类似，但返回`MultiDataSet`。前者的实现包括：

- `RecordReaderDataSetIterator`从`RecordReader`读取数据，可指定`RecordReader`、`WritableConverter`、批次大小、首个标签列、最后标签列、可能的标签数、最大批次数、回归/分类
- `CachingDataSetIterator`支持缓存
- `ExistingMiniBatchDataSetIterator`读入现有的小批次数据
- `KFoldIterator`支持k趟交叉验证
- `MiniBatchFileDataSetIterator`支持把数据分成小批次
- `MultipleEpochsIterator`支持多趟处理
- `SamplingDataSetIterator`支持随机抽样
- `ViewIterator`支持视图

它们可以通过`setPreProcessor(DataSetPreProcessor preProcessor)`方法设置预处理器：

- `ImagePreProcessingScaler`通常用于把0到255转化为0到1
- `NormalizerMinMaxScaler`通常用于把最小值和最大值分别放到0和1
- `NormalizerStandardize`通常用于把各特征分别正规化为均值0方差1
- `VGG16ImagePreProcessor`通常用于减去平均RGB值

注意部分需要统计值的预处理器用之前需要调用调用其`fit`方法。另外它们也有适用于`MultiDataSet`的版本如`ImageMultiPreProcessingScaler`、`MultiNormalizerHybrid`、`MultiNormalizerMinMaxScaler`、`MultiNormalizerStandardize`。

## 网络配置

人工神经网络配置用类`MultiLayerConfiguration`的对象表示，要配置它可以使用的链式API，先创建Builder：`new NeuralNetConfiguration.Builder()`，然后通过调用它的各个方法进行配置，最后调用`build()`。

### 训练配置

#### 激活函数

网络中的激活函数（神经元映射）可以用`activation`方法配置，部分常见值有（也可传递实现`IActivation`的类的对象）：

| `Activation`枚举常量 | 说明                                                         |
| :------------------- | :----------------------------------------------------------- |
| `CUBE`               | f(x)=x3f(x)=x3                                               |
| `ELU`                | f(x)={x,x>0α(exp(x)−1.0),其它f(x)={x,x>0α(exp⁡(x)−1.0),其它   |
| `HARDSIGMOID`        | f(x)=min{1,max{0,0.2x+0.5}}f(x)=min{1,max{0,0.2x+0.5}}       |
| `HARDTANH`           | f(x)={1,x>1x,其它f(x)={1,x>1x,其它                           |
| `IDENTITY`           | f(x)=xf(x)=x                                                 |
| `LEAKYRELU`          | f(x)=max{0,x}+αmin{0,x}f(x)=max{0,x}+αmin{0,x} 其中默认 α=0.01α=0.01 |
| `RATIONALTANH`       | f(x)=sgn(x)(1−1/(1+\|x\|+x2+1.41645x4))∼tanh(x)f(x)=sgn(x)(1−1/(1+\|x\|+x2+1.41645x4))∼tanh⁡(x) |
| `RECTIFIEDTANH`      | f(x)=max{0,tanh(x)}f(x)=max{0,tanh⁡(x)}                       |
| `RELU`               | f(x)={x,x>00,其它f(x)={x,x>00,其它                           |
| `RELU6`              | f(x)=min{max{x,θ},6}f(x)=min{max{x,θ},6}                     |
| `RRELU`              | f(x)=max{0,x}+alphamin{0,x}f(x)=max{0,x}+alphamin{0,x}       |
| `SELU`               | 正规化指数线性单位                                           |
| `SIGMOID`            | f(x)=1/(1+exp(−x))f(x)=1/(1+exp⁡(−x))                         |
| `SOFTMAX`            | fi(x)=exp(xi−θ)/∑jexp(xj−θ)fi(x)=exp⁡(xi−θ)/∑jexp⁡(xj−θ)       |
| `SOFTPLUS`           | f(x)=log(1+ex)f(x)=log⁡(1+ex)                                 |
| `SOFTSIGN`           | f(x)=x/(1+\|x\|)f(x)=x/(1+\|x\|)                             |
| `SWISH`              | f(x)=x/(1+e−x)f(x)=x/(1+e−x)                                 |
| `TANH`               | tanhtanh                                                     |
| `THRESHOLDEDRELU`    | f(x)={x,x>θ0,其它f(x)={x,x>θ0,其它                           |

#### 参数初始值

网络中的初始参数设置方式可以用`weightInit`方法配置（类似有`biasInit`），部分常见值有：

| `WeightInit`枚举常量          | 说明                                               |
| :---------------------------- | :------------------------------------------------- |
| `DISTRIBUTION`                | 由`dist`方法指定分布给出                           |
| `ZERO`                        | 0                                                  |
| `ONES`                        | 1                                                  |
| `SIGMOID_UNIFORM`             | 均匀分布 U(-r,r) 其中 r=4*sqrt(6/(fanIn + fanOut)) |
| `NORMAL`                      | 正态分布，均值 0 ，方差 1/sqrt(fanIn)              |
| `LECUN_UNIFORM`               | 均匀分布 U[-a,a] 其中 a=3/sqrt(fanIn).             |
| `UNIFORM`                     | 均匀分布 U[-a,a] 其中 a=1/sqrt(fanIn)              |
| `XAVIER`                      | 正态分布，均值 0, 方差 2.0/(fanIn + fanOut)        |
| `XAVIER_UNIFORM`              | 均匀分布 U(-s,s) 其中 s = sqrt(6/(fanIn + fanOut)) |
| `XAVIER_FAN_IN`               | 正态分布，均值0, 方差 1/fanIn                      |
| `RELU`                        | 正态分布，方差 2.0/nIn                             |
| `RELU_UNIFORM`                | 均匀分布 U(-s,s) 其中 s = sqrt(6/fanIn)            |
| `IDENTITY`                    | 单位方阵（只适用于方阵参数）                       |
| `VAR_SCALING_NORMAL_FAN_IN`   | 正态分布，均值 0, 方差 1.0/(fanIn)                 |
| `VAR_SCALING_NORMAL_FAN_OUT`  | 正态分布，均值 0, 方差 1.0/(fanOut)                |
| `VAR_SCALING_NORMAL_FAN_AVG`  | 正态分布，均值 0, 方差 1.0/((fanIn + fanOut)/2)    |
| `VAR_SCALING_UNIFORM_FAN_IN`  | 均匀分布 U[-a,a] 其中 a=3.0/(fanIn)                |
| `VAR_SCALING_UNIFORM_FAN_OUT` | 均匀分布 U[-a,a] 其中 a=3.0/(fanOut)               |
| `VAR_SCALING_UNIFORM_FAN_AVG` | 均匀分布 U[-a,a] 其中 a=3.0/((fanIn + fanOut)/2)   |

#### 优化器

网络中的优化器可以用`updater`方法配置（类似有`biasUpdater`），部分常见类有`AdaDelta`、`AdaGrad`、`AdaMax`、`Adam`、`AMSGrad`、`Nadam`、`Nesterovs`、`NoOp`、`RmsProp`、`Sgd`。

支持学习率的优化器也支持学习率调度，以便在不同迭代使用不同的学习率（通常在后面迭代使用更小的学习率），以下是一些实现`ISchedule`的类：

| 类                    | 说明                                                         |
| :-------------------- | :----------------------------------------------------------- |
| `ExponentialSchedule` | value(i) = initialValue * gamma^i                            |
| `InverseSchedule`     | value(i) = initialValue * (1 + gamma * i)^(-power)           |
| `MapSchedule`         | 基于用户提供的映射必须为iteration/epoch为 0时提供值          |
| `PolySchedule`        | value(i) = initialValue * (1 + i/maxIter)^(-power)           |
| `SigmoidSchedule`     | value(i) = initialValue * 1.0 / (1 + exp(-gamma * (iter - stepSize))) |
| `StepSchedule`        | value(i) = initialValue * gamma^( floor(iter/step) )         |

#### 正则化

可以用`l1(0.1)`、`l2(0.2)`对参数作L1、L2正则化。可以用`l1Bias(0.1)`、`l2Bias(0.2)`对偏移作L1、L2正则化。另外每轮迭代后可以作梯度正规化和其它约束。

#### 预防过度拟合

如要在训练阶段中修改激活的值，可用`dropOut`方法设置保持概率或修改方法：

| 类                | 说明                                    |
| :---------------- | :-------------------------------------- |
| `AlphaDropout`    | 企图同时保持均值和方差                  |
| `Dropout`         | 每个激活x以概率1-p置为0，以概率p设为x/p |
| `GaussianDropout` | 加入乘性1均值的高斯噪声                 |
| `GaussianNoise`   | 加入加性0均值的高斯噪声                 |

如要在训练阶段把修改参数的值，可用`weightNoise`方法设置：

| 类            | 说明                                    |
| :------------ | :-------------------------------------- |
| `DropConnect` | 每个参数x以概率1-p置为0，以概率p设为x/p |
| `WeightNoise` | 把加性或乘性的特定分布噪声加入到权重    |

### 层

要创建`MultiLayerNetwork`，调用`list()`方法后再使用`layer`方法可以加入层。另外可以设置以下选项：

- `pretrain(boolean)`方法可设置非监督训练
- `backprop(boolean)`方法可设置向后传播
- `setInputType(InputType)`方法可设置输入类型

#### 前馈层

前馈层是基本的层的构造。

| 类               | 说明                             |
| :--------------- | :------------------------------- |
| `DenseLayer`     | 全连通层                         |
| `EmbeddingLayer` | 输入正整数输出向量，只能用于首层 |

#### 输出层

输出层用作最后一层，可设置损失函数。

| 类                      | 说明                                                         |
| :---------------------- | :----------------------------------------------------------- |
| `OutputLayer`           | 标准的MLP/CNN分类/回归输出层，内置全连通层，二维输入和输出（每个样本一个行向量） |
| `LossLayer`             | 无参输出层，只有损失和激活函数，二维输入和输出（每个样本一个行向量），要求 nIn = nOut |
| `RnnOutputLayer`        | 用于反馈神经网络，3维（时间序列）输入和输出，内置时分全连通层 |
| `RnnLossLayer`          | `RnnOutputLayer`的无参版本，3维输入和输出                    |
| `CnnLossLayer`          | CNN中对每个位置作出预测，无参数，输入输出形如[minibatch, depth, height, width] |
| `Yolo2OutputLayer`      | 用于图像中对象检测                                           |
| `CenterLossOutputLayer` | `OutputLayer`企图最小化类中激活间距离的变种                  |

#### 卷积层

卷积层用于构建卷积神经网格，通常在图像处理中用于提取特征。

| 类                                   | 说明                                                        |
| :----------------------------------- | :---------------------------------------------------------- |
| `ConvolutionLayer`/`Convolution2D`   | 标准二维卷积层，输入输出形如 [minibatch,depth,height,width] |
| `Convolution1DLayer`/`Convolution1D` | 标准一维卷积层                                              |
| `Deconvolution2DLayer`               | 转置卷积，输出通常比输入大                                  |
| `SeparableConvolution2DLayer`        | 分深度的卷积层                                              |
| `SubsamplingLayer`                   | 通过最大值、平均或p范数缩小                                 |
| `Subsampling1DLayer`                 | 一维的采样                                                  |
| `Upsampling2D`                       | 通过重复行/列的值放大                                       |
| `Upsampling1D`                       | 一维的放大                                                  |
| `Cropping2D`                         | 裁剪层                                                      |
| `ZeroPaddingLayer`                   | 在边沿填充0                                                 |
| `ZeroPadding1DLayer`                 | 一维版本的填充                                              |
| `SpaceToDepth`                       | 把两个空间维数据按块转换为通道维                            |
| `SpaceToBatch`                       | 把两个空间维数据按块转换为批次维                            |

#### 反馈层

反馈层用于构建反馈神经网格，通常用于处理时间序列如文本。

| 类              | 说明                                                         |
| :-------------- | :----------------------------------------------------------- |
| `LSTM`          | 没有窥孔连接的LSTM RNN，支持CuDNN                            |
| `GravesLSTM`    | 有窥孔连接LSTM RNN，不支持CuDNN (故对于GPU, 宜用LSTM)        |
| `Bidirectional` | 把单向的RNN包装成双向RNN（前向和反向有独立的参数）           |
| `SimpleRnn`     | 标准/‘vanilla’ RNN层，由于大多长时依赖而不实用               |
| `LastTimeStep`  | 提取所包装（非双向）RNN层的最后时间步把[minibatch, size, timeSeriesLength]转换为 [minibatch, size] |

#### 非监督层

| 类                       | 说明                                   |
| :----------------------- | :------------------------------------- |
| `VariationalAutoencoder` | 编码解码器的可变实现，支持多种重构分布 |
| `AutoEncoder`            | 标准去噪自动编码器层                   |

#### 其它层

| 类                           | 说明                                                         |
| :--------------------------- | :----------------------------------------------------------- |
| `GlobalPoolingLayer`         | 求和、平均、最大值或p范数，对RNN/时间序列输入[minibatch, size, timeSeriesLength]输出[minibatch, size]，对CNN输入[minibatch, depth, h, w]输出[minibatch, depth] |
| `ActivationLayer`            | 对输入应用激活函数                                           |
| `DropoutLayer`               | 把丢弃实现为层                                               |
| `BatchNormalization`         | 批次正规化 2d (前馈), 3d (时间序列，参数与时间无关) 或 4d (CNN，参数与空间位置无关) |
| `LocalResponseNormalization` | CNN的局部响应正规化层，不常用                                |
| `FrozenLayer`                | 用于转移学习的冻结层（进一步训练时参数不变）                 |

### 顶点

要创建更灵活的`ComputationGraph`，可以调用`graphBuilder()`后使用`addVertex`方法。

| 类                   | 说明                                             |
| :------------------- | :----------------------------------------------- |
| `ElementWiseVertex`  | 对输入元素进行按分量运算如加、减、乘、平均、最值 |
| `L2NormalizeVertex`  | 用L2范数正规化输入                               |
| `L2Vertex`           | 计算两个数组间的L2距离                           |
| `MergeVertex`        | 沿维数1合并输入产生更大的输出数组                |
| `PreprocessorVertex` | 只有`InputPreProcessor`                          |
| `ReshapeVertex`      | 进行任意数组重整，但通常应首先考虑预处理器       |
| `ScaleVertex`        | 把输入乘以一个常数                               |
| `ShiftVertex`        | 把输入加上一个常数                               |
| `StackVertex`        | 沿维数0合并输入产生更大的输出数组                |
| `SubsetVertex`       | 沿维数1（nOut/通道）取输入的子集                 |
| `UnstackVertex`      | 沿维数0（小批次）取输入的子集                    |

## 训练与使用

有个神经网络配置`conf`后，可以通过`new MultiLayerNetwork(conf)`创建网络，然后应该调用`init()`初始化它。

为了在训练过程中能了解神经网络的状态，可以通过调用`setListeners(TrainingListener...)`注册监听器，它们会在训练期每个迭代完成后（或其它元事件）被调用。以下是一些有用的监听器：

| 类                               | 用途                                   |
| :------------------------------- | :------------------------------------- |
| `ScoreIterationListener`         | 每若干个迭代记录损失函数得分到日志     |
| `PerformanceListener`            | 每若干个迭代记录性能信息到日志         |
| `EvaluativeListener`             | 每若干个迭代用测试集评估性能           |
| `CheckpointListener`             | 周期性地保存检查点                     |
| `StatsListener`                  | 用于web训练界面                        |
| `CollectScoresIterationListener` | 每若干个迭代记录损失函数得分到一个列表 |
| `TimeIterationListener`          | 估计训练所需的剩余时间                 |

例如为了可视化地观察，可以使用以下代码在`localhost:9000`设立基于web的用户界面：

```
UIServer uiServer = UIServer.getInstance();
StatsStorage statsStorage = new InMemoryStatsStorage();
网络.setListeners(new StatsListener(statsStorage),new ScoreIterationListener(10));
uiServer.attach(statsStorage);
```

要训练网络只用调用`fit`方法传入数据集迭代器，然后可以用`evaluate`等方法评估分类或回归效果。

训练完的网络可以保存起来供以后使用，这时可以使用`ModelSerializer`类的静态方法`writeModel`和`restoreMultiLayerNetwork`可以把网络序列化和反序列化。这使得增量训练成为可能。

最后指出，有时我们希望微调一个现有的神经网络，修改部分层和部分参数而保持其它部分不变，这称为转移学习。`TransferLearning.Builder`类可以做这种事情。

## 下一步

我们仅仅介绍了deeplearning4j的基本用法，但它其实还有很多其它功能，例如：

- 当需要训练大型神经网络时，可以借助Spark用多台机器实现分布式训练（推荐的梯度分享实现需要依赖项`dl4j-spark-parameterserver_2.11`，老的参数平均实现则需要依赖项`dl4j-spark_2.11`）。
- 神经网络配置中有许多元参数如学习率，Arbiter可以自动寻找适合您的数据的元参数（需要依赖项`arbiter-deeplearning4j`、`arbiter-ui_2.11`）。
- 如果希望处理自然语言，不但有分句、分词、词频计算等基本工具，还有生成和比对语义散列等工具。
- 如果您有现成的Keras的HDF5格式模型，运气不太差的话可以通过`KerasModelImport.importKerasSequentialModelAndWeights`之类的方法导入它（需要依赖项`deeplearning4j-modelimport`）。
- zoo提供了一些现成的模型（部分更提供训练过的参数）如AlexNet、Darknet19、FaceNetNN4Small2、InceptionResNetV1、LeNet、ResNet50、SimpleCNN、TextGenerationLSTM、TinyYOLO、VGG16、VGG19，可以直接使用或作为修改的基础（需要依赖项`deeplearning4j-zoo`）。
- 如果您需要与底层的张量`INDArray`打交道，Nd4j库提供了相关的运算。

更详细的信息参见[API 文档](https://deeplearning4j.org/api/latest/)，另外可以参考官方的[例子](https://github.com/deeplearning4j/dl4j-examples)。
