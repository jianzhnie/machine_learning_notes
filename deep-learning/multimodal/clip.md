# CLIP && DALL-E

# CLIP

## 1. 动机

虽然深度学习在CV领域很成功，但是：

- typical vision datasets are labor intensive and costly to create while teaching only a narrow set of [visual concepts]()（当前的CV数据集标注劳动密集，成本高昂；）
-  standard vision models are good at one task and one task only, and require significant effort to adapt to a new task; （模型在单一任务上优秀，但难迁移到新任务）
- and models that perform well on benchmarks have disappointingly poor performance on stress tests, casting doubt on the entire deep learning approach to computer vision.（泛化性和[鲁棒性]()堪忧）

## 2. CLIP解决方案概述

a、互联网上较容易搜集到大量成对的文本和图像，对于任何一个图像文本对而言，文本其实可以认为是图像的标签。也就是说，互联网上天然就存在已经标注好的CV数据集，这解决了“动机”中的问题a。

b、而互联网上存在的这些已经标注好的CV数据集数量不仅大而且差异也大，当我们在这样的数据集上训练一个表达能力足够强的模型时，这个模型就能具备较强的泛化能力，较容易迁移到其他新任务上，这缓解了“动机”中的问题b和问题c。

上述两段话是CLIP解决方案的粗线条概括。

###  2. 1 CLIP 数据集

“CLIP的[解决方案概述]()”中的a比较容易解决：

- 确定一系列query，然后通过搜索引擎（通用搜索引擎如Google等，或者垂直领域搜索引擎Twitter等）搜索图片。

- 最后通过50万条query，搜索得到4亿个图像文本对，这样一个大规模的数据量，对应了“CLIP的解决方案概述”中的b。

​		CLIP提出了WebImageText数据集，该数据集包含4亿通过互联网得到的“image-text”对，包含的文本数据规模与GPT-2相当；与之相比， `VilBERT` 使用的Conceptual Captions数据集仅包含330万“image-text”对. WebImageText数据集的构造方式论文中并未详细介绍，只是简单提到由50万query以及每个query对应的2万张图片组成。

- CLIP超大的数据规模一方面有效减少了噪声、确保模型可以在ImageNet等图像分类数据集上有比较好的泛化能力；
- 另一方面也导致模型的训练效率成为问题，需要大算力的支撑或者更有效的训练方法。

综上所述，CLIP巧妙地将图文匹配任务和图像分类任务进行了关联，使得可以直接使用自然语言作为监督信号，虽然模型结构的设计还有改进空间，但毋庸置疑的是CLIP是openAI对大规模数据和大算力的又一次成功应用。

CLIP 的思路看起来很简单，看下图就知道了，简单来说CLIP是将 Text Decoder 从文本中提取的语义特征和Image Decoder从图像中提取的语义特征进行匹配训练：

## 3. 模型结构

### 3.1模型结构

![img](clip.assets/overview-a-20211204160940484.svg)

![img](clip.assets/overview-b.svg)

CLIP pre-trains an image encoder and a text encoder to predict which images were paired with which texts in our dataset. We then use this behavior to turn CLIP into a zero-shot classifier. We convert all of a dataset’s classes into captions such as “a photo of a *dog*” and predict the class of the caption CLIP estimates best pairs with a given image.

训练的伪代码也很simple（[https://github.com/openai/CLIP](https://link.zhihu.com/?target=https%3A//github.com/openai/CLIP)）：

```python
# image_encoder - ResNet or Vision Transformer
# text_encoder - CBOW or Text Transformer
# I[n, h, w, c] - minibatch of aligned images
# T[n, l] - minibatch of aligned texts
# W_i[d_i, d_e] - learned proj of image to embed
# W_t[d_t, d_e] - learned proj of text to embed
# t - learned temperature parameter
# extract feature representations of each modality
I_f = image_encoder(I) #[n, d_i]
T_f = text_encoder(T) #[n, d_t]
# joint multimodal embedding [n, d_e]
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)
# scaled pairwise cosine similarities [n, n]
logits = np.dot(I_e, T_e.T) * np.exp(t)
# symmetric loss function
labels = np.arange(n)
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss = (loss_i + loss_t)/2
```

### 3.2 算法原理

CLIP的基本算法原理相对比较简单，为了对image和text建立联系，首先分别对image和text进行特征提取：

- 其中image特征提取的backbone 选择的是 `ResNet `系列 或者` Vision Transformer`；
- 文本Encoder选择了``Transformer`结构， 如 bert 模型;
- 特征提取之后，由于做了normalize，直接相乘来计算余弦距离，
- 同一Pair对的结果趋近于1，不同pair对的结果趋近于0，因为就可以采用[对比损失loss]()（info-nce-loss），熟悉这个loss的同学应该都清楚，这种计算loss方式效果与batch size有很大关系，一般需要比较大的batch size才能有效果。
- 图片Encoder和文本Encoder均是从头训练，未加载现有的训练好的模型参数。

​	CLIP 在编码过程中没有进行文本和图片之间的交互而是直接通过计算文本编码和图像编码的余弦相似度来预测对应的图片和文本是否匹配，这与之前的一些多模态工作不同，比如图2所示的ViLBert和图3所示的VL-BERT[3]都在编码过程中进行了图片和文本之间的交互.

CLIP可能是出于训练效率的考虑选择了较为简洁的模型结构，但仍然会让人有疑问：能否通过引入之前多模态工作中的一些图像和文本联合编码的设计来降低[噪声]()，进而提升模型效果？能否通过加载预训练好的图片Encoder和文本Encoder参数来提升训练效率？

![img](clip.assets/v2-658afc5e6cf2e76b3e8a1b940d1909d2_720w.jpg)

![img](clip.assets/v2-4df5587e61a940f20aa305a39b0e427d_1440w.jpg)图3  VL-BERT的结构

## 4. 应用

### **图像分类：**

利用clip进行图像分类有两种方式:

- 一种是直接利用[zero-shot]() 方式进行预测， 将 text 假设为 a photo of [object], 分别对image 和 text进行特征提取以及余弦距离，当object为目标类别时，相似度最高，即为预测结果，通过实验惊奇的发现，直接利用zero-shot 的方式进行预测能够达到 76.2% 的acc，而且泛化性更好；
- 还有一种方式就是再重新finetune，同样也是对类别设计几种不同的文本，这样效果能够达到sota的水平.

## 5. 一些思考

### 5.1 **NLP supervision如何理解。**

[NLP supervision]()其实可以理解为**多维度**的标签，常见的分类任务只有**一维**，比如ImageNet-1K是一维，这一维的shape为1000，而NLP supervision可以认为是多维，比如[三维](https://www.zhihu.com/search?q=三维&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A1678007012})，shape为(A, B, C)，每一维描述一个concept。

经典图像分类的一维标签只包含一种数据集创建者自己设定的较粗的单种 concept，比如ImageNet-1K里的 “图像包含[egyptian cat]()、Persian cat、[tiger cat]()、alley cat等1000种类别中的哪一种”。

而这个单种concept可以由多种concept组合而来，一方面可以减小歧义，一方面方便迁移，比如颜色、大小等等，NLP supervision就可以达到类似的一种效果，比如“a photo of guacamole, a type of food” 这一句话告诉我们这是一张图片，图片里包含的是食物，这个食物是酸橘汁腌鱼。

### 5.2 **zero shot如何理解**

我们通过CLIP训练出来一个模型之后，满足以下条件的新任务都可以直接[zero shot]()进行识别：

1. 我们能够用文字描述清楚这个新分类任务中每个类别；

2. 这个描述对应的概念在CLIP的训练集中出现过。这在经典[一维标签]()的图像分类中是不可实现的。

CLIP这种方法把分类转换为了**跨模态检索**，模型足够强的情况下，检索会比分类扩展性强。比如人脸识别，如果我们把人脸识别建模为分类任务，当gallery里新增加人脸后，类别数就变大了，我们就需要重新训练模型、更新类别数；如果我们将人脸识别建模为检索，当gallery里新增加人脸后，我们用已有的模型提取这个人脸的特征，后续流程不用变，也不用重新训练模型。

从检索这个角度来看，CLIP的zero shot其实就是把分类问题转化为了检索问题。

总结来看，CLIP能够zero shot识别，而且效果不错的原因在于：

1. 训练集够大，zero shot任务的图像分布在训练集中有类似的，zero shot任务的concept在训练集中有相近的；

2. 将分类问题转换为检索问题。

### 5.3 **concept的唬人之处**

CLIP这篇文章里提到通过NLP supervision能够让模型学到concept的概念，concept这个比较唬人，实际情况可能比我们想象的要差一些。

OpenAI没放数据集，我们只知道有50万query，但是这些query是怎么样的我们不得而知，比如是很简单的描述还是较复杂的描述。如果只是简单的描述，比如“black cat”、“lecture room”之类，那NLP encoder学到的所谓concept可能比较低级，稍微比[bag of words]()好一点，毕竟50万的量不是很大，query又简单。

我猜想情况可能近乎于如此，论据如下：

1、原文里作者说到 “we found CLIP's performance to be less sensitive to the capacity of the text encoder”；

2、zero shot时，[prompt engineering]()和prompt ensemble影响较大，原文里说到“When considered together, prompt engineering and ensembling improve ImageNet accuracy by almost 5%”。

### 5.4 **数据问题**

一个强大的方法不仅需要依赖强大的[模型结构]()，还仰仗大规模的训练集。模型结构决定下限，数据集决定上限。CLIP这种方法的上限如何，query的数量和质量至关重要。

如果图像文本对仅仅通过搜索的方式在互联网上获取，感觉文本不太可能复杂，这个会限制CLIP的上限。如果能找到一种获取大量图像文本对，而且文本还比较复杂，那么CLIP这种方法前景会非常不错。

**相比于传统图像分类方法的优势。**

这是显而易见的：每张图像的标签不再是一个名词，而是一个句子，因此以往被强行分成同类的图像，就有了“无限细粒度”的标签。例如ImageNet给图片打的标签是“[金毛寻回犬]()”，而这种配对的例子，就可以学习“金毛寻回犬”身处不同环境、在做不同事情的细微差别。

**相比于传统图像分类方法的劣势。**

主要还是文本和图像的配对关联性不够强。这是为什么作者反复强调要收集巨大的数据集，因为他们必须通过大数据的方式来压制[噪声]()。从这个观点出发，我们可以看出些许未来的趋势。

### 5.5 图像和文本之间的交互方式

直接用文本的encoding结果做为图像的监督信号，显然噪声太大了；能否借鉴captioning等方向的做法，允许图像和文本在encoding过程中多次交互，从而提升效果？当然，这里还是涉及到语言模型太大，无法高效训练。不过，OpenAI也可以选择暴力出奇迹，直接从头训练大规模的跨模态预训练模型。只是这样做的话，400M的数据集可能就太小了。

# DALL-E

今年1月份openAI发布了DALL-E模型，能够根据文本生成效果惊艳的图像，并且参数量达到了120亿，被称为“图像版GPT-3”。

最近，openAI放出了DALL-E的论文和部分代码，使得大家能够进一步一窥究竟。根据本次开出的论文《[Zero-Shot Text-to-Image Generation]()，简单整理了一下DALL-E的整体架构，如图1所示，DALL-E的推理主要分为三个阶段，其中前两个阶段对应论文中的Stage One和 Stage Two。

![img](clip.assets/v2-9058b74d5a3aa63aebd8d4e1149f5e83_1440w.jpg)

​																	图1 DALL-E的整体架构

- 第一个阶段，将256×256的图片分为32×32个patch，然后使用训练好的[离散VAE模型]()的encoder将每个patch映射到大小为8192的词表中，最终一张图片转为用1024个token表示。
- 在第二个阶段，使用[BPE-encoder]()对文本进行编码，得到最多256个token，token数不满256的话padding到256；再将256个文本token与1024个图像token进行拼接，得到长度为1280的数据；最终将拼接的数据输入训练好的具有120亿参数的[Transformer模型]()。
- 在第三个阶段，对模型生成的图像进行采样，并使用同期发布的CLIP模型对采样结果进行排序，从而得到与文本最匹配的生成图像。

DALLE采样生成512个图片会用CLIP模型做rerank，CLIP给了论文和模型forward的代码，具体来说是在网上收集了4亿的图文对，然后在超大batch内做基于图文表征的对比学习。

DALLE包括三个独立训练得到的模型：dVAE，Transformer和CLIP，其中dVAE的训练与VAE基本相同，Transformer采用类似GPT-3的生成式预训练方法。下面对DALL-E采用的dVAE模型和Transformer模型做简单介绍，对CLIP感兴趣的朋友可以参考。

- **dVAE**

dVAE主要用来为图像的每个patch生成token表示，这次openAI开出的代码就是dVAE的推理代码。dVAE的encoder和decoder的机构较为简单，都是由bottleneck-style的res-block组成，但与常见的VAE相比，dVAE有以下两点区别：

1、dVAE的encoder是将图像的patch映射到8192的词表中，论文中将其分布设为

在词表向量上的[均匀分类分布]()，这是一个离散分布，由于不可导的问题，此时不能采用重参数技巧。DALL-E使用了[Gumbel-SoftMax trick]()来解决这个问题，对Gumbel-SoftMax trick感兴趣的朋友可以参考。

2、在重建图像时，真实的像素值是在一个[有界区间]()内，而VAE中使用的Gaussian

分布和Laplace分布都是在整个实数集上，这造成了不匹配的问题。为了解决这个问题，论文中提出了logit-Laplace分布，如下式所示：



![img](clip.assets/v2-f9467177341c842a1415bc815cd1da40_b.jpg)![img](clip.assets/v2-f9467177341c842a1415bc815cd1da40_1440w.jpg)



- **Transformer**

Dall-E中的Transformer结构由64层attention层组成，每层的注意力头数为62，每个注意力头的维度为64，因此，每个token的向量表示维度为3968。如图2所示，attention层使用了行注意力mask、列注意力mask和[卷积注意力mask]()三种稀疏注意力。

![img](clip.assets/v2-c5d07263f046757cd866c63ac8972eb4_b.jpg)![img](clip.assets/v2-c5d07263f046757cd866c63ac8972eb4_1440w.jpg)																							图2 Transformer使用的3种稀疏注意力

Transformer的输入如图3所示，其中[pad embd]()通过学习得到，根据论文介绍，为每个位置都训练了一个pad embd，即256个pad embd，在对文本token进行pad时，使用对应位置的pad embd。

![img](clip.assets/v2-f70f7f5a59d7735baaff70aca947911b_b.jpg)![img](clip.assets/v2-f70f7f5a59d7735baaff70aca947911b_1440w.jpg)																				图3 Transformer输入示意图（假设文本最大长度6)

总的来说，目前公开的DALL-E的实现在模型结构上并没有太多创新，而是合理利用了现有的模型结构进行组合，并采用了一些trick解决了遇到的问题，从而在大数据集上训练得到超大规模的模型，取得了令人惊艳的效果，这也符合openAI的一贯风格。但无论如何，DALL-E在[深度学习](https://www.zhihu.com/search?q=深度学习&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A"1764970196"})能力边界探索的道路上又前进了一步，也再一次展示了大数据和超大规模模型的魅力。美中不足的是，DALL-E包含了三个模块，更像是一个pipeline，而对于普通的研究者来说，要运行这样一个复杂的大规模模型是一件很困难的事情。
