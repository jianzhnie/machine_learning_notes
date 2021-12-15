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
