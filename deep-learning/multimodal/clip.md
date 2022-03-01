# CLIP

## 1、动机

虽然深度学习在CV领域很成功，但是：

- typical vision datasets are labor intensive and costly to create while teaching only a narrow set of [visual concepts](https://www.zhihu.com/search?q=visual+concepts&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A1669790824})（当前的CV数据集标注劳动密集，成本高昂；）
-  standard vision models are good at one task and one task only, and require significant effort to adapt to a new task; （模型在单一任务上优秀，但难迁移到新任务）
- and models that perform well on benchmarks have disappointingly poor performance on stress tests, casting doubt on the entire deep learning approach to computer vision.（泛化性和[鲁棒性](https://www.zhihu.com/search?q=鲁棒性&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A1669790824})堪忧）

## 2、CLIP解决方案概述

a、互联网上较容易搜集到大量成对的文本和图像，对于任何一个图像文本对而言，文本其实可以认为是图像的标签。也就是说，互联网上天然就存在已经标注好的CV数据集，这解决了“动机”中的问题a。

b、而互联网上存在的这些已经标注好的CV数据集数量不仅大而且差异也大，当我们在这样的数据集上训练一个表达能力足够强的模型时，这个模型就能具备较强的泛化能力，较容易迁移到其他新任务上，这缓解了“动机”中的问题b和问题c。

上述两段话是CLIP解决方案的粗线条概括。

OpenAI的这项新工作CLIP可以解决上述问题，思路看起来很简单，看下图就知道了，简单来说CLIP是将Text Decoder从文本中提取的语义特征和Image Decoder从图像中提取的语义特征进行匹配训练：

### 3、CLIP细节

“CLIP的[解决方案概述](https://www.zhihu.com/search?q=解决方案概述&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A1678007012})”中的a比较容易解决：

- 确定一系列query，然后通过搜索引擎（通用搜索引擎如Google等，或者垂直领域搜索引擎Twitter等）搜索图片。

- 最后通过50万条query，搜索得到4亿个图像文本对，这样一个大规模的数据量，对应了“CLIP的解决方案概述”中的b。

既然我们有了图像文本对，那么最直接训练方法就是[metric learning](https://www.zhihu.com/search?q=metric+learning&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A1678007012})，CLIP的确也是这么干的。

### 4、一些思考

**NLP supervision如何理解。**

[NLP supervision](https://www.zhihu.com/search?q=NLP+supervision&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A1678007012})其实可以理解为**多维度**的标签，常见的分类任务只有**一维**，比如ImageNet-1K是一维，这一维的shape为1000，而NLP supervision可以认为是多维，比如[三维](https://www.zhihu.com/search?q=三维&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A1678007012})，shape为(A, B, C)，每一维描述一个concept。

经典图像分类的一维标签只包含一种数据集创建者自己设定的较粗的单种concept，比如ImageNet-1K里的“图像包含[egyptian cat](https://www.zhihu.com/search?q=egyptian+cat&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A1678007012})、Persian cat、[tiger cat](https://www.zhihu.com/search?q=tiger+cat&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A1678007012})、alley cat等1000种类别中的哪一种”。

而这个单种concept可以由多种concept组合而来，一方面可以减小歧义，一方面方便迁移，比如颜色、大小等等，NLP supervision就可以达到类似的一种效果，比如“a photo of guacamole, a type of food”这一句话告诉我们这是一张图片，图片里包含的是食物，这个食物是酸橘汁腌鱼。

**zero shot如何理解**

我们通过CLIP训练出来一个模型之后，满足以下条件的新任务都可以直接[zero shot](https://www.zhihu.com/search?q=zero+shot&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A1678007012})进行识别：

1、我们能够用文字描述清楚这个新分类任务中每个类别；

2、这个描述对应的概念在CLIP的训练集中出现过。这在经典[一维标签](https://www.zhihu.com/search?q=一维标签&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A1678007012})的图像分类中是不可实现的。

CLIP这种方法把分类转换为了**跨模态检索**，模型足够强的情况下，检索会比分类扩展性强。比如人脸识别，如果我们把人脸识别建模为分类任务，当gallery里新增加人脸后，类别数就变大了，我们就需要重新训练模型、更新类别数；如果我们将人脸识别建模为检索，当gallery里新增加人脸后，我们用已有的模型提取这个人脸的特征，后续流程不用变，也不用重新训练模型。

从检索这个角度来看，CLIP的zero shot其实就是把分类问题转化为了检索问题。

总结来看，CLIP能够zero shot识别，而且效果不错的原因在于：

1、训练集够大，zero shot任务的图像分布在训练集中有类似的，zero shot任务的concept在训练集中有相近的；

2、将分类问题转换为检索问题。

**concept的唬人之处**

CLIP这篇文章里提到通过NLP supervision能够让模型学到concept的概念，concept这个比较唬人，实际情况可能比我们想象的要差一些。

OpenAI没放数据集，我们只知道有50万query，但是这些query是怎么样的我们不得而知，比如是很简单的描述还是较复杂的描述。如果只是简单的描述，比如“black cat”、“lecture room”之类，那NLP encoder学到的所谓concept可能比较低级，稍微比[bag of words](https://www.zhihu.com/search?q=bag+of+words&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A1678007012})好一点，毕竟50万的量不是很大，query又简单。

我猜想情况可能近乎于如此，论据如下：

1、原文里作者说到“we found CLIP's performance to be less sensitive to the capacity of the text encoder”；

2、zero shot时，[prompt engineering](https://www.zhihu.com/search?q=prompt+engineering&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A1678007012})和prompt ensemble影响较大，原文里说到“When considered together, prompt engineering and ensembling improve ImageNet accuracy by almost 5%”。

**数据问题**

一个强大的方法不仅需要依赖强大的[模型结构](https://www.zhihu.com/search?q=模型结构&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A1678007012})，还仰仗大规模的训练集。模型结构决定下限，数据集决定上限。CLIP这种方法的上限如何，query的数量和质量至关重要。

如果图像文本对仅仅通过搜索的方式在互联网上获取，感觉文本不太可能复杂，这个会限制CLIP的上限。如果能找到一种获取大量图像文本对，而且文本还比较复杂，那么CLIP这种方法前景会非常不错。

**相比于传统图像分类方法的优势。**

这是显而易见的：每张图像的标签不再是一个名词，而是一个句子，因此以往被强行分成同类的图像，就有了“无限细粒度”的标签。例如ImageNet给图片打的标签是“[金毛寻回犬](https://www.zhihu.com/search?q=金毛寻回犬&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A"1670115915"})”，而这种配对的例子，就可以学习“金毛寻回犬”身处不同环境、在做不同事情的细微差别。

**相比于传统图像分类方法的劣势。**

主要还是文本和图像的配对关联性不够强。这是为什么作者反复强调要收集巨大的数据集，因为他们必须通过大数据的方式来压制[噪声](https://www.zhihu.com/search?q=噪声&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A"1670115915"})。从这个观点出发，我们可以看出些许未来的趋势（见下面第2和第3点）。

#### 图像和文本之间的交互方式。

直接用文本的encoding结果做为图像的监督信号，显然噪声太大了；能否借鉴captioning等方向的做法，允许图像和文本在encoding过程中多次交互，从而提升效果？当然，这里还是涉及到语言模型太大，无法高效训练。不过，OpenAI也可以选择暴力出奇迹，直接从头训练大规模的跨模态预训练模型。只是这样做的话，400M的数据集可能就太小了。
