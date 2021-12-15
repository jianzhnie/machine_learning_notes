# Self-Supervised Learning

I first got introduced to self-supervised learning in a [talk](https://www.youtube.com/watch?v=7I0Qt7GALVk&t=2639s) by Yann Lecun, where he introduced the “cake analogy” to illustrate the importance of self-supervised learning. In the talk, he said:

> “If intelligence is a cake, the bulk of the cake is self-supervised learning, the icing on the cake is supervised learning, and the cherry on the cake is reinforcement learning (RL).”

Though the analogy is [debated](https://www.dropbox.com/s/fdw7q8mx3x4wr0c/2017_12_xx_NIPS-keynote-final.pdf?dl=0), we have seen the impact of self-supervised learning in the Natural Language Processing field where recent developments (Word2Vec, Glove, ELMO, BERT) have embraced self-supervision and achieved state of the art results.

![img](https://amitness.com/images/self-supervised-nlp-to-vision.png)

Curious to know the current state of self-supervised learning in the Computer Vision field, I read up on existing literature on self-supervised learning applied to computer vision through a [recent survey paper](https://arxiv.org/abs/1902.06162) by Jing et. al.

In this post, I will explain what is self-supervised learning and summarize the patterns of problem formulation being used in self-supervised learning with visualizations.

## Why Self-Supervised Learning?

- Supervised learning requires usually a lot of labelled data. Getting good quality labelled data is an expensive and time-consuming task specially for a complex task such as object detection, instance segmentation where more detailed annotations are desired. 
-  There are also fields such as the medical field where getting enough data is a challenge itself. Thus, a major bottleneck in current supervised learning paradigm is the label generation part.
- On the other hand, the unlabeled data is readily available in abundance. The motivation behind Self-supervised learning is to learn useful representations of the data from unlabelled pool of data using self-supervision first and then fine-tune the representations with few labels for the supervised downstream task. 
- The downstream task could be as simple as image classification or complex task such as semantic segmentation, object detection, etc.

Lately, in natural language processing, Transformer models have achieved a lot of success. Transformers like Bert[1], T5[2], etc. applied the idea of self-supervision to NLP tasks. They first train the model with large unlabelled data and then fine-tuning the model with few labelled data examples. Similar self-supervised learning methods have been researched for computer vision as well and in this post, I will try to cover a few of those.

The fundamental idea for self-supervised learning is to create some auxiliary pre-text task for the model from the input data itself such that while solving the auxiliary task, the model learns the underlying structure of the data(for instance the structure of the object in case of image data). Many self-supervised learning methods have been researched but contrastive learning methods seem to be work better than others for computer vision, hence in this post, I would concentrate on contrastive learning-based self-supervised learning methods

![Manual Annotation in Supervised Learning](https://amitness.com/images/supervised-manual-annotation.png)

## What is Self-Supervised Learning?

Self supervised learning is a method that poses the following question to formulate an unsupervised learning problem as a supervised one:

> Can we design the task in such a way that we can generate virtually unlimited labels from our existing images and use that to learn the representations?

![Automating manual labeling with Self Supervised Learning](https://amitness.com/images/supervised-automated.png)

In self-supervised learning, we replace the human annotation block by creatively exploiting some property of data to set up a pseudo-supervised task. For example, here instead of labeling images as cat/dog, we could instead rotate them by 0/90/180/270 degrees and train a model to predict rotation. We can generate virtually unlimited training data from millions of images we have freely available on the internet.

![Self-supervised Learning Workflow Diagram](https://amitness.com/images/self-supervised-workflow.png)

Figure: End to End Workflow of Self-Supervised Learning

Once we learn representations from these millions of images, we can use transfer learning to fine-tune it on some supervised task like image classification of cats vs dogs with very few examples.

![img](https://amitness.com/images/self-supervised-finetuning.png)



Suppose we have a function f(represented by any deep network Resnet50 for example), given an input x, it gives us the features f(x) as output.

Contrastive Learning states that for any positive pairs x1 and x2, the respective outputs f(x1) and f(x2) should be similar to each other and for a negative input x3, f(x1) and f(x2) both should be dissimilar to f(x3).

![Contrastive Learning](https://miro.medium.com/max/1120/1*fdAU4VJtnclv0rGrfzUu4g.png)

Contrastive Learning Idea (Image by Author)

The positive pair could be two crops of same image(lets say top-left and bottom right), two frames of same video file, two augmented views(horizontally flipped version for instance) of same image, etc. and respective negatives could be a crop from different image, frame from different video, augmented view of different image, etc.

The idea of contrastive learning was first introduced in this paper “[Representation learning with contrastive predictive coding](https://arxiv.org/abs/1807.03748)”[3] by Aaron van den Oord et al. from DeepMind. The formulated contrastive learning task gave a strong basis for learning useful representations of the image data which is described next.

## Survey of Self-Supervised Learning Methods

Let’s now understand the various approaches researchers have proposed to exploit image and video properties and apply self-supervised learning for representation learning.

## A. Self-Supervised Learning from Image

### Pattern 1: Reconstruction（重建）

#### 1. **Image Colorization**（图像着色）

Formulation:

> What if we prepared pairs of (grayscale, colorized) images by applying grayscale to millions of images we have freely available?

![Data Generation for Image Colorization](https://amitness.com/images/ss-colorization-data-gen.png)

We could use an encoder-decoder architecture based on a fully convolutional neural network and compute the L2 loss between the predicted and actual color images.

![Architecture for Image Colorization](https://amitness.com/images/ss-image-colorization.png)

To solve this task, the model has to learn about different objects present in the image and related parts so that it can paint those parts in the same color. Thus, representations learned are useful for downstream tasks.![Learning to colorize images](https://amitness.com/images/ss-colorization-learning.png)

**Papers:**
[Colorful Image Colorization](https://arxiv.org/abs/1603.08511) | [Real-Time User-Guided Image Colorization with Learned Deep Priors](https://arxiv.org/abs/1705.02999) | [Let there be Color!: Joint End-to-end Learning of Global and Local Image Priors for Automatic Image Colorization with Simultaneous Classification](http://iizuka.cs.tsukuba.ac.jp/projects/colorization/en/)

#### 2. **Image Superresolution** （超分辨重建）

Formulation:

> What if we prepared training pairs of (small, upscaled) images by downsampling millions of images we have freely available?

![Training Data for Superresolution](https://amitness.com/images/ss-superresolution-training-gen.png)

GAN based models such as [SRGAN](https://arxiv.org/abs/1609.04802) are popular for this task. A generator takes a low-resolution image and outputs a high-resolution image using a fully convolutional network. The actual and generated images are compared using both mean-squared-error and content loss to imitate human-like quality comparison. A binary-classification discriminator takes an image and classifies whether it’s an actual high-resolution image(1) or a fake generated superresolution image(0). This interplay between the two models leads to generator learning to produce images with fine details.

![Architecture for SRGAN](https://amitness.com/images/ss-srgan-architecture.png)

Both generator and discriminator learn semantic features that can be used for downstream tasks.

**Papers**:
[Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802)

#### 3. **Image Inpainting**（图像修复）

Formulation:

> What if we prepared training pairs of (corrupted, fixed) images by randomly removing part of images?

![Training Data for Image Inpainting](https://amitness.com/images/ss-image-inpainting-data-gen.png)

Similar to superresolution, we can leverage a GAN-based architecture where the Generator can learn to reconstruct the image while discriminator separates real and generated images.

![Architecture for Image Inpainting](https://amitness.com/images/ss-inpainting-architecture.png)

For downstream tasks, [Pathak et al.](https://arxiv.org/abs/1604.07379) have shown that semantic features learned by such a generator give 10.2% improvement over random initialization on the [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html) semantic segmentation challenge while giving <4% improvements over classification and object detection.

**Papers**:
[Context encoders: Feature learning by inpainting](https://arxiv.org/abs/1604.07379)

#### 4. **Cross-Channel Prediction**（跨通道预测）

Formulation:

> What if we predict one channel of the image from the other channel and combine them to reconstruct the original image?

Zhang et al. used this idea in their paper called “Split-Brain Autoencoder”. To understand the idea of the paper, let’s take an example of a color image of tomato.

![img](https://amitness.com/images/split-brain-autoencoder.png)

Example adapted from “Split-Brain Autoencoder” paper

For this color image, we can split it into grayscale and color channels. Then, for the grayscale channel, we predict the color channel and for the color channel part, we predict the grayscale channel. The two predicted channels $$ X1$$  and $$X2 $$  are combined to get back a reconstruction of the original image. We can compare this reconstruction to the original color image to get a loss and improve the model.

This same setup can be applied for images with depth as well where we use the color channels and the depth channels from a `RGB-HHA` image to predict each other and compare output image and original image.

![img](https://amitness.com/images/split-brain-autoencoder-rgbhha.png)

Example adapted from “Split-Brain Autoencoder” paper

**Papers**:
[Split-Brain Autoencoders: Unsupervised Learning by Cross-Channel Prediction](https://arxiv.org/abs/1611.09842)

### Pattern 2: Common Sense Tasks（常识任务）

#### 1. **Image Jigsaw Puzzle **(拼图)

Formulation:

> What if we prepared training pairs of (shuffled, ordered) puzzles by randomly shuffling patches of images?

![Training Data For Image Jigsaw Puzzle](https://amitness.com/images/ss-image-jigsaw-data.png)

Even with only 9 patches, there can be 362880 possible puzzles. To overcome this, only a subset of possible permutations is used such as 64 permutations with the highest hamming distance.![Number of Permutations in Image Jigsaw](https://amitness.com/images/ss-jigsaw-permutations.png)

Suppose we use a permutation that changes the image as shown below. Let’s use the permutation number 64 from our total available 64 permutations.![Example of single permutation in jigsaw](https://amitness.com/images/ss-jigsaw-permutation-64.png)

Now, to recover back the original patches, [Noroozi et al.](https://arxiv.org/abs/1603.09246) proposed a neural network called context-free network (CFN) as shown below. Here, the individual patches are passed through the same siamese convolutional layers that have shared weights. Then, the features are combined in a fully-connected layer. In the output, the model has to predict which permutation was used from the 64 possible classes. If we know the permutation, we can solve the puzzle.![Architecture for Image Jigsaw Task](https://amitness.com/images/ss-jigsaw-architecture.png)

To solve the Jigsaw puzzle, the model needs to learn to identify how parts are assembled in an object, relative positions of different parts of objects and shape of objects. Thus, the representations are useful for downstream tasks in classification and detection.

**Papers**:
[Unsupervised learning of visual representations by solving jigsaw puzzles](https://arxiv.org/abs/1603.09246)

#### 2. **Context Prediction**（上下文预测）

Formulation:

> What if we prepared training pairs of (image-patch, neighbor) by randomly taking an image patch and one of its neighbors around it from large, unlabeled image collection?

![Training Data for Context Prediction](https://amitness.com/images/ss-context-prediction-gen.png)

To solve this pre-text task, [Doersch et al.](https://arxiv.org/abs/1505.05192) used an architecture similar to that of a jigsaw puzzle. We pass the patches through two siamese ConvNets to extract features, concatenate the features and do a classification over 8 classes denoting the 8 possible neighbor positions.![Architecture for Context Prediction](https://amitness.com/images/ss-context-prediction-architecture.png)

**Papers**:
[Unsupervised Visual Representation Learning by Context Prediction](https://arxiv.org/abs/1505.05192)

#### 3. **Geometric Transformation Recognition** (几何变换识别)

Formulation:

> What if we prepared training pairs of (rotated-image, rotation-angle) by randomly rotating images by (0, 90, 180, 270) from large, unlabeled image collection?

![Training Data for Geometric Transformation](https://amitness.com/images/ss-geometric-transformation-gen.png)

To solve this pre-text task, [Gidaris et al.](https://arxiv.org/abs/1803.07728) propose an architecture where a rotated image is passed through a ConvNet and the network has to classify it into 4 classes(0/90/270/360 degrees).![Architecture for Geometric Transformation Predction](https://amitness.com/images/ss-geometric-transformation-architecture.png)

Though a very simple idea, the model has to understand location, types and pose of objects in an image to solve this task and as such, the representations learned are useful for downstream tasks.

**Papers**:
[Unsupervised Representation Learning by Predicting Image Rotations](https://arxiv.org/abs/1803.07728)

### Pattern 3: Automatic Label Generation(自动标签生成)

#### 1. **Image Clustering**（图像聚类）

Formulation:

> What if we prepared training pairs of (image, cluster-number) by performing clustering on large, unlabeled image collection?

![Training Data for Image Clustering](https://amitness.com/images/ss-image-clustering-gen.png)

To solve this pre-text task, [Caron et al.](https://arxiv.org/abs/1807.05520) propose an architecture called deep clustering. Here, the images are first clustered and the clusters are used as classes. The task of the ConvNet is to predict the cluster label for an input image.![Architecture for Deep Clustering](https://amitness.com/images/ss-deep-clustering-architecture.png)

**Papers**:

- [Deep clustering for unsupervised learning of visual features](https://amitness.com/2020/04/deepcluster/)
- [Self-labelling via simultaneous clustering and representation learning](https://amitness.com/2020/04/illustrated-self-labelling/)
- [CliqueCNN: Deep Unsupervised Exemplar Learning](https://arxiv.org/abs/1608.08792)

#### 2. **Synthetic Imagery** （合成图像）

Formulation:

> What if we prepared training pairs of (image, properties) by generating synthetic images using game engines and adapting it to real images?

![Training Data for Sythetic Imagery](https://amitness.com/images/synthetic-imagery-data.png)

To solve this pre-text task, [Ren et al.](https://arxiv.org/pdf/1711.09082.pdf) propose an architecture where weight-shared ConvNets are trained on both synthetic and real images and then a discriminator learns to classify whether ConvNet features fed to it is of a synthetic image or a real image. Due to adversarial nature, the shared representations between real and synthetic images get better.![Architecture for Synthetic Image Training](https://amitness.com/images/ss-synthetic-image-architecture.png)

## B. Self-Supervised Learning From Video

### 1. **Frame Order Verification** （视频帧序验证）

Formulation:

> What if we prepared training pairs of (video frames, correct/incorrect order) by shuffling frames from videos of objects in motion?

![Training Data for Video Order](https://amitness.com/images/ss-frame-order-data-gen.png)

To solve this pre-text task, [Misra et al.](https://arxiv.org/pdf/1711.09082.pdf) propose an architecture where video frames are passed through weight-shared ConvNets and the model has to figure out whether the frames are in the correct order or not. In doing so, the model learns not just spatial features but also takes into account temporal features.

![Architecture for Frame Order Verification](https://amitness.com/images/ss-temporal-order-architecture.png)

**Papers**:

- [Shuffle and Learn: Unsupervised Learning using Temporal Order Verification](https://arxiv.org/abs/1603.08561)
- [Self-Supervised Video Representation Learning With Odd-One-Out Networks](https://arxiv.org/abs/1611.06646)



## Methods for SSL : Contrastive Learning

### AutoRegressive Model

說到理解資料，最直觀的機率模型就是建立 Data Likelihood，而在 Computer Vision 的領域中，最強大的莫過於 PixelCNN 這種 Pixel-by-Pixel 的建模方法了。利用 Chain Rule ，Network 能夠完整 Encode 所有資訊。

![img](Self-Supervised%20Learning.assets/1f2gbhX1iTDl6Q8IAFZi47Q-16395640025672.png)

[Conditional Image Generation with PixelCNN Decoders](https://papers.nips.cc/paper/6527-conditional-image-generation-with-pixelcnn-decoders.pdf)

但是 Representation Learning 在乎的並不是整個 Data Distribution; 而是怎麼得到更抽象、High-level 的表示法。End-to-End Training 讓人們發現，Deep Neural Network 是有能力解構資料的 Hierarchical Internal Representation 的，何不利用這種強大的能力呢？

也就是 **Learn the dataset, not the data points**.

### InfoNCE

DeepMind 在 2017 年 [InfoNCE](https://arxiv.org/abs/1807.03748) 提出一種基於 Mutual Information 修改 AutoRegrssive 的 Loss Function，稱為 InfoNCE。

![img](Self-Supervised%20Learning.assets/1c6KwTTF2eRTznJrcSpUdIw.png)

![img](Self-Supervised%20Learning.assets/1c6KwTTF2eRTznJrcSpUdIw.png)

從圖上說明是相當直觀的，模型基於看過的 Data 提取 Context (也就是 Feature) 並且對未來做預測。並且 Loss Function 的目標是讓 Data 與 Context 的 Mutual Information 越大越好。

![img](Self-Supervised%20Learning.assets/1iU830SUntaAF2e2KHp-8JQ-16395640939366.png)

Mutual Information 是廣義上的 Correlation Function。(當我們完全不了解系統的 Dynamics 或更深入的行為時，Mutual Information 依舊能作為估算) 它量化了我們能從 Context 中得到多少 Data 的資訊，稱為 Data 與 Context 之間的 Mutual Information。

![img](Self-Supervised%20Learning.assets/1fS_DOP7CJrhF5hbsrBlW_A-16395640939367.png)

![img](Self-Supervised%20Learning.assets/1fS_DOP7CJrhF5hbsrBlW_A.png)

首先，為了最大化 Mutual Information 讓 Network Model Distribution Ratio (而不像 generative model 是單純 model distribution)；並且用簡單的 Linear Map 作為從 Context 到 Future Data 的預測函數。

InfoNCE 寫法如下。其實他與常用的 Cross Entropy Loss 並沒有太大區別，差異只在於這個 Loss 並不是用於分類，而是注重在對資料的本身做預測。如果用先前 Time Series 的例子就是預測未來。

![img](Self-Supervised%20Learning.assets/1D7ufCqXQ7DWa4rTxexQIxw-163956409393710.png)

[Learning Deep Representations of Fine-grained Visual Descriptions](https://arxiv.org/abs/1605.05395)

我們把所唯一正確的預測稱為 Positive Sample; 其他的預測通通為 Negative Samples。文章接下來都使用 Contrastive Loss 來表示這種 Training 方法。

另外 InfoNCE 有一個 Weak Lower-Bound 在描述 N 的重要，也就是越多的 Negative Samples 時，Loss Function 越能控制 Mutual Information ，並且是以 Log 的方式 Scale (這給了 Metric Learning 一點 Hint, Batch Size 可能隨著 log scale)。

### CPC

第一個成功在 Image Classification 實踐出 InfoNCE 的是 CPC 這篇文章 (基本上是 DeepMind 同一個 team 的作品)。很直觀的利用在圖片上裁切 Patch 的方式，做出 Positive & Negative samples，實現 Contrastive Loss 。

![img](Self-Supervised%20Learning.assets/1Xrj-eSRDPxRjG0SVrWrGdg-163956425183112.png)

[Data-Efficient Image Recognition with Contrastive Predictive Coding](https://arxiv.org/abs/1905.09272)

這邊用到了三個 Network，分別是 feature extractor, context prediction 跟 downstream task network。這是因問 SSL 的 evaluation 方式不同的關係，這邊簡單說明一下。

![img](Self-Supervised%20Learning.assets/1rI7MQMbTsKHAarwTXWAGjA-163956425183114.png)

SSL 訓練出來的模型基本上不能直接使用，通常只能作為很強的 Pretrained Model。 因此要評估 Pretrained Model 好壞通常做 Linear Evaluation ，Fine-tune 一個 Linear Classifier 看能達到多少的準確度 (為了公平，通常這個 classifier 會用 grid search 得到)。

研究後來發現， SSL Pretrained Model 不僅得到 Linear Separable 的 Feature Space; 並且這些 Feature 是很豐富的，因為只需要少量的 Data 就可以達到很好的效果，這稱為 Efficient Classification Evaluation。像常常會測試，拿 ImageNet (有 1000 類 一千四百萬張圖片) 1% 的資料量 (也就是每個類別 Randomly choose 12 張圖片) 來訓練。這種 Evaluation 凸顯出 Feature 是能廣泛描述各種類別的，因此只要取少少的 Samples 就可以達到效果。

第三種 Evaluation 就是將 Pretrained Model 運用在各種 Vision Task 上，例如拿到 Object Detection 或 Segmentation 任務上依舊能表現不錯。

![img](Self-Supervised%20Learning.assets/1B7HlDIH7DHQYTTP5BL3rsg-163956425183116.png)

回到 CPC 這篇文章，ResNet-50 Linear Protocol 能達到 Top-1 71.5% 的準確率；在 Efficient Classification Protocol 上 ，能比原本 Supervised Learning 的方式省掉至少 50% ~ 80% 的資料(這邊是參數更多的 ResNet)。意味著透過 SSL Pretrained Model，我能夠少一些人工標記一樣能達到原本 Supervised Learning 的準確度。

![img](Self-Supervised%20Learning.assets/12LCysmeOI6pfSSWPWAnOBQ.png)

![img](Self-Supervised%20Learning.assets/12LCysmeOI6pfSSWPWAnOBQ-163956425183118.png)

### Contrastive Predictive Coding(CPC)

The central idea of CPC is to first divide the whole image into a coarse grid and given the upper few rows of the image, the task is to predict the lower rows of the same image. The motivation is, to accomplish this task, the model has to learn the structure of the object in the image(*for example seeing the face of a dog, the model should predict that it would have 4 legs*) and this would give us a useful representations for downstream tasks.

The whole auxiliary task could be summarized in 3 steps.

1. Given the 256 x 256 image, divide it into a 7x7 grid with each cell of size 64px and 32px overlap with neighboring cells.
2. Use an encoder model(g_enc) such as Resnet-50 to encode each grid cell into a 1024 dimension vector. The whole image is now transformed into 7x7x1024 tensor.
3. Given the top 3 rows of the transformed grid(7x7x1024 tensors), generate the last 3 rows of it(i.e. 3x3x1024 tensor). An auto-regressive generative model g_ar(PixelCNN for instance) is used for predicting the bottom 3 rows. PixelCNN in itself needs a separate post and is kept outside the scope of this article. In a 3000 ft overview, PixelCNN creates a context vector c_t from the given 3 top rows and sequentially predicts the bottom tows(z_t+2, z_t+3, z_t+4 from the figure below).

The below image depicts the task pictorially.

![img](https://miro.medium.com/max/60/1*TofsYD68wREzt9Keuu0now.png?q=20)

![img](https://miro.medium.com/max/1120/1*TofsYD68wREzt9Keuu0now.png)

Image Credits: Aaron van den Oord et al. [Representation learning with contrastive predictive coding](https://arxiv.org/abs/1807.03748)[3]

To train this model effectively, a loss function is required to enforce the similarity between positive pairs(correct patch prediction) and negative pairs(incorrect patch). For calculating the loss, the set *X* of N patches is used where *X* is the set of *N-1* negative samples and 1 positive sample(correct path). The *N-1* negatives are sampled randomly from all available patches of the same image(expect the correct patch) and different images in the batch. This loss is termed as InfoNCE loss where NCE stands for Noise Contrastive and it is shown below.

![img](https://miro.medium.com/max/60/1*6LCZvT6jJVuZpmErjSgMRQ.png?q=20)

![img](https://miro.medium.com/max/906/1*6LCZvT6jJVuZpmErjSgMRQ.png)

Here q is the network prediction, k+ is the positive patch(correct patch) and k- represents a set of N-1 negative patches. Note that k+, k- and q, all are in representation space i.e. output of g_enc and not into original image space.

In simple terms, the formula is equivalent to the log_softmax function. To calculate the similarity, the dot product is used. Take a dot product of all N samples with the prediction q and then calculate the log of softmax of the similarity score of the positive sample with the prediction q.

In order to validate the richness of the representations learnt by CPC, a **linear evaluation protocol** is used. A linear classifier is trained on top of the output of the frozen encoder model(g_enc) using the Imagenet dataset and then it is evaluated for the classification accuracy of the learnt classifier model on the Imagenet Val/Test set. Note that during this whole training process of the linear classifier, the backbone model(g_enc) is fixed and is not trained at all. The table below shows that the classification accuracy of CPC representations outperformed all the other methods introduced before CPC with 48.7% top-1 acc.

![img](https://miro.medium.com/max/60/1*eV_ro_d1GgwCmed3SWgGJg.png?q=20)

![img](https://miro.medium.com/max/835/1*eV_ro_d1GgwCmed3SWgGJg.png)

Imagenet Top-1% Accuracy of the Linear Classifier Trained on top of CPC representations[3]

Although CPC outperformed other unsupervised learning methods for representation learning, the classification accuracy was still very far from the supervised counterpart(Resnet-50 with 100% labels on the Imagenet has 76.5% top-1 accuracy). This idea of image crop discrimination was extended to instance discrimination and tightened the gap between self-supervised learning and supervised learning methods.

###  CMC

CPC 帶來巨大的好處，但什麼事情是重要的？難道以後都需要將一張圖切很多 Patch 來預測嗎？並不盡然。

在 CMC 這邊文章中表明了，使用不同場景 (View Point, Depth, Color Space) 來計算 Contrastive Loss 能達到非常好的效果，因此 Contrastive 本身 (也就是辨認 Positive & Negative Sample 之間的 Consistency) 才是關鍵。

![img](Self-Supervised%20Learning.assets/13vb6xn4jYbT9r2b3LM9ZIg.png)

![img](Self-Supervised%20Learning.assets/13vb6xn4jYbT9r2b3LM9ZIg-163956444011020.png)

[Contrastive Multiview Coding](https://arxiv.org/abs/1906.05849)

另外 Google 做了大規模的定性實驗，找出了幾個對 Visual Representation 最有影響的因子，因為篇幅關係就節錄下列重點

- Pretext Task 不一定能在 Downstream Task 上達到好的效果
- ResNet 的 skip-connection 能防止 feature quality 下降
- 增大 Model Size 與增加 Embedding Dimension 能有效提升 Performance

![img](Self-Supervised%20Learning.assets/1a5t2hYWMgwradO1hRI2tBg-163956444011022.png)

[Revisiting Self-Supervised Visual Representation Learning](https://arxiv.org/abs/1901.09005)

到目前為止基本上定調了 SSL 的走向

1. Contrastive Learning 能從 Data 中獲得相當豐富的資訊，不需要拘泥在 Patch 上
2. 使用 ResNet 這種 Backbone (而非早期 paper 強調VGG 能得到更好的 representation)

接下來的文章，都基於這樣的前提來 Improve 。

### Instance Discrimination Methods**

Instance Discrimination applies the concept of contrastive learning to whole image instance. Instance Discrimination method constraints that two augmented versions of the same image(positive pair) should have similar representations and two augmented versions of the different image(negative pair) should have different representations.

![img](https://miro.medium.com/max/60/1*_76TAj0aS5PUImFm_omqBA.png?q=20)

![img](https://miro.medium.com/max/1120/1*_76TAj0aS5PUImFm_omqBA.png)

Instance Discrimination. Image Credits: Deep Unsupervised Learning — P. Abbeel, P. Chen, J. Ho, A. Srinivas, A. Li, W. Yan — L7 Self-Supervised Learning

Two papers **MoCo** and **SimCLR** worked on the idea of instance discrimination around the same time. Their main objective is, under a certain kind of image augmentations, the learnt representations should be invariant. These certain image augmentations include horizontal flip, a random crop of a certain size, color channel distortion, gaussian blur, etc. Intuitively, these augmentations although change the input image, but does not change the class of the input image(a cat would be a cat after flipping and cropping as well) and hence their representations should also not change.

The whole method is as follows.

1. Given an image, create an image pair<x1, x2> with two randomly augmented versions of this image. Sample N-2 negative samples by randomly taking an augmented version of any other image in the dataset.
2. Pass this image pair to the encoder model(g_enc), and get the representations. Also, obtain the representations of the N-2 negatives.
3. Similar to CPC, apply InfoNCE loss to induce similarity between positive pairs and dissimilarity between negative pairs.

The major difference between SimCLR and MoCo is how they handle the negative samples.

### SimCLR: Simple Framework for Contrastive Learning

![img](Self-Supervised%20Learning.assets/18r66yZ1KWn_f0B9DuUNC7g.png)

另外兩點相當有意思，一點是 Data Augmentation 對 Contrastive Learning 的重要性; 一點是利用一個 Non-Linear Map 來避免 Feature 的 Information Loss。

![img](Self-Supervised%20Learning.assets/1OTkOQk-Po2KBvVtB4D4vBg.png)

SimCLR 的演算法相當簡單

SimCLR 做了大量的 Augmentation ，並且是 Augmentation 的組合。

![img](Self-Supervised%20Learning.assets/1J3FnhzcGbC3K_eaiVyyBWg.png)

![img](Self-Supervised%20Learning.assets/1J3FnhzcGbC3K_eaiVyyBWg-163956472336742.png)

用到幾種常見的 Data Augmentation

在實驗中發現， Color Distortion + Random Crop 效果提升的相當顯著。這是因為原本的 Random Crop 切出來的圖片 Distribution 其實相差不大，可以說是無效的 Patch （尤其對於 Contrastive Learning 來說相當不好），這兩種 Operation 混合後會讓 Distribution 大相徑庭，能產生更多有效的 Negative Samples。

![img](Self-Supervised%20Learning.assets/1SYCANHWgED_j1r4LMAz5dw.png)

如果有仔細看 CPC 原文的讀者也會發現，CPC 中提到的使用 Layer Normalization 取代 Batch Normalization 以避免 Model 太容易受到 Patch 的統計性質混淆有異曲同工之妙。

文章另一個亮點是，在算 Loss 之前加上一個 Layer，避免 Visual Representation 直接丟給 Contrastive Loss Function 計算。原因是這種比較 Similarity 的 Loss Function 可能會把一些資訊給丟掉。

![img](Self-Supervised%20Learning.assets/1MwUtfx2ZxPNlaxGMi-rcSg-163956472336846.png)

[Bag of Tricks and A Strong Baseline for Deep Person Re-identification](http://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf)

文中做了一些實驗，像是顏色、旋轉這種資訊，就會大幅度的被刪除；而加上一個 Nonlinar Map，這樣可以大幅度地保存 Information。這跟 ReID 中一篇有名的文章 [Bag of Tricks and A Strong Baseline for Deep Person Re-identification](http://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf) 的一種架構有點相似，讓不同層的 Feature 給不同的 Loss 免於 Information Loss。

![img](Self-Supervised%20Learning.assets/1szWadzxscM5wJ3QChpqUaw-163956472336848.png)

這張圖表現出了一件事， Contrastive Learning 是一種能從 Data 本身獲取資訊的 Loss Function ；而且 Data 本身的資訊量遠遠多出 Label 很多，因此就算經過非常長時間的 Training，Model 並沒有任何 Overfit 的跡象。

SimCLR considers all the images in the current batch as negative samples. Trained in this way, SimCLR representations achieve a top-1% accuracy of 69.3% on the Imagenet with the linear evaluation protocol described in the CPC section.

![img](https://miro.medium.com/max/998/1*-804ozRWp0Mmmd0LvZIN1A.png)

SimCLR linear classifier result on Imagenet[5]

In practice, InfoNCE loss performance is dependent upon the number of negatives and it requires a high number of negatives while calculating the loss term. Hence, simCLR is trained with a high number of batches(as big as 8k) for best results which are very computationally demanding and require multi-GPU training. This is considered as the main drawback of simCLR method.

### Momentum Contrast(MoCo)

這篇 MoCo 是 Kaiming He 在 FAIR (又是與 RGB 一起) 第一次對 SSL 問題提出的文章。算是一個相當 Engineering 的解法，來有效增加 Batch Size ，提升 Performance。

![img](Self-Supervised%20Learning.assets/1-yOt4YwRpNxy4unemT21Rw-163956453782824.png)

[Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722)

首先，我們可以完全忘掉過去 AutoRegressive 預測未來的觀點；或切 Patch 預測圖片結構。 MoCo 完全專注在 Contrastive Loss 上，將這個問題想像成有一個很大的 Dictionary ，Network 的目的就是一個 Encoder 要將圖片 Encode 成唯一的一把 Key ，此時要如何做到讓 Key Space Large and Consistent 是最重要的。

首先借鑒了另一篇 SSL 的文章 Memory Bank ，建一個 Bank 來存下所有的 Key （或稱為 Feature) 。這個方法相對把所有圖塞進 Batch 少用很多記憶體，但對於很大的 Dataset 依舊難以 Scale Up。

[Unsupervised Feature Learning via Non-Parametric Instance-level Discrimination](https://arxiv.org/abs/1805.01978)

MoCo 改善了 Bank ，用一個 Dynamic Queue 來取代，但是單純這樣做的話是行不通的，因為每次個 Key 會受到 Network 改變太多，Contrastive Loss 無法收斂。因此 MoCo 將 feature extractor 拆成兩個獨立的 Network: Encoder 與 Momentum Encoder。

![img](Self-Supervised%20Learning.assets/1eSvccPz9z2zUbd255y4HMA-163956453782828.png)

MoCo Algorithm

我們可以想像成這樣的情境，Momentum Encoder 隨著 Training Update 很慢，因此能提供很穩定的 Key ，也就是 Momentum Encoder 把這個 Key Space 先擺好; 當新的 Positive 經過 Encoder 進來時，跟所有其他的 Negative Sample 計算 Similarity ，如果 New Key 與原本的 Key 太近容易混淆，這時候的 Loss 絕大部分會 Update 給 Encoder (相當於找一個比較空的區域放 Key, 而不影響原本的其他 Key)。

![img](Self-Supervised%20Learning.assets/17m6jpsQfNRKyrCDuYls73A-163956453782830.png)

等 Encoder Update 完後，在用 Momentum Update Slow Encoder。並將這次的 Batch 放進 Dynamic Queue 中。

從以下實驗可以看到，MoCo 的表現幾乎與暴力將 Batch Size 增大得到的效果一樣，但是 Batch Size 沒辦法 Scale Up； Memory Bank 與 MoCo 有著一樣的 Scaling Property，但 MoCo 的 Momentum Update 能提供穩定的 Key Space 讓整體 Performance 可以提升約 2%。

![img](Self-Supervised%20Learning.assets/1OuYHo2eZyXI9sckQsM4A-w.png)

Momentum Contrast(MoCo) on the other hand, keeps a separate buffer of negatives(as high as 8k) and uses them for calculating the InfoNCE loss. This allows them to train MoCo with smaller batch sizes without compromising on accuracy.

MoCo keeps all recent mini-batches in fixed-size buffer for negatives(shown as x_0, x_1,x_2 in below image). To achieve superior results, a momentum encoder(**Θ_k**) is used which has exact architecture as the encoder (**Θ_q**)but the weights are slowly moving towards the actual encoder(shown below in the image).

![img](https://miro.medium.com/max/60/1*LiFISdNYgO-T38rz8KihFw.png?q=20)

![img](https://miro.medium.com/max/573/1*LiFISdNYgO-T38rz8KihFw.png)

Momentum encoder Update step[4]

The only role of the momentum encoder is to generate representations(k_0, k_1, k_2… in the image below) out of the negative samples. Mind that momentum encoder does not update the weights through backpropagation which makes the method more memory efficient and allows to keep a large buffer of negatives in memory.
Now in a short summary, given the first input x_query, the encoder generates the representations q, which is matched against another augmented version of the same image x_query(not shown in the image below) and also matched with the N negatives provided by the momentum encoder. Then the loss term is calculated using InfoNCE loss described in CPC section.

![img](https://miro.medium.com/max/60/1*5C0hQvRvTr3qgWJbLDWzIg.png?q=20)

![img](https://miro.medium.com/max/666/1*5C0hQvRvTr3qgWJbLDWzIg.png)

Image credits: He, Kaiming, et al. “Momentum contrast for unsupervised visual representation learning.”[4]

It might be difficult for the readers to understand the relevance of momentum encoder and application of MoCo only through the post, so please for more details, I highly recommend reading the [original paper](https://arxiv.org/abs/2002.05709) . I have just covered the main highlights of the idea.

In the second version of MoCo, the representations attained 71.1% accuracy on the Imagenet under linear evaluation protocol which went further close to the supervised Resnet-50 model(76.5%).

![img](https://miro.medium.com/max/60/1*rtPKRA8vV5biaXWU90RgGQ.png?q=20)

![img](https://miro.medium.com/max/1050/1*rtPKRA8vV5biaXWU90RgGQ.png)

MoCo linear classifier result on Imagenet[6]

### Bootstrap your own Latent(BYOL)

Although MoCo showed good results but the dependency on negative samples has complicated the method. Recently BYOL[7] was introduced based on the instance discrimination method and it has shown that using two networks similar to MoCo, better visual representations could be learnt even without negatives. Their method achieves 74.3% top-1 classification accuracy on ImageNet under linear evaluation protocol using Resnet50 and further reduces the gap with their supervised counterpart using wider and deeper Resnets. The results are shown below.

![img](https://miro.medium.com/max/60/1*6nah3xfeUKiJgX72JPcN4g.png?q=20)

![img](https://miro.medium.com/max/861/1*6nah3xfeUKiJgX72JPcN4g.png)

Image Credits: Bootstrap your own latent-a new approach to self-supervised learning[7]

The actual BYOL training method is worthy of a separate post and I would leave it for future posts.

[UPDATE]: Post for BYOL https://nilesh0109.medium.com/hands-on-review-byol-bootstrap-your-own-latent-67e4c5744e1b

**TL; DR:**

- Self-supervised learning is a representation learning method where a supervised task is created out of the unlabelled data.
- Self-supervised learning is used to reduce the data labelling cost and leverage the unlabelled data pool.
- Some of the popular self-supervised tasks are based on contrastive learning. Examples of contrastive learning methods are BYOL, MoCo, SimCLR, etc.

Below is the list of references used for writing this post.


## References

0. “[Self-Supervised Visual Feature Learning with Deep Neural Networks: A Survey.](https://arxiv.org/abs/1902.06162)” Jing, et al.

1. [Unsupervised Representation Learning by Predicting Image Rotations](https://arxiv.org/abs/1803.07728)
2. [Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles](https://arxiv.org/abs/1603.09246)
3. [Tracking Emerges by Colorizing Videos](https://eccv2018.org/openaccess/content_ECCV_2018/papers/Carl_Vondrick_Self-supervised_Tracking_by_ECCV_2018_paper.pdf)
4. [Conditional Image Generation with PixelCNN Decoders](https://papers.nips.cc/paper/6527-conditional-image-generation-with-pixelcnn-decoders.pdf)
5. [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)
6. [Learning Deep Representations of Fine-grained Visual Descriptions](https://arxiv.org/abs/1605.05395)
7. [Data-Efficient Image Recognition with Contrastive Predictive Coding](https://arxiv.org/abs/1905.09272)
8. [Contrastive Multiview Coding](https://arxiv.org/abs/1906.05849)
9. [Revisiting Self-Supervised Visual Representation Learning](https://arxiv.org/abs/1901.09005)
10. [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722)
11. [Unsupervised Feature Learning via Non-Parametric Instance-level Discrimination](https://arxiv.org/abs/1805.01978)
12. [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
13. [Billion-scale semi-supervised learning for image classification](https://arxiv.org/abs/1905.00546)
14. [Self-training with Noisy Student improves ImageNet classification](https://arxiv.org/abs/1911.04252)
15. [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382)
16. [RandAugment: Practical automated data augmentation with a reduced search space]()

