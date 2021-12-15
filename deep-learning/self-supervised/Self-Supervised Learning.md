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



## Methods for SSL

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

### **Instance Discrimination Methods**

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

### SimCLR

SimCLR considers all the images in the current batch as negative samples. Trained in this way, SimCLR representations achieve a top-1% accuracy of 69.3% on the Imagenet with the linear evaluation protocol described in the CPC section.

![img](https://miro.medium.com/max/60/1*-804ozRWp0Mmmd0LvZIN1A.png?q=20)

![img](https://miro.medium.com/max/998/1*-804ozRWp0Mmmd0LvZIN1A.png)

SimCLR linear classifier result on Imagenet[5]

In practice, InfoNCE loss performance is dependent upon the number of negatives and it requires a high number of negatives while calculating the loss term. Hence, simCLR is trained with a high number of batches(as big as 8k) for best results which are very computationally demanding and require multi-GPU training. This is considered as the main drawback of simCLR method.

### Momentum Contrast(MoCo)

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

- “[Self-Supervised Visual Feature Learning with Deep Neural Networks: A Survey.](https://arxiv.org/abs/1902.06162)” Jing, et al.

