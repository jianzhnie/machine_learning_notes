# [The Illustrated SimCLR Framework](https://amitness.com/2020/03/illustrated-simclr/)

In recent years, [numerous self-supervised learning methods](https://amitness.com/2020/02/illustrated-self-supervised-learning/) have been proposed for learning image representations, each getting better than the previous. But, their performance was still below the supervised counterparts.

This changed when **Chen et. al** proposed a new framework in their research paper “[SimCLR: A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)”. The SimCLR paper not only improves upon the previous state-of-the-art self-supervised learning methods but also beats the supervised learning method on ImageNet classification when scaling up the architecture.

In this article, I will explain the key ideas of the framework proposed in the research paper using diagrams.

## The Nostalgic Intuition[Permalink](https://amitness.com/2020/03/illustrated-simclr/#the-nostalgic-intuition)

As a kid, I remember we had to solve such puzzles in our textbook.

![Find a Pair Exercise](https://amitness.com/images/contrastive-find-a-pair.png)

The way a child would solve it is by looking at the picture of the animal on the left side, know its a cat, then search for a cat on the right side.

![Child Matching Animal Pairs](https://amitness.com/images/contrastive-puzzle.gif)

> “Such exercises were prepared for the child to be able to recognize an object and contrast that to other objects. Can we similarly teach machines?”

It turns out that we can through a technique called **Contrastive Learning**. It attempts to teach machines to distinguish between similar and dissimilar things.

![Contrastive Learning Block](https://amitness.com/images/simclr-contrastive-learning.png)

## Problem Formulation for Machines[Permalink](https://amitness.com/2020/03/illustrated-simclr/#problem-formulation-for-machines)

To model the above exercise for a machine instead of a child, we see that we require 3 things:

### 1. Examples of similar and dissimilar images[Permalink](https://amitness.com/2020/03/illustrated-simclr/#1-examples-of-similar-and-dissimilar-images)

We would require example pairs of images that are similar and images that are different for training a model.

![Pair of similar and dissimilar images](https://amitness.com/images/contrastive-need-one.png)

The supervised school of thought would require a human to manually annotate such pairs. To automate this, we could leverage [self-supervised learning](https://amitness.com/2020/02/illustrated-self-supervised-learning/). But how do we formulate it?

![Manually Labeling pairs of Images](https://amitness.com/images/contrastive-supervised-approach.png)

![Self-supervised Approach to Labeling Images](https://amitness.com/images/contrastive-self-supervised-approach.png)

### 2. Ability to know what an image represents[Permalink](https://amitness.com/2020/03/illustrated-simclr/#2-ability-to-know-what-an-image-represents)

We need some mechanism to get representations that allow the machine to understand an image.

![Converting Image to Representations](https://amitness.com/images/image-representation.png)

### 3. Ability to quantify if two images are similar[Permalink](https://amitness.com/2020/03/illustrated-simclr/#3-ability-to-quantify-if-two-images-are-similar)

We need some mechanism to compute the similarity of two images.

![Computing Similarity between Images](https://amitness.com/images/image-similarity.png)

## The SimCLR Framework Approach[Permalink](https://amitness.com/2020/03/illustrated-simclr/#the-simclr-framework-approach)

The paper proposes a framework called “**SimCLR**” for modeling the above problem in a self-supervised manner. It blends the concept of *Contrastive Learning* with a few novel ideas to learn visual representations without human supervision.

## SimCLR Framework[Permalink](https://amitness.com/2020/03/illustrated-simclr/#simclr-framework)

The idea of SimCLR framework is very simple. An image is taken and random transformations are applied to it to get a pair of two augmented images xixi and xjxj. Each image in that pair is passed through an encoder to get representations. Then a non-linear fully connected layer is applied to get representations z. The task is to maximize the similarity between these two representations zizi and zjzj for the same image.

![General Architecture of the SimCLR Framework](https://amitness.com/images/simclr-general-architecture.png)

## Step by Step Example[Permalink](https://amitness.com/2020/03/illustrated-simclr/#step-by-step-example)

Let’s explore the various components of the SimCLR framework with an example. Suppose we have a training corpus of millions of unlabeled images.

![Corpus of millions of images](https://amitness.com/images/simclr-raw-data.png)

### **1. Self-supervised Formulation** [Data Augmentation][Permalink](https://amitness.com/2020/03/illustrated-simclr/#1-self-supervised-formulation-data-augmentation)

First, we generate batches of size N from the raw images. Let’s take a batch of size N = 2 for simplicity. In the paper, they use a large batch size of 8192.

![A single batch of images](https://amitness.com/images/simclr-single-batch.png)

The paper defines a random transformation function T that takes an image and applies a combination of `random (crop + flip + color jitter + grayscale)`.

![Random Augmentation on Image](https://amitness.com/images/simclr-random-transformation-function.gif)

For each image in this batch, a random transformation function is applied to get a pair of 2 images. Thus, for a batch size of 2, we get 2*N = 2*2 = 4 total images.

![Augmenting images in a batch for SimCLR](https://amitness.com/images/simclr-batch-data-preparation.png)

### 2. Getting Representations [Base Encoder][Permalink](https://amitness.com/2020/03/illustrated-simclr/#2-getting-representations-base-encoder)

Each augmented image in a pair is passed through an encoder to get image representations. The encoder used is generic and replaceable with other architectures. The two encoders shown below have shared weights and we get vectors hihi and hjhj.

![Encoder part of SimCLR](https://amitness.com/images/simclr-encoder-part.png)

In the paper, the authors used [ResNet-50](https://arxiv.org/abs/1512.03385) architecture as the ConvNet encoder. The output is a 2048-dimensional vector h.

![ResNet-50 as encoder in SimCLR](https://amitness.com/images/simclr-paper-encoder.png)

### 3. Projection Head[Permalink](https://amitness.com/2020/03/illustrated-simclr/#3-projection-head)

The representations hihi and hjhj of the two augmented images are then passed through a series of non-linear **Dense -> Relu -> Dense** layers to apply non-linear transformation and project it into a representation zizi and zjzj. This is denoted by g(.)g(.) in the paper and called projection head.

![Projection Head Component of SimCLR](https://amitness.com/images/simclr-projection-head-component.png)

### 4. Tuning Model: [Bringing similar closer][Permalink](https://amitness.com/2020/03/illustrated-simclr/#4-tuning-model-bringing-similar-closer)

Thus, for each augmented image in the batch, we get embedding vectors zz for it.

![Projecting image to embedding vectors](https://amitness.com/images/simclr-projection-vectors.png)

From these embedding, we calculate the loss in following steps:

#### a. Calculation of Cosine Similarity[Permalink](https://amitness.com/2020/03/illustrated-simclr/#a-calculation-of-cosine-similarity)

Now, the similarity between two augmented versions of an image is calculated using cosine similarity. For two augmented images xixi and xjxj, the cosine similarity is calculated on its projected representations zizi and zjzj.

![Cosine similarity between image embeddings](https://amitness.com/images/simclr-cosine-similarity.png)

si,j=zTizj(τ||zi||||zj||)si,j=ziTzj(τ||zi||||zj||)

where

- ττ is the adjustable temperature parameter. It can scale the inputs and widen the range `[-1, 1]` of cosine similarity
- ∥zi∥‖zi‖ is the norm of the vector.

The pairwise cosine similarity between each augmented image in a batch is calculated using the above formula. As shown in the figure, in an ideal case, the similarities between augmented images of cats will be high while the similarity between cat and elephant images will be lower.

![Pairwise cosine similarity between 4 images](https://amitness.com/images/simclr-pairwise-similarity.png)

#### b. Loss Calculation[Permalink](https://amitness.com/2020/03/illustrated-simclr/#b-loss-calculation)

SimCLR uses a contrastive loss called “**NT-Xent loss**” (**Normalized Temperature-Scaled Cross-Entropy Loss**). Let see intuitively how it works.

First, the augmented pairs in the batch are taken one by one.

![Example of a single batch in SimCLR](https://amitness.com/images/simclr-augmented-pairs-batch.png)

Next, we apply the softmax function to get the probability of these two images being similar.

![Softmax Calculation on Image Similarities](https://amitness.com/images/simclr-softmax-calculation.png)

This softmax calculation is equivalent to getting the probability of the second augmented cat image being the most similar to the first cat image in the pair. Here, all remaining images in the batch are sampled as a dissimilar image (negative pair). Thus, we don’t need specialized architecture, memory bank or queue need by previous approaches like [InstDisc](https://arxiv.org/pdf/1805.01978.pdf), [MoCo](https://arxiv.org/abs/1911.05722) or [PIRL](https://amitness.com/2020/03/illustrated-pirl/).

![Interpretation of Softmax Function](https://amitness.com/images/simclr-softmax-interpretation.png)

Then, the loss is calculated for a pair by taking the negative of the log of the above calculation. This formulation is the Noise Contrastive Estimation(NCE) Loss.

l(i,j)=−logexp(si,j)∑2Nk=1l[k!=i]exp(si,k)l(i,j)=−logexp(si,j)∑k=12Nl[k!=i]exp(si,k)

![Calculation of Loss from softmax](https://amitness.com/images/simclr-softmax-loss.png)

We calculate the loss for the same pair a second time as well where the positions of the images are interchanged.

![Calculation of loss for exchanged pairs of images](https://amitness.com/images/simclr-softmax-loss-inverted.png)

Finally, we compute loss over all the pairs in the batch of size N=2 and take an average.

L=12NN∑k=1[l(2k−1,2k)+l(2k,2k−1)]L=12N∑k=1N[l(2k−1,2k)+l(2k,2k−1)]

![Total loss in SimCLR](https://amitness.com/images/simclr-total-loss.png)

Based on the loss, the encoder and projection head representations improves over time and the representations obtained place similar images closer in the space.

## Downstream Tasks[Permalink](https://amitness.com/2020/03/illustrated-simclr/#downstream-tasks)

Once the SimCLR model is trained on the contrastive learning task, it can be used for transfer learning. For this, the representations from the encoder are used instead of representations obtained from the projection head. These representations can be used for downstream tasks like ImageNet Classification.

![Using SimCLR for downstream tasks](https://amitness.com/images/simclr-downstream.png)

## Objective Results[Permalink](https://amitness.com/2020/03/illustrated-simclr/#objective-results)

SimCLR outperformed previous self-supervised methods on ImageNet. The below image shows the top-1 accuracy of linear classifiers trained on representations learned with different self-supervised methods on ImageNet. The gray cross is supervised ResNet50 and SimCLR is shown in bold.

![Performance of SimCLR on ImageNet](https://amitness.com/images/simclr-performance.png)

Source: [SimCLR paper](https://arxiv.org/abs/2002.05709)

- On ImageNet [ILSVRC-2012](http://image-net.org/challenges/LSVRC/2012/), it achieves 76.5% top-1 accuracy which is 7% improvement over previous SOTA self-supervised method [Contrastive Predictive Coding](https://arxiv.org/abs/1905.09272) and on-par with supervised ResNet50.
- When trained on 1% of labels, it achieves 85.8% top-5 accuracy outperforming AlexNet with 100x fewer labels

## SimCLR Code[Permalink](https://amitness.com/2020/03/illustrated-simclr/#simclr-code)

The official implementation of SimCLR in Tensorflow by the paper authors is available on [GitHub](https://github.com/google-research/simclr). They also provide [pretrained models](https://github.com/google-research/simclr#pre-trained-models-for-simclrv1) for 1x, 2x, and 3x variants of the ResNet50 architectures using Tensorflow Hub.

There are various unofficial SimCLR PyTorch implementations available that have been tested on small datasets like [CIFAR-10](https://github.com/leftthomas/SimCLR) and [STL-10](https://github.com/Spijkervet/SimCLR).

## Conclusion[Permalink](https://amitness.com/2020/03/illustrated-simclr/#conclusion)

Thus, SimCLR provides a strong framework for doing further research in this direction and improve the state of self-supervised learning for Computer Vision.

## Citation Info (BibTex)[Permalink](https://amitness.com/2020/03/illustrated-simclr/#citation-info-bibtex)

If you found this blog post useful, please consider citing it as:

```
@misc{chaudhary2020simclr,
  title   = {The Illustrated SimCLR Framework},
  author  = {Amit Chaudhary},
  year    = 2020,
  note    = {\url{https://amitness.com/2020/03/illustrated-simclr}}
}
```

## References[Permalink](https://amitness.com/2020/03/illustrated-simclr/#references)

- [“A Simple Framework for Contrastive Learning of Visual Representations”](https://arxiv.org/abs/2002.05709)
- [“On Calibration of Modern Neural Networks”](https://arxiv.org/pdf/1706.04599.pdf)
- [“Distilling the Knowledge in a Neural Network”](https://arxiv.org/pdf/1503.02531.pdf)
- [“SimCLR Slides, Google Brain Team”](https://docs.google.com/presentation/d/1ccddJFD_j3p3h0TCqSV9ajSi2y1yOfh0-lJoK29ircs/edit#slide=id.g8c1b8d6efd_0_1)

