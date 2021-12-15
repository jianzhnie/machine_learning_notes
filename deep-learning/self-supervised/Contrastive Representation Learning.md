# Contrastive Representation Learning

> The main idea of contrastive learning is to learn representations such that similar samples stay close to each other, while dissimilar ones are far apart. Contrastive learning can be applied to both supervised and unsupervised data and has been shown to achieve good performance on a variety of vision and language tasks.

The goal of contrastive representation learning is to learn such an embedding space in which similar sample pairs stay close to each other while dissimilar ones are far apart. Contrastive learning can be applied to both supervised and unsupervised settings. When working with unsupervised data, contrastive learning is one of the most powerful approaches in [self-supervised learning]()



[toc]

## Contrastive Training Objectives

In early versions of loss functions for contrastive learning, only one positive and one negative sample are involved. The trend in recent training objectives is to include multiple positive and negative pairs in one batch.

### Contrastive Loss

**Contrastive loss** ([Chopra et al. 2005](http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf)) is one of the earliest training objectives used for deep metric learning in a contrastive fashion.

Given a list of input samples ${x_i}$, each has a corresponding label $y_i \in \{1, \dots, L\}$among $L$ classes. We would like to learn a function $f_\theta(.): \mathcal{X}\to\mathbb{R}^d$ that encodes $x_i$ into an embedding vector such that examples from the same class have similar embeddings and samples from different classes have very different ones. Thus, contrastive loss takes a pair of inputs $(x_i,x_j)$ and minimizes the embedding distance when they are from the same class but maximizes the distance otherwise.
$$
\mathcal{L}_\text{cont}(\mathbf{x}_i, \mathbf{x}_j, \theta) = \mathbb{1}[y_i=y_j] \| f_\theta(\mathbf{x}_i) - f_\theta(\mathbf{x}_j) \|^2_2 + \mathbb{1}[y_i\neq y_j]\max(0, \epsilon - \|f_\theta(\mathbf{x}_i) - f_\theta(\mathbf{x}_j)\|_2)^2
$$
where $ϵ$ is a hyperparameter, defining the lower bound distance between samples of different classes.

### Triplet Loss

**Triplet loss** was originally proposed in the FaceNet ([Schroff et al. 2015](https://arxiv.org/abs/1503.03832)) paper and was used to learn face recognition of the same person at different poses and angles.

![Triplet loss](https://lilianweng.github.io/lil-log/assets/images/triplet-loss.png)

Fig. 1. Illustration of triplet loss given one positive and one negative per anchor. (Image source: [Schroff et al. 2015](https://arxiv.org/abs/1503.03832))

Given one anchor input $x$ , we select one positive sample $x^+$ and one negative $x^−$, meaning that $x^+$and $x$ belong to the same class and $x^−$ is sampled from another different class. Triplet loss learns to minimize the distance between the anchor $x$ and positive $x+$ and maximize the distance between the anchor $x$ and negative $x−$ at the same time with the following equation:

$$
\mathcal{L}_\text{triplet}(\mathbf{x}, \mathbf{x}^+, \mathbf{x}^-) = \sum_{\mathbf{x} \in \mathcal{X}} \max\big( 0, \|f(\mathbf{x}) - f(\mathbf{x}^+)\|^2_2 - \|f(\mathbf{x}) - f(\mathbf{x}^-)\|^2_2 + \epsilon \big)
$$
where the margin parameter $ϵ$ is configured as the minimum offset between distances of similar vs dissimilar pairs.

It is crucial to select challenging $x−$ to truly improve the model.

### Lifted Structured Loss

**Lifted Structured Loss** ([Song et al. 2015](https://arxiv.org/abs/1511.06452)) utilizes all the pairwise edges within one training batch for better computational efficiency.

![Lifted structured loss](https://lilianweng.github.io/lil-log/assets/images/lifted-structured-loss.png)

Fig. 2. Illustration compares contrastive loss, triplet loss and lifted structured loss. Red and blue edges connect similar and dissimilar sample pairs respectively. (Image source: [Song et al. 2015](https://arxiv.org/abs/1511.06452))

Let $D_{ij} = \| f(\mathbf{x}_i) - f(\mathbf{x}_j) \|_2$ , a structured loss function is defined as

$$
\begin{aligned}
\mathcal{L}_\text{struct} &= \frac{1}{2\vert \mathcal{P} \vert} \sum_{(i,j) \in \mathcal{P}} \max(0, \mathcal{L}_\text{struct}^{(ij)})^2 \\
\text{where } \mathcal{L}_\text{struct}^{(ij)} &= D_{ij} + \color{red}{\max \big( \max_{(i,k)\in \mathcal{N}} \epsilon - D_{ik}, \max_{(j,l)\in \mathcal{N}} \epsilon - D_{jl} \big)}
\end{aligned}
$$
where $P$ contains the set of positive pairs and $N$ is the set of negative pairs. Note that the dense pairwise squared distance matrix can be easily computed per training batch.

The red part in $\mathcal{L}_\text{struct}^{(ij)}$ is used for mining hard negatives. However, it is not smooth and may cause the convergence to a bad local optimum in practice. Thus, it is relaxed to be:

$$
\mathcal{L}_\text{struct}^{(ij)} = D_{ij} + \log \Big( \sum_{(i,k)\in\mathcal{N}} \exp(\epsilon - D_{ik}) + \sum_{(j,l)\in\mathcal{N}} \exp(\epsilon - D_{jl}) \Big)
$$
In the paper, they also proposed to enhance the quality of negative samples in each batch by actively incorporating difficult negative samples given a few random positive pairs.

### N-pair Loss

**Multi-Class N-pair loss** ([Sohn 2016](https://papers.nips.cc/paper/2016/hash/6b180037abbebea991d8b1232f8a8ca9-Abstract.html)) generalizes triplet loss to include comparison with multiple negative samples.

Given a $(N+1)$-tuplet of training samples, $\{ \mathbf{x}, \mathbf{x}^+, \mathbf{x}^-_1, \dots, \mathbf{x}^-_{N-1} \}$, including one positive and $N−1$ negative ones, $N-pair $ loss is defined as:

$$
\begin{aligned}
\mathcal{L}_\text{N-pair}(\mathbf{x}, \mathbf{x}^+, \{\mathbf{x}^-_i\}^{N-1}_{i=1})
&= \log\big(1 + \sum_{i=1}^{N-1} \exp(f(\mathbf{x})^\top f(\mathbf{x}^-_i) - f(\mathbf{x})^\top f(\mathbf{x}^+))\big) \\
&= -\log\frac{\exp(f(\mathbf{x})^\top f(\mathbf{x}^+))}{\exp(f(\mathbf{x})^\top f(\mathbf{x}^+)) + \sum_{i=1}^{N-1} \exp(f(\mathbf{x})^\top f(\mathbf{x}^-_i))}
\end{aligned}
$$


If we only sample one negative sample per class, it is equivalent to the `softmax` loss for multi-class classification.

### NCE

**Noise Contrastive Estimation**, short for **NCE**, is a method for estimating parameters of a statistical model, proposed by [Gutmann & Hyvarinen](http://proceedings.mlr.press/v9/gutmann10a.html) in 2010. The idea is to run logistic regression to tell apart the target data from noise. Read more on how NCE is used for learning word embedding [here](https://lilianweng.github.io/lil-log/2017/10/15/learning-word-embedding.html#noise-contrastive-estimation-nce).

Let $x$ be the target sample $∼P(x|C=1;θ)=p_θ(x) $and $x$ be the noise sample $∼P(x~|C=0)=q(x~)$. Note that the logistic regression models the `logit` (i.e. log-odds) and in this case we would like to model the `logit` of a sample $u$ from the target data distribution instead of the noise distribution:
$$
\ell_\theta(\mathbf{u}) = \log \frac{p_\theta(\mathbf{u})}{q(\mathbf{u})} = \log p_\theta(\mathbf{u}) - \log q(\mathbf{u})
$$
After converting `logits` into probabilities with sigmoid $σ(.)$ we can apply cross entropy loss:

$$
\begin{aligned}
\mathcal{L}_\text{NCE} &= - \frac{1}{N} \sum_{i=1}^N \big[ \log \sigma (\ell_\theta(\mathbf{x}_i)) + \log (1 - \sigma (\ell_\theta(\tilde{\mathbf{x}}_i))) \big] \\
\text{ where }\sigma(\ell) &= \frac{1}{1 + \exp(-\ell)} = \frac{p_\theta}{p_\theta + q}
\end{aligned}
$$


Here I listed the original form of NCE loss which works with only one positive and one noise sample. In many follow-up works, contrastive loss incorporating multiple negative samples is also broadly referred to as NCE.

### InfoNCE

The **InfoNCE loss** in CPC ([Contrastive Predictive Coding](https://lilianweng.github.io/lil-log/2019/11/10/self-supervised-learning.html#contrastive-predictive-coding); [van den Oord, et al. 2018](https://arxiv.org/abs/1807.03748)), inspired by [NCE](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#NCE), uses categorical cross-entropy loss to identify the positive sample amongst a set of unrelated noise samples.

Given a context vector $c$, the positive sample should be drawn from the conditional distribution $p(x|c)$, while $N−1$ negative samples are drawn from the proposal distribution $p(x)$, independent from the context $c$. For brevity, let us label all the samples as $X=\{ \mathbf{x}_i \}^N_{i=1}$ among which only one of them $x_{pos}$ is a positive sample. The probability of we detecting the positive sample correctly is:

$$
p(C=\texttt{pos} \vert X, \mathbf{c})
= \frac{p(x_\texttt{pos} \vert \mathbf{c}) \prod_{i=1,\dots,N; i \neq \texttt{pos}} p(\mathbf{x}_i)}{\sum_{j=1}^N \big[ p(\mathbf{x}_j \vert \mathbf{c}) \prod_{i=1,\dots,N; i \neq j} p(\mathbf{x}_i) \big]}
= \frac{ \frac{p(\mathbf{x}_\texttt{pos}\vert c)}{p(\mathbf{x}_\texttt{pos})} }{ \sum_{j=1}^N \frac{p(\mathbf{x}_j\vert \mathbf{c})}{p(\mathbf{x}_j)} }
= \frac{f(\mathbf{x}_\texttt{pos}, \mathbf{c})}{ \sum_{j=1}^N f(\mathbf{x}_j, \mathbf{c}) }
$$


where the scoring function is $f(\mathbf{x}, \mathbf{c}) \propto \frac{p(\mathbf{x}\vert\mathbf{c})}{p(\mathbf{x})}$

The InfoNCE loss optimizes the negative log probability of classifying the positive sample correctly:
$$
\mathcal{L}_\text{InfoNCE} = - \mathbb{E} \Big[\log \frac{f(\mathbf{x}, \mathbf{c})}{\sum_{\mathbf{x}' \in X} f(\mathbf{x}', \mathbf{c})} \Big]
$$
The fact that$ f(x,c)$ estimates the density ratio $p(x|c)p(x)$ has a connection with mutual information optimization. To maximize the the mutual information between input $x$ and context vector $c$, we have:

$$
I(\mathbf{x}; \mathbf{c}) = \sum_{\mathbf{x}, \mathbf{c}} p(\mathbf{x}, \mathbf{c}) \log\frac{p(\mathbf{x}, \mathbf{c})}{p(\mathbf{x})p(\mathbf{c})} = \sum_{\mathbf{x}, \mathbf{c}} p(\mathbf{x}, \mathbf{c})\log\color{blue}{\frac{p(\mathbf{x}|\mathbf{c})}{p(\mathbf{x})}}
$$


where the logarithmic term in blue is estimated by $ f.$

For sequence prediction tasks, rather than modeling the future observations $p_k(\mathbf{x}_{t+k} \vert \mathbf{c}_t)$ directly (which could be fairly expensive), CPC models a density function to preserve the mutual information between $x_{t+k}$ and $ct$:
$$
f_k(\mathbf{x}_{t+k}, \mathbf{c}_t) = \exp(\mathbf{z}_{t+k}^\top \mathbf{W}_k \mathbf{c}_t) \propto \frac{p(\mathbf{x}_{t+k}\vert\mathbf{c}_t)}{p(\mathbf{x}_{t+k})}
$$


where $z_{t+k}$ is the encoded input and $W_k$ is a trainable weight matrix.

### Soft-Nearest Neighbors Loss

**Soft-Nearest Neighbors Loss** ([Salakhutdinov & Hinton 2007](http://proceedings.mlr.press/v2/salakhutdinov07a.html), [Frosst et al. 2019](https://arxiv.org/abs/1902.01889)) extends it to include multiple positive samples.

Given a batch of samples, $p_k(\mathbf{x}_{t+k} \vert \mathbf{c}_t)$ where $y_i$ is the class label of $x_i$ and a function $f(.,.)$ for measuring similarity between two inputs, the soft nearest neighbor loss at temperature $τ$ is defined as:
$$
\mathcal{L}_\text{snn} = -\frac{1}{B}\sum_{i=1}^B \log \frac{\sum_{i\neq j, y_i = y_j, j=1,\dots,B} \exp(- f(\mathbf{x}_i, \mathbf{x}_j) / \tau)}{\sum_{i\neq k, k=1,\dots,B} \exp(- f(\mathbf{x}_i, \mathbf{x}_k) /\tau)}
$$


The temperature $τ$ is used for tuning how concentrated the features are in the representation space. For example, when at low temperature, the loss is dominated by the small distances and widely separated representations cannot contribute much and become irrelevant.

### Common Setup

We can loosen the definition of “classes” and “labels” in soft nearest-neighbor loss to create positive and negative sample pairs out of unsupervised data by, for example, applying data augmentation to create noise versions of original samples.

Most recent studies follow the following definition of contrastive learning objective to incorporate multiple positive and negative samples. According to the setup in ([Wang & Isola 2020](https://arxiv.org/abs/2005.10242)), let $p_{data}(.)$ be the data distribution over $R^n$ and $p_{pos}(.,.) $be the distribution of positive pairs over $R^{n×n}$ These two distributions should satisfy:

- Symmetry: $\forall \mathbf{x}, \mathbf{x}^+, p_\texttt{pos}(\mathbf{x}, \mathbf{x}^+) = p_\texttt{pos}(\mathbf{x}^+, \mathbf{x})$
- Matching marginal: $\forall \mathbf{x}, \int p_\texttt{pos}(\mathbf{x}, \mathbf{x}^+) d\mathbf{x}^+ = p_\texttt{data}(\mathbf{x})$
- To learn an encoder $f(x)$to learn a *L2-normalized feature vector*, the contrastive learning objective is:

$$
\begin{aligned}
\mathcal{L}_\text{contrastive}
&= \mathbb{E}_{(\mathbf{x},\mathbf{x}^+)\sim p_\texttt{pos}, \{\mathbf{x}^-_i\}^M_{i=1} \overset{\text{i.i.d}}{\sim} p_\texttt{data} } \Big[ -\log\frac{\exp(f(\mathbf{x})^\top f(\mathbf{x}^+) / \tau)}{ \exp(f(\mathbf{x})^\top f(\mathbf{x}^+) / \tau) + \sum_{i=1}^M \exp(f(\mathbf{x})^\top f(\mathbf{x}_i^-) / \tau)} \Big] & \\
&\approx \mathbb{E}_{(\mathbf{x},\mathbf{x}^+)\sim p_\texttt{pos}, \{\mathbf{x}^-_i\}^M_{i=1} \overset{\text{i.i.d}}{\sim} p_\texttt{data} }\Big[ - f(\mathbf{x})^\top f(\mathbf{x}^+) / \tau + \log\big(\sum_{i=1}^M \exp(f(\mathbf{x})^\top f(\mathbf{x}_i^-) / \tau)\big) \Big] & \scriptstyle{\text{; Assuming infinite negatives}} \\
&= -\frac{1}{\tau}\mathbb{E}_{(\mathbf{x},\mathbf{x}^+)\sim p_\texttt{pos}}f(\mathbf{x})^\top f(\mathbf{x}^+) + \mathbb{E}_{ \mathbf{x} \sim p_\texttt{data}} \Big[ \log \mathbb{E}_{\mathbf{x}^- \sim p_\texttt{data}} \big[ \sum_{i=1}^M \exp(f(\mathbf{x})^\top f(\mathbf{x}_i^-) / \tau)\big] \Big] &
\end{aligned}
$$

## Key Ingredients

### Heavy Data Augmentation

Given a training sample, data augmentation techniques are needed for creating noise versions of itself to feed into the loss as positive samples. Proper data augmentation setup is critical for learning good and generalizable embedding features. It introduces the non-essential variations into examples without modifying semantic meanings and thus encourages the model to learn the essential part of the representation. For example, experiments in [SimCLR](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#simclr) showed that the composition of random cropping and random color distortion is crucial for good performance on learning visual representation of images.

### Large Batch Size

Using a large batch size during training is another key ingredient in the success of many contrastive learning methods (e.g. [SimCLR](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#simclr), [CLIP](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#clip)), especially when it relies on in-batch negatives. Only when the batch size is big enough, the loss function can cover a diverse enough collection of negative samples, challenging enough for the model to learn meaningful representation to distinguish different examples.

### Hard Negative Mining

Hard negative samples should have different labels from the anchor sample, but have embedding features very close to the anchor embedding. With access to ground truth labels in supervised datasets, it is easy to identify task-specific hard negatives. For example when learning sentence embedding, we can treat sentence pairs labeled as “contradiction” in `NLI` datasets as hard negative pairs (e.g. [SimCSE](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#dropout-and-cutoff), or use top incorrect candidates returned by `BM25` with most keywords matched as hard negative samples ([DPR](https://lilianweng.github.io/lil-log/2020/10/29/open-domain-question-answering.html#DPR); [Karpukhin et al., 2020](https://arxiv.org/abs/2004.04906)).

However, it becomes tricky to do hard negative mining when we want to remain unsupervised. Increasing training batch size or [memory bank](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#memory-bank) size implicitly introduces more hard negative samples, but it leads to a heavy burden of large memory usage as a side effect.

[Chuang et al. (2020)](https://arxiv.org/abs/2007.00224) studied the sampling bias in contrastive learning and proposed debiased loss. In the unsupervised setting, since we do not know the ground truth labels, we may accidentally sample false negative samples. Sampling bias can lead to significant performance drop.

![Sampling bias](https://lilianweng.github.io/lil-log/assets/images/contrastive-sampling-bias.png)

*Fig. 3. Sampling bias which refers to false negative samples in contrastive learning can lead to a big performance drop. (Image source: [Chuang et al., 2020](https://arxiv.org/abs/2007.00224))*

Let us assume the probability of anchor class $c$ is uniform $ρ(c)=η+$ and the probability of observing a different class is $η−=1−η+$

- The probability of observing a positive example for $x$ is $p^+_x(\mathbf{x}')=p(\mathbf{x}'\vert \mathbf{h}_{x'}=\mathbf{h}_x)$;
- The probability of getting a negative sample for $x$ is $p^-_x(\mathbf{x}')=p(\mathbf{x}'\vert \mathbf{h}_{x'}\neq\mathbf{h}_x)$.

When we are sampling $x−$ , we cannot access the true $p^{−x}(x−)$and thus $x−$ may be sampled from the (undesired) anchor class cc with probability η+η+. The actual sampling data distribution becomes:

$$
p(\mathbf{x}') = \eta^+ p^+_x(\mathbf{x}') + \eta^- p_x^-(\mathbf{x}')
$$

Thus we can use $p^-_x(\mathbf{x}') = (p(\mathbf{x}') - \eta^+ p^+_x(\mathbf{x}'))/\eta^-$ for sampling $x−$ to debias the loss. With $N$ samples $\{\mathbf{u}_i\}^N_{i=1}$ from $p$ and $M$ samples $\{ \mathbf{v}_i \}_{i=1}^M$ from $p^+_{x}$ , we can estimate the expectation of the second term $\mathbb{E}_{\mathbf{x}^-\sim p^-_x}[\exp(f(\mathbf{x})^\top f(\mathbf{x}^-))]$ in the denominator of contrastive learning loss:
$$
g(\mathbf{x}, \{\mathbf{u}_i\}^N_{i=1}, \{\mathbf{v}_i\}_{i=1}^M) = \max\Big\{ \frac{1}{\eta^-}\Big( \frac{1}{N}\sum_{i=1}^N \exp(f(\mathbf{x})^\top f(\mathbf{u}_i)) - \frac{\eta^+}{M}\sum_{i=1}^M \exp(f(\mathbf{x})^\top f(\mathbf{v}_i)) \Big), \exp(-1/\tau) \Big\
$$
where $τ$ is the temperature and $exp(−1/τ)$is the theoretical lower bound of $\mathbb{E}_{\mathbf{x}^-\sim p^-_x}[\exp(f(\mathbf{x})^\top f(\mathbf{x}^-))]$

The final debiased contrastive loss looks like:
$$
\mathcal{L}^{N,M}_\text{debias}(f) = \mathbb{E}_{\mathbf{x},\{\mathbf{u}_i\}^N_{i=1}\sim p;\;\mathbf{x}^+, \{\mathbf{v}_i\}_{i=1}^M\sim p^+} \Big[ -\log\frac{\exp(f(\mathbf{x})^\top f(\mathbf{x}^+)}{\exp(f(\mathbf{x})^\top f(\mathbf{x}^+) + N g(x,\{\mathbf{u}_i\}^N_{i=1}, \{\mathbf{v}_i\}_{i=1}^M)} \Big]
$$


![Debiased t-SNE vis](https://lilianweng.github.io/lil-log/assets/images/contrastive-debias-t-SNE.png)

*Fig. 4. t-SNE visualization of learned representation with debiased contrastive learning. (Image source: [Chuang et al., 2020](https://arxiv.org/abs/2007.00224))*

Following the above annotation, [Robinson et al. (2021)](https://arxiv.org/abs/2010.04592) modified the sampling probabilities to target at hard negatives by up-weighting the probability $p^{−x}(x′)$ to be proportional to its similarity to the anchor sample. The new sampling probability $q_{β}(x−)$ is:
$$
q_\beta(\mathbf{x}^-) \propto \exp(\beta f(\mathbf{x})^\top f(\mathbf{x}^-)) \cdot p(\mathbf{x}^-)
$$
where $β$ is a hyperparameter to tune.

We can estimate the second term in the denominator $\mathbb{E}_{\mathbf{x}^- \sim q_\beta} [\exp(f(\mathbf{x})^\top f(\mathbf{x}^-))] $using importance sampling where both the partition functions $Z_\beta, Z^+_\beta$ can be estimated empirically.

$$
\begin{aligned}
\mathbb{E}_{\mathbf{u} \sim q_\beta} [\exp(f(\mathbf{x})^\top f(\mathbf{u}))] &= \mathbb{E}_{\mathbf{u} \sim p} [\frac{q_\beta}{p}\exp(f(\mathbf{x})^\top f(\mathbf{u}))] = \mathbb{E}_{\mathbf{u} \sim p} [\frac{1}{Z_\beta}\exp((\beta + 1)f(\mathbf{x})^\top f(\mathbf{u}))] \\
\mathbb{E}_{\mathbf{v} \sim q^+_\beta} [\exp(f(\mathbf{x})^\top f(\mathbf{v}))] &= \mathbb{E}_{\mathbf{v} \sim p^+} [\frac{q^+_\beta}{p}\exp(f(\mathbf{x})^\top f(\mathbf{v}))] = \mathbb{E}_{\mathbf{v} \sim p} [\frac{1}{Z^+_\beta}\exp((\beta + 1)f(\mathbf{x})^\top f(\mathbf{v}))]
\end{aligned}
$$


![Pseudo code](https://lilianweng.github.io/lil-log/assets/images/contrastive-hard-negatives-code.png)

*Fig. 5. Pseudo code for computing NCE loss, debiased contrastive loss, and hard negative sample objective when setting M=1. (Image source: [Robinson et al., 2021](https://arxiv.org/abs/2010.04592) )*

## Vision: Image Embedding

### Image Augmentations

Most approaches for contrastive representation learning in the vision domain rely on creating a noise version of a sample by applying a sequence of data augmentation techniques. The augmentation should significantly change its visual appearance but keep the semantic meaning unchanged.

#### Basic Image Augmentation

There are many ways to modify an image while retaining its semantic meaning. We can use any one of the following augmentation or a composition of multiple operations.

- Random cropping and then resize back to the original size.
- Random color distortions
- Random Gaussian blur
- Random color jittering
- Random horizontal flip
- Random grayscale conversion
- Multi-crop augmentation: Use two standard resolution crops and sample a set of additional low resolution crops that cover only small parts of the image. Using low resolution crops reduces the compute cost. ([SwAV](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#swav))
- And many more …

#### Augmentation Strategies

Many frameworks are designed for learning good data augmentation strategies (i.e. a composition of multiple transforms). Here are a few common ones.

- [AutoAugment](https://lilianweng.github.io/lil-log/2019/05/05/domain-randomization.html#AutoAugment) ([Cubuk, et al. 2018](https://arxiv.org/abs/1805.09501)): Inspired by [NAS](https://lilianweng.github.io/lil-log/2020/08/06/neural-architecture-search.html), AutoAugment frames the problem of learning best data augmentation operations (i.e. shearing, rotation, invert, etc.) for image classification as an RL problem and looks for the combination that leads to the highest accuracy on the evaluation set.
- RandAugment ([Cubuk et al., 2019](https://arxiv.org/abs/1909.13719)): RandAugment greatly reduces the search space of AutoAugment by controlling the magnitudes of different transformation operations with a single magnitude parameter.
- PBA (Population based augmentation; [Ho et al., 2019](https://arxiv.org/abs/1905.05393)): PBA combined PBT ([Jaderberg et al, 2017](https://arxiv.org/abs/1711.09846)) with AutoAugment, using the evolutionary algorithm to train a population of children models in parallel to evolve the best augmentation strategies.
- UDA (Unsupervised Data Augmentation; [Xie et al., 2019](https://arxiv.org/abs/1904.12848)): Among a set of possible augmentation strategies, UDA selects those to minimize the KL divergence between the predicted distribution over an unlabeled example and its unlabeled augmented version.

#### Image Mixture

Image mixture methods can construct new training examples from existing data points.

- Mixup ([Zhang et al., 2018](https://arxiv.org/abs/1710.09412)): It runs global-level mixture by creating a weighted pixel-wise combination of two existing images $I_1$ and $I_2$: $I_\text{mixup} \gets \alpha I_1 + (1-\alpha) I_2$ and $α∈[0,1]$.
- Cutmix ([Yun et al., 2019](https://arxiv.org/abs/1905.04899)): Cutmix does region-level mixture by generating a new example by combining a local region of one image with the rest of the other image. $I_\text{cutmix} \gets \mathbf{M}_b \odot I_1 + (1-\mathbf{M}_b) \odot I_2$, where $\mathbf{M}_b \in \{0, 1\}^I$ is a binary mask and ⊙⊙ is element-wise multiplication. It is equivalent to filling the cutout ([DeVries & Taylor 2017](https://arxiv.org/abs/1708.04552)) region with the same region from another image.
- MoCHi (“Mixing of Contrastive Hard Negatives”; [Kalantidis et al. 2020](https://arxiv.org/abs/2010.01028)): Given a query $q$, MoCHi maintains a queue of $K$ negative features  $Q=\{\mathbf{n}_1, \dots, \mathbf{n}_K \}$and sorts these negative features by similarity to the query, $q^⊤n$, in descending order. The first $N$ items in the queue are considered as the hardest negatives, $Q_N$. Then synthetic hard examples can be generated by $\mathbf{h} = \tilde{\mathbf{h}} / \|\tilde{\mathbf{h}}\|$ where  $\tilde{\mathbf{h}} = \alpha\mathbf{n}_i + (1-\alpha) \mathbf{n}_j$and $α∈(0,1)$. Even harder examples can be created by mixing with the query feature,  $\mathbf{h}' = \tilde{\mathbf{h}'} / \|\tilde{\mathbf{h}'}\|_2$ where $\tilde{\mathbf{h}'} = \beta\mathbf{q} + (1-\beta) \mathbf{n}_j$ and $β∈(0,0.5)$.

### Parallel Augmentation

This category of approaches produce two noise versions of one anchor image and aim to learn representation such that these two augmented samples share the same embedding.

#### SimCLR

**SimCLR** ([Chen et al, 2020](https://arxiv.org/abs/2002.05709)) proposed a simple framework for contrastive learning of visual representations. It learns representations for visual inputs by maximizing agreement between differently augmented views of the same sample via a contrastive loss in the latent space.

![SimCLR](https://lilianweng.github.io/lil-log/assets/images/SimCLR.png)

*Fig. 6. A simple framework for contrastive learning of visual representations. (Image source: [Chen et al, 2020](https://arxiv.org/abs/2002.05709))*

- 1) Randomly sample a minibatch of $N$ samples and each sample is applied with two different data augmentation operations, resulting in $2N$ augmented samples in total.

$\tilde{\mathbf{x}}_i = t(\mathbf{x}),\quad\tilde{\mathbf{x}}_j = t'(\mathbf{x}),\quad t, t' \sim \mathcal{T}$

where two separate data augmentation operators, $t$ and $t′$, are sampled from the same family of augmentations $T$. Data augmentation includes random crop, resize with random flip, color distortions, and Gaussian blur.

- 2) Given one positive pair, other $2(N−1)$ data points are treated as negative samples. The representation is produced by a base encoder $f(.)$:

$\mathbf{h}_i = f(\tilde{\mathbf{x}}_i),\quad \mathbf{h}_j = f(\tilde{\mathbf{x}}_j)$

- 3) The contrastive learning loss is defined using cosine similarity $sim(.,.)$. Note that the loss operates on an extra projection layer of the representation $g(.)$ rather than on the representation space directly. But only the representation $h$ is used for downstream tasks.
  $$
  \begin{aligned}
  \mathbf{z}_i &= g(\mathbf{h}_i),\quad
  \mathbf{z}_j = g(\mathbf{h}_j) \\
  \mathcal{L}_\text{SimCLR}^{(i,j)} &= - \log\frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_k) / \tau)}
  \end{aligned}
  $$


where $\mathbb{1}_{[k \neq i]}$ is an indicator function: 1 if $k≠i$ 0 otherwise.

SimCLR needs a large batch size to incorporate enough negative samples to achieve good performance.

![SimCLR Algorithm](https://lilianweng.github.io/lil-log/assets/images/SimCLR-algo.png)

*Fig. 7. The algorithm for SimCLR. (Image source: [Chen et al, 2020](https://arxiv.org/abs/2002.05709)).*

#### Barlow Twins

**Barlow Twins** ([Zbontar et al. 2021](https://arxiv.org/abs/2103.03230)) feeds two distorted versions of samples into the same network to extract features and learns to make the *cross-correlation matrix* between these two groups of output features close to the identity. The goal is to keep the representation vectors of different distorted versions of one sample similar, while minimizing the redundancy between these vectors.

![Barlow twins](https://lilianweng.github.io/lil-log/assets/images/barlow-twins.png)

*Fig. 8. Illustration of Barlow Twins learning pipeline. (Image source: [Zbontar et al. 2021](https://arxiv.org/abs/2103.03230)).*

Let $C$ be a cross-correlation matrix computed between outputs from two identical networks along the batch dimension. $C$ is a square matrix with the size same as the feature network’s output dimensionality. Each entry in the matrix $C_{ij}$ is the cosine similarity between network output vector dimension at index $i,j$ and batch index $b, \mathbf{z}_{b,i}^A , \mathbf{z}_{b,j}^B$, with a value between -1 (i.e. perfect anti-correlation) and 1 (i.e. perfect correlation).

$$
\begin{aligned}
\mathcal{L}_\text{BT} &= \underbrace{\sum_i (1-\mathcal{C}_{ii})^2}_\text{invariance term} + \lambda \underbrace{\sum_i\sum_{i\neq j} \mathcal{C}_{ij}^2}_\text{redundancy reduction term} \\ \text{where } \mathcal{C}_{ij} &= \frac{\sum_b \mathbf{z}^A_{b,i} \mathbf{z}^B_{b,j}}{\sqrt{\sum_b (\mathbf{z}^A_{b,i})^2}\sqrt{\sum_b (\mathbf{z}^B_{b,j})^2}}
\end{aligned}
$$


Barlow Twins is competitive with SOTA methods for self-supervised learning. It naturally avoids trivial constants (i.e. collapsed representations), and is robust to different training bat\begin{aligned}
\mathcal{L}_\text{BT} &= \underbrace{\sum_i (1-\mathcal{C}_{ii})^2}_\text{invariance term} + \lambda \underbrace{\sum_i\sum_{i\neq j} \mathcal{C}_{ij}^2}_\text{redundancy reduction term} \\ \text{where } \mathcal{C}_{ij} &= \frac{\sum_b \mathbf{z}^A_{b,i} \mathbf{z}^B_{b,j}}{\sqrt{\sum_b (\mathbf{z}^A_{b,i})^2}\sqrt{\sum_b (\mathbf{z}^B_{b,j})^2}}
\end{aligned}ch sizes.

![Barlow twins algo](https://lilianweng.github.io/lil-log/assets/images/barlow-twins-algo.png)

*Fig. 9. Algorithm of Barlow Twins in Pytorch style pseudo code. (Image source: [Zbontar et al. 2021](https://arxiv.org/abs/2103.03230)).*

#### BYOL

Different from the above approaches, interestingly, **BYOL** (Bootstrap Your Own Latent; [Grill, et al 2020](https://arxiv.org/abs/2006.07733)) claims to achieve a new state-of-the-art results *without using egative samples*. It relies on two neural networks, referred to as *online* and *target* networks that interact and learn from each other. The target network (parameterized by $ξ$) has the same architecture as the online one (parameterized by $θ$), but with polyak averaged weights, $\xi \leftarrow \tau \xi + (1-\tau) \theta$.

The goal is to learn a presentation $y$ that can be used in downstream tasks. The online network parameterized by $θ$ contains:

- An encoder $fθ$;
- A projector $gθ$;
- A predictor $qθ$.

The target network has the same network architecture, but with different parameter $ξ$, updated by polyak averaging $θ: ξ←τξ+(1−τ)θ$.

![BYOL](https://lilianweng.github.io/lil-log/assets/images/BYOL.png)

*Fig. 10. The model architecture of BYOL. After training, we only care about $fθ$  for producing representation, $y=f_θ(x)$, and everything else is discarded. $sg$ means stop gradient. (Image source: [Grill, et al 2020](https://arxiv.org/abs/2006.07733))*

Given an image xx, the BYOL loss is constructed as follows:

- Create two augmented views: $v=t(x);v′=t′(x)$ with augmentations sampled $t∼T,t′∼T′$;
- Then they are encoded into representations, $yθ=fθ(v),y′=fξ(v′)$;
- Then they are projected into latent variables, $\mathbf{z}_\theta=g_\theta(\mathbf{y}_\theta), \mathbf{z}'=g_\xi(\mathbf{y}')$;
- The online network outputs a prediction $qθ(zθ)$;
- Both $qθ(zθ)$ and $z′$ are $L2-normalized$, giving us $\bar{q}_\theta(\mathbf{z}_\theta) = q_\theta(\mathbf{z}_\theta) / \| q_\theta(\mathbf{z}_\theta) \|$and $\bar{\mathbf{z}'} = \mathbf{z}' / \|\mathbf{z}'\|$;
- The loss $\mathcal{L}^\text{BYOL}_\theta$ is MSE between L2-normalized prediction $q¯θ(z)$ and $z′¯$;
- The other symmetric loss $\tilde{\mathcal{L}}^\text{BYOL}_\theta$ can be generated by switching $v′$ and $v$; that is, feeding $v′$ to online network and $v$ to target network.
- The final loss is $\mathcal{L}^\text{BYOL}_\theta + \tilde{\mathcal{L}}^\text{BYOL}_\theta$ and only parameters $θ$ are optimized.

Unlike most popular contrastive learning based approaches, BYOL does not use negative pairs. Most bootstrapping approaches rely on pseudo-labels or cluster indices, but BYOL directly boostrapps the latent representation.

It is quite interesting and surprising that *without* negative samples, BYOL still works well. Later I ran into this [post](https://untitled-ai.github.io/understanding-self-supervised-contrastive-learning.html) by Abe Fetterman & Josh Albrecht, they highlighted two surprising findings while they were trying to reproduce BYOL:

1. BYOL generally performs no better than random when *batch normalization is removed*.
2. The presence of batch normalization implicitly causes a form of contrastive learning. They believe that using negative samples is important for avoiding model collapse (i.e. what if you use all-zeros representation for every data point?). Batch normalization injects dependency on negative samples *inexplicitly* because no matter how similar a batch of inputs are, the values are re-distributed (spread out $∼N(0,10)$ and therefore batch normalization prevents model collapse. Strongly recommend you to read the [full article](https://untitled-ai.github.io/understanding-self-supervised-contrastive-learning.html) if you are working in this area.

### Memory Bank

Computing embeddings for a large number of negative samples in every batch is extremely expensive. One common approach is to store the representation in memory to trade off data staleness for cheaper compute.

#### Instance Discrimination with Memoy Bank

**Instance contrastive learning** ([Wu et al, 2018](https://arxiv.org/abs/1805.01978v1)) pushes the class-wise supervision to the extreme by considering each instance as *a distinct class of its own*. It implies that the number of “classes” will be the same as the number of samples in the training dataset. Hence, it is unfeasible to train a softmax layer with these many heads, but instead it can be approximated by [NCE](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#nce).

![Instance contrastive learning](https://lilianweng.github.io/lil-log/assets/images/instance-level-discrimination.png)

*Fig. 11. The training pipeline of instance-level contrastive learning. The learned embedding is L2-normalized. (Image source: [Wu et al, 2018](https://arxiv.org/abs/1805.01978v1))*

Let $v=f_θ(x)$ be an embedding function to learn and the vector is normalized to have $∥v∥=1$. A non-parametric classifier predicts the probability of a sample $v$ belonging to class ii with a temperature parameter $τ$:

$$
P(C=i\vert \mathbf{v}) = \frac{\exp(\mathbf{v}_i^\top \mathbf{v} / \tau)}{\sum_{j=1}^n \exp(\mathbf{v}_j^\top \mathbf{v} / \tau)}
$$


Instead of computing the representations for all the samples every time, they implement an **Memory Bank** for storing sample representation in the database from past iterations. Let $V={v_i}$ be the memory bank and $f_i=f_θ(x_i)$ be the feature generated by forwarding the network. We can use the representation from the memory bank $v_i$ instead of the feature forwarded from the network $f_i$ when comparing pairwise similarity.

The denominator theoretically requires access to the representations of all the samples, but that is too expensive in practice. Instead we can estimate it via Monte Carlo approximation using a random subset of MM indices $\{j_k\}_{k=1}^M$.


$$
P(i\vert \mathbf{v})
= \frac{\exp(\mathbf{v}^\top \mathbf{f}_i / \tau)}{\sum_{j=1}^N \exp(\mathbf{v}_j^\top \mathbf{f}_i / \tau)}
\simeq \frac{\exp(\mathbf{v}^\top \mathbf{f}_i / \tau)}{\frac{N}{M} \sum_{k=1}^M \exp(\mathbf{v}_{j_k}^\top \mathbf{f}_i / \tau)}
$$
Because there is only one instance per class, the training is unstable and fluctuates a lot. To improve the training smoothness, they introduced an extra term for positive samples in the loss function based on the [proximal optimization method](https://web.stanford.edu/~boyd/papers/prox_algs.html). The final NCE loss objective looks like:

$$
\begin{aligned}
\mathcal{L}_\text{instance} &= - \mathbb{E}_{P_d}\big[\log h(i, \mathbf{v}^{(t-1)}_i) - \lambda \|\mathbf{v}^{(t)}_i - \mathbf{v}^{(t-1)}_i\|^2_2\big] - M\mathbb{E}_{P_n}\big[\log(1 - h(i, \mathbf{v}'^{(t-1)})\big] \\
h(i, \mathbf{v}) &= \frac{P(i\vert\mathbf{v})}{P(i\vert\mathbf{v}) + MP_n(i)} \text{ where the noise distribution is uniform }P_n = 1/N
\end{aligned}
$$
where ${v^{(t−1)}}$ are embeddings stored in the memory bank from the previous iteration. The difference between iterations  $\|\mathbf{v}^{(t)}_i - \mathbf{v}^{(t-1)}_i\|^2_2$will gradually vanish as the learned embedding converges.

#### MoCo & MoCo-V2

**Momentum Contrast** (**MoCo**; [He et al, 2019](https://arxiv.org/abs/1911.05722)) provides a framework of unsupervised learning visual representation as a *dynamic dictionary look-up*. The dictionary is structured as a large FIFO queue of encoded representations of data samples.

Given a query sample $x_q$, we get a query representation through an encoder $q=f_q(x_q)$. A list of key representations ${k1,k2,…}$ in the dictionary are encoded by a momentum encoder $k_i=f_k(x^k_i)$. Let’s assume among them there is a single *positive* key $k^+$ in the dictionary that matches $q$ . In the paper, they create $k^+$ using a noise copy of $x^q$ with different [augmentation](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#image-augmentations). Then the [InfoNCE](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#infonce) contrastive loss with temperature $τ$ is used over one positive and $N−1$ negative samples:
$$
\mathcal{L}_\text{MoCo} = - \log \frac{\exp(\mathbf{q} \cdot \mathbf{k}^+ / \tau)}{\sum_{i=1}^N \exp(\mathbf{q} \cdot \mathbf{k}_i / \tau)}
$$


Compared to the [memory bank](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#instance-discrimination-with-memoy-bank), a queue-based dictionary in MoCo enables us to reuse representations of immediately preceding mini-batches of data.

The MoCo dictionary is not differentiable as a queue, so we cannot rely on back-propagation to update the key encoder $f_k$. One naive way might be to use the same encoder for both $f_q$ and $f_k$. Differently, MoCo proposed to use a momentum-based update with a momentum coefficient $m∈[0,1)$. Say, the parameters of $f_q$ and $f_k$ are labeled as θqθq and θkθk, respectively.

θk←mθk+(1−m)θqθk←mθk+(1−m)θq

![MoCo](https://lilianweng.github.io/lil-log/assets/images/MoCo.png)

*Fig. 12. Illustration of how Momentum Contrast (MoCo) learns visual representations. (Image source: [He et al, 2019](https://arxiv.org/abs/1911.05722))*

The advantage of MoCo compared to [SimCLR](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#simclr) is that MoCo decouples the batch size from the number of negatives, but SimCLR requires a large batch size in order to have enough negative samples and suffers performance drops when their batch size is reduced.

Two designs in SimCLR, namely, (1) an MLP projection head and (2) stronger data augmentation, are proved to be very efficient. **MoCo V2** ([Chen et al, 2020](https://arxiv.org/abs/2003.04297)) combined these two designs, achieving even better transfer performance with no dependency on a very large batch size.

#### CURL

**CURL** ([Srinivas, et al. 2020](https://arxiv.org/abs/2004.04136)) applies the above ideas in [Reinforcement Learning](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html). It learns a visual representation for RL tasks by matching embeddings of two data-augmented versions, oqoq and okok, of the raw observation oo via contrastive loss. CURL primarily relies on random crop data augmentation. The key encoder is implemented as a momentum encoder with weights as EMA of the query encoder weights, same as in [MoCo](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#moco--moco-v2).

One significant difference between RL and supervised visual tasks is that RL depends on *temporal consistency* between consecutive frames. Therefore, CURL applies augmentation consistently on each stack of frames to retain information about the temporal structure of the observation.

![CURL](https://lilianweng.github.io/lil-log/assets/images/CURL.png)

*Fig. 13. The architecture of CURL. (Image source: [Srinivas, et al. 2020](https://arxiv.org/abs/2004.04136))*

### Feature Clustering

#### DeepCluster

**DeepCluster** ([Caron et al. 2018](https://arxiv.org/abs/1807.05520)) iteratively clusters features via k-means and uses cluster assignments as pseudo labels to provide supervised signals.

![DeepCluster](https://lilianweng.github.io/lil-log/assets/images/deepcluster.png)

*Fig. 14. Illustration of DeepCluster method which iteratively clusters deep features and uses the cluster assignments as pseudo-labels. (Image source: [Caron et al. 2018](https://arxiv.org/abs/1807.05520))*

In each iteration, DeepCluster clusters data points using the prior representation and then produces the new cluster assignments as the classification targets for the new representation. However this iterative process is prone to trivial solutions. While avoiding the use of negative pairs, it requires a costly clustering phase and specific precautions to avoid collapsing to trivial solutions.

#### SwAV

**SwAV** (*Swapping Assignments between multiple Views*; [Caron et al. 2020](https://arxiv.org/abs/2006.09882)) is an online contrastive learning algorithm. It computes a code from an augmented version of the image and tries to predict this code using another augmented version of the same image.

![SwAV](https://lilianweng.github.io/lil-log/assets/images/SwAV.png)

*Fig. 15. Comparison of SwAV and [contrastive instance learning](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#instance-discrimination-with-memoy-bank). (Image source: [Caron et al. 2020](https://arxiv.org/abs/2006.09882))*

Given features of images with two different augmentations, ztzt and zszs, SwAV computes corresponding codes qtqt and qsqs and the loss quantifies the fit by swapping two codes using ℓ(.)ℓ(.) to measure the fit between a feature and a code.

LSwAV(zt,zs)=ℓ(zt,qs)+ℓ(zs,qt)LSwAV(zt,zs)=ℓ(zt,qs)+ℓ(zs,qt)

The swapped fit prediction depends on the cross entropy between the predicted code and a set of KK trainable prototype vectors C={c1,…,cK}C={c1,…,cK}. The prototype vector matrix is shared across different batches and represents *anchor clusters* that each instance should be clustered to.

ℓ(zt,qs)=−∑kq(k)slogp(k)t where p(k)t=exp(z⊤tck/τ)∑k′exp(z⊤tck′/τ)ℓ(zt,qs)=−∑kqs(k)log⁡pt(k) where pt(k)=exp⁡(zt⊤ck/τ)∑k′exp⁡(zt⊤ck′/τ)

In a mini-batch containing BB feature vectors Z=[z1,…,zB]Z=[z1,…,zB], the mapping matrix between features and prototype vectors is defined as Q=[q1,…,qB]∈RK×B+Q=[q1,…,qB]∈R+K×B. We would like to maximize the similarity between the features and the prototypes:

maxQ∈Qwhere QTr(Q⊤C⊤Z)+εH(Q)={Q∈RK×B+∣Q1B=1K1K,Q⊤1K=1B1B}maxQ∈QTr(Q⊤C⊤Z)+εH(Q)where Q={Q∈R+K×B∣Q1B=1K1K,Q⊤1K=1B1B}

where HH is the entropy, H(Q)=−∑ijQijlogQijH(Q)=−∑ijQijlog⁡Qij, controlling the smoothness of the code. The coefficient ϵϵ should not be too large; otherwise, all the samples will be assigned uniformly to all the clusters. The candidate set of solutions for QQ requires every mapping matrix to have each row sum up to 1/K1/K and each column to sum up to 1/B1/B, enforcing that each prototype gets selected at least B/KB/K times on average.

SwAV relies on the iterative Sinkhorn-Knopp algorithm ([Cuturi 2013](https://arxiv.org/abs/1306.0895)) to find the solution for QQ.

### Working with Supervised Datasets

#### CLIP

**CLIP** (*Contrastive Language-Image Pre-training*; [Radford et al. 2021](https://arxiv.org/abs/2103.00020)) jointly trains a text encoder and an image feature extractor over the pretraining task that predicts which caption goes with which image.

![CLIP](https://lilianweng.github.io/lil-log/assets/images/CLIP.png)

*Fig. 16. Illustration of CLIP contrastive pre-training over text-image pairs. (Image source: [Radford et al. 2021](https://arxiv.org/abs/2103.00020))*

Given a batch of NN (image, text) pairs, CLIP computes the dense cosine similarity matrix between all N×NN×N possible (image, text) candidates within this batch. The text and image encoders are jointly trained to maximize the similarity between NN correct pairs of (image, text) associations while minimizing the similarity for N(N−1)N(N−1) incorrect pairs via a symmetric cross entropy loss over the dense matrix.

See the numy-like pseudo code for CLIP in Fig. 17.

![CLIP pseudo code](https://lilianweng.github.io/lil-log/assets/images/CLIP-algo.png)

*Fig. 17. CLIP algorithm in Numpy style pseudo code. (Image source: [Radford et al. 2021](https://arxiv.org/abs/2103.00020))*

Compared to other methods above for learning good visual representation, what makes CLIP really special is *“the appreciation of using natural language as a training signal”*. It does demand access to supervised dataset in which we know which text matches which image. It is trained on 400 million (text, image) pairs, collected from the Internet. The query list contains all the words occurring at least 100 times in the English version of Wikipedia. Interestingly, they found that Transformer-based language models are 3x slower than a bag-of-words (BoW) text encoder at zero-shot ImageNet classification. Using contrastive objective instead of trying to predict the exact words associated with images (i.e. a method commonly adopted by image caption prediction tasks) can further improve the data efficiency another 4x.

![CLIP efficiency](https://lilianweng.github.io/lil-log/assets/images/CLIP-efficiency.png)

*Fig. 18. Using bag-of-words text encoding and contrastive training objectives can bring in multiple folds of data efficiency improvement. (Image source: [Radford et al. 2021](https://arxiv.org/abs/2103.00020))*

CLIP produces good visual representation that can non-trivially transfer to many CV benchmark datasets, achieving results competitive with supervised baseline. Among tested transfer tasks, CLIP struggles with very fine-grained classification, as well as abstract or systematic tasks such as counting the number of objects. The transfer performance of CLIP models is smoothly correlated with the amount of model compute.

#### Supervised Contrastive Learning

There are several known issues with cross entropy loss, such as the lack of robustness to noisy labels and the possibility of poor margins. Existing improvement for cross entropy loss involves the curation of better training data, such as label smoothing and data augmentation. **Supervised Contrastive Loss** ([Khosla et al. 2021](https://arxiv.org/abs/2004.11362)) aims to leverage label information more effectively than cross entropy, imposing that normalized embeddings from the same class are closer together than embeddings from different classes.

![SupCon](https://lilianweng.github.io/lil-log/assets/images/sup-con.png)

*Fig. 19. Supervised vs self-supervised contrastive losses. Supervised contrastive learning considers different samples from the same class as positive examples, in addition to augmented versions. (Image source: [Khosla et al. 2021](https://arxiv.org/abs/2004.11362))*

Given a set of randomly sampled nn (image, label) pairs, {xi,yi}ni=1{xi,yi}i=1n, 2n2n training pairs can be created by applying two random augmentations of every sample, {x~i,y~i}2ni=1{x~i,y~i}i=12n.

Supervised contrastive loss LsupconLsupcon utilizes multiple positive and negative samples, very similar to [soft nearest-neighbor loss](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#soft-nearest-neighbors-loss):

Lsupcon=−∑i=12n12|Ni|−1∑j∈N(yi),j≠ilogexp(zi⋅zj/τ)∑k∈I,k≠iexp(zi⋅zk/τ)Lsupcon=−∑i=12n12|Ni|−1∑j∈N(yi),j≠ilog⁡exp⁡(zi⋅zj/τ)∑k∈I,k≠iexp⁡(zi⋅zk/τ)

where zk=P(E(xk~))zk=P(E(xk~)), in which E(.)E(.) is an encoder network (augmented image mapped to vector) P(.)P(.) is a projection network (one vector mapped to another). Ni={j∈I:y~j=y~i}Ni={j∈I:y~j=y~i} contains a set of indices of samples with label yiyi. Including more positive samples into the set $N_i$ leads to improved results.

According to their experiments, supervised contrastive loss:

- does outperform the base cross entropy, but only by a small amount.
- outperforms the cross entropy on robustness benchmark (ImageNet-C, which applies common naturally occuring perturbations such as noise, blur and contrast changes to the ImageNet dataset).
- is less sensitive to hyperparameter changes.

## Language: Sentence Embedding

In this section, we focus on how to learn sentence embedding.

### Text Augmentation

Most contrastive methods in vision applications depend on creating an augmented version of each image. However, it is more challenging to construct text augmentation which does not alter the semantics of a sentence. In this section we look into three approaches for augmenting text sequences, including lexical edits, back-translation and applying cutoff or dropout.

#### Lexical Edits

**EDA** (*Easy Data Augmentation*; [Wei & Zou 2019](https://arxiv.org/abs/1901.11196)) defines a set of simple but powerful operations for text augmentation. Given a sentence, EDA randomly chooses and applies one of four simple operations:

1. Synonym replacement (SR): Replace nn random non-stop words with their synonyms.
2. Random insertion (RI): Place a random synonym of a randomly selected non-stop word in the sentence at a random position.
3. Random swap (RS): Randomly swap two words and repeat nn times.
4. Random deletion (RD): Randomly delete each word in the sentence with probability pp.

where p=αp=α and n=α×sentence_lengthn=α×sentence_length, with the intuition that longer sentences can absorb more noise while maintaining the original label. The hyperparameter αα roughly indicates the percent of words in one sentence that may be changed by one augmentation.

EDA is shown to improve the classification accuracy on several classification benchmark datasets compared to baseline without EDA. The performance lift is more significant on a smaller training set. All the four operations in EDA help improve the classification accuracy, but get to optimal at different αα’s.

![EDA classification](https://lilianweng.github.io/lil-log/assets/images/EDA-exp1.png)

*Fig. 20. EDA leads to performance improvement on several classification benchmarks. (Image source: [Wei & Zou 2019](https://arxiv.org/abs/1901.11196))*

In **Contextual Augmentation** ([Sosuke Kobayashi, 2018](https://arxiv.org/abs/1805.06201)), new substitutes for word wiwi at position ii can be smoothly sampled from a given probability distribution, p(.∣S∖{wi})p(.∣S∖{wi}), which is predicted by a bidirectional LM like BERT.

#### Back-translation

**CERT** (*Contrastive self-supervised Encoder Representations from Transformers*; [Fang et al. (2020)](https://arxiv.org/abs/2005.12766); [code](https://github.com/UCSD-AI4H/CERT)) generates augmented sentences via **back-translation**. Various translation models for different languages can be employed for creating different versions of augmentations. Once we have a noise version of text samples, many contrastive learning frameworks introduced above, such as [MoCo](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#moco--moco-v2), can be used to learn sentence embedding.

#### Dropout and Cutoff

[Shen et al. (2020)](https://arxiv.org/abs/2009.13818) proposed to apply **Cutoff** to text augmentation, inspired by [cross-view training](https://lilianweng.github.io/lil-log/2019/01/31/generalized-language-models.html#cross-view-training). They proposed three cutoff augmentation strategies:

1. *Token cutoff* removes the information of a few selected tokens. To make sure there is no data leakage, corresponding tokens in the input, positional and other relevant embedding matrices should all be zeroed out.,
2. *Feature cutoff* removes a few feature columns.
3. *Span cutoff* removes a continuous chunk of texts.

![Text cutoff](https://lilianweng.github.io/lil-log/assets/images/text-cutoff.png)

*Fig. 21. Schematic illustration of token, feature and span cutoff augmentation strategies. (Image source: [Shen et al. 2020](https://arxiv.org/abs/2009.13818))*

Multiple augmented versions of one sample can be created. When training, [Shen et al. (2020)](https://arxiv.org/abs/2009.13818) applied an additional KL-divergence term to measure the consensus between predictions from different augmented samples.

**SimCSE** ([Gao et al. 2021](https://arxiv.org/abs/2104.08821); [code](https://github.com/princeton-nlp/SimCSE)) learns from unsupervised data by predicting a sentence from itself with only **dropout** noise. In other words, they treat dropout as data augmentation for text sequences. A sample is simply fed into the encoder twice with different dropout masks and these two versions are the positive pair where the other in-batch samples are considered as negative pairs. It feels quite similar to the cutoff augmentation, but dropout is more flexible with less well-defined semantic meaning of what content can be masked off.

![SimCSE](https://lilianweng.github.io/lil-log/assets/images/SimCSE.png)

*Fig. 22. SimCSE creates augmented samples by applying different dropout masks. The supervised version leverages NLI datasets to predict positive (entailment) or negative (contradiction) given a pair of sentences. (Image source: [Gao et al. 2021](https://arxiv.org/abs/2104.08821))*

They ran experiments on 7 STS (Semantic Text Similarity) datasets and computed cosine similarity between sentence embeddings. They also tried out an optional MLM auxiliary objective loss to help avoid catastrophic forgetting of token-level knowledge. This aux loss was found to help improve performance on transfer tasks, but a consistent drop on the main STS tasks.

![SimCSE experiments](https://lilianweng.github.io/lil-log/assets/images/SimCSE-STS-exp.png)

*Fig. 23. Experiment numbers on a collection of STS benchmarks with SimCES. (Image source: [Gao et al. 2021](https://arxiv.org/abs/2104.08821))*

### Supervision from NLI

The pre-trained BERT sentence embedding without any fine-tuning has been found to have poor performance for semantic similarity tasks. Instead of using the raw embeddings directly, we need to refine the embedding with further fine-tuning.

**Natural Language Inference (NLI)** tasks are the main data sources to provide supervised signals for learning sentence embedding; such as [SNLI](https://nlp.stanford.edu/projects/snli/), [MNLI](https://cims.nyu.edu/~sbowman/multinli/), and [QQP](https://www.kaggle.com/c/quora-question-pairs).

#### Sentence-BERT

**SBERT (Sentence-BERT)** ([Reimers & Gurevych, 2019](https://arxiv.org/abs/1908.10084)) relies on siamese and triplet network architectures to learn sentence embeddings such that the sentence similarity can be estimated by cosine similarity between pairs of embeddings. Note that learning SBERT depends on supervised data, as it is fine-tuned on several NLI datasets.

They experimented with a few different prediction heads on top of BERT model:

- Softmax classification objective: The classification head of the siamese network is built on the concatenation of two embeddings f(x),f(x′)f(x),f(x′) and |f(x)−f(x′)||f(x)−f(x′)|. The predicted output is y^=softmax(Wt[f(x);f(x′);|f(x)−f(x′)|])y^=softmax(Wt[f(x);f(x′);|f(x)−f(x′)|]). They showed that the most important component is the element-wise difference |f(x)−f(x′)||f(x)−f(x′)|.
- Regression objective: This is the regression loss on cos(f(x),f(x′))cos⁡(f(x),f(x′)), in which the pooling strategy has a big impact. In the experiments, they observed that `max` performs much worse than `mean` and `CLS`-token.
- Triplet objective: max(0,∥f(x)−f(x+)∥−∥f(x)−f(x−)∥+ϵ)max(0,‖f(x)−f(x+)‖−‖f(x)−f(x−)‖+ϵ), where x,x+,x−x,x+,x− are embeddings of the anchor, positive and negative sentences.

In the experiments, which objective function works the best depends on the datasets, so there is no universal winner.

![SBERT](https://lilianweng.github.io/lil-log/assets/images/SBERT.png)

*Fig. 24. Illustration of Sentence-BERT training framework with softmax classification head and regression head. (Image source: [Reimers & Gurevych, 2019](https://arxiv.org/abs/1908.10084))*

The [SentEval](https://github.com/facebookresearch/SentEval) library ([Conneau and Kiela, 2018](https://arxiv.org/abs/1803.05449)) is commonly used for evaluating the quality of learned sentence embedding. SBERT outperformed other baselines at that time (Aug 2019) on 5 out of 7 tasks.

![SBERT SentEval results](https://lilianweng.github.io/lil-log/assets/images/SBERT-SentEval.png)

*Fig. 25. The performance of Sentence-BERT on the SentEval benchmark. (Image source: [Reimers & Gurevych, 2019](https://arxiv.org/abs/1908.10084))*

#### BERT-flow

The embedding representation space is deemed *isotropic* if embeddings are uniformly distributed on each dimension; otherwise, it is *anisotropic*. [Li et al, (2020)](https://arxiv.org/abs/2011.05864) showed that a pre-trained BERT learns a non-smooth *anisotropic* semantic space of sentence embeddings and thus leads to poor performance for text similarity tasks without fine-tuning. Empirically, they observed two issues with BERT sentence embedding: Word frequency biases the embedding space. High-frequency words are close to the origin, but low-frequency ones are far away from the origin. Low-frequency words scatter sparsely. The embeddings of low-frequency words tend to be farther to their kk-NN neighbors, while the embeddings of high-frequency words concentrate more densely.

**BERT-flow** ([Li et al, 2020](https://arxiv.org/abs/2011.05864); [code](https://github.com/bohanli/BERT-flow)) was proposed to transform the embedding to a smooth and isotropic Gaussian distribution via [normalizing flows](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html#what-is-normalizing-flows).

![BERT-flow](https://lilianweng.github.io/lil-log/assets/images/BERT-flow.png)

*Fig. 26. Illustration of the flow-based calibration over the original sentence embedding space in BERT-flow. (Image source: [Li et al, 2020](https://arxiv.org/abs/2011.05864))*

Let UU be the observed BERT sentence embedding space and ZZ be the desired latent space which is a standard Gaussian. Thus, pZpZ is a Gaussian density function and fϕ:Z→Ufϕ:Z→U is an invertible transformation:

z∼pZ(z)u=fϕ(z)z=f−1ϕ(u)z∼pZ(z)u=fϕ(z)z=fϕ−1(u)

A flow-based generative model learns the invertible mapping function by maximizing the likelihood of UU’s marginal:

maxϕEu=BERT(s),s∼D[logpZ(f−1ϕ(u))+log∣∣det∂f−1ϕ(u)∂u∣∣]maxϕEu=BERT(s),s∼D[log⁡pZ(fϕ−1(u))+log⁡|det∂fϕ−1(u)∂u|]

where ss is a sentence sampled from the text corpus DD. Only the flow parameters ϕϕ are optimized while parameters in the pretrained BERT stay unchanged.

BERT-flow was shown to improve the performance on most STS tasks either with or without supervision from NLI datasets. Because learning normalizing flows for calibration does not require labels, it can utilize the entire dataset including validation and test sets.

#### Whitening Operation

[Su et al. (2021)](https://arxiv.org/abs/2103.15316) applied **whitening** operation to improve the [isotropy](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#isotropy) of the learned representation and also to reduce the dimensionality of sentence embedding.

They transform the mean value of the sentence vectors to 0 and the covariance matrix to the identity matrix. Given a set of samples {xi}Ni=1{xi}i=1N, let x~ix~i and Σ~Σ~ be the transformed samples and corresponding covariance matrix:

μx~i=1N∑i=1NxiΣ=1N∑i=1N(xi−μ)⊤(xi−μ)=(xi−μ)WΣ~=W⊤ΣW=I thus Σ=(W−1)⊤W−1μ=1N∑i=1NxiΣ=1N∑i=1N(xi−μ)⊤(xi−μ)x~i=(xi−μ)WΣ~=W⊤ΣW=I thus Σ=(W−1)⊤W−1

If we get [SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition) decomposition of Σ=UΛU⊤Σ=UΛU⊤, we will have W−1=Λ−−√U⊤W−1=ΛU⊤ and W=UΛ−1−−−√W=UΛ−1. Note that within SVD, UU is an orthogonal matrix with column vectors as eigenvectors and ΛΛ is a diagonal matrix with all positive elements as sorted eigenvalues.

A dimensionality reduction strategy can be applied by only taking the first kk columns of WW, named `Whitening`-kk.

![Whitening-SBERT](https://lilianweng.github.io/lil-log/assets/images/whitening-SBERT.png)

*Fig. 27. Pseudo code of the whitening-kk operation. (Image source: [Su et al. 2021](https://arxiv.org/abs/2103.15316))*

Whitening operations were shown to outperform BERT-flow and achieve SOTA with 256 sentence dimensionality on many STS benchmarks, either with or without NLI supervision.

### Unsupervised Sentence Embedding Learning

#### Context Prediction

**Quick-Thought (QT) vectors** ([Logeswaran & Lee, 2018](https://arxiv.org/abs/1803.02893)) formulate sentence representation learning as a *classification* problem: Given a sentence and its context, a classifier distinguishes context sentences from other contrastive sentences based on their vector representations ([“cloze test”](https://lilianweng.github.io/lil-log/2019/01/31/generalized-language-models.html#MLM)). Such a formulation removes the softmax output layer which causes training slowdown.

![Quick-Thought vectors](https://lilianweng.github.io/lil-log/assets/images/quick-thought.png)

*Fig. 28. Illustration of how Quick-Thought sentence embedding vectors are learned. (Image source: [Logeswaran & Lee, 2018](https://arxiv.org/abs/1803.02893))*

Let f(.)f(.) and g(.)g(.) be two functions that encode a sentence ss into a fixed-length vector. Let C(s)C(s) be the set of sentences in the context of ss and S(s)S(s) be the set of candidate sentences including only one sentence sc∈C(s)sc∈C(s) and many other non-context negative sentences. Quick Thoughts model learns to optimize the probability of predicting the only true context sentence sc∈S(s)sc∈S(s). It is essentially NCE loss when considering the sentence (s,sc)(s,sc) as the positive pairs while other pairs (s,s′)(s,s′) where s′∈S(s),s′≠scs′∈S(s),s′≠sc as negatives.

LQT=−∑s∈D∑sc∈C(s)logp(sc|s,S(s))=−∑s∈D∑sc∈C(s)exp(f(s)⊤g(sc))∑s′∈S(s)exp(f(s)⊤g(s′))LQT=−∑s∈D∑sc∈C(s)log⁡p(sc|s,S(s))=−∑s∈D∑sc∈C(s)exp⁡(f(s)⊤g(sc))∑s′∈S(s)exp⁡(f(s)⊤g(s′))

#### Mutual Information Maximization

**IS-BERT (Info-Sentence BERT)** ([Zhang et al. 2020](https://arxiv.org/abs/2009.12061); [code](https://github.com/yanzhangnlp/IS-BERT)) adopts a self-supervised learning objective based on *mutual information maximization* to learn good sentence embeddings in the *unsupervised* manners.

![IS-BERT](https://lilianweng.github.io/lil-log/assets/images/IS-BERT.png)

*Fig. 29. Illustration of Info-Sentence BERT. (Image source: [Zhang et al. 2020](https://arxiv.org/abs/2009.12061))*

IS-BERT works as follows:

1. Use BERT to encode an input sentence ss to a token embedding of length ll, h1:lh1:l.
2. Then apply 1-D conv net with different kernel sizes (e.g. 1, 3, 5) to process the token embedding sequence to capture the n-gram local contextual dependencies: ci=ReLU(w⋅hi:i+k−1+b)ci=ReLU(w⋅hi:i+k−1+b). The output sequences are padded to stay the same sizes of the inputs.
3. The final local representation of the ii-th token F(i)θ(x)Fθ(i)(x) is the concatenation of representations of different kernel sizes.
4. The global sentence representation Eθ(x)Eθ(x) is computed by applying a mean-over-time pooling layer on the token representations Fθ(x)={F(i)θ(x)∈Rd}li=1Fθ(x)={Fθ(i)(x)∈Rd}i=1l.

Since the mutual information estimation is generally intractable for continuous and high-dimensional random variables, IS-BERT relies on the Jensen-Shannon estimator ([Nowozin et al., 2016](https://arxiv.org/abs/1606.00709), [Hjelm et al., 2019](https://arxiv.org/abs/1808.06670)) to maximize the mutual information between Eθ(x)Eθ(x) and F(i)θ(x)Fθ(i)(x).

IJSDω(F(i)θ(x);Eθ(x))=Ex∼P[−sp(−Tω(F(i)θ(x);Eθ(x)))]−Ex∼P,x′∼P~[sp(Tω(F(i)θ(x′);Eθ(x)))]IωJSD(Fθ(i)(x);Eθ(x))=Ex∼P[−sp(−Tω(Fθ(i)(x);Eθ(x)))]−Ex∼P,x′∼P~[sp(Tω(Fθ(i)(x′);Eθ(x)))]

where Tω:F×E→RTω:F×E→R is a learnable network with parameters ωω, generating discriminator scores. The negative sample x′x′ is sampled from the distribution P~=PP~=P. And sp(x)=log(1+ex)sp(x)=log⁡(1+ex) is the softplus activation function.

The unsupervised numbers on SentEval with IS-BERT outperforms most of the unsupervised baselines (Sep 2020), but unsurprisingly weaker than supervised runs. When using labelled NLI datasets, IS-BERT produces results comparable with SBERT (See Fig. 25 & 30).

![IS-BERT SentEval results](https://lilianweng.github.io/lil-log/assets/images/IS-BERT-SentEval.png)

*Fig. 30. The performance of IS-BERT on the SentEval benchmark. (Image source: [Zhang et al. 2020](https://arxiv.org/abs/2009.12061))*

## References

[1] Sumit Chopra, Raia Hadsell and Yann LeCun. [“Learning a similarity metric discriminatively, with application to face verification.”](http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf) CVPR 2005.

[2] Florian Schroff, Dmitry Kalenichenko and James Philbin. [“FaceNet: A Unified Embedding for Face Recognition and Clustering.”](https://arxiv.org/abs/1503.03832) CVPR 2015.

[3] Hyun Oh Song et al. [“Deep Metric Learning via Lifted Structured Feature Embedding.”](https://arxiv.org/abs/1511.06452) CVPR 2016. [[code](https://github.com/rksltnl/Deep-Metric-Learning-CVPR16)]

[4] Ruslan Salakhutdinov and Geoff Hinton. [“Learning a Nonlinear Embedding by Preserving Class Neighbourhood Structure”](http://proceedings.mlr.press/v2/salakhutdinov07a.html) AISTATS 2007.

[5] Michael Gutmann and Aapo Hyvärinen. [“Noise-contrastive estimation: A new estimation principle for unnormalized statistical models.”](http://proceedings.mlr.press/v9/gutmann10a.html) AISTATS 2010.

[6] Kihyuk Sohn et al. [“Improved Deep Metric Learning with Multi-class N-pair Loss Objective”](https://papers.nips.cc/paper/2016/hash/6b180037abbebea991d8b1232f8a8ca9-Abstract.html) NIPS 2016.

[7] Nicholas Frosst, Nicolas Papernot and Geoffrey Hinton. [“Analyzing and Improving Representations with the Soft Nearest Neighbor Loss.”](http://proceedings.mlr.press/v97/frosst19a.html) ICML 2019

[8] Tongzhou Wang and Phillip Isola. [“Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere.”](https://arxiv.org/abs/2005.10242) ICML 2020. [[code](https://ssnl.github.io/hypersphere/)]

[9] Zhirong Wu et al. [“Unsupervised feature learning via non-parametric instance-level discrimination.”](https://arxiv.org/abs/1805.01978) CVPR 2018.

[10] Ekin D. Cubuk et al. [“AutoAugment: Learning augmentation policies from data.”](https://arxiv.org/abs/1805.09501) arXiv preprint arXiv:1805.09501 (2018).

[11] Daniel Ho et al. [“Population Based Augmentation: Efficient Learning of Augmentation Policy Schedules.”](https://arxiv.org/abs/1905.05393) ICML 2019.

[12] Ekin D. Cubuk & Barret Zoph et al. [“RandAugment: Practical automated data augmentation with a reduced search space.”](https://arxiv.org/abs/1909.13719) arXiv preprint arXiv:1909.13719 (2019).

[13] Hongyi Zhang et al. [“mixup: Beyond Empirical Risk Minimization.”](https://arxiv.org/abs/1710.09412) ICLR 2017.

[14] Sangdoo Yun et al. [“CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features.”](https://arxiv.org/abs/1905.04899) ICCV 2019.

[15] Yannis Kalantidis et al. [“Mixing of Contrastive Hard Negatives”](https://arxiv.org/abs/2010.01028) NeuriPS 2020.

[16] Ashish Jaiswal et al. [“A Survey on Contrastive Self-Supervised Learning.”](https://arxiv.org/abs/2011.00362) arXiv preprint arXiv:2011.00362 (2021)

[17] Jure Zbontar et al. [“Barlow Twins: Self-Supervised Learning via Redundancy Reduction.”](https://arxiv.org/abs/2103.03230) arXiv preprint arXiv:2103.03230 (2021) [[code](https://github.com/facebookresearch/barlowtwins)]

[18] Alec Radford, et al. [“Learning Transferable Visual Models From Natural Language Supervision”](https://arxiv.org/abs/2103.00020) arXiv preprint arXiv:2103.00020 (2021)

[19] Mathilde Caron et al. [“Unsupervised Learning of Visual Features by Contrasting Cluster Assignments (SwAV).”](https://arxiv.org/abs/2006.09882) NeuriPS 2020.

[20] Mathilde Caron et al. [“Deep Clustering for Unsupervised Learning of Visual Features.”](https://arxiv.org/abs/1807.05520) ECCV 2018.

[21] Prannay Khosla et al. [“Supervised Contrastive Learning.”](https://arxiv.org/abs/2004.11362) NeurIPS 2020.

[22] Aaron van den Oord, Yazhe Li & Oriol Vinyals. [“Representation Learning with Contrastive Predictive Coding”](https://arxiv.org/abs/1807.03748) arXiv preprint arXiv:1807.03748 (2018).

[23] Jason Wei and Kai Zou. [“EDA: Easy data augmentation techniques for boosting performance on text classification tasks.”](https://arxiv.org/abs/1901.11196) EMNLP-IJCNLP 2019.

[24] Sosuke Kobayashi. [“Contextual Augmentation: Data Augmentation by Words with Paradigmatic Relations.”](https://arxiv.org/abs/1805.06201) NAACL 2018

[25] Hongchao Fang et al. [“CERT: Contrastive self-supervised learning for language understanding.”](https://arxiv.org/abs/2005.12766) arXiv preprint arXiv:2005.12766 (2020).

[26] Dinghan Shen et al. [“A Simple but Tough-to-Beat Data Augmentation Approach for Natural Language Understanding and Generation.”](https://arxiv.org/abs/2009.13818) arXiv preprint arXiv:2009.13818 (2020) [[code](https://github.com/dinghanshen/cutoff)]

[27] Tianyu Gao et al. [“SimCSE: Simple Contrastive Learning of Sentence Embeddings.”](https://arxiv.org/abs/2104.08821) arXiv preprint arXiv:2104.08821 (2020). [[code](https://github.com/princeton-nlp/SimCSE)]

[28] Nils Reimers and Iryna Gurevych. [“Sentence-BERT: Sentence embeddings using Siamese BERT-networks.”](https://arxiv.org/abs/1908.10084) EMNLP 2019.

[29] Jianlin Su et al. [“Whitening sentence representations for better semantics and faster retrieval.”](https://arxiv.org/abs/2103.15316) arXiv preprint arXiv:2103.15316 (2021). [[code](https://github.com/bojone/BERT-whitening)]

[30] Yan Zhang et al. [“An unsupervised sentence embedding method by mutual information maximization.”](https://arxiv.org/abs/2009.12061) EMNLP 2020. [[code](https://github.com/yanzhangnlp/IS-BERT)]

[31] Bohan Li et al. [“On the sentence embeddings from pre-trained language models.”](https://arxiv.org/abs/2011.05864) EMNLP 2020.

[32] Lajanugen Logeswaran and Honglak Lee. [“An efficient framework for learning sentence representations.”](https://arxiv.org/abs/1803.02893) ICLR 2018.

[33] Joshua Robinson, et al. [“Contrastive Learning with Hard Negative Samples.”](https://arxiv.org/abs/2010.04592) ICLR 2021.

[34] Ching-Yao Chuang et al. [“Debiased Contrastive Learning.”](https://arxiv.org/abs/2007.00224) NeuriPS 2020.
