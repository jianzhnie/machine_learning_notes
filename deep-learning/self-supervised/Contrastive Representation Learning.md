# Contrastive Representation Learning

May 31, 2021 by Lilian Weng [representation-learning ](https://lilianweng.github.io/lil-log/tag/representation-learning) [long-read ](https://lilianweng.github.io/lil-log/tag/long-read) [language-model ](https://lilianweng.github.io/lil-log/tag/language-model) [unsupervised-learning ](https://lilianweng.github.io/lil-log/tag/unsupervised-learning)

> The main idea of contrastive learning is to learn representations such that similar samples stay close to each other, while dissimilar ones are far apart. Contrastive learning can be applied to both supervised and unsupervised data and has been shown to achieve good performance on a variety of vision and language tasks.

The goal of contrastive representation learning is to learn such an embedding space in which similar sample pairs stay close to each other while dissimilar ones are far apart. Contrastive learning can be applied to both supervised and unsupervised settings. When working with unsupervised data, contrastive learning is one of the most powerful approaches in [self-supervised learning](https://lilianweng.github.io/lil-log/2019/11/10/self-supervised-learning.html).

- Contrastive Training Objectives
  - [Contrastive Loss](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#contrastive-loss)
  - [Triplet Loss](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#triplet-loss)
  - [Lifted Structured Loss](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#lifted-structured-loss)
  - [N-pair Loss](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#n-pair-loss)
  - [NCE](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#nce)
  - [InfoNCE](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#infonce)
  - [Soft-Nearest Neighbors Loss](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#soft-nearest-neighbors-loss)
  - [Common Setup](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#common-setup)
- Key Ingredients
  - [Heavy Data Augmentation](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#heavy-data-augmentation)
  - [Large Batch Size](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#large-batch-size)
  - [Hard Negative Mining](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#hard-negative-mining)
- Vision: Image Embedding
  - [Image Augmentations](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#image-augmentations)
  - [Parallel Augmentation](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#parallel-augmentation)
  - [Memory Bank](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#memory-bank)
  - [Feature Clustering](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#feature-clustering)
  - [Working with Supervised Datasets](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#working-with-supervised-datasets)
- Language: Sentence Embedding
  - [Text Augmentation](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#text-augmentation)
  - [Supervision from NLI](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#supervision-from-nli)
  - [Unsupervised Sentence Embedding Learning](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#unsupervised-sentence-embedding-learning)
- [References](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#references)

## Contrastive Training Objectives

In early versions of loss functions for contrastive learning, only one positive and one negative sample are involved. The trend in recent training objectives is to include multiple positive and negative pairs in one batch.

### Contrastive Loss

**Contrastive loss** ([Chopra et al. 2005](http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf)) is one of the earliest training objectives used for deep metric learning in a contrastive fashion.

Given a list of input samples {xi}{xi}, each has a corresponding label yi∈{1,…,L}yi∈{1,…,L} among LL classes. We would like to learn a function fθ(.):X→Rdfθ(.):X→Rd that encodes xixi into an embedding vector such that examples from the same class have similar embeddings and samples from different classes have very different ones. Thus, contrastive loss takes a pair of inputs (xi,xj)(xi,xj) and minimizes the embedding distance when they are from the same class but maximizes the distance otherwise.

Lcont(xi,xj,θ)=1[yi=yj]∥fθ(xi)−fθ(xj)∥22+1[yi≠yj]max(0,ϵ−∥fθ(xi)−fθ(xj)∥2)2Lcont(xi,xj,θ)=1[yi=yj]‖fθ(xi)−fθ(xj)‖22+1[yi≠yj]max(0,ϵ−‖fθ(xi)−fθ(xj)‖2)2

where ϵϵ is a hyperparameter, defining the lower bound distance between samples of different classes.

### Triplet Loss

**Triplet loss** was originally proposed in the FaceNet ([Schroff et al. 2015](https://arxiv.org/abs/1503.03832)) paper and was used to learn face recognition of the same person at different poses and angles.

![Triplet loss](https://lilianweng.github.io/lil-log/assets/images/triplet-loss.png)

Fig. 1. Illustration of triplet loss given one positive and one negative per anchor. (Image source: [Schroff et al. 2015](https://arxiv.org/abs/1503.03832))

Given one anchor input xx, we select one positive sample x+x+ and one negative x−x−, meaning that x+x+ and xx belong to the same class and x−x− is sampled from another different class. Triplet loss learns to minimize the distance between the anchor xx and positive x+x+ and maximize the distance between the anchor xx and negative x−x− at the same time with the following equation:

Ltriplet(x,x+,x−)=∑x∈Xmax(0,∥f(x)−f(x+)∥22−∥f(x)−f(x−)∥22+ϵ)Ltriplet(x,x+,x−)=∑x∈Xmax(0,‖f(x)−f(x+)‖22−‖f(x)−f(x−)‖22+ϵ)

where the margin parameter ϵϵ is configured as the minimum offset between distances of similar vs dissimilar pairs.

It is crucial to select challenging x−x− to truly improve the model.

### Lifted Structured Loss

**Lifted Structured Loss** ([Song et al. 2015](https://arxiv.org/abs/1511.06452)) utilizes all the pairwise edges within one training batch for better computational efficiency.

![Lifted structured loss](https://lilianweng.github.io/lil-log/assets/images/lifted-structured-loss.png)

Fig. 2. Illustration compares contrastive loss, triplet loss and lifted structured loss. Red and blue edges connect similar and dissimilar sample pairs respectively. (Image source: [Song et al. 2015](https://arxiv.org/abs/1511.06452))

Let Dij=∥f(xi)−f(xj)∥2Dij=‖f(xi)−f(xj)‖2, a structured loss function is defined as

Lstructwhere L(ij)struct=12|P|∑(i,j)∈Pmax(0,L(ij)struct)2=Dij+max(max(i,k)∈Nϵ−Dik,max(j,l)∈Nϵ−Djl)Lstruct=12|P|∑(i,j)∈Pmax(0,Lstruct(ij))2where Lstruct(ij)=Dij+max(max(i,k)∈Nϵ−Dik,max(j,l)∈Nϵ−Djl)

where PP contains the set of positive pairs and NN is the set of negative pairs. Note that the dense pairwise squared distance matrix can be easily computed per training batch.

The red part in L(ij)structLstruct(ij) is used for mining hard negatives. However, it is not smooth and may cause the convergence to a bad local optimum in practice. Thus, it is relaxed to be:

L(ij)struct=Dij+log(∑(i,k)∈Nexp(ϵ−Dik)+∑(j,l)∈Nexp(ϵ−Djl))Lstruct(ij)=Dij+log⁡(∑(i,k)∈Nexp⁡(ϵ−Dik)+∑(j,l)∈Nexp⁡(ϵ−Djl))

In the paper, they also proposed to enhance the quality of negative samples in each batch by actively incorporating difficult negative samples given a few random positive pairs.

### N-pair Loss

**Multi-Class N-pair loss** ([Sohn 2016](https://papers.nips.cc/paper/2016/hash/6b180037abbebea991d8b1232f8a8ca9-Abstract.html)) generalizes triplet loss to include comparison with multiple negative samples.

Given a (N+1)(N+1)-tuplet of training samples, {x,x+,x−1,…,x−N−1}{x,x+,x1−,…,xN−1−}, including one positive and N−1N−1 negative ones, N-pair loss is defined as:

LN-pair(x,x+,{x−i}N−1i=1)=log(1+∑i=1N−1exp(f(x)⊤f(x−i)−f(x)⊤f(x+)))=−logexp(f(x)⊤f(x+))exp(f(x)⊤f(x+))+∑N−1i=1exp(f(x)⊤f(x−i))LN-pair(x,x+,{xi−}i=1N−1)=log⁡(1+∑i=1N−1exp⁡(f(x)⊤f(xi−)−f(x)⊤f(x+)))=−log⁡exp⁡(f(x)⊤f(x+))exp⁡(f(x)⊤f(x+))+∑i=1N−1exp⁡(f(x)⊤f(xi−))

If we only sample one negative sample per class, it is equivalent to the softmax loss for multi-class classification.

### NCE

**Noise Contrastive Estimation**, short for **NCE**, is a method for estimating parameters of a statistical model, proposed by [Gutmann & Hyvarinen](http://proceedings.mlr.press/v9/gutmann10a.html) in 2010. The idea is to run logistic regression to tell apart the target data from noise. Read more on how NCE is used for learning word embedding [here](https://lilianweng.github.io/lil-log/2017/10/15/learning-word-embedding.html#noise-contrastive-estimation-nce).

Let xx be the target sample ∼P(x|C=1;θ)=pθ(x)∼P(x|C=1;θ)=pθ(x) and x~x~ be the noise sample ∼P(x~|C=0)=q(x~)∼P(x~|C=0)=q(x~). Note that the logistic regression models the logit (i.e. log-odds) and in this case we would like to model the logit of a sample uu from the target data distribution instead of the noise distribution:

ℓθ(u)=logpθ(u)q(u)=logpθ(u)−logq(u)ℓθ(u)=log⁡pθ(u)q(u)=log⁡pθ(u)−log⁡q(u)

After converting logits into probabilities with sigmoid σ(.)σ(.), we can apply cross entropy loss:

LNCE where σ(ℓ)=−1N∑i=1N[logσ(ℓθ(xi))+log(1−σ(ℓθ(x~i)))]=11+exp(−ℓ)=pθpθ+qLNCE=−1N∑i=1N[log⁡σ(ℓθ(xi))+log⁡(1−σ(ℓθ(x~i)))] where σ(ℓ)=11+exp⁡(−ℓ)=pθpθ+q

Here I listed the original form of NCE loss which works with only one positive and one noise sample. In many follow-up works, contrastive loss incorporating multiple negative samples is also broadly referred to as NCE.

### InfoNCE

The **InfoNCE loss** in CPC ([Contrastive Predictive Coding](https://lilianweng.github.io/lil-log/2019/11/10/self-supervised-learning.html#contrastive-predictive-coding); [van den Oord, et al. 2018](https://arxiv.org/abs/1807.03748)), inspired by [NCE](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#NCE), uses categorical cross-entropy loss to identify the positive sample amongst a set of unrelated noise samples.

Given a context vector cc, the positive sample should be drawn from the conditional distribution p(x|c)p(x|c), while N−1N−1 negative samples are drawn from the proposal distribution p(x)p(x), independent from the context cc. For brevity, let us label all the samples as X={xi}Ni=1X={xi}i=1N among which only one of them xposxpos is a positive sample. The probability of we detecting the positive sample correctly is:

p(C=pos|X,c)=p(xpos|c)∏i=1,…,N;i≠posp(xi)∑Nj=1[p(xj|c)∏i=1,…,N;i≠jp(xi)]=p(xpos|c)p(xpos)∑Nj=1p(xj|c)p(xj)=f(xpos,c)∑Nj=1f(xj,c)p(C=pos|X,c)=p(xpos|c)∏i=1,…,N;i≠posp(xi)∑j=1N[p(xj|c)∏i=1,…,N;i≠jp(xi)]=p(xpos|c)p(xpos)∑j=1Np(xj|c)p(xj)=f(xpos,c)∑j=1Nf(xj,c)

where the scoring function is f(x,c)∝p(x|c)p(x)f(x,c)∝p(x|c)p(x).

The InfoNCE loss optimizes the negative log probability of classifying the positive sample correctly:

LInfoNCE=−E[logf(x,c)∑x′∈Xf(x′,c)]LInfoNCE=−E[log⁡f(x,c)∑x′∈Xf(x′,c)]

The fact that f(x,c)f(x,c) estimates the density ratio p(x|c)p(x)p(x|c)p(x) has a connection with mutual information optimization. To maximize the the mutual information between input xx and context vector cc, we have:

I(x;c)=∑x,cp(x,c)logp(x,c)p(x)p(c)=∑x,cp(x,c)logp(x|c)p(x)I(x;c)=∑x,cp(x,c)log⁡p(x,c)p(x)p(c)=∑x,cp(x,c)log⁡p(x|c)p(x)

where the logarithmic term in blue is estimated by ff.

For sequence prediction tasks, rather than modeling the future observations pk(xt+k|ct)pk(xt+k|ct) directly (which could be fairly expensive), CPC models a density function to preserve the mutual information between xt+kxt+k and ctct:

fk(xt+k,ct)=exp(z⊤t+kWkct)∝p(xt+k|ct)p(xt+k)fk(xt+k,ct)=exp⁡(zt+k⊤Wkct)∝p(xt+k|ct)p(xt+k)

where zt+kzt+k is the encoded input and WkWk is a trainable weight matrix.

### Soft-Nearest Neighbors Loss

**Soft-Nearest Neighbors Loss** ([Salakhutdinov & Hinton 2007](http://proceedings.mlr.press/v2/salakhutdinov07a.html), [Frosst et al. 2019](https://arxiv.org/abs/1902.01889)) extends it to include multiple positive samples.

Given a batch of samples, {xi,yi)}Bi=1{xi,yi)}i=1B where yiyi is the class label of xixi and a function f(.,.)f(.,.) for measuring similarity between two inputs, the soft nearest neighbor loss at temperature ττ is defined as:

Lsnn=−1B∑i=1Blog∑i≠j,yi=yj,j=1,…,Bexp(−f(xi,xj)/τ)∑i≠k,k=1,…,Bexp(−f(xi,xk)/τ)Lsnn=−1B∑i=1Blog⁡∑i≠j,yi=yj,j=1,…,Bexp⁡(−f(xi,xj)/τ)∑i≠k,k=1,…,Bexp⁡(−f(xi,xk)/τ)

The temperature ττ is used for tuning how concentrated the features are in the representation space. For example, when at low temperature, the loss is dominated by the small distances and widely separated representations cannot contribute much and become irrelevant.

### Common Setup

We can loosen the definition of “classes” and “labels” in soft nearest-neighbor loss to create positive and negative sample pairs out of unsupervised data by, for example, applying data augmentation to create noise versions of original samples.

Most recent studies follow the following definition of contrastive learning objective to incorporate multiple positive and negative samples. According to the setup in ([Wang & Isola 2020](https://arxiv.org/abs/2005.10242)), let pdata(.)pdata(.) be the data distribution over RnRn and ppos(.,.)ppos(.,.) be the distribution of positive pairs over Rn×nRn×n. These two distributions should satisfy:

- Symmetry: ∀x,x+,ppos(x,x+)=ppos(x+,x)∀x,x+,ppos(x,x+)=ppos(x+,x)
- Matching marginal: ∀x,∫ppos(x,x+)dx+=pdata(x)∀x,∫ppos(x,x+)dx+=pdata(x)

To learn an encoder f(x)f(x) to learn a *L2-normalized feature vector*, the contrastive learning objective is:

Lcontrastive=E(x,x+)∼ppos,{x−i}Mi=1∼i.i.dpdata[−logexp(f(x)⊤f(x+)/τ)exp(f(x)⊤f(x+)/τ)+∑Mi=1exp(f(x)⊤f(x−i)/τ)]≈E(x,x+)∼ppos,{x−i}Mi=1∼i.i.dpdata[−f(x)⊤f(x+)/τ+log(∑i=1Mexp(f(x)⊤f(x−i)/τ))]=−1τE(x,x+)∼pposf(x)⊤f(x+)+Ex∼pdata[logEx−∼pdata[∑i=1Mexp(f(x)⊤f(x−i)/τ)]]; Assuming infinite negativesLcontrastive=E(x,x+)∼ppos,{xi−}i=1M∼i.i.dpdata[−log⁡exp⁡(f(x)⊤f(x+)/τ)exp⁡(f(x)⊤f(x+)/τ)+∑i=1Mexp⁡(f(x)⊤f(xi−)/τ)]≈E(x,x+)∼ppos,{xi−}i=1M∼i.i.dpdata[−f(x)⊤f(x+)/τ+log⁡(∑i=1Mexp⁡(f(x)⊤f(xi−)/τ))]; Assuming infinite negatives=−1τE(x,x+)∼pposf(x)⊤f(x+)+Ex∼pdata[log⁡Ex−∼pdata[∑i=1Mexp⁡(f(x)⊤f(xi−)/τ)]]

## Key Ingredients

### Heavy Data Augmentation

Given a training sample, data augmentation techniques are needed for creating noise versions of itself to feed into the loss as positive samples. Proper data augmentation setup is critical for learning good and generalizable embedding features. It introduces the non-essential variations into examples without modifying semantic meanings and thus encourages the model to learn the essential part of the representation. For example, experiments in [SimCLR](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#simclr) showed that the composition of random cropping and random color distortion is crucial for good performance on learning visual representation of images.

### Large Batch Size

Using a large batch size during training is another key ingredient in the success of many contrastive learning methods (e.g. [SimCLR](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#simclr), [CLIP](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#clip)), especially when it relies on in-batch negatives. Only when the batch size is big enough, the loss function can cover a diverse enough collection of negative samples, challenging enough for the model to learn meaningful representation to distinguish different examples.

### Hard Negative Mining

Hard negative samples should have different labels from the anchor sample, but have embedding features very close to the anchor embedding. With access to ground truth labels in supervised datasets, it is easy to identify task-specific hard negatives. For example when learning sentence embedding, we can treat sentence pairs labelled as “contradiction” in NLI datasets as hard negative pairs (e.g. [SimCSE](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#dropout-and-cutoff), or use top incorrect candidates returned by BM25 with most keywords matched as hard negative samples ([DPR](https://lilianweng.github.io/lil-log/2020/10/29/open-domain-question-answering.html#DPR); [Karpukhin et al., 2020](https://arxiv.org/abs/2004.04906)).

However, it becomes tricky to do hard negative mining when we want to remain unsupervised. Increasing training batch size or [memory bank](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#memory-bank) size implicitly introduces more hard negative samples, but it leads to a heavy burden of large memory usage as a side effect.

[Chuang et al. (2020)](https://arxiv.org/abs/2007.00224) studied the sampling bias in contrastive learning and proposed debiased loss. In the unsupervised setting, since we do not know the ground truth labels, we may accidentally sample false negative samples. Sampling bias can lead to significant performance drop.

![Sampling bias](https://lilianweng.github.io/lil-log/assets/images/contrastive-sampling-bias.png)

*Fig. 3. Sampling bias which refers to false negative samples in contrastive learning can lead to a big performance drop. (Image source: [Chuang et al., 2020](https://arxiv.org/abs/2007.00224))*

Let us assume the probability of anchor class cc is uniform ρ(c)=η+ρ(c)=η+ and the probability of observing a different class is η−=1−η+η−=1−η+.

- The probability of observing a positive example for xx is p+x(x′)=p(x′|hx′=hx)px+(x′)=p(x′|hx′=hx);
- The probability of getting a negative sample for xx is p−x(x′)=p(x′|hx′≠hx)px−(x′)=p(x′|hx′≠hx).

When we are sampling x−x− , we cannot access the true p−x(x−)px−(x−) and thus x−x− may be sampled from the (undesired) anchor class cc with probability η+η+. The actual sampling data distribution becomes:

p(x′)=η+p+x(x′)+η−p−x(x′)p(x′)=η+px+(x′)+η−px−(x′)

Thus we can use p−x(x′)=(p(x′)−η+p+x(x′))/η−px−(x′)=(p(x′)−η+px+(x′))/η− for sampling x−x− to debias the loss. With NN samples {ui}Ni=1{ui}i=1N from pp and MM samples {vi}Mi=1{vi}i=1M from p+xpx+ , we can estimate the expectation of the second term Ex−∼p−x[exp(f(x)⊤f(x−))]Ex−∼px−[exp⁡(f(x)⊤f(x−))] in the denominator of contrastive learning loss:

g(x,{ui}Ni=1,{vi}Mi=1)=max{1η−(1N∑i=1Nexp(f(x)⊤f(ui))−η+M∑i=1Mexp(f(x)⊤f(vi))),exp(−1/τ)}g(x,{ui}i=1N,{vi}i=1M)=max{1η−(1N∑i=1Nexp⁡(f(x)⊤f(ui))−η+M∑i=1Mexp⁡(f(x)⊤f(vi))),exp⁡(−1/τ)}

where ττ is the temperature and exp(−1/τ)exp⁡(−1/τ) is the theoretical lower bound of Ex−∼p−x[exp(f(x)⊤f(x−))]Ex−∼px−[exp⁡(f(x)⊤f(x−))].

The final debiased contrastive loss looks like:

LN,Mdebias(f)=Ex,{ui}Ni=1∼p;x+,{vi}Mi=1∼p+[−logexp(f(x)⊤f(x+)exp(f(x)⊤f(x+)+Ng(x,{ui}Ni=1,{vi}Mi=1)]LdebiasN,M(f)=Ex,{ui}i=1N∼p;x+,{vi}i=1M∼p+[−log⁡exp⁡(f(x)⊤f(x+)exp⁡(f(x)⊤f(x+)+Ng(x,{ui}i=1N,{vi}i=1M)]

![Debiased t-SNE vis](https://lilianweng.github.io/lil-log/assets/images/contrastive-debias-t-SNE.png)

*Fig. 4. t-SNE visualization of learned representation with debiased contrastive learning. (Image source: [Chuang et al., 2020](https://arxiv.org/abs/2007.00224))*

Following the above annotation, [Robinson et al. (2021)](https://arxiv.org/abs/2010.04592) modified the sampling probabilities to target at hard negatives by up-weighting the probability p−x(x′)px−(x′) to be proportional to its similarity to the anchor sample. The new sampling probability qβ(x−)qβ(x−) is:

qβ(x−)∝exp(βf(x)⊤f(x−))⋅p(x−)qβ(x−)∝exp⁡(βf(x)⊤f(x−))⋅p(x−)

where ββ is a hyperparameter to tune.

We can estimate the second term in the denominator Ex−∼qβ[exp(f(x)⊤f(x−))]Ex−∼qβ[exp⁡(f(x)⊤f(x−))] using importance sampling where both the partition functions Zβ,Z+βZβ,Zβ+ can be estimated empirically.

Eu∼qβ[exp(f(x)⊤f(u))]Ev∼q+β[exp(f(x)⊤f(v))]=Eu∼p[qβpexp(f(x)⊤f(u))]=Eu∼p[1Zβexp((β+1)f(x)⊤f(u))]=Ev∼p+[q+βpexp(f(x)⊤f(v))]=Ev∼p[1Z+βexp((β+1)f(x)⊤f(v))]Eu∼qβ[exp⁡(f(x)⊤f(u))]=Eu∼p[qβpexp⁡(f(x)⊤f(u))]=Eu∼p[1Zβexp⁡((β+1)f(x)⊤f(u))]Ev∼qβ+[exp⁡(f(x)⊤f(v))]=Ev∼p+[qβ+pexp⁡(f(x)⊤f(v))]=Ev∼p[1Zβ+exp⁡((β+1)f(x)⊤f(v))]

![Pseudo code](https://lilianweng.github.io/lil-log/assets/images/contrastive-hard-negatives-code.png)

*Fig. 5. Pseudo code for computing NCE loss, debiased contrastive loss, and hard negative sample objective when setting M=1M=1. (Image source: [Robinson et al., 2021](https://arxiv.org/abs/2010.04592) )*

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
- UDA (Unsupervised Data Augmentation; [Xie et al., 2019](https://arxiv.org/abs/1904.12848)): Among a set of possible augmentation strategies, UDA selects those to minimize the KL divergence between the predicted distribution over an unlabelled example and its unlabelled augmented version.

#### Image Mixture

Image mixture methods can construct new training examples from existing data points.

- Mixup ([Zhang et al., 2018](https://arxiv.org/abs/1710.09412)): It runs global-level mixture by creating a weighted pixel-wise combination of two existing images I1I1 and I2I2: Imixup←αI1+(1−α)I2Imixup←αI1+(1−α)I2 and α∈[0,1]α∈[0,1].
- Cutmix ([Yun et al., 2019](https://arxiv.org/abs/1905.04899)): Cutmix does region-level mixture by generating a new example by combining a local region of one image with the rest of the other image. Icutmix←Mb⊙I1+(1−Mb)⊙I2Icutmix←Mb⊙I1+(1−Mb)⊙I2, where Mb∈{0,1}IMb∈{0,1}I is a binary mask and ⊙⊙ is element-wise multiplication. It is equivalent to filling the cutout ([DeVries & Taylor 2017](https://arxiv.org/abs/1708.04552)) region with the same region from another image.
- MoCHi (“Mixing of Contrastive Hard Negatives”; [Kalantidis et al. 2020](https://arxiv.org/abs/2010.01028)): Given a query qq, MoCHi maintains a queue of KK negative features Q={n1,…,nK}Q={n1,…,nK} and sorts these negative features by similarity to the query, q⊤nq⊤n, in descending order. The first NN items in the queue are considered as the hardest negatives, QNQN. Then synthetic hard examples can be generated by h=h~/∥h~∥h=h~/‖h~‖ where h~=αni+(1−α)njh~=αni+(1−α)nj and α∈(0,1)α∈(0,1). Even harder examples can be created by mixing with the query feature, h′=h′~/∥h′~∥2h′=h′~/‖h′~‖2 where h′~=βq+(1−β)njh′~=βq+(1−β)nj and β∈(0,0.5)β∈(0,0.5).

### Parallel Augmentation

This category of approaches produce two noise versions of one anchor image and aim to learn representation such that these two augmented samples share the same embedding.

#### SimCLR

**SimCLR** ([Chen et al, 2020](https://arxiv.org/abs/2002.05709)) proposed a simple framework for contrastive learning of visual representations. It learns representations for visual inputs by maximizing agreement between differently augmented views of the same sample via a contrastive loss in the latent space.

![SimCLR](https://lilianweng.github.io/lil-log/assets/images/SimCLR.png)

*Fig. 6. A simple framework for contrastive learning of visual representations. (Image source: [Chen et al, 2020](https://arxiv.org/abs/2002.05709))*

\1) Randomly sample a minibatch of NN samples and each sample is applied with two different data augmentation operations, resulting in 2N2N augmented samples in total.

x~i=t(x),x~j=t′(x),t,t′∼Tx~i=t(x),x~j=t′(x),t,t′∼T

where two separate data augmentation operators, tt and t′t′, are sampled from the same family of augmentations TT. Data augmentation includes random crop, resize with random flip, color distortions, and Gaussian blur.

\2) Given one positive pair, other 2(N−1)2(N−1) data points are treated as negative samples. The representation is produced by a base encoder f(.)f(.):

hi=f(x~i),hj=f(x~j)hi=f(x~i),hj=f(x~j)

\3) The contrastive learning loss is defined using cosine similarity sim(.,.)sim(.,.). Note that the loss operates on an extra projection layer of the representation g(.)g(.) rather than on the representation space directly. But only the representation hh is used for downstream tasks.

ziL(i,j)SimCLR=g(hi),zj=g(hj)=−logexp(sim(zi,zj)/τ)∑2Nk=11[k≠i]exp(sim(zi,zk)/τ)zi=g(hi),zj=g(hj)LSimCLR(i,j)=−log⁡exp⁡(sim(zi,zj)/τ)∑k=12N1[k≠i]exp⁡(sim(zi,zk)/τ)

where 1[k≠i]1[k≠i] is an indicator function: 1 if k≠ik≠i 0 otherwise.

SimCLR needs a large batch size to incorporate enough negative samples to achieve good performance.

![SimCLR Algorithm](https://lilianweng.github.io/lil-log/assets/images/SimCLR-algo.png)

*Fig. 7. The algorithm for SimCLR. (Image source: [Chen et al, 2020](https://arxiv.org/abs/2002.05709)).*

#### Barlow Twins

**Barlow Twins** ([Zbontar et al. 2021](https://arxiv.org/abs/2103.03230)) feeds two distorted versions of samples into the same network to extract features and learns to make the *cross-correlation matrix* between these two groups of output features close to the identity. The goal is to keep the representation vectors of different distorted versions of one sample similar, while minimizing the redundancy between these vectors.

![Barlow twins](https://lilianweng.github.io/lil-log/assets/images/barlow-twins.png)

*Fig. 8. Illustration of Barlow Twins learning pipeline. (Image source: [Zbontar et al. 2021](https://arxiv.org/abs/2103.03230)).*

Let CC be a cross-correlation matrix computed between outputs from two identical networks along the batch dimension. CC is a square matrix with the size same as the feature network’s output dimensionality. Each entry in the matrix CijCij is the cosine similarity between network output vector dimension at index i,ji,j and batch index bb, zAb,izb,iA and zBb,jzb,jB, with a value between -1 (i.e. perfect anti-correlation) and 1 (i.e. perfect correlation).

LBTwhere Cij=∑i(1−Cii)2invariance term+λ∑i∑i≠jC2ijredundancy reduction term=∑bzAb,izBb,j∑b(zAb,i)2−−−−−−−−√∑b(zBb,j)2−−−−−−−−√LBT=∑i(1−Cii)2⏟invariance term+λ∑i∑i≠jCij2⏟redundancy reduction termwhere Cij=∑bzb,iAzb,jB∑b(zb,iA)2∑b(zb,jB)2

Barlow Twins is competitive with SOTA methods for self-supervised learning. It naturally avoids trivial constants (i.e. collapsed representations), and is robust to different training batch sizes.

![Barlow twins algo](https://lilianweng.github.io/lil-log/assets/images/barlow-twins-algo.png)

*Fig. 9. Algorithm of Barlow Twins in Pytorch style pseudo code. (Image source: [Zbontar et al. 2021](https://arxiv.org/abs/2103.03230)).*

#### BYOL

Different from the above approaches, interestingly, **BYOL** (Bootstrap Your Own Latent; [Grill, et al 2020](https://arxiv.org/abs/2006.07733)) claims to achieve a new state-of-the-art results *without using egative samples*. It relies on two neural networks, referred to as *online* and *target* networks that interact and learn from each other. The target network (parameterized by ξξ) has the same architecture as the online one (parameterized by θθ), but with polyak averaged weights, ξ←τξ+(1−τ)θξ←τξ+(1−τ)θ.

The goal is to learn a presentation yy that can be used in downstream tasks. The online network parameterized by θθ contains:

- An encoder fθfθ;
- A projector gθgθ;
- A predictor qθqθ.

The target network has the same network architecture, but with different parameter ξξ, updated by polyak averaging θθ: ξ←τξ+(1−τ)θξ←τξ+(1−τ)θ.

![BYOL](https://lilianweng.github.io/lil-log/assets/images/BYOL.png)

*Fig. 10. The model architecture of BYOL. After training, we only care about fθfθ for producing representation, y=fθ(x)y=fθ(x), and everything else is discarded. sgsg means stop gradient. (Image source: [Grill, et al 2020](https://arxiv.org/abs/2006.07733))*

Given an image xx, the BYOL loss is constructed as follows:

- Create two augmented views: v=t(x);v′=t′(x)v=t(x);v′=t′(x) with augmentations sampled t∼T,t′∼T′t∼T,t′∼T′;
- Then they are encoded into representations, yθ=fθ(v),y′=fξ(v′)yθ=fθ(v),y′=fξ(v′);
- Then they are projected into latent variables, zθ=gθ(yθ),z′=gξ(y′)zθ=gθ(yθ),z′=gξ(y′);
- The online network outputs a prediction qθ(zθ)qθ(zθ);
- Both qθ(zθ)qθ(zθ) and z′z′ are L2-normalized, giving us q¯θ(zθ)=qθ(zθ)/∥qθ(zθ)∥q¯θ(zθ)=qθ(zθ)/‖qθ(zθ)‖ and z′¯=z′/∥z′∥z′¯=z′/‖z′‖;
- The loss LBYOLθLθBYOL is MSE between L2-normalized prediction q¯θ(z)q¯θ(z) and z′¯z′¯;
- The other symmetric loss L~BYOLθL~θBYOL can be generated by switching v′v′ and vv; that is, feeding v′v′ to online network and vv to target network.
- The final loss is LBYOLθ+L~BYOLθLθBYOL+L~θBYOL and only parameters θθ are optimized.

Unlike most popular contrastive learning based approaches, BYOL does not use negative pairs. Most bootstrapping approaches rely on pseudo-labels or cluster indices, but BYOL directly boostrapps the latent representation.

It is quite interesting and surprising that *without* negative samples, BYOL still works well. Later I ran into this [post](https://untitled-ai.github.io/understanding-self-supervised-contrastive-learning.html) by Abe Fetterman & Josh Albrecht, they highlighted two surprising findings while they were trying to reproduce BYOL:

1. BYOL generally performs no better than random when *batch normalization is removed*.
2. The presence of batch normalization implicitly causes a form of contrastive learning. They believe that using negative samples is important for avoiding model collapse (i.e. what if you use all-zeros representation for every data point?). Batch normalization injects dependency on negative samples *inexplicitly* because no matter how similar a batch of inputs are, the values are re-distributed (spread out ∼N(0,1∼N(0,1) and therefore batch normalization prevents model collapse. Strongly recommend you to read the [full article](https://untitled-ai.github.io/understanding-self-supervised-contrastive-learning.html) if you are working in this area.

### Memory Bank

Computing embeddings for a large number of negative samples in every batch is extremely expensive. One common approach is to store the representation in memory to trade off data staleness for cheaper compute.

#### Instance Discrimination with Memoy Bank

**Instance contrastive learning** ([Wu et al, 2018](https://arxiv.org/abs/1805.01978v1)) pushes the class-wise supervision to the extreme by considering each instance as *a distinct class of its own*. It implies that the number of “classes” will be the same as the number of samples in the training dataset. Hence, it is unfeasible to train a softmax layer with these many heads, but instead it can be approximated by [NCE](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#nce).

![Instance contrastive learning](https://lilianweng.github.io/lil-log/assets/images/instance-level-discrimination.png)

*Fig. 11. The training pipeline of instance-level contrastive learning. The learned embedding is L2-normalized. (Image source: [Wu et al, 2018](https://arxiv.org/abs/1805.01978v1))*

Let v=fθ(x)v=fθ(x) be an embedding function to learn and the vector is normalized to have ∥v∥=1‖v‖=1. A non-parametric classifier predicts the probability of a sample vv belonging to class ii with a temperature parameter ττ:

P(C=i|v)=exp(v⊤iv/τ)∑nj=1exp(v⊤jv/τ)P(C=i|v)=exp⁡(vi⊤v/τ)∑j=1nexp⁡(vj⊤v/τ)

Instead of computing the representations for all the samples every time, they implement an **Memory Bank** for storing sample representation in the database from past iterations. Let V={vi}V={vi} be the memory bank and fi=fθ(xi)fi=fθ(xi) be the feature generated by forwarding the network. We can use the representation from the memory bank vivi instead of the feature forwarded from the network fifi when comparing pairwise similarity.

The denominator theoretically requires access to the representations of all the samples, but that is too expensive in practice. Instead we can estimate it via Monte Carlo approximation using a random subset of MM indices {jk}Mk=1{jk}k=1M.

P(i|v)=exp(v⊤fi/τ)∑Nj=1exp(v⊤jfi/τ)≃exp(v⊤fi/τ)NM∑Mk=1exp(v⊤jkfi/τ)P(i|v)=exp⁡(v⊤fi/τ)∑j=1Nexp⁡(vj⊤fi/τ)≃exp⁡(v⊤fi/τ)NM∑k=1Mexp⁡(vjk⊤fi/τ)

Because there is only one instance per class, the training is unstable and fluctuates a lot. To improve the training smoothness, they introduced an extra term for positive samples in the loss function based on the [proximal optimization method](https://web.stanford.edu/~boyd/papers/prox_algs.html). The final NCE loss objective looks like:

Linstanceh(i,v)=−EPd[logh(i,v(t−1)i)−λ∥v(t)i−v(t−1)i∥22]−MEPn[log(1−h(i,v′(t−1))]=P(i|v)P(i|v)+MPn(i) where the noise distribution is uniform Pn=1/NLinstance=−EPd[log⁡h(i,vi(t−1))−λ‖vi(t)−vi(t−1)‖22]−MEPn[log⁡(1−h(i,v′(t−1))]h(i,v)=P(i|v)P(i|v)+MPn(i) where the noise distribution is uniform Pn=1/N

where {v(t−1)}{v(t−1)} are embeddings stored in the memory bank from the previous iteration. The difference between iterations ∥v(t)i−v(t−1)i∥22‖vi(t)−vi(t−1)‖22 will gradually vanish as the learned embedding converges.

#### MoCo & MoCo-V2

**Momentum Contrast** (**MoCo**; [He et al, 2019](https://arxiv.org/abs/1911.05722)) provides a framework of unsupervised learning visual representation as a *dynamic dictionary look-up*. The dictionary is structured as a large FIFO queue of encoded representations of data samples.

Given a query sample xqxq, we get a query representation through an encoder q=fq(xq)q=fq(xq). A list of key representations {k1,k2,…}{k1,k2,…} in the dictionary are encoded by a momentum encoder ki=fk(xki)ki=fk(xik). Let’s assume among them there is a single *positive* key k+k+ in the dictionary that matches qq. In the paper, they create k+k+ using a noise copy of xqxq with different [augmentation](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#image-augmentations). Then the [InfoNCE](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#infonce) contrastive loss with temperature ττ is used over one positive and N−1N−1 negative samples:

LMoCo=−logexp(q⋅k+/τ)∑Ni=1exp(q⋅ki/τ)LMoCo=−log⁡exp⁡(q⋅k+/τ)∑i=1Nexp⁡(q⋅ki/τ)

Compared to the [memory bank](https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html#instance-discrimination-with-memoy-bank), a queue-based dictionary in MoCo enables us to reuse representations of immediately preceding mini-batches of data.

The MoCo dictionary is not differentiable as a queue, so we cannot rely on back-propagation to update the key encoder fkfk. One naive way might be to use the same encoder for both fqfq and fkfk. Differently, MoCo proposed to use a momentum-based update with a momentum coefficient m∈[0,1)m∈[0,1). Say, the parameters of fqfq and fkfk are labeled as θqθq and θkθk, respectively.

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

------

Cited as:

```
@article{weng2021contrastive,
  title   = "Contrastive Representation Learning",
  author  = "Weng, Lilian",
  journal = "lilianweng.github.io/lil-log",
  year    = "2021",
  url     = "https://lilianweng.github.io/lil-log/2021/05/31/contrastive-representation-learning.html"
}
```

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