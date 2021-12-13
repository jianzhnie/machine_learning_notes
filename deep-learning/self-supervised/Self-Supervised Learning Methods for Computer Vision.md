# Self-Supervised Learning Methods for Computer Vision

Self-supervised Learning is an unsupervised learning method where the supervised learning task is created out of the unlabelled input data.

This task could be as simple as given the upper-half of the image, predict the lower-half of the same image, or given the grayscale version of the colored image, predict the RGB channels of the same image, etc.

![Learn from yourself](https://miro.medium.com/max/60/1*lavhAToKGO9Bo4j3U7n4TA.jpeg?q=20)

![Learn from yourself](https://miro.medium.com/max/1120/1*lavhAToKGO9Bo4j3U7n4TA.jpeg)

Image by Author

**Why Self-supervised Learning?**

Supervised learning requires usually a lot of labelled data. Getting good quality labelled data is an expensive and time-consuming task specially for a complex task such as object detection, instance segmentation where more detailed annotations are desired. On the other hand, the unlabelled data is readily available in abundance. The motivation behind Self-supervised learning is to learn useful representations of the data from unlabelled pool of data using self-supervision first and then fine-tune the representations with few labels for the supervised downstream task. The downstream task could be as simple as image classification or complex task such as semantic segmentation, object detection, etc.

Lately, in natural language processing, Transformer models have achieved a lot of success. Transformers like Bert[1], T5[2], etc. applied the idea of self-supervision to NLP tasks. They first train the model with large unlabelled data and then fine-tuning the model with few labelled data examples. Similar self-supervised learning methods have been researched for computer vision as well and in this post, I will try to cover a few of those.

The fundamental idea for self-supervised learning is to create some auxiliary pre-text task for the model from the input data itself such that while solving the auxiliary task, the model learns the underlying structure of the data(for instance the structure of the object in case of image data). Many self-supervised learning methods have been researched but contrastive learning methods seem to be work better than others for computer vision, hence in this post, I would concentrate on contrastive learning-based self-supervised learning methods

**What is Contrastive Learning?**

suppose we have a function f(represented by any deep network Resnet50 for example), given an input x, it gives us the features f(x) as output.

Contrastive Learning states that for any positive pairs x1 and x2, the respective outputs f(x1) and f(x2) should be similar to each other and for a negative input x3, f(x1) and f(x2) both should be dissimilar to f(x3).

![Contrastive Learning](https://miro.medium.com/max/60/1*fdAU4VJtnclv0rGrfzUu4g.png?q=20)

![Contrastive Learning](https://miro.medium.com/max/1120/1*fdAU4VJtnclv0rGrfzUu4g.png)

Contrastive Learning Idea (Image by Author)

The positive pair could be two crops of same image(lets say top-left and bottom right), two frames of same video file, two augmented views(horizontally flipped version for instance) of same image, etc. and respective negatives could be a crop from different image, frame from different video, augmented view of different image, etc.

The idea of contrastive learning was first introduced in this paper “[Representation learning with contrastive predictive coding](https://arxiv.org/abs/1807.03748)”[3] by Aaron van den Oord et al. from DeepMind. The formulated contrastive learning task gave a strong basis for learning useful representations of the image data which is described next.

# Contrastive Predictive Coding(CPC)

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

InfoNCE loss[4]

Here q is the network prediction, k+ is the positive patch(correct patch) and k- represents a set of N-1 negative patches. Note that k+, k- and q, all are in representation space i.e. output of g_enc and not into original image space.

In simple terms, the formula is equivalent to the log_softmax function. To calculate the similarity, the dot product is used. Take a dot product of all N samples with the prediction q and then calculate the log of softmax of the similarity score of the positive sample with the prediction q.

In order to validate the richness of the representations learnt by CPC, a **linear evaluation protocol** is used. A linear classifier is trained on top of the output of the frozen encoder model(g_enc) using the Imagenet dataset and then it is evaluated for the classification accuracy of the learnt classifier model on the Imagenet Val/Test set. Note that during this whole training process of the linear classifier, the backbone model(g_enc) is fixed and is not trained at all. The table below shows that the classification accuracy of CPC representations outperformed all the other methods introduced before CPC with 48.7% top-1 acc.

![img](https://miro.medium.com/max/60/1*eV_ro_d1GgwCmed3SWgGJg.png?q=20)

![img](https://miro.medium.com/max/835/1*eV_ro_d1GgwCmed3SWgGJg.png)

Imagenet Top-1% Accuracy of the Linear Classifier Trained on top of CPC representations[3]

Although CPC outperformed other unsupervised learning methods for representation learning, the classification accuracy was still very far from the supervised counterpart(Resnet-50 with 100% labels on the Imagenet has 76.5% top-1 accuracy). This idea of image crop discrimination was extended to instance discrimination and tightened the gap between self-supervised learning and supervised learning methods.

# **Instance Discrimination Methods**

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

# SimCLR

SimCLR considers all the images in the current batch as negative samples. Trained in this way, SimCLR representations achieve a top-1% accuracy of 69.3% on the Imagenet with the linear evaluation protocol described in the CPC section.

![img](https://miro.medium.com/max/60/1*-804ozRWp0Mmmd0LvZIN1A.png?q=20)

![img](https://miro.medium.com/max/998/1*-804ozRWp0Mmmd0LvZIN1A.png)

SimCLR linear classifier result on Imagenet[5]

In practice, InfoNCE loss performance is dependent upon the number of negatives and it requires a high number of negatives while calculating the loss term. Hence, simCLR is trained with a high number of batches(as big as 8k) for best results which are very computationally demanding and require multi-GPU training. This is considered as the main drawback of simCLR method.

# Momentum Contrast(MoCo)

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

# Bootstrap your own Latent(BYOL)

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

1. Devlin, Jacob, et al. “Bert: Pre-training of deep bidirectional transformers for language understanding.” *arXiv preprint arXiv:1810.04805* (2018).
2. Raffel, Colin, et al. “Exploring the limits of transfer learning with a unified text-to-text transformer.” *arXiv preprint arXiv:1910.10683* (2019).
3. Oord, Aaron van den, Yazhe Li, and Oriol Vinyals. “Representation learning with contrastive predictive coding.” *arXiv preprint arXiv:1807.03748* (2018).
4. He, Kaiming, et al. “Momentum contrast for unsupervised visual representation learning.” *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2020.
5. Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. *arXiv preprint arXiv:2002.05709*.
6. Chen, X., Fan, H., Girshick, R., & He, K. (2020). Improved baselines with momentum contrastive learning. *arXiv preprint arXiv:2003.04297*
7. Grill, Jean-Bastien, et al. “Bootstrap your own latent-a new approach to self-supervised learning.” *Advances in Neural Information Processing Systems* 33 (2020).