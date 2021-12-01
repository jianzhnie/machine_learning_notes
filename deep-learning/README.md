# 深度学习论文

（这里引用采用的是 semanticscholar，是因为它提供 API 可以自动获取，不用手动更新。）

### 计算机视觉 - CNN


| 已录制 | 年份 | 名字                                                         | 简介                 | 引用                                                         |
| ------ | ---- | ------------------------------------------------------------ | -------------------- | -----------------------------------------------------------: |
| ✅      | 2012 | [AlexNet](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) | 深度学习热潮的奠基作           | 73342 ([link](https://www.semanticscholar.org/paper/ImageNet-classification-with-deep-convolutional-Krizhevsky-Sutskever/abd1c342495432171beb7ca8fd9551ef13cbd0ff)) |
| | 2014 | [VGG](https://arxiv.org/pdf/1409.1556.pdf) | 使用 3x3 卷积构造更深的网络           | 55856 ([link](https://www.semanticscholar.org/paper/Very-Deep-Convolutional-Networks-for-Large-Scale-Simonyan-Zisserman/eb42cf88027de515750f230b23b1a057dc782108)) |
| | 2014 | [GoogleNet](https://arxiv.org/pdf/1409.4842.pdf) | 使用并行架构构造更深的网络           | 26878 ([link](https://www.semanticscholar.org/paper/Going-deeper-with-convolutions-Szegedy-Liu/e15cf50aa89fee8535703b9f9512fca5bfc43327)) |
|  ✅  | 2015 |  [ResNet](https://arxiv.org/pdf/1512.03385.pdf) | 构建深层网络都要有的残差连接。       | 80816 ([link](https://www.semanticscholar.org/paper/Deep-Residual-Learning-for-Image-Recognition-He-Zhang/2c03df8b48bf3fa39054345bafabfeff15bfd11d)) |
|  | 2017 | [MobileNet](https://arxiv.org/pdf/1704.04861.pdf) | 适合终端设备的小CNN           | 8695 ([link](https://www.semanticscholar.org/paper/MobileNets%3A-Efficient-Convolutional-Neural-Networks-Howard-Zhu/3647d6d0f151dc05626449ee09cc7bce55be497e)) |
| | 2019 | [EfficientNet](https://arxiv.org/pdf/1905.11946.pdf) | 通过架构搜索得到的CNN           | 3426 ([link](https://www.semanticscholar.org/paper/EfficientNet%3A-Rethinking-Model-Scaling-for-Neural-Tan-Le/4f2eda8077dc7a69bb2b4e0a1a086cf054adb3f9)) |
| | 2019 | [MoCo](https://arxiv.org/pdf/1911.05722.pdf) | 无监督训练效果也很好           | 2011 ([link](https://www.semanticscholar.org/paper/Momentum-Contrast-for-Unsupervised-Visual-Learning-He-Fan/ec46830a4b275fd01d4de82bffcabe6da086128f)) |
| | 2021 |  [Non-deep networks](https://arxiv.org/pdf/2110.07641.pdf) | 让不深的网络也能在ImageNet刷到SOTA           | 0 ([link](https://www.semanticscholar.org/paper/Non-deep-Networks-Goyal-Bochkovskiy/0d7f6086772079bc3e243b7b375a9ca1a517ba8b)) |

### 计算机视觉 - Transformer

| 已录制 | 年份 | 名字                                                         | 简介                 | 引用                                                         |
| ------ | ---- | ------------------------------------------------------------ | -------------------- | -----------------------------------------------------------: |
| ✅ | 2020 | [ViT](https://arxiv.org/pdf/2010.11929.pdf) | Transformer杀入CV界           | 1527 ([link](https://www.semanticscholar.org/paper/An-Image-is-Worth-16x16-Words%3A-Transformers-for-at-Dosovitskiy-Beyer/7b15fa1b8d413fbe14ef7a97f651f47f5aff3903)) |
| | 2021 |  [CLIP](https://openai.com/blog/clip/) | 图片和文本之间的对比学习           | 399 ([link](https://www.semanticscholar.org/paper/Learning-Transferable-Visual-Models-From-Natural-Radford-Kim/6f870f7f02a8c59c3e23f407f3ef00dd1dcf8fc4)) |
| | 2021 | [Swin Transformer](https://arxiv.org/pdf/2103.14030.pdf) | 多层次的Vision Transformer           | 384 ([link](https://www.semanticscholar.org/paper/Swin-Transformer%3A-Hierarchical-Vision-Transformer-Liu-Lin/c8b25fab5608c3e033d34b4483ec47e68ba109b7)) |
| | 2021 | [MLP-Mixer](https://arxiv.org/pdf/2105.01601.pdf) | 使用MLP替换self-attention    | 137 ([link](https://www.semanticscholar.org/paper/MLP-Mixer%3A-An-all-MLP-Architecture-for-Vision-Tolstikhin-Houlsby/2def61f556f9a5576ace08911496b7c7e4f970a4)) |
| | 2021 | [MAE](https://arxiv.org/pdf/2111.06377.pdf) | BERT的CV版     | 4 ([link](https://www.semanticscholar.org/paper/Masked-Autoencoders-Are-Scalable-Vision-Learners-He-Chen/c1962a8cf364595ed2838a097e9aa7cd159d3118)) |

### 计算机视觉 - GAN

| 已录制 | 年份 | 名字                                              | 简介         |                                                         引用 |
| ------ | ---- | ------------------------------------------------- | ------------ | -----------------------------------------------------------: |
|  ✅ | 2014 | [GAN](https://papers.nips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf) | 生成模型的开创工作           | 26024 ([link](https://www.semanticscholar.org/paper/Generative-Adversarial-Nets-Goodfellow-Pouget-Abadie/54e325aee6b2d476bbbb88615ac15e251c6e8214)) |
|  | 2015 | [DCGAN](https://arxiv.org/pdf/1511.06434.pdf) | 使用CNN的GAN  | 9022 ([link](https://www.semanticscholar.org/paper/Unsupervised-Representation-Learning-with-Deep-Radford-Metz/8388f1be26329fa45e5807e968a641ce170ea078)) |
|  | 2016 | [pix2pix](https://arxiv.org/pdf/1611.07004.pdf) |   | 9752 ([link](https://www.semanticscholar.org/paper/Image-to-Image-Translation-with-Conditional-Isola-Zhu/8acbe90d5b852dadea7810345451a99608ee54c7)) |
|  | 2016 | [SRGAN](https://arxiv.org/pdf/1609.04802.pdf) | 图片超分辨率  | 5524 ([link](https://www.semanticscholar.org/paper/Photo-Realistic-Single-Image-Super-Resolution-Using-Ledig-Theis/df0c54fe61f0ffb9f0e36a17c2038d9a1964cba3)) |
|  | 2017 | [WGAN](https://arxiv.org/abs/1701.07875) | 训练更加容易  | 2620 ([link](https://www.semanticscholar.org/paper/Wasserstein-GAN-Arjovsky-Chintala/2f85b7376769473d2bed56f855f115e23d727094)) |
|  | 2017 | [CycleGAN](https://arxiv.org/abs/1703.10593) |   | 3401 ([link](https://www.semanticscholar.org/paper/Unpaired-Image-to-Image-Translation-Using-Networks-Zhu-Park/c43d954cf8133e6254499f3d68e45218067e4941)) |
|  | 2019 | [StyleGAN](https://arxiv.org/abs/1812.04948) |   | 2708 ([link](https://www.semanticscholar.org/paper/A-Style-Based-Generator-Architecture-for-Generative-Karras-Laine/ceb2ebef0b41e31c1a21b28c2734123900c005e2)) |

### 计算机视觉 - Object Detection

| 已录制 | 年份 | 名字                                              | 简介         |                                                         引用 |
| ------ | ---- | ------------------------------------------------- | ------------ | -----------------------------------------------------------: |
|        | 2014 | [R-CNN](https://arxiv.org/pdf/1311.2524v5.pdf)    | Two-stage     | 15545 ([link](https://www.semanticscholar.org/paper/2f4df08d9072fc2ac181b7fced6a245315ce05c8)) |
|        | 2015 | [Fast R-CNN](http://arxiv.org/abs/1504.08083v2)   |               | 12578 ([link](https://www.semanticscholar.org/paper/7ffdbc358b63378f07311e883dddacc9faeeaf4b)) |
|        | 2015 | [Faster R-CNN](http://arxiv.org/abs/1506.01497v3) |               | 28396 ([link](https://www.semanticscholar.org/paper/424561d8585ff8ebce7d5d07de8dbf7aae5e7270)) |
|        | 2016 | [SSD](http://arxiv.org/abs/1512.02325v5)          | Single stage  | 13449 ([link](https://www.semanticscholar.org/paper/4d7a9197433acbfb24ef0e9d0f33ed1699e4a5b0)) |
|        | 2016 | [YOLO](http://arxiv.org/abs/1506.02640v5)         |               | 14099 ([link](https://www.semanticscholar.org/paper/f8e79ac0ea341056ef20f2616628b3e964764cfd)) |
|        | 2017 | [Mask R-CNN](http://arxiv.org/abs/1703.06870v3)   |               | 3580 ([link](https://www.semanticscholar.org/paper/ea99a5535388196d0d44be5b4d7dd02029a43bb2)) |
|        | 2017 | [YOLOv2](http://arxiv.org/abs/1612.08242v1)       |               | 6915 ([link](https://www.semanticscholar.org/paper/7d39d69b23424446f0400ef603b2e3e22d0309d6)) |
|        | 2018 | [YOLOv3](http://arxiv.org/abs/1804.02767v1)       |               | 7002 ([link](https://www.semanticscholar.org/paper/e4845fb1e624965d4f036d7fd32e8dcdd2408148)) |
|        | 2019 | [CentorNet](https://arxiv.org/pdf/1904.07850.pdf) | Anchor free   | 773 ([link](https://www.semanticscholar.org/paper/Objects-as-Points-Zhou-Wang/6a2e2fd1b5bb11224daef98b3fb6d029f68a73f2)) |
|        | 2020 | [DETR](https://arxiv.org/pdf/2005.12872.pdf)      | Transformer   | 1053 ([link](https://www.semanticscholar.org/paper/End-to-End-Object-Detection-with-Transformers-Carion-Massa/962dc29fdc3fbdc5930a10aba114050b82fe5a3e)) |


### 自然语言处理 - Transformer

| 已录制 | 年份 | 名字                                                         | 简介                 | 引用                                                         |
| ------ | ---- | ------------------------------------------------------------ | -------------------- | -----------------------------------------------------------: |
| ✅ | 2017 | [Transformer](https://arxiv.org/abs/1706.03762) | 继MLP、CNN、RNN后的第四大类架构           | 26029 ([link](https://www.semanticscholar.org/paper/Attention-is-All-you-Need-Vaswani-Shazeer/204e3073870fae3d05bcbc2f6a8e263d9b72e776)) |
| | 2018 | [GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) | 使用 Transformer 来做预训练       | 2752 ([link](https://www.semanticscholar.org/paper/Improving-Language-Understanding-by-Generative-Radford-Narasimhan/cd18800a0fe0b668a1cc19f2ec95b5003d0a5035)) |
| ✅ | 2018 | [BERT](https://arxiv.org/abs/1810.04805) | Transformer一统NLP的开始          | 25340 ([link](https://www.semanticscholar.org/paper/BERT%3A-Pre-training-of-Deep-Bidirectional-for-Devlin-Chang/df2b0e26d0599ce3e70df8a9da02e51594e0e992)) |
|  | 2019 | [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)  |       | 4534 ([link](https://www.semanticscholar.org/paper/Language-Models-are-Unsupervised-Multitask-Learners-Radford-Wu/9405cc0d6169988371b2755e573cc28650d14dfe)) |
|  | 2020 |  [GPT-3](https://arxiv.org/abs/2005.14165) | 朝着zero-shot learning迈了一大步          | 2548 ([link](https://www.semanticscholar.org/paper/Language-Models-are-Few-Shot-Learners-Brown-Mann/6b85b63579a916f705a8e10a49bd8d849d91b1fc)) |


### 通用技术

| 已录制 | 年份 | 名字                                                         | 简介                 | 引用                                                         |
| ------ | ---- | ------------------------------------------------------------ | -------------------- | -----------------------------------------------------------: |
| | 2014 | [Adam](https://arxiv.org/abs/1412.6980) | 深度学习里最常用的优化算法之一           | 77401 ([link](https://www.semanticscholar.org/paper/Adam%3A-A-Method-for-Stochastic-Optimization-Kingma-Ba/a6cb366736791bcccc5c8639de5a8f9636bf87e8)) |
| | 2016 |  [为什么超大的模型泛化性不错](https://arxiv.org/abs/1611.03530)   |       | 3112 ([link](https://www.semanticscholar.org/paper/Understanding-deep-learning-requires-rethinking-Zhang-Bengio/54ddb00fa691728944fd8becea90a373d21597cf)) |
| | 2017 | [为什么Momentum有效](https://distill.pub/2017/momentum/) | Distill的可视化介绍    | 116 ([link](https://www.semanticscholar.org/paper/Why-Momentum-Really-Works-Goh/3e8ccf9d3d843c9855c5d76ab66d3e775384da72)) |

### 其他领域

| 已录制 | 年份 | 名字                                                         | 简介                 | 引用                                                         |
| ------ | ---- | ------------------------------------------------------------ | -------------------- | -----------------------------------------------------------: |
|  | 2014 | [Two-stream networks](https://arxiv.org/abs/1406.2199) | 首次超越手工特征的视频分类架构           | 5093 ([link](https://www.semanticscholar.org/paper/Two-Stream-Convolutional-Networks-for-Action-in-Simonyan-Zisserman/67dccc9a856b60bdc4d058d83657a089b8ad4486)) |
| | 2016 | [AlphaGo](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf) | 强化学习出圈         | 10257 ([link](https://www.semanticscholar.org/paper/Mastering-the-game-of-Go-with-deep-neural-networks-Silver-Huang/846aedd869a00c09b40f1f1f35673cb22bc87490)) |
|  ✅ |  2021 | [图神经网络介绍](https://distill.pub/2021/gnn-intro/) | GNN的可视化介绍          | 4 ([link](https://www.semanticscholar.org/paper/A-Gentle-Introduction-to-Graph-Neural-Networks-S%C3%A1nchez-Lengeling-Reif/2c0e0440882a42be752268d0b64243243d752a74)) |

TODO：

1. Out-of-distribution
1. AlphaFold
1. Anchor-free object detection
1. Knowledge graph