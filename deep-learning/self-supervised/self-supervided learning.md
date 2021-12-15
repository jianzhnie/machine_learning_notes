##  self-supervided learning

最近，发表的文章也显示SSL得到的模型表现在ImageNet dataset上逐渐逼近了传统Supervised Learning的表现了。
在实际应用中（例如业界中已经部署的模型），Self Supervised Learning 未必能直接有效的提升Performance，但阅读这些文章还是能得到不少启发。例如我们能对以下Supervised Learning问题有更多想法：

- 如果要Deep Network学习到有用的信息，人工标记（Manual-Label）是必要的吗？
- 数据（Data）本身带有的信息是否比标记（Label）更加丰富？
- 我们能将每张图视为一个类别（Class）；甚至每一个Pixel都视为一个类别吗？

以上问题可能有点天马行空，如果在实际应用上我们能思考：

- 在Representation Learning中，如何能等价的增大Batch Size？如何能维持Embedding Space的稳定性？
- 在Deep Network一定是最后一层具有最丰富的Representation吗？
- 听说Deep Network的Capacity很强大，但时至今日，我们是否已经达到Model能负荷的上限？（例如ResNet-50有24M个参数，号称拥有`大数据`的人们，是否已经触碰到Effective Upper-Bound of ResNet-50's Model Complexity？）
- 如果Model Complexity远超乎我们想像，那什么样的Training Procedure能最有效率的将信息储存于Deep Network中？

- Data Augmentation是学习Deep Learning一定会接触到的方法，它只是一个方便Training的Trick呢？还是他对Network有特殊意义？
这些问题目前没人能给出确切答案，但在接下来的文章中必然能带来更多想法与启发。

# Before SSL Revolution: Pretext Task

早期在探索 SSL 的想法相對單純，如果沒有得到 Label ，就利用 Rule-Based 的方法產生一些 Label 。比較著名的方法為

- Rotation
- Jigsaw Puzzle
- Colorization

“Unsupervised Representation Learning by Predicting Image Rotations” 一文中提出 Rotation 可謂是 SSL 之濫觴。將給定的 Image 旋轉固定的 0, 90, 180, 270 四種角度，讓 Network 預測看到的圖是原圖被轉了幾度後的結果。

![img](https://miro.medium.com/max/60/1*eOCvCMQCSEGWoeKBPRFBNg.png?q=20)

![img](https://miro.medium.com/max/1400/1*eOCvCMQCSEGWoeKBPRFBNg.png)

Unsupervised Representation Learning by Predicting Image Rotations https://arxiv.org/pdf/1803.07728.pdf

乍聽之下，預測旋轉角度挺沒意思的，但是如果 Training Data 是整個 ImageNet 時，這樣的任務就變成相當有趣了。Network 必須了解了什麼是物件後，才能了解旋轉。這也貫策了 SSL 的基本想法：Knowledge Before Task.

![img](https://miro.medium.com/max/60/1*Eobu7VnaRjET_Kk0-iU7aw.png?q=20)

![img](https://miro.medium.com/max/1400/1*Eobu7VnaRjET_Kk0-iU7aw.png)

Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles https://arxiv.org/abs/1603.09246

![img](https://miro.medium.com/max/60/1*QZKTvtlBi4yuhJa_3dF4Yg.png?q=20)

![img](https://miro.medium.com/max/1400/1*QZKTvtlBi4yuhJa_3dF4Yg.png)

Tracking Emerges by Colorizing Videos

後來的 Jigsaw 與 Colorization 基本上延續了這種探索 Image 本質的想法，設計出能幫助理解 Content 的輔助任務，因此這類方法統稱為 Pretext Task (委託任務)。其中 Colorization 這個方法，能延伸出 Tracking 的效果，著實讓人震驚。

# Huge Improvement: Contrastive Learning

## “Data-Centric” Loss

另一派做法是從設計 Loss Function 下手，找出能探索資料本身性質的 Loss。

其中 Energy-Based Model 算是 LeCun 從以前到現在不斷推廣的，這種模型與具有強烈的物理意義，可能會是明日之星；而這篇文章整理的是另一種以 Mutual Information 為主的 Loss Function。

## AutoRegressive Model

說到理解資料，最直觀的機率模型就是建立 Data Likelihood，而在 Computer Vision 的領域中，最強大的莫過於 PixelCNN 這種 Pixel-by-Pixel 的建模方法了。利用 Chain Rule ，Network 能夠完整 Encode 所有資訊。

![img](https://miro.medium.com/max/60/1*f2gbhX1iTDl6Q8IAFZi47Q.png?q=20)

![img](https://miro.medium.com/max/1400/1*f2gbhX1iTDl6Q8IAFZi47Q.png)

[Conditional Image Generation with PixelCNN Decoders](https://papers.nips.cc/paper/6527-conditional-image-generation-with-pixelcnn-decoders.pdf)

但是 Representation Learning 在乎的並不是整個 Data Distribution; 而是怎麼得到更抽象、High-level 的表示法。End-to-End Training 讓人們發現，Deep Neural Network 是有能力解構資料的 Hierarchical Internal Representation 的，何不利用這種強大的能力呢？

也就是 **Learn the dataset, not the data points**.

## InfoNCE

DeepMind 在 2017 年 (https://arxiv.org/abs/1807.03748) 提出一種基於 Mutual Information 修改 AutoRegrssive 的 Loss Function，稱為 InfoNCE。

![img](https://miro.medium.com/max/60/1*c6KwTTF2eRTznJrcSpUdIw.png?q=20)

![img](https://miro.medium.com/max/1400/1*c6KwTTF2eRTznJrcSpUdIw.png)

https://arxiv.org/abs/1807.03748

從圖上說明是相當直觀的，模型基於看過的 Data 提取 Context (也就是 Feature) 並且對未來做預測。並且 Loss Function 的目標是讓 Data 與 Context 的 Mutual Information 越大越好。

![img](https://miro.medium.com/max/60/1*iU830SUntaAF2e2KHp-8JQ.png?q=20)

![img](https://miro.medium.com/max/1400/1*iU830SUntaAF2e2KHp-8JQ.png)

Mutual Information 是廣義上的 Correlation Function。(當我們完全不了解系統的 Dynamics 或更深入的行為時，Mutual Information 依舊能作為估算) 它量化了我們能從 Context 中得到多少 Data 的資訊，稱為 Data 與 Context 之間的 Mutual Information。

![img](https://miro.medium.com/max/60/1*fS_DOP7CJrhF5hbsrBlW_A.png?q=20)

![img](https://miro.medium.com/max/1400/1*fS_DOP7CJrhF5hbsrBlW_A.png)

首先，為了最大化 Mutual Information 讓 Network Model Distribution Ratio (而不像 generative model 是單純 model distribution)；並且用簡單的 Linear Map 作為從 Context 到 Future Data 的預測函數。

InfoNCE 寫法如下。其實他與常用的 Cross Entropy Loss 並沒有太大區別，差異只在於這個 Loss 並不是用於分類，而是注重在對資料的本身做預測。如果用先前 Time Series 的例子就是預測未來。

![img](https://miro.medium.com/max/60/1*D7ufCqXQ7DWa4rTxexQIxw.png?q=20)

![img](https://miro.medium.com/max/1400/1*D7ufCqXQ7DWa4rTxexQIxw.png)

[Learning Deep Representations of Fine-grained Visual Descriptions](https://arxiv.org/abs/1605.05395)

我們把所唯一正確的預測稱為 Positive Sample; 其他的預測通通為 Negative Samples。文章接下來都使用 Contrastive Loss 來表示這種 Training 方法。

另外 InfoNCE 有一個 Weak Lower-Bound 在描述 N 的重要，也就是越多的 Negative Samples 時，Loss Function 越能控制 Mutual Information ，並且是以 Log 的方式 Scale (這給了 Metric Learning 一點 Hint, Batch Size 可能隨著 log scale)。

## CPC: Contrastive Predictive Coding

第一個成功在 Image Classification 實踐出 InfoNCE 的是 CPC 這篇文章 (基本上是 DeepMind 同一個 team 的作品)。很直觀的利用在圖片上裁切 Patch 的方式，做出 Positive & Negative samples，實現 Contrastive Loss 。

![img](https://miro.medium.com/max/60/1*Xrj-eSRDPxRjG0SVrWrGdg.png?q=20)

![img](https://miro.medium.com/max/1400/1*Xrj-eSRDPxRjG0SVrWrGdg.png)

[Data-Efficient Image Recognition with Contrastive Predictive Coding](https://arxiv.org/abs/1905.09272)

這邊用到了三個 Network，分別是 feature extractor, context prediction 跟 downstream task network。這是因問 SSL 的 evaluation 方式不同的關係，這邊簡單說明一下。

![img](https://miro.medium.com/max/60/1*rI7MQMbTsKHAarwTXWAGjA.png?q=20)

![img](https://miro.medium.com/max/1400/1*rI7MQMbTsKHAarwTXWAGjA.png)

SSL 訓練出來的模型基本上不能直接使用，通常只能作為很強的 Pretrained Model。 因此要評估 Pretrained Model 好壞通常做 Linear Evaluation ，Fine-tune 一個 Linear Classifier 看能達到多少的準確度 (為了公平，通常這個 classifier 會用 grid search 得到)。

研究後來發現， SSL Pretrained Model 不僅得到 Linear Separable 的 Feature Space; 並且這些 Feature 是很豐富的，因為只需要少量的 Data 就可以達到很好的效果，這稱為 Efficient Classification Evaluation。像常常會測試，拿 ImageNet (有 1000 類 一千四百萬張圖片) 1% 的資料量 (也就是每個類別 Randomly choose 12 張圖片) 來訓練。這種 Evaluation 凸顯出 Feature 是能廣泛描述各種類別的，因此只要取少少的 Samples 就可以達到效果。

第三種 Evaluation 就是將 Pretrained Model 運用在各種 Vision Task 上，例如拿到 Object Detection 或 Segmentation 任務上依舊能表現不錯。

![img](https://miro.medium.com/max/60/1*B7HlDIH7DHQYTTP5BL3rsg.png?q=20)

![img](https://miro.medium.com/max/1400/1*B7HlDIH7DHQYTTP5BL3rsg.png)

回到 CPC 這篇文章，ResNet-50 Linear Protocol 能達到 Top-1 71.5% 的準確率；在 Efficient Classification Protocol 上 ，能比原本 Supervised Learning 的方式省掉至少 50% ~ 80% 的資料(這邊是參數更多的 ResNet)。意味著透過 SSL Pretrained Model，我能夠少一些人工標記一樣能達到原本 Supervised Learning 的準確度。

![img](https://miro.medium.com/max/60/1*2LCysmeOI6pfSSWPWAnOBQ.png?q=20)

![img](https://miro.medium.com/max/1400/1*2LCysmeOI6pfSSWPWAnOBQ.png)

## What Important?

CPC 帶來巨大的好處，但什麼事情是重要的？難道以後都需要將一張圖切很多 Patch 來預測嗎？並不盡然。

在 CMC 這邊文章中表明了，使用不同場景 (View Point, Depth, Color Space) 來計算 Contrastive Loss 能達到非常好的效果，因此 Contrastive 本身 (也就是辨認 Positive & Negative Sample 之間的 Consistency) 才是關鍵。

![img](https://miro.medium.com/max/60/1*3vb6xn4jYbT9r2b3LM9ZIg.png?q=20)

![img](https://miro.medium.com/max/1400/1*3vb6xn4jYbT9r2b3LM9ZIg.png)

[Contrastive Multiview Coding](https://arxiv.org/abs/1906.05849)

另外 Google 做了大規模的定性實驗，找出了幾個對 Visual Representation 最有影響的因子，因為篇幅關係就節錄下列重點

- Pretext Task 不一定能在 Downstream Task 上達到好的效果
- ResNet 的 skip-connection 能防止 feature quality 下降
- 增大 Model Size 與增加 Embedding Dimension 能有效提升 Performance

![img](https://miro.medium.com/max/60/1*a5t2hYWMgwradO1hRI2tBg.png?q=20)

![img](https://miro.medium.com/max/1400/1*a5t2hYWMgwradO1hRI2tBg.png)

[Revisiting Self-Supervised Visual Representation Learning](https://arxiv.org/abs/1901.09005)

到目前為止基本上定調了 SSL 的走向

1. Contrastive Learning 能從 Data 中獲得相當豐富的資訊，不需要拘泥在 Patch 上
2. 使用 ResNet 這種 Backbone (而非早期 paper 強調VGG 能得到更好的 representation)

接下來的文章，都基於這樣的前提來 Improve 。

## MoCo: Momentum Contrast

這篇 MoCo 是 Kaiming He 在 FAIR (又是與 RGB 一起) 第一次對 SSL 問題提出的文章。算是一個相當 Engineering 的解法，來有效增加 Batch Size ，提升 Performance。

![img](https://miro.medium.com/max/60/1*-yOt4YwRpNxy4unemT21Rw.png?q=20)

![img](https://miro.medium.com/max/1400/1*-yOt4YwRpNxy4unemT21Rw.png)

[Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722)

首先，我們可以完全忘掉過去 AutoRegressive 預測未來的觀點；或切 Patch 預測圖片結構。 MoCo 完全專注在 Contrastive Loss 上，將這個問題想像成有一個很大的 Dictionary ，Network 的目的就是一個 Encoder 要將圖片 Encode 成唯一的一把 Key ，此時要如何做到讓 Key Space Large and Consistent 是最重要的。

首先借鑒了另一篇 SSL 的文章 Memory Bank ，建一個 Bank 來存下所有的 Key （或稱為 Feature) 。這個方法相對把所有圖塞進 Batch 少用很多記憶體，但對於很大的 Dataset 依舊難以 Scale Up。

![img](https://miro.medium.com/max/60/1*_UUAF2ilNvvkIjTgkBlT0g.png?q=20)

![img](https://miro.medium.com/max/1400/1*_UUAF2ilNvvkIjTgkBlT0g.png)

[Unsupervised Feature Learning via Non-Parametric Instance-level Discrimination](https://arxiv.org/abs/1805.01978)

MoCo 改善了 Bank ，用一個 Dynamic Queue 來取代，但是單純這樣做的話是行不通的，因為每次個 Key 會受到 Network 改變太多，Contrastive Loss 無法收斂。因此 MoCo 將 feature extractor 拆成兩個獨立的 Network: Encoder 與 Momentum Encoder。

![img](https://miro.medium.com/max/52/1*eSvccPz9z2zUbd255y4HMA.png?q=20)

![img](https://miro.medium.com/max/1110/1*eSvccPz9z2zUbd255y4HMA.png)

MoCo Algorithm

我們可以想像成這樣的情境，Momentum Encoder 隨著 Training Update 很慢，因此能提供很穩定的 Key ，也就是 Momentum Encoder 把這個 Key Space 先擺好; 當新的 Positive 經過 Encoder 進來時，跟所有其他的 Negative Sample 計算 Similarity ，如果 New Key 與原本的 Key 太近容易混淆，這時候的 Loss 絕大部分會 Update 給 Encoder (相當於找一個比較空的區域放 Key, 而不影響原本的其他 Key)。

![img](https://miro.medium.com/max/60/1*7m6jpsQfNRKyrCDuYls73A.png?q=20)

![img](https://miro.medium.com/max/1174/1*7m6jpsQfNRKyrCDuYls73A.png)

等 Encoder Update 完後，在用 Momentum Update Slow Encoder。並將這次的 Batch 放進 Dynamic Queue 中。

從以下實驗可以看到，MoCo 的表現幾乎與暴力將 Batch Size 增大得到的效果一樣，但是 Batch Size 沒辦法 Scale Up； Memory Bank 與 MoCo 有著一樣的 Scaling Property，但 MoCo 的 Momentum Update 能提供穩定的 Key Space 讓整體 Performance 可以提升約 2%。

![img](https://miro.medium.com/max/60/1*OuYHo2eZyXI9sckQsM4A-w.png?q=20)

![img](https://miro.medium.com/max/1400/1*OuYHo2eZyXI9sckQsM4A-w.png)

## SimCLR: Simple Framework for Contrastive Learning

在 MoCo 之後發出來的論文是出自今年 (2020) 產量很高的 Hinton 之手。

![img](https://miro.medium.com/max/60/1*5F3CVyVQll94xu2JLY2IYw.png?q=20)

![img](https://miro.medium.com/max/1050/1*5F3CVyVQll94xu2JLY2IYw.png)

Hinton 今年 (2020/03 之前) 發出了三篇有趣的文章

並且達到 SSL 目前 Linear Protocol State-of-The-Art 的紀錄。

![img](https://miro.medium.com/max/60/1*mONHfgM1B3oRmZknzprtrQ.png?q=20)

![img](https://miro.medium.com/max/1280/1*mONHfgM1B3oRmZknzprtrQ.png)

[A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)

文章更專注在 Contrastive Loss ，並且發現幾點重要因素，包括先前已知的 Batch Size, Embedding Size (與 Normalization)。

![img](https://miro.medium.com/max/60/1*8r66yZ1KWn_f0B9DuUNC7g.png?q=20)

![img](https://miro.medium.com/max/1400/1*8r66yZ1KWn_f0B9DuUNC7g.png)

另外兩點相當有意思，一點是 Data Augmentation 對 Contrastive Learning 的重要性; 一點是利用一個 Non-Linear Map 來避免 Feature 的 Information Loss。

![img](https://miro.medium.com/max/60/1*OTkOQk-Po2KBvVtB4D4vBg.png?q=20)

![img](https://miro.medium.com/max/1400/1*OTkOQk-Po2KBvVtB4D4vBg.png)

SimCLR 的演算法相當簡單

SimCLR 做了大量的 Augmentation ，並且是 Augmentation 的組合。

![img](https://miro.medium.com/max/60/1*J3FnhzcGbC3K_eaiVyyBWg.png?q=20)

![img](https://miro.medium.com/max/1400/1*J3FnhzcGbC3K_eaiVyyBWg.png)

用到幾種常見的 Data Augmentation

在實驗中發現， Color Distortion + Random Crop 效果提升的相當顯著。這是因為原本的 Random Crop 切出來的圖片 Distribution 其實相差不大，可以說是無效的 Patch （尤其對於 Contrastive Learning 來說相當不好），這兩種 Operation 混合後會讓 Distribution 大相徑庭，能產生更多有效的 Negative Samples。

![img](https://miro.medium.com/max/60/1*SYCANHWgED_j1r4LMAz5dw.png?q=20)

![img](https://miro.medium.com/max/1400/1*SYCANHWgED_j1r4LMAz5dw.png)

如果有仔細看 CPC 原文的讀者也會發現，CPC 中提到的使用 Layer Normalization 取代 Batch Normalization 以避免 Model 太容易受到 Patch 的統計性質混淆有異曲同工之妙。

文章另一個亮點是，在算 Loss 之前加上一個 Layer，避免 Visual Representation 直接丟給 Contrastive Loss Function 計算。原因是這種比較 Similarity 的 Loss Function 可能會把一些資訊給丟掉。

![img](https://miro.medium.com/max/60/1*MwUtfx2ZxPNlaxGMi-rcSg.png?q=20)

![img](https://miro.medium.com/max/1400/1*MwUtfx2ZxPNlaxGMi-rcSg.png)

[Bag of Tricks and A Strong Baseline for Deep Person Re-identification](http://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf)

文中做了一些實驗，像是顏色、旋轉這種資訊，就會大幅度的被刪除；而加上一個 Nonlinar Map，這樣可以大幅度地保存 Information。這跟 ReID 中一篇有名的文章 [Bag of Tricks and A Strong Baseline for Deep Person Re-identification](http://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf) 的一種架構有點相似，讓不同層的 Feature 給不同的 Loss 免於 Information Loss。

最後呢，有錢人的研究也是樸實無華且枯燥的，這就是標題 Simple 的來源。

![img](https://miro.medium.com/max/60/1*szWadzxscM5wJ3QChpqUaw.png?q=20)

![img](https://miro.medium.com/max/1400/1*szWadzxscM5wJ3QChpqUaw.png)

這張圖表現出了一件事， Contrastive Learning 是一種能從 Data 本身獲取資訊的 Loss Function ；而且 Data 本身的資訊量遠遠多出 Label 很多，因此就算經過非常長時間的 Training，Model 並沒有任何 Overfit 的跡象。

![img](https://miro.medium.com/max/60/1*b-4UNXwvCreiQsoJoaDeuA.png?q=20)

![img](https://miro.medium.com/max/1400/1*b-4UNXwvCreiQsoJoaDeuA.png)

各方法用到的 Resources

Self-Supervised Learning 到目前爲止展現了相當大的潛力，尤其是 Contrastive Learning 的方法，在資源豐沛的情況下，可以佣簡單的架構便達到逼近 Supervised Learning 的效果，甚至在 Model Capacity 增加的情況下，能與 Supervised Learning 平起平坐。但如果要完全超過 Supervised Learning 的表現要怎麼做呢？

# Semi-Supervised Learning

## Teacher-Student

Teacher-Student (Knowledge Distillation) 是一種相當有效率的做法，他的精神類似 SSL 並且能從 Data 本身取得更多的資訊。

![img](https://miro.medium.com/max/60/1*zK1nifJjLQNnjdowmK9Ntg.png?q=20)

![img](https://miro.medium.com/max/1400/1*zK1nifJjLQNnjdowmK9Ntg.png)

[Billion-scale semi-supervised learning for image classification](https://arxiv.org/abs/1905.00546)

利用有 Label 的資料訓練 Teacher ，並讓 Teacher 標記 Unlabel Data ，再交給 Student 學習。在實驗上也可以發現，這種方法能隨這資料量越多越有效，並且在 Supervised Learning 方法已經 Overfit 時，Semi-Supervised 還可以繼續學習。

![img](https://miro.medium.com/max/60/1*hBviUp62-aaR1IrDn0EIvw.png?q=20)

![img](https://miro.medium.com/max/1400/1*hBviUp62-aaR1IrDn0EIvw.png)

## Noisy-Student

Google 也在這個方法上做了修改，讓 Student 隨著過程中能增大，並且加入更多干擾，確保學到的東西能 General 到真實情況。

![img](https://miro.medium.com/max/60/1*Qr-pMjvTMQ6f0fR9HZ7TCg.png?q=20)

![img](https://miro.medium.com/max/1400/1*Qr-pMjvTMQ6f0fR9HZ7TCg.png)

[Self-training with Noisy Student improves ImageNet classification](https://arxiv.org/abs/1911.04252)

這裡的干擾有兩個：Data 上的 Augmentation 與 Architecture 上的 Randomness。

![img](https://miro.medium.com/max/60/1*__Z4H5sHpsEKhLiTETrFvw.png?q=20)

![img](https://miro.medium.com/max/1400/1*__Z4H5sHpsEKhLiTETrFvw.png)

[Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382), [RandAugment: Practical automated data augmentation with a reduced search space](https://arxiv.org/abs/1909.13719)

一方面讓 Student 有 Ensemble 的特性; 一放面讓 Student 隨著 Knowledge 增加可以增大。這個方法下能達到目前 ImageNet 最高分 Top-1 Accuracy 88.4%。

![img](https://miro.medium.com/max/60/1*IONjMY5MxE6WvY7g-XSfIQ.png?q=20)

![img](https://miro.medium.com/max/1400/1*IONjMY5MxE6WvY7g-XSfIQ.png)

最後用這張圖總結資料與 Network 的發展：

- 如果在單純 Supervised Learning 的情況下，研究 Architecture 帶來的效果是相當顯著的，人類目前得到 10 % 的進步。
- 如果在限定架構的情況下，透過有效率的 Training 流程，從 Data 本身學習 (Unlabelled Data) ，也能得到顯著提升。從 76% 能到 81.2% ; 從 85.5% 能到目前的 SOTA (88.4%)

Self-Supervised Learning 與 Semi-Supervised Learning 想傳達的概念不盡相同，如果我們能從 Data 本身中，有效率的獲取資訊，那將比傳統的 Manual Labelling 表現更好、學到的資訊更豐富並且更能隨著問題 Scaling Up。

# References

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
16. [RandAugment: Practical automated data augmentation with a reduced search space](https://arxiv.org/abs/1909.13719)