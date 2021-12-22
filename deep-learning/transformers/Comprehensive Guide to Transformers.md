# Comprehensive Guide to Transformers

- 8 mins read
- Author Ahmed Hashesh
- Updated December 13th, 2021

You have a piece of paper with text on it, and you want to build a model that can translate this text to another language. How do you approach this?

The first problem is the variable size of the text. There’s no linear algebra model that can deal with vectors with varying dimensions.

The default way of dealing with such problems is to use the bag-of-words Model ([1](https://en.wikipedia.org/wiki/Bag-of-words_model)). In this model, data will be a vector of a massive number, as big as the number of words in a language, and most of the vector elements will be zeros, as most of the terms are not used in this text. To minimize the size of the vector for computation, we store only the positions of the presented words.

However, the Bag-of-Words Model ignores the ordering of the words, which is critical. For example: “**Work to live**” is different from “**Live to Work**.” To keep the data order, we will increase the dimension of the graph (**n-gram**) to add the order into our equation.

In n-gram models, the probability of a word depends on the (**n-1**) previous comments, which means that the model will not correlate with words earlier than (**n-1**). To overcome that, we will have to increase n, which will increase the computation complexity exponentially ([2](http://d2l.ai/chapter_recurrent-neural-networks/rnn.html)).

So, here are the problems we have so far:

1. Variable length of the text.
2. The massive size of data after applying the Bag-of-Words Model.
3. The computation cost as we increase the dimensionality.

Seems like we need a new model, which doesn’t depend on the Bag-of-Words. This is where the RNN Model ([3](https://youtu.be/0TMKCQiT1T0)) comes into play.

## Recurrent Neural Networks (RNN)

The RNN is the same as the n-gram model, except that the output of the current input will depend on the output of all the previous computations. The RNN has its internal state which works as a memory. It’s very suitable for natural language processing and speech recognition.

![Transformers-rnn-train](https://i0.wp.com/neptune.ai/wp-content/uploads/Transformers-rnn-train.png?resize=538%2C235&ssl=1)*Character Level Language model based on RNN [[source](https://d2l.ai/chapter_recurrent-neural-networks/rnn.html)]*

This diagram shows that the input at a time (t+6) depends on the hidden state of each previous step and the current input. It allows the network to keep a history of previously learned parameters and use it to predict the following output, which overcomes the problem of word order and removes the computation cost, as we’ll just pass the words individually on our model.

This model seems to be perfect, but in practice it has some issues ([5](http://d2l.ai/chapter_recurrent-neural-networks/bptt.html)):

1. Vanishing or exploding gradients problem.
2. We can’t parallelize the computations, as the output depends on previous calculations.

Okay, the RNN model is not perfect. So, it was modified further to overcome these imperfections.

### LEARN MORE

[Recurrent Neural Network Guide – a Deep Dive in RNN](https://neptune.ai/blog/recurrent-neural-network-guide)

### Long short-term memory (LSTM)

This particular kind of RNN adds a forget mechanism, as the LSTM unit is divided into cells. Each cell takes three inputs: :

- current input,
- hidden state,
- memory state of the previous step ([6](http://d2l.ai/chapter_recurrent-modern/lstm.html#gated-memory-cell)).

These inputs go through gates:

- input gate,
- forget gate,
- output gate.

Gates regulate the data to and from the cell.

The forget gate decides when to remember, and when to skip the inputs from previous hidden states. This design was mainly created to overcome the vanishing and exploding gradients problems.

![Transformers-LSTM3-chain](https://i0.wp.com/neptune.ai/wp-content/uploads/Transformers-lstm-3.png?resize=648%2C326&ssl=1)*Computing the hidden states in LSTM [[source](https://d2l.ai/chapter_recurrent-modern/lstm.html)]*

LSTM was able to overcome vanishing and exploding gradients in the RNN model, but there are still problems inherited from the RNN model, like:

1. No parallelization, we still have a sequential path for the data, even more complicated than before.
2. Hardware resources are still a problem.

The solutions are not enough to overcome the memory and parallelism problem. So, it’s time to introduce another model.

![Transformers-lstm-3](https://i0.wp.com/neptune.ai/wp-content/uploads/Transformers-lstm-3-1.png?resize=648%2C326&ssl=1)LSTM Module [[source](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)]

## Transformers

In the previous sections, we introduced the problems we face, and some of the proposed solutions that fixed part of the issues. But there’s still room for research.

We mentioned the variable-length problem on the sequence to sequence translation, which was not addressed yet.

Trying to fix this problem, a model was introduced in 2017 that depends on attention mechanisms. Instead of processing tokens separately, we divide text into segments and learn the dependencies between them.

This model was designed based on another architecture, consisting of two main components.

The input is passed first through an encoder.

This encoder will take a variable-length input and convert it into a hidden state with a fixed length.

Then the hidden state will go through the second component, which is a decoder that converts the fixed-length state to an output of variable length.

This architecture is called the ***encoder-decoder architecture\*** ([4](http://d2l.ai/chapter_recurrent-modern/encoder-decoder.html#sec-encoder-decoder)).

![Transformers-encoder-decoder](https://i0.wp.com/neptune.ai/wp-content/uploads/Transformers-encoder-decoder.png?resize=475%2C54&ssl=1)*Encoder-decoder architecture [[source](https://d2l.ai/chapter_recurrent-modern/encoder-decoder.html)]*

![Transformers-encoder-decoder](https://i0.wp.com/neptune.ai/wp-content/uploads/Transformers-seq2seq.png?resize=555%2C175&ssl=1)*Sequence to sequence learning [[source](https://d2l.ai/chapter_recurrent-modern/seq2seq.html)]*

In Transformers, the input tokens are converted into vectors, and then we add some positional information (positional encoding) to take the order of the tokens into account during the concurrent processing of the model.

The transformers modified this model to make it resistant to the previous problems we discussed, using stacked self-attention and fully connected layers for both the encoder and the decoder.

### READ ALSO

[10 Things You Need to Know About BERT and the Transformer Architecture That Are Reshaping the AI Landscape](https://neptune.ai/blog/bert-and-the-transformer-architecture-reshaping-the-ai-landscape)

**The encoder:** composed of a stack of multiple identical layers, each layer containing two sublayers, multi-headed self-attention mechanism followed by residual connections, and simple-wise fully connected feed-forward network.

**The decoder:** consists of a stack of multiple layers, three sublayers each; the first two layers are the same as the encoder layers, and the third is multi-head attention over the output of the encoder stack.

![Transformer](https://i0.wp.com/neptune.ai/wp-content/uploads/Transformer.png?resize=445%2C574&ssl=1)*The transformer architecture [[source](https://d2l.ai/chapter_attention-mechanisms/transformer.html)]*

### Attention mechanisms

This model was inspired by the human vision system ([7](https://arxiv.org/pdf/1706.03762.pdf)). As a brain receives a massive input of information from the eyes, more than the brain can process at a time, the attention cues in the eye sensory system make humans capable of paying attention to a fraction of what the eyes receive.

We can apply this methodology to the problem at hand. If we know the parts that can affect our translation, we can focus on those parts and ignore the other useless information.

This will affect the system’s performance. While you’re reading this article, you’re paying attention to this article and ignoring the rest of the world. This comes with a cost that can be described as the opportunity cost.

We can select from different types of attention mechanisms, like attention pooling and fully-connected layers.

In attention pooling, inputs to the attention system can be divided into three types:

- the **Keys** (the nonvolitional cues),
- the **Queries** (Volitional Cues),
- the **Values** (the sensory inputs).

We can visualize the attention weights between the Keys and Queries. The Values and the Keys are the encoder’s hidden states, and the query is the output of the previous decoder.

![Attention weight matrix](https://i0.wp.com/neptune.ai/wp-content/uploads/Attention-weight-matrix.png?resize=430%2C462&ssl=1)*Visualizing the attention weight matrix ([source](http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/))*

#### Scaled Dot-Product Attention

The Scaled Dot-Product is a more efficient computation design for the scoring function.

We calculate dot product over the input Queries (Q) and Keys (K), with the same vector length (d). Then we scale them to make sure that the variance remains with different vector lengths, and then apply a softmax function to get the weights on the values (V).

![img](https://lh3.googleusercontent.com/V3B4YS73j04B9PfJUin1VGq9yDw3c0Pu_XzRpT551kMUi6QOAJZ6HTb5ugPqjvn36E-tlKhow3SpbMKK8HOawByb9LHMZpWJ3PlGgX1JT86FGqpeNvtvNfyEJ_gfU3eRTs9gTeVr)

![Scaled dot product attention](https://i0.wp.com/neptune.ai/wp-content/uploads/Scaled-dot-product-attention.png?resize=251%2C284&ssl=1)

![Multi head attention](https://i0.wp.com/neptune.ai/wp-content/uploads/Multi-head-attention.png?resize=247%2C316&ssl=1)

*Scaled Dot-Product Attention & Multi-Head Attention [[source](https://arxiv.org/pdf/1706.03762.pdf)]*

#### Multi-Head Attention

We perform the single Dot-Product Attention **h** time in parallel.

![img](https://lh5.googleusercontent.com/TRczGhxeIT_e6lV3wvIqCwC5BQVyTcTmf-rrMLTrolyv0KV_fT0H3DHLNwWJSHmEwCBYvngaqJPJZVVfBVMXLUq1NvWl19pFu4Q4Ntfv-F_a-gGhZreeyCpbEPuuJgmTkE8wtyTs)

W is the Weights of the keys, Queries, and Values, and O is the output Linear transformation.

The Multi-head attention is used in our model in:

- The decoder layers; Queries is the output of the previous decoder layer, and Keys is the output of the encoder.
- The encoder self-attention layers; the keys, Queries, and values are from the previous layer of the encoder.

#### Self-Attention

A particular attention mechanism where the Queries, Keys, and Values are all from the same source. Self-attention (Intra-attention) is faster than recurrent layers when the sequence length (n) is smaller than the representation dimensionality (d).

Self-Attention is used to learn the correlation of different words of a single sentence, to compute a representation of the same sentence.

### The position-wise feed-forward Network

Each layer of the Encoder and the Decoder contains a fully connected feed-forward network which transforms the representation in each position with the identical feed-forward network of two linear transformations and a ReLU activation function.

![img](https://lh5.googleusercontent.com/2uEu94gCqg71Y6Y1AAq9cDhBFN2m4eUwmNxaQGUk563Q9O_MKB7j8Va-NImEr5cNTEjnudaBPfpbYmwtfuZ-fPVMf7785VTCD0wz9aF0NxICHGlLE9GJWK834NiFmBUM311IZfDA)

### Embeddings and softmax

Convert the input and output tokens into vectors of the model dimension, and convert the output of the decoder into predicted probabilities.

### CHECK ALSO

[Training, Visualizing, and Understanding Word Embeddings: Deep Dive Into Custom Datasets](https://neptune.ai/blog/word-embeddings-deep-dive-into-custom-datasets)

### Positional encoding

As the model has no recurrence and no convolution, A layer was added to make use of the sequence order. At the end of the encoder and decoder stacks, The information injected contains information about the relative or absolute position of the token in this sequence.

### The Vanilla Transformer summary

The Vanilla Transformer is a great model to overcome the RNN models shortcomings, but it still has two issues:

- **Limited Context Dependency:** for character-level language modeling, the model was found to outperform the LSTM. However, as the model was designed to be trained over separated fixed-length segments of a few hundred characters with no information to correlate between the segments, this introduced a problem that there’s no long-term dependency information kept beyond the configured context length. The limited context dependency also makes the mode unable to correlate with any word that appeared several segments ago.
- **Context Fragmentation:** in the first few symbols of each segment, there’s no context information stored as the model is trained from scratch for each segment, leading to performance issues.

So, we still need another enhancement to address these issues and overcome these shortcomings.

![img](https://i0.wp.com/neptune.ai/wp-content/uploads/Vanilla-transformer.gif)*Vanilla Transformer with a segment of 4 [[source](https://ai.googleblog.com/2019/01/transformer-xl-unleashing-potential-of.html)]*

## Transformer-XL

The transformer XL is a newer version from the Transformer (it’s extra long). It is derived from the vanilla Transformer, but introduces the recurrence mechanism and relative positional encoding.

In Transformer-XL, instead of computing the hidden state from scratch for each segment, the model will keep the hidden state of the previously learned segments and use it for the current segment.

The Model solves the problems introduced in the vanilla transformer model, and overcomes the long-term dependency problem. Another advantage is that it also solves the context fragmentation problem caused by the usage of recently initialized or empty context information. Hence the new model can now be used for character-level language modeling and word-level modeling.

### Recurrence mechanism

To preserve the dependencies between segments, Transformer-XL introduced this mechanism. The Transformer-XL will process the first segment the same as the vanilla transformer would, and then keep the hidden layer’s output while processing the next segment.

Recurrence can also speed up the evaluation. We can use the previous segment representations instead of being computed from scratch during the evaluation phase.

So, the input for each layer will be a concatenated form of the following:

- The output of the previous layer, the same as in the vanilla Transformer (grey arrows in the following figure).
- The previously processed hidden layer output (green arrows in the following figure) as an extended context.

![Transformer-XL](https://i0.wp.com/neptune.ai/wp-content/uploads/Transformer-XL.gif)*Transformer-XL with a segment length of 4 [[source](https://ai.googleblog.com/2019/01/transformer-xl-unleashing-potential-of.html)]*

### Relative positional encoding

The recurrence mechanism seems to solve all our problems. Still, the recurrence mechanism introduced another issue: positional information stored in the hidden state reused from the previous segment.

As in the vanilla transformer, the positional information provided by the positional encoding step can lead us to have some tokens from different segments with the same positional encoding, although they differ in their position and importance.

The fundamental concept added in this model is only to encode the relative positional information in the hidden state, enough to know the positional offset between each Key and its Query, and enough to make the predictions.

### Transformer XL summary

The Transformer-XL combined the recurrence with the attention mechanisms converting the vanilla transformer model, which suffered from context fragmentation and limited context-dependency into word-level language modeling, and enhanced its evaluation speed by adding the recurrence mechanism and relative position encoding.

This results in long-term dependency enhancement. According to the original paper of Transformer-XL, it can learn dependency 80% longer than RNNs, and 450% longer than the vanilla transformers, and achieves better performance on the long and short sequences up to 1800+ times faster than the vanilla transformer.

This model was implemented in TensorFlow and PyTorch, and is available [open source](https://github.com/kimiyoung/transformer-xl/).

## Compressive Transformer

One drawback of keeping all these hidden states is that it increases the computation cost of attending every time-step, and the storage cost of keeping all this information.

Several methods have been created to reduce the computational cost of attention, such as sparse access mechanisms, but this doesn’t solve the storage costs.

The compressive transformer is a simple extension of the transformer, inspired by the concept of sleep. Sleep is known to compress the memory, which improves the reasoning ability.

The compressive transformer uses attention to select information from the past, and then compress it into a compressed memory. This compression is done through a neural network trained with a loss function to keep the relevant information.

![Compressive transformer](https://i0.wp.com/neptune.ai/wp-content/uploads/Compressive-transformer.gif)*Compressive transformer [[source](https://deepmind.com/blog/article/A_new_model_and_dataset_for_long-range_memory)]*

### Compression functions

Built over the Transformer-XL. XL keeps the past activations for each layer and discards them only when they are old. The compressive model was implemented to compress the old memories instead of discarding them.

This model uses a variety of compression functions:

- **Max/Mean Pooling.**
- **ID convolution.**
- **Dilated Convolution.**
- **Most used.**

Pooling is considered the fastest and simplest. Most used compression function is inspired by the garbage collection mechanism in differential neural computers, and data is stored by its average usage. The convolutional compression function has some weights to train.

### Compressive Transformer summary

The compressive transform is helpful in long-range modeling. If this doesn’t apply to your project, then compressive transform won’t add any benefits.

As you can see from the following comparison, the results are very close to transformer-XL, but with massive benefits of optimizing the memory usage.

![Compressive transformer results](https://i0.wp.com/neptune.ai/wp-content/uploads/Compressive-transformer-results.png?resize=576%2C175&ssl=1)*Comparison results from the original paper [[source](https://deepmind.com/blog/article/A_new_model_and_dataset_for_long-range_memory)]*

## Related Work

### Reformer

Replaces the dot-product attention by locality-sensitive hashing, which changed the complexity of the model from O(L2) to be O(Llog L) and used a reversible version of the residual layers instead of using the standard residual layer. These changes reduced the computational cost and made the model compete with the state of transformer models while being faster.

## Conclusion

And that’s it. We explored three types of Transformer models, and hopefully now it’s clear why they came to life.

Inspired by human vision and memory, these models are bringing us closer to models that truly work like a human brain. We’re still far from it, but Transformers are a big step in the right direction.

Thanks for reading!

### Other resources about Transformers

- [Dive into Deep Learning Reference book](https://d2l.ai/index.html) – An online book contains an explanation of most of the deep learning algorithms with good code samples.
- [Attention is all what you need](https://arxiv.org/abs/1706.03762) – The original paper where the transformers were first introduced.
- [The Transformer – Attention is all you need](https://mchromiak.github.io/articles/2017/Sep/12/Transformer-Attention-is-all-you-need/#.YIDgVpAzbIX) – An article illustrates the Transformers with a lot of details and code samples.
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Compressive Transformer vs. LSTM](https://medium.com/ml2b/introduction-to-compressive-transform-53acb767361e)
- [Visualizing A Neural Machine Translation Model](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
- [Reformers: The efficient transformers](https://arxiv.org/pdf/2001.04451.pdf)
- [Image Transformer](https://arxiv.org/pdf/1802.05751.pdf)
- [Transformer-XL: Attentive Language Models](https://arxiv.org/pdf/1901.02860.pdf)
- [Compressive Transformers for Long-Range Sequence Modelling](https://arxiv.org/abs/1911.05507)
- [Transformer-XL Explained](https://towardsdatascience.com/transformer-xl-explained-combining-transformers-and-rnns-into-a-state-of-the-art-language-model-c0cfe9e5a924)
