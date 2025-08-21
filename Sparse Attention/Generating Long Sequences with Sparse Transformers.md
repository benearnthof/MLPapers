https://arxiv.org/pdf/1904.10509

### Abstract
Transformers are well suited for modelling sequences and can, in theory, be extended to arbitrarily long sequences. The computational time and memory complexity of vanilla attention, however, grows quadratically with length of the input sequences. This paper introduces a selection of sparse factorizations of the attention matrix which reduce this to $O(n \sqrt{n})$.
Furthermore Child et al. present several architectures suitable to deep network training, recompute attention matrices to reduce the memory footprint of the attention operation, and deliver fast, custom attention kernels to exploit GPU architecture for efficient training. 
### Introduction
The central problem in CNN-based generative modeling is the small receptive field of such approaches, given classical approaches. Several architectural changes have been introduced over the years to address these issues -- dilated convolutions, pixelcnn, Unet, wavenet, etc. 
The Transformer [[Attention Is All You Need]] excels on many long sequence tasks, due to its ability to model arbitrary dependencies in a constant number of layers. As each self-attention layer has a global receptive field, the network can allocate representational capacity to the input regions for which it is most useful. This less informative inductive prior allows transformers to outperform other architectures, with fixed connectivity patterns, at generating diverse data types. Dot product attention grows quadratically, both in the number of operations, and memory footprint, with the sequence length, which renders them unfit for very long sequences. 

**Sparse Attention** addresses these issues by introducing several *Factorizations* of the attention matrix which scale with $O(n \sqrt[p]{n})$ given sequence length $n$ and $p$ the number of separate attention heads. This works by separating the full attention computation into several faster attention operations which, when recombined, can approximate the dense attention operation. 

The core contributions are: Sparse Attention Kernels, Restructured Residual Blocks & Weight Initialization schemes to improve training of very deep networks, Recomputation of attention weights during the backward pass -- reducing memory footprint during training. 

Notes on Related work: Approaches like Clockwork RNN or LSTM seem archaic in context, given the simplicity of the Attention operation. Sparse Attention is orthogonal to these approaches.
### Background 
To best apply classical Transformers to a generative task we must formalize it in the context of autoregressive sequence generation, where the joint probability of a sequence $\mathbf{x}=\left\{x_1, x_2, \ldots, x_n\right\}$ is modeled as the product of conditional probability distributions, parametrized by a network $\theta$.
$$
p(\mathbf{x})=\prod_{i=1}^n p\left(x_i \mid x_1, \ldots, x_{i-1} ; \theta\right)
$$
The easiest way to accomplish this is to treat input data, like images, text, audio, etc. As sequences of discrete tokens. The network $\theta$ ingests any given input sequence, and outputs a categorical distribution over the $v$ (size of vocabulary) possible values of the next token, using the softmax function. The training objective is to maximize the log-probability of the data w.r.t. to the model parameters $\theta$.

### Factorized Self-Attention
The motivation for the introduction of factorized attention is based on an empirical study, performed on a 128-layer deep network, trained on CIFAR-10 with full attention. When inspecting the attention weights for any given layer, with respect to the autoregressive masks present during inference it becomes clear that, similar to filters learned in traditional CNNs, the network decomposes the generative task into several factorized suboperations. The core idea is then to test if several Sparse Transformer layers (strided, fixed, etc.) are able to match performance of the full attention computation. 
It is clear that several sparse attention patterns are present in most layers across all data points, but some layers clearly exhibit global attention patterns, as well as data-dependent sparsity patterns, suggesting that some level of global dependency should be retained in the full model to avoid too much degradation in performance. 

Self-attention layers map matrices of input embeddings $X$ to an output matrix and are parametrized by a connectivity pattern $S=\left\{S_1, \ldots, S_n\right\}$, where $S_i$ denotes the set of indices of the input vectors to which the $i$th output vector attends. The output vector is a weighted sum of transformations of the input vectors: 
$$
\begin{gathered}
\operatorname{Attend}(X, S)=\left(a\left(\mathbf{x}_i, S_i\right)\right)_{i \in\{1, \ldots, n\}} \\
a\left(\mathbf{x}_i, S_i\right)=\operatorname{softmax}\left(\frac{\left(W_q \mathbf{x}_i\right) K_{S_i}^T}{\sqrt{d}}\right) V_{S_i} \\
K_{S_i}=\left(W_k \mathbf{x}_j\right)_{j \in S_i} \quad V_{S_i}=\left(W_v \mathbf{x}_j\right)_{j \in S_i}
\end{gathered}
$$
Where any $W_{k}$ represents the weight matrices transforming a given $x_i$ into the respective key, query, and value vectors. $d$ is the inner dimension of the queries and keys. Each individual output is a sum of the values weighted by the scaled dot-product similarity of the keys and queries. 
Vanilla self-attention defines $S_i=\{j:j\leq i\}$, allowing every element to attend to all previous positions, including its own position. 
**Factorized self-attention** instead has $p$ separate attention heads that each operate on a subset of the index set. The paper only considers *valid* factorizations, meaning that every partial index set is defined such that $i$ can attend to $j$ through a path of locations with maximum length $p+1$.
Combining multiple such layers allows one to retain the ability of Transformers to propagate signals from arbitrary input to arbitrary output positions, while reducing the effective computation to the order of $O(n \sqrt[p]{n})$. Splitting the task into a series of only locally connected layers also may be a useful inductive bias for certain domains. 

### Sparse Transformers
Regular attention is basically an affine transformation followed by a nonlinearity. Factorized attention splits the full operation into multiple residual blocks, which can either be interleaved sequentially or at a ratio. One may also merge heads together by having another head attend to the data that two separate factorized heads each attend to, or simply use standard Multi Head Attention [[Attention Is All You Need]] where $n_h$ individual attention products are computed in parallel and then concatenated along the feature dimension. Here the dimensions of the weight matrices are each reduced by a factor of $1/n_h$, such that the total number of parameters remains invariant across all desirable values.

TODO: Need to see how they implement the kernels for this...
### Scaling to hundreds of layers
[[Identity Mappings in Deep Residual Networks]] introduces pre-activation residual blocks with dropout and layer normalization. This paper uses GLU activation functions. 
### Modeling diverse data types
The paper uses learned embeddings to encode structural information about the data and factorized attention patterns to stabilize performance. The authors one hot encode each data point according to its position in the sequence then se a linear embedding operation to obtain flexible, learnable positional embeddings. 

### Saving memory by recomputing attention weights
Gradient checkpointing [[Training Deep Nets with Sublinear Memory Cost]] is particularly effective for self-attention layers for long sequences, as their memory cost is particularly high compared to the cost of computing them. Recomputing attention and feed-forward blocks is used during the backward pass & to further simplify the implementation dropout is only applied after each residual addition. 

### Efficient block-sparse attention kernels
TODO: Look at CUDA implementation of block transpose aggregate operations

### Notable training techniques
Mixed precision training & gradient accumulation across multiple GPUs. Gradient Clipping, weight decay, cosine decay learning rate annealing [[GPT-1]]
