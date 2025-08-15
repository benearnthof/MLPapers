https://arxiv.org/pdf/2103.14636
2023

Sequence models have to tackle time dependencies present in the data. RNNs do this implicitly, by computing a new internal representation for every time step. Architectures like RNNs, LSTMs, or GRUs are limited, however, by a one-to-one correspondence between the shapes of input and output tensors. [[A Practical Survey on Faster and Lighter Transformers]] adresses this by introducing an encoder that first processes the entire sequence to produce a hidden representation that is then fed to a decoder model, which then produces an output sequence in an autoregressive manner.

General methods to increase performance in training and inference of neural networks are often orthogonal, and consequently, several of them may be combined to precisely tune the network's capacity, computational cost, and memory usage. 

Alternative Transformers are categorized depending on whether they sparsify attention, factorize it, or modify the network architecture directly. [[Efficient Transformers]]

### The Transformer
Very nice chapter on the transformer block noting the residual connections [[Deep Residual Learning for Image Recognition]] and [[Layer Normalization]]. [[Reconciling Modern Deep Learning with Traditional Optimization Analyses]][[How does BatchNorm Help Optimization]]

### General Approaches
This section introduces techniques that apply to almost all neural network architectures. 

**Gradient Checkpointing** [[Training Deep Nets with Sublinear Memory Cost]]
Intermediate results (activations) required to compute the gradients during the backward pass are stored in GPU memory.

**Reversible Layers** [[NICE]], [[Reformer]]
Somewhat resembling resnets, reversible layers are constructed in such a way that activations can be reconstructed from the subsequent layer.

**Parameter Sharing**
Separate different network components share the same parameters. An extreme case of this would be regular, vanilla RNNs where multiple time steps implicitly share the exact same weights. [[Linformer]] and [[Reformer]] share projection matrices across heads and layers, and queries and key parameters, respectively. [[ALBERT]] shared *all* parameters between layers.

**Pruning** [[Optimal Brain Damage]]
Removing weights that have only a negligible impact on performance *after* training. 
[[Dropping Layers]], [[Are Sixteen Heads Really Better than One]], [[The Lottery Ticket Hypothesis]], [[When BERT Plays the Lottery]]
Two major drawbacks: A large model must be trained initially and, crucially, sparse models are unoptimized for modern GPUs and tensor processing units. 

**Knowledge Distillation**
[[Distilling the Knowledge of a Neural Network]][[Do Deep Nets Really Need to be Deep]]
Teacher-Student Approach, Useful only at Inference time. [[DistilBERT]] Makes models lighter but still requires a large base model and student models may significantly degrade performance.

**Mixed-Precision**
[[Mixed Precision Training]] stores and computes weights, activations and gradients in half-precision. A master copy of the weights is stored in single-precision for numerical stability. 
[[Quantization Aware Training]] Introduced 8-bit quantization of weights and activations and uses a straight-through estimator to approximate gradients of nondifferentiable operations during training. [[Estimating or Propagating Gradients Through Stochastic Neurons]]
[[Iterative Product Quantization]]  replaces vectors of weights by their assigned centroid and their respective quantization. 
Pruning reduces the number of parameters directly, while quantization reduces the number of bits required to store each parameter. 

**Micro-Batching** 
Improving model performance by increasing model capacity and data throughput. [[GPipe]] adresses issues arising from large mini batches by model parallelism. We basically distribute a model across GPUs by separating sequential layers into a GPU parallel pipeline. The forward pass is done sequentially by splitting each minibatch across devices, the backward pass simply accumulates gradients from all micro-batches. Waiting times between GPUs can be utilized for [[Moccasin - Tensor Rematerialization]] This can cause minor discrepancies with models that use [[Batch Normalization]].

**Mixture of Experts**
The core idea is to train multiple networks called experts, each of which specializes only in a subset of the data. A manager or "router" then decides which expert to dispatch input data towards. In most practical applications a single network is used, whose layers are composed of multiple separate subsets, effectively resulting in a sparsely activated model. This is similar to the Native Sparse Attention Algorithm proposed in [[Native Sparse Attention]].
Increasing the number of experts does keep the computational cost constant, since the number of *active* experts remains constant. This also integrates very well with device parallelism, since one can easily distribute experts across GPUs or even machines. 
The method is complex to deploy in practice however, since one does incur a communication cost between devices, a computational cost to designate the correct experts, and it does make training unstable. [[Switch Transformers]] adresses these issues, demonstrating that a trillion parameter model can be trained on bfloat16. 

**Sample-Efficient Objective**
Pretraining extremely large models remains prohibitively expensive for most practical applications. In some cases the need for pretraining cannot be avoided however. [[ELECTRA]] Introduces a novel, sample efficient pre-training task called "replaced token detection", resulting in higher quality embeddings given the same amount of training data, model capacity, and compute. 

**Parameter Initialization Strategies**
Transformers are notoriously hard to train, requiring carefully tuned optimizers with adaptive learning rates, learning rate schedulers, and large batches. [[Understanding the Difficulty of Training Transformers]] and [[Improving Transformer Optimization Through Better Initialization]] propose initialization methods leading to smoother optimization and better generalization properties of trained models. In essence they introduce Adaptive Model Initialization (Admin) that controls the transformer layers dependency on the residual branches with a new parameter $\omega$ that scales the magnitude of the residual connection by an adjustable factor.
This works in three stages: Standard initialization [[Understanding the Difficulty of Training Deep Feedforward Neural Networks]], [[Delving Deep into Rectifiers]], suitable for the architecture. Passing a batch of data through the model and recording activations, and finally: Initializing $\omega$ depending on the variance of the model activations. After training $\omega$ is discarded.
Transformers do have other issues during training however: Vanilla SGD is not sufficient for these models, leading researchers to utilize optimizers that rely on adaptive learning rates. This leads to problematically large variance during the early stages of optimization, resulting in convergence issues [[On the Variance of the Adaptive Learning Rate and Beyond]]. They solve this with a technique they coin "T-Fixup" eliminating the need for layer norm and warmup. 
[[On Layer Normalization in the Transformer Architecture]]

**Architecture Search**
The problem of finding architectures that achieve the best performance with the fewest possible operations, and a minimal memory footprint in a discrete search space is an NP-hard combinatorial optimization problem. 
The biggest constraint for NAS is the enormous amount of computing power required, since, at least in the naïve case, candidate models must be trained before evaluation. Further, applying this to transformers is prohibitively expensive because of the already extraordinary training requirements. 
[[The Evolved Transformer]] modifies tournament selection evolutionary architecture search with Progressive Dynamic Hurdles, automatically allocating resources to more promising architectures. 
Other handcrafted architectures, like [[Lite Transformer]] outperform the Evolved Transformer for mobile NLP while requiring about 14000x less GPU time. 

**Conditional Computing**
[[Deep Learning of Representations]] A simple way of decreasing the total computational resources required for training and inference, would be to dynamically reduce the amount of computation allocated to simple problems. Many modern LLM services already deploy such routing. Bengio introduced conditional computing, which dynamically adapts the models computational graph as a function of the input. [[Adaptive Computation Time for Recurrent Neural Networks]]
[[Universal Transformers]] applied ACT to transformers with a recurrent mechanism for the architectures depth. 
Other dynamic architectures like [[Adaptive Attention Spans]] and [[Depth Adaptive Transformers]] sought to address the limitations of fixed context lengths and the application of the same exact layers at every depth.
### Specialized Approaches
The attention weight matrix is dominated by a few large values and is approximately low rank. This motivates two distinct lines of research: Sparse Attention and Factorized Attention. 

**Sparse Attention**
Due to the exponential nature of the Softmax operation, only a few positions are strongly attended to. Consequently it would be (at least conceptually) simple to sparsify this matrix by setting the contributions of irrelevant cells to zero (well, negative infinity, really, since the softmax of -inf is zero).  But since we don't know ahead of time which tokens are going to be relevant for the current computation we must find a way to efficiently sparsify the QK matrix. In general, there are three different ways to implement this: Fixed and random sparsity patterns, learned and adaptive patterns, clustering & locality sensitive hashing. 

**Fixed and Random Sparse Patterns**
[[Longformer]], [[Generating Long Sequences with Sparse Transformers]], [[Star Transformer]], [[Enhancing the Locality]], [[Blockwise Self-Attention for Long Document Understanding]], [[Transformer on a Diet]],[[Big Bird - Transformers for Longer Sequences]]
Star Transformer considers a fixed sparse pattern by only allowing attention between adjacent positions and a fixed global token every token can attend to. 
Sparse Transformers introduced strided and fixed attention patterns, allowing the ith output position to attend to the jth input position if one of the conditions 
(𝑖 + 𝑠) > 𝑗 > (𝑖 − 𝑠) or (𝑖 − 𝑗) mod 𝑠 = 0 
is satisfied. (With stride n close to sqrt(n))
Cascade Transformers [[Transformer on a Diet]] rely on exponentially growing the attention window by number of layers.
[[Enhancing the Locality]] introduced the LogSparse-Transformer for forecasting fine-grained time series with strong long-term dependencies. 
[[Blockwise Self-Attention for Long Document Understanding]] introduced BlockBERT, relying on block-wise attention that splits input sequences into separete, non-overlapping, chunks, that are then fully attended to internally.
[[Longformer]] reduces the computational complexity of attention to O(n) by using a combination of sliding window and global attention. Sliding window attention comes coupled with an inductive bias towards locality, which is ameliorated by introducing special, preselected, tokens that are able to attend to the global context, in turn reducing the maximum path length between any two positions in an input sequence to two. 
[[Big Bird - Transformers for Longer Sequences]] Is quite similar, using a mix of random, sliding window, and global attention. 

**Learned and Adaptive Sparse Patterns**
[[Adaptively Sparse Transformers]] [[SparseBERT]] [[Sparse Sinkhorn Attention]]
Sinkhorn attention extends block sparse attention by learning an importance score for every block. In the limiting case this does not outperform vanilla attention however. 
SparseBERT proposed to learn sparsity patterns for each task in an end-to-end fashion wih the Differentiable Attention Mask algorithm, calculating an attention mask with GumbelSigmoid noise. 
Adaptively sparse transformers replaced softmax by an alpha-entmax function, a differentiable generalization of the softmax that pushes small weights to be exactly zero. 
Nonetheless, Adaptively Sparse Transformers compute attention scores for each pair of queries and keys, resulting in models that consume less memory but are 25% slower then the original transformer in terms of tokens per second. 
Moreover, unstructured sparse attention (whether fixed, random or learned) does not benefit frmo efficient, hardware aware, implementations and therefore cannot result in memory and or computational improvements. [[The Hardware Lottery]]

**Clustering and Locality-Sensitive Hashing**
[[Reformer]], [[Routing Transformers]]
The Softmax function is dominated by the largest values, that is the key and query pairs that have the largest dot product. Therefore one may approximate attention by computing only the scores for the keys and queries with the highest similarity. This is a form of adaptive sparsity, since the patterns depend on the data, but they are conceptually different. 
Reformer selects the set of keys that a query can attend to by grouping them with angular multi-round locality-sensitive hashing. Such a hashing scheme has a high probability of assigning the same value to similar vectors. 
Routing Transformers introduced a clustering based method that relies on the observation that for the Maximum Inner Product Search problem, when the norm of every $K_j$ is constant, the problem is equivalent to nearest neighbor search. They utilize an online mini-batch version of k-means and a set of centroids, dictating which keys queries may attend to. 

**Low-Rank Factorization**
[[Linformer]] demonstrated that the attention matrix is approximately low rank. Consequently one may approximate attention by factorizing it into the product of two separate matrices of lower dimensions, thus reducing the transformers computational complexity. 
[[Synthesizer]], [[Nyströmformer]]
Formally the low-rank attention is given by:
$$
\operatorname{Attention}(X)=\underbrace{\operatorname{Softmax}\left(\frac{Q K^{\top}}{\sqrt{d}}\right)}_{n \times n} \underbrace{V}_{n \times d} \approx \underbrace{\operatorname{Softmax}\left(\frac{Q(E K)^{\top}}{\sqrt{d}}\right)}_{n \times k} \underbrace{F V}_{k \times d}
$$
Where E and F are two linear projection matrices learned during training. The authors showed that E and F could be shared across heads and layers with virtually no performance penalty. 
Synthesizer learn compatibility scores without computing the pairwise dot products between queries and keys. They do this by learning dense projection matrices to compress inputs. Although Synthesizers eliminate the need to compute pairwise dot products, their complexity remains quadratic with respect to the input length. 
Nyströmformer relies on the Nyström method to generate a low-rank approximation of the Softmax matrix. The Nyström method was first proposed to speed up kernel machines in [[Using the Nyström Method to Speed up Kernel Machines]]. Provided that the number of landmarks chosen to approximate the original attention matrix is constant and much smaller than the sequence length, the Nyströmformer complexity is O(n). 

**Kernel Attention**
[[Fast Autoregressive Transformers with Linear Attention]] interpreted Softmax as a kernel, decomposed it as an inner product in the right vector space, and rearranged its computations in a way that reduced complexity. 
[[Rethinking Attention with Performers]] later demonstrated that this can be extended by introducing a kernel with randomized mappings. 

**Clustering and LSH**
[[Clustered Attention]] improves on the clustering methods introduced previously by considering the k keys with the highest attentio nfor each cluster. Compared to Reformer, this method is significantly faster. 

### Architectural Changes
The two subsections here introduce methods that improve upon the transformers complexity by modifying the model architecture while preserving vanilla attention. 

**Memory**
[[Transformer-XL]], [[Compressive Transformers]]
Transformer-XL relies on segment-based recurrence between windows. This is implemented by storing the representations of previous windows in a first in first out memory. 
This model, while achieving great performance, cannot capture dependencies outside the FIFO memory range and is only compatible with autoregressive tasks. 
Compressive Transformer adds a compressed memory to the FIFO memory, allowing the model to store more contextual information than Transformer-XL alone. 

**Sequence Compression**
[[Funnel Transformer]] argues, that the complete sequence of hidden states may contain significant redundancy that the model may not have to preserve token-level information. Funnel Transformer reduces computational cost by gradually reducing the length of the hidden states with pooling. 

