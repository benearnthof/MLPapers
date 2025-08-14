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

**Conditional Computing**

