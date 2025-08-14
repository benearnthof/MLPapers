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

**Mixture of Experts**

**Sample-Efficient Objective**

**Parameter Initialization Strategies**

**Architecture Search**

**Conditional Computing**

