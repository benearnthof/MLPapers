Fast and Memory-Efficient Exact Attention with IO-Awareness
https://arxiv.org/abs/2205.14135
https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad
[[Online normalizer calculations for softmax]]
[[Self-attention Does Not Need On2 Memory]]
https://www.adept.ai/blog/flashier-attention
https://tridao.me/blog/2024/flash3/
https://arxiv.org/abs/2208.07339
https://arxiv.org/abs/2307.08691
https://github.com/mattneary/attention
https://pytorch.org/blog/inside-the-matrix/

This paper is the first publication that presented custom attention kernels for training, meaning forward and backward computation that implements exact attention to efficiently use GPU memory. It is the first in a series of three FlashAttention publications [[FlashAttention 2]] [[FlashAttention 3]] that were later extended by [[Native Sparse Attention]]

The missing piece the authors address in FlashAttention is the lack of IO-awareness of prior attention implementations, meaning that reads and writes between different levels of GPU memory become prominent bottlenecks when scaling models to the level modern LLMs are at. 
FlashAttention is an *exact* attention algorithm that uses tiling to reduce the number of memory reads and writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM. 
(SRAM -- static RAM is cache that is close to the streaming multiprocessors and tensor cores)

FlashAttention is also extended to block-sparse attention, yielding up to 15% decreases in wall clock time during training.

### Introduction
Transformers [[Attention Is All You Need]] have become larger [[Language Models are Few Shot Learners]] and deeper [[Deepnet]] but training them on longer context windows remains a challenge because of the quadratic time and memory complexity of the attention operation. 
Approximate attention in the form of sparse- or low-rank methods [[Reformer]], [[Routing Transformers]],[[Rethinking Attention with Performers]], [[Transformers are RNNs]], [[Linformer]] or their combinations: [[Longformer]], [[Scatterbrain]], [[Big Bird - Transformers for Longer Sequences]] have already been proposed, but while these methods reduce the computational complexity compared to vanilla attention, in practice they often do not yield wall clock speed ups. 

The missing piece is the lack of I/O-Awareness [[IO-Complexity]] specific to the GPU Artchitecture available during training [[Dissecting the NVIDIA Volta GPU]]
For modern Transformer based models the larges bottleneck is memory bandwidth and access speed [[Data Movement Is All You Need]]. [[Computer Architecture A Quantitative Approach]]

The main goal of FlashAttention is to avoid having to read and write the entire attention matrix to and from HBM. This requires: 
* Computing the softmax reduction without access to the complete input
* Not storing the large intermediate attention matrix for the backward pass
To solve these issues FlashAttention uses **tiling** of the softmax reduction and **recomputation** of attention on-chip in the backward pass by storing the softmax normalization factor from the forward pass. 
Despite the increase in FLOPs due to recomputation FlashAttention runs faster *and* uses less memory than vanilla attention. 

The benefits of FlashAttention and Block-Sparse FlashAttention are: 
* Faster Model Training
* Higher Quality Models (Potential to scale transformers to longer sequences)
* A standard Benchmark for Attention

From what I can gather FlashAttention's tiled computation avoids having to offload large matrix operations to (relatively) slow HBM, by computing smaller chunks directly in SRAM and then writing the results to HBM before finally reducing in the softmax step. 
### Background
[[Demystifying NVIDIA Ampere]], [[Dissecting the Ampere GPU]]
The Ampere GPU lineup has up to 80GB of HBM with 2TB/s bandwidth, but only 192KB of on-chip SRAM for every one of the 108 streaming multiprocessors (with bandwidth of up to 19TB/s). The on chip SRAM is an order of magnitude faster than HBM but many orders of magnitude smaller.  Compute has become faster relative to memory speed, thus ops are limited by HBM access. 

GPU kernels deploy a massive amount of parallel threads to execute operations. They load inputs from HBM to registers and SRAM, then compute on SMs and then write the outputs back to HBM. 
[[Arithmetic Intensity]]

Kernel Fusion: If there are multiple operations to be performed on the same input we can load the input once from HBM and perform the operations in order. Compilers can automatically fuse many elementwise operations. 
In the context of LLM training, the intermediate values still need to be written to HBM, because they are used during the backward pass, thus reducing the effectiveness of naïve kernel fusion. 
### The Standard Attention Implementation
Given input sequences $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times d}$ where $N$ is the sequence length and $d$ is the head dimension, we want to compute the attention output $\mathbf{O} \in \mathbb{R}^{N \times d}$ :
$$
\mathbf{S}=\mathbf{Q} \mathbf{K}^{\top} \in \mathbb{R}^{N \times N}, \quad \mathbf{P}=\operatorname{softmax}(\mathbf{S}) \in \mathbb{R}^{N \times N}, \quad \mathbf{O}=\mathbf{P V} \in \mathbb{R}^{N \times d}
$$
where softmax is applied row-wise.
Standard attention implementations materialize the matrices $S$ and $P$ to HBM, which requires $O\left(N^2\right)$ memory. In modern transformers $N \gg d$ (GPT2 1024 vs 64)
Most of the operations are memory bound, even streaming multiprocessors with limited SBM can perform GEMMs on matrices of these sizes quite comfortably, but require offloading to HBM.
This results in large wall-clock times for attention computations in practice.
Other elementwise operations like masking or dropout further exacerbate this problem which is why other papers have attempted to fuse several elementwise operations, like masking with softmax [[MegatronLM]].

The standard implementation (roughly) does the following:
1. Load Q, K from HBM, compute S=QK^T write S to HBM
2. Read S from HBM, compute P = softmax(S), write P to HBM
3. Load P and V from HBM, compute O = PV and write O to HBM

### FlashAttention: Algorithm, Analysis and Extensions
**Tiling:** FlashAttention computes attention blockwise. Softmax couples columns of $K$, so the authors decompose the large softmax operation via scaling [[Reformer]],[[Online normalizer calculations for softmax]], [[Self-attention Does Not Need On2 Memory]]
The core idea is that a vector softmax can be decomposed into multiple separate softmax operations and then recombined by tracking some scaling factors. This allows us to compute row-wise softmax of the full QK^T matrix in blocks. 

**Recomputation:** [[Training Deep Nets with Sublinear Memory Cost]]
The backward pass requires $S$ and $P$ to calculate the gradients w.r.t. the QKV matrices, but by storing the output $O$ and the softmax normalization statistics  we can recompute the attention matrix $S$ and $P$ easily from the blocks of QKV we already have in SRAM. This is a form of selective gradient checkpointing. [[Evaluating Derivatives]]

**Kernel Fusion**: Tiling allows us to combine operations such as masking and dropout with the CUDA kernel executed on the same data, usually left to compiler. 

