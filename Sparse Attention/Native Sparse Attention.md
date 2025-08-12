Hardware-Aligned and Natively Trainable Sparse Attention
https://arxiv.org/abs/2502.11089
https://github.com/a-hamdi/native-sparse-attention

NSA, a Natively trainable Sparse Attention mechanism integrates algorithmic advances with hardware-aligned optimizations. 
Dynamic, hierarchical sparse strategy, combining coarse token compression with fine token selection. This preserves both global coherence and local precision. 

Keyword: [[Arithmetic Intensity]] -- How much of computation time is spent actually calculating vs how much time do we need to fetch data from memory. We want to optimize this with the available hardware in mind. 

NSA maintains performance on downstream tasks when compared to Full Attention and achieves substantial speedups on 64k-length sequences across decoding, forward- and backward propagation. 

A common optimization method for Transformer based sequence models is KV-Caching. Since we are not only interested in generating a single token based on a sequence of input tokens, but would like to generate longer sequences of tokens for common downstream tasks like text or image generation, inference is usually performed in two stages: Prefill and Decoding. Prefill "fills" up the KQV matrices based on the input token sequences, allowing us to speed up inference since we only need to recompute a small subset of matrix entries for each subsequently generated token. Many efficient KV-Caching and compression strategies have been proposed [[ClusterKV]],[[Adaptive KV-Cache Compression for LLMs]], [[H2O Heavy Hitter Oracle]], [[SnapKV]], [[LLM x MapReduce]], [[SeerAttention]], [[Quest Query Aware Sparsity]], [[InfLLM Training Free Long Context Extrapolation]], [[MagicPIG LSH Sampling for Efficient LLM Generation]], [[HashAttention]]
but these can only speed up training to a point, since datasets are vast and contain many novel samples. 

Many of these approaches fail to deliver speedups comparable to their theoretical gains in practice, moreover they lack training time efficiency.

NSA reduces per-query computationby organizing keys and values into temporal blocks and processing them through three separate attention paths: 

* Compressed Coarse-Grained tokens
* Selectively retained fine-grained tokens
* Sliding windows for local contextual information

The Authors implemented specialized GPU kernels in Triton to fully exploit the available Hardware.

[[GQA Generalized Multi Query Attention]] and [[MQA Fast Transformer Decoding]] require a ton of memory access volume of KV-Cache, which can reduce computation operations at the cost of increased memory latency. Because memory accesses for these techniques are not blocked this may also cause a performance bottleneck since fetching data from VRAM can be quite slow when compared to SRAM.

### The Myth of Trainable Sparsity
* Applying sparsity post-hoc forces models to deviate from their pretrained optimization trajectory, leading to performance degradation. [[MagicPIG LSH Sampling for Efficient LLM Generation]] demonstrates that th etop 20% of Attention can only cover 70% of the total attention scores. 
* Training efficiency is crucial since trillions of operations are required to train LLMs to convergence. Even marginal gains in attention computation at training time would yield massive reductions in energy costs. Existing sparse attention methods mostly target the inference stage, when most of the work has already been done. 
* Non trainable components like SimilarityHash based selection or k-means clustering [[MagicPIG LSH Sampling for Efficient LLM Generation]] [[ClusterKV]] Do not provide gradient flow through these components, limiting the model's ability to learn optimal sparse patterns.
* Inefficient BackProp: Token-granular selection strategies like [[HashAttention]] require a large amount of individual tokens from the KV-Cache during attention computation. This leads to non-contiguous memory accesses, which slows down computation during training. Techniques like [[FlashAttention]] instead rely on contiguous memory accesses through GPU optimized kernels achieving higher throughput.

### Token Compression
Token Compression aggregates sequential blocks of keys or values into block-level representations. The authors use a strided MLP to compress blocks of keys and values into a compact representation. This captures coarse-grained, higher-level semantic information and reduces computational overhead for downstream attention computations.
$$
\tilde{K}_t^{\mathrm{cmp}}=f_K^{\mathrm{cmp}}\left(\mathbf{k}_{: t}\right)=\left\{\varphi\left(\mathbf{k}_{i d+1: i d+l}\right) \left\lvert\, 0 \leqslant i \leqslant\left\lfloor\frac{t-l}{d}\right\rfloor\right.\right\}
$$
### Token Selection
By only using compressed keys the model would lose fine grained local information, leading the authors to resort to the direct selection of specific, individual tokens. How can this be done in a way that is both efficient and dynamically identifies the most relevant tokens?

#### Blockwise Selection
NSA processes key and value sequences in spacially contiguous blocks, which is key for an efficient GPU implementation. The actual selection process then follows distributional patterns inherent to attention scores. [[Minference]] demonstrated that attention scores often exhibit spatial continuity, suggesting that neighboring keys tend to share similar importance levels. 
Key, and value sequences are first divided separately into blocks and then assigned importance scores that determine if they are selected for downstream processing or discarded. 
#### Importance Score Computation
Computing block importance scores would be quite inefficient in practice, but we can leverage intermediate attention scores from the prior attention computation already performed during token compression. When compression and selection blocks share the same blocking scheme, we can directly obtain the selection block importance scores by calculating 
$$
\operatorname{Softmax}\left(\mathbf{q}_t^T \tilde{K}_t^{\mathrm{cmp}}\right)
$$
Where $\tilde{K}_t^{\mathrm{cm}}$ is the respective compression score for the current block $t$. Should the blocking schemes differ one can derive auxiliary scores via their spatial relationship. See equation 9 in the paper. 
#### Top-n Block Selection
After block importance scores have been obtained only those tokens within the top-n sparse blocks ranked by their block importance are retained. The selected blocks are then concatenated for further processing. 
### Sliding Window
To avoid the learning process being dominated by strong local dependencies in data NSA introduces a third attention branch, using a sliding window approach to allow the model to learn local features separate from the global compressed features and highly specific top-n selected features. The three separate information branches are then aggregated through a learned gating mechanism -- basically a weighted sum. 
### Kernel Design
The authors implement NSA as custom, efficient GPU kernels with triton. Multi head attention is memory-intensive and inefficient for decoding, they focus on architectures with shared KV caches like GQA and MQA. 
Compression and Sliding window attention are already compatible with available [[FlashAttention]] kernels, the authors introduce a specialized kernel for sparse selection attention. 
They follow a different query grouping strategy, loading all query heads within each GQA group into SRAM, for each position on the query sequence. 
In essence they utilize three strategies: 
* Group-Centric Data Loading: For each inner loop, load all heads' queries in the GQA group at position $t$ and their shared sparse key/value blocks indices $I_t$
* Shared KV Fetching: In the inner loop, sequentiall load continuous key/value blocks indexed by $I_t$ into SRAM to minimize memory loads.
* Outer Loop on Grid: Since the inner loop length remains nearly identical for different query blocks, query and output loops remain in tritons grid scheduler to simplify and optimize the kernel. 

