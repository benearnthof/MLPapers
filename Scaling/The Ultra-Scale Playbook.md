https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=high-level_overview

Rough overview of techniques: 
* Data Parallelism
* Tensor Parallelism
* Pipeline Parallelism
* Context Parallelism
* ZeRO (https://arxiv.org/abs/1910.02054)
* Kernel Fusion

The core problem is that ultra-scale training is extremely expensive, so we want to use the available hardware to its fullest potential, eliminating downtime and minimizing communication overhead. Every dimension of parallelism incurs a debt to training throughput. Sooner or later that debt is paid.

We want to find a delicate balance between memory usage, compute efficiency, and communication overhead during training. If GPUs spend more time offloading to CPU or being forced to idle because of synchronization issues we are wasting potential compute resources. If models don't fit into memory we cannot train at all. 

On large batch training: [[An Empirical Model of Large-Batch Training]]

For LLMs batch size is commonly reported as batch size tokens (bst)
Sweet spot for training LLMs is on th eorder of 4-60 million tokens per batch.

Basically any deep model stores 4 items in GPU HBM: 
* Model weights
* Model gradients
* Activations needed to compute the gradients
* Optimizer state
Some memory is reserved to initialize CUDA kernels, some memory is also used for buffers and intermediate results. Additionally, some memory may be unavailable due to fragmentation. 

Visualizing Memory Traces
	https://zdevito.github.io/2022/12/09/memory-traces.html

Counting Model Parameters in Transformers
	https://michaelwornow.net/2024/01/18/counting-params-in-transformer
	https://blog.eleuther.ai/transformer-math/

Parameter and optimizer state memory requirement depend on model architecture, precision, and optimizer type (adam with momentum and variance requires 4 bytes per param in fp32) 

On calculating activation memory: 
	https://web.archive.org/web/20250308172134/https://www.determined.ai/blog/act-mem-2

Activation memory scales linearly in batch size but quadratically in sequence length
	[[Reducing Activation Recomputation in Large Transformer Models]]

Selective Activation Recomputation is they key: 
	Expensive Feedforward computations should be checkpointed
	Attention operations are pretty cheap and can be rematerialized
	For GPT-3 selective recomputation would lead to a 70% activation memory reduction at a 2.7% increase in compute cost

Gradient accumulation linearly decreases the activation memory footprint but also scales the time required to compute gradients for each batch linearly.

## Data Parallelism
(Gradient accumulation across multiple GPUs)

Because we dispatch different micro batches to each respective GPU we cannot just update the copies of the model on each of them separately. We use **all-reduce** to average the gradients from the model instances before the optimizer step.

Naively waiting for all GPUs to sync would lead to downtime. This is why we employ the following optimizations, to overlap communication and computation.

### Overlapping Gradient Synchronization with Backward Pass
