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
The depth of the neural network to be optimized is the main reason this is useful at all. While the gradients of the preceding layers are still being computed, the results for the last layers (the first layers in the backward pass) can already be communicated across devices and the optimization steps for these layers can ideally be completed while the rest of the backward pass is still running. This communication computation overlap decreases idle time and thus increases throughput.

In pytorch this can be done by attaching an all-reduce hook to each parameter: 
```python
def register_backward_hook(self, hook):
    """
    Registers a backward hook for all parameters of the model that 
    require gradients.
    """
    for p in self.module.parameters():
        if p.requires_grad is True:
            p.register_post_accumulate_grad_hook(hook)
```

### Bucketing Gradients
Instead of calling multiple small all-reduce operations for every leaf tensor we instead pack them into buckets before executing an all-reduce operation for the entire bucket at once. 
[[PyTorch Distributed]] pages 4 & 5 are a good reference here.
Parameter-to-Bucket Mapping has considerable impact on DDP speed. Buckets are always created on the same device as the parameters copied into them, this naturally has to take device affinity into consideration as models grow and require multiple GPUs to fit them entirely. DDP launches allreduce in the reverse order of model.parameters(). The key ingredient for performance is finding a balanced bucket size that takes advantage of inter device bandwidth yet still allows communication/computation overlap to increase throughput.

### Interplay with Gradient Accumulation
If we're utilizing gradient accumulation to increase batch size we perform multiple forward and backward passes before updating the parameters. In a naïve implementation one might trigger an allreduce after each backward pass during the accumulation, we can prevent this by adding a model.no_sync() decorator that disables gradient sync during the backward passes that don't need reduction. 

## Zero Redundancy Optimizer [[ZeRO]]
Data parallelism introduces significant memory redundancy since, at least for a basic implementation, we keep the full optimizer state and model parameters in each GPU. The three levels of ZeRO will introduce Optimizer State Partitioning, Gradient Partitioning, and Parameter Partitioning.

In vanilla DP, all ranks gather the same gradients after the backward pass and simultaneously perform identical optimizer steps. It seems like a natural idea to offload this with the help of some slight additional communication overhead which, again, we might overlap with computation to further decrease impedance on throughput while drastically cutting down on memory usage.

### ZeRO-1: Partitioning Optimizer States
In ZeRO-1, the optimizer states are partitioned into $N_d$ equal parts, where $N_d$ is the degree of Data-Parallel instances. This means each GPU only keeps track of $\frac{1}{N_d}$ of the total optimizer state and during each optimization operation only $\frac{1}{N_d}$ of the FP32 weights are being updated.
To compute the forward pass, each instance needs access to the full model parameters, this means we require an additional **all-gather** operation after the optimizer step, such that each GPU has the complete set of updated weights at its disposal. 
We can outline the algorithm as follows: 

1. Perform BF16 forward pass on each replica, with different micro-batches across replicas
2. Perform backward pass with full set of gradients on each replica
3. Perform **reduce-scatter** on the gradients 
4. Each replica performs an optimizer step on its local optimizer states to get $\frac{1}{N_d}$ updated FP32 parameters and convert to BF16 parameters
5. Perform **all-gather** on BF16 parameters to send missing slices to each replica.

### ZeRO-2: Adding Gradient Partitioning
In a nutshell, each GPU only keeps the gradients respective to its optimizer state, further reducing allocated memory for each instance. Depending on the implementation this can achieve even better throughput than ZeRO-1 with almost no real additional overhead since we are already performing a reduce-scatter for the gradients and an all-gather over the parameters anyway. 

### ZeRO-3: Adding Parameter Partitioning (FSDP)
As we perform the forward pass and sequentially go through layers we gather the necessary parameters on demand and immediately flush them from memory when they are no longer required. We basically exploit the sequential forward computation to minimize allocated memory even further, at the cost of all-gather operations at runtime. The backward pass also employs this strategy, just in reverse order.
This results in $2 * num_{layers}-1$ additional all-gather operations per training step, each of which incurs a small latency overhead. This can be minimized in practice by *prefetching* the required parameters (or gradients) for the subsequent (or previous) layers respectively. 
Disregarding communication overhead, this could in theory allow us to drive memory usage down indefinitely, as long as we're able to scale up the number of GPUs we're using. In practice though, one starts to run into diminishing returns in throughput at around 64 nodes / 512 GPUs. 

## Tensor Parallelism
