Contextual Sparsity for Efficient LLMs at Inference Time
https://proceedings.mlr.press/v202/liu23am/liu23am.pdf

[[Approximate Nearest Neighbors]]

The core idea seems similar to speculative decoding/LSH etc.

The authors train a lookahead predictor to predict contextual sparsity on the fly, for a frozen, pretrained model.

They predict a relevant subset of attention heads or MLP parameters and only load them for the computation (?) 
In section 4.3 they propose an asynchronous predictor similar to a classic branch predictor [[A Study of Branch Prediction Strategies]] to avoid sequential overhead.

[[Deep Compression]]
[[Quantization and Training of Neural Networks]]
[[Data-Free Quantization]]
[[Improving Neural Network Quantization without Retraining]]
[[Distilling the Knowledge of a Neural Network]]
[[Training data-efficient image transformers]]
[[ZeroQuant]]
[[LLM.int8()]]
[[GPTQ]]
[[SparseGPT]]
[[SmoothQuant]]

The main reason their approach works is because in massive language models that are sharded across multiple GPUs or even machines, communication overhead for large layers can become a bottleneck. Various strategies to deal with this already exist, they attempt to fit a small predictor on the GPU that is utilized for "branch prediction" of which parameters to load ahead of time, which increases throughput. This, of course, works only at inference time, and in settings where models are sufficiently sharded for this to become an issue. 

[[Fast Attention Requires Bounded Entries]]
[[Pruning Convolutional Neural Networks for Resource Efficient Inference]]
[[Inducing and Exploiting Activation Sparsity for Fast Inference on Deep Neural Networks]]
[[The Lazy Neuron Phenomenon]]

The authors hypothesize that attention performs [[Mean-Shift Algorithm]]
They argue that the softmax can be interpreted as a kernel that performs a single mean shift step for tokens that have embeddings that already point in a similar direction. 

They argue that layer by layer the embeddings remain highly similar to those in neighboring layers due to residual connections.

Because [[Efficient and Robust Approximate Nearest Neighbor Search]] and [[FAISS]] is too slow when compared to matrix operations on GPUs the authors opt for a 2 layer MLP ? XD

To make this work at all they employ
* Asynchronous Execution
* Hardware Aware Implementation https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/
* Kernel Fusion
* Memory Coalescing