[[Generating Long Sequences with Sparse Transformers]]
https://reinforcedknowledge.com/sparse-transformers/
[[Relational inductive biases, deep learning, and graph networks]]
Short survey on CIFAR-10 https://arxiv.org/pdf/2501.06220

Notes on Cosine Learning Rate Annealing [[Chinchilla]]

Attention sinks from the graph perspective
https://publish.obsidian.md/the-tensor-throne/Transformers+as+GNNs/Attention+sinks+from+the+graph+perspective


First steps: 
Archive preprint dataset: 
	Convert Abstracts to HuggingFaceDataset format
	Bytepairencoding of dataset to .bin files for training and test set

CIFAR-10 trained with full attention
	Passing in image data should be straight forward. 
	We flatten images into sequences of 3072 bytes
	https://github.com/kentaroy47/vision-transformers-cifar10
	Modern Vision Transformers use Patch embeddings of course
	Sparse Transformers Paper trains a 128 layer network with full attention with autoregressive masking. 
	From what I understand they just use regular cross entropy loss 
	Key line from the paper: 
		For images, we used data embeddings, where d_data = 3 for the row, column, and channel location of each input byte. 
	They train an unconditioned generator on imagenet64x64 directly from pixels without using a multi-scale architecture (Page 7)
	Training was done on a single V100 GPU node with 8 GPUs total.
	Training Details on CIFAR-10:
		We train strided Sparse Transformers on CIFAR-10 images represented as sequences of 3072 bytes. Models have 2	heads, 128 layers, d = 256, half-size feedforward network and query-key projections, and are trained for 120 epochs with a learning rate of 0.00035 and a dropout rate of 0.25 until validation error stops decreasing.
	Training Details on Imagenet64:
		We used a 48 layer strided Sparse Transformer with 16 attention heads and d = 512, totaling 152 million parameters. We used a stride of 128, a dropout of 0.01, and trained for 70 epochs, which took 7 days on 64 V100 GPUs.
	They represent the CIFAR-10 Images as 3072 entry long byte sequences, with added positional encoding to preserve some of the spatial information
	The fully unconditional models were trained using the maximum likelihood objective.

In [[Image Transformer]] the authors mention that a context of 3000 would be prohibitive? 


Transformer
	Short Multihead Attention Explainer
	Visualize KQV matrices at various stages during training

Block Sparse Attention Repo with Kernels
	https://github.com/mit-han-lab/Block-Sparse-Attention

On the Statistics of Block Sparse Attention
	https://guangxuanx.com/blog/block-sparse-attn-stats.html

Very nice resource on Vision Transformers and Autoregressive Image Models
	https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html

Very useful paper that implements GPT-2 Style image transformers like our repo: 
	[[Generative Pretraining from Pixels]]
	We use the same model code as GPT-2, except that we initialize weights in the layer-dependent fashion as in Sparse Transformer (Child et al., 2019) and zero-initialize all projections producing logits

Blogpost exploring block sparsity
	https://pytorch.org/blog/speeding-up-vits/

[[Spartan]]

Semi sparse linear layers for 6% training speedup
	https://github.com/pytorch/ao/tree/main/torchao/sparsity/training

N:M structured sparsity
	https://pytorch.org/blog/accelerating-neural-network-training/

Nice summary slides on slaying OOMs
	https://christianjmills.com/posts/mastering-llms-course-notes/conference-talk-012/


https://docs.runpod.io/instant-clusters/pytorch
```bash
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ens1
torchrun \
  --nproc_per_node=$NUM_TRAINERS \
  --nnodes=$NUM_NODES \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
torch-demo/main.py
```
### DeepSpeed Features

Drop-in replacement via config:
* Flops Profiler
* Wandb monitoring
* Simplified Data Loader (Should look into this for Imagenet)
*  Random Layerwise token dropping https://www.deepspeed.ai/tutorials/data-efficiency/  [[Random-LTD]]
* Progressive Layer Dropping (Up to 2.5x convergence speedup for pre-training) [[Accelerating Training of Transformer-Based Language Models with Progressive Layer Dropping]]; https://www.deepspeed.ai/tutorials/progressive_layer_dropping/
* Automatic DDP with Mixed Precision
* Activation Checkpointing (same as in torch, we skip it for the most part)
* Gradient Accumulation (May increase throughput slightly via reduction of communication)
* CPU/NVMe offloading
* ZeRO: Easy to set up wtih config, various optimizers are supported [[Turing-NLG]]; LAMB, 1-bit ADAM, 0/1 Adam, 1-bit LAMB https://deepspeed.readthedocs.io/en/latest/optimizers.html
* Data Parallelism: Given by default with more than 1 GPU

Straightforward to implement:
* Pipeline Parallelism: https://www.deepspeed.ai/tutorials/pipeline/ Needs tweaks in nn Modules and training loop, pretty customizable with custom schedules etc.
* Context Parallelism: Via Ring Attention/DeepSpeed Ulysses. https://www.deepspeed.ai/tutorials/ds-sequence/ (maybe for imagenet 12288 sequence length; 4x sequence parallelism would let us keep the training config from cifar-10) Does integrate with sparse attention out of the box
* Sparse Attention: Drop in kernels supported by all of the above https://www.deepspeed.ai/tutorials/sparse-attention/
* MoE: Should require only a few tweaks to nn Modules 

Requires major adjustments:
* Tensor Parallelism / Model Parallelism: Maybe config argument works ? https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/huggingface-tp/README.md Large memory reduction, large communication overhead

### Large Batch Training

[[Accurate Large Minibatch SGD]]