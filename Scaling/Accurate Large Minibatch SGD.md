Training ImageNet in 1 Hour (on 256 P100 GPUs xdd)
https://arxiv.org/pdf/1706.02677

Napkin math: (rounded up)
1xP100: 18.7 TFLOPS Half Precision
256xP100:  4787.2 TFLOPS Half Precision total
RTX4000: 14.24 TFLOPS (Closest match on Runpod)
1xA100: 312 TFLOPS bf16
Equivalent A100: ~16
1xH100 SXM: 1979 TFLOPS 
Equivalent H100: ~ 3
1xB200: 4.5 PFLOPS
Equivalent B200: ~ 1.07

So if throughput is absolutely maximal, which of course it usually is not during training, we could complete the 3600s x 4787.2 TFLOPS = 17233920 TFLOPs in about 17233920 / 4500 = ~3829 seconds or around 77 Minutes? Is that realistic? Probably not. 

Oh Imagenet the authors demonstrated no accuracy loss when training with minibatch sizes of up to 8192. By adopting a linear scaling rule for adjusting learning rates as a function of minibatch size, coupled with a new warmup scheme to overcome optimization challenges in early training. 

ResNet-50 ~35 million parameters (~100MB at bf16)

Earlier works like [[One Weird Trick for Parallelizing Convolutional Neural Networks]] already utilize learning rate adjustments based on batch size, but no large scale empirical tests have been performed to truly push this to the limit. 

The authors first give a quick review of Minibatch SGD [[SGD A Stochastic Approximation Method]]

They provide the following learning rate scaling rule: 
### Linear Scaling Rule
When the minibatch size is multiplied by a factor of *k*, multiply the learning rate by *k*.

They justify this by noting that under exactly equivalent network conditions and regular data, the sequential gradient steps of k separate minibatches would follow approximately the same path of a single large minibatch of size k * batch_size with adjusted learning rate $k \times \alpha$. 
This does **not** hold directly after initialization, which is why they introduce a warmup phase, and it also must be noted that one cannot scale minibatch size indefinitely. For batch sizes larger than ~8000 accuracy tends to degrade rapidly. 

[[Revisiting Distributed Synchronous SGD]] also used this linear scaling rule but did not establish a small minibatch baseline. 

[[Optimization Methods for Large-Scale Machine Learning]] gives theoretical reasons why learning rate should not exceed a certain threshold, regardless of batch size

### Warmup
[[Deep Residual Learning for Image Recognition]] introduced the idea of learning rate warmup.
The constant warmup introduced there is not suited to large scale training, the rough spike in learning rate at the transition boundary causes an equivalent spike in training error. This is why this paper introduces *gradual warmup*.
Regarding [[Batch Normalization]], they keep the per worker batch size at 32, computing batch statistics explicitly **not** across all workers. 

### Pitfalls of Large Batch SGD
**Weight decay**: Scaling the cross-entropy loss is not equivalent to scaling the learning rate. 
**Momentum Correction**: Apply momentum correction after changing the learning rate if using SGD with momentum. 
**Gradient Aggregation**: Because [[MPI Message Passing Interface (Textbook)]] allreduce performs summing, not averaging, the authors recommend absorbing the total GPU scaling factor (1/256 * minibatch_size in their case) into the loss directly instead of scaling by the per worker batch size.
**Per Epoch Shuffling**: Is more efficient than sampling each minibatch separately.

### Gradient Aggregation
The authors note that beyond the scale of 256 GPUs, communication overlap with the backward computation may become problematic but at their scale this was not an issue yet. They point to [[Quantized Neural Networks]] for further discussion.

For inter-node communication the authors implement custom optimized algorithms, which is probably the most interesting contribution of the paper. 

