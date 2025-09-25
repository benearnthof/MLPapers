Training BERT in 76 Minutes
https://arxiv.org/pdf/1904.00962

[[Accurate Large Minibatch SGD]] introduced the idea of warmup to aid in convergence of large minibatch SGD, when combined with linear scaling of learning rate. 

[[LARS Large Batch Training of Convolutional Networks]] was initially introduced to accelerate large batch training for resnets, but this technique does not work well with transformers.

In general we want to increase batch size as much as possible to decrease the variance of stochastic gradients and thus get away with larger optimization steps.

[[Image Classification at Supercomputer Scale]] 
Achieves top1 accuracy of 76.3% on ImageNet in 2.2 Minutes. Training throughput of over 1.05 million images per second !

### LARS:
1. Compute gradients
2. Initialize Trust Ratio (Weights/Gradients) based
3. compute norm of gradients
4. perform layer wise weight decay
5. Compute the new trust ratio
6. update momentum
7. update weights
=> Layer wise adaptive learning rates

### LAMB:
1. Compute Gradients
2. Compute first moment
3. compute second moment
4. bias correction for first moment
5. bias correction for second moment
6. init trust ratio
7. compute norm of gradients
8. elementwise weight decay
9. compute trust ratio
10. update the weights
=> Adaptive *element* wise updates

