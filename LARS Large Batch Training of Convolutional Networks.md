https://arxiv.org/pdf/1708.03888
Proposes Layer-wise Adaptive Rate Scaling (LARS) to help overcome optimization problems manifesting in the large batch training regime.

Refers to [[One Weird Trick for Parallelizing Convolutional Neural Networks]] and [[Accurate Large Minibatch SGD]] as inspiration.

The authors observe the ratio between the norm of layer weights and the norm of gradient updates. They find that if this ratio is too large training becomes unstable. 
They propose LARS to fix this. In essence LARS sets a separate learning rate for each *layer* which aids in stability and allows large batch training (32K) without accuracy loss.

[[Train Longer Generalize Better]] tried square root scaling of LR with "Ghost Batch Norm" but despite that the accuracy still dropped by ~4 percentage points. 

### LARS

Uses local LR $\lambda^l$ for the respective layer $l$ which is computed by dividing the norm of the weights in layer l by the norm of its gradients, multiplied with a "trust coefficient". This can be extended to incorporate weight decay.

