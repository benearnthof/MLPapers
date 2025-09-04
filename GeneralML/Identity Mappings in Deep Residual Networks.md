https://arxiv.org/abs/1603.05027

Paper by Kaiming He, following up on their seminal work in [[Deep Residual Learning for Image Recognition]] by presenting a new "pre activation" residual unit that outperforms other skip connection types like scaling, gating, and 1x1 convolutions. 
[[Long Short-Term Memory]]
[[Highway Networks]]
[[Training Very Deep Networks]]

The authors make the argument that without residual connections the feature computations of a plain network (when omitting the [[Batch Normalization]] and [[Rectified Linear Units]]) boils down to a series of matrix-vector *products*.
When introducing residual units we can obtain a nice additive form for the feature computation and backward pass. This, they argue, implies that gradients of a layer never vanish even when the weights become arbitrarily small, which in turn greatly aids optimization during training.

When generalizing residual connections from identity mappings to general mappings of the form 
$$
\mathbf{x}_{l+1}=\lambda_l \mathbf{x}_l+\mathcal{F}\left(\mathbf{x}_l, \mathcal{W}_l\right),
$$
Where $\lambda_l$ is a multiplicative factor, we can obtain the output of any layer $L$, again omitting BN and ReLU layers, like so:
$$
\mathbf{x}_L=\left(\prod_{i=l}^{L-1} \lambda_i\right) \mathbf{x}_l+\sum_{i=l}^{L-1} \hat{\mathcal{F}}\left(\mathbf{x}_i, \mathcal{W}_i\right)
$$
This in turn yields the gradient, w.r.t the parameters of the layer $x_l$ as: 
$$
\frac{\partial \mathcal{E}}{\partial \mathbf{x}_l}=\frac{\partial \mathcal{E}}{\partial \mathbf{x}_L}\left(\left(\prod_{i=l}^{L-1} \lambda_i\right)+\frac{\partial}{\partial \mathbf{x}_l} \sum_{i=l}^{L-1} \hat{\mathcal{F}}\left(\mathbf{x}_i, \mathcal{W}_i\right)\right)
$$
Meaning that for any factor $\lambda \neq 1$ one may obtain arbitrarily large (exploding) or arbitrarily small (vanishing) gradients by constructing sufficiently deep networks. Both of these situations ultimately result in network behavior that greatly complicates training.

The authors experiment with various types of skip connections: 
* Additive 
* constant scaling
* exclusive gating
* shortcut-only gating
* conv shortcuts [[Network In Network]], [[Going Deeper with Convolutions]]
* [[Dropout]] shortcuts

They also benchmark the order of operations for additive residual units, where the full pre-activation block scores best. x_out = x_in + BN(ReLU(weight(BN(ReLU(weight(x))))))

They argue pre-activation eases optimization and improves regularization. (I mean technically improving regularization would directly lead to easier optimization so they are kind of two sides of the same coin). 

Training details: 
Their largest ResNet has 1001 layers, summing up to about 10M parameters. Training Hyperparameters are as follows:
* Translation and flipping augmentation
* Learning rate 0.1 (!) wtf
* learning rate warmup for 400 iterations and learning rate decay at 32k and 48k iterations respectively. 
* Batch size of 128 on two GPUs in parallel. 
* Weight decay of 0.0001
* momentum of 0.9
* weight initialization according to [[Delving Deep into Rectifiers]]

On imagenet they use 8GPUs with batch size 32 each
