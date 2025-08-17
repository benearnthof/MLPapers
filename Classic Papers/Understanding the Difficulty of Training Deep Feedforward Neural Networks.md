https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
Another Banger Classic Paper I Need to revisit.

Introduces Xavier Initialization, He init was introduced later for the training of networks using ReLU activation functions. 

[[Deep Sparse Rectifier Neural Networks]] is authored by the same researchers.

Historical Background: Many recent (prior to 2006) methods struggled with consistently, or even successfully at all, training deep neural networks. Many recent advances highlighted the success of deep architectures. The authors observe that the choice of activation function may lead some neurons to saturate, destabilizing the training process. They note that these neurons are able to desaturate on their own, albeit slowly. They add that training is more difficult when the singular values of the Jacobian associated with each layer are far from 1, and propose a novel layer initialization scheme to ameliorate these issues. 

[[The Difficulty of Training Deep Architectures]] already demonstrated that unsupervised pre-training acts as a regularizer that initializes the parameters in a "better" basin of attraction of the optimization procedure. This paper focuses solely on the effects of depth on the training of deep NNs.

Core idea: Unsupervised pretraining acts as a form of initialization and has a drastic impact on performance. A good and bad initialization scheme should have equally drastic impacts on performance. 

It's quite astounding to see how far we have come in less than 20 years. They publish a paper with experiments done on mnist, cifar 10, and Small-ImageNet. They use up to 5 hidden layers of dimension 1000, softmax logistic regression as output layer, negative log-likelihood loss and batch size of 10. They perform learning rate fine tuning by validation set error after 5 million updates. 

The authors compare sigmoid, tanh, and softsign activation functions. 
Biases were initialized as 0, Weights were initialized uniformly. 

We now know that this is not ideal but it is interesting to see that an obvious maxentropy choice like the multivariate normal is not far older. 

[[Efficient BackProp]] already showed in 1998 that the sigmoid activation function can induce problematic singular values in the Hessian of any layer.

The authors demonstrate that the activations, even for relatively shallow networks, tend to saturate for sigmoid activations, slowing down training by drowning out gradient flow. 
Tanh provides stronger gradients. 

But even for the tanh the layers saturate, one after another. Softsign also saturates, but here the layers do so in unison. 

The authors show that when analyzing the variance of gradients during BackProp, they may still vanish or explode for deep networks, despite identical initialization and layer width. This is Similar to a problem observed for RNNs.  [[Learning Long-Term Dependencies with Gradient Descent is Difficult]]

They note that cross entropy loss should lead to less plateaux.

The authors derive the now well known Xavier Initialization:
$$
W \sim U\left[-\frac{\sqrt{6}}{\sqrt{n_j+n_{j+1}}}, \frac{\sqrt{6}}{\sqrt{n_j+n_{j+1}}}\right]
$$
Where n is the layer size (assuming all layers are of the same size).

