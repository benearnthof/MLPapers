Loss Deceleration and Zero-Sum Learning
https://arxiv.org/pdf/2506.05447

How exactly does scaling aid in improving language models? 
The authors note empirically that *loss deceleration* occurs during training: 
Loss deceleration, the slowdown of optimization after an initial phase of rapid improvement. They attribute this to what they call *Zero-sum Learning* or the fact that once a smaller model is saturated per sample gradient updates are opposed to one another, causing the model to stall in a local optimum. They find that scaling the model mitigates this by "decreasing the loss at which deceleration occurs" and "improving the log-log linear rate at which loss decreases after deceleration of convergence."

Initially I'm skeptical of these claims as larger models have more capacity, my gut feeling is that this serves as some kind of implicit regularization but let's see how their claims hold up.

It is empirically known that scaling up models increases performance in terms achieved cross entropy loss with power-law behavior. [[Scaling Laws for Neural Language Models]]

While some other works try to address the theoretical understanding in terms of model capacity and learning theoretic analyses [[Learning Curve Theory]], [[Explaining Neural Scaling Laws]], [[The Quantization Model of Neural Scaling]], [[Scaling Laws from the Data Manifold Dimension]], this work focuses on the impact of scaling data size and model capacity on the training dynamics.

