https://proceedings.neurips.cc/paper_files/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf
Classic Paper that uses information-theoretic ideas to derive nearly optimal schemes for adapting the size of a neural network.

By removing unimportant weights from a network one can expect better generalization, smaller dependency on training data, improved convergence rates. 
They use second order information to make a tradeoff between network complexity and training set error. 

Considerations inspired by regularization literature: Minimizing a cost function comprised of the sum of training error and some measure of the network complexity. (Penalization term, Lasso, Ridge, Elastic Net, etc.) Later literature builds on this with various domain specific regularization methods. [[Akaike Information Criterion]]

Naïve deletion of the smallest parameters and then retraining amounts to simple continuous weight decay during training. 
Such pruning can increase robustness of the network but slows down training and requires fine tuning of the pruning coefficient to avoid catastrophic effects.

*One of the main points of this paper is to move beyond the approximation that "magnitude equals saliency", and propose a theoretically justified saliency measure*.

### Optimal Brain Damage
The core Idea is to use the amount of change in the objective function caused by deletion of a parameter as a measure of saliency. Since it would be almost infeasible to evaluate the objective function on the entire training set for every individual parameter, the authors approximate this with a second order Taylor expansion. 

Since the hessian is enormous even for sub 10k parameter networks they use diagonal approximation and "extremal" approximation, in the sense that we want to prune after training, thus it is reasonable to assume that we are in a local optimum of the loss landscape. 

This reduces the approximation of the loss function w.r.t. the parameters to $\delta E=\frac{1}{2} \sum_i h_{i i} \delta u_i^2$.

Now we need a fast method to compute the diagonal entries (the second order gradients.)
Fortunately this is of the same order of complexity as first order gradients, since we're only interested in diagonal entries and are working with simple feedforward networks in this case.

### OBD Recipe: 

1. Pick network architecture
2. Train to convergence
3. Compute second derivatives $h_{kk}$ for each parameeter
4. Compute the saliencies for each parameter $s_k = h_{kk}u^2_k/2$
5. Delete the parameters of lowest saliency
6. Iterate to step 2

This cannot exploit modern hardware efficiently though, but is quite useful for model compression.

In the published version of the paper the authors use second order techniques to derive a measure of informational content of a model, which touches on the principle of [[Minimal Description Length]]
