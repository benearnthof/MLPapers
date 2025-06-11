https://arxiv.org/abs/2210.02747
https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html#gaussian-probability-paths
https://ai.meta.com/research/publications/flow-matching-guide-and-code/
https://dl.heeere.com/conditional-flow-matching/blog/conditional-flow-matching/
https://rectifiedflow.github.io/
https://diffusionflow.github.io/
https://diffusion.csail.mit.edu/docs/lecture-notes.pdf

## Introduction

Flow matching is a novel framework for generative modeling that leverages insights from statistical physics, differential geometry, and statistical learning to arrive at a method for learning statistical distributions that outperforms other methods and pushes the state-of-the-art in fields such as conditional generation of images [[Stable Diffusion 3]], viedeos [[Movie Gen]], speech [[Voicebox]], audio [[Audiobox]], robotics [[Pi_zero]]], and proteomics [[SE(3) Flow Matching]].

Most important questions: 
	* Design Choices for Flow Matching
	* Can Flow Matching be extended to domains other than images?
	* How to actually implement Flow Matching with PyTorch
	* Mathematical Derivation of Flow Matching and Connections to other Generative Models

Flow Matching is based on learning a velocity field / vector field. Each velocity field defines a flow $\psi_{t}$ by solving (integrating) an [[ODE]]. 
A flow $\psi$ is a **deterministic**, **time-continuous**, and **bijective** transformation of the d-dimensional Euclidean space $\mathbb{R}^d$. 
The goal of Flow Matching is to learn a flow in such a way, that it transforms any sample $X_0 \sim p$ drawn from a source distribution $p$ into a target sample $X_1:=\psi_1\left(X_0\right)$ such that the resulting target distribution $X_1 \sim q$ has a desired target distribution $q$. 

Flow Models are a more general class of models, the first member of this class of generative models was introduced as [[Continuous Normalizing Flows]]  [[Neural ODE]]  [[FFJORD]] 
Training CNFs requires simulation and its differentiation during training, leading these models to incur a large computational cost during training. This led directly to the development of simulation free Flow Model variants like [[Normalizing Flows on Manifolds]] and [[Moser Flow]] which, in turn, led to modern Flow Matching algorithms like [[Rectified Flow]], [[Stochastic Interpolants]], [[Action Matching]], [[Minibatch OT]], and [[Iterative alpha Blending]].

A minimal Flow Matching implementation consists of two steps: 
1. Pick a probability path that interpolates between source $p$ and target $q$ distributions. 
2. Train a velocity field (or a neural network approximation of it) that defines the flow transformation $\psi_{t}$ and implements $p_t$

The Flow Matching paradigm can easily be extended to accommodate discrete state spaces [[Discrete State Spaces]], inductive priors such as Riemannian Manifolds[[Flow Matching on General Geometries]], and Continuous Time Markov Processes [[Generator Matching]], where, up to differences in the neural architecture, the Flow Matching recipe stays the same. 

Diffusion Models can be seen as a special case of Flow Matching, depending on the chosen formulation we arrive at several connections:
	* [[Nonequilibrium Thermodynamics]] and [[DDPM]] introduce Diffusion Models as discrete time Gaussian Processes
	* [[Score-based generative modeling SDEs]] introduce them as solutions to [[SDE]]s 

Diffusion Models build the probability path $p_t$ by interpolating between target and source by a forward noising process, that can be modeled by specific [[SDE]]s. These SDEs are chosen to have marginal probabilities obtainable analytically as a closed form solution, and thus can be used to parametrize the generator of the diffusion process (drift and diffusion coefficients) via the score function [[Score Matching]]. This parametrization is based on a reversal of the forward process. [[Reverse-Time Diffusion Equation Models]]. 

Diffusion models learn the score function of the marginal probabilities. It can be shown that this is, up to normalizing constants, equivalent to learning to predict the noise, denoising, and v-prediction. [[Variational Diffusion Models]] [[Progressive Distillation]] 
[[Diffusion Bridges]] design $p_t$ by extending diffusion models to arbitrary source-target couplings, again leveraging [[SDE]]s with suitable marginals, a clever application of [[Doob's h-transform]].
[[Schrödinger Bridge Matching]] shows that linear versions of Flow Matching arise as limiting cases of Bridge Matching. 
## Key Concepts

Given access to a training dataset of samples from a target distribution $q$ over $\mathbb{R}^d$ we want to train a model capable of generating new samples from $q$. Flow Matching does this by constructing a **probability path** $\left(p_t\right)_{0 \leq t \leq 1}$ from a knows source distribution $p_0 = p$ to the target data distribution $p_1$, where each $p_t$ is a distribution over $\mathbb{R}^d$ itself. Specifically, FM is trained via a simple regression objective to train a neural network to predict the instantaneous velocities of samples in $\mathbb{R}^d$ given the current sampling time ${0 \leq t \leq 1}$. After training the velocity prediction network to completion we can then sample from the target distribution by first sampling from the source at time $t = 0$, and then following the respective probability path by integrating along the velocity field, much like we would with a numerical ODE solver, by repeatedly taking small steps along the current velocity predicted by the network, and updating the sample position accordingly. If a network of sufficient capacity has been trained on sufficient data this will then yield the desired sample from the target distribution, up to the numerical errors compounded during integration. We will discuss considerations on picking optimal step sizes, and how to decrease the total number of integration steps later in detail. 

More specifically: 
Any Ordinary Differential Equations ([[ODE]]) can be represented by a time-dependent vector field  
	$u:[0,1] \times \mathbb{R}^d \rightarrow \mathbb{R}^d$
Which, in our case, is parametrized as a Neural Network and optimized during training. 
This vector field determines a time-dependent **flow** $\psi:[0,1] \times \mathbb{R}^d \rightarrow \mathbb{R}^d$, defined as 
	$\frac{\mathrm{d}}{\mathrm{d} t} \psi_t(x)=u_t\left(\psi_t(x)\right)$,
Where $\psi_t:=\psi(t, x)$ and $\psi_0(x)=x$. Thus, at time $t < 1$ the velocity field $u_t$ will transport a sample in the to the target distribution, generating the probability path $p_t$ if its flow $\psi_t$ satisfies
	$X_t:=\psi_t\left(X_0\right) \sim p_t$ for $X_0 \sim p_0$.
This means that, in order to sample from $p_t$ we only have to solve the ODE by integrating along our parametrized velocity field $u_t^\theta$. 
By parametrizing the velocity field in terms of a neural network, we can then use standard stochastic optimization techniques to train this network and obtain a close approximation of this vector field. Here one may note that these claims only hold if we choose an appropriate training objective. We will discuss this in detail later on, with particular focus on data dependend coupling [[Stochastic Interpolants]] and the Conditional Flow Matching Objective. 

#### Designing the Probability Paths

In the simplest case of Flow Matching we define our source distribution as a simple multivariate Gaussian with mean $0$ and identity covariance matrix $I$.
	$p:=p_0=\mathcal{N}(x \mid 0, I)$
Now we construct the probability path $p_t$ as the aggregation of the conditional probability paths $p_{t \mid 1}\left(x \mid x_1\right)$, each conditioned on one of the data samples $X_1 = x_1$ comprising the training set. 

Using the definition of continuous conditional distributions 
	$f_{Y \mid X}(y \mid x)=\frac{f_{X, Y}(x, y)}{f_X(x)}$,
and rearranging to
	$f_{Y \mid X}(y \mid x) f_X(x)=f_{X, Y}(x, y)=f_{X \mid Y}(x \mid y) f_Y(y)$
we obtain expressions for the joint density. Applying this, we obtain an analogous expression for the joint density of our probability path:
	$p_{t \mid 1}\left(x \mid x_1\right) q\left(x_1\right) \mathrm{d} x_1$
We can then finally arrive at the desired solution for our target probability path by integration: 
	$p_t(x)=\int p_{t \mid 1}\left(x \mid x_1\right) q\left(x_1\right) \mathrm{d} x_1$
Where $p_{t \mid 1}\left(x \mid x_1\right)=\mathcal{N}\left(x \mid t x_1,(1-t)^2 I\right)$. 

This is also known as the *conditional optimal-transport* or *linear* path which has a couple of desirable properties that we will later discuss. 

Such a probability path we can define a random variable $X_t \sim p_t$ by simply sampling from source 
$X_0 \sim p$ and target $X_1 \sim q$ and linearly interpolating between them: 
	$X_t=t X_1+(1-t) X_0 \sim p_t$

Constructing a Loss for this appears intractable at first but one can exploit conditioning to arrive at a surrogate objective that leads to the same gradients (up to a constant). We will discuss this in detail later. 
## Flow Models

This section introduces Flows as general mathematical objects. This will encompass introducing them as the simplest, deterministic form of Continuous Time Markov Processes, highlighting their capability of transforming any source into any target distribution, their efficient sampling, and, finally, their unbiasedness when it comes to likelihood estimation. 

#### Mathematical Background

For now we are going to introduce Flow Matching for $d$-dimensional source distributions mapping to $d$-dimensional target samples. We endow a standard $d$-dimensional Euclidean space $x=\left(x^1, \ldots, x^d\right) \in \mathbb{R}^d$ with the standard inner product $\langle x, y\rangle=\sum_{i=1}^d x^i y^i$ and norm $\|x\|=\sqrt{\langle x, x\rangle}$. 
Further, consider random variables $X \in \mathbb{R}^d$ with *continuous* probability density functions $p_X: \mathbb{R}^d \rightarrow \mathbb{R}_{\geq 0}$ such that we obtain the usual [[Measure Theoretic Probability]]  of an event $A$ by integrating over the respective set A: $\mathbb{P}(X \in A)=\int_A f(x) d x$ 

Notation for PDF of RV $X_t$: $p_t$

Most common source distribution is the d-dimensional isotropic Gaussian: 

	$\mathcal{N}\left(x \mid \mu, \sigma^2 I\right)=\left(2 \pi \sigma^2\right)^{-\frac{d}{2}} \exp \left(-\frac{\|x-\mu\|_2^2}{2 \sigma^2}\right)$
With known mean vector $\mu$ and diagonal covariance matrix $\sigma I$. 
Not only does it have a lot of nice mathematical properties that make marginalization easy and closed form solutions possible, it is also the highest entropy distribution for the class of distributions with equivalent moments. This will become important later. 

The expectation of a RV is the constant vector closest to X in the least-squares sense: 

	$\mathbb{E}[X]=\underset{z \in \mathbb{R}^d}{\arg \min } \int\|x-z\|^2 p_X(x) \mathrm{d} x=\int x p_X(x) \mathrm{d} x$

Proof for the one dimensional case: 
Let $c \in \mathbb{R}$ and define: 
	$f(c):=\mathbb{E}\left[(X-c)^2\right]$
To find the value of c that minimizes this expression we differentiate with respect to c and set to zero as usual: 
	$f^{\prime}(c)=\mathbb{E}[-2(X-c)]=-2 \mathbb{E}[X]+2 c$  
	$-2 \mathbb{E}[X]+2 c=0 \quad \Rightarrow \quad c=\mathbb{E}[X]$
We see that the Expectation functional minimizes the squared difference.
Note: One could also use tools from Functional Analysis to show that Expectation is an orthogonal projection onto the subspace of constant random variables, meaning that we obtain a vector that minimizes squared differences. This also extends to conditional expectation where we obtain conditional expectations as the best predictor of the original random variable that is measurable in the sub sigma algebra we condition on. 

To compute the expectation of functions of RVs we can simply integrate like so: 
	$\mathbb{E}[f(X)]=\int f(x) p_X(x) \mathrm{d} x$
Which does require a theorem to prove but is commonly just known as the *Law of the Unconscious Statistician*.

Two random variables $X, Y \in \mathbb{R}^d$ have a joint density $p_{X, Y}(x, y)$. To obtain the marginal distribution for either of them, we integrate with respect to the other like so: 
	$\int p_{X, Y}(x, y) \mathrm{d} y=p_X(x)$ and $\int p_{X, Y}(x, y) \mathrm{d} x=p_Y(y)$
We obtain the conditional PDFs via Bayes's rule: 
	$p_{Y \mid X}(y \mid x)=\frac{p_{X \mid Y}(x \mid y) p_Y(y)}{p_X(x)}$
As mentioned above, the conditional expectation is the best approximating function when conditioning on prior information. $\mathbb{E}[X \mid Y=y]$ and $\mathbb{E}[X \mid Y]$ are two different objects: 
$\mathbb{E}[X \mid Y=y]$ is a function $\mathbb{R}^d \rightarrow \mathbb{R}^d$ since we condition on a single realization $y$, whereas
$\mathbb{E}[X \mid Y]$ is a random variable that takes values in $\mathbb{R}^d$. 
This is useful in the tower property (Law of total expectation):
	$\mathbb{E}[\mathbb{E}[X \mid Y]]=\mathbb{E}[X]$

Push forward of a RV: 
given a RV $X \sim p_X$ we can construct another RV $Y=\psi(X)$, where $\psi: \mathbb{R}^d \rightarrow \mathbb{R}^d$ is a $C^1$ diffeomorphism (an invertible function with continuous partial derivatives of order 1). 
Because just mapping X to $\psi(X)$ does not guarantee that Y has a valid PDF we must correct for potential nonlinear transformations by locally approximating such transformations linearly via the Jacobian. This is also why we require the existence of partial derivatives of order 1 -- to guarantee the existence of the Jacobian. https://angeloyeo.github.io/2020/07/24/Jacobian_en.html 
Using the change of variables formula we get: 
	$\mathbb{E}[f(Y)]=\mathbb{E}[f(\psi(X))]=\int f(\psi(x)) p_X(x) \mathrm{d} x=\int f(y) p_X\left(\psi^{-1}(y)\right)\left|\operatorname{det} \partial_y \psi^{-1}(y)\right| \mathrm{d} y$
Leading us to the PDF  of $Y$ is: 
	$p_Y(y)=p_X\left(\psi^{-1}(y)\right)\left|\operatorname{det} \partial_y \psi^{-1}(y)\right|$.
Which is in general shortened to the push-forward operator: 
	$\left[\psi_{\sharp} p_X\right](y):=p_X\left(\psi^{-1}(y)\right)\left|\operatorname{det} \partial_y \psi^{-1}(y)\right|$.

#### Flows as generative Models
A **flow** is a time-dependent mapping $\psi:[0,1] \times \mathbb{R}^d \rightarrow \mathbb{R}^d$ implementing $\psi:(t, x) \mapsto \psi_\iota(x)$. Such that the function $\psi_\iota(x)$ is a $C^r$ diffeomorphism in $x$ for all $t \in[0,1]$.
A **flow model** is a *continuous-time Markov process* $\left(X_t\right)_{0 \leq t \leq 1}$ defined by applying a flow $\psi_t$ to the RV $X_0$: 
	$X_\iota=\psi_\iota\left(X_0\right), \quad t \in[0,1]$, where $X_0 \sim p$.
The Markov property holds because for any choice of $0 \leq t<s \leq 1$, we obtain 
	$X_s=\psi_s\left(X_0\right)=\psi_s\left(\psi_t^{-1}\left(\psi_\iota\left(X_0\right)\right)\right)=\psi_{s \mid t}\left(X_\iota\right)$
which implies that states later than $t$ depend only on $X_t$.
Much more interestingly, this dependence is *deterministic* for flow models. 
(? Why is that the case?)

3.41 Equivalence between flows and velocity fields




## Flow Matching

## Non-Euclidean Flow Matching

## Continuous Time Markov Chain Models

## Discrete Flow Matching

## Continuous Time Markov Process Models

## Generator Matching  

## Relation to Diffusion and other Denoising Models

## Appendix A: Proofs

