ai.meta.com/research/publications/flow-matching-guide-and-code/
# Introduction


# Quick Tour and Key Concepts
# Flow Models

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
A **flow** is a time-dependent mapping $\psi:[0,1] \times \mathbb{R}^d \rightarrow \mathbb{R}^d$ implementing $\psi:(t, x) \mapsto \psi_t(x)$. Such that the function $\psi_t(x)$ is a $C^r$ diffeomorphism in $x$ for all $t \in[0,1]$.
A **flow model** is a *continuous-time Markov process* $\left(X_t\right)_{0 \leq t \leq 1}$ defined by applying a flow $\psi_t$ to the RV $X_0$: 
	$X_t=\psi_t\left(X_0\right), \quad t \in[0,1]$, where $X_0 \sim p$.
The Markov property holds because for any choice of $0 \leq t<s \leq 1$, we obtain 
	$X_s=\psi_s\left(X_0\right)=\psi_s\left(\psi_t^{-1}\left(\psi_t\left(X_0\right)\right)\right)=\psi_{s \mid t}\left(X_t\right)$
which implies that states later than $t$ depend only on $X_t$.
Much more interestingly, this dependence is *deterministic* for flow models. 
(? Why is that the case?)

#### Equivalence between flows and velocity fields
We can define a flow $\psi$ in terms of a *velocity field* $u:[0,1] \times \mathbb{R}^d \rightarrow \mathbb{R}^d$
Where $u:$$[0,1]$ parametrizes the time interval, implementing $u:(t, x) \mapsto u_t(x)$
via the ODE:
	$\frac{\mathrm{d}}{\mathrm{d} t} \psi_t(x)=u_t\left(\psi_t(x)\right)$ (flow ODE)
	$\psi_0(x)=x$ (flow initial conditions)
We see that the time derivative of our flow at time $t$ is equivalent to the velocity field evaluated at time $t$.
Under the condition of local Lipschitzness a unique solution exists for this ODE locally. To guarantee the existence of flow almost everywhere and at least until time $t=1$ we will later utilize integrability. 

We have thus shown that a velocity field defines a flow uniquely, but does the converse also hold? 
For a given flow $\psi_t$ we want to extract its defining velocity field $u_t(x)$, as this would allow us to obtain a velocity field from any interpolating flow. Considering
	$\frac{\mathrm{d}}{\mathrm{d} t} \psi_t\left(x^{\prime}\right)=u_t\left(\psi_t\left(x^{\prime}\right)\right)$
we exploit the fact that $\psi_t$ is an invertible diffeomorphism for every $t \in[0,1]$. 
Let $x^{\prime}=\psi_t^{-1}(x)$ and plug this into the case above we get:
	$u_t(x)=\dot{\psi}_t\left(\psi_t^{-1}(x)\right)$,  $(*)$ 
	where $\dot{\psi}_t:=\frac{\mathrm{d}}{\mathrm{d} t} \psi_t$.
But $\frac{\mathrm{d}}{\mathrm{d} t} \psi_t(x)=u_t\left(\psi_t(x)\right)$ as defined in the flow ODE in the first place, thus, when plugging $(*)$ into our flow ODE we see that both flow $\psi_t$ and velocity field $u_t$ are equivalent. 

#### Computing target samples from source targets

Once we have trained a sufficiently capacious neural net to approximate the velocity field we can then obtain target samples by approximating the solution to the flow ODE via numerical methods for [[ODE]]s. Given some initial condition $X_0=x_0$ we could for example use the Euler method, with the update rule: 
	$X_{t+h}=X_t+h u_t\left(X_t\right)$
where $h$ is a step size hyper-parameter, usually defined by $h=n^{-1}>0$ with n defining the amount of sample steps we're willing to take. Here the Euler method coincides with the first-order Taylor expansion of $X_t$: 
	$X_{t+h}=X_t+h \dot{X}_t+o(h)=X_t+h u_t\left(X_t\right)+o(h)$, 
meaning that for smaller and smaller step sizes $h \rightarrow 0$, the approximation error we incur with the Euler method vanishes. Other solvers like the Second order Midpoint-, or Runge-Kutta-Method, are often better in practice as they have better error guarantees, allowing for a smaller number of total function evaluations.

#### Probability Paths and the Continuity Equation

#### Instantaneous Change of Variables

#### Training Flow Models with Simulation


# Flow Matching

#### Data

#### Probability Paths

#### Generating Velocity Fields

#### Conditioning and the Marginalization Trick

#### Flow Matching Loss

#### Conditional Generation

#### Optimal Transport

#### Affine Conditional Flows

#### Data Couplings

#### Guidance