Blogpost by Terence Tao
https://terrytao.wordpress.com/2010/10/02/245a-notes-4-modes-of-convergence/

Convergence for sequences of numbers to a number x and even vectors to a vector x is very straight forward. 
It means that for any arbitrarily small epsilon > 0 we can always find some index n for which any appropriately chosen norm between  sequence elements and x will be smaller than epsilon. 

(The Cauchy criterion applies this to consecutive sequence elements.)

However once we consider functions instead of finite vectors they may approach a limiting function $f$ in any number of inequivalent ways. 
We can interpret a function as an infinite vector of numbers, meaning that we need to be careful in the notions of convergence we define on such infinite objects. 

As undergraduates we learn about two fundamental modes of convergence:
1. Pointwise convergence: Meaning that for any arbitrary point $x \in X$ $f_n(x)$ converges to $f(x)$.
2. Uniform Convergence: Meaning that for every $\varepsilon > 0$, there exists $N$ such that for every $n \geq N$  $\left|f_n(x)-f(x)\right| \leq \varepsilon$ for **every** $x \in X$. Essentially, for uniform convergence the time $N$ at which $f_n(x)$ must be permanently $\varepsilon$-close to $f(x)$ is not permitted to depend on $x$, but must instead be chosen uniformly in $x$.

Uniform convergence implies pointwise convergence but not conversely.

The measure theoretic extensions of these modes of convergence are: 

1. We say that $f_n$ converges to $f$ pointwise almost everywhere if, for ( $\mu$ -)almost everywhere $x \in X, f_n(x)$ converges to $f(x)$. (**Convergence almost surely**)
2. We say that $f_n$ converges to $f$ uniformly almost everywhere, essentially uniformly, or in $L^{\infty}$ norm if, for every $\varepsilon>0$, there exists $N$ such that for every $n \geq N,\left|f_n(x)-f(x)\right| \leq \varepsilon$ for $\mu$-almost every $x \in X$.
3. We say that $f_n$ converges to $f$ almost uniformly if, for every $\varepsilon>0$, there exists an exceptional set $E \in \mathcal{B}$ of measure $\mu(E) \leq \varepsilon$ such that $f_n$ converges uniformly to $f$ on the complement of $E$.
4. We say that $f_n$ converges to $f$ in $L^1$ norm if the quantity $\left\|f_n-f\right\|_{L^1(\mu)}=\int_X\left|f_n(x)-f(x)\right| d \mu$ converges to 0 as $n \rightarrow \infty$ (**Convergence in mean**)
5. We say that $f_n$ converges to $f$ in measure if, for every $\varepsilon>0$, the measures $\mu\left(\left\{x \in X:\left|f_n(x)-f(x)\right| \geq \varepsilon\right\}\right)$ converge to zero as $n \rightarrow \infty$. (**Convergence in probability**)

We can easily show that these notions of measure have the linearity property. 

To put it more informally: when the height goes to zero, then one has convergence to zero in all modes except possibly for $L^1$ convergence, which requires that the product of the height and the width goes to zero. If instead the height is bounded away from zero and the width is positive, then we never have uniform or $L^\infty$ convergence, but we have convergence in measure if the width goes to zero, we have almost uniform convergence if the tail support (which has larger measure than the width) has measure that goes to zero, we have pointwise almost everywhere convergence if the tail support shrinks to a null set, and pointwise convergence if the tail support shrinks to the empty set.

