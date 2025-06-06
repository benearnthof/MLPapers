Physical Phenomenon: Brownian Motion with Drag 

Ornstein-Uhlenbeck process models a mean-reverting stochastic system 
(position of particle buffeted by random hits and pulled towards stable equilibrium by a linear force)

$d X_t=\theta\left(\mu-X_t\right) d t+\sigma d B_t$

Where $\theta$ is the speed of reversion, $\mu$ is the long-term mean, $\sigma$ corresponds to volatility (noise) and ${B_{t}}$ is standard Brownian motion.

This is directly related to the Langevin Equation: 
$m \frac{d v}{d t}=-\gamma v+\eta(t)$
Where a Hooke-like drag or friction force acts on the velocity of a particle and $\eta(t)$ is modeled as white noise. Simply dividing by m and using basic stochastic calculus we obtain a SDE in Ornstein-Uhlenbeck form.

Applications of this can be found in Finance (Interest rate models, volatility models), Neuroscience (Membrane potential dynamics), Biology (Trait evolution under stabilizing selection, phylogenetic comparative methods), Control theory (noisy sensors and regulators)

