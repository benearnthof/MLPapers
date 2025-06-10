https://arxiv.org/pdf/2305.19947

Direct connection to Stochastic Differential Equations (SDEs) and other [[Marginal Preserving ODEs]]
Neatly unifies generative models in a single, broader (in the sense of less informative inductive priors) framework.

GPDM analyzes ODE-based sampling of Variance exploding [[Score-based generative modeling SDEs]] SDE formulation and highlights structures in their sampling dynamics.

Source and Target distributions are smoothly connected with both
	Quasi-Linear sampling trajectory and
	Implicit denoising trajectory (that converges faster)

Denoising trajectory governs curvature of corresponding sampling trajectory 
Finite differences of Denoising trajectory lead yield various [[Second Order Samplers]] used in practice.

Theoretical connection between optimal ODE-based sampling and classic [[Mean-Shift Algorithm]]

Once a denoising model (with sufficient capacity) has been trained we can then:
	Run SDE backwards with numerical solvers
	Simulate the corresponding marginal preserving ODE and obtain deterministic samples [[Score-based generative modeling SDEs]] [[Karras Elucidating]]

Variance Exploding variation [[Karras Elucidating]] is used as a basis of the paper

Figure 1, Geometric Insight: 
	Initial samples from the noise distribution are drawn and then transported to the data distribution by the trained denoising model. Why is the Noise distribution a "shell" around the data distribution? Because high dimensional gaussians have a typical set [[Typicality]] that resembles a high dimensional "soap bubble", "annulus", or the shell of a hypersphere. 
	This also removes space for potential misinterpretation of the "Sand Castle Analogy" for OT based [[Flow Matching for Generative Modeling]]. The source distribution already surrounds the data distribution, since the data distribution is embedded on a lower dimensional submanifold in data space. We can map from any source point to at least some data point, we only need some form of conditioning or [[Guidance]] to control the generative process.
	This also directly underscores the potential benefits of noise scheduling [[Nonequilibrium Thermodynamics]] or varying step sizes when integrating the (sampling from) the ODE: 
	We want to spend as little time near the noise shell as we can and invest denoising steps near the data manifold. But we must also be careful to avoid overshooting the data manifold, which would lead to paths that are not straight and thus also more expensive to compute.

The (straight) implicit denoising trajectory leads directly to a point very close to the data manifold. [[Rectified Flow]]

Each Euler step $t$ during sampling moves the given sample to an updated location that is a convex combination of annealed mean shift and its current position $t_{-1}$ [[Mean-Shift Algorithm]]
This also guarantees (under mild conditions) that the likelihood of the sample at point $t_{+1}$ is always greater than the likelihood of the sample at point $t$.

This also sheds light on [[Consistency Models]] and latent interpolation with trained models.

2: Preliminaries
	