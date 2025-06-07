https://arxiv.org/abs/1503.03585

Achieves both Flexibility and Tractability
Iteratively destroy structure in data through a forward diffusion process
Learn a reverse diffusion process that restores structure in data

Allows flexibly learning, sampling, and evaluation of probabilities in deep generative models
* Computation of conditional and posterior probabilities ? 

Historically probabilistic models suffer from tractability vs flexibility tradeoff: 
	At some point integrals become intractable
	MCMC can be used but suffers from similar issues in high dimensions
	Restricting ourselves to tractable models cannot express semantically rich datasets
	Methods that ameliorate these problems:
		Mean field theory 
			https://journals.aps.org/pre/abstract/10.1103/PhysRevE.58.2302
		Variational Bayes
			 https://arxiv.org/pdf/1312.6114
			 https://people.eecs.berkeley.edu/~jordan/papers/variational-intro.pdf
		Contrastive Divergence
			https://www.gatsby.ucl.ac.uk/publications/tr/tr01-002.pdf
			https://www.cs.toronto.edu/~fritz/absps/tr00-004.pdf
		Minimum Probability Flow
			https://arxiv.org/pdf/0906.4779
			https://arxiv.org/pdf/2007.09240
		KL Contraction
			https://papers.nips.cc/paper_files/paper/2011/file/a3f390d88e4c41f2747bfa2f1b5f87db-Paper.pdf
		Scoring Rules
			https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf
		[[Score Matching]]
			https://jmlr.org/papers/volume6/hyvarinen05a/hyvarinen05a.pdf
		Pseudolikelihood
			https://www2.stat.duke.edu/~scs/Courses/Stat376/Papers/GibbsFieldEst/BesagPseudoLik1975.pdf
		Belief Propagation
			https://www.cs.ubc.ca/~murphyk/papers/loopy_uai99.pdf

DPM is introduced as a novel way of defining probabilistic models that allows
	Extreme structural flexibility
	Exact sampling
	Easy multiplication with other distributions for posterior calculation
	Cheap evaluation of the model likelihood

Motivating Idea:
	Using a Markov Chain to gradually convert one distribution into another
	Derivation in Statistical Physics: https://arxiv.org/pdf/cond-mat/9707325
		Jarzynski Equation
	Application for Sequential Monte Carlo: https://arxiv.org/pdf/physics/9803008
		Annealed Importance Sampling (AIS)
		Reverse Annealing https://arxiv.org/abs/1412.8566
	DPM Gradually convert a high dimensional Gaussian into the target distribution through a diffusion process 
		Connection to [[Typicality]]
	Explicitly define the stochastic model as the endpoint of the MC
	Each step in the diffusion chain has an analytically evaluable probability 
		The full chain can be evaluated
	This method can capture data distributions of arbitrary form
		Why does this hold for arbitrary target distributions?

Related Ideas:
	Wake Sleep Algorithm https://www.cs.toronto.edu/~fritz/absps/ws.pdf
	Joint Top Down, bottom up https://ieeexplore.ieee.org/document/1640965/
	Deep Generative Stochastic Networks https://arxiv.org/pdf/1306.1091
	Bayesian Neural Networks & Density Networks https://www.inference.org.uk/mackay/ch_learning.pdf
	Generative Topographic Mapping https://www.microsoft.com/en-us/research/wp-content/uploads/1998/01/bishop-gtm-ncomp-98.pdf
	Langevin Dynamics are the stochastic realization of the Fokker-Planck equation:
		https://www.physik.uni-augsburg.de/theo1/hanggi/History/Langevin1908.pdf
		Shows how to define a Gaussian diffusion process with arbitrary target distribution as equilibrium
	Fokker-Planck optimization:
		https://www.researchgate.net/publication/2457042_Nonconvex_Optimization_Using_A_Fokker-Planck_Learning_Machine
	Kolmogorov forward and backward equations show that for many forward diffusion processes the reverse diffusion process can be described using the same functional form
		file:///D:/Downloads/1166219215-1.pdf
	

"Training Models with thousands of layers (or time steps)"
	Relation to neural differential equations? 

2.1. Forward Trajectory
	Data distribution is gradually converted into a well behaved and analytically tractable distribution by repeated application of a Markov diffusion kernel.
	The Forward Trajectory corresponds to starting at the data distribution and performing T steps of diffusion
		$q\left(\mathbf{x}^{(0 \cdots T)}\right)=q\left(\mathbf{x}^{(0)}\right) \prod_{t=1}^T q\left(\mathbf{x}^{(t)} \mid \mathbf{x}^{(t-1)}\right)$
	Gaussian Forward Diffusion Kernel:
		$q\left(\mathbf{x}^{(t)} \mid \mathbf{x}^{(t-1)}\right)=\quad \mathcal{N}\left(\mathbf{x}^{(t)} ; \mathbf{x}^{(t-1)} \sqrt{1-\beta_t}, \mathbf{I} \beta_t\right)$
	Gaussian Reverse Diffusion Kernel
		$p\left(\mathbf{x}^{(t-1)} \mid \mathbf{x}^{(t)}\right)=\mathcal{N}\left(\mathbf{x}^{(t-1)} ; \mathbf{f}_\mu\left(\mathbf{x}^{(t)}, t\right), \mathbf{f}_{\Sigma}\left(\mathbf{x}^{(t)}, t\right)\right)$
	Continuous Diffusion: 
		Limit of discrete diffusion for arbitrarily small stepsize $\beta$
		(Feller 1949): Reversal has same functional form as the forward process
		Interesting Paper on Reversibility in thermodynamics https://arxiv.org/pdf/1201.6381
		https://terpconnect.umd.edu/~cjarzyns/CHEM-CHPH-PHYS_703_Spr_20/resources/Jarzynski_AnnuRevCondMattPhys_2_329_2011.pdf
	During learning only the mean and covariance need to be estimated.
		Here one can simply plug in any general function estimator:
			MLP, regression, function fitting, unet, transformers etc.
2.3 Model Probability: 
	$p\left(\mathbf{x}^{(0)}\right)=\int d \mathbf{x}^{(1 \cdots T)} p\left(\mathbf{x}^{(0 \cdots T)}\right)$
	Seems intractable at first but with inspiration fro annealed importance sampling and the Jarzynski equation we instead evaluate the relative probability of the forward process and reverse trajectories averaged over forward trajectories
	Multiply by one: 
		$p\left(\mathbf{x}^{(0)}\right)=\int d \mathbf{x}^{(1 \cdots T)} p\left(\mathbf{x}^{(0 \cdots T)}\right) \frac{q\left(\mathbf{x}^{(1 \cdots T)} \mid \mathbf{x}^{(0)}\right)}{q\left(\mathbf{x}^{(1 \cdots T)} \mid \mathbf{x}^{(0)}\right)}$
	Swap factors:
		$=\int d \mathbf{x}^{(1 \cdots T)} q\left(\mathbf{x}^{(1 \cdots T)} \mid \mathbf{x}^{(0)}\right) \frac{p\left(\mathbf{x}^{(0 \cdots T)}\right)}{q\left(\mathbf{x}^{(1 \cdots T)} \mid \mathbf{x}^{(0)}\right)}$
	Factor integral:
		$=\int d \mathbf{x}^{(1 \cdots T)} q\left(\mathbf{x}^{(1 \cdots T)} \mid \mathbf{x}^{(0)}\right) \cdot p\left(\mathbf{x}^{(T)}\right) \prod_{t=1}^T \frac{p\left(\mathbf{x}^{(t-1)} \mid \mathbf{x}^{(t)}\right)}{q\left(\mathbf{x}^{(t)} \mid \mathbf{x}^{(t-1)}\right)}$
	In the time limit the forward and backward processes are identical, this corresponds to the case of a quasi-static process  in statistical physics.

2.4 Training
	Training amounts to maximizing the model log likelihood: 
	$\begin{aligned} L= & \int d \mathbf{x}^{(0)} q\left(\mathbf{x}^{(0)}\right) \log p\left(\mathbf{x}^{(0)}\right) \\ = & \int d \mathbf{x}^{(0)} q\left(\mathbf{x}^{(0)}\right) . \\ & \quad \log \left[\begin{array}{c}\int d \mathbf{x}^{(1 \cdots T)} q\left(\mathbf{x}^{(1 \cdots T)} \mid \mathbf{x}^{(0)}\right) . \\ p\left(\mathbf{x}^{(T)}\right) \prod_{t=1}^T \frac{p\left(\mathbf{x}^{(t-1)} \mid \mathbf{x}^{(t)}\right)}{q\left(\mathbf{x}^{(t)} \mid \mathbf{x}^{(t-1)}\right)}\end{array}\right]\end{aligned}$
	Which can be approximated with a lower bound we can derive using [[Jensens Inequality]]
	Further, this can be reduced to a sum of: 
		KL Divergence of Source and Target
		And Entropies: $H_q\left(\mathbf{X}^{(T)} \mid \mathbf{X}^{(0)}\right)-H_q\left(\mathbf{X}^{(1)} \mid \mathbf{X}^{(0)}\right)-H_p\left(\mathbf{X}^{(T)}\right)$
		This is a direct analogue of how the log likelihood bounds in Variational Bayesian Methods are derived.
	Thus we can reduce the estimation of a probability distribution to regression on functions that modifiy the mean and covariance of a sequence of Gaussians.
	Noise Scheduling Considerations: 
	The authors claim here that setting the correct diffusion rate $\beta_t$ is crucial for the performance of the model, making connections to [[Annealed Importance Sampling]] and [[Jarzynski Thermo]]. Later analysis further analyzes these claims, from the perspective of [[Flow Matching for Generative Modeling]] and [[Typicality]] This makes intuitive sense since one does not want to waste interpolation steps near the noise distribution. It is much more sensible to decrease step size near the target distribution for example. Direct connection to typicality: [[Geometric Perspective on Diffusion Models]]
	