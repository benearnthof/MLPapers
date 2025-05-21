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

Motivating Idea is the 