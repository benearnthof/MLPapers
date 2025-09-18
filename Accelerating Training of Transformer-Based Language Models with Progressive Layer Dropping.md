https://arxiv.org/pdf/2010.13369

Performance Claim: 25% of time reduction per sample
2.5x faster pretraining than baseline while maintaining similar accuracy on downstream tasks

The authors cite [[On Large-Batch Training for Deep Learning]] to argue that large batch training leads to sharp local minima with poor generalizability ? This is no longer an issue with specific large batch techniques. 

The central idea of this paper, Progressive Layer Dropping, is inspired by [[Deep Networks with Stochastic Depth]].
They propose a layer dropping schedule that both slowly increases the probability of dropping out a layer as training progresses *and* distributes a global layer dropping rate across all of the blocks in the transformer to favor different layers. 

They include a discussion on PreLN vs PostLN models, noting that [[Layer Normalization]] before the rest of the block performs better overall. 

In essence the Switchable Transformer Block specified in the paper adjusts the usual forward:
```python
def forward(self, x):
	# Standard forward pass
	x = x + self.attn(self.ln_1(x))
	x = x + self.mlp(self.ln_2(x))
	return x
```
To a "Gated" mechanism that stochastically drops out each sublayer like so: 
$$
\begin{aligned}
h_i & =x_i+G_i \times f_{S-A T T N}\left(f_{L N}\left(x_i\right)\right) \times \frac{1}{p_i} \\
x_{i+1} & =h_i+G_i \times f_{F F N}\left(f_{L N}\left(h_i\right)\right) \times \frac{1}{p_i}
\end{aligned}
$$
Where the factor $G_i$ is drawn from a Bernoulli distribution with parameter $p_i$ equivalent to the current schedule. $G_i \sim B\left(1, p_i\right)$ The $p_i$ also serves as a normalization factor to keep the scale of each input/output roughly equivalent through the network, as during inference all layers are always kept. 
The layer then looks like this: 
```python
def forward(self, x):
	p_i = self.pld_scheduler()
	G_i = torch.bernoulli(p_i)
	x = x + G_i * self.attn(self.ln_1(x)) / p_i
	x = x + G_i * self.mlp(self.ln_2(x)) / p_i
	return x 
```

### A Progressive Layer Dropping Schedule
Inspired by prior work on curriculum learning [[Curriculum Learning]] the authors propose a progressive schedule $\theta(t)$ -- a temporal schedule for the expected number of ST blocks that are retained. Starting from the initial condition $\theta(0) = 1$ where no layer drop is performed, layer ddrop is gradually introduced. Eventually (i.e., when t is sufficiently large) $\theta(t) \rightarrow \bar{\theta}$ where $\bar{\theta}$ is a limit value, to be taken as $0.5 \leq \bar{\theta} \leq 0.9$ 
The authors use the following function: 
$\bar{\theta}(t)=(1-\bar{\theta}) \exp (-\gamma \cdot t)+\bar{\theta}, \gamma>0$
Where $\gamma$ is following the heuristic $\gamma = \frac{100}{T}$ where it is assumed that $T$ is in the order of 10^4 to 10^5 when training transformer networks. 

## Distribution along the depth dimension
The above schedule assumes all gates in ST blocks take the same $p$ value at each step $t$. However the lower layers of the networks should be more reliably present during training. Therefore we distribute the global $\bar{\theta}$ across the entire stack, such that lower layers have lower drop probability, linearly scaled by their depth according to: 
$$
p_l(t)=\frac{i}{L}(1-\bar{\theta}(t))
$$
Combining this with the proposed scheduler we obtain: 
$$
\theta_i(t)=\frac{i}{L}(1-(1-\bar{\theta}(t)) \exp (-\gamma \cdot t)-\bar{\theta}(t))
$$
