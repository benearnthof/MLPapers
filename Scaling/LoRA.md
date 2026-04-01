Low-Rank Adaptation of Large Language Models
https://arxiv.org/abs/2106.09685

General Idea: Freeze pre-trained model weights and inject trainable rank decomposition matrices into each layer of the Transformer.
LoRA performs on-par or better than fine-tuning in model quality on [[RoBERTa]], [[DeBERTa]], [[GPT-2]], & [[GPT-3]], despite having fewer trainable parameters, higher throughput, &, unlike adapters, no additional inference latency.

In a nutshell we keep the pretrained weight matrices of each layer frozen, but add a low-rank decomposition of each W to the forward pass:
$h=W_0 x+\Delta W x=W_0 x+B A x$
Where $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$, and the rank $r \ll \min (d, k)$.

The decomposition allows us to avoid incurring inference latency since we can just calculate $W_{task} = W_0 + BA$ and perform inference as normal. For a separate task one would add $B^{\prime} A^{\prime}$ respectively.

