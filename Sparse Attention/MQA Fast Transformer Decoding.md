https://arxiv.org/abs/1911.02150
A more efficient alternative to vanilla MHA, an extension of this idea is [[GQA Generalized Multi Query Attention]]
## Abstract
Transformer training is fast and efficient because of the multiple dimensions of parallelism exploitable in modern architectures. Autoregressive inference however is slow, because we cannot effectively parallelize across the sequence length, thus requiring repeated loading of large key and value tensors. Multi Query Attention (MQA) shares key and values across all of the different heads, drastically decreasing the memory bandwidth requirements of incremental decoding. 

## Introduction
Transformers [[Attention Is All You Need]], have emerged as the state of the art for sequence models. 
MQA shares key and value tensors across heads, greatly speeding up inference with only minor quality degradation. 

## Background: Neural Attention
Neural Attention, originally proposed by Bahdanau eta al in [[Neural Machine Translation by Jointly Learning to Align and Translate]] takes in a query vector q and produces an output vector y by multiplying q with the key and value matrices K and V. The output y is computed as a weighted sum of the different value vectors, where weights are derived by comparing the query to the keys. 

### Dot-Product Attention 
In the basic MHA implementation we simply perform one big linear projection on the inputs in the beginning, and then split the outputs first in three chunks of size n_embd. 
Each of the kqv chunks of size [batch, sequence_length, n_embed] is then reshaped to [batch, nhead, sequence_length, n_embed//nhead]. These chunks are then processed in parallel.
```python
	...
	self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False) 
	...

def forward(self, x):
	B, T, C = x.size() # batch_size, seq_len, embedding_dim
	# calculate query, key, values for all heads in batch and move head forward to be the batch dim
	q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
	k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
	q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
	v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

	# causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
	if self.flash:
		# efficient attention using Flash Attention CUDA kernels
		y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
	else:
		# manual implementation of attention
		att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
		att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
		att = F.softmax(att, dim=-1)
		att = self.attn_dropout(att)
		y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
	y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
```
This is already batched in two different ways: First, we generate queries from n different positions in the input sequence via causal mask, second we process multiple batches of independent sequences at once.

This is fine for training as input sequences are known ahead of time and have a known maximum length, but for incremental decoding the outputs of each iteration are passed into the model again and again, each subsequent step requiring us to update the k, q and v matrices. At inference time these memory accesses become a major bottleneck. 
This paper posits the reduction of the sizes of the K and V tensors by removing their "heads" dimension, while maintaining the "heads" dimension in the queries. 

## Multi Query Attention

MQA is identical to the batched MHA, except for the fact that different heads share a single set of keys and values. 

```python
class  MultiQueryAttention(Attention):
    r"""
    https://arxiv.org/pdf/1911.02150.pdf
    """
    def __init__(self, word_size, embed_dim, n_query:int=8) -> None:
        super().__init__(word_size, embed_dim)
        self.n_query = n_query
        self.proj = nn.Linear(embed_dim * n_query, embed_dim)
        self.querys = nn.ModuleList([
            nn.Linear(in_features=word_size, out_features=embed_dim, bias=True)
            for _ in range(n_query)
        ])
        self.key = nn.Linear(word_size, embed_dim)
        self.value = nn.Linear(word_size, embed_dim)

    def forward(self, x: Tensor, mask:Optional[BoolTensor]=None) -> Tensor:
        K = self.key(x)
        V = self.value(x)
        Z_s = torch.cat([
            self.self_attention(query(x), K, V, mask) for query in self.querys
        ], dim=1)
        Z = self.proj(Z_s)
        return Z
```
We see that MQA reuses K and V projections as input for each separate attention head.

This results in slight gains during training and substantial gains in terms of inference throughput, at the cost of marginal dips in model performance on the sequence to sequence benchmarks presented in the paper. 

