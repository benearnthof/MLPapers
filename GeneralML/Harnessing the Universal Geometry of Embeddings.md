https://arxiv.org/pdf/2505.12540

Translating vector embeddings from one vector space to another without any paired data. 
Do vector spaces have the same dimension or is this solved by concatenation or some architectural trick? 

No encoders no other predefined sets of matches. Completely unsupervised. 
An adversary with access to a database of only embeddign vectors can extract sensitive information about underlying documents, sufficient for classification and attribute inference.

Their model learns a latent space in which unpaired embedding samples from different models are closely aligned. 

### Introduction
In theory, the same piece of text taken from the same training corpus should yield embeddings that point in very similar directions in the embedding vector space. But because of random initialization and architectural choices, different models encode text into completely different and incompatible vector spaces. 

[[The Platonic Representation Hypothesis]] conjectures that all image models of sufficient size converge to the same latent representation. 
This paper proposes a stronger, constructive argument: The universal latent structure of text representations can be learned, and furthermore, harnessed to translate representations from one space to another without any paired data or encoders. 

Could we use this to construct a twin translation model? We train two separate text transformers, then use this model to map the embeddings of the input language to those of the target language. No paired data required. We could either extend this architecture with [[Flow Matching for Generative Modeling]] or experiment with a "decoder only" translation model. (Or combine both methods). Flow Matching will probably not work for this since we would require paired data and don't have a closed form source distribution. 

Figure 2 is interesting: Given only a vector database from an unknown model, vec2vec translates the database into the space of a known model using latent structure alone. 
So this method is probably not accurate enough for machine translation. But they do only use an MLP with a couple of residual and normalization layers. 

The Authors draw heavily from other literature about [[Word Translation Without Parallel Data]]: [[Unsupervised Alignment of Embeddings with Wasserstein Procrustes]] [[Unsupervised Multilingual Word Embeddings]] and unsupervised image to image translation techniques. 
Doublecheck: is [[Dynadiff]] relevant? 

vec2vec uses adversarial losses and cycle consistency to learn to encode embeddings into a shared latent space and decode with minimal loss. [[CycleGAN]]

vec2vec preserves not only relative geometry of the embeddings but also the internal semantics of their underlying inputs, allowing the authors to perform attribute inference and inversion **without any knowledge about the model that produced the embeddings.**

### Problem statement
An example application would be data extraction from a vector database dump. We only have access to a large set of high dimensional document embedding vectors but have no information about the model architecture used to generate the embedding vectors. Can we still recover information about the original documents from this? 
vec2vec claims this is possible. 

We have to assume structural information about the original data (text vs image) and distributional information (english, chinese, etc.). To extract information we only use a separate encoder $M_2$ that one can query at will to generate new embeddings from the vectors. To extract useful information we may translate the set of database vectors into the output space of $M_2$ and then apply techniques such as inversion that take advantage of the encoder. 

So the core idea is that any semantically persistent transformation preserves enough of the local and global geometry of the database vectors to allow us to "decode" them in some sense. 
#### Limits
[[Gromov-Wasserstein Alignment of Word Embedding Spaces]] and other similar "matching" methods struggle significantly when it is not possible to query the original encoder $M_1$ Unsupervised embeddign translation seeks to generate a translation that is close to $v_1 = M_2(d_1)$ without access to $d_1$, $M_1$, or $v_1$.
### vec2vec
Unsupervised translation is inspired by cycle-consistency (mappings to and from an embedding space should end up in the same place they started in) and indistinguishability (embeddings for the same text from either space should have identical latents)

The authors use a GAN based architecture with additional regularization based on 
* Reconstruction quality
* Cycle-consistency
* Vector space preservation

This ensures that both local and global semantics are preserved under these transformations.



