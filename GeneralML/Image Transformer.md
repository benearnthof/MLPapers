https://arxiv.org/pdf/1802.05751

The authors build on the previous successes of models like PixelCNN and PixelRNN and try to strike a balance between both models. 
CNNs are highly parallelizable and parameter efficient, achieving great throughput and high trainability, but suffer from limited receptive fields when deployed on larger images. RNNs have a virtually unlimited receptive field but do suffer from difficult training and high inference cost for long sequences. 
This paper builds on [[Attention Is All You Need]], [[PixelCNN]], [[The Neural Autoregressive Distribution Estimator]], [[Modeling High-Dimensional Discrete Data with Multi-Layer Neural Networks]]

They introduce a specific, locally restricted form of multi-head self-attention that can be interpreted as a sparsely parametrized form of gated convolution. This effectively decouples the size of the receptive field from the number of parameters, allowing for larger receptive fields than vanilla PixelCNN.

They model the color channels of the output pixels as discrete values generated from a multinomial distribution, implemented with a softmax layer.

**For categories**, each of the input pixels' three color channels is encoded using a channel-specific set of 256 d-dimensional embedding vectors of the respective intensity values ranging from 0-255. 
For output intensities they share a single, separate set of 256 d-dimensional embeddings across the channels. 

**For ordinal values** they run a 1x3 strided convolution to combine the three channels per pixel to form an input representation with shape [h, w, d]

They add a d-dimensional encoding of coordinates of that pixel [[Positional Encoding]]
They compare sin/cosine embeddings and learned position embeddings. 

The authors use query blocks to reduce memory consumption. 

