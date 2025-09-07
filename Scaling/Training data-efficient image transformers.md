& distillation through attention
https://arxiv.org/abs/2012.12877

The paper produces competitive, convolution-free transformers by training on Imagenet only. They claim training on a single computer (so a max of 8 GPUs?) for three days. 
(53 hours of pre-training, and optionally 20 hours of fine-tuning)

They claim training on 8 V100 GPUs. V100s were released in 2017, so 24GB RTX A5000 should be able to achieve equivalent performance. On demand such an instance costs about $2.16/h (disregarding additional disk space) for 73h that would amount to just shy of $160. ($110) for finetuning. Not bad. Edit: They claim that their smallest models were trained on only 4 GPUs, halving training costs. 

The github repo leads to a swarm of implementations building on the methods proposed here: 
[[Going Deeper with Image Transformers]]
[[ResMLP]]
[[PatchConvnet]]
[[Three things everyone should know about Vision Transformers]]
[[DeiT III  Revenge of the ViT]]
[[Cosub - Co-training 2L Submodels for Visual Recognition]]

Transformers have already shown great performance on end-to-end vision tasks: 
[[DETR End-to-End Object Detection with Transformers]]
[[An image is worth 16x16 words]]

They present key ingredients for successful training such as hyper-parameter choices and repeated augmentation.

They introduce a token-based distillation strategy specific to transformers and show that it "advantageously replaces the usual distillation".

## Related Works
[[Generative Pretraining from Pixels]]
[[Bag of Tricks for Image Classification with Convolutional Neural Networks]]
[[An image is worth 16x16 words]]
[[Attention Is All You Need]]
[[Squeeze-and-Excitation Networks]]
[[Distilling the Knowledge of a Neural Network]]
[[Revisiting Knowledge Distillation via Label Smoothing Regularization]]

To obtain a transformer block for images the authors use a FFN that expands the input of size D to 4D and back to D. (Some sort of fully connected mixing?)
Each patch is projected with a linear layer that conserves its overall dimension 3 × 16 × 16 = 768

[[Fixing the Train-test resolution discrepancy]] highlights that it is desirable to pretrain on smaller resolutions and finetune on the target resolution. Positional encodings are simply interpolated in this case. 

They utilize strong convnet teachers with soft and or hard distillation to outperform the teacher models when analyzed for accuracy vs throughput. 

## The Secret Sauce 
* Initialization & Choice of Hyperparameters: [[How to Start Training]] Weights are initialized with a truncated normal distribution
* Data-Augmentation: They use Rand-Augment and **do not** use dropout [[RandAugment]]
* Regularization & Optimizers: They scale the learning rate according to the batch size as recommended in [[Accurate Large Minibatch SGD]]
* They do use minimal weight decay, as large weight decay seemed to impede convergence
* They employ [[Deep Networks with Stochastic Depth]] which facilitates convergence of deep transformers
* EMA advantage seems to disappear after finetuning
* They finetune on larger resolutions [[Fixing the Train-test resolution discrepancy]]


[[Deep Networks with Stochastic Depth]]
