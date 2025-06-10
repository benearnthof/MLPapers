https://arxiv.org/abs/2210.02747
https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html#gaussian-probability-paths
https://ai.meta.com/research/publications/flow-matching-guide-and-code/
https://dl.heeere.com/conditional-flow-matching/blog/conditional-flow-matching/
https://rectifiedflow.github.io/
https://diffusionflow.github.io/
https://diffusion.csail.mit.edu/docs/lecture-notes.pdf

## Introduction

Flow matching is a novel framework for generative modeling that leverages insights from statistical physics, differential geometry, and statistical learning to arrive at a method for learning statistical distributions that outperforms other methods and pushes the state-of-the-art in fields such as conditional generation of images [[Stable Diffusion 3]], viedeos [[Movie Gen]], speech [[Voicebox]], audio [[Audiobox]], robotics [[Pi_zero]]], and proteomics [[SE(3) Flow Matching]].

Most important questions: 
	* Design Choices for Flow Matching
	* Can Flow Matching be extended to domains other than images?
	* How to actually implement Flow Matching with PyTorch
	* Mathematical Derivation of Flow Matching and Connections to other Generative Models

Flow Matching is based on learning a velocity field / vector field. Each velocity field defines a flow $\psi_{t}$ by solving (integrating) an [[ODE]]. 
A flow $\psi$ is a **deterministic**, **time-continuous**, and **bijective** transformation of the d-dimensional Euclidean space $\mathbb{R}^d$. 
The goal of Flow Matching is to learn a flow in such a way, that it transforms any sample $X_0 \sim p$ drawn from a source distribution $p$ into a target sample $X_1:=\psi_1\left(X_0\right)$ such that the resulting target distribution $X_1 \sim q$ has a desired target distribution $q$. 

Flow Models are a more general class of models, the first member of this class of generative models was introduced as [[Continuous Normalizing Flows]]  [[Neural ODE]]  [[FFJORD]] 
Training CNFs requires simulation and its differentiation during training, leading these models to incur a large computational cost during training. This led directly to the development of simulation free Flow Model variants like [[Normalizing Flows on Manifolds]] and [[Moser Flow]] which, in turn, led to modern Flow Matching algorithms like [[Rectified Flow]], [[Stochastic Interpolants]], [[Action Matching]], [[Minibatch OT]], and [[Iterative alpha Blending]].

A minimal Flow Matching implementation consists of two steps: 
1. Pick a probability path that interpolates between source $p$ and target $q$ distributions. 
2. Train a velocity field (or a neural network approximation of it) that defines the flow transformation $\psi_{t}$ and implements $p_t$

The Flow Matching paradigm can easily be extended to accommodate discrete state spaces [[Discrete State Spaces]], inductive priors such as Riemannian Manifolds[[Flow Matching on General Geometries]], and Continuous Time Markov Processes [[Generator Matching]], where, up to differences in the neural architecture, the Flow Matching recipe stays the same. 




## Key Concepts

## Flow Models

## Flow Matching

## Non-Euclidean Flow Matching

## Continuous Time Markov Chain Models

## Discrete Flow Matching

## Continuous Time Markov Process Models

## Generator Matching  

## Relation to Diffusion and other Denoising Models

## Appendix A: Proofs

