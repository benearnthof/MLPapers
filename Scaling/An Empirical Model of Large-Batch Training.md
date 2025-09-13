https://arxiv.org/pdf/1812.06162

CIFAR-10: Critical Batch Size of 300 (Start) to 900 (Average)
There is a pareto frontier for each level of desired training error. Larger batches require less optimizer steps to reach the desired level of performance, but we have to pay the extra cost of additional compute. In general we have to either optimize longer on smaller batches, or see more total training examples in larger batches but take less optimization steps in total. (If that makes sense).
