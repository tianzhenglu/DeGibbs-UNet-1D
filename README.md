# DeGibbs-UNet-1D
U-Net convolutional network to remove Gibbs noise in 1D time signal

Author: Tianzheng Lu        


INTRODUCTION

This U-Net is proposed to eliminate noise introduced by Gibb's phenoemon in the Fourier series. Traditional filters such as $\sigma$ approximation smooth the signal, while in the author's application[1] smoothing the signal leads to extra error. Thus author tried convolutional neural network to solve this problem.

This work is mainly based on two existing works:
1. Muckley, Matthew J., et al. "Training a neural network for Gibbs and noise removal in diffusion MRI." Magnetic resonance in medicine 85.1 (2021): 413-428.
2. https://github.com/milesial/Pytorch-UNet

[1] Lu, T., Legrand, M. Nonsmooth modal analysis via the boundary element method for one-dimensional bar systems. Nonlinear Dyn 107, 227–246 (2022). https://doi.org/10.1007/s11071-021-06994-z
