# Wasserstein GANs with Gradient Penalty

## [Assignment -- WGAN](C1W3_WGAN_GP.ipynb)

[The PyTorch-lightning version](../../gans/models/WGAN_GP/) and [notebook](myC1W3_WGAN_GP.ipynb)

## SN-GAN

In this [notebook](SNGAN.ipynb), you'll learn about and implement spectral normalization, a weight normalization technique to stabilize the training of the discriminator, as proposed in Spectral Normalization for Generative Adversarial Networks (Miyato et al. 2018).

## ProteinGAN

The goal of this notebook is to demonstrate that core GAN ideas can be applied outside of the image domain. In this notebook, you will be able to play around with a pre-trained ProteinGAN model to see how it can be used in bioinformatics to generate functional molecules.  
   
[Notebook](C1W3_ProteinGAN_Optional.ipynb)  
   
Paper: Expanding functional protein sequence spaces using generative adversarial networks, Nature Machine Intelligence (2021).

## The WGAN and WGAN-GP Papers

[Wasserstein GAN](https://arxiv.org/abs/1701.07875) (Arjovsky, Chintala, and Bottou, 2017) 

[Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028) (Gulrajani et al., 2017)

## WGAN Walkthrough

Want another explanation of WGAN? This article provides a great walkthrough of how WGAN addresses the difficulties of training a traditional GAN with a focus on the loss functions.

[From GAN to WGAN](https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html) (Weng, 2017)

## [Lecture notes](C1_W3.pdf)