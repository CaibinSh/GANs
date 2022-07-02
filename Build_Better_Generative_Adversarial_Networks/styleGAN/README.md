# StyleGAN

Amazed by StyleGAN's capabilities? Take a look at the original paper! Note that it may take a few extra moments to load because of the high-resolution images.

[A Style-Based Generator Architecture for Generative Adversarial Networks](1812.04948.pdf) (Karras, Laine, and Aila, 2019)

# Lecture Note

Lecture note from Deeplearning.AI - [C2_W3.pdf](C2_W3.pdf)


# Programming Assignment: Components of StyleGAN
In this notebook, you're going to implement various components of StyleGAN, including the truncation trick, the mapping layer, noise injection, adaptive instance normalization (AdaIN), and progressive growing.  

Assignment notebook - [StyleGAN](C2W3_Assignment.ipynb)
[PyTorch-lightning version](myC2W3_styleGAN.ipynb)

# Components of StyleGAN2
In this [notebook](StyleGAN2), you're going to learn about StyleGAN2, from the paper Analyzing and Improving the Image Quality of StyleGAN ([Karras et al., 2019](1912.04958.pdf)), and how it builds on StyleGAN. This is the V2 of StyleGAN, so be prepared for even more extraordinary outputs.

# Components of BigGAN
In this [notebook](BigGAN.ipynb), you'll learn about and implement the components of BigGAN, the first large-scale GAN architecture proposed in [Large Scale GAN Training for High Fidelity Natural Image Synthesis](1809.11096.pdf) (Brock et al. 2019). BigGAN performs a conditional generation task, so unlike StyleGAN, it conditions on a certain class to generate results. BigGAN is based mainly on empirical results and shows extremely good results when trained on ImageNet and its 1000 classes

# StyleGAN Walkthrough and Beyond
Want another explanation of StyleGAN? This article provides a great walkthrough of StyleGAN and even discusses StyleGAN's successor: StyleGAN2!

GAN — [StyleGAN & StyleGAN2 (Hui, 2020)](https://jonathan-hui.medium.com/gan-stylegan-stylegan2-479bdf256299)

# Finetuning Notebook: FreezeD

In this [notebook](https://colab.research.google.com/github/https-deeplearning-ai/GANs-Public/blob/master/C2W3_FreezeD_(Optional).ipynb), you'll learn about and implement the fine-tuning approach proposed in [Freeze the Discriminator: a Simple Baseline for Fine-Tuning GANs](2002.10964) (Mo et al. 2020), which introduces the concept of freezing the upper layers of the discriminator in fine-tuning. Specifically, you'll fine-tune a pretrained StyleGAN to generate anime faces from human faces.

Due to the size and dependence on recent PyTorch features (e.g. Automatic Mixed Precision (AMP)), this is provided as a Colab notebook. 