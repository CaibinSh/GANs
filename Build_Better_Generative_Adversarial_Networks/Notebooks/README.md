# (Optional Notebook) GAN Debiasing
Please note that this is an optional notebook that is meant to introduce more advanced concepts, if you're up for a challenge. So, don't worry if you don't completely follow every step!

Notebook link: https://colab.research.google.com/github/https-deeplearning-ai/GANs-Public/blob/master/C2W2_GAN_Debiasing_(Optional).ipynb

[Fair Attribute Classification through Latent Space De-biasing](https://princetonvisualai.github.io/gan-debiasing/). Vikram V. Ramaswamy, Sunnie S. Y. Kim, Olga Russakovsky. CVPR 2021.

Fairness in visual recognition is becoming a prominent and critical topic of discussion as recognition systems are deployed at scale in the real world. Models trained from data in which target labels are correlated with protected attributes (i.e. gender, race) are known to learn and perpetuate those correlations.

In this notebook, you will learn about Fair Attribute Classification through Latent Space De-biasing (Ramaswamy et al. 2020) that introduces a method for training accurate target classifiers while mitigating biases that stem from these correlations. Specifically, this work uses GANs to generate realistic-looking images and perturb these images in the underlying latent space to generate training data that is balanced for each protected attribute. They augment the original dataset with this perturbed generated data, and empirically demonstrate that target classifiers trained on the augmented dataset exhibit a number of both quantitative and qualitative benefits.



# (Optional Notebook) NeRF: Neural Radiance Fields
Please note that this is an optional notebook meant to introduce more advanced concepts. If you’re up for a challenge, take a look and don’t worry if you can’t follow everything. There is no code to implement—only some cool code for you to learn and run!

Click [this link](C2W2_Optional_Notebook_NeRF.ipynb) to access the Colab notebook!

In this notebook, you'll learn how to use Neural Radiance Fields to generate new views of a complex 3D scene using only a couple input views, first proposed by [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis (Mildenhall et al. 2020)](NeRF.pdf). Though 2D GANs have seen success in high-resolution image synthesis, NeRF has quickly become a popular technique to enable high-resolution 3D-aware GANs.