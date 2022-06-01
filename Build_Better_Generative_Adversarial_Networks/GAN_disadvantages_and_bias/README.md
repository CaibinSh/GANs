# GAN disadvantages and bias

## (Optional Notebook) GAN Debiasing
Please note that this is an optional notebook that is meant to introduce more advanced concepts, if you're up for a challenge. So, don't worry if you don't completely follow every step!

Notebook link: https://colab.research.google.com/github/https-deeplearning-ai/GANs-Public/blob/master/C2W2_GAN_Debiasing_(Optional).ipynb

[Fair Attribute Classification through Latent Space De-biasing](https://princetonvisualai.github.io/gan-debiasing/). Vikram V. Ramaswamy, Sunnie S. Y. Kim, Olga Russakovsky. CVPR 2021.

Fairness in visual recognition is becoming a prominent and critical topic of discussion as recognition systems are deployed at scale in the real world. Models trained from data in which target labels are correlated with protected attributes (i.e. gender, race) are known to learn and perpetuate those correlations.

In this notebook, you will learn about Fair Attribute Classification through Latent Space De-biasing (Ramaswamy et al. 2020) that introduces a method for training accurate target classifiers while mitigating biases that stem from these correlations. Specifically, this work uses GANs to generate realistic-looking images and perturb these images in the underlying latent space to generate training data that is balanced for each protected attribute. They augment the original dataset with this perturbed generated data, and empirically demonstrate that target classifiers trained on the augmented dataset exhibit a number of both quantitative and qualitative benefits.



## (Optional Notebook) NeRF: Neural Radiance Fields
Please note that this is an optional notebook meant to introduce more advanced concepts. If you’re up for a challenge, take a look and don’t worry if you can’t follow everything. There is no code to implement—only some cool code for you to learn and run!

Click [this link](C2W2_Optional_Notebook_NeRF.ipynb) to access the Colab notebook!

In this notebook, you'll learn how to use Neural Radiance Fields to generate new views of a complex 3D scene using only a couple input views, first proposed by [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis (Mildenhall et al. 2020)](NeRF.pdf). Though 2D GANs have seen success in high-resolution image synthesis, NeRF has quickly become a popular technique to enable high-resolution 3D-aware GANs.

## [Lecture note](C2_W2.pdf)

## Score-based Generative Modeling

[notebook](C2W2_Optional_Notebook_Score_Based_Generative_Modeling.ipynb)

This is a hitchhiker's guide to score-based generative models, a family of approaches based on estimating gradients of the data distribution. They have obtained high-quality samples comparable to GANs (like below, figure from this paper) without requiring adversarial training, and are considered by some to be the new contender to GANs.

## Machine Bias
Before going into the discussion on bias in machine learning, please read this case study to gain an understanding of the impact these biases can have on real lives: 

Machine Bias (Angwin, Larson, Mattu, and Kirchner, 2016): https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing

## Fairness Definition

To understand some of the existing definitions of fairness and their relationships, please read the following paper and view the Google glossary entry for fairness: 

Fairness Definitions Explained (Verma and Rubin, 2018): https://fairware.cs.umass.edu/papers/Verma.pdf

Machine Learning Glossary: Fairness (2020): https://developers.google.com/machine-learning/glossary/fairness

## Finding Bias

Now that you've seen how complex fairness is, how do you find bias in existing material (models, datasets, frameworks, etc.) and how can you prevent it? These two readings offer some insight into how bias was detected and some avenues where it may have been introduced.

Does Object Recognition Work for Everyone? (DeVries, Misra, Wang, and van der Maaten, 2019): https://arxiv.org/abs/1906.02659

What a machine learning tool that turns Obama white can (and can't) tell us about AI bias (Vincent, 2020): https://www.theverge.com/21298762/face-depixelizer-ai-machine-learning-tool-pulse-stylegan-obama-bias

## [Assignment -- Bias](C2W2_Assignment.ipynb)

## [Lecture note](C2_W2.pdf)