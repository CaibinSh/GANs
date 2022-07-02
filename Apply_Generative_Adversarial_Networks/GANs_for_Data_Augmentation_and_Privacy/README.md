# GANs for Data Augmentation and Privacy

## Automated Data Augmentation

RandAugment: Practical automated data augmentation with a reduced search space (Cubuk, Zoph, Shlens, and Le, 2019): https://arxiv.org/abs/1909.13719

## [Lecture Note](C3_W1.pdf)

## GTN

Click on this link to access the optional Colab notebook.

In this [notebook](C3W1_Generative_Teaching_Networks_Optional.ipynb), you'll be implementing a Generative Teaching Network (GTN), first introduced in Generative Teaching Networks: Accelerating Neural Architecture Search by Learning to Generate Synthetic Training Data (Such et al. 2019). Essentially, a GTN is composed of a generator (i.e. teacher), which produces synthetic data, and a student, which is trained on this data for some task. The key difference between GTNs and GANs is that GTN models work cooperatively (as opposed to adversarially).

## Talking Heads
Fascinated by how you can use GANs to create talking heads and deepfakes? Take a look at the paper!

Few-Shot Adversarial Learning of Realistic Neural Talking Head Models (Zakharov, Shysheya, Burkov, and Lempitsky, 2019): https://arxiv.org/abs/1905.08233

## De-identification
Curious to learn more about how you can de-identify (anonymize) a face while preserving essential facial attributes in order to conceal an identity? Check out this paper!

De-identification without losing faces (Li and Lyu, 2019): https://arxiv.org/abs/1902.04202

## GAN Fingerprints
Concerned about distinguishing between real images and fake GAN generated images? See how GANs leave fingerprints!

Attributing Fake Images to GANs: Learning and Analyzing GAN Fingerprints (Yu, Davis, and Fritz, 2019): https://arxiv.org/abs/1811.08180

## [Assignment](C3W1_Assignment.ipynb)