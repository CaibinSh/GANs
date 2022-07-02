# Image-to-Image Translation with Pix2Pix

## The Pix2Pix Paper
Image-to-Image Translation with Conditional Adversarial Networks (Isola, Zhu, Zhou, and Efros, 2018): https://arxiv.org/abs/1611.07004

## [Lecture Note](C3_W2.pdf)

## Assignment
### [AssignmentA: U-Net architecture](C3W2A_Assignment.ipynb)
### [AssignmentB: Pix2Pix](C3W2B_Assignment.ipynb)

## Pix2PixHD (Optional)
In this [notebook](C3W2_Pix2PixHD.ipynb), you will learn about Pix2PixHD, which synthesizes high-resolution images from semantic label maps. Proposed in [High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs](https://arxiv.org/abs/1711.11585) (Wang et al. 2018), Pix2PixHD improves upon Pix2Pix via multiscale architecture, improved adversarial loss, and instance maps.

## Super-resolution GAN (SRGAN) (Optional)
In this [notebook](C3W2_SRGAN.ipynb), you will learn about Super-Resolution GAN (SRGAN), a GAN that enhances the resolution of images by 4x, proposed in [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802) (Ledig et al. 2017). You will also implement the architecture and training in full and be able to train it on the CIFAR dataset.

## More Work Using PatchGAN
Want to see how a GAN can fill-in cropped-out portions of an image? Read about how PGGAN does that by using PatchGAN!

Patch-Based Image Inpainting with Generative Adversarial Networks (Demir and Unal, 2018): https://arxiv.org/abs/1803.07422

## GauGAN (Optional)
In this [notebook](C3W2_GauGAN.ipynb), you will learn about GauGAN, which synthesizes high-resolution images from semantic label maps, which you implement and train. GauGAN is based around a special denormalization technique proposed in [Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) (Park et al. 2019)