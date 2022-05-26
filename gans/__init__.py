from .__version__ import __version__
from .datasets.MNIST import MNISTDataModule
from .models.firstGAN.myGANs import GANs
from .models.DCGAN.DCGAN import DCGAN
from .models.WGAN_GP.WGAN_GP import WGAN_GP
from .models.CGAN.CGAN import CGAN
from .models.ControllableGAN.ControllableGAN import ControllableGAN