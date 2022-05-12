from .__version__ import __version__
from .generator import generator, get_noise
from .discriminator import discriminator
from .myGANs import GANs, MNISTDataModule
from .loss_function import get_disc_loss, get_gen_loss