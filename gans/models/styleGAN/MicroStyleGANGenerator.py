import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from .MappingLayers import MappingLayers
from .MicroStyleGANGeneratorBlock import MicroStyleGANGeneratorBlock

class MicroStyleGANGenerator(LightningModule):
    """Micro StyleGAN Generator Class

    Args:
        z_dim (int): the dimension of the noise vector
        map_hidden_dim (int): the mapping inner dimension
        w_dim (int): the dimension of the intermediate noise vector
        in_chan (int): the dimension of the constant input, usually w_dim
        out_chan (int): the number of channels wanted in the output
        kernel_size (int): the size of the convolving kernel
        hidden_chan (int): the inner dimension
    """        
    def __init__(self, z_dim, map_hidden_dim, w_dim, in_chan, out_chan, kernel_size, hidden_chan) -> None:
 
        super().__init__()
        self.map = MappingLayers(z_dim, map_hidden_dim, w_dim)
        self.starting_constant = nn.Parameter(torch.randn(1, in_chan, 4, 4))
        self.block0 = MicroStyleGANGeneratorBlock(in_chan, hidden_chan, w_dim, kernel_size, 4, use_upsample=False)
        self.block1 = MicroStyleGANGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 8)
        self.block2 = MicroStyleGANGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 16)
        self.block1_to_image = nn.Conv2d(hidden_chan, out_chan, kernel_size=1)
        self.block2_to_image = nn.Conv2d(hidden_chan, out_chan, kernel_size=1)
        self.alpha = 0.2
    
    def upsample_to_match_size(self, smaller_image, bigger_image):
        """Function for upsampling an image to the size of another: given two images (smaller and bigger),
        upsamples the first to have the same dimensions as the second.

        Args:
            smaller_image (tensor): the smaller image to upsample
            bigger_image (tensor): the bigger image whose dimensions will be upsampled to
        """        
        return F.interpolate(smaller_image, size=bigger_image.shape[-2:], mode="bilinear")
    
    def forward(self, noise, return_intermediate=False):
        """Function for completing a forward pass of MicroStyleGANGenerator: give noise,
        computes a styleGAN iteration.

        Args:
            noise (tensor): a noise tensor with dimensions (n_samples, z_dim)
            return_intermediate (bool, optional): true to return the images as well (for testing). Defaults to False.
        """        
        x = self.starting_constant
        w = self.map(noise)
        x = self.block0(x, w)
        x_small = self.block1(x, w)
        x_small_image = self.block1_to_image(x_small)
        x_big = self.block2(x_small, w)
        x_big_image = self.block2_to_image(x_big)
        x_small_upsample = self.upsample_to_match_size(x_small_image, x_big_image)

        interpolation = torch.lerp(input=x_small_upsample, end=x_big_image, weight=self.alpha)

        if return_intermediate:
            return interpolation, x_small_upsample, x_big_image
        return interpolation
    
    # for unit test
    def get_self(self):
        return self