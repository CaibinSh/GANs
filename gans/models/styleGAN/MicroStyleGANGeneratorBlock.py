from torch import nn
from pytorch_lightning import LightningModule
from .InjectNoise import InjectNoise
from .AdaIN import AdaIN

class MicroStyleGANGeneratorBlock(LightningModule):
    """Micro StyleGAN Generator Block Class

    Args:
        in_chan (int): the number of channels in the input
        out_chan (int): the number of channels in the output
        w_dim (int): the dimension of the intermediate noise vector
        kernel_size (int): the size of the convolving kernel
        starting_size (int): the size of the starting image
    """    

    def __init__(self, in_chan, out_chan, w_dim, kernel_size, starting_size, use_upsample=True) -> None:
        super().__init__()
        self.use_upsample = use_upsample
        if self.use_upsample:
            self.upsample = nn.Upsample((starting_size, starting_size), mode="bilinear")
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size, padding=1)
        self.inject_noise = InjectNoise(out_chan)
        self.adain = AdaIN(out_chan, w_dim)
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x, w):
        """Function for completing a forward pass of MicroStyleGANGeneratorBlock: given an x and w,
        computes a styleGAN generator block

        Args:
            x (tensor): the input into the generator, feature map of shape (n_samples, channels, width, height)
            w (tensor): the intermediate noise vector
        """        
        if self.use_upsample:
            x = self.upsample(x)
        x = self.conv(x)
        x = self.inject_noise(x)
        x = self.adain(x, w)
        x = self.activation(x)

        return x
    
    # Unit test
    def get_self(self):
        return self