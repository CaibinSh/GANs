import torch
from torch import nn
from pytorch_lightning import LightningModule

class InjectNoise(LightningModule):
    """Inject Noise Class

    Args:
        channels (int): the number of channels the image has
    """    

    def __init__(self, channels) -> None:        
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(1, channels, 1, 1)
        )
    
    # For unit test
    def forward(self, image):
        """Function for completing a forward pass of InjectNoise: given an image,
        returns the image with random noise added.

        Args:
            image (tensor): the feature map of shape (n_sample, channels, width, height)

        Returns:
            tensor: an image with random noise added
        """        
        noise_shape = (image.shape[0], 1, image.shape[2], image.shape[3])
        noise = torch.randn(noise_shape)
        return image + noise * self.weight
    
    # For unit test
    def get_weight(self):
        return self.weight
    
    # For unit test
    def get_self(self):
        return self