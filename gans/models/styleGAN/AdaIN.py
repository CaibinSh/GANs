from torch import nn
from pytorch_lightning import LightningModule

class AdaIN(LightningModule):
    """AdaIN class: Adaptive Instance Normalization 

    Args:
        channels (int): the number of channels the image has
        w_dim (int): the dimension of the intemediate noise vector
    """

    def __init__(self, channels, w_dim) -> None:
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.style_scale_transform = nn.Linear(w_dim, channels)
        self.style_shift_transform = nn.Linear(w_dim, channels)

    def forward(self, image, w):
        """Function for completing a forward pass of AdaIN: given an image and intermediate noise vector w,
        returns the normalized image that has been scaled and shifted by the style.

        Args:
            image (tensor): the feature map of shape (n_samples, channels, width, height)
            w (tensor): the intermediate noise vector
        """        
        normalize_image = self.instance_norm(image)
        style_scale = self.style_scale_transform(w)[:, :, None, None]
        style_shift = self.style_shift_transform(w)[:, :, None, None]
    
        # calculate the transformed image
        transformed_image = normalize_image * style_scale + style_shift
        return transformed_image
    
    # for unit test
    def get_style_scale_transform(self):
        return self.style_scale_transform
    
    # for unit test
    def get_style_shift_transform(self):
        return self.style_shift_transform

    # for unit test
    def get_self(self):
        return self