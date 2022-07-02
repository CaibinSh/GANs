import torch.nn as nn

class discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
    hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, im_chan=1, hidden_dim=16, spectral_norm=True):
        super().__init__()
        self.discriminator = nn.Sequential(
            self.make_discriminator_block(im_chan, hidden_dim, spectral_norm=spectral_norm),
            self.make_discriminator_block(hidden_dim, hidden_dim * 2, spectral_norm=spectral_norm),
            self.make_discriminator_block(hidden_dim * 2, 1, final_layer=True, spectral_norm=spectral_norm)
        )

    def make_discriminator_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False, spectral_norm=True):
        '''
        Function to return a sequence of operations corresponding to a discriminator block of DCGAN, 
        corresponding to a convolution, a batchnorm (except for in the last layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
            spectral_norm: whether to implement spectral normalization
        '''
        if not final_layer:
            return nn.Sequential(
                nn.utils.spectral_norm(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=output_channels,
                        kernel_size=kernel_size,
                        stride=stride
                    )
                ) if spectral_norm else nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    stride=stride
                ),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(negative_slope=0.2)
            )
        else:
            return nn.Sequential(
                nn.utils.spectral_norm(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=output_channels,
                        kernel_size=kernel_size,
                        stride=stride
                    )
                ) if spectral_norm else nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    stride=stride
                )
            )
    
    def forward(self, image):
        '''
        Function for completing a forward pass of the discriminator: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_dim)
        '''
        disc_pred = self.discriminator(image)

        return disc_pred.view(len(disc_pred), -1)
    