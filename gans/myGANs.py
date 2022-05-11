import torch
from torch import nn

from .generator import generator
from .discriminator import discriminator

class myGANs:
    '''
    myGANs Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
          (MNIST images are 28 x 28 = 784 so that is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, batch_size = 128, lr = 0.00001, device='cuda') -> None:
        super().__init__()
        self.batch_size = batch_size
        self.gen = generator(z_dim).to(device)
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr)
        self.disc = discriminator().to(device)
        self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=lr)
    
    def train(
        self,
        n_epochs=200,
        criterion=nn.BCEWithLogitsLoss(),
    ):
        pass

    # Load MNIST dataset as tensors
    


# Set your parameters

# z_dim = 64
# display_step = 5000

def dataloader(
        MNIST('.', download=True, transform=transforms.ToTensor()),
        batch_size=128,
        shuffle=True
    ):
        return DataLoader(
            MNIST('.', download=True, transform=transforms.ToTensor()),
            batch_size=batch_size,
            shuffle=shuffle
    )