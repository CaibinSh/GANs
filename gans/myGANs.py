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

    # Load MNIST dataset as tensors
    def load(
        self,
        MNIST('.', download=True, transform=transforms.ToTensor()),
        shuffle=True
    ):
        return DataLoader(
            MNIST('.', download=True, transform=transforms.ToTensor()),
            batch_size=self.batch_size,
            shuffle=shuffle
    )


# Set your parameters

z_dim = 64
display_step = 5000


# GRADED FUNCTION: get_disc_loss
def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    '''
    Return the loss of the discriminator given inputs.
    Parameters:
        gen: the generator model, which returns an image given z-dimensional noise
        disc: the discriminator model, which returns a single-dimensional prediction of real/fake
        criterion: the loss function, which should be used to compare 
               the discriminator's predictions to the ground truth reality of the images 
               (e.g. fake = 0, real = 1)
        real: a batch of real images
        num_images: the number of images the generator should produce, 
                which is also the length of the real images
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    Returns:
        disc_loss: a torch scalar loss value for the current batch
    '''
    #     These are the steps you will need to complete:
    #       1) Create noise vectors and generate a batch (num_images) of fake images. 
    #            Make sure to pass the device argument to the noise.
    #       2) Get the discriminator's prediction of the fake image 
    #            and calculate the loss. Don't forget to detach the generator!
    #            (Remember the loss function you set earlier -- criterion. You need a 
    #            'ground truth' tensor in order to calculate the loss. 
    #            For example, a ground truth tensor for a fake image is all zeros.)
    #       3) Get the discriminator's prediction of the real image and calculate the loss.
    #       4) Calculate the discriminator's loss by averaging the real and fake loss
    #            and set it to disc_loss.
    #     Note: Please do not use concatenation in your solution. The tests are being updated to 
    #           support this, but for now, average the two losses as described in step (4).
    #     *Important*: You should NOT write your own loss function here - use criterion(pred, true)!
    #### START CODE HERE ####
    noise = get_noise(num_images, z_dim, device=device)
    fake_images = gen(noise).detach()
    pred = disc(fake_images)
    loss_fake = criterion(pred, torch.zeros_like(pred))
    ground_truth = disc(real)
    loss_real = criterion(ground_truth, torch.ones_like(ground_truth))
    disc_loss = (loss_fake + loss_real) / 2
    #### END CODE HERE ####
    return disc_loss