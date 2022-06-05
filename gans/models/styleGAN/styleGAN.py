from turtle import forward
from scipy.stats import truncnorm
import torch
from torch import nn
from pytorch_lightning import LightningModule

from .MicroStyleGANGenerator import MicroStyleGANGenerator

class styleGAN(LightningModule):
    """_summary_

    Args:
        n_samples (int, optional): _description_. Defaults to 10.
        z_dim (int, optional): _description_. Defaults to 128.
        map_hidden_dim (int, optional): _description_. Defaults to 1024.
        w_dim (int, optional): _description_. Defaults to 496.
        in_chan (int, optional): _description_. Defaults to 512.
        out_chan (int, optional): _description_. Defaults to 3.
        kernel_size (int, optional): _description_. Defaults to 3.
        hidden_chan (int, optional): _description_. Defaults to 256.
        truncation (float, optional): _description_. Defaults to 0.7.
    """  
    def __init__(
        self,
        z_dim = 128,
        map_hidden_dim = 1024,
        w_dim = 496,
        in_chan = 512,
        out_chan = 3,
        kernel_size = 3,
        hidden_chan = 256,
    ) -> None:
            
        super().__init__()
        self.save_hyperparameters()
        self.styleGAN = MicroStyleGANGenerator(
            z_dim = z_dim,
            map_hidden_dim = map_hidden_dim,
            w_dim = w_dim,
            in_chan = in_chan,
            out_chan = out_chan,
            kernel_size = kernel_size,
            hidden_chan = hidden_chan
        )

    def forward(self, noise):
        return self.styleGAN(noise)

    def validation_step(self, val_batch, batch_idx):
        x, _ = val_batch
        generated_images = self(x)
        return generated_images

    def on_validation_end(self):
        truncation = 0.7
        test_samples = 10

        z = get_truncated_noise(n_samples=test_samples, z_dim=self.hparams.z_dim, truncation=truncation) * 10
        z = z.type_as(self.styleGAN.map.mapping[0].weight)

        # log sampled images
        sample_imgs = self(z)

        return sample_imgs


def get_truncated_noise(n_samples, z_dim, truncation):
    """Function for creating truncated noise vector: given the dimensions (n_samples, z_dim)
    and truncation value, creates a tensor of that shape filled with random numbers from the 
    truncated normal distribution

    Args:
        n_samples (int): the number of samples to generate
        z_dim (int): the dimension of the noise vector
        truncation (float): the truncation value, non-negative
    """

    truncated_noise = truncnorm.rvs(
        -truncation, truncation, size=(n_samples, z_dim)
    )
    return torch.Tensor(truncated_noise)