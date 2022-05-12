import os
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

import torch.nn.functional as F
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torchvision.datasets import MNIST


from torchvision import transforms
from torchvision.datasets import MNIST

from .generator import generator
from .discriminator import discriminator
from .loss_function import get_disc_loss, get_gen_loss

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")

class GANs(LightningModule):
    '''
    myGANs Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
          (MNIST images are 28 x 28 = 784 so that is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, z_dim=64, lr=0.00001, num_images=25):
        super().__init__()
        self.save_hyperparameters()
        # self.batch_size = batch_size
        self.gen = generator(z_dim=z_dim, im_dim=784, hidden_dim=128)
        self.disc = discriminator(im_dim=784, hidden_dim=128)
        self.criterion = nn.BCEWithLogitsLoss()
        self.z_dim = z_dim
        self.num_images = num_images
        self.lr=lr

    def forward(self, z):
        return self.gen(z)

    # def adversarial_loss(self, y_hat, y):
    #     return self.criterion(y_hat, y)

    def training_step(self, train_batch, batch_idx, optimizer_idx):
        cur_batch_size = len(train_batch)
        # if test_generator:
        #     try:
        #         assert lr > 0.0000002 or (gen.gen[0][0].weight.grad.abs().max() < 0.0005 and epoch == 0)
        #         assert torch.any(gen.gen[0][0].weight.detach().clone() != old_generator_weights)
        #     except:
        #         error = True
        #         print("Runtime tests have failed")

        # train generator
        if optimizer_idx == 0:

            gen_loss = get_gen_loss(self.gen, self.disc, self.criterion, cur_batch_size, self.z_dim)
            # Update gradients
            gen_loss.backward(retain_graph=True)

            self.log("gen_loss", gen_loss, prog_bar=True)
            return gen_loss

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples
            real, _ = train_batch
            real = real.view(real.size(0), -1)

            disc_loss = get_disc_loss(self.gen, self.disc, self.criterion, real, self.num_images, self.z_dim)

            # Update gradients
            disc_loss.backward(retain_graph=True)

            self.log("disc_loss", disc_loss, prog_bar=True)
            return disc_loss

    def configure_optimizers(self):
        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=self.lr)
        disc_opt = torch.optim.Adam(self.disc.parameters(), lr=self.lr)
        return gen_opt, disc_opt
        
class MNISTDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = PATH_DATASETS,
        batch_size: int = 128,
        num_workers: int = 24,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.ToTensor()

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, download=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)
