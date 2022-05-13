import os
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

# import torch.nn.functional as F
from pytorch_lightning import LightningDataModule, LightningModule
from torchvision.datasets import MNIST

import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST

from .generator import generator
from .discriminator import discriminator

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count() - 4)

class GANs(LightningModule):
    '''
    myGANs Class
    Values:
        latent_dim: the dimension of the noise vector, a scalar
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
          (MNIST images are 28 x 28 = 784 so that is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self,
    channels,
    width,
    height,
    z_dim: int = 10,
    latent_dim: int = 64,
    lr: float = 0.00001,
    batch_size: int = 128,
    **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        data_shape = (channels, width, height)
        # self.batch_size = batch_size
        self.generator = generator(z_dim=self.hparams.z_dim, latent_dim=self.hparams.latent_dim, img_shape=data_shape)
        self.discriminator = discriminator(img_shape=data_shape)
        self.validation_z = torch.randn(8, self.hparams.z_dim)
        self.example_input_array = torch.zeros(2, self.hparams.z_dim)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        criterion = nn.BCEWithLogitsLoss()
        return criterion(y_hat, y)

    def training_step(self, train_batch, batch_idx, optimizer_idx):
        cur_batch_size = len(train_batch)
        imgs, _ = train_batch
        
         # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.z_dim)
        z = z.type_as(imgs)

        # train generator
        if optimizer_idx == 0:

            # generate images
            self.generated_imgs = self(z)

            # log sampled images
            sample_imgs = self.generated_imgs[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            # self.logger.experiment.add_image("generated_images", grid, 0)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            self.log("g_loss", g_loss, prog_bar=True)
            
            return g_loss

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)

            fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            self.log("d_loss", d_loss, prog_bar=True)
            return d_loss

    def configure_optimizers(self):
        gen_opt = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr)
        disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr)
        return gen_opt, disc_opt

    def on_validation_epoch_end(self):
        z = self.validation_z.type_as(self.generator.generator[0].weight)

        # log sampled images
        sample_imgs = self(z)
        # grid = torchvision.utils.make_grid(sample_imgs)
        # self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
        return sample_imgs# grid

class MNISTDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = PATH_DATASETS,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.ToTensor()
        
        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (1, 28, 28)
        self.num_classes = 10

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
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