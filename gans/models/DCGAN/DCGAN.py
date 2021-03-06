import torch
import torchvision
from torch import nn
from pytorch_lightning import LightningModule

from .generator import generator, get_noise
from .discriminator import discriminator


class DCGAN(LightningModule):
    '''
    DCGAN Class
    Values:
        hidden_dim: the dimension of the noise vector, a scalar
        im_dim: the dimension of the images, fitted for the dataset used, a scalar
          (MNIST images are 28 x 28 = 784 so that is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self,
    im_chan: int = 1,
    z_dim: int = 10,
    hidden_dim: int = 64,
    lr: float = 0.0002,
    beta_1: float = 0.5,
    beta_2: float = 0.999,
    spectral_norm: bool = True,
    **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.z_dim = z_dim
        self.generator = generator(z_dim=self.hparams.z_dim, hidden_dim=self.hparams.hidden_dim, im_chan=self.hparams.im_chan).apply(self.weights_init)
        self.discriminator = discriminator(im_chan=self.hparams.im_chan, hidden_dim=self.hparams.hidden_dim, spectral_norm=spectral_norm).apply(self.weights_init)
    
    def weights_init(self, m):
        """initialize the weights to the normal distribution with mean 0 and standard deviation 0.02

        Args:
            m (nn.Module): nn.Module, generator or discriminator
        """
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        criterion = nn.BCEWithLogitsLoss()
        return criterion(y_hat, y)

    def training_step(self, train_batch, batch_idx, optimizer_idx):

        imgs, _ = train_batch

        # train generator
        if optimizer_idx == 0:
            
            # sample noise
            fake_noise = get_noise(len(imgs), self.z_dim)
            fake_noise = fake_noise.type_as(imgs)

            # generate images
            self.generated_imgs = self(fake_noise)

            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self(fake_noise)), valid)
            
            self.log("loss/g_loss", g_loss, prog_bar=True)
            
            return g_loss

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples
            
            # sample noise
            fake_noise2 = get_noise(len(imgs), self.z_dim)
            fake_noise2 = fake_noise2.type_as(imgs)
            
            # how well can it label as real?
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)

            fake_loss = self.adversarial_loss(self.discriminator(self(fake_noise2).detach()), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            self.log("loss/d_loss", d_loss, prog_bar=True)
            return d_loss

    def configure_optimizers(self):
        gen_opt = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta_1, self.hparams.beta_2))
        disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta_1, self.hparams.beta_2))
        return gen_opt, disc_opt
    
    def validation_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self):
        
        z = get_noise(n_samples=25, z_dim=self.hparams.z_dim)
        z = z.type_as(self.generator.generator[0][0].weight)

        # log sampled images
        sample_imgs = self(z)

        image_unflat = sample_imgs.detach().cpu().view(-1, 1, 28, 28)
        grid = torchvision.utils.make_grid(image_unflat, nrow=5)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
        return grid # sample_imgs# 
