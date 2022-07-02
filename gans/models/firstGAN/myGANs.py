import torch
from torch import nn
from pytorch_lightning import LightningModule
import torchvision

from .generator import generator
from .discriminator import discriminator

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
    **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        data_shape = (channels, width, height)
        self.generator = generator(z_dim=self.hparams.z_dim, latent_dim=self.hparams.latent_dim, img_shape=data_shape)
        self.discriminator = discriminator(img_shape=data_shape)
        self.example_input_array = torch.zeros(2, self.hparams.z_dim)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        criterion = nn.BCEWithLogitsLoss()
        return criterion(y_hat, y)

    def training_step(self, train_batch, batch_idx, optimizer_idx):

        imgs, _ = train_batch
        
         # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.z_dim)
        z = z.type_as(imgs)

        # train generator
        if optimizer_idx == 0:

            # generate images
            self.generated_imgs = self(z)

            # log sampled images
            # sample_imgs = self.generated_imgs[:6]
            # grid = torchvision.utils.make_grid(sample_imgs).detach()
            # self.logger.experiment.add_image("generated_images", grid, 0)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            
            self.log("loss/g_loss", g_loss, prog_bar=True)
            
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
            self.log("loss/d_loss", d_loss, prog_bar=True)
            return d_loss

    def configure_optimizers(self):
        gen_opt = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr)
        disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr)
        return gen_opt, disc_opt
    
    def validation_step(self, batch, batch_idx):
        pass

    def on_validation_epoch_end(self):
        
        validation_z = torch.randn(25, self.hparams.z_dim)
        z = validation_z.type_as(self.generator.generator[0].weight)

        # log sampled images
        sample_imgs = self(z)

        image_unflat = sample_imgs.detach().cpu().view(-1, 1, 28, 28)
        grid = torchvision.utils.make_grid(image_unflat, nrow=5)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
        return grid